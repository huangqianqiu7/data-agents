"""
Phase 2 functional tests for the new ``BaseTool`` subclasses.

Coverage:

  - ``AnswerTool``: success path returns ``is_terminal=True`` with a real
    ``AnswerTable``; validation failures return ``ok=False`` with
    ``error_kind="validation"``.
  - ``ListContextTool`` / ``ReadCsvTool`` / ``ReadJsonTool`` / ``ReadDocTool``:
    end-to-end on a synthetic in-memory task; success returns ``ok=True``
    with the expected payload shape; missing-file errors come back as
    ``ok=False`` with ``error_kind="runtime"``.
  - ``InspectSqliteSchemaTool``: schema returned for a real sqlite db.
  - ``call_tool_with_timeout`` mapping of pydantic validation errors to
    ``error_kind="validation"`` and timeouts to ``error_kind="timeout"``.
  - Factory determinism — the returned tool list contains exactly the
    eight expected tool classes, in deterministic order.

Tests run without network access.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import time
from pathlib import Path

from data_agent_langchain.benchmark.schema import AnswerTable, PublicTask, TaskAssets, TaskRecord
from data_agent_langchain.tools.answer import AnswerTool
from data_agent_langchain.tools.execute_context_sql import ExecuteContextSqlTool
from data_agent_langchain.tools.factory import create_all_tools
from data_agent_langchain.tools.inspect_sqlite_schema import InspectSqliteSchemaTool
from data_agent_langchain.tools.list_context import ListContextTool
from data_agent_langchain.tools.read_csv import ReadCsvTool
from data_agent_langchain.tools.read_doc import ReadDocTool
from data_agent_langchain.tools.read_json import ReadJsonTool
from data_agent_langchain.tools.timeout import call_tool_with_timeout
from data_agent_langchain.tools.tool_runtime import ToolRuntime, ToolRuntimeResult


# ---------------------------------------------------------------------------
# Synthetic task fixture
# ---------------------------------------------------------------------------

def _make_task(tmp_path: Path) -> PublicTask:
    """Build a synthetic ``PublicTask`` rooted at *tmp_path* with stock files."""
    task_dir = tmp_path / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)

    # ----- CSV -----
    csv_path = context_dir / "matches.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        w.writerow([1, 95])
        w.writerow([2, 80])
        w.writerow([3, 70])

    # ----- JSON -----
    json_path = context_dir / "info.json"
    json_path.write_text(json.dumps({"hello": "world"}, indent=2), encoding="utf-8")

    # ----- DOC (markdown) -----
    doc_path = context_dir / "README.md"
    doc_path.write_text("# title\n\nbody", encoding="utf-8")

    # ----- SQLite -----
    sqlite_path = context_dir / "data.sqlite"
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("CREATE TABLE players(id INTEGER, name TEXT)")
    conn.executemany(
        "INSERT INTO players VALUES (?, ?)", [(1, "Alice"), (2, "Bob")],
    )
    conn.commit()
    conn.close()

    record = TaskRecord(task_id="task_test", difficulty="easy", question="q?")
    assets = TaskAssets(task_dir=task_dir, context_dir=context_dir)
    return PublicTask(record=record, assets=assets)


def _runtime_for(task: PublicTask) -> ToolRuntime:
    return ToolRuntime(
        task_dir=str(task.task_dir),
        context_dir=str(task.context_dir),
        python_timeout_s=30.0,
        sql_row_limit=200,
        max_obs_chars=3000,
    )


# ---------------------------------------------------------------------------
# AnswerTool
# ---------------------------------------------------------------------------

def test_answer_tool_success_terminal(tmp_path):
    task = _make_task(tmp_path)
    tool = AnswerTool(task=task, runtime=_runtime_for(task))
    result: ToolRuntimeResult = tool.invoke(
        {"columns": ["score"], "rows": [["80.5"]]},
    )
    assert result.ok
    assert result.is_terminal
    assert isinstance(result.answer, AnswerTable)
    assert result.answer.columns == ["score"]
    assert result.answer.rows == [["80.5"]]
    assert result.content["status"] == "submitted"
    assert result.content["column_count"] == 1
    assert result.content["row_count"] == 1


def test_answer_tool_empty_columns_fails_validation(tmp_path):
    task = _make_task(tmp_path)
    tool = AnswerTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"columns": [], "rows": []})
    assert not result.ok
    assert result.is_terminal is False
    assert result.error_kind == "validation"
    assert "non-empty list of strings" in result.content["error"]


def test_answer_tool_row_length_mismatch(tmp_path):
    task = _make_task(tmp_path)
    tool = AnswerTool(task=task, runtime=_runtime_for(task))
    # 2 columns but only 1 value per row -> mismatch
    result = tool.invoke({"columns": ["a", "b"], "rows": [["x"]]})
    assert not result.ok
    assert result.error_kind == "validation"
    assert "match the number of columns" in result.content["error"]


def test_answer_tool_non_list_columns_fails(tmp_path):
    """Pydantic schema rejects ``columns`` of the wrong type via ValidationError;
    the wrapper maps that to ``error_kind="validation"``."""
    task = _make_task(tmp_path)
    tool = AnswerTool(task=task, runtime=_runtime_for(task))
    result = call_tool_with_timeout(
        tool,
        {"columns": "not-a-list", "rows": []},
        timeout_seconds=5.0,
    )
    assert not result.ok
    assert result.error_kind == "validation"


# ---------------------------------------------------------------------------
# ListContextTool
# ---------------------------------------------------------------------------

def test_list_context_returns_tree(tmp_path):
    task = _make_task(tmp_path)
    tool = ListContextTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"max_depth": 4})
    assert result.ok
    paths = sorted(e["path"] for e in result.content["entries"])
    assert "matches.csv" in paths
    assert "info.json" in paths
    assert "README.md" in paths
    assert "data.sqlite" in paths


# ---------------------------------------------------------------------------
# ReadCsvTool
# ---------------------------------------------------------------------------

def test_read_csv_preview_columns_and_rows(tmp_path):
    task = _make_task(tmp_path)
    tool = ReadCsvTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "matches.csv", "max_rows": 2})
    assert result.ok
    assert result.content["columns"] == ["id", "score"]
    assert result.content["rows"][:2] == [["1", "95"], ["2", "80"]]
    assert result.content["row_count"] == 3


def test_read_csv_missing_file_returns_validation_error(tmp_path):
    task = _make_task(tmp_path)
    tool = ReadCsvTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "does_not_exist.csv"})
    assert not result.ok
    assert result.error_kind == "validation"
    assert "Missing context asset" in result.content["error"]


# ---------------------------------------------------------------------------
# ReadJsonTool / ReadDocTool
# ---------------------------------------------------------------------------

def test_read_json_preview_shape(tmp_path):
    task = _make_task(tmp_path)
    tool = ReadJsonTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "info.json", "max_chars": 100})
    assert result.ok
    assert "hello" in result.content["preview"]


def test_read_doc_preview_shape(tmp_path):
    task = _make_task(tmp_path)
    tool = ReadDocTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "README.md", "max_chars": 100})
    assert result.ok
    assert "title" in result.content["preview"]


def test_read_json_missing_file_returns_validation_error(tmp_path):
    task = _make_task(tmp_path)
    tool = ReadJsonTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "does_not_exist.json"})
    assert not result.ok
    assert result.error_kind == "validation"
    assert "Missing context asset" in result.content["error"]


def test_read_doc_missing_file_returns_validation_error(tmp_path):
    task = _make_task(tmp_path)
    tool = ReadDocTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "does_not_exist.md"})
    assert not result.ok
    assert result.error_kind == "validation"
    assert "Missing context asset" in result.content["error"]


# ---------------------------------------------------------------------------
# InspectSqliteSchemaTool / ExecuteContextSqlTool
# ---------------------------------------------------------------------------

def test_inspect_sqlite_schema(tmp_path):
    task = _make_task(tmp_path)
    tool = InspectSqliteSchemaTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "data.sqlite"})
    assert result.ok
    table_names = [t["name"] for t in result.content["tables"]]
    assert "players" in table_names


def test_execute_context_sql_select(tmp_path):
    task = _make_task(tmp_path)
    tool = ExecuteContextSqlTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke(
        {"path": "data.sqlite", "sql": "SELECT name FROM players ORDER BY id"},
    )
    assert result.ok
    assert result.content["rows"] == [["Alice"], ["Bob"]]


def test_inspect_sqlite_missing_db_returns_validation_error(tmp_path):
    task = _make_task(tmp_path)
    tool = InspectSqliteSchemaTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "missing.sqlite"})
    assert not result.ok
    assert result.error_kind == "validation"
    assert "Missing context asset" in result.content["error"]


def test_execute_context_sql_missing_db_returns_validation_error(tmp_path):
    task = _make_task(tmp_path)
    tool = ExecuteContextSqlTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "missing.sqlite", "sql": "SELECT 1"})
    assert not result.ok
    assert result.error_kind == "validation"
    assert "Missing context asset" in result.content["error"]


def test_execute_context_sql_schema_mismatch_returns_structured_missing_table(tmp_path):
    task = _make_task(tmp_path)
    tool = ExecuteContextSqlTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "data.sqlite", "sql": "SELECT * FROM missing_table"})
    assert not result.ok
    assert result.error_kind == "runtime"
    assert result.content["sql_error_kind"] == "schema_mismatch"
    assert result.content["missing_kind"] == "table"
    assert result.content["missing_identifier"] == "missing_table"
    assert result.content["path"] == "data.sqlite"
    assert result.content["available_tables"] == ["players"]
    assert "missing_table" in result.content["error"]


def test_execute_context_sql_schema_mismatch_returns_structured_missing_column(tmp_path):
    task = _make_task(tmp_path)
    tool = ExecuteContextSqlTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke({"path": "data.sqlite", "sql": "SELECT missing_col FROM players"})
    assert not result.ok
    assert result.error_kind == "runtime"
    assert result.content["sql_error_kind"] == "schema_mismatch"
    assert result.content["missing_kind"] == "column"
    assert result.content["missing_identifier"] == "missing_col"
    assert result.content["path"] == "data.sqlite"
    assert result.content["available_tables"] == ["players"]


def test_execute_context_sql_rejects_writes(tmp_path):
    task = _make_task(tmp_path)
    tool = ExecuteContextSqlTool(task=task, runtime=_runtime_for(task))
    result = tool.invoke(
        {"path": "data.sqlite", "sql": "DELETE FROM players"},
    )
    assert not result.ok
    assert result.error_kind == "runtime"
    assert "read-only" in result.content["error"].lower()
    assert "sql_error_kind" not in result.content


# ---------------------------------------------------------------------------
# call_tool_with_timeout
# ---------------------------------------------------------------------------

def test_call_tool_with_timeout_maps_timeout(tmp_path):
    """A tool that hangs longer than the budget returns error_kind='timeout'."""
    task = _make_task(tmp_path)
    runtime = _runtime_for(task)

    class _SleepyInput:
        pass

    # Build a slow tool that just sleeps; we use ListContextTool here with
    # a monkey-patched ``_run`` that sleeps to keep the rest of the
    # construction valid.
    tool = ListContextTool(task=task, runtime=runtime)

    def _slow_run(*_args, **_kwargs):
        time.sleep(0.5)
        return ToolRuntimeResult(ok=True, content={})

    object.__setattr__(tool, "_run", _slow_run)

    result = call_tool_with_timeout(tool, {"max_depth": 4}, timeout_seconds=0.05)
    assert not result.ok
    assert result.error_kind == "timeout"
    assert "timed out" in result.content["error"]


def test_call_tool_with_timeout_maps_validation(tmp_path):
    """Pydantic ValidationError surfaces as error_kind='validation'."""
    task = _make_task(tmp_path)
    runtime = _runtime_for(task)
    tool = ReadCsvTool(task=task, runtime=runtime)
    # ``max_rows`` rejects 0 (ge=1 constraint); should produce validation error
    result = call_tool_with_timeout(
        tool, {"path": "matches.csv", "max_rows": 0}, timeout_seconds=5.0,
    )
    assert not result.ok
    assert result.error_kind == "validation"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_returns_eight_tools_in_deterministic_order(tmp_path):
    task = _make_task(tmp_path)
    tools = create_all_tools(task, _runtime_for(task))
    names = [t.name for t in tools]
    assert names == [
        "answer",
        "execute_context_sql",
        "execute_python",
        "inspect_sqlite_schema",
        "list_context",
        "read_csv",
        "read_doc",
        "read_json",
    ]
    # Each tool received the same task / runtime
    for tool in tools:
        # PrivateAttrs are accessed via ``_attr`` on the instance.
        assert tool._task is task
        assert tool._runtime.context_dir == str(task.context_dir)


# ---------------------------------------------------------------------------
# ExecutePythonInput schema hints (task_11 trace 改进, 方案一)
# ---------------------------------------------------------------------------

def test_execute_python_input_code_description_warns_about_extra_arguments():
    """``ExecutePythonInput.code`` 的 ``Field.description`` 必须告诉模型：
    整段代码留在 ``code`` 字符串里，不要把内层 dict 字面量的 key 拆出来
    当成额外 tool 参数（否则会被 ``extra_forbidden`` 拒绝）。

    回归源：artifacts/runs/20260510T090648Z/task_11/trace.json step 6/8 ——
    模型在 tool_calling 模式下生成 code 字符串时，把 ``{'SEX': p['SEX'],
    'Diagnosis': p['Diagnosis']}`` 的 key 误提到 args 顶层，触发 Pydantic
    ``extra_forbidden`` 校验失败，浪费了 3 个 step。
    """
    from data_agent_langchain.tools.execute_python import ExecutePythonInput

    desc = (ExecutePythonInput.model_fields["code"].description or "")
    lower = desc.lower()
    # 必须告诉模型整段 code 是单个字符串。
    assert "single string" in lower, desc
    # 必须明确提到 dict 字面量场景。
    assert "dict" in lower, desc
    # 必须点名 extra_forbidden 错误代号，让模型把 hint 与运行时错误对齐。
    assert "extra_forbidden" in desc, desc
