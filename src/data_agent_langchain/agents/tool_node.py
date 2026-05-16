"""
自定义 ``tool_node``（§6.3 / C2）。

刻意不使用 LangGraph 预置 ``ToolNode``，原因：

  - 它返回 ``ToolMessage`` 而不是变更 ``RunState``；
  - 它依赖 ``return_direct=True`` 处理终止 answer（这会丢失结构化的
    ``AnswerTable`` 输出）；
  - 它不写本项目 ``trace.json`` 期望的 ``StepRecord`` schema。

本节点执行 ``state["action"]`` 指向的工具（通过 ``create_all_tools``
解析），用 ``call_tool_with_timeout`` 包一层硬超时防止挂死，然后：

  1. 无论成败都通过 reducer 追加一条 ``StepRecord``。
  2. 设置显式路由字段 ``last_tool_ok`` / ``last_tool_is_terminal`` /
     ``last_error_kind``（C11 / E15），让条件边不必去看 ``steps[-1]`` 猜。
  3. 终止 answer 命中时，把 ``state["answer"]`` 显式写好，外层图据此走
     条件边到 ``finalize_node``。
  4. 数据预览成功时把 ``preview_done = True``；``list_context`` 成功时
     额外把 ``discovery_done = True``。

如果上一个节点写了 ``skip_tool=True``（gate L1/L2 阻塞）或
``last_error_kind="parse_error"``，本节点直接 no-op 跳过工具执行 ——
这样 gate L1/L2 路径不会重跑原 action，也避免 v4 M1 fall-through 下的
重复计数。

错误码映射（v4 E15）：

  ``ToolRuntimeResult.error_kind``  →  ``RunState["last_error_kind"]``
  -----------------------------------------------------------------
  ``"timeout"``                      →  ``"tool_timeout"``
  ``"validation"``                   →  ``"tool_validation"``
  ``"runtime"``                      →  ``"tool_error"``

未注册的工具名直接映射到 ``last_error_kind="unknown_tool"``，不经过
``ToolRuntimeResult``（它们根本走不到 wrapper 那一步）。
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data_agent_langchain.agents.gate import DATA_PREVIEW_ACTIONS
from data_agent_langchain.agents.runtime import StepRecord
from data_agent_langchain.config import AppConfig, default_app_config
from data_agent_langchain.memory.records import DatasetKnowledgeRecord
from data_agent_langchain.memory.writers.store_backed import StoreBackedMemoryWriter
from data_agent_langchain.observability.events import dispatch_observability_event
from data_agent_langchain.runtime.context import get_current_app_config
from data_agent_langchain.runtime.rehydrate import build_runtime, rehydrate_task
from data_agent_langchain.runtime.state import RunState
from data_agent_langchain.tools.factory import create_all_tools
from data_agent_langchain.tools.timeout import call_tool_with_timeout
from data_agent_langchain.tools.tool_runtime import ToolRuntimeResult

logger = logging.getLogger(__name__)


# v4 E15：tool error_kind → state last_error_kind 的映射。模块加载时即
# 冻结，任意一边的字段名改动都会在 import 阶段就暴露出来，避免错误被
# 静默吞掉。
_ERROR_KIND_MAP: dict[str, str] = {
    "timeout": "tool_timeout",
    "validation": "tool_validation",
    "runtime": "tool_error",
}

_KNOWN_PATH_ACTION_INPUTS: dict[str, str] = {
    "execute_context_sql": "path",
    "inspect_sqlite_schema": "path",
    "read_csv": "path",
    "read_doc": "path",
    "read_json": "path",
}


# ---------------------------------------------------------------------------
# 公共节点
# ---------------------------------------------------------------------------

def tool_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    """带超时包装地执行 ``state["action"]``。"""
    # 上游已经发出了跳过信号（gate L1/L2 阻塞，或 parse_error）。
    if state.get("skip_tool"):
        return {}
    if state.get("last_error_kind") == "parse_error":
        return {}

    action = state.get("action") or ""
    if not action:
        # 防御性分支：parse_action_node 跑过之后理论上不该出现这种情况。
        return _emit_unknown_tool(state, "(missing)")

    app_config = _safe_get_app_config()

    try:
        task = rehydrate_task(state)
        runtime = build_runtime(task, app_config)
        tools = {t.name: t for t in create_all_tools(task, runtime)}
    except Exception as exc:
        # 重建任务 / 运行时失败按 runtime 错误处理（与工具体内崩溃同级别）。
        logger.exception("[tool_node] rehydrate failed: %s", exc)
        return _emit_runtime_failure(state, action, str(exc))

    if action not in tools:
        return _emit_unknown_tool(state, action)

    tool = tools[action]
    timeout_s = float(getattr(app_config.agent, "tool_timeout_s", 180.0))
    action_input = dict(state.get("action_input") or {})

    known_path_failure = _known_path_failure_result(
        state=state,
        action=action,
        action_input=action_input,
        app_config=app_config,
    )
    if known_path_failure is not None:
        dispatch_observability_event(
            "tool_call",
            {"tool": action, "step_index": int(state.get("step_index", 0) or 0), "ok": False},
            config,
        )
        return {
            "steps": [_step_record_from_result(state, action, action_input, known_path_failure)],
            "last_tool_ok": False,
            "last_tool_is_terminal": False,
            "last_error_kind": _map_error_kind(known_path_failure),
        }

    result: ToolRuntimeResult = call_tool_with_timeout(tool, action_input, timeout_s)
    dispatch_observability_event(
        "tool_call",
        {"tool": action, "step_index": int(state.get("step_index", 0) or 0), "ok": result.ok},
        config,
    )

    update = {
        "steps": [_step_record_from_result(state, action, action_input, result)],
        "last_tool_ok": result.ok,
        "last_tool_is_terminal": bool(result.is_terminal),
        "last_error_kind": _map_error_kind(result),
    }

    if result.is_terminal and result.answer is not None:
        update["answer"] = result.answer

    if result.ok:
        if action == "list_context":
            update["discovery_done"] = True
            update["known_paths"] = _extract_known_paths(result.content)
        if action in DATA_PREVIEW_ACTIONS:
            update["preview_done"] = True
        memory_cfg = getattr(app_config, "memory", None)
        if (
            action in _DATASET_KNOWLEDGE_ACTIONS
            and memory_cfg is not None
            and memory_cfg.mode != "disabled"
        ):
            try:
                from data_agent_langchain.memory.factory import build_store, build_writer

                store = build_store(memory_cfg)
                writer = build_writer(memory_cfg, store=store)
                dataset_name = Path(state.get("dataset_root", "") or ".").name or "default"
                content = result.content if isinstance(result.content, dict) else {}
                _maybe_write_dataset_knowledge(
                    writer=writer,
                    dataset=dataset_name,
                    action=action,
                    action_input=action_input,
                    content=content,
                )
            except Exception as exc:
                logger.warning("[tool_node] memory subsystem error: %s", exc)

    return update


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _safe_get_app_config() -> AppConfig:
    try:
        return get_current_app_config()
    except RuntimeError:
        return default_app_config()


def _extract_known_paths(content: dict[str, Any]) -> list[str]:
    entries = content.get("entries") if isinstance(content, Mapping) else None
    if not isinstance(entries, list):
        return []
    paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("kind") != "file":
            continue
        path = entry.get("path")
        if isinstance(path, str) and path.strip():
            paths.add(_normalize_context_path(path))
    return sorted(paths)


def _known_path_failure_result(
    *,
    state: RunState,
    action: str,
    action_input: dict[str, Any],
    app_config: AppConfig,
) -> ToolRuntimeResult | None:
    if not bool(getattr(app_config.agent, "enforce_known_path_only", False)):
        return None
    if not bool(state.get("discovery_done", False)):
        return None
    raw_known_paths = state.get("known_paths") or []
    if not raw_known_paths:
        return None
    path_key = _KNOWN_PATH_ACTION_INPUTS.get(action)
    if path_key is None:
        return None
    raw_path = action_input.get(path_key)
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    requested_path = _normalize_context_path(raw_path)
    available_paths = sorted(
        {
            _normalize_context_path(path)
            for path in raw_known_paths
            if isinstance(path, str) and path.strip()
        }
    )
    if requested_path in available_paths:
        return None
    return ToolRuntimeResult(
        ok=False,
        content={
            "tool": action,
            "error": _known_path_error_message(requested_path, available_paths),
            "rejected_path": requested_path,
            "available_paths": available_paths,
            "validation": "known_path_only",
        },
        error_kind="validation",
    )


def _known_path_error_message(rejected_path: str, available_paths: list[str]) -> str:
    available = ", ".join(available_paths) if available_paths else "(none)"
    return (
        f"Path '{rejected_path}' is not present in the current task context. "
        f"Available files from list_context: {available}. "
        f"Do not retry '{rejected_path}' unless a later list_context output shows it exists."
    )


def _normalize_context_path(path: str) -> str:
    text = path.replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def _step_record_from_result(
    state: RunState,
    action: str,
    action_input: dict[str, Any],
    result: ToolRuntimeResult,
) -> StepRecord:
    """把 ``ToolRuntimeResult`` 拼装成一条 ``StepRecord``。

    schema 与 legacy ``ReActAgent`` / ``PlanAndSolveAgent`` 一致：

      observation = {"ok": bool, "tool": str, "content": dict}

    失败时 ``content.error`` 字段携带具体错误消息，让 LLM 下一轮
    自行纠错。
    """
    observation: dict[str, Any] = {
        "ok": result.ok,
        "tool": action,
        "content": result.content,
    }
    return StepRecord(
        step_index=int(state.get("step_index", 0) or 0),
        thought=state.get("thought", "") or "",
        action=action,
        action_input=action_input,
        raw_response=state.get("raw_response", "") or "",
        observation=observation,
        ok=result.ok,
        phase=state.get("phase", "") or "",
        plan_progress=state.get("plan_progress", "") or "",
        plan_step_description=state.get("plan_step_description", "") or "",
    )


def _map_error_kind(result: ToolRuntimeResult) -> str | None:
    """把 ``ToolRuntimeResult.error_kind`` 翻译成 ``RunState`` 枚举值。"""
    if result.ok:
        return None
    return _ERROR_KIND_MAP.get(result.error_kind or "runtime", "tool_error")


def _emit_unknown_tool(state: RunState, action: str) -> dict[str, Any]:
    """``action`` 不在工具注册表中的失败路径。"""
    error_msg = (
        f"Unknown tool '{action}'. "
        f"Available tools: answer, execute_context_sql, execute_python, "
        f"inspect_sqlite_schema, list_context, read_csv, read_doc, read_json"
    )
    return {
        "steps": [
            StepRecord(
                step_index=int(state.get("step_index", 0) or 0),
                thought=state.get("thought", "") or "",
                action=action,
                action_input=dict(state.get("action_input") or {}),
                raw_response=state.get("raw_response", "") or "",
                observation={"ok": False, "tool": action, "error": error_msg},
                ok=False,
                phase=state.get("phase", "") or "",
                plan_progress=state.get("plan_progress", "") or "",
                plan_step_description=state.get("plan_step_description", "") or "",
            )
        ],
        "last_tool_ok": False,
        "last_tool_is_terminal": False,
        "last_error_kind": "unknown_tool",
    }


def _emit_runtime_failure(state: RunState, action: str, msg: str) -> dict[str, Any]:
    """重建任务 / 运行时构建崩溃的失败路径。"""
    return {
        "steps": [
            StepRecord(
                step_index=int(state.get("step_index", 0) or 0),
                thought=state.get("thought", "") or "",
                action=action,
                action_input=dict(state.get("action_input") or {}),
                raw_response=state.get("raw_response", "") or "",
                observation={"ok": False, "tool": action, "error": msg},
                ok=False,
                phase=state.get("phase", "") or "",
                plan_progress=state.get("plan_progress", "") or "",
                plan_step_description=state.get("plan_step_description", "") or "",
            )
        ],
        "last_tool_ok": False,
        "last_tool_is_terminal": False,
        "last_error_kind": "tool_error",
    }


_DATASET_KNOWLEDGE_ACTIONS: dict[str, str] = {
    "read_csv": "csv",
    "read_json": "json",
    "read_doc": "doc",
    "inspect_sqlite_schema": "sqlite",
}


def _maybe_write_dataset_knowledge(
    *,
    writer: StoreBackedMemoryWriter,
    dataset: str,
    action: str,
    action_input: dict[str, Any],
    content: dict[str, Any],
) -> None:
    """Write DatasetKnowledgeRecord after successful preview tools; skip safely on missing fields."""
    file_kind = _DATASET_KNOWLEDGE_ACTIONS.get(action)
    if file_kind is None:
        return
    file_path = (
        action_input.get("file_path")
        or action_input.get("path")
        or content.get("file_path")
    )
    if not isinstance(file_path, str) or not file_path:
        return
    schema = _extract_dataset_schema(action, content)
    if not schema:
        return
    row_count = content.get("row_count_estimate")
    if row_count is None:
        row_count = content.get("row_count")
    if not isinstance(row_count, int):
        row_count = None
    columns = content.get("columns")
    if isinstance(columns, list) and columns:
        sample_columns = [str(c) for c in columns]
    elif action == "inspect_sqlite_schema":
        sample_columns = _sqlite_table_names(content)
    else:
        sample_columns = []
    record = DatasetKnowledgeRecord(
        file_path=file_path,
        file_kind=file_kind,  # type: ignore[arg-type]
        schema=schema,
        row_count_estimate=row_count,
        sample_columns=sample_columns,
    )
    try:
        writer.write_dataset_knowledge(dataset, record)
    except Exception as exc:
        logger.warning("[tool_node] memory write skipped: %s", exc)


def _extract_dataset_schema(action: str, content: dict[str, Any]) -> dict[str, str]:
    schema_src = content.get("dtypes")
    if schema_src is None:
        schema_src = content.get("schema")
    if isinstance(schema_src, dict):
        return {str(k): str(v) for k, v in schema_src.items() if str(k)}
    if action == "inspect_sqlite_schema":
        return _schema_from_sqlite_tables(content)
    return _infer_schema_from_preview(content)


def _schema_from_sqlite_tables(content: dict[str, Any]) -> dict[str, str]:
    tables = content.get("tables")
    if not isinstance(tables, list):
        return {}
    schema: dict[str, str] = {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        name = table.get("name")
        create_sql = table.get("create_sql")
        if isinstance(name, str) and name and isinstance(create_sql, str) and create_sql:
            schema[name] = create_sql
    return schema


def _sqlite_table_names(content: dict[str, Any]) -> list[str]:
    tables = content.get("tables")
    if not isinstance(tables, list):
        return []
    names: list[str] = []
    for table in tables:
        if isinstance(table, dict) and isinstance(table.get("name"), str):
            names.append(table["name"])
    return names


def _infer_schema_from_preview(content: dict[str, Any]) -> dict[str, str]:
    columns = content.get("columns")
    rows = content.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return {}
    sample_row = rows[0] if rows else []
    schema: dict[str, str] = {}
    for index, column in enumerate(columns):
        value = (
            sample_row[index]
            if isinstance(sample_row, list) and index < len(sample_row)
            else None
        )
        schema[str(column)] = _infer_scalar_type(value)
    return schema


def _infer_scalar_type(value: Any) -> str:
    if value is None:
        return "string"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    text = str(value)
    try:
        int(text)
    except ValueError:
        try:
            float(text)
        except ValueError:
            return "string"
        return "float"
    return "int"


__all__ = ["tool_node", "_maybe_write_dataset_knowledge"]
