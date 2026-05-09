"""
Built-in tool handler implementations.

Each handler follows the :data:`ToolHandler` signature::

    (task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult

Handlers are registered with the :class:`ToolRegistry` via
:func:`create_default_tool_registry`.
"""
from __future__ import annotations

from typing import Any

from data_agent_common.benchmark.schema import AnswerTable, PublicTask
from data_agent_common.exceptions import ToolValidationError
from data_agent_common.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_common.tools.python_exec import (
    EXECUTE_PYTHON_TIMEOUT_SECONDS,
    execute_python_code,
)
from data_agent_common.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema
from data_agent_refactored.tools.registry import ToolExecutionResult

# ``EXECUTE_PYTHON_TIMEOUT_SECONDS`` is re-exported from common so the legacy
# ``ToolRegistry`` description and the new LangGraph backend both reference
# the same source of truth (LANGCHAIN_MIGRATION_PLAN.md v4 E11).
__all__ = [
    "EXECUTE_PYTHON_TIMEOUT_SECONDS",
    "handle_answer",
    "handle_execute_context_sql",
    "handle_execute_python",
    "handle_inspect_sqlite_schema",
    "handle_list_context",
    "handle_read_csv",
    "handle_read_doc",
    "handle_read_json",
]


# ---------------------------------------------------------------------------
# Individual handlers
# ---------------------------------------------------------------------------

def handle_list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """List files and directories under the task context folder."""
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def handle_read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """Preview the first *max_rows* rows of a CSV file."""
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def handle_read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """Preview a JSON file (character-limited)."""
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def handle_read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """Preview a text document (character-limited)."""
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def handle_inspect_sqlite_schema(
    task: PublicTask, action_input: dict[str, Any]
) -> ToolExecutionResult:
    """Inspect tables and columns in a SQLite database file."""
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def handle_execute_context_sql(
    task: PublicTask, action_input: dict[str, Any]
) -> ToolExecutionResult:
    """Execute a read-only SQL query against a SQLite database."""
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def handle_execute_python(
    task: PublicTask, action_input: dict[str, Any]
) -> ToolExecutionResult:
    """Execute arbitrary Python code with the task context as working directory."""
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def handle_answer(_task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """Validate and submit the final answer table (terminal tool)."""
    columns = action_input.get("columns")
    rows = action_input.get("rows")

    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(item, str) for item in columns)
    ):
        raise ToolValidationError("answer.columns must be a non-empty list of strings.")

    if not isinstance(rows, list):
        raise ToolValidationError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ToolValidationError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ToolValidationError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)

    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )
