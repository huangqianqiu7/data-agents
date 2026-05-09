"""
Backwards-compat shim — exceptions moved to ``data_agent_common.exceptions``.

Per LANGCHAIN_MIGRATION_PLAN.md §3.1 (C7), the exception hierarchy is shared
between the legacy hand-written backend (this package) and the new LangGraph
backend (``data_agent_langchain``). The canonical home is now
``data_agent_common.exceptions``; this module re-exports every public name
so existing imports continue to work unchanged.
"""
from __future__ import annotations

from data_agent_common.exceptions import (
    AgentError,
    ConfigError,
    ContextAssetNotFoundError,
    ContextPathEscapeError,
    DataAgentError,
    DatasetError,
    InvalidRunIdError,
    ModelCallError,
    ModelResponseParseError,
    ReadOnlySQLViolationError,
    RunnerError,
    TaskNotFoundError,
    ToolError,
    ToolTimeoutError,
    ToolValidationError,
    UnknownToolError,
)

__all__ = [
    "AgentError",
    "ConfigError",
    "ContextAssetNotFoundError",
    "ContextPathEscapeError",
    "DataAgentError",
    "DatasetError",
    "InvalidRunIdError",
    "ModelCallError",
    "ModelResponseParseError",
    "ReadOnlySQLViolationError",
    "RunnerError",
    "TaskNotFoundError",
    "ToolError",
    "ToolTimeoutError",
    "ToolValidationError",
    "UnknownToolError",
]
