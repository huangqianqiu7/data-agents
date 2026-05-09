"""
Custom exception hierarchy for the data-agent system.

All domain-specific exceptions inherit from ``DataAgentError`` so that
callers can catch the full family with a single ``except DataAgentError``.

This module is the canonical home of the exception hierarchy; both
``data_agent_refactored`` and ``data_agent_langchain`` import from here.
"""
from __future__ import annotations


class DataAgentError(Exception):
    """Base exception for all data-agent errors."""


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------
class ConfigError(DataAgentError):
    """Raised when the application configuration is invalid or missing."""


# ---------------------------------------------------------------------------
# Benchmark / dataset errors
# ---------------------------------------------------------------------------
class DatasetError(DataAgentError):
    """Raised for dataset-level structural problems (missing dirs, bad JSON)."""


class TaskNotFoundError(DatasetError):
    """Raised when a requested task ID does not exist in the dataset."""


# ---------------------------------------------------------------------------
# Tool errors
# ---------------------------------------------------------------------------
class ToolError(DataAgentError):
    """Base class for tool-related errors."""


class UnknownToolError(ToolError):
    """Raised when the agent requests a tool name that is not registered."""

    def __init__(self, tool_name: str, available_tools: frozenset[str] | set[str]) -> None:
        self.tool_name = tool_name
        self.available_tools = available_tools
        super().__init__(
            f"Unknown tool '{tool_name}'. "
            f"Available tools: {', '.join(sorted(available_tools))}"
        )


class ToolTimeoutError(ToolError):
    """Raised when a tool execution exceeds its time budget."""

    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout_seconds}s. "
            f"Try simplifying your code or query."
        )


class ToolValidationError(ToolError):
    """Raised when tool input validation fails (e.g. bad answer format)."""


class ContextPathEscapeError(ToolError):
    """Raised when a relative path attempts to escape the task context directory."""

    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        super().__init__(f"Path escapes context dir: {relative_path}")


class ContextAssetNotFoundError(ToolError):
    """Raised when a referenced file does not exist inside the context directory."""

    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        super().__init__(
            f"Missing context asset: {relative_path}. "
            f"Call `list_context` first to discover the actual files available."
        )


class ReadOnlySQLViolationError(ToolError):
    """Raised when a SQL statement is not read-only."""


# ---------------------------------------------------------------------------
# Agent / model errors
# ---------------------------------------------------------------------------
class AgentError(DataAgentError):
    """Base class for agent-level errors."""


class ModelCallError(AgentError):
    """Raised when a model API call fails after exhausting all retries."""

    def __init__(self, attempts: int, last_error: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Model call failed after {attempts} attempts: {last_error}")


class ModelResponseParseError(AgentError):
    """Raised when the model's JSON response cannot be parsed."""


# ---------------------------------------------------------------------------
# Runner errors
# ---------------------------------------------------------------------------
class RunnerError(DataAgentError):
    """Base class for runner-level errors."""


class InvalidRunIdError(RunnerError):
    """Raised when the supplied run_id is syntactically invalid."""
