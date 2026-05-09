"""
Tool registry — the single dispatch point between agents and tool handlers.

Responsibilities:
  1. Define :class:`ToolSpec` (metadata shown to the LLM) and
     :class:`ToolExecutionResult` (standardised handler return type).
  2. Implement :class:`ToolRegistry` with a ``register()`` method that
     supports plugin-style extension.
  3. Provide :func:`create_default_tool_registry` to assemble the built-in
     tool set in one call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from data_agent_common.benchmark.schema import AnswerTable, PublicTask
from data_agent_common.exceptions import UnknownToolError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Metadata describing a tool (serialised into the system prompt)."""
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Standardised return value from any tool handler."""
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


# Type alias for handler callables.
ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ToolRegistry:
    """Central registry mapping tool names to specs and handlers.

    Use :meth:`register` to add new tools at runtime (plugin pattern).
    """
    specs: dict[str, ToolSpec] = field(default_factory=dict)
    handlers: dict[str, ToolHandler] = field(default_factory=dict)

    # -- Plugin-style registration -----------------------------------------

    def register(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a tool with its spec and handler.

        Args:
            spec: Tool metadata (name, description, input schema).
            handler: Callable implementing the tool logic.
            overwrite: If *False* (default) and the tool name is already
                registered, a ``ValueError`` is raised.
        """
        if spec.name in self.specs and not overwrite:
            raise ValueError(f"Tool '{spec.name}' is already registered.")
        self.specs[spec.name] = spec
        self.handlers[spec.name] = handler
        logger.debug("Registered tool: %s", spec.name)

    # -- Prompt generation -------------------------------------------------

    def describe_for_prompt(self) -> str:
        """Render all registered tools as plain text for the system prompt."""
        lines: list[str] = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    # -- Execution ---------------------------------------------------------

    def execute(
        self,
        task: PublicTask,
        action: str,
        action_input: dict[str, Any],
    ) -> ToolExecutionResult:
        """Dispatch *action* to the corresponding handler."""
        if action not in self.handlers:
            raise UnknownToolError(action, frozenset(self.handlers.keys()))
        return self.handlers[action](task, action_input)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_default_tool_registry() -> ToolRegistry:
    """Create a :class:`ToolRegistry` pre-loaded with the 8 built-in tools."""
    # Import here to avoid a circular dependency at module level.
    from data_agent_refactored.tools.handlers import (
        EXECUTE_PYTHON_TIMEOUT_SECONDS,
        handle_answer,
        handle_execute_context_sql,
        handle_execute_python,
        handle_inspect_sqlite_schema,
        handle_list_context,
        handle_read_csv,
        handle_read_doc,
        handle_read_json,
    )

    registry = ToolRegistry()

    _TOOL_DEFS: list[tuple[ToolSpec, ToolHandler]] = [
        (
            ToolSpec(
                name="answer",
                description="Submit the final answer table. This is the only valid terminating action.",
                input_schema={
                    "columns": ["column_name"],
                    "rows": [["value_1"]],
                },
            ),
            handle_answer,
        ),
        (
            ToolSpec(
                name="execute_context_sql",
                description="Run a read-only SQL query against a sqlite/db file inside context.",
                input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
            ),
            handle_execute_context_sql,
        ),
        (
            ToolSpec(
                name="execute_python",
                description=(
                    "Execute arbitrary Python code with the task context directory as the "
                    "working directory. The tool returns the code's captured stdout as `output`. "
                    f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
                ),
                input_schema={
                    "code": "import os\nprint(sorted(os.listdir('.')))",
                },
            ),
            handle_execute_python,
        ),
        (
            ToolSpec(
                name="inspect_sqlite_schema",
                description="Inspect tables and columns in a sqlite/db file inside context.",
                input_schema={"path": "relative/path/to/file.sqlite"},
            ),
            handle_inspect_sqlite_schema,
        ),
        (
            ToolSpec(
                name="list_context",
                description="List files and directories available under context.",
                input_schema={"max_depth": 4},
            ),
            handle_list_context,
        ),
        (
            ToolSpec(
                name="read_csv",
                description="Read a preview of a CSV file inside context.",
                input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
            ),
            handle_read_csv,
        ),
        (
            ToolSpec(
                name="read_doc",
                description="Read a text-like document inside context.",
                input_schema={"path": "relative/path/to/file.md", "max_chars": 4000},
            ),
            handle_read_doc,
        ),
        (
            ToolSpec(
                name="read_json",
                description="Read a preview of a JSON file inside context.",
                input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
            ),
            handle_read_json,
        ),
    ]

    for spec, handler in _TOOL_DEFS:
        registry.register(spec, handler)

    return registry
