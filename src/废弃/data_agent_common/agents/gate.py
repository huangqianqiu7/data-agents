"""
Data-preview gate constants and predicates.

The gate logic lives in two places per backend (the legacy hand-written
loop in ``data_agent_refactored.agents.plan_solve_agent._handle_gate_block``
and the new LangGraph ``data_agent_langchain.agents.gate.gate_node``), but
the constants they read MUST be byte-for-byte identical or parity tests
break (LANGCHAIN_MIGRATION_PLAN.md §7).

Constants:
  - :data:`DATA_PREVIEW_ACTIONS` — tool names that count as "data inspection".
  - :data:`GATED_ACTIONS` — tool names blocked until a preview occurred.
  - :data:`GATE_REMINDER` — message returned to the LLM on a gate block.

Functions:
  - :func:`has_data_preview` — return ``True`` iff at least one preview
    step succeeded in the given step list.
"""
from __future__ import annotations

from collections.abc import Iterable

from data_agent_common.agents.runtime import StepRecord


# Tool names that qualify as "data inspection" (unlocking gated tools).
DATA_PREVIEW_ACTIONS: frozenset[str] = frozenset({
    "read_csv",
    "read_json",
    "read_doc",
    "inspect_sqlite_schema",
})

# Tool names that require a prior data preview before use.
GATED_ACTIONS: frozenset[str] = frozenset({
    "execute_python",
    "execute_context_sql",
    "answer",
})

# Reminder message injected as the observation on a gate block.
GATE_REMINDER: str = (
    "Blocked: inspect data first. Call read_csv / read_json / read_doc / "
    "inspect_sqlite_schema before using execute_python, execute_context_sql, or answer."
)


def has_data_preview(steps: Iterable[StepRecord]) -> bool:
    """Return ``True`` if at least one data-preview tool succeeded.

    Accepts any iterable of ``StepRecord`` so the function works equally
    well with the legacy ``AgentRuntimeState.steps`` list and the new
    LangGraph ``RunState["steps"]``.
    """
    return any(s.ok and s.action in DATA_PREVIEW_ACTIONS for s in steps)


__all__ = [
    "DATA_PREVIEW_ACTIONS",
    "GATED_ACTIONS",
    "GATE_REMINDER",
    "has_data_preview",
]
