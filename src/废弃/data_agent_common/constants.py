"""
Cross-backend string constants.

Per LANGCHAIN_MIGRATION_PLAN.md v4 E1, these strings must stay byte-for-byte
identical between the legacy hand-written agent loop and the new LangGraph
backend, otherwise golden-trace parity tests fail.

If a constant is added here, both backends MUST import from this module —
do not duplicate the literal value anywhere.
"""
from __future__ import annotations

# Used by Plan-and-Solve when the model's plan is exhausted but no answer
# has been submitted yet. Both backends append this string to the plan and
# stay on the last index until the model finally calls `answer`.
#
# Source of truth: src/data_agent_refactored/agents/plan_solve_agent.py
# (the original `_FALLBACK_STEP` private constant on line 513).
FALLBACK_STEP_PROMPT: str = "Call the answer tool with the final result table."
