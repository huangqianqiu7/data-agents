"""
Phase 1 byte-for-byte parity assertions for cross-backend constants.

These constants must NEVER drift between the legacy hand-written backend
and the new LangGraph backend, otherwise prompt strings sent to the LLM
diverge and golden-trace parity (LANGCHAIN_MIGRATION_PLAN.md D14) breaks.

The plan calls these out explicitly:
  - v4 E1: ``FALLBACK_STEP_PROMPT == "Call the answer tool with the final result table."``
  - v4 E2: ``SANITIZE_ACTIONS == frozenset({"__error__", "__replan_failed__"})``
"""
from __future__ import annotations


def test_fallback_step_prompt_byte_for_byte():
    """Match the legacy private constant from plan_solve_agent.py:513 exactly."""
    from data_agent_common.constants import FALLBACK_STEP_PROMPT

    assert FALLBACK_STEP_PROMPT == "Call the answer tool with the final result table."


def test_sanitize_actions_byte_for_byte():
    """Match the legacy ``_SANITIZE_ACTIONS`` set from context_manager.py exactly."""
    from data_agent_common.agents.sanitize import SANITIZE_ACTIONS

    assert SANITIZE_ACTIONS == frozenset({"__error__", "__replan_failed__"})
    assert isinstance(SANITIZE_ACTIONS, frozenset)


def test_legacy_sanitize_alias_points_to_common_object():
    """The legacy private alias must be the same frozenset as the common one."""
    from data_agent_common.agents.sanitize import SANITIZE_ACTIONS
    from data_agent_refactored.agents.context_manager import _SANITIZE_ACTIONS

    assert _SANITIZE_ACTIONS is SANITIZE_ACTIONS


def test_error_sanitized_response_byte_for_byte():
    """The sanitized JSON response placeholder must match its historical form."""
    import json

    from data_agent_common.agents.sanitize import ERROR_SANITIZED_RESPONSE

    expected = json.dumps(
        {
            "thought": "My previous response had a format error. "
                       "I must return a valid JSON with keys: thought, action, action_input.",
            "action": "__error__",
            "action_input": {},
        },
        ensure_ascii=False,
    )
    assert ERROR_SANITIZED_RESPONSE == expected


def test_legacy_error_sanitized_response_is_same_string_object():
    """``data_agent_refactored.agents.context_manager.ERROR_SANITIZED_RESPONSE``
    must be the exact same string object exported by common (no shadowing).
    """
    from data_agent_common.agents.sanitize import ERROR_SANITIZED_RESPONSE as common
    from data_agent_refactored.agents.context_manager import (
        ERROR_SANITIZED_RESPONSE as legacy,
    )

    assert common == legacy


def test_execute_python_timeout_seconds_is_30():
    """Legacy ToolRegistry hard-codes 30s; the parity registry imports the constant."""
    from data_agent_common.tools.python_exec import EXECUTE_PYTHON_TIMEOUT_SECONDS

    assert EXECUTE_PYTHON_TIMEOUT_SECONDS == 30


def test_execute_python_timeout_legacy_path_resolves_same():
    """Legacy ``data_agent_refactored.tools.handlers.EXECUTE_PYTHON_TIMEOUT_SECONDS``
    must be the same int as the common one (no duplicate value)."""
    from data_agent_common.tools.python_exec import (
        EXECUTE_PYTHON_TIMEOUT_SECONDS as common,
    )
    from data_agent_refactored.tools.handlers import (
        EXECUTE_PYTHON_TIMEOUT_SECONDS as legacy,
    )

    assert common == legacy
