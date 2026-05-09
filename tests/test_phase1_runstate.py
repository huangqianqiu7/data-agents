"""
Phase 1 RunState shape verification.

Confirms that the ``RunState`` TypedDict declared in
``data_agent_langchain.runtime.state``:

  - Carries the v4 routing fields (gate_decision / skip_tool / last_*).
  - Declares ``steps`` with an ``Annotated[list, operator.add]`` reducer
    so concurrent node updates concatenate (LANGCHAIN_MIGRATION_PLAN.md
    §5.1 / C1).
  - Distinguishes ``tool_validation`` from ``tool_error`` in
    ``LastErrorKind`` (v4 E15).
  - Exposes the ``subgraph_exit`` enum (v3 D7 / v4 E14).

These are static-shape checks; they don't require LangGraph to be
installed because we only inspect the type annotations.
"""
from __future__ import annotations

import operator
import typing
from typing import get_type_hints


def test_run_state_has_required_routing_fields():
    """All explicit routing fields from §5.1 are declared."""
    from data_agent_langchain.runtime.state import RunState

    annotations = RunState.__annotations__
    expected_fields = {
        "task_id", "question", "difficulty", "dataset_root",
        "task_dir", "context_dir",
        "mode", "action_mode",
        "plan", "plan_index", "replan_used",
        "steps",
        "answer", "failure_reason",
        "discovery_done", "preview_done", "known_paths",
        "consecutive_gate_blocks", "gate_decision", "skip_tool",
        "raw_response", "thought", "action", "action_input",
        "last_tool_ok", "last_tool_is_terminal", "last_error_kind",
        "subgraph_exit",
        "step_index", "max_steps",
        "phase", "plan_progress", "plan_step_description",
    }
    missing = expected_fields - annotations.keys()
    assert not missing, f"RunState is missing required fields: {missing}"


def test_run_state_steps_uses_operator_add_reducer():
    """``steps`` MUST use ``Annotated[list, operator.add]`` (C1)."""
    from data_agent_langchain.runtime.state import RunState

    hints = get_type_hints(RunState, include_extras=True)
    steps_hint = hints["steps"]
    # ``Annotated[list[StepRecord], operator.add]`` exposes its metadata
    # via ``typing.get_args`` — the second element should be operator.add.
    assert typing.get_origin(steps_hint) is not None
    args = typing.get_args(steps_hint)
    assert len(args) >= 2, f"Expected Annotated[..., reducer], got {steps_hint!r}"
    assert operator.add in args, (
        "RunState['steps'] must be Annotated with operator.add as the reducer "
        "(LANGCHAIN_MIGRATION_PLAN.md §5.1 / C1)."
    )


def test_last_error_kind_distinguishes_validation_and_error():
    """v4 E15: tool_validation and tool_error are separate literals."""
    from data_agent_langchain.runtime.state import LastErrorKind

    args = set(typing.get_args(LastErrorKind))
    assert "tool_validation" in args, "Expected tool_validation literal (v4 E15)"
    assert "tool_error" in args, "Expected tool_error literal (v4 E15)"
    assert "tool_timeout" in args
    assert "parse_error" in args
    assert "unknown_tool" in args
    assert "model_error" in args
    assert "gate_block" in args
    assert "max_steps_exceeded" in args


def test_subgraph_exit_has_three_states():
    """v3 D7 / v4 E14: subgraph_exit literal carries continue/done/replan_required."""
    from data_agent_langchain.runtime.state import SubgraphExit

    args = set(typing.get_args(SubgraphExit))
    assert args == {"continue", "done", "replan_required"}


def test_gate_decision_has_three_states():
    """gate_decision literal: pass / block / forced_inject."""
    from data_agent_langchain.runtime.state import GateDecision

    args = set(typing.get_args(GateDecision))
    assert args == {"pass", "block", "forced_inject"}


def test_app_config_contextvar_raises_when_unset():
    """``get_current_app_config`` must raise a clear error when not initialised."""
    import pytest

    from data_agent_langchain.runtime.context import (
        _APP_CONFIG,
        get_current_app_config,
    )

    # Reset the contextvar in this test scope (use a fresh token).
    token = _APP_CONFIG.set(None)
    try:
        with pytest.raises(RuntimeError, match="AppConfig not initialised"):
            get_current_app_config()
    finally:
        _APP_CONFIG.reset(token)
