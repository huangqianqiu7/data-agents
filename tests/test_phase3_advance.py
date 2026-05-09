"""
Phase 3 — unit tests for ``advance_node`` (rules 1-6).

Covers:
  - Rule 1: ``answer`` set or ``last_tool_is_terminal`` → done.
  - Rule 2: ``step_index >= max_steps`` → done.
  - Rule 3: ``last_error_kind == gate_block`` → continue.
  - Rule 5 (v4 M5): only ``tool_*`` errors trigger replan_required;
    ``parse_error`` / ``model_error`` / ``unknown_tool`` do not.
  - Rule 5: replan budget exhausted → continue (never replan).
  - Plan-and-Solve plan_index advance + FALLBACK_STEP append (D5).
"""
from __future__ import annotations

from data_agent_common.benchmark.schema import AnswerTable
from data_agent_common.constants import FALLBACK_STEP_PROMPT
from data_agent_langchain.agents.advance_node import (
    _REPLAN_TRIGGER_ERROR_KINDS,
    advance_node,
)


def _state(**kwargs) -> dict:
    base = {
        "step_index": 1,
        "max_steps": 16,
        "answer": None,
        "last_tool_ok": True,
        "last_tool_is_terminal": False,
        "last_error_kind": None,
        "mode": "react",
        "replan_used": 0,
    }
    base.update(kwargs)
    return base


# --- Rule 1: terminal --------------------------------------------------

def test_rule1_answer_set_yields_done():
    at = AnswerTable(columns=["x"], rows=[[1]])
    assert advance_node(_state(answer=at)) == {"subgraph_exit": "done"}


def test_rule1_last_tool_is_terminal_yields_done():
    assert advance_node(_state(last_tool_is_terminal=True)) == {"subgraph_exit": "done"}


# --- Rule 2: step budget ------------------------------------------------

def test_rule2_step_budget_exhausted_yields_done():
    assert advance_node(_state(step_index=16, max_steps=16)) == {"subgraph_exit": "done"}


# --- Rule 3: gate_block continues --------------------------------------

def test_rule3_gate_block_continues():
    update = advance_node(_state(last_tool_ok=False, last_error_kind="gate_block"))
    assert update["subgraph_exit"] == "continue"


# --- Rule 5: M5 white-list ---------------------------------------------

def test_rule5_white_list_contents():
    """Sanity-check that the v4 M5 white-list has exactly the three tool_* kinds."""
    assert _REPLAN_TRIGGER_ERROR_KINDS == frozenset(
        {"tool_error", "tool_timeout", "tool_validation"}
    )


def test_rule5_tool_error_triggers_replan():
    update = advance_node(_state(
        last_tool_ok=False, last_error_kind="tool_error", replan_used=0,
    ))
    assert update["subgraph_exit"] == "replan_required"


def test_rule5_tool_timeout_triggers_replan():
    update = advance_node(_state(
        last_tool_ok=False, last_error_kind="tool_timeout",
    ))
    assert update["subgraph_exit"] == "replan_required"


def test_rule5_tool_validation_triggers_replan():
    update = advance_node(_state(
        last_tool_ok=False, last_error_kind="tool_validation",
    ))
    assert update["subgraph_exit"] == "replan_required"


def test_rule5_parse_error_does_not_replan():
    update = advance_node(_state(
        last_tool_ok=False, last_error_kind="parse_error",
    ))
    assert update["subgraph_exit"] == "continue"


def test_rule5_model_error_does_not_replan():
    update = advance_node(_state(
        last_tool_ok=False, last_error_kind="model_error",
    ))
    assert update["subgraph_exit"] == "continue"


def test_rule5_unknown_tool_does_not_replan():
    update = advance_node(_state(
        last_tool_ok=False, last_error_kind="unknown_tool",
    ))
    assert update["subgraph_exit"] == "continue"


def test_rule5_default_replan_budget_allows_three_used_replans():
    update = advance_node(_state(
        last_tool_ok=False, last_error_kind="tool_error", replan_used=3,
    ))
    assert update["subgraph_exit"] == "replan_required"


def test_rule5_replan_budget_exhausted_yields_continue():
    update = advance_node(_state(
        last_tool_ok=False, last_error_kind="tool_error", replan_used=4,
    ))
    # default max_replans is 4; replan_used == 4 → cannot replan.
    assert update["subgraph_exit"] == "continue"


# --- Rule 4 (Plan-and-Solve): plan_index advance + FALLBACK_STEP -------

def test_plan_solve_advance_index_on_success():
    update = advance_node(_state(
        mode="plan_solve",
        plan=["A", "B", "C"],
        plan_index=0,
        last_tool_ok=True,
    ))
    assert update["plan_index"] == 1


def test_plan_solve_appends_fallback_when_plan_exhausted():
    update = advance_node(_state(
        mode="plan_solve",
        plan=["A"],
        plan_index=0,
        last_tool_ok=True,  # advances plan_index → 1, len(plan) == 1 → fallback
    ))
    assert update["plan"][-1] == FALLBACK_STEP_PROMPT
    assert update["plan_index"] == len(update["plan"]) - 1


def test_plan_solve_no_double_fallback():
    update = advance_node(_state(
        mode="plan_solve",
        plan=["A", FALLBACK_STEP_PROMPT],
        plan_index=1,  # already on fallback
        last_tool_ok=False,  # no advance
        last_error_kind=None,
    ))
    # No double-append, plan stays the same length.
    assert update.get("plan", ["A", FALLBACK_STEP_PROMPT]) == ["A", FALLBACK_STEP_PROMPT]


# --- Default rule 6: continue -----------------------------------------

def test_rule6_default_continue():
    update = advance_node(_state(last_tool_ok=True, last_error_kind=None))
    assert update["subgraph_exit"] == "continue"
