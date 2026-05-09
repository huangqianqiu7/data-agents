"""
Phase 2.5 working-memory tests.

LANGCHAIN_MIGRATION_PLAN.md §8: ``memory/working.py`` is the new home of
the scratchpad logic that used to live in
``data_agent_refactored.agents.context_manager``. The contract has three
moving parts:

    1. ``render_step_messages`` returns the right LangChain messages for
       each ``action_mode``.
    2. ``SANITIZE_ACTIONS`` steps replace the raw response with the
       ``ERROR_SANITIZED_RESPONSE`` placeholder.
    3. The L2 selector pins successful data-preview steps and FIFO-evicts
       non-pinned steps when over budget.

The cross-backend parity assertions at the end ensure that the langchain
local copies of ``build_observation_prompt`` and ``DATA_PREVIEW_ACTIONS``
remain byte-for-byte compatible with the legacy reference (still archived
under ``src/废弃/``).
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from data_agent_langchain.agents.runtime import StepRecord
from data_agent_langchain.agents.sanitize import (
    ERROR_SANITIZED_RESPONSE,
    SANITIZE_ACTIONS,
)
from data_agent_langchain.memory.working import (
    build_scratchpad_messages,
    render_step_messages,
    select_steps_for_context,
    truncate_observation,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _step(
    *,
    idx: int,
    action: str,
    ok: bool = True,
    raw: str = "",
    obs: dict[str, Any] | None = None,
    thought: str = "",
    action_input: dict[str, Any] | None = None,
) -> StepRecord:
    """Build a StepRecord with sane defaults for unit tests."""
    return StepRecord(
        step_index=idx,
        thought=thought,
        action=action,
        action_input=dict(action_input or {}),
        raw_response=raw,
        observation=dict(obs or {"ok": ok, "tool": action, "content": "x"}),
        ok=ok,
    )


# ---------------------------------------------------------------------------
# truncate_observation
# ---------------------------------------------------------------------------

def test_truncate_observation_passthrough_when_short():
    obs = {"ok": True, "tool": "list_context", "content": "abc"}
    assert truncate_observation(obs, max_chars=100) == obs


def test_truncate_observation_truncates_long_content():
    obs = {"ok": True, "tool": "read_doc", "content": "x" * 500}
    out = truncate_observation(obs, max_chars=50)
    assert len(out["content"]) > 50  # includes the truncation marker
    assert out["content"].startswith("x" * 50)
    assert "truncated" in out["content"]
    # Other fields preserved.
    assert out["ok"] is True
    assert out["tool"] == "read_doc"


# ---------------------------------------------------------------------------
# render_step_messages — json_action mode
# ---------------------------------------------------------------------------

def test_render_step_messages_json_action_pair():
    step = _step(idx=1, action="list_context", raw='{"thought":"a","action":"list_context","action_input":{}}')
    msgs = render_step_messages(step, action_mode="json_action")
    assert len(msgs) == 2
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].content == step.raw_response
    assert isinstance(msgs[1], HumanMessage)
    assert msgs[1].content.startswith("Observation:")


def test_render_step_messages_json_action_sanitize_action():
    """A SANITIZE_ACTIONS step replaces raw_response with the placeholder."""
    step = _step(idx=2, action="__error__", ok=False, raw="garbled")
    msgs = render_step_messages(step, action_mode="json_action")
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].content == ERROR_SANITIZED_RESPONSE


# ---------------------------------------------------------------------------
# render_step_messages — tool_calling mode
# ---------------------------------------------------------------------------

def test_render_step_messages_tool_calling_with_tool_calls():
    tool_calls = [{"name": "list_context", "args": {"max_depth": 4}, "id": "call_1", "type": "tool_call"}]
    step = _step(idx=1, action="list_context", raw=json.dumps(tool_calls))
    msgs = render_step_messages(step, action_mode="tool_calling")
    assert len(msgs) == 2
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].tool_calls == tool_calls
    assert isinstance(msgs[1], ToolMessage)
    assert msgs[1].tool_call_id == "call_1"


def test_render_step_messages_tool_calling_falls_back_when_no_tool_calls():
    """If raw_response has no tool_calls shape, fall back to json_action layout."""
    step = _step(idx=1, action="list_context", raw="not json")
    msgs = render_step_messages(step, action_mode="tool_calling")
    assert isinstance(msgs[0], AIMessage)
    assert isinstance(msgs[1], HumanMessage)


def test_render_step_messages_tool_calling_sanitize_uses_placeholder():
    """Sanitize-action steps in tool_calling mode use the placeholder + json_action shape."""
    step = _step(idx=1, action="__replan_failed__", ok=False, raw="anything")
    msgs = render_step_messages(step, action_mode="tool_calling")
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].content == ERROR_SANITIZED_RESPONSE
    assert isinstance(msgs[1], HumanMessage)


# ---------------------------------------------------------------------------
# select_steps_for_context — pinning + FIFO eviction
# ---------------------------------------------------------------------------

def test_select_steps_pins_data_preview_actions():
    """Successful read_csv / read_json / read_doc / inspect_sqlite_schema are pinned."""
    steps = [
        _step(idx=1, action="read_csv", ok=True, raw="r1", obs={"ok": True, "content": "a" * 50}),
        _step(idx=2, action="execute_python", ok=True, raw="p1", obs={"ok": True, "content": "b" * 50}),
        _step(idx=3, action="read_json", ok=True, raw="r2", obs={"ok": True, "content": "c" * 50}),
        _step(idx=4, action="execute_python", ok=True, raw="p2", obs={"ok": True, "content": "d" * 50}),
    ]
    kept, n_omitted = select_steps_for_context(
        steps,
        base_tokens=0,
        max_context_tokens=80,  # tight budget
        max_obs_chars=200,
    )
    # Pinned indices (read_csv, read_json) must always survive.
    assert 0 in kept
    assert 2 in kept
    # At least one execute_python should be evicted under the tight budget.
    assert n_omitted >= 1


def test_select_steps_empty_returns_empty():
    kept, n = select_steps_for_context(
        [],
        base_tokens=0,
        max_context_tokens=1000,
        max_obs_chars=200,
    )
    assert kept == []
    assert n == 0


def test_select_steps_base_over_budget_still_keeps_one_observation_window():
    """When base prompt alone exceeds the budget, give history a small head-room."""
    steps = [_step(idx=1, action="list_context", ok=True, raw="r", obs={"ok": True, "content": "x"})]
    kept, _n = select_steps_for_context(
        steps,
        base_tokens=10_000,
        max_context_tokens=100,
        max_obs_chars=200,
    )
    # Step survives because head-room (max_obs_chars) is granted.
    assert kept == [0]


# ---------------------------------------------------------------------------
# build_scratchpad_messages — top-level entry point
# ---------------------------------------------------------------------------

def test_build_scratchpad_messages_appends_to_base():
    """Base messages come first, untouched; history follows."""
    base: list[BaseMessage] = [
        SystemMessage(content="sys"),
        HumanMessage(content="user"),
    ]
    steps = [_step(idx=1, action="list_context", ok=True)]
    result = build_scratchpad_messages(steps, base, action_mode="json_action")
    assert result[0] is base[0]
    assert result[1] is base[1]
    assert len(result) == len(base) + 2  # one step → one (assistant, user) pair


def test_build_scratchpad_messages_empty_steps_returns_base_copy():
    base: list[BaseMessage] = [SystemMessage(content="sys")]
    result = build_scratchpad_messages([], base, action_mode="json_action")
    assert result == base
    assert result is not base  # copy, not aliased


def test_build_scratchpad_messages_emits_omission_summary():
    """When eviction happens, a summary message is inserted."""
    base: list[BaseMessage] = [SystemMessage(content="sys")]
    steps = [
        _step(idx=i, action="execute_python", ok=False, raw="x" * 200, obs={"ok": False, "content": "y" * 200})
        for i in range(1, 6)
    ]
    # Budget tight enough to force at least one eviction.
    result = build_scratchpad_messages(
        steps, base,
        action_mode="json_action",
        max_obs_chars=50,
        max_context_tokens=80,
    )
    # First non-base message should be the omission summary.
    omission = result[1]
    assert isinstance(omission, HumanMessage)
    assert "earlier step(s) omitted" in omission.content


# ---------------------------------------------------------------------------
# Cross-backend parity sanity (legacy archive under src/废弃/)
# ---------------------------------------------------------------------------

def test_observation_prompt_byte_for_byte_with_legacy():
    """``build_observation_prompt`` is shared between backends (Phase 1.5)."""
    from data_agent_common.agents.prompts import build_observation_prompt as common_fn
    from data_agent_refactored.agents.prompt import build_observation_prompt as legacy_fn
    from data_agent_langchain.agents.observation_prompt import (
        build_observation_prompt as new_fn,
    )

    obs = {"ok": True, "tool": "list_context", "content": {"x": [1, 2, 3]}}
    assert common_fn(obs) == legacy_fn(obs)
    assert common_fn is legacy_fn  # actually the same function object (refactored re-exports common)
    # langchain owns its own copy; assert byte-for-byte parity.
    assert new_fn(obs) == common_fn(obs)


def test_data_preview_actions_identity_across_packages():
    """Legacy backend (common+refactored) shares one frozenset; langchain has a byte-equal copy."""
    from data_agent_common.agents.gate import DATA_PREVIEW_ACTIONS as common_set
    from data_agent_refactored.agents.data_preview_gate import (
        DATA_PREVIEW_ACTIONS as legacy_set,
    )
    from data_agent_langchain.tools.descriptions import (
        DATA_PREVIEW_ACTIONS as new_set,
    )

    # The legacy backend (common+refactored) still shares one frozenset.
    assert common_set is legacy_set
    # data_agent_langchain owns its own copy; assert byte-for-byte parity.
    assert new_set == common_set


def test_sanitize_actions_match_common():
    """langchain SANITIZE_ACTIONS must match common's set byte-for-byte."""
    from data_agent_common.agents.sanitize import SANITIZE_ACTIONS as common_set

    assert SANITIZE_ACTIONS == common_set