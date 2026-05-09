"""
Phase 3 — unit tests for ``parse_action_node`` (v4 M2 contract).

Covers:
  - Successful JSON-action parse fills thought / action / action_input.
  - Successful tool-calling parse with one tool_call.
  - Multi tool_calls rejection (C15) → parse_error step.
  - Empty / malformed payloads → parse_error step.
  - parse_error path appends a step using state["step_index"] (D2:
    parse_action does NOT advance step_index).
  - parse_error path forwards phase / plan_progress / plan_step_description.
"""
from __future__ import annotations

import json

from data_agent_langchain.agents.parse_action import parse_action_node


def _state(raw: str = "", *, mode: str = "json_action", **extra) -> dict:
    base = {
        "raw_response": raw,
        "action_mode": mode,
        "step_index": 3,
        "phase": "execution",
        "plan_progress": "1/3",
        "plan_step_description": "List context",
    }
    base.update(extra)
    return base


# --- json_action mode ---------------------------------------------------

def test_parse_json_action_success():
    raw = '```json\n{"thought":"t","action":"list_context","action_input":{"max_depth":4}}\n```'
    update = parse_action_node(_state(raw))
    assert update == {
        "thought": "t",
        "action": "list_context",
        "action_input": {"max_depth": 4},
    }


def test_parse_json_action_malformed_yields_parse_error():
    update = parse_action_node(_state("not a json"))
    assert update["last_tool_ok"] is False
    assert update["last_error_kind"] == "parse_error"
    [step] = update["steps"]
    assert step.action == "__error__"
    assert step.ok is False
    assert step.observation["ok"] is False
    assert "Parse failed" in step.observation["error"]
    # D2: parse_action does NOT advance step_index — uses current value.
    assert step.step_index == 3
    # phase / plan_* are forwarded from state.
    assert step.phase == "execution"
    assert step.plan_progress == "1/3"
    assert step.plan_step_description == "List context"


def test_parse_json_action_missing_action_yields_parse_error():
    raw = '{"thought":"t","action":"","action_input":{}}'
    update = parse_action_node(_state(raw))
    assert update["last_error_kind"] == "parse_error"
    [step] = update["steps"]
    assert step.action == "__error__"


# --- tool_calling mode --------------------------------------------------

def test_parse_tool_calling_single_call_success():
    raw = json.dumps([{"name": "list_context", "args": {"max_depth": 4}}])
    update = parse_action_node(_state(raw, mode="tool_calling"))
    assert update == {
        "thought": "",
        "action": "list_context",
        "action_input": {"max_depth": 4},
    }


def test_parse_tool_calling_dict_payload_treated_as_single_call():
    raw = json.dumps({"name": "answer", "args": {"columns": ["x"], "rows": [[1]]}})
    update = parse_action_node(_state(raw, mode="tool_calling"))
    assert update["action"] == "answer"
    assert update["action_input"]["columns"] == ["x"]


def test_parse_tool_calling_empty_yields_parse_error():
    update = parse_action_node(_state("[]", mode="tool_calling"))
    assert update["last_error_kind"] == "parse_error"
    [step] = update["steps"]
    assert "no tool_calls" in step.observation["error"].lower()


def test_parse_tool_calling_multi_calls_rejected():
    raw = json.dumps([
        {"name": "list_context", "args": {}},
        {"name": "read_csv", "args": {"path": "x.csv"}},
    ])
    update = parse_action_node(_state(raw, mode="tool_calling"))
    assert update["last_error_kind"] == "parse_error"
    [step] = update["steps"]
    assert "multiple" in step.observation["error"].lower()


def test_parse_tool_calling_missing_name_yields_parse_error():
    raw = json.dumps([{"args": {}}])
    update = parse_action_node(_state(raw, mode="tool_calling"))
    assert update["last_error_kind"] == "parse_error"


def test_parse_tool_calling_blank_raw_yields_parse_error():
    update = parse_action_node(_state("", mode="tool_calling"))
    assert update["last_error_kind"] == "parse_error"
