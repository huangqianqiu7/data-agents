"""
Phase 3 — unit tests for ``gate_node`` (L1 / L2 / L3 escalation).

Covers:
  - Pass-through: ungated action → gate_decision="pass", skip_tool=False.
  - L1 block: first time a gated action is requested without preview.
  - L1 with preview already done → pass-through.
  - L2: ``consecutive_gate_blocks == max_gate_retries`` → plan rewrite.
  - L3 (v4 M1): ``consecutive_gate_blocks >= max + 2`` → forced inject
    list_context AND ``skip_tool=False`` (fall-through).
  - L3 forced step is appended in the same node call.
  - parse_error transparency: gate_node returns immediately.
  - Disabling ``require_data_preview_before_compute`` skips gating entirely.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from data_agent_common.agents.gate import GATE_REMINDER
from data_agent_langchain.agents.gate import gate_node
from data_agent_langchain.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    ToolsConfig,
)
from data_agent_langchain.runtime.context import _APP_CONFIG


@pytest.fixture()
def synthetic_task(tmp_path: Path) -> dict:
    """Create a minimal task tree with task_test/context/* and return base state."""
    root = tmp_path
    task_dir = root / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_test", "difficulty": "easy", "question": "Q?"}),
        encoding="utf-8",
    )
    csv_path = context_dir / "matches.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id"])
        w.writerow([1])
    return {"dataset_root": str(root), "task_id": "task_test"}


@pytest.fixture()
def app_config_in_context():
    """Install a fresh AppConfig in the contextvar; restore on teardown."""
    cfg = AppConfig(
        dataset=DatasetConfig(),
        agent=AgentConfig(max_gate_retries=2),
        tools=ToolsConfig(),
    )
    token = _APP_CONFIG.set(cfg)
    try:
        yield cfg
    finally:
        _APP_CONFIG.reset(token)


def _state(task_state: dict, *, action: str, blocks: int = 0, **extra) -> dict:
    base = {
        **task_state,
        "step_index": 1,
        "thought": "",
        "raw_response": "",
        "phase": "execution",
        "plan_progress": "",
        "plan_step_description": "",
        "action": action,
        "action_input": {},
        "consecutive_gate_blocks": blocks,
        "steps": [],
        "skip_tool": False,
        "last_error_kind": None,
    }
    base.update(extra)
    return base


# --- pass-through ----------------------------------------------------

def test_gate_passes_when_action_not_gated(synthetic_task, app_config_in_context):
    state = _state(synthetic_task, action="list_context")
    update = gate_node(state)
    assert update["gate_decision"] == "pass"
    assert update["skip_tool"] is False
    assert update["consecutive_gate_blocks"] == 0


def test_gate_passes_when_preview_already_done(synthetic_task, app_config_in_context):
    """If steps contain a successful read_csv, gating clears."""
    from data_agent_common.agents.runtime import StepRecord
    preview = StepRecord(
        step_index=1, thought="", action="read_csv", action_input={},
        raw_response="", observation={"ok": True}, ok=True,
    )
    state = _state(synthetic_task, action="execute_python", steps=[preview])
    update = gate_node(state)
    assert update["gate_decision"] == "pass"
    assert update["skip_tool"] is False


def test_gate_parse_error_transparency(synthetic_task, app_config_in_context):
    state = _state(synthetic_task, action="", last_error_kind="parse_error")
    update = gate_node(state)
    assert update["gate_decision"] == "pass"
    assert update["skip_tool"] is True


# --- L1 block --------------------------------------------------------

def test_gate_L1_blocks_gated_action_without_preview(synthetic_task, app_config_in_context):
    state = _state(synthetic_task, action="execute_python", blocks=0)
    update = gate_node(state)
    assert update["gate_decision"] == "block"
    assert update["skip_tool"] is True
    assert update["consecutive_gate_blocks"] == 1
    assert update["last_error_kind"] == "gate_block"
    [step] = update["steps"]
    assert step.action == "execute_python"
    assert step.observation["error"] == GATE_REMINDER
    assert step.ok is False


# --- L2: plan rewrite ------------------------------------------------

def test_gate_L2_rewrites_current_plan_step(synthetic_task, app_config_in_context):
    state = _state(
        synthetic_task,
        action="execute_python",
        blocks=1,  # this call -> 2 == max_gate_retries
        plan=["A", "B", "C"],
        plan_index=1,
    )
    update = gate_node(state)
    assert update["gate_decision"] == "block"
    assert update["skip_tool"] is True
    assert update["consecutive_gate_blocks"] == 2
    new_plan = update["plan"]
    assert new_plan[1].startswith("MANDATORY: Inspect data files first")
    assert new_plan[0] == "A"
    assert new_plan[2] == "C"


def test_gate_L2_no_plan_no_rewrite(synthetic_task, app_config_in_context):
    state = _state(synthetic_task, action="execute_python", blocks=1)
    update = gate_node(state)
    assert "plan" not in update
    assert update["consecutive_gate_blocks"] == 2


# --- L3: forced inject + fall-through ---------------------------------

def test_gate_L3_forces_list_context_and_falls_through(synthetic_task, app_config_in_context):
    state = _state(
        synthetic_task,
        action="execute_python",
        blocks=3,  # this call -> 4 == max(2) + 2
    )
    update = gate_node(state)
    # v4 M1: skip_tool=False -> tool_node will execute the original action.
    assert update["gate_decision"] == "forced_inject"
    assert update["skip_tool"] is False
    assert update["consecutive_gate_blocks"] == 0  # reset
    steps = update["steps"]
    assert len(steps) == 2
    assert steps[0].action == "execute_python"
    assert steps[1].action == "list_context"
    assert steps[1].ok is True
    assert update["last_tool_ok"] is True
    assert update["discovery_done"] is True


# --- Configuration toggle ---------------------------------------------

def test_gate_disabled_by_config(synthetic_task):
    cfg = AppConfig(agent=AgentConfig(require_data_preview_before_compute=False))
    token = _APP_CONFIG.set(cfg)
    try:
        state = _state(synthetic_task, action="execute_python")
        update = gate_node(state)
        assert update["gate_decision"] == "pass"
    finally:
        _APP_CONFIG.reset(token)
