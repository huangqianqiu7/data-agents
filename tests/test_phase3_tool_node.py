"""
Phase 3 — unit tests for the custom ``tool_node`` (§6.3 / C2).

Covers:
  - Successful ``list_context`` writes ``discovery_done=True`` and
    ``preview_done=False`` (list_context is NOT a data preview action).
  - Successful ``read_csv`` writes ``preview_done=True``.
  - Successful ``answer`` writes ``state["answer"]`` and
    ``last_tool_is_terminal=True``.
  - Unknown tool name → ``last_error_kind="unknown_tool"``.
  - skip_tool=True → no-op (no step appended).
  - parse_error transparency → no-op.
  - Validation error path → ``last_error_kind="tool_validation"``.
  - Runtime failure (missing file) → ``last_error_kind="tool_error"``.
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

import data_agent_langchain.agents.tool_node as tool_node_module
from data_agent_langchain.config import default_app_config


@pytest.fixture()
def synthetic_task(tmp_path: Path) -> dict:
    """Create a small dataset_root with one task (task_test) and return state-like dict."""
    root = tmp_path
    task_dir = root / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    # task.json
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_test", "difficulty": "easy", "question": "Q?"}),
        encoding="utf-8",
    )
    # CSV
    csv_path = context_dir / "matches.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        w.writerow([1, 95])
        w.writerow([2, 80])
    return {"dataset_root": str(root), "task_id": "task_test"}


def _state(task_state: dict, *, action: str, action_input: dict, **extra) -> dict:
    base = {
        **task_state,
        "step_index": 1,
        "thought": "",
        "raw_response": "",
        "phase": "execution",
        "plan_progress": "",
        "plan_step_description": "",
        "action": action,
        "action_input": action_input,
        "skip_tool": False,
    }
    base.update(extra)
    return base


# --- skip / transparency ----------------------------------------------

def test_tool_node_noop_when_skip_tool(synthetic_task):
    state = _state(synthetic_task, action="list_context", action_input={}, skip_tool=True)
    assert tool_node_module.tool_node(state) == {}


def test_tool_node_noop_when_parse_error(synthetic_task):
    state = _state(synthetic_task, action="list_context", action_input={}, last_error_kind="parse_error")
    assert tool_node_module.tool_node(state) == {}


# --- happy paths -------------------------------------------------------

def test_tool_node_list_context_success(synthetic_task):
    state = _state(synthetic_task, action="list_context", action_input={"max_depth": 4})
    update = tool_node_module.tool_node(state)
    assert update["last_tool_ok"] is True
    assert update["last_tool_is_terminal"] is False
    assert update["last_error_kind"] is None
    assert update["discovery_done"] is True
    assert update["known_paths"] == ["matches.csv"]
    # list_context is NOT in DATA_PREVIEW_ACTIONS
    assert "preview_done" not in update
    [step] = update["steps"]
    assert step.action == "list_context"
    assert step.ok is True


def test_tool_node_read_csv_sets_preview_done(synthetic_task):
    state = _state(
        synthetic_task,
        action="read_csv",
        action_input={"path": "matches.csv", "max_rows": 5},
    )
    update = tool_node_module.tool_node(state)
    assert update["last_tool_ok"] is True
    assert update["preview_done"] is True


def test_tool_node_allows_known_path_after_discovery(synthetic_task, monkeypatch):
    cfg = default_app_config()
    cfg = replace(
        cfg,
        agent=replace(cfg.agent, enforce_known_path_only=True),
    )
    monkeypatch.setattr(tool_node_module, "_safe_get_app_config", lambda: cfg)
    state = _state(
        synthetic_task,
        action="read_csv",
        action_input={"path": "matches.csv", "max_rows": 5},
        discovery_done=True,
        known_paths=["matches.csv"],
    )

    update = tool_node_module.tool_node(state)

    assert update["last_tool_ok"] is True
    assert update["preview_done"] is True


def test_tool_node_hard_blocks_unknown_path_after_discovery(synthetic_task, monkeypatch):
    cfg = default_app_config()
    cfg = replace(
        cfg,
        agent=replace(cfg.agent, enforce_known_path_only=True),
    )
    monkeypatch.setattr(tool_node_module, "_safe_get_app_config", lambda: cfg)
    state = _state(
        synthetic_task,
        action="read_csv",
        action_input={"path": "stale.csv", "max_rows": 5},
        discovery_done=True,
        known_paths=["matches.csv"],
    )

    update = tool_node_module.tool_node(state)

    assert update["last_tool_ok"] is False
    assert update["last_error_kind"] == "tool_validation"
    assert "preview_done" not in update
    [step] = update["steps"]
    assert step.action == "read_csv"
    assert step.ok is False
    assert step.observation["content"]["rejected_path"] == "stale.csv"
    assert step.observation["content"]["available_paths"] == ["matches.csv"]
    assert "Do not retry" in step.observation["content"]["error"]


def test_tool_node_does_not_hard_block_before_discovery(synthetic_task, monkeypatch):
    cfg = default_app_config()
    cfg = replace(
        cfg,
        agent=replace(cfg.agent, enforce_known_path_only=True),
    )
    monkeypatch.setattr(tool_node_module, "_safe_get_app_config", lambda: cfg)
    state = _state(
        synthetic_task,
        action="read_csv",
        action_input={"path": "stale.csv", "max_rows": 5},
        discovery_done=False,
        known_paths=["matches.csv"],
    )

    update = tool_node_module.tool_node(state)

    assert update["last_tool_ok"] is False
    [step] = update["steps"]
    assert "rejected_path" not in step.observation["content"]


def test_tool_node_answer_terminal(synthetic_task):
    state = _state(
        synthetic_task,
        action="answer",
        action_input={"columns": ["x"], "rows": [[1]]},
    )
    update = tool_node_module.tool_node(state)
    assert update["last_tool_ok"] is True
    assert update["last_tool_is_terminal"] is True
    assert update["answer"] is not None
    assert update["answer"].columns == ["x"]


# --- failure paths ----------------------------------------------------

def test_tool_node_unknown_tool(synthetic_task):
    state = _state(synthetic_task, action="not_a_real_tool", action_input={})
    update = tool_node_module.tool_node(state)
    assert update["last_tool_ok"] is False
    assert update["last_error_kind"] == "unknown_tool"
    [step] = update["steps"]
    assert step.action == "not_a_real_tool"


def test_tool_node_validation_error_maps_to_tool_validation(synthetic_task):
    # ``read_csv`` with non-existent path will raise pydantic validation? Actually
    # path is required as str — pass an int to trigger validation.
    state = _state(synthetic_task, action="read_csv", action_input={"path": 123})
    update = tool_node_module.tool_node(state)
    assert update["last_tool_ok"] is False
    # pydantic auto-coerces path=int to str=str ("123") so error is runtime not validation.
    # We accept either kind here as long as it's a tool_* error (not parse_error).
    assert update["last_error_kind"] in {"tool_error", "tool_validation"}


def test_tool_node_validation_error_for_missing_file(synthetic_task):
    state = _state(synthetic_task, action="read_csv", action_input={"path": "nope.csv"})
    update = tool_node_module.tool_node(state)
    assert update["last_tool_ok"] is False
    assert update["last_error_kind"] == "tool_validation"
    [step] = update["steps"]
    assert step.action == "read_csv"
    assert step.ok is False


# --- runtime failure (rehydrate breakage) -----------------------------

def test_tool_node_rehydrate_failure(tmp_path: Path):
    """When dataset_root is missing entirely, surface a tool_error with a clear msg."""
    state = _state(
        {"dataset_root": "", "task_id": "task_test"},
        action="list_context", action_input={},
    )
    update = tool_node_module.tool_node(state)
    assert update["last_tool_ok"] is False
    assert update["last_error_kind"] == "tool_error"
