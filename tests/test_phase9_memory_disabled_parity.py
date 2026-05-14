from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from langchain_core.language_models import FakeListChatModel

import data_agent_langchain.agents.planner_node as planner_module
from data_agent_langchain.config import MemoryConfig, default_app_config


def _make_task(dataset_root: Path) -> dict:
    task_dir = dataset_root / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_test", "difficulty": "easy", "question": "Q?"}),
        encoding="utf-8",
    )
    with (context_dir / "x.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["a"], [1]])
    return {"dataset_root": str(dataset_root), "task_id": "task_test"}


def _base_state(task_state: dict) -> dict:
    return {
        **task_state,
        "mode": "plan_solve",
        "action_mode": "json_action",
        "steps": [],
        "plan_index": 0,
        "replan_used": 0,
        "step_index": 0,
        "max_steps": 20,
    }


def test_planner_node_disabled_memory_mode_produces_no_memory_hits(
    tmp_path: Path, monkeypatch
):
    cfg = replace(
        default_app_config(),
        memory=MemoryConfig(mode="disabled", path=tmp_path),
    )
    monkeypatch.setattr(planner_module, "_safe_get_app_config", lambda: cfg)
    llm = FakeListChatModel(
        responses=['```json\n{"plan":["List files","Read data"]}\n```']
    )

    out = planner_module.planner_node(
        _base_state(_make_task(tmp_path / "ds")),
        config={"configurable": {"llm": llm}},
    )

    assert out["plan"] == ["List files", "Read data"]
    assert "memory_hits" not in out or out["memory_hits"] == []
