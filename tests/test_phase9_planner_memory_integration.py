from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import FakeListChatModel

import data_agent_langchain.agents.planner_node as planner_module
from data_agent_langchain.config import MemoryConfig, default_app_config
from data_agent_langchain.memory.base import MemoryRecord
from data_agent_langchain.memory.factory import build_store
from data_agent_langchain.memory.types import MemoryHit


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


def _base_state(task_state: dict, **extra) -> dict:
    state = {
        **task_state,
        "mode": "plan_solve",
        "action_mode": "json_action",
        "steps": [],
        "plan_index": 0,
        "replan_used": 0,
        "step_index": 0,
        "max_steps": 20,
    }
    state.update(extra)
    return state


def _planner_config():
    return {
        "configurable": {
            "llm": FakeListChatModel(
                responses=['```json\n{"plan":["List files","Read data"]}\n```']
            )
        }
    }


def test_planner_node_adds_dataset_memory_hits_when_enabled(
    tmp_path: Path, monkeypatch
):
    memory_cfg = MemoryConfig(mode="read_only_dataset", path=tmp_path)
    store = build_store(memory_cfg)
    store.put(
        MemoryRecord(
            id="dk:ds:a.csv",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={
                "file_path": "a.csv",
                "file_kind": "csv",
                "schema": {"x": "int"},
                "row_count_estimate": 1,
            },
            created_at=datetime(2026, 1, 1),
        )
    )
    app_config = replace(default_app_config(), memory=memory_cfg)
    monkeypatch.setattr(planner_module, "_safe_get_app_config", lambda: app_config)

    output = planner_module.planner_node(
        _base_state(_make_task(tmp_path / "ds")),
        config=_planner_config(),
    )

    assert output["plan"] == ["List files", "Read data"]
    assert output["plan_index"] == 0
    assert output["replan_used"] == 0
    assert output["memory_hits"][0].record_id == "dk:ds:a.csv"


def test_planner_node_adds_dataset_memory_hits_when_plan_generation_falls_back(
    tmp_path: Path, monkeypatch
):
    memory_cfg = MemoryConfig(mode="read_only_dataset", path=tmp_path)
    store = build_store(memory_cfg)
    store.put(
        MemoryRecord(
            id="dk:ds:fallback.csv",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={
                "file_path": "fallback.csv",
                "file_kind": "csv",
                "schema": {"value": "int"},
                "row_count_estimate": 3,
            },
            created_at=datetime(2026, 1, 1),
        )
    )
    app_config = replace(default_app_config(), memory=memory_cfg)
    monkeypatch.setattr(planner_module, "_safe_get_app_config", lambda: app_config)

    output = planner_module.planner_node(
        _base_state(_make_task(tmp_path / "ds")),
        config={"configurable": {"llm": FakeListChatModel(responses=["not json"])}},
    )

    assert output["plan"] == planner_module.FALLBACK_PLAN
    [step] = output["steps"]
    assert step.action == "__plan_generation__"
    assert step.ok is False
    assert step.observation == {"ok": False, "plan": planner_module.FALLBACK_PLAN}
    assert output["memory_hits"][0].record_id == "dk:ds:fallback.csv"


def test_planner_node_omits_memory_hits_when_memory_disabled(
    tmp_path: Path, monkeypatch
):
    app_config = replace(
        default_app_config(),
        memory=MemoryConfig(mode="disabled", path=tmp_path),
    )
    monkeypatch.setattr(planner_module, "_safe_get_app_config", lambda: app_config)

    output = planner_module.planner_node(
        _base_state(_make_task(tmp_path / "ds")),
        config=_planner_config(),
    )

    assert output["plan"] == ["List files", "Read data"]
    assert output["plan_index"] == 0
    assert output["replan_used"] == 0
    assert "memory_hits" not in output


def test_planner_node_recalls_with_planner_node_label_and_runtime_config(
    tmp_path: Path, monkeypatch
):
    calls = []
    app_config = replace(
        default_app_config(),
        memory=MemoryConfig(mode="read_only_dataset", path=tmp_path),
    )
    monkeypatch.setattr(planner_module, "_safe_get_app_config", lambda: app_config)

    def fake_recall_dataset_facts(memory_cfg, dataset, node, config):
        calls.append(
            {
                "memory_cfg": memory_cfg,
                "dataset": dataset,
                "node": node,
                "config": config,
            }
        )
        return [
            MemoryHit(
                record_id="dk:ds:a.csv",
                namespace="dataset:ds",
                score=1.0,
                summary="File: a.csv",
            )
        ]

    monkeypatch.setattr(
        planner_module, "recall_dataset_facts", fake_recall_dataset_facts
    )
    runtime_config = _planner_config()

    output = planner_module.planner_node(
        _base_state(_make_task(tmp_path / "ds")),
        config=runtime_config,
    )

    assert calls == [
        {
            "memory_cfg": app_config.memory,
            "dataset": "ds",
            "node": "planner_node",
            "config": runtime_config,
        }
    ]
    assert output["memory_hits"][0].record_id == "dk:ds:a.csv"
