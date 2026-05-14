from __future__ import annotations

from datetime import datetime
from pathlib import Path

from data_agent_langchain.config import MemoryConfig
from data_agent_langchain.memory.base import MemoryRecord
from data_agent_langchain.memory.factory import build_store
from data_agent_langchain.memory.types import MemoryHit


def _memory_cfg(tmp_path: Path, *, mode: str = "read_only_dataset") -> MemoryConfig:
    return MemoryConfig(mode=mode, path=tmp_path, retrieval_max_results=3)


def test_recall_dataset_facts_returns_memory_hits(tmp_path: Path):
    from data_agent_langchain.agents.memory_recall import recall_dataset_facts

    cfg = _memory_cfg(tmp_path)
    store = build_store(cfg)
    store.put(
        MemoryRecord(
            id="dk:ds:a.csv",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={
                "file_path": "a.csv",
                "file_kind": "csv",
                "schema": {"x": "int", "y": "str"},
                "row_count_estimate": 12,
                "sample_columns": ["x", "y"],
                "answer": "must not leak",
            },
            created_at=datetime(2026, 1, 1),
        )
    )

    hits = recall_dataset_facts(cfg, dataset="ds", node="planner", config=None)

    assert len(hits) == 1
    assert isinstance(hits[0], MemoryHit)
    assert hits[0].record_id == "dk:ds:a.csv"
    assert hits[0].namespace == "dataset:ds"
    assert hits[0].score == 1.0
    assert "a.csv" in hits[0].summary
    assert "csv" in hits[0].summary
    assert "x, y" in hits[0].summary
    assert "Rows~: 12" in hits[0].summary
    assert "must not leak" not in hits[0].summary


def test_recall_disabled_returns_empty(monkeypatch):
    from data_agent_langchain.agents import memory_recall

    def fail_build_store(_memory_cfg):
        raise AssertionError("disabled recall should not build or read store")

    monkeypatch.setattr(memory_recall, "build_store", fail_build_store)

    hits = memory_recall.recall_dataset_facts(
        MemoryConfig(mode="disabled"), dataset="ds", node="planner", config=None
    )

    assert hits == []


def test_recall_unknown_mode_returns_empty(monkeypatch):
    from data_agent_langchain.agents import memory_recall

    def fail_build_store(_memory_cfg):
        raise AssertionError("unknown mode should fail closed")

    monkeypatch.setattr(memory_recall, "build_store", fail_build_store)

    hits = memory_recall.recall_dataset_facts(
        MemoryConfig(mode="disable"), dataset="ds", node="planner", config=None
    )

    assert hits == []


def test_recall_clamps_negative_k(tmp_path: Path, monkeypatch):
    from data_agent_langchain.agents import memory_recall

    events = []
    monkeypatch.setattr(
        memory_recall,
        "dispatch_observability_event",
        lambda name, data, config: events.append((name, data)),
    )
    cfg = MemoryConfig(
        mode="read_only_dataset", path=tmp_path, retrieval_max_results=-1
    )
    store = build_store(cfg)
    store.put(
        MemoryRecord(
            id="dk:ds:a.csv",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={
                "file_path": "a.csv",
                "file_kind": "csv",
                "schema": {"a": "string"},
                "row_count_estimate": 1,
            },
            created_at=datetime(2026, 1, 1),
        )
    )

    hits = memory_recall.recall_dataset_facts(
        cfg, dataset="ds", node="planner", config=None
    )

    assert hits == []
    assert events[0][1]["k"] == 0
    assert events[0][1]["hit_ids"] == []


def test_recall_summary_falls_back_to_schema_keys(tmp_path: Path):
    from data_agent_langchain.agents.memory_recall import recall_dataset_facts

    cfg = _memory_cfg(tmp_path)
    store = build_store(cfg)
    store.put(
        MemoryRecord(
            id="dk:ds:b.csv",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={
                "file_path": "b.csv",
                "file_kind": "csv",
                "schema": {"first": "string", "second": "int"},
                "row_count_estimate": None,
            },
            created_at=datetime(2026, 1, 1),
        )
    )

    [hit] = recall_dataset_facts(cfg, dataset="ds", node="planner", config=None)

    assert "first, second" in hit.summary
    assert "Rows~: ?" in hit.summary


def test_recall_dispatches_memory_recall_event(tmp_path: Path, monkeypatch):
    from data_agent_langchain.agents import memory_recall

    events = []

    def fake_dispatch(name, data, config):
        events.append((name, data, config))

    monkeypatch.setattr(memory_recall, "dispatch_observability_event", fake_dispatch)
    cfg = _memory_cfg(tmp_path)
    store = build_store(cfg)
    store.put(
        MemoryRecord(
            id="dk:ds:a.csv",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={
                "file_path": "a.csv",
                "file_kind": "csv",
                "schema": {"x": "int"},
                "row_count_estimate": None,
            },
            created_at=datetime(2026, 1, 1),
        )
    )
    runnable_config = {"callbacks": []}

    hits = memory_recall.recall_dataset_facts(
        cfg, dataset="ds", node="model", config=runnable_config
    )

    assert [hit.record_id for hit in hits] == ["dk:ds:a.csv"]
    assert events == [
        (
            "memory_recall",
            {
                "node": "model",
                "namespace": "dataset:ds",
                "k": 3,
                "hit_ids": ["dk:ds:a.csv"],
                "reason": "exact_namespace",
            },
            runnable_config,
        )
    ]
