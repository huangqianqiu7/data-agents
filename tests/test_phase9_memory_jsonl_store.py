from pathlib import Path

from data_agent_langchain.memory.base import MemoryRecord
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore


def test_put_then_list(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    rec = MemoryRecord(
        id="dk:ds:a.csv",
        namespace="dataset:ds",
        kind="dataset_knowledge",
        payload={"file_path": "a.csv"},
    )
    store.put(rec)
    results = store.list("dataset:ds", limit=5)
    assert len(results) == 1
    assert results[0].id == "dk:ds:a.csv"
    assert results[0].payload["file_path"] == "a.csv"


def test_list_returns_recent_first(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    for i in range(3):
        store.put(
            MemoryRecord(
                id=f"x{i}",
                namespace="dataset:ds",
                kind="dataset_knowledge",
                payload={"i": i},
            )
        )
    results = store.list("dataset:ds", limit=2)
    assert [r.id for r in results] == ["x2", "x1"]


def test_namespace_isolated(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(
        MemoryRecord(
            id="a",
            namespace="dataset:ds1",
            kind="dataset_knowledge",
            payload={},
        )
    )
    assert store.list("dataset:ds2") == []


def test_empty_namespace_returns_empty(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    assert store.list("dataset:none") == []


def test_get_returns_latest_or_none(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(
        MemoryRecord(
            id="a",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={"v": 1},
        )
    )
    store.put(
        MemoryRecord(
            id="a",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={"v": 2},
        )
    )
    rec = store.get("dataset:ds", "a")
    assert rec is not None
    assert rec.payload["v"] == 2
    assert store.get("dataset:ds", "missing") is None


def test_delete_writes_tombstone(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(
        MemoryRecord(
            id="a",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={},
        )
    )
    store.delete("dataset:ds", "a")
    assert store.list("dataset:ds") == []
