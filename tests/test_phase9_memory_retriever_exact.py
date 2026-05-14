from datetime import datetime, timedelta
from pathlib import Path

from data_agent_langchain.memory.base import MemoryRecord
from data_agent_langchain.memory.retrievers.exact import ExactNamespaceRetriever
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore


def _seed(store: JsonlMemoryStore, n: int) -> None:
    base_time = datetime(2026, 1, 1)
    for i in range(n):
        store.put(
            MemoryRecord(
                id=f"r{i}",
                namespace="dataset:ds",
                kind="dataset_knowledge",
                payload={"i": i},
                created_at=base_time + timedelta(minutes=i),
            )
        )


def test_retrieve_returns_up_to_k(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    _seed(store, 5)
    store.put(
        MemoryRecord(
            id="other",
            namespace="dataset:other",
            kind="dataset_knowledge",
            payload={"i": "other"},
            created_at=datetime(2026, 1, 2),
        )
    )
    retriever = ExactNamespaceRetriever(store)
    results = retriever.retrieve("ignored", namespace="dataset:ds", k=2)
    assert len(results) == 2
    assert all(r.reason == "exact_namespace" for r in results)
    assert all(r.score == 1.0 for r in results)
    assert all(r.record.namespace == "dataset:ds" for r in results)
    assert [r.record.id for r in results] == ["r4", "r3"]
    assert [r.record.payload for r in results] == [{"i": 4}, {"i": 3}]


def test_retrieve_empty_namespace_returns_empty(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    retriever = ExactNamespaceRetriever(store)
    assert retriever.retrieve("q", namespace="dataset:none", k=5) == []
