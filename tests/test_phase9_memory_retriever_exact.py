from pathlib import Path

from data_agent_langchain.memory.base import MemoryRecord
from data_agent_langchain.memory.retrievers.exact import ExactNamespaceRetriever
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore


def _seed(store: JsonlMemoryStore, n: int) -> None:
    for i in range(n):
        store.put(
            MemoryRecord(
                id=f"r{i}",
                namespace="dataset:ds",
                kind="dataset_knowledge",
                payload={"i": i},
            )
        )


def test_retrieve_returns_up_to_k(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    _seed(store, 5)
    retriever = ExactNamespaceRetriever(store)
    results = retriever.retrieve("ignored", namespace="dataset:ds", k=2)
    assert len(results) == 2
    assert all(r.reason == "exact_namespace" for r in results)
    assert all(r.score == 1.0 for r in results)


def test_retrieve_empty_namespace_returns_empty(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    retriever = ExactNamespaceRetriever(store)
    assert retriever.retrieve("q", namespace="dataset:none", k=5) == []
