"""Retriever that returns latest records from an exact namespace."""
from __future__ import annotations

from data_agent_langchain.memory.base import MemoryStore, RetrievalResult


class ExactNamespaceRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def retrieve(
        self, query: str, *, namespace: str, k: int = 5
    ) -> list[RetrievalResult]:
        records = self._store.list(namespace, limit=k)
        return [
            RetrievalResult(record=rec, score=1.0, reason="exact_namespace")
            for rec in records
        ]


__all__ = ["ExactNamespaceRetriever"]
