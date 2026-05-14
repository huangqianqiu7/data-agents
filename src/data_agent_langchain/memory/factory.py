"""Memory subsystem factories; call at process/node boundaries only."""
from __future__ import annotations

from data_agent_langchain.config import MemoryConfig
from data_agent_langchain.memory.base import MemoryStore, Retriever
from data_agent_langchain.memory.retrievers.exact import ExactNamespaceRetriever
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore
from data_agent_langchain.memory.writers.store_backed import StoreBackedMemoryWriter


def build_store(cfg: MemoryConfig) -> MemoryStore:
    if cfg.store_backend == "jsonl":
        return JsonlMemoryStore(root=cfg.path)
    raise ValueError(f"unsupported store_backend: {cfg.store_backend!r}")


def build_retriever(cfg: MemoryConfig, *, store: MemoryStore) -> Retriever:
    if cfg.retriever_type == "exact":
        return ExactNamespaceRetriever(store)
    raise ValueError(f"unsupported retriever_type: {cfg.retriever_type!r}")


def build_writer(cfg: MemoryConfig, *, store: MemoryStore) -> StoreBackedMemoryWriter:
    return StoreBackedMemoryWriter(store, mode=cfg.mode)


__all__ = ["build_retriever", "build_store", "build_writer"]
