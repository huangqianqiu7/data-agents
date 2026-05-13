from pathlib import Path

from data_agent_langchain.config import MemoryConfig
from data_agent_langchain.memory.factory import (
    build_retriever,
    build_store,
    build_writer,
)
from data_agent_langchain.memory.retrievers.exact import ExactNamespaceRetriever
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore
from data_agent_langchain.memory.writers.store_backed import StoreBackedMemoryWriter


def test_build_store_jsonl(tmp_path: Path):
    cfg = MemoryConfig(mode="full", store_backend="jsonl", path=tmp_path)
    store = build_store(cfg)
    assert isinstance(store, JsonlMemoryStore)


def test_build_retriever_exact(tmp_path: Path):
    cfg = MemoryConfig(mode="full", retriever_type="exact", path=tmp_path)
    store = build_store(cfg)
    retriever = build_retriever(cfg, store=store)
    assert isinstance(retriever, ExactNamespaceRetriever)


def test_build_writer_carries_mode(tmp_path: Path):
    cfg = MemoryConfig(mode="read_only_dataset", path=tmp_path)
    store = build_store(cfg)
    writer = build_writer(cfg, store=store)
    assert isinstance(writer, StoreBackedMemoryWriter)
    assert writer._mode == "read_only_dataset"  # noqa: SLF001
