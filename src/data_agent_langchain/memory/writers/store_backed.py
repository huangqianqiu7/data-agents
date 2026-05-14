"""Mode-aware MemoryWriter implementation."""
from __future__ import annotations

from dataclasses import asdict

from data_agent_langchain.memory.base import MemoryRecord, MemoryStore
from data_agent_langchain.memory.records import (
    DatasetKnowledgeRecord,
    ToolPlaybookRecord,
)


_VALID_MEMORY_MODES = frozenset({"disabled", "read_only_dataset", "full"})


class StoreBackedMemoryWriter:
    def __init__(self, store: MemoryStore, *, mode: str) -> None:
        if mode not in _VALID_MEMORY_MODES:
            raise ValueError(f"unsupported memory mode: {mode!r}")
        self._store = store
        self._mode = mode

    def write_dataset_knowledge(
        self, dataset: str, record: DatasetKnowledgeRecord
    ) -> None:
        if self._mode == "disabled":
            return
        self._store.put(
            MemoryRecord(
                id=f"dk:{dataset}:{record.file_path}",
                namespace=f"dataset:{dataset}",
                kind="dataset_knowledge",
                payload=asdict(record),
            )
        )

    def write_tool_playbook(
        self, dataset: str, tool_name: str, record: ToolPlaybookRecord
    ) -> None:
        if self._mode in {"disabled", "read_only_dataset"}:
            return
        self._store.put(
            MemoryRecord(
                id=f"tp:{dataset}:{tool_name}",
                namespace=f"dataset:{dataset}/tool:{tool_name}",
                kind="tool_playbook",
                payload=asdict(record),
            )
        )


__all__ = ["StoreBackedMemoryWriter"]
