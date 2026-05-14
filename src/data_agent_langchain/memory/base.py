"""Memory layer protocols and base data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable


RecordKind = Literal["working", "dataset_knowledge", "tool_playbook", "corpus"]


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """Generic key/value envelope for cross-task memory."""

    id: str
    namespace: str
    kind: RecordKind
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    record: MemoryRecord
    score: float
    reason: str


@runtime_checkable
class MemoryStore(Protocol):
    def put(self, record: MemoryRecord) -> None: ...
    def get(self, namespace: str, record_id: str) -> MemoryRecord | None: ...
    def list(self, namespace: str, *, limit: int = 100) -> list[MemoryRecord]: ...
    def delete(self, namespace: str, record_id: str) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self, query: str, *, namespace: str, k: int = 5
    ) -> list[RetrievalResult]: ...


@runtime_checkable
class MemoryWriter(Protocol):
    def write_dataset_knowledge(self, dataset: str, record: Any) -> None: ...
    def write_tool_playbook(
        self, dataset: str, tool_name: str, record: Any
    ) -> None: ...


__all__ = [
    "MemoryRecord",
    "MemoryStore",
    "MemoryWriter",
    "RecordKind",
    "Retriever",
    "RetrievalResult",
]
