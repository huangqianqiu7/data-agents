"""Memory package exports."""
from __future__ import annotations

from importlib import import_module

from data_agent_langchain.memory.base import (
    MemoryRecord,
    MemoryStore,
    MemoryWriter,
    RecordKind,
    Retriever,
    RetrievalResult,
)
from data_agent_langchain.memory.factory import (
    build_retriever,
    build_store,
    build_writer,
)
from data_agent_langchain.memory.records import (
    DatasetKnowledgeRecord,
    ToolPlaybookRecord,
)
from data_agent_langchain.memory.types import MemoryHit

_WORKING_EXPORTS = {
    "build_scratchpad_messages",
    "render_step_messages",
    "select_steps_for_context",
    "truncate_observation",
}


def __getattr__(name: str):
    if name in _WORKING_EXPORTS:
        value = getattr(import_module("data_agent_langchain.memory.working"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetKnowledgeRecord",
    "MemoryHit",
    "MemoryRecord",
    "MemoryStore",
    "MemoryWriter",
    "RecordKind",
    "Retriever",
    "RetrievalResult",
    "ToolPlaybookRecord",
    "build_retriever",
    "build_scratchpad_messages",
    "build_store",
    "build_writer",
    "render_step_messages",
    "select_steps_for_context",
    "truncate_observation",
]
