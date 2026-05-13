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
    "MemoryRecord",
    "MemoryStore",
    "MemoryWriter",
    "RecordKind",
    "Retriever",
    "RetrievalResult",
    "build_scratchpad_messages",
    "render_step_messages",
    "select_steps_for_context",
    "truncate_observation",
]
