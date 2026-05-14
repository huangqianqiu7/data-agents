"""Lightweight summary types written into RunState."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryHit:
    record_id: str
    namespace: str
    score: float
    summary: str


__all__ = ["MemoryHit"]
