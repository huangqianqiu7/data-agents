"""Append-only JSONL MemoryStore implementation.

Each namespace maps to one file. `_safe_filename` replaces `:` and `/` with
`__` to avoid path traversal. Deletes are represented with tombstone rows.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from data_agent_langchain.memory.base import MemoryRecord, RecordKind


def _safe_filename(namespace: str) -> str:
    return namespace.replace(":", "__").replace("/", "__") + ".jsonl"


def _record_to_json(rec: MemoryRecord) -> str:
    data = asdict(rec)
    data["created_at"] = rec.created_at.isoformat()
    return json.dumps(data, ensure_ascii=False, default=str)


def _record_from_json(line: str) -> MemoryRecord | None:
    obj: dict[str, Any] = json.loads(line)
    if "_tombstone" in obj:
        return None
    created_at = datetime.fromisoformat(obj["created_at"])
    return MemoryRecord(
        id=obj["id"],
        namespace=obj["namespace"],
        kind=obj["kind"],
        payload=obj.get("payload", {}),
        metadata=obj.get("metadata", {}),
        created_at=created_at,
    )


class JsonlMemoryStore:
    """Append-only JSONL store split by namespace."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str) -> Path:
        return self._root / _safe_filename(namespace)

    def put(self, record: MemoryRecord) -> None:
        with self._path(record.namespace).open("a", encoding="utf-8") as fp:
            fp.write(_record_to_json(record) + "\n")

    def _iter_active(self, namespace: str) -> list[MemoryRecord]:
        path = self._path(namespace)
        if not path.exists():
            return []
        active_by_id: dict[str, MemoryRecord] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if "_tombstone" in obj:
                active_by_id.pop(obj["_tombstone"], None)
                continue
            rec = _record_from_json(raw)
            if rec is not None:
                active_by_id[rec.id] = rec
        return list(active_by_id.values())

    def get(self, namespace: str, record_id: str) -> MemoryRecord | None:
        latest: MemoryRecord | None = None
        for rec in self._iter_active(namespace):
            if rec.id == record_id:
                latest = rec
        return latest

    def list(self, namespace: str, *, limit: int = 100) -> list[MemoryRecord]:
        ordered = sorted(
            self._iter_active(namespace), key=lambda r: r.created_at, reverse=True
        )
        return ordered[:limit]

    def delete(self, namespace: str, record_id: str) -> None:
        with self._path(namespace).open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"_tombstone": record_id}) + "\n")


__all__ = ["JsonlMemoryStore"]
