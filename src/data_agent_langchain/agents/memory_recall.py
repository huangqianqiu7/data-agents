"""Shared memory recall helpers for graph nodes."""
from __future__ import annotations

from typing import Any

from data_agent_langchain.config import MemoryConfig
from data_agent_langchain.memory.factory import build_retriever, build_store
from data_agent_langchain.memory.types import MemoryHit
from data_agent_langchain.observability.events import dispatch_observability_event


_RECALL_ENABLED_MODES = frozenset({"read_only_dataset", "full"})


def recall_dataset_facts(
    memory_cfg: MemoryConfig,
    dataset: str,
    node: str,
    config: Any | None,
) -> list[MemoryHit]:
    """Recall whitelisted dataset facts for a dataset namespace."""
    if memory_cfg.mode not in _RECALL_ENABLED_MODES:
        return []

    namespace = f"dataset:{dataset}"
    k = max(0, int(memory_cfg.retrieval_max_results))
    store = build_store(memory_cfg)
    retriever = build_retriever(memory_cfg, store=store)
    results = retriever.retrieve("", namespace=namespace, k=k)
    hits = [
        MemoryHit(
            record_id=result.record.id,
            namespace=result.record.namespace,
            score=result.score,
            summary=_dataset_fact_summary(result.record.payload),
        )
        for result in results
    ]
    dispatch_observability_event(
        "memory_recall",
        {
            "node": node,
            "namespace": namespace,
            "k": k,
            "hit_ids": [hit.record_id for hit in hits],
            "reason": "exact_namespace",
        },
        config=config,
    )
    return hits


def _dataset_fact_summary(payload: dict[str, Any]) -> str:
    file_path = payload.get("file_path") or "?"
    file_kind = payload.get("file_kind") or "?"
    columns = _columns_text(payload)
    rows = payload.get("row_count_estimate")
    rows_text = str(rows) if rows is not None else "?"
    return f"File: {file_path}  Kind: {file_kind}  Columns: {columns}  Rows~: {rows_text}"


def _columns_text(payload: dict[str, Any]) -> str:
    sample_columns = payload.get("sample_columns")
    if isinstance(sample_columns, list) and sample_columns:
        return ", ".join(str(column) for column in sample_columns)

    schema = payload.get("schema")
    if isinstance(schema, dict) and schema:
        return ", ".join(str(column) for column in schema.keys())

    return "?"


__all__ = ["recall_dataset_facts"]
