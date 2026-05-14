"""``MetricsCollector`` 聚合 memory_rag 事件测试（M4.6.1）。"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from data_agent_langchain.observability.metrics import MetricsCollector


def _payload(tmp_path: Path) -> dict:
    """读取测试输出的 ``metrics.json``。"""
    return json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))


def test_metrics_omits_memory_rag_without_rag_events(tmp_path: Path) -> None:
    """无 RAG 事件时不输出 ``memory_rag``，保持 baseline metrics 稳定。"""
    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    collector.on_custom_event("memory_recall", {"kind": "dataset_knowledge"}, run_id=uuid4())
    collector.on_chain_end({}, run_id=uuid4())

    assert "memory_rag" not in _payload(tmp_path)


def test_metrics_aggregates_index_built_and_corpus_recall(tmp_path: Path) -> None:
    """index built 与 corpus recall 应聚合进 ``memory_rag``。"""
    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    run_id = uuid4()
    collector.on_custom_event(
        "memory_rag_index_built",
        {
            "task_id": "task_1",
            "doc_count": 2,
            "chunk_count": 5,
            "model_id": "stub",
            "dimension": 16,
            "elapsed_ms": 7,
        },
        run_id=run_id,
    )
    collector.on_custom_event(
        "memory_recall",
        {
            "kind": "corpus_task",
            "node": "task_entry_node",
            "hit_chunk_ids": ["doc-a#0000"],
        },
        run_id=run_id,
    )
    collector.on_chain_end({}, run_id=run_id)

    memory_rag = _payload(tmp_path)["memory_rag"]
    assert memory_rag["task_index_built"] is True
    assert memory_rag["task_doc_count"] == 2
    assert memory_rag["task_chunk_count"] == 5
    assert memory_rag["shared_collections_loaded"] == 0
    assert memory_rag["recall_count"] == {"task_entry": 1}
    assert memory_rag["skipped"] == []


def test_metrics_aggregates_skipped_reasons_with_counts(tmp_path: Path) -> None:
    """``memory_rag_skipped`` 应按 reason 去重计数。"""
    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    run_id = uuid4()
    collector.on_custom_event(
        "memory_rag_skipped",
        {"reason": "no_documents", "task_id": "task_1"},
        run_id=run_id,
    )
    collector.on_custom_event(
        "memory_rag_skipped",
        {"reason": "no_documents", "task_id": "task_1"},
        run_id=run_id,
    )
    collector.on_custom_event(
        "memory_rag_skipped",
        {"reason": "index_timeout", "task_id": "task_1"},
        run_id=run_id,
    )
    collector.on_chain_end({}, run_id=run_id)

    memory_rag = _payload(tmp_path)["memory_rag"]
    assert memory_rag["task_index_built"] is False
    assert memory_rag["skipped"] == [
        {"reason": "index_timeout", "count": 1},
        {"reason": "no_documents", "count": 2},
    ]

