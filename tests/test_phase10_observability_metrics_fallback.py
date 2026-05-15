"""Bug 5 回归：``MetricsCollector`` 必须能通过 events fallback 路径接收
``memory_rag_*`` 事件，并把它们汇总到 ``metrics.json`` 的 ``memory_rag`` 段。

场景：
  - ``runner._build_and_set_corpus_handles`` 在 ``compiled.invoke`` 之前调用
    ``factory.build_task_corpus`` → 后者通过 ``dispatch_observability_event``
    发出 ``memory_rag_index_built`` / ``memory_rag_skipped`` 事件。
  - 这阶段没有 LangGraph callback manager → ``dispatch_custom_event``
    抛 ``RuntimeError``，由 events.py fallback 路径把事件路由给已注册的
    ``MetricsCollector.on_observability_event``。
  - 最终 ``metrics.json`` 的 ``memory_rag`` 段必须包含这些事件汇总。
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from data_agent_langchain.observability.events import (
    dispatch_observability_event,
    register_fallback_handler,
    unregister_fallback_handler,
)
from data_agent_langchain.observability.metrics import MetricsCollector


@pytest.fixture(autouse=True)
def _clear_fallback_handlers():
    """每个 test 自动清理 fallback list（防御性）。"""
    from data_agent_langchain.observability import events as events_mod

    events_mod._FALLBACK_HANDLERS.clear()
    yield
    events_mod._FALLBACK_HANDLERS.clear()


def test_metrics_collector_exposes_on_observability_event_public_method() -> None:
    """``MetricsCollector`` 应提供 ``on_observability_event(name, data)`` public method。"""
    metrics = MetricsCollector(task_id="task_x", output_dir=Path("/tmp"))
    assert callable(getattr(metrics, "on_observability_event", None)), (
        "MetricsCollector must expose on_observability_event(name, data) for events fallback path"
    )


def test_metrics_collector_aggregates_memory_rag_index_built_via_fallback(
    tmp_path: Path,
) -> None:
    """通过 fallback 路径 dispatch ``memory_rag_index_built``，metrics.json 应包含 ``memory_rag.task_index_built=True``。"""
    out = tmp_path / "task_xy"
    metrics = MetricsCollector(task_id="task_xy", output_dir=out)

    register_fallback_handler(metrics.on_observability_event)
    try:
        dispatch_observability_event(
            "memory_rag_index_built",
            {
                "task_id": "task_xy",
                "doc_count": 2,
                "chunk_count": 62,
                "model_id": "microsoft/harrier-oss-v1-270m",
                "dimension": 768,
                "elapsed_ms": 60000,
            },
        )
        # 触发最外层 on_chain_end 写出 metrics.json
        metrics.on_chain_end({}, parent_run_id=None, run_id=uuid4())
    finally:
        unregister_fallback_handler(metrics.on_observability_event)

    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "memory_rag" in payload, f"expected memory_rag段, got: {payload.keys()}"
    rag = payload["memory_rag"]
    assert rag["task_index_built"] is True
    assert rag["task_doc_count"] == 2
    assert rag["task_chunk_count"] == 62


def test_metrics_collector_aggregates_memory_rag_skipped_via_fallback(
    tmp_path: Path,
) -> None:
    """通过 fallback 路径 dispatch 多个 ``memory_rag_skipped``，metrics.json ``memory_rag.skipped`` 应正确聚合。"""
    out = tmp_path / "task_z"
    metrics = MetricsCollector(task_id="task_z", output_dir=out)

    register_fallback_handler(metrics.on_observability_event)
    try:
        dispatch_observability_event("memory_rag_skipped", {"reason": "index_timeout"})
        dispatch_observability_event("memory_rag_skipped", {"reason": "index_timeout"})
        dispatch_observability_event("memory_rag_skipped", {"reason": "no_documents"})
        metrics.on_chain_end({}, parent_run_id=None, run_id=uuid4())
    finally:
        unregister_fallback_handler(metrics.on_observability_event)

    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "memory_rag" in payload
    skipped_entries = {entry["reason"]: entry["count"] for entry in payload["memory_rag"]["skipped"]}
    assert skipped_entries == {"index_timeout": 2, "no_documents": 1}


def test_metrics_collector_on_observability_event_matches_on_custom_event_logic(
    tmp_path: Path,
) -> None:
    """``on_observability_event`` 与 ``on_custom_event`` 在处理同一事件时聚合结果应等价。"""
    metrics_a = MetricsCollector(task_id="a", output_dir=tmp_path / "a")
    metrics_b = MetricsCollector(task_id="b", output_dir=tmp_path / "b")

    # A 走 on_custom_event（LangChain 原路径）
    metrics_a.on_custom_event("memory_rag_index_built", {"doc_count": 3, "chunk_count": 10}, run_id=uuid4())
    metrics_a.on_custom_event("memory_rag_skipped", {"reason": "index_timeout"}, run_id=uuid4())
    metrics_a.on_chain_end({}, parent_run_id=None, run_id=uuid4())

    # B 走 on_observability_event（fallback 路径）
    metrics_b.on_observability_event("memory_rag_index_built", {"doc_count": 3, "chunk_count": 10})
    metrics_b.on_observability_event("memory_rag_skipped", {"reason": "index_timeout"})
    metrics_b.on_chain_end({}, parent_run_id=None, run_id=uuid4())

    payload_a = json.loads((tmp_path / "a" / "metrics.json").read_text(encoding="utf-8"))
    payload_b = json.loads((tmp_path / "b" / "metrics.json").read_text(encoding="utf-8"))

    assert payload_a["memory_rag"] == payload_b["memory_rag"]


def test_metrics_collector_does_not_double_count_when_both_paths_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """如果 LangGraph context 内 ``dispatch_custom_event`` 成功了，
    fallback 路径不应额外触发，metrics 不会重复计数（events.py 的责任，
    这里做集成验证）。"""
    from data_agent_langchain.observability import events as events_mod
    from uuid import uuid4 as _uuid4

    # 模拟 LangGraph runtime 内的 dispatch_custom_event 成功
    on_custom_calls: list[tuple[str, dict]] = []

    def fake_dispatch_custom_event(name, data, config=None):
        # 模拟 LangChain callback manager 把事件路由到 callback handler
        # 这里我们只记录，由 test 末尾驱动 callback 调用
        on_custom_calls.append((name, dict(data)))

    monkeypatch.setattr(events_mod, "dispatch_custom_event", fake_dispatch_custom_event)

    out = tmp_path / "task_double"
    metrics = MetricsCollector(task_id="task_double", output_dir=out)

    register_fallback_handler(metrics.on_observability_event)
    try:
        dispatch_observability_event("memory_rag_skipped", {"reason": "index_timeout"})
        # 模拟 LangChain 把 dispatch_custom_event 转发给 metrics.on_custom_event
        for name, data in on_custom_calls:
            metrics.on_custom_event(name, data, run_id=_uuid4())
        metrics.on_chain_end({}, parent_run_id=None, run_id=_uuid4())
    finally:
        unregister_fallback_handler(metrics.on_observability_event)

    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    # fallback 路径在 dispatch_custom_event 成功时不应被调，所以只算一次。
    skipped_entries = {entry["reason"]: entry["count"] for entry in payload["memory_rag"]["skipped"]}
    assert skipped_entries == {"index_timeout": 1}, (
        f"expected exactly 1 skip count (only via on_custom_event), got: {skipped_entries}"
    )
