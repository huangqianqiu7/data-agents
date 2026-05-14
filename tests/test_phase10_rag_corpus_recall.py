"""``recall_corpus_snippets`` 的 task corpus 召回测试（M4.5.1）。"""
from __future__ import annotations

import json
from hashlib import sha1
from types import SimpleNamespace
from typing import Any

import pytest

from data_agent_langchain.config import CorpusRagConfig, MemoryConfig
from data_agent_langchain.memory.base import MemoryRecord, RetrievalResult
from data_agent_langchain.runtime.context import (
    clear_current_corpus_handles,
    set_current_corpus_handles,
)


@pytest.fixture(autouse=True)
def _clear_corpus_context():
    """每个用例后清理 contextvar，避免 task corpus handles 泄漏到后续测试。"""
    clear_current_corpus_handles()
    yield
    clear_current_corpus_handles()


class SpyRetriever:
    """记录 retrieve 调用的轻量 retriever。"""

    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, str, int | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        namespace: str,
        k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, namespace, k))
        return list(self.results)


class RaisingRetriever:
    """模拟底层向量检索异常。"""

    def retrieve(self, query: str, *, namespace: str, k: int | None = None):
        raise RuntimeError("boom")


def _handles(retriever: Any) -> SimpleNamespace:
    """构造与 ``TaskCorpusHandles`` 形状兼容的测试句柄。"""
    return SimpleNamespace(
        retriever=retriever,
        embedder=SimpleNamespace(model_id="stub-deterministic-dim16"),
    )


def _result(
    *,
    chunk_id: str = "doc-a#0000",
    namespace: str = "corpus_task:task_1",
    text: str = "schema details",
    doc_id: str = "doc-a",
    source_path: str = "README.md",
    doc_kind: str = "markdown",
    score: float = 0.91,
) -> RetrievalResult:
    """构造 corpus chunk 的 ``RetrievalResult``。"""
    return RetrievalResult(
        record=MemoryRecord(
            id=chunk_id,
            namespace=namespace,
            kind="corpus",
            payload={
                "text": text,
                "doc_id": doc_id,
                "source_path": source_path,
                "doc_kind": doc_kind,
            },
            metadata={},
        ),
        score=score,
        reason="vector_cosine",
    )


def test_recall_returns_empty_when_rag_disabled() -> None:
    """``rag.enabled=False`` 时不调用 retriever，直接返回空列表。

    Bug 1 修复后签名为 ``MemoryConfig``；本用例 ``mode="read_only_dataset"``
    让 mode 守卫先放行，再由 ``rag.enabled=False`` 短路。
    """
    from data_agent_langchain.agents.corpus_recall import recall_corpus_snippets

    retriever = SpyRetriever([_result()])
    set_current_corpus_handles(_handles(retriever))

    hits = recall_corpus_snippets(
        MemoryConfig(mode="read_only_dataset", rag=CorpusRagConfig(enabled=False)),
        task_id="task_1",
        query="where is schema?",
        node="planner",
        config=None,
    )

    assert hits == []
    assert retriever.calls == []


def test_recall_returns_empty_without_context_handles() -> None:
    """RAG 开启但 contextvar 无 handles 时 fail-closed 返回空列表。"""
    from data_agent_langchain.agents.corpus_recall import recall_corpus_snippets

    hits = recall_corpus_snippets(
        MemoryConfig(mode="read_only_dataset", rag=CorpusRagConfig(enabled=True)),
        task_id="task_1",
        query="where is schema?",
        node="planner",
        config=None,
    )

    assert hits == []


def test_recall_returns_memory_hits_and_dispatches_event(monkeypatch) -> None:
    """命中时返回 ``MemoryHit``，并派发不含原始 query 的 ``memory_recall`` 事件。"""
    import data_agent_langchain.agents.corpus_recall as corpus_recall

    captured: list[tuple[str, dict[str, Any], Any | None]] = []
    monkeypatch.setattr(
        corpus_recall,
        "dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data, config)),
    )

    retriever = SpyRetriever([_result(text="schema details for columns")])
    set_current_corpus_handles(_handles(retriever))
    memory_cfg = MemoryConfig(
        mode="read_only_dataset",
        rag=CorpusRagConfig(enabled=True, retrieval_k=3),
    )

    hits = corpus_recall.recall_corpus_snippets(
        memory_cfg,
        task_id="task_1",
        query="where is schema?",
        node="planner",
        config={"callbacks": []},
    )

    assert retriever.calls == [("where is schema?", "corpus_task:task_1", 3)]
    assert len(hits) == 1
    assert hits[0].record_id == "doc-a#0000"
    assert hits[0].namespace == "corpus_task:task_1"
    assert hits[0].score == 0.91
    assert hits[0].summary == "[markdown] README.md: schema details for columns"

    assert captured and captured[0][0] == "memory_recall"
    event = captured[0][1]
    assert event["kind"] == "corpus_task"
    assert event["node"] == "planner"
    assert event["namespace"] == "corpus_task:task_1"
    assert event["query_digest"] == sha1("where is schema?".encode("utf-8")).hexdigest()[:8]
    assert event["hit_chunk_ids"] == ["doc-a#0000"]
    assert event["hit_doc_ids"] == ["doc-a"]
    assert event["model_id"] == "stub-deterministic-dim16"
    assert event["reason"] == "vector_cosine"
    assert "where is schema?" not in json.dumps(event, ensure_ascii=False)


def test_recall_dispatches_skipped_when_retriever_raises(monkeypatch) -> None:
    """retriever 异常时返回空列表并派发 ``retrieve_failed``。"""
    import data_agent_langchain.agents.corpus_recall as corpus_recall

    captured: list[tuple[str, dict[str, Any], Any | None]] = []
    monkeypatch.setattr(
        corpus_recall,
        "dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data, config)),
    )
    set_current_corpus_handles(_handles(RaisingRetriever()))

    hits = corpus_recall.recall_corpus_snippets(
        MemoryConfig(mode="read_only_dataset", rag=CorpusRagConfig(enabled=True)),
        task_id="task_1",
        query="where is schema?",
        node="planner",
        config=None,
    )

    assert hits == []
    assert captured and captured[0][0] == "memory_rag_skipped"
    assert captured[0][1]["reason"] == "retrieve_failed"
    assert captured[0][1]["task_id"] == "task_1"


def test_recall_respects_prompt_budget_chars() -> None:
    """返回的 summary 累计长度不超过 ``prompt_budget_chars``。"""
    from data_agent_langchain.agents.corpus_recall import recall_corpus_snippets

    long_text = "x" * 240
    retriever = SpyRetriever([
        _result(chunk_id="doc-a#0000", text=long_text),
        _result(chunk_id="doc-a#0001", text="second chunk"),
    ])
    set_current_corpus_handles(_handles(retriever))
    memory_cfg = MemoryConfig(
        mode="read_only_dataset",
        rag=CorpusRagConfig(enabled=True, prompt_budget_chars=80),
    )

    hits = recall_corpus_snippets(
        memory_cfg,
        task_id="task_1",
        query="where is schema?",
        node="planner",
        config=None,
    )

    assert hits
    assert sum(len(hit.summary) for hit in hits) <= 80
    assert hits[0].summary.endswith("...")

