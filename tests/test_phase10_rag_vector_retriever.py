"""``VectorCorpusRetriever`` 行为测试（M4.3.3）。

按 ``01-design-v2.md §4.10`` 验证：

  - 命中场景：``retrieve`` 返回 ``list[RetrievalResult]``，``reason="vector_cosine"``，
    并把 ``doc_index`` 的 ``source_path`` / ``doc_kind`` 补回 ``payload``。
  - ``k=0`` 或 ``k<0`` 短路返回 ``[]``，**不**调用 embedder。
  - embedder 抛异常 → 返回 ``[]``（由 ``recall_corpus_snippets`` 负责 dispatch
    事件，retriever 本身静默）。
  - store 抛异常 → 同上。
  - 满足 v2 ``Retriever`` Protocol（``isinstance(..., Retriever) is True``）。
"""
from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock

import pytest

from data_agent_langchain.memory.base import (
    MemoryRecord,
    Retriever,
    RetrievalResult,
)
from data_agent_langchain.memory.rag.documents import CorpusDocument
from data_agent_langchain.memory.rag.embedders import DeterministicStubEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_doc(doc_id: str, source_path: str, doc_kind: str = "markdown") -> CorpusDocument:
    return CorpusDocument(
        doc_id=doc_id,
        source_path=source_path,
        doc_kind=doc_kind,  # type: ignore[arg-type]
        bytes_size=100,
        char_count=80,
        collection="task_corpus",
    )


def _make_hit(
    chunk_id: str,
    doc_id: str,
    text: str,
    namespace: str = "corpus_task:t1",
    score: float = 0.5,
) -> RetrievalResult:
    """构造 ``ChromaCorpusStore.query_by_vector`` 形态的返回值。"""
    record = MemoryRecord(
        id=chunk_id,
        namespace=namespace,
        kind="corpus",
        payload={
            "text": text,
            "doc_id": doc_id,
            "ord": 0,
            "char_offset": 0,
            "char_length": len(text),
        },
        metadata={},
    )
    return RetrievalResult(record=record, score=score, reason="vector_cosine")


# ---------------------------------------------------------------------------
# 命中场景
# ---------------------------------------------------------------------------


def test_retriever_returns_hits_with_source_path_and_doc_kind() -> None:
    """命中时应把 ``doc_index`` 的 ``source_path`` / ``doc_kind`` 补回 payload。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    doc = _make_doc("d1", "guides/intro.md", "markdown")
    store = MagicMock()
    store.query_by_vector.return_value = [_make_hit("d1#0000", "d1", "hello world")]
    embedder = DeterministicStubEmbedder(dim=8)

    retriever = VectorCorpusRetriever(
        store=store, embedder=embedder, doc_index={"d1": doc}, k=4
    )
    out = retriever.retrieve("what?", namespace="corpus_task:t1")

    assert len(out) == 1
    res = out[0]
    assert res.reason == "vector_cosine"
    assert res.record.kind == "corpus"
    assert res.record.payload["source_path"] == "guides/intro.md"
    assert res.record.payload["doc_kind"] == "markdown"
    # 原 payload 保留。
    assert res.record.payload["text"] == "hello world"
    assert res.record.payload["doc_id"] == "d1"


def test_retriever_uses_default_k_when_none() -> None:
    """``retrieve(query)`` 不传 k 时应用构造时的 ``k`` 默认值。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    store = MagicMock()
    store.query_by_vector.return_value = []
    embedder = DeterministicStubEmbedder(dim=8)

    retriever = VectorCorpusRetriever(
        store=store, embedder=embedder, doc_index={}, k=7
    )
    retriever.retrieve("q", namespace="ns")

    store.query_by_vector.assert_called_once()
    assert store.query_by_vector.call_args.kwargs["k"] == 7


def test_retriever_passes_explicit_k_through() -> None:
    """``retrieve(query, k=3)`` 显式 k 应覆盖构造时的默认值。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    store = MagicMock()
    store.query_by_vector.return_value = []
    retriever = VectorCorpusRetriever(
        store=store,
        embedder=DeterministicStubEmbedder(dim=8),
        doc_index={},
        k=4,
    )
    retriever.retrieve("q", namespace="ns", k=3)

    assert store.query_by_vector.call_args.kwargs["k"] == 3


def test_retriever_handles_unknown_doc_id_gracefully() -> None:
    """``doc_index`` 中找不到 ``doc_id`` 时应返回结果但不补 source_path / doc_kind。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    store = MagicMock()
    store.query_by_vector.return_value = [_make_hit("d2#0000", "d2", "x")]

    retriever = VectorCorpusRetriever(
        store=store,
        embedder=DeterministicStubEmbedder(dim=8),
        doc_index={"d1": _make_doc("d1", "a.md")},  # 不含 d2
        k=4,
    )
    out = retriever.retrieve("q", namespace="ns")

    assert len(out) == 1
    # 未补 source_path / doc_kind。
    assert "source_path" not in out[0].record.payload
    assert "doc_kind" not in out[0].record.payload


def test_retriever_preserves_record_score_and_namespace() -> None:
    """``score`` / ``namespace`` 应原样透传，不被 retriever 改动。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    hit = _make_hit("d1#0000", "d1", "x", namespace="corpus_task:abc", score=0.73)
    store = MagicMock()
    store.query_by_vector.return_value = [hit]

    retriever = VectorCorpusRetriever(
        store=store,
        embedder=DeterministicStubEmbedder(dim=8),
        doc_index={"d1": _make_doc("d1", "a.md")},
        k=4,
    )
    out = retriever.retrieve("q", namespace="corpus_task:abc")
    assert out[0].score == 0.73
    assert out[0].record.namespace == "corpus_task:abc"


# ---------------------------------------------------------------------------
# k=0 / k<0 短路
# ---------------------------------------------------------------------------


def test_retriever_k_zero_returns_empty_and_does_not_call_embedder() -> None:
    """``k=0`` 短路返回 ``[]``，**不**调用 embedder 与 store。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    store = MagicMock()
    embedder = MagicMock()
    retriever = VectorCorpusRetriever(
        store=store, embedder=embedder, doc_index={}, k=4
    )
    out = retriever.retrieve("q", namespace="ns", k=0)
    assert out == []
    embedder.embed_query.assert_not_called()
    store.query_by_vector.assert_not_called()


def test_retriever_negative_k_returns_empty() -> None:
    """``k < 0`` 同样短路。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    store = MagicMock()
    embedder = MagicMock()
    retriever = VectorCorpusRetriever(
        store=store, embedder=embedder, doc_index={}, k=4
    )
    assert retriever.retrieve("q", namespace="ns", k=-2) == []
    embedder.embed_query.assert_not_called()


def test_retriever_construct_with_k_zero_always_returns_empty() -> None:
    """构造 ``k=0`` 时即便不显式传 k，也短路。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    store = MagicMock()
    embedder = MagicMock()
    retriever = VectorCorpusRetriever(
        store=store, embedder=embedder, doc_index={}, k=0
    )
    assert retriever.retrieve("q", namespace="ns") == []


# ---------------------------------------------------------------------------
# 异常容错
# ---------------------------------------------------------------------------


def test_retriever_returns_empty_when_embedder_raises() -> None:
    """``embed_query`` 抛异常 → 返回 ``[]``，不抛出（fail-closed）。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    store = MagicMock()
    embedder = MagicMock()
    embedder.embed_query.side_effect = RuntimeError("boom")

    retriever = VectorCorpusRetriever(
        store=store, embedder=embedder, doc_index={}, k=4
    )
    out = retriever.retrieve("q", namespace="ns")
    assert out == []
    store.query_by_vector.assert_not_called()


def test_retriever_returns_empty_when_store_raises() -> None:
    """``store.query_by_vector`` 抛异常 → 返回 ``[]``。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    store = MagicMock()
    store.query_by_vector.side_effect = RuntimeError("boom")
    embedder = DeterministicStubEmbedder(dim=8)

    retriever = VectorCorpusRetriever(
        store=store, embedder=embedder, doc_index={}, k=4
    )
    out = retriever.retrieve("q", namespace="ns")
    assert out == []


# ---------------------------------------------------------------------------
# Protocol 兼容
# ---------------------------------------------------------------------------


def test_retriever_satisfies_v2_retriever_protocol() -> None:
    """``VectorCorpusRetriever`` 应满足 v2 ``Retriever`` Protocol。"""
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    retriever = VectorCorpusRetriever(
        store=MagicMock(),
        embedder=DeterministicStubEmbedder(dim=8),
        doc_index={},
        k=4,
    )
    assert isinstance(retriever, Retriever)
