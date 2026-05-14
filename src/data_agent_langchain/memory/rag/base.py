"""Corpus RAG store 协议（M4.3.1，B2 决策）。

按 ``01-design-v2.md §4.9`` 定义独立的 :class:`CorpusStore` Protocol：

  - 与 v2 ``data_agent_langchain.memory.base.MemoryStore`` **解耦**，**不**复用
    其 ``put`` / ``get`` / ``list`` / ``delete`` KV 接口（B2 决策守护）。
  - 仅暴露 corpus 专用的 ``upsert_chunks`` / ``query_by_vector`` / ``close``
    与 ``namespace`` / ``dimension`` 两个 property。
  - 召回结果借用 v2 ``RetrievalResult`` 类型，使上层 ``VectorCorpusRetriever``
    与 v2 ``Retriever`` Protocol 兼容；``RetrievalResult.record`` 的
    ``MemoryRecord.kind == "corpus"``（v2 ``RecordKind`` 已预留）。

为什么不复用 v2 ``MemoryStore``？

  - 语义不同：``MemoryStore`` 是 KV envelope；corpus store 是向量索引。强行
    把 ``put`` 映射到 ``upsert_chunks`` 会把 chunk 与 KV record 的字段语义混
    在一起，导致写入路径上的 redact 守护更难维护。
  - 实现成本不对称：``ChromaCorpusStore`` 没有 ``get(record_id)`` 的 O(1) 路径
    —— chromadb 是按向量查的，不是按 ID 查；强行实现会引入误用。
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from data_agent_langchain.memory.base import RetrievalResult
from data_agent_langchain.memory.rag.documents import CorpusChunk


@runtime_checkable
class CorpusStore(Protocol):
    """corpus 专属 store 接口，与 v2 MemoryStore 解耦（B2）。"""

    @property
    def namespace(self) -> str:
        """绑定的 namespace 字符串（如 ``"corpus_task:<task_id>"``）。"""
        ...

    @property
    def dimension(self) -> int:
        """向量维度；与 ``embedder.dimension`` 一致。"""
        ...

    def upsert_chunks(self, chunks: Sequence[CorpusChunk]) -> None:
        """写入或更新一批 chunk；空输入应该 no-op。"""
        ...

    def query_by_vector(
        self, vector: Sequence[float], *, k: int
    ) -> list[RetrievalResult]:
        """按向量召回 top-k chunk；``k <= 0`` 时返回 ``[]``。"""
        ...

    def close(self) -> None:
        """释放底层资源（chroma client 等）。"""
        ...


__all__ = ["CorpusStore"]
