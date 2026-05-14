"""``VectorCorpusRetriever``：corpus 语义的向量检索器（M4.3.3）。

按 ``01-design-v2.md §4.10`` 实现：

  - 适配 v2 ``Retriever`` Protocol：``retrieve(query, *, namespace, k=None)``。
  - 内部流程：``embedder.embed_query(query)`` → ``store.query_by_vector(vec, k)``
    → 用 ``doc_index`` 把 ``source_path`` / ``doc_kind`` 补回 payload。
  - **fail-closed**：embedder / store 抛任何异常都返回 ``[]``，不向上抛；
    事件 dispatch 由调用方 ``recall_corpus_snippets`` 负责，retriever 自身静默。
  - **k 短路**：``k <= 0`` 直接返回 ``[]``，**不**调用 embedder（性能与可观测）。
  - ``doc_index`` 保留 ``source_path`` / ``doc_kind`` 元数据，避免把这些字符串
    冗余写进 chroma metadata 占空间。
"""
from __future__ import annotations

from typing import Mapping

from data_agent_langchain.memory.base import MemoryRecord, RetrievalResult
from data_agent_langchain.memory.rag.base import CorpusStore
from data_agent_langchain.memory.rag.documents import CorpusDocument
from data_agent_langchain.memory.rag.embedders.base import Embedder


class VectorCorpusRetriever:
    """corpus 专属向量检索器；满足 v2 ``Retriever`` Protocol。"""

    __slots__ = ("_store", "_embedder", "_doc_index", "_k")

    def __init__(
        self,
        *,
        store: CorpusStore,
        embedder: Embedder,
        doc_index: Mapping[str, CorpusDocument],
        k: int,
    ) -> None:
        self._store = store
        self._embedder = embedder
        # 拷贝 doc_index 为普通 dict，避免外部 mapping 在 retriever 生命周期内被改。
        self._doc_index: dict[str, CorpusDocument] = dict(doc_index)
        self._k = int(k)

    def retrieve(
        self,
        query: str,
        *,
        namespace: str,
        k: int | None = None,
    ) -> list[RetrievalResult]:
        """召回 corpus chunk。

        ``namespace`` 参数由 v2 ``Retriever`` Protocol 要求；本实现忽略它，
        因为 store 在构造时已绑定 namespace（每 task 一个 store 实例）。
        """
        actual_k = self._k if k is None else int(k)
        if actual_k <= 0:
            return []

        # 1) 编码 query → 向量。embedder 抛错 → 静默返回 []。
        try:
            vector = self._embedder.embed_query(query)
        except Exception:
            return []

        # 2) 向量检索。store 抛错 → 静默返回 []。
        try:
            raw = self._store.query_by_vector(vector, k=actual_k)
        except Exception:
            return []

        # 3) 把 doc_index 中的 source_path / doc_kind 补回 payload。
        results: list[RetrievalResult] = []
        for hit in raw:
            doc_id = hit.record.payload.get("doc_id")
            doc = self._doc_index.get(doc_id) if doc_id is not None else None
            payload = dict(hit.record.payload)
            if doc is not None:
                payload["source_path"] = doc.source_path
                payload["doc_kind"] = doc.doc_kind
            results.append(
                RetrievalResult(
                    record=MemoryRecord(
                        id=hit.record.id,
                        namespace=hit.record.namespace,
                        kind="corpus",
                        payload=payload,
                        metadata=hit.record.metadata,
                    ),
                    score=hit.score,
                    reason="vector_cosine",
                )
            )
        return results


__all__ = ["VectorCorpusRetriever"]
