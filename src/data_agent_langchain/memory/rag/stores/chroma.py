"""``ChromaCorpusStore``：chromadb 后端的 :class:`CorpusStore` 实现（M4.3.2）。

按 ``01-design-v2.md §4.9`` / ``D2`` / ``D10`` 实现：

  - 仅提供 :meth:`ChromaCorpusStore.ephemeral` 类方法构造 ``EphemeralClient``
    实例（D2：``persistent_readonly`` / ``persistent_writable`` 拆到独立提案）。
  - 用 ``Settings(anonymized_telemetry=False)`` 关闭 chroma telemetry，避免
    评测态触发外部网络。
  - collection name = ``f"memrag_{sha1(namespace)[:16]}"``（D10）：避免
    ``a:b`` / ``a__b`` 这类 namespace 在 chroma collection name 字符集限制下
    发生 collision。
  - 显式禁用 chroma 内置 ``embedding_function``（``embedding_function=None`` +
    手动 ``upsert(embeddings=...)``），杜绝 chroma 默认调用 OpenAI 等外部
    embed function 的可能。
  - **不**实现 v2 ``MemoryStore`` 的 ``put`` / ``get`` / ``list`` / ``delete``
    KV 方法（B2 决策）。任何想用 v2 ``MemoryStore`` Protocol 的代码不能误把
    chroma store 当 KV 用。

**启动期 import 边界（D11）**：本模块**顶层**只 import 标准库与 v2 类型；
``chromadb`` / ``chromadb.config`` 仅在 :meth:`ephemeral` 与 :meth:`__init__`
内**方法级延迟 import**。
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

from data_agent_langchain.memory.base import MemoryRecord, RetrievalResult
from data_agent_langchain.memory.rag.documents import CorpusChunk
from data_agent_langchain.memory.rag.embedders.base import Embedder


class ChromaCorpusStore:
    """实现 :class:`~data_agent_langchain.memory.rag.base.CorpusStore`
    的 chromadb 后端（仅 ephemeral 模式）。
    """

    __slots__ = (
        "_client",
        "_namespace",
        "_embedder",
        "_distance",
        "_collection",
        "_collection_name",
    )

    @classmethod
    def ephemeral(
        cls,
        namespace: str,
        embedder: Embedder,
        *,
        distance: str = "cosine",
    ) -> "ChromaCorpusStore":
        """创建 ephemeral（内存）store。子进程退出即释放。"""
        # 方法级延迟 import：``rag.enabled=false`` / chromadb 未装路径不触发。
        import chromadb
        from chromadb.config import Settings

        client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
        return cls(
            client,
            namespace=namespace,
            embedder=embedder,
            distance=distance,
        )

    def __init__(
        self,
        client: Any,
        *,
        namespace: str,
        embedder: Embedder,
        distance: str,
    ) -> None:
        self._client = client
        self._namespace = namespace
        self._embedder = embedder
        self._distance = distance

        # collection name = ``f"memrag_{sha1(namespace)[:16]}"``（D10）。
        safe = hashlib.sha1(namespace.encode("utf-8")).hexdigest()[:16]
        collection_name = f"memrag_{safe}"
        # Bug 4：把 collection name 缓存到 slot，``close()`` 调用
        # ``delete_collection`` 时不再依赖 ``self._collection.name``
        # （collection 对象在 delete 后再访问属性可能不可靠）。
        self._collection_name = collection_name
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance, "ns": namespace},
            # 显式禁用 chroma 内置 embed function；我们手动传 embeddings。
            embedding_function=None,
        )

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    def upsert_chunks(self, chunks: Sequence[CorpusChunk]) -> None:
        """写入或更新一批 chunk。

        - 空输入直接返回，不调用 chroma 也不调用 embedder。
        - 非空：用本 store 的 embedder 编码 chunk 文本，再走
          ``collection.upsert(ids=, embeddings=, documents=, metadatas=)``。
        - metadata 仅保存 ``doc_id`` / ``ord`` / ``char_offset`` /
          ``char_length``；**不**写 ``source_path`` / ``doc_kind`` 等字段
          （后者由 retriever 用 ``doc_index`` 反查）。
        """
        if not chunks:
            return

        chunks_list = list(chunks)
        texts = [c.text for c in chunks_list]
        vectors = self._embedder.embed_documents(texts)
        ids = [c.chunk_id for c in chunks_list]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "ord": c.ord,
                "char_offset": c.char_offset,
                "char_length": c.char_length,
            }
            for c in chunks_list
        ]
        self._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    def query_by_vector(
        self, vector: Sequence[float], *, k: int
    ) -> list[RetrievalResult]:
        """按向量召回 top-k chunk。

        - ``k <= 0`` 短路返回 ``[]``，不调用 chroma。
        - 调用 ``collection.query(query_embeddings=[vec], n_results=k,
          include=["documents", "metadatas", "distances"])``。
        - 把每条命中转为 :class:`RetrievalResult`，``record.kind == "corpus"``，
          ``reason == "vector_cosine"``。
        - cosine score 公式：``score = 1 - dist/2 ∈ [-1, 1]``，越大越相关。
        """
        if k <= 0:
            return []

        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=int(k),
            include=["documents", "metadatas", "distances"],
        )

        # chroma 返回 batch list（外层是 query batch），我们只查一个 vector，
        # 故取 ``result[<key>][0]``。
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        out: list[RetrievalResult] = []
        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            record = MemoryRecord(
                id=chunk_id,
                namespace=self._namespace,
                kind="corpus",
                payload={
                    "text": text,
                    "doc_id": meta["doc_id"],
                    "ord": meta["ord"],
                    "char_offset": meta["char_offset"],
                    "char_length": meta["char_length"],
                },
                metadata={},
            )
            # cosine distance ∈ [0, 2] → score ∈ [-1, 1]，越大越相关；
            # 其他 distance 直接取负，保持 "score 越大越相关" 的方向语义。
            score = (
                1.0 - float(dist) / 2.0
                if self._distance == "cosine"
                else -float(dist)
            )
            out.append(
                RetrievalResult(record=record, score=score, reason="vector_cosine")
            )
        return out

    def close(self) -> None:
        """释放 collection 并丢弃 chroma client 引用。

        Bug 4 修复（chromadb EphemeralClient 进程内单例隐患）：

          - chromadb ``EphemeralClient`` 在同一 Python 进程内是单例缓存设计，
            ``self._client = None`` 仅丢引用并**不会**真正释放底层 collection。
            collection 仍在进程内累积，跨调用（同 namespace ⇒ 同 collection
            name = ``memrag_<sha1(namespace)[:16]>``）会继承之前写入的 chunk，
            导致：(1) 测试隔离失败；(2) 单进程多 task 内存累积；(3) shared /
            persistent backend 落地后隐患放大。
          - 修复：在丢引用前显式调 ``client.delete_collection(name=...)``，让
            collection 在 chroma 内部真正释放。

        守卫顺序：

          1. ``self._client is None`` → ``close()`` 已被调过，幂等 no-op。
          2. ``client.delete_collection`` 抛任何异常（collection 已不存在 /
             chroma 内部状态异常等）→ **静默吞掉**，与 store 其他方法的
             fail-closed 风格一致。

        ``EphemeralClient`` 本身没有显式 ``close`` API，丢引用让 GC 回收。
        """
        if self._client is None:
            return
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            # delete_collection 抛错不影响 close 语义。
            pass
        self._client = None


__all__ = ["ChromaCorpusStore"]
