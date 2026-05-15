"""Corpus RAG 工厂（M4.4.3）。

按 ``01-design-v2.md §4.11`` 实现：

  - :func:`build_embedder` —— 根据 ``CorpusRagConfig`` 决定返回
    ``DeterministicStubEmbedder`` / ``HarrierEmbedder`` / ``None``。
  - :func:`build_task_corpus` —— 子进程入口构造 per-task corpus 索引（
    ``loader.scan`` → ``redactor.filter_text`` → ``chunker.chunk`` →
    ``ChromaCorpusStore.ephemeral`` → ``store.upsert_chunks`` →
    ``VectorCorpusRetriever``），返回 :class:`TaskCorpusHandles` 或 ``None``。

事件：

  - ``memory_rag_index_built``：成功构建时 dispatch，payload 含 ``task_id`` /
    ``doc_count`` / ``chunk_count`` / ``model_id`` / ``dimension`` /
    ``elapsed_ms``。
  - ``memory_rag_skipped``：失败 / 跳过时 dispatch，``reason`` ∈
    ``{"no_documents", "index_timeout", "embedder_load_failed",
    "shared_corpus_not_implemented", ...}``。

**启动期 import 边界（D11）**：本模块顶层只 import 标准库与 rag 子包的纯类型
模块；``HarrierEmbedder`` / ``ChromaCorpusStore`` 都是方法级延迟 import 它们
各自的重依赖，本 factory 可以直接 ``from ... import HarrierEmbedder`` 因为
class 定义不触发 torch / chromadb 加载。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_langchain.config import CorpusRagConfig
from data_agent_langchain.memory.rag.base import CorpusStore
from data_agent_langchain.memory.rag.chunker import MarkdownAwareChunker
from data_agent_langchain.memory.rag.documents import CorpusChunk, CorpusDocument
from data_agent_langchain.memory.rag.embedders import (
    DeterministicStubEmbedder,
    Embedder,
)
from data_agent_langchain.memory.rag.loader import Loader
from data_agent_langchain.memory.rag.redactor import Redactor
from data_agent_langchain.memory.rag.retrievers.vector import VectorCorpusRetriever
from data_agent_langchain.observability.events import dispatch_observability_event


# ``stub`` backend 用的固定 dim：不引入 ``embedder_stub_dim`` 配置字段污染
# 生产配置；16 足够给行为测试使用，且与 vector 计算复杂度无关（stub 不参与
# 真实评测）。
_STUB_EMBEDDER_DIM: int = 16


@dataclass(frozen=True, slots=True)
class TaskCorpusHandles:
    """子进程入口构建出的 task corpus 句柄；通过 ``runtime/context`` contextvar
    暴露给 graph node。

    Attributes:
        embedder: 本 task 共享的 embedder 实例。
        store: chroma ephemeral store 实例（绑定 ``corpus_task:<task_id>``）。
        retriever: 已注入 ``doc_index`` 的向量检索器，供 ``recall_corpus_snippets``
            直接调用 ``retriever.retrieve(query, namespace=...)``。
    """

    embedder: Embedder
    store: CorpusStore
    retriever: VectorCorpusRetriever


def build_embedder(
    cfg: CorpusRagConfig,
    *,
    config: Any | None = None,
) -> Embedder | None:
    """根据 ``cfg`` 构造 embedder。

    - ``cfg.enabled=False`` → 返回 ``None``。
    - ``cfg.embedder_backend == "stub"`` → 返回 ``DeterministicStubEmbedder``。
    - ``cfg.embedder_backend == "sentence_transformer"`` → 方法级延迟 import
      ``HarrierEmbedder`` 并构造；任何异常（``ImportError`` / ``OSError`` /
      ``Exception``）都被捕获，返回 ``None`` 并 dispatch
      ``memory_rag_skipped(reason="embedder_load_failed")``。
    """
    if not cfg.enabled:
        return None

    if cfg.embedder_backend == "stub":
        return DeterministicStubEmbedder(dim=_STUB_EMBEDDER_DIM)

    if cfg.embedder_backend == "sentence_transformer":
        try:
            # 方法级延迟 import：避免 ``rag.enabled=false`` 路径触发 torch 加载。
            from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
                HarrierEmbedder,
            )

            return HarrierEmbedder(
                model_id=cfg.embedder_model_id,
                device=cfg.embedder_device,
                dtype=cfg.embedder_dtype,
                query_prompt_name=cfg.embedder_query_prompt_name,
                max_seq_len=cfg.embedder_max_seq_len,
                batch_size=cfg.embedder_batch_size,
                cache_dir=cfg.embedder_cache_dir,
            )
        except Exception as exc:
            dispatch_observability_event(
                "memory_rag_skipped",
                {
                    "reason": "embedder_load_failed",
                    "backend": "sentence_transformer",
                    "model_id": cfg.embedder_model_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                config=config,
            )
            return None

    # 未知 backend：fail-closed。
    dispatch_observability_event(
        "memory_rag_skipped",
        {
            "reason": "embedder_load_failed",
            "backend": cfg.embedder_backend,
            "error": "unknown embedder_backend",
        },
        config=config,
    )
    return None


def build_task_corpus(
    cfg: CorpusRagConfig,
    *,
    task_id: str,
    task_input_dir: Path,
    embedder: Embedder,
    config: Any | None = None,
) -> TaskCorpusHandles | None:
    """子进程入口构造 per-task corpus 索引。

    流程（``§4.11``）：

      1. 检查 ``cfg.enabled and cfg.task_corpus``；否则返回 ``None``。
      2. ``shared_corpus=True`` → dispatch ``shared_corpus_not_implemented``
         事件（task_corpus 路径仍继续）。
      3. ``Loader.scan(task_input_dir)`` → 文档列表；空列表 → 返回 ``None``
         + dispatch ``no_documents``。
      4. 对每个文档读正文 → ``Redactor.filter_text`` 过滤（命中 patterns 的整段
         丢弃）→ ``MarkdownAwareChunker.chunk`` 切片。
      5. ``ChromaCorpusStore.ephemeral(namespace=f"corpus_task:{task_id}",
         embedder=embedder)``。
      6. ``store.upsert_chunks(all_chunks)``（含 embedder 调用）。
      7. 全程包在 ``cfg.task_corpus_index_timeout_s`` 超时检查内；超时 → 返回
         ``None`` + dispatch ``index_timeout``。
      8. 成功 → dispatch ``memory_rag_index_built``，返回
         :class:`TaskCorpusHandles`。
    """
    if not (cfg.enabled and cfg.task_corpus):
        return None

    # shared_corpus 未实现：dispatch 警告但 task_corpus 仍继续。
    if cfg.shared_corpus:
        dispatch_observability_event(
            "memory_rag_skipped",
            {
                "reason": "shared_corpus_not_implemented",
                "shared_collections": list(cfg.shared_collections),
            },
            config=config,
        )

    start_t = perf_counter()
    timeout_s = float(cfg.task_corpus_index_timeout_s)

    def _check_timeout(stage: str) -> bool:
        """返回 ``True`` 表示超时。"""
        if perf_counter() - start_t > timeout_s:
            dispatch_observability_event(
                "memory_rag_skipped",
                {
                    "reason": "index_timeout",
                    "task_id": task_id,
                    "stage": stage,
                    "timeout_s": timeout_s,
                },
                config=config,
            )
            return True
        return False

    # 步骤 3：扫描目录。
    redactor = Redactor(
        redact_patterns=cfg.redact_patterns,
        redact_filenames=cfg.redact_filenames,
    )
    loader = Loader(redactor=redactor, max_docs_per_task=cfg.max_docs_per_task)
    docs = loader.scan(task_input_dir)
    if not docs:
        dispatch_observability_event(
            "memory_rag_skipped",
            {
                "reason": "no_documents",
                "task_id": task_id,
                "context_dir": str(task_input_dir),
            },
            config=config,
        )
        return None

    if _check_timeout("after_scan"):
        return None

    # 步骤 4：读取 + redact + chunk。
    chunker = MarkdownAwareChunker(
        chunk_size_chars=cfg.chunk_size_chars,
        chunk_overlap_chars=cfg.chunk_overlap_chars,
        max_chunks_per_doc=cfg.max_chunks_per_doc,
    )
    all_chunks: list[CorpusChunk] = []
    kept_docs: list[CorpusDocument] = []
    for doc in docs:
        if _check_timeout("during_chunk"):
            return None
        try:
            raw_text = loader.read_document_text(doc, task_input_dir)
        except OSError:
            continue
        filtered = redactor.filter_text(raw_text)
        if not filtered:
            # 命中 redact_patterns 的整段被丢弃。
            continue
        chunks = chunker.chunk(doc, filtered)
        if not chunks:
            continue
        all_chunks.extend(chunks)
        kept_docs.append(doc)

    if not all_chunks:
        dispatch_observability_event(
            "memory_rag_skipped",
            {
                "reason": "no_documents",
                "task_id": task_id,
                "context_dir": str(task_input_dir),
                "note": "所有文档被 redact 或切片为空",
            },
            config=config,
        )
        return None

    if _check_timeout("before_upsert"):
        return None

    # 步骤 5 + 6：构造 store 并写入。Bug 2 修复：任何异常（chromadb 缺失 /
    # ChromaCorpusStore.ephemeral 抛错 / store.upsert_chunks 失败）都必须
    # fail-closed 并 dispatch ``chroma_store_load_failed`` 落 trace，不能
    # 让异常一路传到 runner 被吞掉。
    namespace = f"corpus_task:{task_id}"
    try:
        from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

        store = ChromaCorpusStore.ephemeral(
            namespace=namespace,
            embedder=embedder,
            distance=cfg.vector_distance,
        )
        store.upsert_chunks(all_chunks)
    except Exception as exc:
        dispatch_observability_event(
            "memory_rag_skipped",
            {
                "reason": "chroma_store_load_failed",
                "task_id": task_id,
                "namespace": namespace,
                "error": f"{type(exc).__name__}: {exc}",
            },
            config=config,
        )
        return None

    if _check_timeout("after_upsert"):
        # 超时但已经写入；仍然 fail-closed 返回 None。
        store.close()
        return None

    # 步骤 8：构造 retriever，dispatch 成功事件。
    doc_index = {d.doc_id: d for d in kept_docs}
    retriever = VectorCorpusRetriever(
        store=store,
        embedder=embedder,
        doc_index=doc_index,
        k=cfg.retrieval_k,
    )
    elapsed_ms = int((perf_counter() - start_t) * 1000)
    dispatch_observability_event(
        "memory_rag_index_built",
        {
            "task_id": task_id,
            "doc_count": len(kept_docs),
            "chunk_count": len(all_chunks),
            "model_id": embedder.model_id,
            "dimension": embedder.dimension,
            "elapsed_ms": elapsed_ms,
        },
        config=config,
    )
    return TaskCorpusHandles(embedder=embedder, store=store, retriever=retriever)


__all__ = ["TaskCorpusHandles", "build_embedder", "build_task_corpus"]
