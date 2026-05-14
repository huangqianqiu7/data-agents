"""task corpus RAG 召回 helper（M4.5.1）。"""
from __future__ import annotations

import hashlib
from typing import Any

from data_agent_langchain.config import MemoryConfig
from data_agent_langchain.memory.types import MemoryHit
from data_agent_langchain.observability.events import dispatch_observability_event
from data_agent_langchain.runtime.context import get_current_corpus_handles


# 与 ``agents/memory_recall.py:_RECALL_ENABLED_MODES`` 对齐：仅在
# ``read_only_dataset`` / ``full`` 两种 mode 下才允许召回；``disabled``
# 强制关闭 corpus RAG（见 ``CorpusRagConfig`` docstring 的 mode × rag 交叉
# 决策表）。Bug 1 修复点：之前仅检查 ``cfg.enabled``，与设计意图脱节。
_RECALL_ENABLED_MODES = frozenset({"read_only_dataset", "full"})


def recall_corpus_snippets(
    memory_cfg: MemoryConfig,
    *,
    task_id: str,
    query: str,
    node: str,
    config: Any | None,
) -> list[MemoryHit]:
    """从当前 task corpus 中召回片段，并转换为 ``RunState.memory_hits`` 可承载的摘要。

    Bug 1 修复后签名扩展为 ``memory_cfg: MemoryConfig``（与
    ``recall_dataset_facts`` 对称）；内部按以下顺序短路：

      1. ``memory_cfg.mode`` 不在 ``_RECALL_ENABLED_MODES`` 中（如 ``"disabled"``）
         → returns ``[]``，不调 retriever。
      2. ``memory_cfg.rag.enabled=False`` → returns ``[]``，不调 retriever。
      3. ``contextvar`` 无 handles（runner fail-closed 时）→ returns ``[]``。
      4. ``retrieval_k <= 0`` → returns ``[]``，不调 retriever。
      5. retriever 异常 → 派发 ``memory_rag_skipped(reason="retrieve_failed")``
         并 returns ``[]``。

    该函数只读取 runner 放进 contextvar 的 ``TaskCorpusHandles``，不会把 query
    写入任何 store；观测事件只记录 ``sha1(query)`` 的短摘要。
    """
    if memory_cfg.mode not in _RECALL_ENABLED_MODES:
        return []

    cfg = memory_cfg.rag
    if not cfg.enabled:
        return []

    handles = get_current_corpus_handles()
    if handles is None:
        return []

    k = max(0, int(cfg.retrieval_k))
    if k <= 0:
        return []

    namespace = f"corpus_task:{task_id}"
    try:
        results = handles.retriever.retrieve(query, namespace=namespace, k=k)
    except Exception as exc:
        dispatch_observability_event(
            "memory_rag_skipped",
            {
                "reason": "retrieve_failed",
                "task_id": task_id,
                "namespace": namespace,
                "node": node,
                "error": f"{type(exc).__name__}: {exc}",
            },
            config=config,
        )
        return []

    hits = _results_to_hits(
        results,
        budget_chars=max(0, int(cfg.prompt_budget_chars)),
    )
    dispatch_observability_event(
        "memory_recall",
        {
            "node": node,
            "namespace": namespace,
            "k": k,
            "kind": "corpus_task",
            "query_digest": hashlib.sha1(query.encode("utf-8")).hexdigest()[:8],
            "hit_ids": [hit.record_id for hit in hits],
            "hit_chunk_ids": [hit.record_id for hit in hits],
            "hit_doc_ids": [_doc_id_from_result(result) for result in results[: len(hits)]],
            "scores": [hit.score for hit in hits],
            "model_id": getattr(getattr(handles, "embedder", None), "model_id", ""),
            "reason": "vector_cosine",
        },
        config=config,
    )
    return hits


def _results_to_hits(results: list[Any], *, budget_chars: int) -> list[MemoryHit]:
    """把 retriever 结果转换为 ``MemoryHit``，并严格遵守 summary 字符预算。"""
    if budget_chars <= 0:
        return []

    hits: list[MemoryHit] = []
    remaining = budget_chars
    for result in results:
        summary = _render_summary(result.record.payload)
        if len(summary) > remaining:
            summary = _truncate_summary(summary, remaining)
        if not summary:
            break
        hits.append(
            MemoryHit(
                record_id=result.record.id,
                namespace=result.record.namespace,
                score=float(result.score),
                summary=summary,
            )
        )
        remaining -= len(summary)
        if remaining <= 0:
            break
    return hits


def _render_summary(payload: dict[str, Any]) -> str:
    """按白名单字段渲染 corpus snippet 摘要。"""
    doc_kind = str(payload.get("doc_kind") or "text")
    source_path = str(payload.get("source_path") or "?")
    text = str(payload.get("text") or "")
    snippet = _truncate_summary(text, 240)
    return f"[{doc_kind}] {source_path}: {snippet}"


def _truncate_summary(text: str, limit: int) -> str:
    """截断字符串并保证返回长度不超过 ``limit``。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _doc_id_from_result(result: Any) -> str:
    """从 retriever 结果中提取 doc_id，缺失时返回空字符串。"""
    return str(result.record.payload.get("doc_id") or "")


__all__ = ["recall_corpus_snippets"]

