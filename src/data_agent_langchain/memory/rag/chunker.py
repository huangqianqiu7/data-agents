"""Corpus RAG 字符窗口切片器（M4.1.4）。

按 ``01-design-v2.md §4.8`` 实现 :class:`CharWindowChunker`：

  - 按字符窗口切片：窗口大小 ``chunk_size_chars``，相邻窗口重叠
    ``chunk_overlap_chars`` 字符。
  - 段落优先：如果当前候选切点 ``±_PARAGRAPH_SEARCH_RADIUS`` 字符内出现
    ``"\\n\\n"``，则把切点移到该段落分隔符之后，让 chunk 在段落边界处结束，
    避免把一段 markdown 段落硬切成两半破坏语义。
  - 每个 doc 最多产出 ``max_chunks_per_doc`` 个 chunk；超出时截断 + dispatch
    一个 ``memory_rag_skipped(reason="max_chunks_truncated")`` 事件。
  - 输出 ``CorpusChunk.chunk_id == f"{doc_id}#{ord:04d}"``，由 ``ord``（从 0
    开始的序号）与 ``doc_id`` 一起重建。

设计偏离：构造器从 ``__init__(cfg)`` 改为接受三个独立参数，与 ``Loader`` /
``Redactor`` 解耦风格一致；``factory.build_task_corpus`` 在 M4.4.3 简单包装。
"""
from __future__ import annotations

from data_agent_langchain.memory.rag.documents import CorpusChunk, CorpusDocument
from data_agent_langchain.observability.events import dispatch_observability_event


# 段落对齐的搜索半径：候选切点 ±100 字符内寻找 ``"\n\n"``；这是经验值，
# 既不会把 chunk 撕得过短，也不会让 chunk 过度漂移失去 size 控制语义。
_PARAGRAPH_SEARCH_RADIUS: int = 100


class CharWindowChunker:
    """字符窗口切片器（带段落优先 + 上限截断）。

    Args:
        chunk_size_chars: 单个 chunk 字符上限；段落对齐路径下实际长度可能略短。
        chunk_overlap_chars: 相邻 chunk 重叠字符数；要求 ``< chunk_size_chars``。
        max_chunks_per_doc: 单 doc 最多产出 chunk 数；超出截断。
    """

    __slots__ = ("_size", "_overlap", "_max_chunks")

    def __init__(
        self,
        *,
        chunk_size_chars: int,
        chunk_overlap_chars: int,
        max_chunks_per_doc: int,
    ) -> None:
        if chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars 必须 > 0")
        if not 0 <= chunk_overlap_chars < chunk_size_chars:
            raise ValueError("chunk_overlap_chars 必须 ∈ [0, chunk_size_chars)")
        if max_chunks_per_doc <= 0:
            raise ValueError("max_chunks_per_doc 必须 > 0")
        self._size = chunk_size_chars
        self._overlap = chunk_overlap_chars
        self._max_chunks = max_chunks_per_doc

    def chunk(self, doc: CorpusDocument, text: str) -> list[CorpusChunk]:
        """切片 ``text`` 并返回 :class:`CorpusChunk` 列表。

        - 空文本或纯空白返回 ``[]``，不发任何事件。
        - 超过 ``max_chunks_per_doc`` 时截断到上限并 dispatch 一次
          ``memory_rag_skipped`` 事件，事件 payload 含 ``doc_id`` / ``limit``。
        """
        if not text or not text.strip():
            return []

        chunks: list[CorpusChunk] = []
        pos = 0
        text_len = len(text)
        truncated = False
        while pos < text_len:
            # 候选切点：固定 size 推算。
            candidate_end = min(pos + self._size, text_len)
            end = self._align_to_paragraph(text, pos, candidate_end, text_len)

            chunk_text = text[pos:end]
            chunks.append(
                CorpusChunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks):04d}",
                    doc_id=doc.doc_id,
                    ord=len(chunks),
                    text=chunk_text,
                    char_offset=pos,
                    char_length=len(chunk_text),
                )
            )

            # 已达上限：截断 + 退出。
            if len(chunks) >= self._max_chunks:
                # 只在还有未覆盖文本时才算"被截断"：``end < text_len``。
                if end < text_len:
                    truncated = True
                break

            # 已覆盖到末尾：正常退出。
            if end >= text_len:
                break

            # 滑动到下一个窗口起点：保持 overlap。
            next_pos = end - self._overlap
            # 防御：next_pos 必须严格前进，否则会无限循环。``end > pos`` 总是成立
            # （chunk_text 非空），但 ``end - overlap`` 可能 <= pos。此时强制 +1。
            if next_pos <= pos:
                next_pos = pos + 1
            pos = next_pos

        if truncated:
            dispatch_observability_event(
                "memory_rag_skipped",
                {
                    "reason": "max_chunks_truncated",
                    "doc_id": doc.doc_id,
                    "limit": self._max_chunks,
                },
            )
        return chunks

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _align_to_paragraph(
        text: str, pos: int, candidate_end: int, text_len: int
    ) -> int:
        """把切点对齐到 ``±_PARAGRAPH_SEARCH_RADIUS`` 内最靠近的 ``"\\n\\n"``。

        - 若已到文本末尾（``candidate_end == text_len``）：不调整。
        - 在 ``[max(pos+1, candidate_end - radius), min(text_len, candidate_end + radius)]``
          区间内找最靠近 ``candidate_end`` 的 ``"\\n\\n"`` 起点 ``i``，把 end
          设为 ``i + 2``（让 ``"\\n\\n"`` 落在当前 chunk 内）。
        - 没有命中则返回原 candidate_end。

        这里"最靠近"的判定：先看 candidate_end **之前**最近的 ``"\\n\\n"``；
        没有再看 candidate_end **之后**最近的；都没有则不调整。这避免了
        chunk 过度向后扩展。
        """
        if candidate_end >= text_len:
            return candidate_end

        radius = _PARAGRAPH_SEARCH_RADIUS
        lo = max(pos + 1, candidate_end - radius)
        hi = min(text_len, candidate_end + radius)

        # 1) 候选切点之前最近的 ``\n\n``。
        before_idx = text.rfind("\n\n", lo, candidate_end)
        if before_idx != -1:
            return before_idx + 2  # 含 ``\n\n`` 在当前 chunk 内
        # 2) 候选切点之后最近的 ``\n\n``。
        after_idx = text.find("\n\n", candidate_end, hi)
        if after_idx != -1:
            return after_idx + 2
        return candidate_end


__all__ = ["CharWindowChunker"]
