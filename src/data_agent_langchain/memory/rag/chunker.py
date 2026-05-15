"""Corpus RAG markdown-aware chunker."""
from __future__ import annotations

from data_agent_langchain.memory.rag.documents import CorpusChunk, CorpusDocument
from data_agent_langchain.observability.events import dispatch_observability_event


_RECURSIVE_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "?", ". ", " ", "")


class MarkdownAwareChunker:
    """Hybrid chunker for markdown headers plus recursive character splitting."""

    __slots__ = (
        "_size",
        "_overlap",
        "_max_chunks",
        "_md_splitter",
        "_recursive_splitter_cls",
    )

    def __init__(
        self,
        *,
        chunk_size_chars: int,
        chunk_overlap_chars: int,
        max_chunks_per_doc: int,
    ) -> None:
        if chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars ?? > 0")
        if not 0 <= chunk_overlap_chars < chunk_size_chars:
            raise ValueError("chunk_overlap_chars ?? ? [0, chunk_size_chars)")
        if max_chunks_per_doc <= 0:
            raise ValueError("max_chunks_per_doc ?? > 0")

        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        self._size = chunk_size_chars
        self._overlap = chunk_overlap_chars
        self._max_chunks = max_chunks_per_doc
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=True,
        )
        self._recursive_splitter_cls = RecursiveCharacterTextSplitter

    def chunk(self, doc: CorpusDocument, text: str) -> list[CorpusChunk]:
        if not text or not text.strip():
            return []

        if doc.doc_kind == "markdown":
            raw_chunks = self._chunk_markdown(text)
        elif doc.doc_kind in {"text", "doc"}:
            raw_chunks = self._chunk_recursive_only(text)
        else:
            return []
        truncated = len(raw_chunks) > self._max_chunks
        if truncated:
            raw_chunks = raw_chunks[: self._max_chunks]

        chunks = [
            CorpusChunk(
                chunk_id=f"{doc.doc_id}#{ord_idx:04d}",
                doc_id=doc.doc_id,
                ord=ord_idx,
                text=chunk_text,
                char_offset=char_offset,
                char_length=len(chunk_text),
            )
            for ord_idx, (chunk_text, char_offset) in enumerate(raw_chunks)
        ]

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

    def _split_recursive(self, text: str, *, chunk_size: int | None = None) -> list[str]:
        size = self._size if chunk_size is None else chunk_size
        overlap = min(self._overlap, max(0, size - 1))
        splitter = self._recursive_splitter_cls(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=list(_RECURSIVE_SEPARATORS),
            length_function=len,
        )
        return splitter.split_text(text)

    def _chunk_markdown(self, text: str) -> list[tuple[str, int]]:
        sections = self._md_splitter.split_text(text)
        if not sections or not any(section.metadata for section in sections):
            return self._chunk_recursive_only(text)

        result: list[tuple[str, int]] = []
        section_search_pos = 0
        for section in sections:
            prefix = self._build_breadcrumb(section.metadata)
            body = section.page_content
            if not body.strip():
                continue

            section_offset = self._find_offset(text, body, section_search_pos)
            section_search_pos = max(section_search_pos, section_offset + 1)
            body_budget = max(1, self._size - len(prefix)) if prefix else self._size
            if prefix and len(prefix) >= self._size:
                body_budget = 1

            parts = (
                [body]
                if len(prefix) + len(body) <= self._size
                else self._split_recursive(body, chunk_size=body_budget)
            )
            local_search_pos = section_offset
            local_overlap = min(self._overlap, max(0, body_budget - 1))
            for part in parts:
                part_offset = self._find_offset(text, part, local_search_pos)
                chunk_text = f"{prefix}{part}" if prefix else part
                result.append((chunk_text, part_offset))
                local_search_pos = max(
                    part_offset + 1,
                    part_offset + max(1, len(part) - local_overlap),
                )
        return result

    def _chunk_recursive_only(self, text: str) -> list[tuple[str, int]]:
        result: list[tuple[str, int]] = []
        search_pos = 0
        for chunk in self._split_recursive(text):
            offset = self._find_offset(text, chunk, search_pos)
            result.append((chunk, offset))
            search_pos = max(offset + 1, offset + max(1, len(chunk) - self._overlap))
        return result

    @staticmethod
    def _find_offset(text: str, needle: str, start_at: int) -> int:
        bounded_start = max(0, min(start_at, len(text)))
        if not needle:
            return bounded_start
        idx = text.find(needle, bounded_start)
        if idx >= 0:
            return idx
        sample = needle[:80].strip()
        if sample:
            idx = text.find(sample, bounded_start)
            if idx >= 0:
                return idx
        return bounded_start

    @staticmethod
    def _build_breadcrumb(metadata: dict[str, str]) -> str:
        parts = [
            value.strip()
            for key in ("h1", "h2", "h3")
            if (value := metadata.get(key))
        ]
        if not parts:
            return ""
        return "> " + " > ".join(parts) + "\n\n"


__all__ = ["MarkdownAwareChunker"]
