from __future__ import annotations

from typing import Any

import pytest

from data_agent_langchain.memory.rag.chunker import MarkdownAwareChunker
from data_agent_langchain.memory.rag.documents import CorpusChunk, CorpusDocument


def _make_doc(
    *,
    doc_id: str = "doc_test",
    doc_kind: str = "markdown",
    source_path: str = "README.md",
) -> CorpusDocument:
    return CorpusDocument(
        doc_id=doc_id,
        source_path=source_path,
        doc_kind=doc_kind,
        bytes_size=0,
        char_count=0,
        collection="task_corpus",
    )


def _chunker(
    *,
    chunk_size_chars: int = 100,
    chunk_overlap_chars: int = 10,
    max_chunks_per_doc: int = 200,
) -> MarkdownAwareChunker:
    return MarkdownAwareChunker(
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        max_chunks_per_doc=max_chunks_per_doc,
    )


def test_empty_text_returns_empty_list() -> None:
    assert _chunker().chunk(_make_doc(), "") == []


def test_whitespace_only_returns_empty_list() -> None:
    assert _chunker().chunk(_make_doc(), "   \n\n   ") == []


def test_chunk_id_format_stable() -> None:
    chunker = _chunker(chunk_size_chars=40, chunk_overlap_chars=5)
    text = "# Title\n\n" + "body " * 40
    first = chunker.chunk(_make_doc(doc_id="stable"), text)
    second = chunker.chunk(_make_doc(doc_id="stable"), text)

    assert first == second
    for index, chunk in enumerate(first):
        assert chunk.chunk_id == f"stable#{index:04d}"
        assert chunk.ord == index


def test_chunk_offsets_monotonic() -> None:
    chunker = _chunker(chunk_size_chars=80, chunk_overlap_chars=15)
    text = "# Title\n\n" + "abcdefgh " * 80
    chunks = chunker.chunk(_make_doc(), text)

    offsets = [chunk.char_offset for chunk in chunks]
    assert offsets == sorted(offsets)


def test_chunker_returns_list_of_corpus_chunk() -> None:
    chunks = _chunker(chunk_size_chars=50, chunk_overlap_chars=10).chunk(
        _make_doc(), "# Title\n\n" + "x " * 100
    )

    assert isinstance(chunks, list)
    assert chunks
    assert all(isinstance(chunk, CorpusChunk) for chunk in chunks)


def test_markdown_splits_on_h1_h2_h3() -> None:
    text = """# Knowledge Guide

## Section A

Alpha body.

## Section B

Beta body.

## Section C

### Detail

Gamma body.
"""

    chunks = _chunker(chunk_size_chars=500, chunk_overlap_chars=20).chunk(_make_doc(), text)

    assert len(chunks) >= 3
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "> Knowledge Guide > Section A" in joined
    assert "> Knowledge Guide > Section B" in joined
    assert "> Knowledge Guide > Section C > Detail" in joined


def test_markdown_chunk_text_contains_header_prefix() -> None:
    text = """# Knowledge Guide

## Core Entities

### Patient

Patient records describe demographics.
"""

    chunks = _chunker(chunk_size_chars=500, chunk_overlap_chars=20).chunk(_make_doc(), text)

    assert chunks[0].text.startswith(
        "> Knowledge Guide > Core Entities > Patient\n\nPatient records"
    )


def test_markdown_large_section_falls_back_to_recursive() -> None:
    body = "\n\n".join(f"Paragraph {i} " + "clinical facts " * 8 for i in range(12))
    text = f"# Knowledge Guide\n\n## Large Section\n\n{body}"

    chunks = _chunker(chunk_size_chars=200, chunk_overlap_chars=20).chunk(_make_doc(), text)

    assert len(chunks) > 1
    assert all(
        chunk.text.startswith("> Knowledge Guide > Large Section\n\n") for chunk in chunks
    )


def test_markdown_prefix_counts_toward_chunk_size() -> None:
    body = "\n\n".join(f"Paragraph {i} " + "evidence " * 10 for i in range(10))
    text = f"# Guide\n\n## Section\n\n{body}"

    chunks = _chunker(chunk_size_chars=160, chunk_overlap_chars=20).chunk(_make_doc(), text)

    assert len(chunks) > 1
    assert all(chunk.char_length <= 160 for chunk in chunks)


def test_markdown_no_headers_falls_back_to_recursive() -> None:
    text = "Plain markdown-looking content without headings. " * 20

    chunks = _chunker(chunk_size_chars=120, chunk_overlap_chars=20).chunk(_make_doc(), text)

    assert len(chunks) > 1
    assert all(not chunk.text.startswith("> ") for chunk in chunks)


def test_markdown_repeated_section_text_offsets_stay_monotonic() -> None:
    repeated = "Repeated body paragraph."
    text = f"# Guide\n\n## One\n\n{repeated}\n\n## Two\n\n{repeated}"

    chunks = _chunker(chunk_size_chars=500, chunk_overlap_chars=20).chunk(_make_doc(), text)

    offsets = [chunk.char_offset for chunk in chunks]
    assert offsets == sorted(offsets)
    assert offsets[1] > offsets[0]


@pytest.mark.parametrize("doc_kind", ["text", "doc"])
def test_non_markdown_doc_kind_uses_recursive_only(doc_kind: str) -> None:
    text = "# This is content, not a markdown header for this doc_kind. " * 20

    chunks = _chunker(chunk_size_chars=120, chunk_overlap_chars=20).chunk(
        _make_doc(doc_kind=doc_kind, source_path="note.txt"), text
    )

    assert len(chunks) > 1
    assert all(not chunk.text.startswith("> ") for chunk in chunks)


def test_json_schema_note_doc_kind_is_not_chunked() -> None:
    text = "{\"schema\": " + "\"field\": \"value\", " * 100 + "}"

    chunks = _chunker(chunk_size_chars=120, chunk_overlap_chars=20).chunk(
        _make_doc(doc_kind="json_schema_note", source_path="json/Patient.json"), text
    )

    assert chunks == []


def test_truncates_when_exceeding_max_chunks() -> None:
    chunks = _chunker(
        chunk_size_chars=50, chunk_overlap_chars=5, max_chunks_per_doc=3
    ).chunk(_make_doc(), "# Title\n\n" + "x " * 500)

    assert len(chunks) == 3


def test_dispatches_skipped_event_when_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "data_agent_langchain.memory.rag.chunker.dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data)),
    )

    _chunker(chunk_size_chars=50, chunk_overlap_chars=5, max_chunks_per_doc=3).chunk(
        _make_doc(doc_id="d"), "# Title\n\n" + "x " * 500
    )

    skipped = [data for name, data in captured if name == "memory_rag_skipped"]
    assert skipped
    assert skipped[0]["reason"] == "max_chunks_truncated"
    assert skipped[0]["doc_id"] == "d"
    assert skipped[0]["limit"] == 3


def test_no_event_when_not_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "data_agent_langchain.memory.rag.chunker.dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data)),
    )

    _chunker(chunk_size_chars=100, chunk_overlap_chars=20).chunk(
        _make_doc(), "# Title\n\nshort body"
    )

    assert captured == []


def test_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        _chunker(chunk_size_chars=0, chunk_overlap_chars=0)


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        _chunker(chunk_size_chars=10, chunk_overlap_chars=-1)
    with pytest.raises(ValueError):
        _chunker(chunk_size_chars=10, chunk_overlap_chars=10)
