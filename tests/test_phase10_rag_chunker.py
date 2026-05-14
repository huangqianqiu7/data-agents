"""``CharWindowChunker`` 字符窗口切片行为测试（M4.1.4）。

按 ``01-design-v2.md §4.8`` 验证：

  - 空文本 / 无意义输入返回 ``[]``。
  - 短于 ``chunk_size_chars`` 的文本返回单 chunk。
  - 长文本切片数 ≈ ``ceil((char_count - overlap) / (size - overlap))``。
  - 相邻窗口重叠 ``chunk_overlap_chars`` 字符（窗口 N 末尾 overlap 字符 ==
    窗口 N+1 开头 overlap 字符）。
  - ``chunk_id`` 形如 ``f"{doc_id}#{ord:04d}"``，且对相同输入稳定。
  - 超过 ``max_chunks_per_doc`` 的部分被截断 + dispatch
    ``memory_rag_skipped(reason="max_chunks_truncated")``。
  - 段落优先：当切点附近 ±100 字符内有 ``\\n\\n``，优先在该处切。

设计偏离：构造器从 ``__init__(cfg)`` 改为接受独立的三个参数
``chunk_size_chars`` / ``chunk_overlap_chars`` / ``max_chunks_per_doc``，
与 ``Loader`` / ``Redactor`` 的解耦风格一致。
"""
from __future__ import annotations

from typing import Any

import pytest

from data_agent_langchain.memory.rag.chunker import CharWindowChunker
from data_agent_langchain.memory.rag.documents import CorpusChunk, CorpusDocument


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_doc(doc_id: str = "doc_test", char_count: int | None = None) -> CorpusDocument:
    """构造一个最小 ``CorpusDocument``。``char_count`` 不参与 chunker 行为，
    可任意。"""
    return CorpusDocument(
        doc_id=doc_id,
        source_path="README.md",
        doc_kind="markdown",
        bytes_size=0,
        char_count=char_count or 0,
        collection="task_corpus",
    )


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------


def test_chunker_empty_text_returns_empty_list() -> None:
    """空文本应返回 ``[]``。"""
    chunker = CharWindowChunker(
        chunk_size_chars=100, chunk_overlap_chars=10, max_chunks_per_doc=200
    )
    assert chunker.chunk(_make_doc(), "") == []


def test_chunker_whitespace_only_text_returns_empty_list() -> None:
    """纯空白文本（``"   "``）应被视为无内容，返回 ``[]``。"""
    chunker = CharWindowChunker(
        chunk_size_chars=100, chunk_overlap_chars=10, max_chunks_per_doc=200
    )
    assert chunker.chunk(_make_doc(), "   \n\n   ") == []


def test_chunker_short_text_yields_single_chunk() -> None:
    """文本长度 < ``chunk_size_chars`` → 单 chunk，且 ``ord=0``。"""
    chunker = CharWindowChunker(
        chunk_size_chars=100, chunk_overlap_chars=10, max_chunks_per_doc=200
    )
    text = "hello world"
    chunks = chunker.chunk(_make_doc(doc_id="abc"), text)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == text
    assert chunk.ord == 0
    assert chunk.chunk_id == "abc#0000"
    assert chunk.char_offset == 0
    assert chunk.char_length == len(text)


def test_chunker_text_equal_to_size_yields_single_chunk() -> None:
    """文本长度恰好等于 ``chunk_size_chars`` → 仍是单 chunk。"""
    chunker = CharWindowChunker(
        chunk_size_chars=20, chunk_overlap_chars=5, max_chunks_per_doc=200
    )
    text = "x" * 20
    chunks = chunker.chunk(_make_doc(), text)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].char_length == 20


# ---------------------------------------------------------------------------
# 长文本切片数量
# ---------------------------------------------------------------------------


def test_chunker_long_text_yields_overlapping_chunks() -> None:
    """长文本应切成多个 chunk，且相邻 chunk 在 ``overlap`` 字符上重叠。"""
    size, overlap = 50, 10
    chunker = CharWindowChunker(
        chunk_size_chars=size, chunk_overlap_chars=overlap, max_chunks_per_doc=200
    )
    # 用纯字符序列保证确定性（无段落分隔，绕开段落对齐启发式）。
    text = "".join(chr(ord("a") + i % 26) for i in range(200))
    chunks = chunker.chunk(_make_doc(), text)

    assert len(chunks) >= 4, f"长 200 字符按 size=50 / overlap=10 应至少 4 chunks，实际 {len(chunks)}"

    # 每个 chunk 长度 ≤ chunk_size_chars。
    for c in chunks:
        assert c.char_length <= size

    # 相邻 chunk 重叠 == overlap：window N 的末尾 overlap 字符 == window N+1 的开头 overlap 字符。
    for i in range(len(chunks) - 1):
        tail = chunks[i].text[-overlap:]
        head = chunks[i + 1].text[:overlap]
        assert tail == head, (
            f"chunk {i} 与 {i+1} overlap 不一致：tail={tail!r} head={head!r}"
        )


def test_chunker_chunk_count_matches_formula() -> None:
    """切片数应等于 ``ceil((char_count - overlap) / (size - overlap))``。"""
    import math

    size, overlap = 100, 20
    n = 500
    chunker = CharWindowChunker(
        chunk_size_chars=size, chunk_overlap_chars=overlap, max_chunks_per_doc=999
    )
    text = "x" * n
    chunks = chunker.chunk(_make_doc(), text)
    expected = math.ceil((n - overlap) / (size - overlap))
    assert len(chunks) == expected, f"期望 {expected} 个 chunks，实际 {len(chunks)}"


def test_chunker_chunk_offsets_are_monotonic() -> None:
    """``char_offset`` 应严格单调递增；末 chunk 覆盖到文本末尾。"""
    chunker = CharWindowChunker(
        chunk_size_chars=80, chunk_overlap_chars=15, max_chunks_per_doc=200
    )
    text = "abcdefgh" * 50  # 400 字符
    chunks = chunker.chunk(_make_doc(), text)

    offsets = [c.char_offset for c in chunks]
    assert offsets == sorted(offsets), f"offset 非单调递增：{offsets}"
    # 末 chunk 必须覆盖到末尾。
    last = chunks[-1]
    assert last.char_offset + last.char_length == len(text), (
        f"末 chunk 未覆盖到文本末尾：offset={last.char_offset} length={last.char_length} text_len={len(text)}"
    )


# ---------------------------------------------------------------------------
# chunk_id 与稳定性
# ---------------------------------------------------------------------------


def test_chunker_chunk_id_format() -> None:
    """``chunk_id`` 形如 ``f"{doc_id}#{ord:04d}"``。"""
    chunker = CharWindowChunker(
        chunk_size_chars=30, chunk_overlap_chars=5, max_chunks_per_doc=200
    )
    text = "x" * 100
    chunks = chunker.chunk(_make_doc(doc_id="myDoc"), text)
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"myDoc#{i:04d}", f"chunk_id 格式漂移：{c.chunk_id}"
        assert c.ord == i


def test_chunker_chunk_id_is_stable_across_calls() -> None:
    """同输入两次切片应产生完全相同的 chunk 列表。"""
    chunker = CharWindowChunker(
        chunk_size_chars=40, chunk_overlap_chars=8, max_chunks_per_doc=200
    )
    text = "abcde" * 40
    a = chunker.chunk(_make_doc(doc_id="d"), text)
    b = chunker.chunk(_make_doc(doc_id="d"), text)
    assert a == b


# ---------------------------------------------------------------------------
# 段落优先
# ---------------------------------------------------------------------------


def test_chunker_prefers_paragraph_boundary_when_available() -> None:
    """切点附近 ±100 字符内有 ``\\n\\n`` 时优先在段落边界切。"""
    size, overlap = 100, 20
    chunker = CharWindowChunker(
        chunk_size_chars=size, chunk_overlap_chars=overlap, max_chunks_per_doc=200
    )
    # 构造：第一段 80 字符 + ``\n\n`` + 第二段 200 字符。
    # 默认切点 = size = 100 = 在第二段内；附近 ±100 字符内有 ``\n\n`` 在 80 处。
    para1 = "a" * 80
    para2 = "b" * 200
    text = para1 + "\n\n" + para2
    chunks = chunker.chunk(_make_doc(), text)

    # 第一个 chunk 应在段落边界处结束（含 ``\n\n``，长度 = 80 或 82，含/不含分隔符）。
    first = chunks[0]
    # 段落对齐意味着第一 chunk 至少包含完整 para1，且不"咬掉"para2 的开头字符。
    # 灵活断言：first.text 不应跨越到 para2（即 ``"b"`` 不出现，或仅在 overlap 中）。
    assert first.text.startswith("a"), first.text[:20]
    # 无段落对齐时 first.text 长 100；有段落对齐应短于 100（在 ``\n\n`` 处切）。
    assert len(first.text) < size, (
        f"段落对齐未生效：first.text 长度 {len(first.text)} 应 < {size}"
    )


def test_chunker_no_paragraph_boundary_falls_back_to_size() -> None:
    """文本内无 ``\\n\\n`` 时，切点回退到 ``size`` 默认位置。"""
    size, overlap = 60, 10
    chunker = CharWindowChunker(
        chunk_size_chars=size, chunk_overlap_chars=overlap, max_chunks_per_doc=200
    )
    text = "x" * 200  # 无 ``\n\n``
    chunks = chunker.chunk(_make_doc(), text)
    # 第一 chunk 长度严格等于 size。
    assert chunks[0].char_length == size


# ---------------------------------------------------------------------------
# 截断
# ---------------------------------------------------------------------------


def test_chunker_truncates_when_exceeding_max_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """切片数超过 ``max_chunks_per_doc`` 时截断到上限。"""
    chunker = CharWindowChunker(
        chunk_size_chars=10, chunk_overlap_chars=2, max_chunks_per_doc=3
    )
    text = "x" * 500  # 远多于 3 个 chunk
    chunks = chunker.chunk(_make_doc(), text)
    assert len(chunks) == 3, f"max_chunks_per_doc=3 但收到 {len(chunks)} 个 chunks"


def test_chunker_dispatches_skipped_event_when_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """截断时 dispatch ``memory_rag_skipped(reason="max_chunks_truncated")``。"""
    captured: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "data_agent_langchain.memory.rag.chunker.dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data)),
    )

    chunker = CharWindowChunker(
        chunk_size_chars=10, chunk_overlap_chars=2, max_chunks_per_doc=3
    )
    chunker.chunk(_make_doc(doc_id="d"), "x" * 500)

    skipped = [d for n, d in captured if n == "memory_rag_skipped"]
    assert skipped, f"未发现 memory_rag_skipped 事件，captured={captured}"
    assert any(d.get("reason") == "max_chunks_truncated" for d in skipped), skipped
    truncated = next(d for d in skipped if d.get("reason") == "max_chunks_truncated")
    assert truncated.get("doc_id") == "d"
    assert truncated.get("limit") == 3


def test_chunker_no_event_when_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未截断时不应 dispatch 任何事件。"""
    captured: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "data_agent_langchain.memory.rag.chunker.dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data)),
    )

    chunker = CharWindowChunker(
        chunk_size_chars=100, chunk_overlap_chars=20, max_chunks_per_doc=200
    )
    chunker.chunk(_make_doc(), "hello world")
    assert not captured, f"未截断却 dispatch 了事件：{captured}"


# ---------------------------------------------------------------------------
# 类型契约
# ---------------------------------------------------------------------------


def test_chunker_returns_list_of_corpus_chunk() -> None:
    """返回值类型必须是 ``list[CorpusChunk]``，方便 store.upsert_chunks 直接消费。"""
    chunker = CharWindowChunker(
        chunk_size_chars=50, chunk_overlap_chars=10, max_chunks_per_doc=200
    )
    chunks = chunker.chunk(_make_doc(), "x" * 200)
    assert isinstance(chunks, list)
    for c in chunks:
        assert isinstance(c, CorpusChunk)
