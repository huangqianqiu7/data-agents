"""``CorpusDocument`` / ``CorpusChunk`` dataclass 形状与不变量测试（M4.1.1）。

按 ``01-design-v2.md §4.4`` 验证：

  - 字段名与顺序（用 ``dataclasses.fields`` 反射）。
  - ``frozen=True`` —— 赋值抛 ``FrozenInstanceError``。
  - ``slots=True`` —— 实例没有 ``__dict__``。
  - 实例 ``pickle.dumps`` / ``pickle.loads`` 互逆（v2 picklability 不变量）。
  - **故意不存在** 的字段：``question`` / ``answer`` / ``hint`` / ``approach`` /
    ``predicted_label`` —— 写入路径禁令（§6.1 数据流禁令）。
"""
from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, fields

import pytest

from data_agent_langchain.memory.rag.documents import CorpusChunk, CorpusDocument


# ---------------------------------------------------------------------------
# CorpusDocument
# ---------------------------------------------------------------------------


_DOC_FIELD_NAMES = (
    "doc_id",
    "source_path",
    "doc_kind",
    "bytes_size",
    "char_count",
    "collection",
)


def test_corpus_document_field_names_match_design() -> None:
    """``CorpusDocument`` 字段名与顺序必须与 ``§4.4`` 设计一致。"""
    actual = tuple(f.name for f in fields(CorpusDocument))
    assert actual == _DOC_FIELD_NAMES, (
        f"字段名/顺序漂移：期望 {_DOC_FIELD_NAMES}，实际 {actual}"
    )


def test_corpus_document_is_frozen() -> None:
    """``frozen=True``：赋值应抛 ``FrozenInstanceError``。"""
    doc = CorpusDocument(
        doc_id="abc123",
        source_path="README.md",
        doc_kind="markdown",
        bytes_size=100,
        char_count=80,
        collection="task_corpus",
    )
    with pytest.raises(FrozenInstanceError):
        doc.doc_id = "tampered"  # type: ignore[misc]


def test_corpus_document_uses_slots() -> None:
    """``slots=True``：实例不应有 ``__dict__``，避免意外属性污染 pickle。"""
    doc = CorpusDocument(
        doc_id="abc123",
        source_path="README.md",
        doc_kind="markdown",
        bytes_size=100,
        char_count=80,
        collection="task_corpus",
    )
    assert not hasattr(doc, "__dict__"), "CorpusDocument 应使用 slots，禁止 __dict__"


def test_corpus_document_is_picklable() -> None:
    """``pickle.dumps`` / ``loads`` 互逆，确保子进程可序列化（v2 §13）。"""
    doc = CorpusDocument(
        doc_id="abc123",
        source_path="data_schema.md",
        doc_kind="text",
        bytes_size=42,
        char_count=40,
        collection="shared:dabench-handbook",
    )
    restored = pickle.loads(pickle.dumps(doc))
    assert restored == doc
    assert isinstance(restored, CorpusDocument)


def test_corpus_document_does_not_have_forbidden_fields() -> None:
    """禁令：``question`` / ``answer`` / ``hint`` / ``approach`` / ``predicted_label``
    不得作为 ``CorpusDocument`` 字段。"""
    forbidden = {"question", "answer", "hint", "approach", "predicted_label"}
    actual = {f.name for f in fields(CorpusDocument)}
    leaked = forbidden & actual
    assert not leaked, f"CorpusDocument 出现禁字段：{leaked}"


# ---------------------------------------------------------------------------
# CorpusChunk
# ---------------------------------------------------------------------------


_CHUNK_FIELD_NAMES = (
    "chunk_id",
    "doc_id",
    "ord",
    "text",
    "char_offset",
    "char_length",
)


def test_corpus_chunk_field_names_match_design() -> None:
    """``CorpusChunk`` 字段名与顺序必须与 ``§4.4`` 设计一致。"""
    actual = tuple(f.name for f in fields(CorpusChunk))
    assert actual == _CHUNK_FIELD_NAMES, (
        f"字段名/顺序漂移：期望 {_CHUNK_FIELD_NAMES}，实际 {actual}"
    )


def test_corpus_chunk_is_frozen() -> None:
    """``frozen=True``：赋值应抛 ``FrozenInstanceError``。"""
    chunk = CorpusChunk(
        chunk_id="abc123#0001",
        doc_id="abc123",
        ord=1,
        text="hello world",
        char_offset=0,
        char_length=11,
    )
    with pytest.raises(FrozenInstanceError):
        chunk.text = "tampered"  # type: ignore[misc]


def test_corpus_chunk_uses_slots() -> None:
    """``slots=True``：实例不应有 ``__dict__``。"""
    chunk = CorpusChunk(
        chunk_id="abc123#0000",
        doc_id="abc123",
        ord=0,
        text="hello",
        char_offset=0,
        char_length=5,
    )
    assert not hasattr(chunk, "__dict__"), "CorpusChunk 应使用 slots，禁止 __dict__"


def test_corpus_chunk_is_picklable() -> None:
    """``pickle.dumps`` / ``loads`` 互逆。"""
    chunk = CorpusChunk(
        chunk_id="abc123#0042",
        doc_id="abc123",
        ord=42,
        text="some text",
        char_offset=120,
        char_length=9,
    )
    restored = pickle.loads(pickle.dumps(chunk))
    assert restored == chunk
    assert isinstance(restored, CorpusChunk)


def test_corpus_chunk_id_format_is_reconstructible() -> None:
    """``chunk_id`` 形如 ``f"{doc_id}#{ord:04d}"``，可由 ``doc_id + ord`` 重建。

    本测试通过手动构造预期 ID 验证设计契约：调用方负责生成 ID，dataclass
    本身不强校验，但格式约定写在 docstring 与 §4.4。
    """
    doc_id = "abc123"
    ord_ = 7
    expected_chunk_id = f"{doc_id}#{ord_:04d}"
    chunk = CorpusChunk(
        chunk_id=expected_chunk_id,
        doc_id=doc_id,
        ord=ord_,
        text="x",
        char_offset=0,
        char_length=1,
    )
    assert chunk.chunk_id == "abc123#0007"
    # 反向重建必须命中。
    assert chunk.chunk_id == f"{chunk.doc_id}#{chunk.ord:04d}"


def test_corpus_chunk_does_not_have_forbidden_fields() -> None:
    """禁令：``question`` / ``answer`` / ``hint`` / ``approach`` / ``summary``
    不得作为 ``CorpusChunk`` 字段。"""
    forbidden = {"question", "answer", "hint", "approach", "summary"}
    actual = {f.name for f in fields(CorpusChunk)}
    leaked = forbidden & actual
    assert not leaked, f"CorpusChunk 出现禁字段：{leaked}"
