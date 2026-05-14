"""``Embedder`` Protocol 与 ``DeterministicStubEmbedder`` 行为测试（M4.2.1）。

按 ``01-design-v2.md §4.5`` 验证：

  - ``Embedder`` 是 ``@runtime_checkable Protocol``，仅暴露 ``model_id`` /
    ``dimension`` / ``embed_documents`` / ``embed_query`` 四个成员。
  - ``DeterministicStubEmbedder`` 满足 ``Embedder`` Protocol 契约：
      * ``dimension`` 等于构造参数 ``dim``。
      * ``model_id == f"stub-deterministic-dim{dim}"``。
      * ``embed_documents`` 对相同输入产出相同向量；不同输入产出不同向量。
      * ``embed_query(text)`` 在 query 文本前注入 ``"q:"`` 前缀，结果与
        ``embed_documents([text])[0]`` 不同。
      * 返回类型严格为 ``list[list[float]]`` / ``list[float]``，不是 numpy ndarray。
      * 所有向量 L2 范数 ≈ 1（归一化）。
"""
from __future__ import annotations

import math
from typing import Sequence

import pytest

from data_agent_langchain.memory.rag.embedders import (
    DeterministicStubEmbedder,
    Embedder,
)


# ---------------------------------------------------------------------------
# Embedder Protocol
# ---------------------------------------------------------------------------


def test_embedder_is_runtime_checkable_protocol() -> None:
    """``Embedder`` 应是 ``@runtime_checkable Protocol``，可用 ``isinstance`` 校验。"""
    embedder = DeterministicStubEmbedder(dim=8)
    assert isinstance(embedder, Embedder), (
        "DeterministicStubEmbedder 应满足 Embedder Protocol（runtime_checkable）"
    )


def test_embedder_protocol_has_only_four_members() -> None:
    """``Embedder`` 仅声明 ``model_id`` / ``dimension`` / ``embed_documents`` /
    ``embed_query`` 四个成员，避免接口蔓延。"""
    members = {
        m for m in dir(Embedder)
        if not m.startswith("_") and m not in {"register"}
    }
    assert members == {
        "model_id",
        "dimension",
        "embed_documents",
        "embed_query",
    }, f"Embedder Protocol 接口漂移：{members}"


# ---------------------------------------------------------------------------
# DeterministicStubEmbedder 基本属性
# ---------------------------------------------------------------------------


def test_stub_dimension_matches_constructor_arg() -> None:
    """``dimension`` 应等于构造时传入的 ``dim``。"""
    assert DeterministicStubEmbedder(dim=8).dimension == 8
    assert DeterministicStubEmbedder(dim=16).dimension == 16
    assert DeterministicStubEmbedder(dim=384).dimension == 384


def test_stub_model_id_format() -> None:
    """``model_id`` 形如 ``f"stub-deterministic-dim{dim}"``。"""
    assert DeterministicStubEmbedder(dim=8).model_id == "stub-deterministic-dim8"
    assert (
        DeterministicStubEmbedder(dim=384).model_id == "stub-deterministic-dim384"
    )


def test_stub_rejects_invalid_dim() -> None:
    """``dim <= 0`` 应抛 ``ValueError``。"""
    with pytest.raises(ValueError):
        DeterministicStubEmbedder(dim=0)
    with pytest.raises(ValueError):
        DeterministicStubEmbedder(dim=-3)


# ---------------------------------------------------------------------------
# embed_documents
# ---------------------------------------------------------------------------


def test_stub_embed_documents_empty_returns_empty_list() -> None:
    """空输入应返回空列表。"""
    embedder = DeterministicStubEmbedder(dim=8)
    assert embedder.embed_documents([]) == []


def test_stub_embed_documents_returns_list_of_list_of_float() -> None:
    """返回类型严格为 ``list[list[float]]``，**不**是 numpy。"""
    embedder = DeterministicStubEmbedder(dim=8)
    out = embedder.embed_documents(["hello", "world"])
    assert isinstance(out, list), f"应返回 list，实际 {type(out).__name__}"
    assert len(out) == 2
    for vec in out:
        assert isinstance(vec, list), f"每个 vector 应是 list，实际 {type(vec).__name__}"
        assert len(vec) == 8
        for x in vec:
            assert isinstance(x, float), f"vector 元素应是 float，实际 {type(x).__name__}"


def test_stub_embed_documents_same_text_returns_same_vector() -> None:
    """相同文本两次调用返回相同向量（确定性）。"""
    embedder = DeterministicStubEmbedder(dim=16)
    a = embedder.embed_documents(["hello world"])[0]
    b = embedder.embed_documents(["hello world"])[0]
    assert a == b, f"同输入向量不一致：a={a[:3]}... b={b[:3]}..."


def test_stub_embed_documents_different_text_returns_different_vector() -> None:
    """不同文本应产出不同向量（哈希分散）。"""
    embedder = DeterministicStubEmbedder(dim=16)
    a = embedder.embed_documents(["hello"])[0]
    b = embedder.embed_documents(["world"])[0]
    assert a != b, "不同文本应产生不同向量"


def test_stub_embed_documents_vectors_are_l2_normalized() -> None:
    """所有 doc 向量的 L2 范数应 ≈ 1（归一化不变量）。"""
    embedder = DeterministicStubEmbedder(dim=32)
    vectors = embedder.embed_documents(["hello", "world", "foo bar baz"])
    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-6, f"L2 范数偏离 1：{norm}"


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


def test_stub_embed_query_returns_list_of_float() -> None:
    """``embed_query`` 返回类型为 ``list[float]``。"""
    embedder = DeterministicStubEmbedder(dim=8)
    vec = embedder.embed_query("question")
    assert isinstance(vec, list)
    assert len(vec) == 8
    for x in vec:
        assert isinstance(x, float)


def test_stub_embed_query_is_l2_normalized() -> None:
    """query 向量 L2 范数应 ≈ 1。"""
    embedder = DeterministicStubEmbedder(dim=16)
    vec = embedder.embed_query("what is X?")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6, f"L2 范数偏离 1：{norm}"


def test_stub_embed_query_differs_from_doc_for_same_text() -> None:
    """query 端注入 ``"q:"`` 前缀，因此与 doc 端同输入向量不同（避免对称性）。"""
    embedder = DeterministicStubEmbedder(dim=16)
    text = "hello world"
    doc_vec = embedder.embed_documents([text])[0]
    query_vec = embedder.embed_query(text)
    assert doc_vec != query_vec, (
        "query 端应在文本前注入 'q:' 前缀，使其与 doc 端同文本向量不同"
    )


def test_stub_embed_query_is_deterministic() -> None:
    """同 query 两次调用应返回相同向量。"""
    embedder = DeterministicStubEmbedder(dim=16)
    a = embedder.embed_query("hello")
    b = embedder.embed_query("hello")
    assert a == b


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------


def test_stub_embed_documents_handles_empty_string() -> None:
    """空字符串也应能产出归一化向量（不抛错）。"""
    embedder = DeterministicStubEmbedder(dim=8)
    vec = embedder.embed_documents([""])[0]
    assert len(vec) == 8
    norm = math.sqrt(sum(x * x for x in vec))
    # 空串可能产出零向量；如此则约定返回零向量或 fallback 到统一向量。
    # 当前实现选择：空串也走 hash 路径（hashlib 接受空 bytes），所以 norm == 1。
    assert abs(norm - 1.0) < 1e-6 or norm == 0.0


def test_stub_embed_documents_unicode() -> None:
    """中文等非 ASCII 文本应能正常 embed。"""
    embedder = DeterministicStubEmbedder(dim=8)
    vec = embedder.embed_documents(["中文测试"])[0]
    assert len(vec) == 8
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6
