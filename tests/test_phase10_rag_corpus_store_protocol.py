"""``CorpusStore`` Protocol 接口契约测试（M4.3.1，B2 守护）。

按 ``01-design-v2.md §4.9`` 验证：

  - ``CorpusStore`` 是 ``@runtime_checkable Protocol``。
  - 仅暴露 ``namespace`` / ``dimension`` / ``upsert_chunks`` / ``query_by_vector``
    / ``close`` 五个成员。
  - **不**暴露 v2 ``MemoryStore`` 的 ``put`` / ``get`` / ``list`` / ``delete``
    —— B2 决策：corpus store 与 KV store 解耦，避免任何代码误把 chroma store
    当 v2 KV memory 用。
"""
from __future__ import annotations

from data_agent_langchain.memory.rag.base import CorpusStore


def test_corpus_store_is_runtime_checkable_protocol() -> None:
    """``CorpusStore`` 应是 ``@runtime_checkable Protocol`` 以便 isinstance 校验。"""
    # ``Protocol`` 实例化会抛 TypeError；``@runtime_checkable`` 标记后可用 isinstance。
    # 测试 protocol 自身的属性。
    from typing import _ProtocolMeta  # type: ignore[attr-defined]

    assert isinstance(CorpusStore, _ProtocolMeta), (
        "CorpusStore 应是 typing.Protocol 子类（带 _ProtocolMeta 元类）"
    )
    # ``runtime_checkable`` 设置 ``_is_runtime_protocol = True``。
    assert getattr(CorpusStore, "_is_runtime_protocol", False), (
        "CorpusStore 应被 @runtime_checkable 装饰"
    )


def test_corpus_store_protocol_has_only_five_members() -> None:
    """``CorpusStore`` 仅声明 5 个成员，避免接口蔓延。"""
    expected = {
        "namespace",
        "dimension",
        "upsert_chunks",
        "query_by_vector",
        "close",
    }
    members = {
        m for m in dir(CorpusStore)
        if not m.startswith("_") and m not in {"register"}
    }
    assert members == expected, (
        f"CorpusStore Protocol 接口漂移：{members}（期望 {expected}）"
    )


def test_corpus_store_protocol_does_not_expose_memory_store_kv_methods() -> None:
    """B2 决策守护：``CorpusStore`` 必须**不**含 v2 ``MemoryStore`` 的 KV 方法。

    这样可以防止下游代码误把 ``ChromaCorpusStore`` 当作 v2 KV memory 用，
    或反过来把 v2 ``MemoryStore`` 当作 corpus store 用。
    """
    forbidden = {"put", "get", "list", "delete"}
    members = {
        m for m in dir(CorpusStore)
        if not m.startswith("_") and m not in {"register"}
    }
    leaked = forbidden & members
    assert not leaked, (
        f"CorpusStore Protocol 出现 v2 MemoryStore 的 KV 方法：{leaked}（B2 违反）"
    )
