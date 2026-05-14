"""``_dataclass_from_dict`` 对嵌套 dataclass / tuple / Optional[Path] 的支持测试（M4.0.1）。

v3 corpus RAG 设计需要在 ``MemoryConfig`` 内嵌套 ``CorpusRagConfig``，并使用
``tuple[str, ...]`` 与 ``Path | None`` 字段。本文件用临时 dataclass 隔离验证
``_dataclass_from_dict`` 在这三类形状上的 round-trip 行为，避免直接耦合
``CorpusRagConfig`` 的具体字段（后者由 M4.4.1 引入）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from data_agent_langchain.config import _dataclass_from_dict, _to_plain_dict


# ---------------------------------------------------------------------------
# 1) 嵌套 dataclass：MemoryConfig.rag = CorpusRagConfig(...) 的形状代理
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Inner:
    name: str = "x"
    enabled: bool = False


@dataclass(frozen=True, slots=True)
class _Outer:
    inner: _Inner = field(default_factory=_Inner)
    title: str = "outer"


def test_nested_dataclass_roundtrip() -> None:
    """嵌套 dataclass 在 to_dict / from_dict 之间互逆，且类型不退化为 dict。"""
    original = _Outer(inner=_Inner(name="hello", enabled=True), title="t")
    payload = _to_plain_dict(original)
    assert payload == {"inner": {"name": "hello", "enabled": True}, "title": "t"}

    restored = _dataclass_from_dict(_Outer, payload)
    assert restored == original
    # 关键断言：内层必须还原为 _Inner 实例，而非 dict（v1 现状会留 dict）。
    assert isinstance(restored.inner, _Inner)


# ---------------------------------------------------------------------------
# 2) tuple[str, ...]：CorpusRagConfig.shared_collections / redact_patterns 的形状
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TupleHolder:
    tags: tuple[str, ...] = ()


def test_tuple_str_field_roundtrip() -> None:
    """tuple[str, ...] 经 ``_to_plain_dict`` 转 list 后，``_dataclass_from_dict`` 应还原回 tuple。"""
    original = _TupleHolder(tags=("a", "b", "c"))
    payload = _to_plain_dict(original)
    # ``_to_plain_dict`` 把 tuple 转为 list 以便 JSON / YAML 序列化。
    assert payload == {"tags": ["a", "b", "c"]}

    restored = _dataclass_from_dict(_TupleHolder, payload)
    assert restored == original
    # 关键断言：必须是 tuple，否则 frozen dataclass 的 hash / 比较会失败。
    assert isinstance(restored.tags, tuple)


def test_tuple_str_field_empty_roundtrip() -> None:
    """空 tuple 也要正确还原（默认值场景）。"""
    original = _TupleHolder()
    payload = _to_plain_dict(original)
    assert payload == {"tags": []}

    restored = _dataclass_from_dict(_TupleHolder, payload)
    assert restored == original
    assert restored.tags == ()
    assert isinstance(restored.tags, tuple)


# ---------------------------------------------------------------------------
# 3) Path | None：CorpusRagConfig.embedder_cache_dir / shared_path 的形状
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OptionalPathHolder:
    cache_dir: Path | None = None


def test_optional_path_field_none() -> None:
    """``Path | None`` 字段为 None 时保持 None，不被强制转 Path。"""
    original = _OptionalPathHolder(cache_dir=None)
    payload = _to_plain_dict(original)
    assert payload == {"cache_dir": None}

    restored = _dataclass_from_dict(_OptionalPathHolder, payload)
    assert restored == original
    assert restored.cache_dir is None


def test_optional_path_field_value(tmp_path: Path) -> None:
    """``Path | None`` 字段为 str 时自动转 Path（与现有 ``path`` 字段行为一致）。"""
    target = tmp_path / "models"
    original = _OptionalPathHolder(cache_dir=target)
    payload = _to_plain_dict(original)
    # ``_to_plain_dict`` 已把 Path 转为 str。
    assert payload == {"cache_dir": str(target)}

    restored = _dataclass_from_dict(_OptionalPathHolder, payload)
    # 关键断言：必须是 Path 实例，否则 ``cfg.cache_dir / "x"`` 这种调用会爆。
    assert isinstance(restored.cache_dir, Path)
    assert restored == original
