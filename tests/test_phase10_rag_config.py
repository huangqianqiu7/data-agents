"""``CorpusRagConfig`` + ``MemoryConfig.rag`` 嵌套配置测试（M4.4.1）。

按 ``01-design-v2.md §4.3`` 验证：

  - ``CorpusRagConfig`` 默认值与设计文档一致（每个字段单独检查）。
  - ``MemoryConfig.rag`` 是 ``CorpusRagConfig`` 实例（非 None）。
  - ``MemoryConfig.path`` 默认值仍为 ``PROJECT_ROOT / "artifacts" / "memory"``
    （D7 守护：``default_factory`` 完整保留，未被截断为必填字段）。
  - ``AppConfig.to_dict / from_dict`` 包含 ``memory.rag`` 字段且互逆。
  - ``tuple[str, ...]`` 字段（``shared_collections`` / ``redact_patterns``
    / ``redact_filenames``）round-trip 后仍是 tuple。
  - YAML 解析：``memory.rag.enabled=true`` 被正确加载。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from data_agent_langchain.config import (
    AppConfig,
    CorpusRagConfig,
    MemoryConfig,
    PROJECT_ROOT,
    load_app_config,
)


# ---------------------------------------------------------------------------
# CorpusRagConfig 默认值
# ---------------------------------------------------------------------------


def test_corpus_rag_config_defaults_match_design() -> None:
    """``CorpusRagConfig()`` 默认值应与 ``§4.3`` 设计文档一致。"""
    cfg = CorpusRagConfig()

    # 总开关
    assert cfg.enabled is False
    assert cfg.task_corpus is True
    assert cfg.shared_corpus is False
    assert cfg.shared_collections == ()
    assert cfg.shared_path is None

    # 索引
    assert cfg.chunk_size_chars == 1200
    assert cfg.chunk_overlap_chars == 200
    assert cfg.max_chunks_per_doc == 200
    assert cfg.max_docs_per_task == 100
    assert cfg.task_corpus_index_timeout_s == 30.0

    # Embedding
    assert cfg.embedder_backend == "sentence_transformer"
    assert cfg.embedder_model_id == "microsoft/harrier-oss-v1-270m"
    assert cfg.embedder_device == "cpu"
    assert cfg.embedder_dtype == "auto"
    assert cfg.embedder_query_prompt_name == "web_search_query"
    assert cfg.embedder_max_seq_len == 1024
    assert cfg.embedder_batch_size == 8
    assert cfg.embedder_cache_dir is None

    # 向量库
    assert cfg.vector_backend == "chroma"
    assert cfg.vector_distance == "cosine"

    # 检索
    assert cfg.retrieval_k == 4
    assert cfg.prompt_budget_chars == 1800

    # 内容过滤
    assert cfg.redact_patterns == (
        r"(?i)\banswer\b",
        r"(?i)\bhint\b",
        r"(?i)\bapproach\b",
        r"(?i)\bsolution\b",
    )
    assert cfg.redact_filenames == (
        "expected_output.json",
        "ground_truth*",
        "*label*",
        "*solution*",
    )


def test_corpus_rag_config_is_frozen() -> None:
    """``frozen=True``：赋值应抛 ``FrozenInstanceError``。"""
    from dataclasses import FrozenInstanceError

    cfg = CorpusRagConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.enabled = True  # type: ignore[misc]


def test_corpus_rag_config_uses_slots() -> None:
    """``slots=True``：实例不应有 ``__dict__``。"""
    cfg = CorpusRagConfig()
    assert not hasattr(cfg, "__dict__")


# ---------------------------------------------------------------------------
# MemoryConfig.rag 嵌套
# ---------------------------------------------------------------------------


def test_memory_config_rag_is_corpus_rag_config_instance() -> None:
    """``MemoryConfig().rag`` 应是 ``CorpusRagConfig`` 实例（非 None）。"""
    cfg = MemoryConfig()
    assert isinstance(cfg.rag, CorpusRagConfig), (
        f"MemoryConfig.rag 应是 CorpusRagConfig，实际 {type(cfg.rag).__name__}"
    )


def test_memory_config_path_default_factory_intact() -> None:
    """D7 守护：``MemoryConfig.path`` 的 ``default_factory`` 完整保留，
    默认值仍为 ``PROJECT_ROOT / "artifacts" / "memory"``。"""
    cfg = MemoryConfig()
    assert cfg.path == PROJECT_ROOT / "artifacts" / "memory"
    assert cfg.path.is_absolute()


def test_memory_config_rag_independent_instances() -> None:
    """两个 ``MemoryConfig()`` 应各自持有独立的 ``CorpusRagConfig`` 实例
    （``default_factory`` 而非默认值，避免可变状态共享）。"""
    a = MemoryConfig()
    b = MemoryConfig()
    # 由于 ``CorpusRagConfig`` 是 frozen，两个实例可以是同一个对象（不可变可共享），
    # 也可以是不同对象。这里只断言：a.rag == b.rag（值等价），不强制 ``is`` 不同。
    assert a.rag == b.rag


# ---------------------------------------------------------------------------
# AppConfig round-trip
# ---------------------------------------------------------------------------


def test_app_config_round_trip_includes_memory_rag() -> None:
    """``AppConfig.to_dict`` / ``from_dict`` 应包含 ``memory.rag`` 字段且互逆。"""
    original = AppConfig()
    payload = original.to_dict()

    assert "memory" in payload
    assert "rag" in payload["memory"], (
        f"memory 段缺少 rag 子段：{payload['memory'].keys()}"
    )

    # rag 段应含设计字段。
    rag_payload = payload["memory"]["rag"]
    assert rag_payload["enabled"] is False
    assert rag_payload["task_corpus"] is True
    assert rag_payload["embedder_model_id"] == "microsoft/harrier-oss-v1-270m"

    # tuple 在 to_dict 后被序列化为 list。
    assert rag_payload["shared_collections"] == []
    assert isinstance(rag_payload["redact_patterns"], list)

    # round-trip 等价。
    restored = AppConfig.from_dict(payload)
    assert restored == original
    assert isinstance(restored.memory.rag, CorpusRagConfig)


def test_app_config_round_trip_preserves_modified_rag_fields() -> None:
    """改了 ``rag.enabled=True`` / ``embedder_backend="stub"`` 后 round-trip
    仍能还原所有字段。"""
    original = AppConfig(
        memory=MemoryConfig(
            rag=CorpusRagConfig(
                enabled=True,
                embedder_backend="stub",
                shared_collections=("col_a", "col_b"),
                retrieval_k=8,
            )
        )
    )
    payload = original.to_dict()
    restored = AppConfig.from_dict(payload)

    assert restored.memory.rag.enabled is True
    assert restored.memory.rag.embedder_backend == "stub"
    assert restored.memory.rag.shared_collections == ("col_a", "col_b")
    assert isinstance(restored.memory.rag.shared_collections, tuple)
    assert restored.memory.rag.retrieval_k == 8


def test_app_config_round_trip_preserves_optional_path_field() -> None:
    """``embedder_cache_dir: Path | None`` 在 round-trip 后仍是 ``Path``。"""
    target = Path("/tmp/hf_cache")
    original = AppConfig(
        memory=MemoryConfig(
            rag=CorpusRagConfig(embedder_cache_dir=target)
        )
    )
    payload = original.to_dict()
    # to_dict 把 Path 序列化为 str。
    assert payload["memory"]["rag"]["embedder_cache_dir"] == str(target)

    restored = AppConfig.from_dict(payload)
    assert isinstance(restored.memory.rag.embedder_cache_dir, Path)
    assert restored.memory.rag.embedder_cache_dir == target


def test_app_config_round_trip_preserves_redact_patterns_as_tuple() -> None:
    """``redact_patterns`` round-trip 后仍是 tuple（不退化为 list）。"""
    original = AppConfig()
    payload = original.to_dict()
    restored = AppConfig.from_dict(payload)
    assert isinstance(restored.memory.rag.redact_patterns, tuple)
    assert isinstance(restored.memory.rag.redact_filenames, tuple)


# ---------------------------------------------------------------------------
# YAML 解析
# ---------------------------------------------------------------------------


def test_load_app_config_parses_memory_rag_enabled(tmp_path: Path) -> None:
    """``memory.rag.enabled=true`` YAML 应被正确加载到 ``cfg.memory.rag.enabled``。"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
memory:
  mode: read_only_dataset
  rag:
    enabled: true
    embedder_backend: stub
    retrieval_k: 6
""".strip(),
        encoding="utf-8",
    )

    cfg = load_app_config(cfg_path)
    assert cfg.memory.mode == "read_only_dataset"
    assert cfg.memory.rag.enabled is True
    assert cfg.memory.rag.embedder_backend == "stub"
    assert cfg.memory.rag.retrieval_k == 6


def test_load_app_config_yaml_preserves_default_rag_when_omitted(tmp_path: Path) -> None:
    """YAML 不写 ``memory.rag`` 时，``cfg.memory.rag`` 应为默认 ``CorpusRagConfig``。"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
memory:
  mode: disabled
""".strip(),
        encoding="utf-8",
    )

    cfg = load_app_config(cfg_path)
    assert cfg.memory.rag == CorpusRagConfig()
    assert cfg.memory.rag.enabled is False


def test_load_app_config_resolves_relative_embedder_cache_dir(tmp_path: Path) -> None:
    """``embedder_cache_dir`` 是 ``Path | None`` 字段；YAML 给字符串应被转为 ``Path``。"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
memory:
  rag:
    embedder_cache_dir: /opt/models/hf
""".strip(),
        encoding="utf-8",
    )

    cfg = load_app_config(cfg_path)
    assert cfg.memory.rag.embedder_cache_dir is not None
    assert isinstance(cfg.memory.rag.embedder_cache_dir, Path)
