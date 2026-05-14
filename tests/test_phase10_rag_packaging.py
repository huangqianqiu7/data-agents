"""``pyproject.toml`` 依赖分组守护测试（M4.4.2，C2 决策）。

按 ``01-design-v2.md §3 Conventions`` 与 ``D3`` 验证：

  - ``[project].dependencies``（运行最小集）**不**含 ``sentence-transformers`` /
    ``chromadb`` / ``torch``。
  - ``[project.optional-dependencies].rag`` 子组**含** ``sentence-transformers``
    与 ``chromadb``（torch 由 sentence-transformers 间接拉，无需在 rag 组内
    显式 pin，除非评测镜像需要严格 pin）。

这是 C2 决策的护栏：rag 重依赖只在 ``pip install ".[rag]"`` 时才装，提交镜像
不带 RAG 时体积零增量。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _load_pyproject() -> dict[str, Any]:
    """以 ``tomllib`` （Python 3.11+）解析仓库根 ``pyproject.toml``。"""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def _names_in_specs(specs: list[str]) -> set[str]:
    """从 ``"sentence-transformers>=3,<4"`` 这类规格字符串中抽出包名（小写）。"""
    out: set[str] = set()
    for spec in specs:
        # 取第一个非字母数字-下划线-连字符的字符之前的部分作为包名。
        name = ""
        for ch in spec.strip():
            if ch.isalnum() or ch in "-_.":
                name += ch
            else:
                break
        out.add(name.lower())
    return out


# ---------------------------------------------------------------------------
# 主依赖（运行最小集）禁包
# ---------------------------------------------------------------------------


def test_pyproject_dependencies_does_not_include_rag_extras() -> None:
    """``[project].dependencies`` 不应含 RAG 重依赖。"""
    pyproject = _load_pyproject()
    deps = pyproject.get("project", {}).get("dependencies", [])
    names = _names_in_specs(list(deps))

    forbidden = {"sentence-transformers", "chromadb", "torch"}
    leaked = forbidden & names
    assert not leaked, (
        f"运行最小集（[project].dependencies）不应含 RAG 重依赖：{leaked}（C2 违反）"
    )


# ---------------------------------------------------------------------------
# rag extra 必含
# ---------------------------------------------------------------------------


def test_pyproject_optional_rag_extra_exists() -> None:
    """``[project.optional-dependencies].rag`` 必须存在。"""
    pyproject = _load_pyproject()
    optional = pyproject.get("project", {}).get("optional-dependencies", {})
    assert "rag" in optional, (
        f"未声明 [project.optional-dependencies].rag，实际 keys: {sorted(optional.keys())}"
    )


def test_pyproject_rag_extra_includes_sentence_transformers() -> None:
    """``rag`` extra 必含 ``sentence-transformers``（HarrierEmbedder 后端）。"""
    pyproject = _load_pyproject()
    rag_specs = pyproject["project"]["optional-dependencies"].get("rag", [])
    names = _names_in_specs(list(rag_specs))
    assert "sentence-transformers" in names, (
        f"rag extra 缺少 sentence-transformers，实际：{sorted(names)}"
    )


def test_pyproject_rag_extra_includes_chromadb() -> None:
    """``rag`` extra 必含 ``chromadb``（ChromaCorpusStore 后端）。"""
    pyproject = _load_pyproject()
    rag_specs = pyproject["project"]["optional-dependencies"].get("rag", [])
    names = _names_in_specs(list(rag_specs))
    assert "chromadb" in names, (
        f"rag extra 缺少 chromadb，实际：{sorted(names)}"
    )
