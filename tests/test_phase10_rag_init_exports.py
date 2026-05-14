"""Bug 3 守卫测试：``data_agent_langchain.memory.rag`` 包顶层不再有 M4 前
``NotImplementedError`` 占位，且只 re-export 轻量纯类型。

设计意图（``02-implementation-plan-v2.md`` §File Structure / D11）：

  - 去掉 ``build_corpus_retriever`` 的 NotImplementedError 占位。
  - 仅顶层 re-export 纯类型 / Protocol：``CorpusDocument`` / ``CorpusChunk`` /
    ``DocKind`` / ``CorpusStore`` / ``Embedder`` / ``DeterministicStubEmbedder``。
  - **不**导出 ``HarrierEmbedder`` / ``ChromaCorpusStore`` / ``factory``，避免
    ``import data_agent_langchain.memory.rag`` 触发 ``torch`` / ``chromadb`` /
    ``sentence_transformers`` 加载（D11 启动期 import 边界）。

实测发现 v3.1 实施漏改了这个文件 —— 它仍然是 M4 启动前的占位 stub。
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Bug 3.1：包顶层不再有 build_corpus_retriever 占位
# ---------------------------------------------------------------------------


def test_rag_package_no_longer_exposes_placeholder_build_corpus_retriever() -> None:
    """``data_agent_langchain.memory.rag`` 不应再导出 ``build_corpus_retriever``。

    回归 Bug 3：占位 API 已被生产 API（``factory.build_task_corpus``）取代，
    残留会让上游误用 / 误抛 ``NotImplementedError``。
    """
    import data_agent_langchain.memory.rag as rag_pkg

    assert not hasattr(rag_pkg, "build_corpus_retriever"), (
        "data_agent_langchain.memory.rag 不应再有占位 build_corpus_retriever；"
        "改用 factory.build_task_corpus / factory.build_embedder。"
    )


# ---------------------------------------------------------------------------
# Bug 3.2：包顶层 re-export 纯类型 / Protocol
# ---------------------------------------------------------------------------


def test_rag_package_re_exports_pure_types_and_protocols() -> None:
    """``rag`` 包应顶层 re-export 这些**轻量**符号供下游直接 import。"""
    import data_agent_langchain.memory.rag as rag_pkg

    expected = {
        "CorpusDocument",
        "CorpusChunk",
        "DocKind",
        "CorpusStore",
        "Embedder",
        "DeterministicStubEmbedder",
    }
    missing = {name for name in expected if not hasattr(rag_pkg, name)}
    assert not missing, (
        f"data_agent_langchain.memory.rag 应顶层导出 {expected}，"
        f"缺失：{missing}"
    )


def test_rag_package_does_not_re_export_heavy_backends() -> None:
    """``rag`` 包**故意**不在 ``__all__`` 中声明重依赖 backend / factory 模块。

    注意：检查 ``__all__`` 而非 ``hasattr``。Python 的 sub-package import
    机制会让 ``from data_agent_langchain.memory.rag.factory import ...``
    自动把 ``factory`` 挂到父 package 上（即 ``rag_pkg.factory``），这是
    Python 行为不可避免；但只要 ``__all__`` 不包含这些名字，下游
    ``from data_agent_langchain.memory.rag import *`` 就不会拉到它们，且
    ``__init__.py`` 自身没有顶层显式 import 这些重依赖（D11 / R5 守护）。
    """
    import data_agent_langchain.memory.rag as rag_pkg

    forbidden = {
        "HarrierEmbedder",  # sentence_transformer / torch backend
        "ChromaCorpusStore",  # chromadb backend
        "VectorCorpusRetriever",  # 通过 factory 间接构造，不直接 export
        "factory",  # factory 模块本身按需 import
    }
    declared_all = set(rag_pkg.__all__ or ())
    leaked = forbidden & declared_all
    assert not leaked, (
        f"data_agent_langchain.memory.rag.__all__ 不应含 {forbidden}；"
        f"实际泄漏：{leaked}"
    )


# ---------------------------------------------------------------------------
# Bug 3.3：D11 启动期 import 边界 —— 顶层 import 不触发重依赖
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "import_target",
    [
        "data_agent_langchain.memory.rag",
        "data_agent_langchain",
    ],
)
def test_rag_package_import_does_not_load_heavy_backends(import_target: str) -> None:
    """``import data_agent_langchain.memory.rag`` (或顶层 package) 不应让
    ``torch`` / ``chromadb`` / ``sentence_transformers`` 出现在 ``sys.modules``。

    这与 ``test_phase10_rag_import_boundary.py`` 的 D11 守护互补 —— 后者已存在
    但只覆盖 ``data_agent_langchain`` 顶层；本测试同时覆盖 ``rag`` 子包。
    """
    script = textwrap.dedent(
        f"""\
        import sys
        import {import_target}  # noqa: F401
        leaked = [m for m in ('torch', 'chromadb', 'sentence_transformers') if m in sys.modules]
        if leaked:
            raise SystemExit(f'leaked modules at import time: {{leaked}}')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{import_target} import 触发了重依赖加载；stderr={result.stderr}"
    )
