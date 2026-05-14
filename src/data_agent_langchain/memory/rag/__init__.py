"""Corpus RAG 子包（M4 已落地，不再是 M4 前 placeholder）。

Bug 3 修复：去掉 ``build_corpus_retriever`` NotImplementedError 占位，仅顶层
re-export 轻量纯类型 / Protocol。

  - :class:`CorpusDocument` / :class:`CorpusChunk` / :data:`DocKind` —— 纯
    dataclass + 字面量类型，无重依赖。
  - :class:`CorpusStore` —— corpus 专属 store Protocol（B2 决策）。
  - :class:`Embedder` —— embedder Protocol（typing 纯接口）。
  - :class:`DeterministicStubEmbedder` —— 测试用确定性 stub，仅依赖 stdlib。

**故意不**顶层导出（D11 启动期 import 边界守护）：

  - ``HarrierEmbedder`` —— 隐藏在 ``embedders.sentence_transformer`` 模块；
    需要 ``torch`` / ``sentence_transformers`` 时由 ``factory`` 按需 import。
  - ``ChromaCorpusStore`` —— 隐藏在 ``stores.chroma`` 模块；需要 ``chromadb``
    时由 ``factory`` 按需 import。
  - ``factory`` 模块本身、``VectorCorpusRetriever`` —— 都通过 ``factory``
    工厂函数构造，不直接 export。

通过这种 re-export 边界，``import data_agent_langchain.memory.rag`` 不会让
``torch`` / ``chromadb`` / ``sentence_transformers`` 进 ``sys.modules``，
让 ``rag.enabled=false`` 路径零开销（计划 D11 / R5）。
"""
from __future__ import annotations

from data_agent_langchain.memory.rag.base import CorpusStore
from data_agent_langchain.memory.rag.documents import (
    CorpusChunk,
    CorpusDocument,
    DocKind,
)
from data_agent_langchain.memory.rag.embedders import (
    DeterministicStubEmbedder,
    Embedder,
)


__all__ = [
    "CorpusChunk",
    "CorpusDocument",
    "CorpusStore",
    "DeterministicStubEmbedder",
    "DocKind",
    "Embedder",
]
