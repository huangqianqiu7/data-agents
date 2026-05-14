"""Corpus RAG embedder 子包。

仅导出**轻量**接口与 stub：

  - :class:`Embedder` —— 协议接口，纯 typing，无重依赖。
  - :class:`DeterministicStubEmbedder` —— 测试用确定性 stub，仅依赖 stdlib。

**故意不**导出 ``HarrierEmbedder`` —— 后者在 ``sentence_transformer.py`` 内
方法级延迟 import ``sentence_transformers`` / ``torch``，仅在 ``factory`` 内
按需 ``from data_agent_langchain.memory.rag.embedders.sentence_transformer import HarrierEmbedder``
显式 import。这避免 ``rag.enabled=false`` 路径触发任何重依赖加载（D11 启动期
import 边界）。
"""
from __future__ import annotations

from data_agent_langchain.memory.rag.embedders.base import Embedder
from data_agent_langchain.memory.rag.embedders.stub import DeterministicStubEmbedder


__all__ = ["Embedder", "DeterministicStubEmbedder"]
