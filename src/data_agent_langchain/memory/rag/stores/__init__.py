"""Corpus RAG store 实现子包。

**故意不**在此 ``__init__`` 中 import ``ChromaCorpusStore`` —— 后者顶层注释
里再次声明：方法级延迟 import ``chromadb``。如需使用：

.. code-block:: python

    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

这样 ``rag.enabled=false`` 与 chromadb 未装的环境也能安全 import 父包。
"""
from __future__ import annotations


__all__: list[str] = []
