"""Corpus RAG ``Embedder`` Protocol（M4.2.1）。

按 ``01-design-v2.md §4.5`` 定义文本 → L2 归一化向量的协议接口。

设计要点：

  - **doc / query 分离**：Harrier 等 instruction-tuned 模型在 query 端必须加
    instruction prompt（``web_search_query``），doc 端不加。Protocol 通过
    ``embed_documents`` / ``embed_query`` 两个分离接口承载该差异。
  - **返回 ``list[list[float]]``**：简化 picklability、跨进程传输、与 chromadb
    ``upsert(embeddings=...)`` 的 list-shape 期望对齐。**不**返回 numpy。
  - **L2 归一化由实现负责**：Protocol 不再约束（避免重复归一化的开销），
    实现者必须保证返回的向量 L2 范数 ≈ 1。
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """文本 → L2 归一化向量。"""

    @property
    def model_id(self) -> str:
        """嵌入模型标识（HF model id 或 stub 标记），用于 metrics 归因。"""
        ...

    @property
    def dimension(self) -> int:
        """输出向量维度；与 chromadb collection 的 ``embedding_function=None``
        + 手动 upsert 路径配合使用。"""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文档侧文本（无 query instruction）。空输入返回 ``[]``。"""
        ...

    def embed_query(self, text: str) -> list[float]:
        """编码 query 文本（应注入 query instruction prompt）。"""
        ...


__all__ = ["Embedder"]
