"""Corpus RAG 测试用确定性 stub embedder（M4.2.1）。

仅用于单元测试和 corpus retriever 的整体连通验证；**不**承担任何真实语义。
基于 ``hashlib.shake_256`` 的可变长输出把文本散列成 ``dim`` 字节，再每字节
归一化为 ``[-1, 1]`` 区间内的 float，最后整体 L2 归一化。

为了让 ``embed_query`` 与 ``embed_documents`` 在同输入下产出**不同**向量
（避免 corpus 召回时 query 与 doc 完全对称的退化），``embed_query`` 在文本前
注入 ``"q:"`` 前缀。这与 Harrier 系列在 query 侧加 instruction prompt 的语义
保持一致（仅退化为最弱形式）。
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence


class DeterministicStubEmbedder:
    """确定性、可复现的 stub embedder，仅用于测试。"""

    __slots__ = ("_dim", "_model_id")

    def __init__(self, *, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim 必须 > 0，实际 {dim}")
        self._dim = dim
        # ``model_id`` 形如 ``"stub-deterministic-dim8"``，用于 metrics 归因区分。
        self._model_id = f"stub-deterministic-dim{dim}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码 doc 侧文本。空输入返回 ``[]``。"""
        if not texts:
            return []
        return [self._hash_to_unit_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """编码 query 文本；前缀 ``"q:"`` 模拟 Harrier 的 query instruction。"""
        return self._hash_to_unit_vector("q:" + text)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _hash_to_unit_vector(self, text: str) -> list[float]:
        """文本 → 单位向量。

        步骤：

        1. ``shake_256`` 输出 ``dim`` 字节（支持任意 dim，无需循环 sha256）。
        2. 每字节 ``b ∈ [0, 255]`` 线性映射到 ``[-1, 1)``：``(b - 128) / 128``。
        3. 整体 L2 归一化；零向量直接返回零向量（理论上 shake_256 不可能产出
           全 128 的 256 字节，所以这条边界仅为防御）。
        """
        digest: bytes = hashlib.shake_256(text.encode("utf-8")).digest(self._dim)
        raw = [(b - 128) / 128.0 for b in digest]
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0.0:
            return raw
        return [x / norm for x in raw]


__all__ = ["DeterministicStubEmbedder"]
