"""``HarrierEmbedder``：sentence-transformers 后端（M4.2.2）。

按 ``01-design-v2.md §4.6`` 实现 Harrier-OSS-v1-270m 等基于
``sentence_transformers.SentenceTransformer`` 的嵌入模型适配。

**启动期 import 边界（D11）**：

  - 本模块**顶层**只 import 标准库与 typing；**不** import ``sentence_transformers``
    或 ``torch``。
  - 重依赖在 :meth:`HarrierEmbedder.__init__` 与 :meth:`_resolve_device` 内
    **方法级延迟 import**。
  - 守护：``test_phase10_rag_embedder_harrier.test_harrier_module_does_not_import_torch_at_module_load``
    + ``test_phase10_rag_import_boundary.py`` 用 subprocess 验证。

**v1 设计 bug 修复（D6）**：

  - ``self._model_id`` 必须赋值（v1 例子漏了，``model_id`` property 返回 None）。
  - ``model_kwargs={"dtype": ...}`` 始终传，包括 ``"auto"``（HF model card
    推荐写法）。

设计偏离：构造器接受独立参数（非 ``CorpusRagConfig`` 整体），与
``Loader`` / ``Redactor`` / ``Chunker`` / ``DeterministicStubEmbedder`` 解耦
风格一致；``factory.build_embedder`` 在 M4.4.3 把 cfg 字段拆开传入。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Sequence


class HarrierEmbedder:
    """sentence-transformers 后端，默认承载 ``microsoft/harrier-oss-v1-270m``。

    Args:
        model_id: HF model id，例如 ``"microsoft/harrier-oss-v1-270m"``。
        device: 设备字面量；``"auto"`` 会尝试 cuda，失败回退 cpu。
        dtype: ``model_kwargs={"dtype": dtype}``；``"auto"`` 让 HF 自行决定。
        query_prompt_name: 调用 ``embed_query`` 时传给 ``model.encode`` 的
            ``prompt_name``；Harrier 系列默认 ``"web_search_query"``。
        max_seq_len: 最大序列长度上限；与 ``model.max_seq_length`` 取 min 钳制。
        batch_size: ``embed_documents`` 单次 forward 的批量。
        cache_dir: HF cache 目录；``None`` 走默认 ``HF_HOME`` / ``~/.cache/huggingface``。
    """

    __slots__ = (
        "_model_id",
        "_query_prompt_name",
        "_batch_size",
        "_model",
        "_dim",
    )

    def __init__(
        self,
        *,
        model_id: str,
        device: Literal["cpu", "cuda", "auto"] = "cpu",
        dtype: Literal["float32", "float16", "auto"] = "auto",
        query_prompt_name: str = "web_search_query",
        max_seq_len: int = 1024,
        batch_size: int = 8,
        cache_dir: Path | None = None,
    ) -> None:
        # 方法级延迟 import：``rag.enabled=false`` 路径不付出 torch 加载代价。
        from sentence_transformers import SentenceTransformer

        # v1 bug 修复 D6：必须赋值，否则 ``model_id`` property 返回 None。
        self._model_id = model_id
        self._query_prompt_name = query_prompt_name
        self._batch_size = batch_size

        resolved_device = self._resolve_device(device)

        self._model = SentenceTransformer(
            model_id,
            device=resolved_device,
            cache_folder=str(cache_dir) if cache_dir is not None else None,
            # v1 bug 修复 D6：``dtype="auto"`` 也要传；HF model card 推荐
            # 始终传 ``model_kwargs={"dtype": ...}``。
            model_kwargs={"dtype": dtype},
        )
        # 钳制最大序列长度：避免某些 model card 默认 8192 token 撑爆显存。
        self._model.max_seq_length = min(
            int(self._model.max_seq_length), int(max_seq_len)
        )
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """解析 ``device`` 参数；``"auto"`` 时尝试 cuda，失败安静回退 cpu。"""
        if requested != "auto":
            return requested
        # 方法级延迟 import：仅 ``auto`` 路径才拉 torch（且即便 torch 未装
        # 也不抛，回退 cpu）。
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码 doc 侧文本（无 query instruction）。

        空输入返回 ``[]`` 且**不**调用 ``model.encode``，避免 sentence-transformers
        对空 batch 的边界行为差异。
        """
        if not texts:
            return []
        vectors: Any = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=False,
            normalize_embeddings=False,  # 由本类自行做 L2 归一化，保持输出一致
        )
        return [self._to_unit_list(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """编码 query 文本，传 ``prompt_name=self._query_prompt_name``。"""
        vectors: Any = self._model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=False,
            normalize_embeddings=False,
            prompt_name=self._query_prompt_name,
        )
        return self._to_unit_list(vectors[0])

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_unit_list(vector: Any) -> list[float]:
        """将向量（可能是 torch.Tensor / numpy ndarray / list）转为 L2 归一化
        ``list[float]``，便于 picklability 与跨进程传输。"""
        # tolist() 同时支持 torch.Tensor / numpy.ndarray / list。
        raw_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        floats: list[float] = [float(x) for x in raw_list]
        norm = math.sqrt(sum(x * x for x in floats))
        if norm == 0.0:
            return floats
        return [x / norm for x in floats]


__all__ = ["HarrierEmbedder"]
