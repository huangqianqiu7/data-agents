"""``HarrierEmbedder`` sentence-transformers 后端测试（M4.2.2）。

按 ``01-design-v2.md §4.6`` 验证：

  - 顶层 import 边界（D11，**不**带 ``slow`` marker）：
    ``from data_agent_langchain.memory.rag.embedders.sentence_transformer import HarrierEmbedder``
    后 ``sys.modules`` 不应含 ``torch``（class 定义不触发 import）。
  - 配置 wiring（用 mock 替换 ``SentenceTransformer``，**不**带 ``slow`` marker）：
      * ``self._model_id`` 必须赋值（v1 bug 修复 D6）。
      * ``model_kwargs={"dtype": ...}`` 始终传，包括 ``"auto"``。
      * ``device`` / ``cache_folder`` / ``max_seq_length`` 正确透传。
      * ``embed_documents([])`` 返回 ``[]``；非空走 ``model.encode``。
      * ``embed_query`` 走 ``prompt_name=query_prompt_name``。
      * batch_size 控制单次 forward 的批量。
  - 真权重集成（**带** ``@pytest.mark.slow``）：本地 HF 缓存可用时验证维度 > 0、
    向量 L2 归一化等。

设计偏离：``HarrierEmbedder.__init__`` 接受独立参数（非 ``CorpusRagConfig``
整体），与 ``Loader`` / ``Redactor`` / ``Chunker`` 解耦风格一致；factory 在
M4.4.3 包装 cfg → 参数。
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# 顶层 import 边界（D11）—— 非 slow，每次必跑
# ---------------------------------------------------------------------------


def test_harrier_module_does_not_import_torch_at_module_load() -> None:
    """``import HarrierEmbedder`` 不应在模块加载阶段拉 ``torch``。

    用 subprocess 启 fresh Python 进程，import 只到 class 层（不构造实例），
    随后断言 ``sys.modules`` 不含 ``torch``。
    这是 D11 启动期 import 边界的核心守护。
    """
    code = """
import sys

# import 模块；不构造任何 HarrierEmbedder 实例。
from data_agent_langchain.memory.rag.embedders.sentence_transformer import HarrierEmbedder  # noqa

leaked = [m for m in ('torch', 'sentence_transformers') if m in sys.modules]
assert not leaked, f'重依赖在模块加载阶段被 import：{leaked}'
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={
            **__import__("os").environ,
            # 让 src 在 sys.path 上；conftest 的逻辑在 subprocess 里不会跑。
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src"),
        },
    )
    assert result.returncode == 0, (
        f"子进程失败：stdout={result.stdout}, stderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# Wiring 单元测试（mock SentenceTransformer，非 slow）
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """替换 ``sentence_transformers.SentenceTransformer`` 为可观察 mock。

    返回 MagicMock 类对象；测试函数可通过 ``mock.call_args`` 访问构造参数，
    通过 ``mock.return_value.encode.return_value`` 配置 ``encode`` 输出。
    """
    fake_class = MagicMock(name="SentenceTransformer")
    fake_instance = fake_class.return_value
    fake_instance.max_seq_length = 8192  # SentenceTransformer 的真实默认值之一
    fake_instance.get_sentence_embedding_dimension.return_value = 384
    # ``encode`` 默认返回单位向量列表（避免下游归一化逻辑抛错）。
    fake_instance.encode.return_value = [
        [1.0] + [0.0] * 383,  # 一个简单的单位向量
    ]

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_class)
    # 同时替换 HarrierEmbedder 内部 lazy import 后 cache 的位置（如果有）。
    return fake_class


def test_harrier_assigns_model_id(mock_sentence_transformer: MagicMock) -> None:
    """v1 bug 修复 D6：``self._model_id`` 必须赋值，``model_id`` property 返回它。"""
    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    embedder = HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
    )
    assert embedder.model_id == "microsoft/harrier-oss-v1-270m"


def test_harrier_passes_dtype_auto_to_model_kwargs(
    mock_sentence_transformer: MagicMock,
) -> None:
    """v1 bug 修复 D6：``dtype="auto"`` 也应通过 ``model_kwargs`` 传给 SentenceTransformer。"""
    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
    )
    args, kwargs = mock_sentence_transformer.call_args
    assert kwargs.get("model_kwargs") == {"dtype": "auto"}, (
        f"应始终传 model_kwargs={{'dtype': 'auto'}}，实际 {kwargs}"
    )


def test_harrier_passes_dtype_float16(
    mock_sentence_transformer: MagicMock,
) -> None:
    """``dtype="float16"`` 也应正确透传。"""
    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="float16",
    )
    args, kwargs = mock_sentence_transformer.call_args
    assert kwargs.get("model_kwargs") == {"dtype": "float16"}


def test_harrier_passes_device_and_cache_folder(
    mock_sentence_transformer: MagicMock, tmp_path: Path
) -> None:
    """``device`` 与 ``cache_dir`` 应正确透传给 SentenceTransformer。"""
    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    cache = tmp_path / "hf_cache"
    HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
        cache_dir=cache,
    )
    args, kwargs = mock_sentence_transformer.call_args
    assert kwargs.get("device") == "cpu"
    assert kwargs.get("cache_folder") == str(cache)


def test_harrier_clamps_max_seq_length(
    mock_sentence_transformer: MagicMock,
) -> None:
    """``max_seq_length`` 应被钳制到 ``min(model.max_seq_length, max_seq_len)``。"""
    fake_instance = mock_sentence_transformer.return_value
    fake_instance.max_seq_length = 8192  # SentenceTransformer 默认很大

    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
        max_seq_len=512,
    )
    # 钳制：min(8192, 512) = 512
    assert fake_instance.max_seq_length == 512


def test_harrier_dimension_from_model(
    mock_sentence_transformer: MagicMock,
) -> None:
    """``dimension`` 应来自 ``model.get_sentence_embedding_dimension()``。"""
    fake_instance = mock_sentence_transformer.return_value
    fake_instance.get_sentence_embedding_dimension.return_value = 768

    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    embedder = HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
    )
    assert embedder.dimension == 768


def test_harrier_embed_documents_empty_returns_empty_list(
    mock_sentence_transformer: MagicMock,
) -> None:
    """空输入返回 ``[]``，不触发 model.encode。"""
    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    embedder = HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
    )
    fake_instance = mock_sentence_transformer.return_value
    assert embedder.embed_documents([]) == []
    fake_instance.encode.assert_not_called()


def test_harrier_embed_documents_returns_list_of_list_of_float(
    mock_sentence_transformer: MagicMock,
) -> None:
    """``embed_documents`` 必须返回 ``list[list[float]]``，**不**是 numpy。"""
    fake_instance = mock_sentence_transformer.return_value
    fake_instance.encode.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    embedder = HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
    )
    out = embedder.embed_documents(["a", "b"])
    assert isinstance(out, list)
    assert len(out) == 2
    for vec in out:
        assert isinstance(vec, list)
        for x in vec:
            assert isinstance(x, float)


def test_harrier_embed_documents_normalizes_output(
    mock_sentence_transformer: MagicMock,
) -> None:
    """``embed_documents`` 输出向量应 L2 归一化（最终责任在 Embedder 实现层）。

    传入未归一化的 mock encode 输出，验证 ``HarrierEmbedder`` 自行归一化。
    """
    fake_instance = mock_sentence_transformer.return_value
    # 未归一化向量（norm = 5）
    fake_instance.encode.return_value = [[3.0, 4.0]]

    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    embedder = HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
    )
    out = embedder.embed_documents(["x"])
    norm = math.sqrt(sum(v * v for v in out[0]))
    assert abs(norm - 1.0) < 1e-6, f"L2 范数偏离 1：{norm}"


def test_harrier_embed_query_uses_prompt_name(
    mock_sentence_transformer: MagicMock,
) -> None:
    """``embed_query`` 必须传 ``prompt_name=query_prompt_name`` 给 model.encode。"""
    fake_instance = mock_sentence_transformer.return_value
    fake_instance.encode.return_value = [[1.0, 0.0, 0.0]]

    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    embedder = HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
        query_prompt_name="web_search_query",
    )
    embedder.embed_query("what is X?")

    # encode 至少被调用过一次（构造时不调，embed_query 时调）。
    assert fake_instance.encode.called
    last_call = fake_instance.encode.call_args
    # 检查 prompt_name 关键字参数。
    assert last_call.kwargs.get("prompt_name") == "web_search_query", (
        f"embed_query 应传 prompt_name='web_search_query'，实际 {last_call.kwargs}"
    )


def test_harrier_embed_query_returns_list_of_float(
    mock_sentence_transformer: MagicMock,
) -> None:
    """``embed_query`` 返回 ``list[float]``，单向量。"""
    fake_instance = mock_sentence_transformer.return_value
    fake_instance.encode.return_value = [[1.0, 0.0, 0.0]]

    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    embedder = HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
    )
    vec = embedder.embed_query("hi")
    assert isinstance(vec, list)
    for x in vec:
        assert isinstance(x, float)


def test_harrier_resolve_device_auto_falls_back_to_cpu_when_cuda_unavailable(
    mock_sentence_transformer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``device="auto"`` 在 CUDA 不可用时应安静回退 ``"cpu"``，不抛错。"""
    # 模拟 torch.cuda.is_available() 返回 False。
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="auto",
        dtype="auto",
    )
    args, kwargs = mock_sentence_transformer.call_args
    assert kwargs.get("device") == "cpu", (
        f"auto 路径应回退 cpu，实际 {kwargs.get('device')}"
    )


def test_harrier_satisfies_embedder_protocol(
    mock_sentence_transformer: MagicMock,
) -> None:
    """``HarrierEmbedder`` 应满足 ``Embedder`` Protocol（runtime_checkable）。"""
    from data_agent_langchain.memory.rag.embedders import Embedder
    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )

    embedder = HarrierEmbedder(
        model_id="microsoft/harrier-oss-v1-270m",
        device="cpu",
        dtype="auto",
    )
    assert isinstance(embedder, Embedder)


# ---------------------------------------------------------------------------
# 真权重集成（slow）—— 仅 --runslow + 本地缓存可用时跑
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_harrier_real_weights_dimension_and_normalization() -> None:
    """加载真 ``microsoft/harrier-oss-v1-270m`` 权重并验证基本不变量。

    没有 HF 缓存或离线时本测试会因 ``OSError`` 而 skip。
    """
    try:
        from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
            HarrierEmbedder,
        )

        embedder = HarrierEmbedder(
            model_id="microsoft/harrier-oss-v1-270m",
            device="cpu",
            dtype="auto",
        )
    except OSError as exc:
        pytest.skip(f"HF 缓存不可用，跳过真权重集成：{exc}")

    assert embedder.dimension > 0
    assert embedder.model_id == "microsoft/harrier-oss-v1-270m"

    docs = embedder.embed_documents(["hello world", "another doc"])
    assert len(docs) == 2
    for vec in docs:
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-3, f"L2 范数偏离 1：{norm}"

    query = embedder.embed_query("what is hello?")
    norm = math.sqrt(sum(x * x for x in query))
    assert abs(norm - 1.0) < 1e-3
