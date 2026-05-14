"""``factory.build_embedder`` / ``build_task_corpus`` 行为测试（M4.4.3）。

按 ``01-design-v2.md §4.11`` 验证：

  ``build_embedder(cfg)``：

    - ``cfg.enabled=False`` → 返回 ``None``。
    - ``embedder_backend="stub"`` → 返回 ``DeterministicStubEmbedder``。
    - ``embedder_backend="sentence_transformer"`` → 返回 ``HarrierEmbedder``
      （用 monkeypatch 把 ``SentenceTransformer`` 替换为 mock，避免加载真权重）。
    - HarrierEmbedder 构造失败（``ImportError`` / ``OSError``）→ 返回 ``None``
      + dispatch ``memory_rag_skipped(reason="embedder_load_failed")``。

  ``build_task_corpus(cfg, *, task_id, task_input_dir, embedder, config)``：

    - ``cfg.enabled=False`` 或 ``cfg.task_corpus=False`` → 返回 ``None``。
    - 目录不存在 → 返回 ``None`` + dispatch ``memory_rag_skipped(reason="no_documents")``。
    - fixture 目录 → 返回 :class:`TaskCorpusHandles`，含 store / retriever /
      embedder；索引内 chunk 数等于 fixture 文档切片总数。
    - 构建成功 → dispatch ``memory_rag_index_built`` 含 doc_count /
      chunk_count / model_id / dimension / elapsed_ms。
    - 超时 → 返回 ``None`` + dispatch ``memory_rag_skipped(reason="index_timeout")``。
    - ``shared_corpus=True`` 但 M4 未实现 → dispatch
      ``memory_rag_skipped(reason="shared_corpus_not_implemented")``（task 继续）。

设计偏离：``stub`` backend 的 dim 不可由 ``CorpusRagConfig`` 配置（不引入
``embedder_stub_dim`` 字段污染 prod 配置）；factory 用固定 ``dim=16`` 构造
``DeterministicStubEmbedder``，足够测试用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from data_agent_langchain.config import CorpusRagConfig
from data_agent_langchain.memory.rag.documents import CorpusChunk
from data_agent_langchain.memory.rag.embedders import (
    DeterministicStubEmbedder,
    Embedder,
)


@pytest.fixture()
def captured_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """替换 factory 内部使用的 dispatch 函数，收集事件。"""
    captured: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "data_agent_langchain.memory.rag.factory.dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data)),
    )
    return captured


# ---------------------------------------------------------------------------
# build_embedder
# ---------------------------------------------------------------------------


def test_build_embedder_returns_none_when_rag_disabled() -> None:
    """``cfg.enabled=False`` 时 ``build_embedder`` 应返回 ``None``。"""
    from data_agent_langchain.memory.rag.factory import build_embedder

    cfg = CorpusRagConfig(enabled=False)
    assert build_embedder(cfg) is None


def test_build_embedder_returns_stub_when_backend_is_stub() -> None:
    """``embedder_backend="stub"`` 应返回 ``DeterministicStubEmbedder`` 实例。"""
    from data_agent_langchain.memory.rag.factory import build_embedder

    cfg = CorpusRagConfig(enabled=True, embedder_backend="stub")
    embedder = build_embedder(cfg)
    assert isinstance(embedder, DeterministicStubEmbedder)
    # dim 固定 16 即可。
    assert embedder.dimension == 16


def test_build_embedder_returns_harrier_when_backend_is_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embedder_backend="sentence_transformer"`` 应返回 ``HarrierEmbedder``。"""
    # 用 mock 替换 SentenceTransformer，避免拉真权重。
    fake_class = MagicMock(name="SentenceTransformer")
    fake_instance = fake_class.return_value
    fake_instance.max_seq_length = 8192
    fake_instance.get_sentence_embedding_dimension.return_value = 384

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_class)

    from data_agent_langchain.memory.rag.embedders.sentence_transformer import (
        HarrierEmbedder,
    )
    from data_agent_langchain.memory.rag.factory import build_embedder

    cfg = CorpusRagConfig(
        enabled=True,
        embedder_backend="sentence_transformer",
        embedder_model_id="microsoft/harrier-oss-v1-270m",
    )
    embedder = build_embedder(cfg)
    assert isinstance(embedder, HarrierEmbedder)
    assert embedder.model_id == "microsoft/harrier-oss-v1-270m"


def test_build_embedder_returns_none_on_harrier_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    captured_events: list[tuple[str, dict[str, Any]]],
) -> None:
    """HarrierEmbedder 构造抛错（如 ``OSError`` HF 缓存不可用）→ 返回 None +
    dispatch ``memory_rag_skipped(reason="embedder_load_failed")``。"""
    # 强制 SentenceTransformer 构造抛 OSError，模拟无 HF 缓存。
    import sentence_transformers

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise OSError("simulated: HF cache not available")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _raise)

    from data_agent_langchain.memory.rag.factory import build_embedder

    cfg = CorpusRagConfig(
        enabled=True,
        embedder_backend="sentence_transformer",
    )
    result = build_embedder(cfg)
    assert result is None

    skipped = [d for n, d in captured_events if n == "memory_rag_skipped"]
    assert any(d.get("reason") == "embedder_load_failed" for d in skipped), (
        f"未捕获 embedder_load_failed 事件，实际 captured={captured_events}"
    )


# ---------------------------------------------------------------------------
# build_task_corpus
# ---------------------------------------------------------------------------


@pytest.fixture()
def context_with_docs(tmp_path: Path) -> Path:
    """构造一个含 3 个 markdown 文档的 task ``context_dir``。"""
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "README.md").write_text("# Title\nIntro paragraph.\n", encoding="utf-8")
    (ctx / "data_schema.md").write_text(
        "# Schema\nA: int\nB: string\nC: float\n", encoding="utf-8"
    )
    (ctx / "notes.txt").write_text("Free notes paragraph.\n", encoding="utf-8")
    return ctx


@pytest.fixture()
def unique_task_id(request: pytest.FixtureRequest) -> str:
    """为每个测试生成唯一 ``task_id``（Bug 4 defensive fix）。

    背景：``chromadb.EphemeralClient`` 在同一 Python 进程内是单例缓存设计，
    collection name = ``f"memrag_{sha1(corpus_task:<task_id>)[:16]}"``；如果所有
    测试都用 ``task_id="t1"``，collection name 固定，跨测试累积 chunk。
    本 fixture 以测试节点 ID 生成唯一后缀 → 不同 collection → 隔离。
    与 ``ChromaCorpusStore.close()`` 的 upstream fix 互补：upstream 保证显式
    close 后释放，本 fixture 保证“从未 close”场景下也隔离。
    """
    return f"t_{abs(hash(request.node.nodeid)) % (10 ** 12):012d}"


def test_build_task_corpus_returns_none_when_rag_disabled(
    context_with_docs: Path,
    unique_task_id: str,
) -> None:
    """``cfg.enabled=False`` 时直接返回 None，不读目录。"""
    from data_agent_langchain.memory.rag.factory import build_task_corpus

    cfg = CorpusRagConfig(enabled=False)
    result = build_task_corpus(
        cfg,
        task_id=unique_task_id,
        task_input_dir=context_with_docs,
        embedder=DeterministicStubEmbedder(dim=16),
    )
    assert result is None


def test_build_task_corpus_returns_none_when_task_corpus_disabled(
    context_with_docs: Path,
    unique_task_id: str,
) -> None:
    """``cfg.task_corpus=False`` 时返回 None。"""
    from data_agent_langchain.memory.rag.factory import build_task_corpus

    cfg = CorpusRagConfig(enabled=True, task_corpus=False)
    result = build_task_corpus(
        cfg,
        task_id=unique_task_id,
        task_input_dir=context_with_docs,
        embedder=DeterministicStubEmbedder(dim=16),
    )
    assert result is None


def test_build_task_corpus_returns_none_when_dir_does_not_exist(
    tmp_path: Path,
    captured_events: list[tuple[str, dict[str, Any]]],
    unique_task_id: str,
) -> None:
    """目录不存在 / 为空 → 返回 None + dispatch ``no_documents``。"""
    from data_agent_langchain.memory.rag.factory import build_task_corpus

    cfg = CorpusRagConfig(enabled=True)
    result = build_task_corpus(
        cfg,
        task_id=unique_task_id,
        task_input_dir=tmp_path / "missing",
        embedder=DeterministicStubEmbedder(dim=16),
    )
    assert result is None
    skipped = [d for n, d in captured_events if n == "memory_rag_skipped"]
    assert any(d.get("reason") == "no_documents" for d in skipped), captured_events


def test_build_task_corpus_returns_handles_for_valid_dir(
    context_with_docs: Path,
    captured_events: list[tuple[str, dict[str, Any]]],
    unique_task_id: str,
) -> None:
    """有文档的目录 → 返回 ``TaskCorpusHandles``，含 store / retriever / embedder。

    需要 chromadb 已装；未装时 skip（等 ``pip install \".[rag]\"`` 后再跑）。
    """
    pytest.importorskip("chromadb")
    from data_agent_langchain.memory.rag.factory import (
        TaskCorpusHandles,
        build_task_corpus,
    )
    from data_agent_langchain.memory.rag.retrievers.vector import (
        VectorCorpusRetriever,
    )

    cfg = CorpusRagConfig(enabled=True, embedder_backend="stub")
    embedder = DeterministicStubEmbedder(dim=16)
    handles = build_task_corpus(
        cfg,
        task_id=unique_task_id,
        task_input_dir=context_with_docs,
        embedder=embedder,
    )

    assert isinstance(handles, TaskCorpusHandles)
    assert handles.embedder is embedder
    assert isinstance(handles.retriever, VectorCorpusRetriever)
    # store 必须满足 CorpusStore Protocol。
    from data_agent_langchain.memory.rag.base import CorpusStore

    assert isinstance(handles.store, CorpusStore)


def test_build_task_corpus_dispatches_index_built_event(
    context_with_docs: Path,
    captured_events: list[tuple[str, dict[str, Any]]],
    unique_task_id: str,
) -> None:
    """构建成功应 dispatch ``memory_rag_index_built``，payload 含必要字段。"""
    pytest.importorskip("chromadb")
    from data_agent_langchain.memory.rag.factory import build_task_corpus

    cfg = CorpusRagConfig(enabled=True)
    embedder = DeterministicStubEmbedder(dim=16)
    build_task_corpus(
        cfg,
        task_id=unique_task_id,
        task_input_dir=context_with_docs,
        embedder=embedder,
    )

    built = [d for n, d in captured_events if n == "memory_rag_index_built"]
    assert len(built) == 1, f"应有 1 个 index_built 事件，实际：{captured_events}"
    evt = built[0]
    assert evt["task_id"] == unique_task_id
    assert evt["doc_count"] == 3  # README.md + data_schema.md + notes.txt
    assert evt["chunk_count"] >= 3  # 至少每 doc 1 个 chunk
    assert evt["model_id"] == "stub-deterministic-dim16"
    assert evt["dimension"] == 16
    assert "elapsed_ms" in evt
    assert evt["elapsed_ms"] >= 0


def test_build_task_corpus_index_can_retrieve_indexed_chunks(
    context_with_docs: Path,
    unique_task_id: str,
) -> None:
    """构建出的 retriever 应能召回到刚索引的 chunk。"""
    pytest.importorskip("chromadb")
    from data_agent_langchain.memory.rag.factory import build_task_corpus

    cfg = CorpusRagConfig(enabled=True, retrieval_k=2)
    embedder = DeterministicStubEmbedder(dim=16)
    handles = build_task_corpus(
        cfg,
        task_id=unique_task_id,
        task_input_dir=context_with_docs,
        embedder=embedder,
    )
    assert handles is not None

    # 召回任意 query，应得到至少 1 条结果。
    results = handles.retriever.retrieve(
        "schema", namespace=f"corpus_task:{unique_task_id}"
    )
    assert len(results) >= 1
    for r in results:
        assert r.record.kind == "corpus"
        # ``source_path`` 应被 retriever 补回（来自 doc_index）。
        assert "source_path" in r.record.payload


def test_build_task_corpus_returns_none_when_timeout(
    context_with_docs: Path,
    captured_events: list[tuple[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    unique_task_id: str,
) -> None:
    """索引时间超过 ``task_corpus_index_timeout_s`` → 返回 None + dispatch
    ``memory_rag_skipped(reason="index_timeout")``。

    通过 monkeypatch 强制 ``perf_counter`` 在多次调用中跳跃，模拟超时。
    """
    # 模拟超时：让 perf_counter 第二次调用比第一次大很多。
    counter = iter([0.0, 100.0, 100.0])

    def _fake_perf() -> float:
        return next(counter, 100.0)

    monkeypatch.setattr(
        "data_agent_langchain.memory.rag.factory.perf_counter", _fake_perf
    )

    from data_agent_langchain.memory.rag.factory import build_task_corpus

    cfg = CorpusRagConfig(enabled=True, task_corpus_index_timeout_s=1.0)
    result = build_task_corpus(
        cfg,
        task_id=unique_task_id,
        task_input_dir=context_with_docs,
        embedder=DeterministicStubEmbedder(dim=16),
    )
    assert result is None
    skipped = [d for n, d in captured_events if n == "memory_rag_skipped"]
    assert any(d.get("reason") == "index_timeout" for d in skipped), captured_events


def test_build_task_corpus_shared_corpus_not_implemented(
    context_with_docs: Path,
    captured_events: list[tuple[str, dict[str, Any]]],
    unique_task_id: str,
) -> None:
    """``shared_corpus=True`` 时 dispatch ``shared_corpus_not_implemented``，
    task_corpus 路径仍正常构建（不阻塞）。"""
    pytest.importorskip("chromadb")
    from data_agent_langchain.memory.rag.factory import build_task_corpus

    cfg = CorpusRagConfig(
        enabled=True,
        task_corpus=True,
        shared_corpus=True,  # M4 未实现
    )
    handles = build_task_corpus(
        cfg,
        task_id=unique_task_id,
        task_input_dir=context_with_docs,
        embedder=DeterministicStubEmbedder(dim=16),
    )
    # task_corpus 仍应成功构建。
    assert handles is not None

    skipped = [d for n, d in captured_events if n == "memory_rag_skipped"]
    assert any(
        d.get("reason") == "shared_corpus_not_implemented" for d in skipped
    ), captured_events
