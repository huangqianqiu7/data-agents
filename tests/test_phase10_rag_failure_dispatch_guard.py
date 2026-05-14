"""Bug 2 守卫测试：RAG 构建失败时必须 dispatch ``memory_rag_skipped`` 事件。

设计意图（``02-implementation-plan-v2.md`` M4.4.3 / M4.4.4）：失败 fail-closed
但 ``skipped`` 必须落 trace / metrics，不能静默降级。

实测发现两处入口缺 dispatch：

  1. ``factory.build_task_corpus`` 在 ``ChromaCorpusStore`` 构造 / ``upsert``
     失败时直接抛异常（runner 捕获后吞掉，事件丢失）。
  2. ``runner._build_and_set_corpus_handles`` 两个 ``except Exception`` 都直接
     ``return``，没 dispatch 任何事件，导致 metrics.json 不含 ``memory_rag``
     段，与 M4.6.1 聚合契约不符。

本测试覆盖两处 dispatch 缺失。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from data_agent_langchain.config import (
    AppConfig,
    CorpusRagConfig,
    DatasetConfig,
    MemoryConfig,
    RunConfig,
)


# 复用 test_phase10_rag_runner.py 的 _run_runner_core helper。
from tests.test_phase10_rag_runner import _run_runner_core


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_task_with_context(tmp_path: Path) -> tuple[str, Path]:
    """构造一个含一篇 markdown 的 task ``context_dir``。"""
    ctx = tmp_path / "task_001" / "context"
    ctx.mkdir(parents=True)
    (ctx / "README.md").write_text(
        "# Demo\nSchema columns A, B.\n", encoding="utf-8"
    )
    return ("task_001", ctx)


# ---------------------------------------------------------------------------
# Bug 2.1：factory.build_task_corpus 在 chroma store 构造失败时必须 dispatch
# ---------------------------------------------------------------------------


def test_build_task_corpus_dispatches_when_chroma_store_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
    fake_task_with_context: tuple[str, Path],
) -> None:
    """``ChromaCorpusStore.ephemeral`` 抛异常时，factory 应 dispatch
    ``memory_rag_skipped(reason="chroma_store_load_failed")`` 并返回 None。

    回归 Bug 2.1：factory.py 步骤 5+6（store 构造 + upsert）当前没包
    try/except，异常一路传到调用方，trace 无任何记录。
    """
    from data_agent_langchain.memory.rag import factory as rag_factory
    from data_agent_langchain.memory.rag.embedders.stub import (
        DeterministicStubEmbedder,
    )

    captured: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        rag_factory,
        "dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data)),
    )

    # 注入 fake chroma store 模块：让 ChromaCorpusStore.ephemeral 抛 RuntimeError。
    fake_chroma_module = MagicMock()
    fake_chroma_module.ChromaCorpusStore.ephemeral.side_effect = RuntimeError(
        "chroma boom"
    )
    monkeypatch.setitem(
        sys.modules,
        "data_agent_langchain.memory.rag.stores.chroma",
        fake_chroma_module,
    )

    result = rag_factory.build_task_corpus(
        CorpusRagConfig(enabled=True, embedder_backend="stub"),
        task_id="task_001",
        task_input_dir=fake_task_with_context[1],
        embedder=DeterministicStubEmbedder(dim=16),
    )

    assert result is None, "store 构造失败应 fail-closed 返回 None"

    skipped_events = [event for event in captured if event[0] == "memory_rag_skipped"]
    assert any(
        event[1].get("reason") == "chroma_store_load_failed"
        for event in skipped_events
    ), (
        f"store 构造失败应 dispatch reason='chroma_store_load_failed'；"
        f"实际 events: {captured}"
    )


def test_build_task_corpus_dispatches_when_upsert_fails(
    monkeypatch: pytest.MonkeyPatch,
    fake_task_with_context: tuple[str, Path],
) -> None:
    """``store.upsert_chunks`` 抛异常时同样应 dispatch
    ``memory_rag_skipped(reason="chroma_store_load_failed")``。
    """
    from data_agent_langchain.memory.rag import factory as rag_factory
    from data_agent_langchain.memory.rag.embedders.stub import (
        DeterministicStubEmbedder,
    )

    captured: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        rag_factory,
        "dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data)),
    )

    # 让 ephemeral 返回一个 upsert_chunks 抛异常的 store。
    fake_store = MagicMock()
    fake_store.upsert_chunks.side_effect = RuntimeError("upsert boom")
    fake_chroma_module = MagicMock()
    fake_chroma_module.ChromaCorpusStore.ephemeral.return_value = fake_store
    monkeypatch.setitem(
        sys.modules,
        "data_agent_langchain.memory.rag.stores.chroma",
        fake_chroma_module,
    )

    result = rag_factory.build_task_corpus(
        CorpusRagConfig(enabled=True, embedder_backend="stub"),
        task_id="task_001",
        task_input_dir=fake_task_with_context[1],
        embedder=DeterministicStubEmbedder(dim=16),
    )

    assert result is None
    skipped_events = [event for event in captured if event[0] == "memory_rag_skipped"]
    assert any(
        event[1].get("reason") == "chroma_store_load_failed"
        for event in skipped_events
    ), (
        f"upsert 失败应 dispatch reason='chroma_store_load_failed'；"
        f"实际 events: {captured}"
    )


# ---------------------------------------------------------------------------
# Bug 2.2：runner._build_and_set_corpus_handles 在两个 except 路径都必须 dispatch
# ---------------------------------------------------------------------------


def test_runner_dispatches_when_factory_import_fails(
    monkeypatch: pytest.MonkeyPatch,
    fake_task_with_context: tuple[str, Path],
) -> None:
    """factory 模块 import 失败时，runner 应 dispatch
    ``memory_rag_skipped(reason="factory_import_failed")`` 而不是静默 return。

    回归 Bug 2.2 第一个 except：runner.py 当前 ``except Exception: return`` 完全
    吞掉异常。
    """
    captured: list[tuple[str, dict[str, Any]]] = []

    # 让 factory 模块被强制 ImportError：把 sys.modules 中已 cache 的 factory
    # 替换为一个 Loader 在 import 时抛错的对象，并 invalidate 已有引用。
    import importlib

    real_factory_module = sys.modules.get(
        "data_agent_langchain.memory.rag.factory"
    )
    monkeypatch.setitem(
        sys.modules,
        "data_agent_langchain.memory.rag.factory",
        None,  # ``None`` 让后续 ``import data_agent_langchain.memory.rag.factory``
        # 抛 ``ImportError`` —— Python 文档明确语义。
    )

    # spy dispatch：runner GREEN 后会从 observability.events import 该函数；
    # 这里同时 patch runner 模块级与 events 模块级，覆盖任意 import 写法。
    import data_agent_langchain.run.runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data)),
        raising=False,
    )

    fake_compiled = MagicMock()
    fake_compiled.invoke.return_value = {}

    cfg = AppConfig(
        dataset=DatasetConfig(root_path=fake_task_with_context[1].parent.parent),
        run=RunConfig(
            output_dir=fake_task_with_context[1].parent.parent / "runs",
            run_id="test_run",
            max_workers=1,
            task_timeout_seconds=0,
        ),
        memory=MemoryConfig(
            mode="read_only_dataset",
            rag=CorpusRagConfig(enabled=True, embedder_backend="stub"),
        ),
    )

    try:
        _run_runner_core(monkeypatch, fake_compiled, cfg, fake_task_with_context[0])
    finally:
        # 还原 factory 模块，避免污染后续测试。
        if real_factory_module is not None:
            sys.modules[
                "data_agent_langchain.memory.rag.factory"
            ] = real_factory_module
        else:
            sys.modules.pop(
                "data_agent_langchain.memory.rag.factory", None
            )
        importlib.invalidate_caches()

    skipped_events = [event for event in captured if event[0] == "memory_rag_skipped"]
    assert skipped_events, (
        f"factory import 失败时 runner 应 dispatch memory_rag_skipped；"
        f"实际 events: {captured}"
    )
    assert any(
        event[1].get("reason") == "factory_import_failed"
        for event in skipped_events
    ), f"应 dispatch reason='factory_import_failed'；实际：{skipped_events}"


def test_runner_dispatches_when_build_embedder_raises_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_task_with_context: tuple[str, Path],
) -> None:
    """``build_embedder`` 抛非 ImportError 异常时，runner 应 dispatch
    ``memory_rag_skipped(reason="unexpected_error")``。

    回归 Bug 2.2 第二个 except：runner.py:288-290 当前 ``except Exception: return``
    不 dispatch，metrics.json 完全没痕迹。
    """
    captured: list[tuple[str, dict[str, Any]]] = []

    import data_agent_langchain.memory.rag.factory as rag_factory
    import data_agent_langchain.run.runner as runner_module

    # build_embedder 抛非 ImportError，触发 runner 的兜底 except。
    def _boom_build_embedder(cfg, *, config=None):  # noqa: ANN001
        raise RuntimeError("embedder boom")

    monkeypatch.setattr(rag_factory, "build_embedder", _boom_build_embedder)

    monkeypatch.setattr(
        runner_module,
        "dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data)),
        raising=False,
    )

    fake_compiled = MagicMock()
    fake_compiled.invoke.return_value = {}

    cfg = AppConfig(
        dataset=DatasetConfig(root_path=fake_task_with_context[1].parent.parent),
        run=RunConfig(
            output_dir=fake_task_with_context[1].parent.parent / "runs",
            run_id="test_run",
            max_workers=1,
            task_timeout_seconds=0,
        ),
        memory=MemoryConfig(
            mode="read_only_dataset",
            rag=CorpusRagConfig(enabled=True, embedder_backend="stub"),
        ),
    )
    _run_runner_core(monkeypatch, fake_compiled, cfg, fake_task_with_context[0])

    skipped_events = [event for event in captured if event[0] == "memory_rag_skipped"]
    assert skipped_events, (
        f"build_embedder 抛 RuntimeError 时 runner 应 dispatch；"
        f"实际 events: {captured}"
    )
    assert any(
        event[1].get("reason") == "unexpected_error"
        for event in skipped_events
    ), f"应 dispatch reason='unexpected_error'；实际：{skipped_events}"
