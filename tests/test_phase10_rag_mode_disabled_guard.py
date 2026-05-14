"""Bug 1 守卫测试：``memory.mode="disabled"`` 必须强制关闭 corpus RAG。

设计意图（``CorpusRagConfig`` docstring 已明确）：

  - ``memory.mode=disabled`` → 无论 ``rag.enabled`` 如何都强制关闭 RAG。
  - ``memory.mode=read_only_dataset`` 或 ``full`` 时 ``rag.enabled=true``
    才走 corpus 召回路径。

实测发现 v3.1 实施漏掉了三处守卫：

  1. ``run/runner.py:_build_and_set_corpus_handles`` 仅检查
     ``cfg.memory.rag.enabled``，无视 ``memory.mode``。
  2. ``agents/corpus_recall.py:recall_corpus_snippets`` 仅检查
     ``cfg.enabled``，无视 ``memory.mode``。
  3. ``agents/planner_node.py`` / ``agents/task_entry_node.py`` 直接传
     ``app_config.memory.rag``，丢失了 mode 信息。

本测试同时覆盖 runner 层与 recall helper 层的守卫；planner / task_entry 层
通过 helper 守卫间接被覆盖（避免重复测试 graph wiring）。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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
from data_agent_langchain.memory.base import MemoryRecord, RetrievalResult


# 复用 test_phase10_rag_runner.py 的 _run_runner_core helper（已 monkeypatch
# DABenchPublicDataset / 编译图 / LLM / metrics / build_run_result 等真依赖）。
from tests.test_phase10_rag_runner import _run_runner_core


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_task_with_context(tmp_path: Path) -> tuple[str, Path]:
    """构造一个含一篇 markdown 的 task ``context_dir``。"""
    ctx = tmp_path / "task_001" / "context"
    ctx.mkdir(parents=True)
    (ctx / "README.md").write_text("# Demo\nSchema columns A, B.\n", encoding="utf-8")
    return ("task_001", ctx)


# ---------------------------------------------------------------------------
# 场景 1：runner 层守卫 —— mode=disabled 时不应构建 corpus 索引
# ---------------------------------------------------------------------------


def test_runner_does_not_build_corpus_when_memory_mode_disabled(
    monkeypatch: pytest.MonkeyPatch,
    fake_task_with_context: tuple[str, Path],
) -> None:
    """mode=disabled + rag.enabled=True → invoke 内 contextvar 必须为 None。

    回归 Bug 1：runner._build_and_set_corpus_handles 当前只检查
    rag.enabled，不检查 memory.mode；与 CorpusRagConfig docstring 不符。
    """
    pytest.importorskip("chromadb")

    from data_agent_langchain.runtime.context import (
        clear_current_corpus_handles,
        get_current_corpus_handles,
    )

    clear_current_corpus_handles()
    seen_inside_invoke: list[Any] = []

    def _fake_invoke(state: Any, config: Any | None = None) -> Any:
        seen_inside_invoke.append(get_current_corpus_handles())
        return state

    fake_compiled = MagicMock()
    fake_compiled.invoke.side_effect = _fake_invoke

    cfg = AppConfig(
        dataset=DatasetConfig(root_path=fake_task_with_context[1].parent.parent),
        run=RunConfig(
            output_dir=fake_task_with_context[1].parent.parent / "runs",
            run_id="test_run",
            max_workers=1,
            task_timeout_seconds=0,  # 同步路径，便于 monkeypatch
        ),
        memory=MemoryConfig(
            # 关键：mode=disabled
            mode="disabled",
            # 但 rag.enabled=True；docstring 说应被 mode 覆盖。
            rag=CorpusRagConfig(enabled=True, embedder_backend="stub"),
        ),
    )
    _run_runner_core(monkeypatch, fake_compiled, cfg, fake_task_with_context[0])

    assert seen_inside_invoke == [None], (
        f"memory.mode='disabled' 应强制关闭 RAG，runner 不应构建 corpus 索引；"
        f"实际 invoke 内 contextvar={seen_inside_invoke}"
    )


# ---------------------------------------------------------------------------
# 场景 2：recall helper 守卫 —— mode=disabled 时不应调用 retriever
# ---------------------------------------------------------------------------


class _SpyRetriever:
    """记录 retrieve 调用的轻量 retriever，用于守卫测试。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        namespace: str,
        k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, namespace, k))
        return []


def _set_handles_with_spy() -> _SpyRetriever:
    """注入一个含 SpyRetriever 的 ``TaskCorpusHandles``，并返回 spy 引用。"""
    from data_agent_langchain.runtime.context import set_current_corpus_handles

    spy = _SpyRetriever()
    set_current_corpus_handles(
        SimpleNamespace(
            retriever=spy,
            embedder=SimpleNamespace(model_id="stub-deterministic-dim16"),
        )
    )
    return spy


def test_recall_corpus_snippets_returns_empty_when_memory_mode_disabled() -> None:
    """mode=disabled + rag.enabled=True + handles 已 set → returns [] 且不调 retriever。

    回归 Bug 1：recall_corpus_snippets 当前签名是 ``(cfg: CorpusRagConfig, ...)``，
    只检查 ``cfg.enabled``。Fix 之后改为 ``(memory_cfg: MemoryConfig, ...)``，
    内部检查 ``mode in {"read_only_dataset", "full"}``。
    """
    from data_agent_langchain.agents.corpus_recall import recall_corpus_snippets
    from data_agent_langchain.runtime.context import clear_current_corpus_handles

    clear_current_corpus_handles()
    spy = _set_handles_with_spy()

    memory_cfg = MemoryConfig(
        mode="disabled",
        rag=CorpusRagConfig(enabled=True, retrieval_k=3),
    )

    # 新签名：第一个位置参数是 memory_cfg（含 mode + rag）。
    hits = recall_corpus_snippets(
        memory_cfg,
        task_id="task_1",
        query="where is schema?",
        node="planner_node",
        config=None,
    )

    assert hits == []
    assert spy.calls == [], (
        f"mode=disabled 应阻止 retriever 调用；实际 retriever 收到："
        f"{spy.calls}"
    )

    clear_current_corpus_handles()


def test_recall_corpus_snippets_works_when_memory_mode_read_only_dataset() -> None:
    """mode=read_only_dataset + rag.enabled=True → 正常调用 retriever（baseline）。

    防止 Bug 1 修复矫枉过正把所有 mode 都拦死。
    """
    from data_agent_langchain.agents.corpus_recall import recall_corpus_snippets
    from data_agent_langchain.runtime.context import (
        clear_current_corpus_handles,
        set_current_corpus_handles,
    )

    clear_current_corpus_handles()

    class _OneHitRetriever:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, int | None]] = []

        def retrieve(
            self,
            query: str,
            *,
            namespace: str,
            k: int | None = None,
        ) -> list[RetrievalResult]:
            self.calls.append((query, namespace, k))
            return [
                RetrievalResult(
                    record=MemoryRecord(
                        id="doc-a#0000",
                        namespace=namespace,
                        kind="corpus",
                        payload={
                            "text": "schema columns A, B",
                            "doc_id": "doc-a",
                            "source_path": "README.md",
                            "doc_kind": "markdown",
                        },
                        metadata={},
                    ),
                    score=0.9,
                    reason="vector_cosine",
                ),
            ]

    retriever = _OneHitRetriever()
    set_current_corpus_handles(
        SimpleNamespace(
            retriever=retriever,
            embedder=SimpleNamespace(model_id="stub-deterministic-dim16"),
        )
    )

    memory_cfg = MemoryConfig(
        mode="read_only_dataset",
        rag=CorpusRagConfig(enabled=True, retrieval_k=3),
    )

    hits = recall_corpus_snippets(
        memory_cfg,
        task_id="task_1",
        query="where is schema?",
        node="planner_node",
        config=None,
    )

    assert len(retriever.calls) == 1, "read_only_dataset 模式应触发 retriever 调用"
    assert len(hits) == 1
    assert hits[0].record_id == "doc-a#0000"
    assert hits[0].summary.startswith("[markdown] README.md:")

    clear_current_corpus_handles()


def test_recall_corpus_snippets_returns_empty_when_memory_mode_full_but_rag_disabled() -> None:
    """mode=full + rag.enabled=False → returns []（rag 总开关守卫，未被 mode 覆盖）。

    防止守卫顺序错乱：mode 是必要条件而非充分条件。
    """
    from data_agent_langchain.agents.corpus_recall import recall_corpus_snippets
    from data_agent_langchain.runtime.context import clear_current_corpus_handles

    clear_current_corpus_handles()
    spy = _set_handles_with_spy()

    memory_cfg = MemoryConfig(
        mode="full",
        rag=CorpusRagConfig(enabled=False),
    )

    hits = recall_corpus_snippets(
        memory_cfg,
        task_id="task_1",
        query="where is schema?",
        node="planner_node",
        config=None,
    )

    assert hits == []
    assert spy.calls == [], "rag.enabled=False 仍应阻止 retriever 调用"

    clear_current_corpus_handles()
