"""``runtime.context`` corpus handles + runner 集成测试（M4.4.4）。

按 ``01-design-v2.md §4.11`` / ``§4.13`` 验证：

  - ``runtime/context.py`` 暴露 ``set_current_corpus_handles`` /
    ``get_current_corpus_handles`` / ``clear_current_corpus_handles`` 三个
    contextvar 操作函数。
  - ``get_current_corpus_handles()`` 在 contextvar 未设置时返回 ``None``
    （fail-closed；与 ``get_current_app_config`` 抛错的语义不同：corpus
    handles 是可选项，没设也不应崩）。
  - ``_run_single_task_core``（runner 子进程入口的同步路径）：
      * ``cfg.memory.rag.enabled=False`` 时 contextvar 仍为 None。
      * ``cfg.memory.rag.enabled=True`` 但 task ``context_dir`` 为空 → contextvar
        仍为 None；``memory_rag_skipped(reason="no_documents")`` 已 dispatch。
      * ``cfg.memory.rag.enabled=True`` + 含文档 + ``embedder_backend="stub"``
        → contextvar 在 invoke 期间被设为 ``TaskCorpusHandles``。

注：M4.4.4 仅验证 contextvar / runner wiring；真召回路径（``recall_corpus_snippets``）
在 M4.5.1 单独测试。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from data_agent_langchain.config import (
    AppConfig,
    CorpusRagConfig,
    DatasetConfig,
    MemoryConfig,
    RunConfig,
)


# ---------------------------------------------------------------------------
# contextvar API
# ---------------------------------------------------------------------------


def test_corpus_handles_contextvar_default_is_none() -> None:
    """``get_current_corpus_handles()`` 默认（未 set）返回 ``None``，不抛。"""
    from data_agent_langchain.runtime.context import (
        clear_current_corpus_handles,
        get_current_corpus_handles,
    )

    # 测试可能在其他测试 set 后跑；先清。
    clear_current_corpus_handles()
    assert get_current_corpus_handles() is None


def test_corpus_handles_contextvar_set_get_round_trip() -> None:
    """set 后 get 应返回相同对象。"""
    from data_agent_langchain.runtime.context import (
        clear_current_corpus_handles,
        get_current_corpus_handles,
        set_current_corpus_handles,
    )

    clear_current_corpus_handles()
    sentinel = object()
    set_current_corpus_handles(sentinel)
    assert get_current_corpus_handles() is sentinel
    clear_current_corpus_handles()
    assert get_current_corpus_handles() is None


def test_corpus_handles_contextvar_clear() -> None:
    """``clear_current_corpus_handles`` 应把 contextvar 置 ``None``。"""
    from data_agent_langchain.runtime.context import (
        clear_current_corpus_handles,
        get_current_corpus_handles,
        set_current_corpus_handles,
    )

    set_current_corpus_handles(object())
    clear_current_corpus_handles()
    assert get_current_corpus_handles() is None


# ---------------------------------------------------------------------------
# Runner 集成
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_task_with_context(tmp_path: Path) -> tuple[str, Path]:
    """返回一个 ``(task_id, task_input_dir)``，``context_dir`` 含 markdown。"""
    ctx = tmp_path / "task_001" / "context"
    ctx.mkdir(parents=True)
    (ctx / "README.md").write_text("# Demo\nThis dataset has columns A, B.\n", encoding="utf-8")
    (ctx / "data_schema.md").write_text("Schema: A int, B str.\n", encoding="utf-8")
    return ("task_001", ctx)


def test_runner_does_not_set_corpus_handles_when_rag_disabled(
    monkeypatch: pytest.MonkeyPatch,
    fake_task_with_context: tuple[str, Path],
) -> None:
    """``rag.enabled=False`` → runner 不构建 corpus，contextvar 在 invoke 内为 None。"""
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

    cfg = _make_test_config(
        task_input_dir=fake_task_with_context[1],
        rag_enabled=False,
    )
    _run_runner_core(monkeypatch, fake_compiled, cfg, fake_task_with_context[0])

    assert seen_inside_invoke == [None], (
        f"rag 关闭时 contextvar 应保持 None，实际：{seen_inside_invoke}"
    )


def test_runner_sets_corpus_handles_when_rag_enabled_with_documents(
    monkeypatch: pytest.MonkeyPatch,
    fake_task_with_context: tuple[str, Path],
) -> None:
    """``rag.enabled=True`` + stub backend + 含文档 → invoke 内 contextvar 非 None。

    需要 chromadb 可用（factory 内部用 ChromaCorpusStore）。未装时 skip。
    """
    pytest.importorskip("chromadb")

    from data_agent_langchain.memory.rag.factory import TaskCorpusHandles
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

    cfg = _make_test_config(
        task_input_dir=fake_task_with_context[1],
        rag_enabled=True,
        embedder_backend="stub",
    )
    _run_runner_core(monkeypatch, fake_compiled, cfg, fake_task_with_context[0])

    assert len(seen_inside_invoke) == 1
    handles = seen_inside_invoke[0]
    assert handles is not None, "rag 开启 + 含文档时应在 invoke 内拿到 handles"
    assert isinstance(handles, TaskCorpusHandles)


def test_runner_clears_corpus_handles_after_invoke(
    monkeypatch: pytest.MonkeyPatch,
    fake_task_with_context: tuple[str, Path],
) -> None:
    """invoke 完成后应清理 contextvar，避免污染同进程下后续 task。"""
    pytest.importorskip("chromadb")

    from data_agent_langchain.runtime.context import (
        clear_current_corpus_handles,
        get_current_corpus_handles,
    )

    clear_current_corpus_handles()
    fake_compiled = MagicMock()
    fake_compiled.invoke.return_value = {}

    cfg = _make_test_config(
        task_input_dir=fake_task_with_context[1],
        rag_enabled=True,
        embedder_backend="stub",
    )
    _run_runner_core(monkeypatch, fake_compiled, cfg, fake_task_with_context[0])

    # invoke 后 contextvar 必须为 None。
    assert get_current_corpus_handles() is None


def test_runner_clears_corpus_handles_when_setup_fails_before_invoke(
    monkeypatch: pytest.MonkeyPatch,
    fake_task_with_context: tuple[str, Path],
) -> None:
    """RAG handles 已构建但图执行前初始化失败时，也必须清理 contextvar。

    回归场景：``_build_and_set_corpus_handles`` 成功后，
    ``_llm_for_action_mode`` / gateway caps 初始化等步骤可能在
    ``compiled.invoke`` 之前抛错。此前清理逻辑只包住 invoke，导致同进程
    路径下后续 task 可能读到上一 task 的 corpus handles。
    """
    from data_agent_langchain.runtime.context import (
        clear_current_corpus_handles,
        get_current_corpus_handles,
        set_current_corpus_handles,
    )

    clear_current_corpus_handles()
    cfg = _make_test_config(
        task_input_dir=fake_task_with_context[1],
        rag_enabled=True,
        embedder_backend="stub",
    )

    # 直接复用 helper 的大部分 monkeypatch，但让 LLM 初始化在 invoke 前失败。
    fake_task = MagicMock()
    fake_task.task_id = fake_task_with_context[0]
    fake_task.context_dir = cfg.dataset.root_path / fake_task.task_id / "context"
    fake_dataset = MagicMock()
    fake_dataset.get_task.return_value = fake_task

    monkeypatch.setattr(
        "data_agent_langchain.run.runner.DABenchPublicDataset",
        lambda *a, **k: fake_dataset,
    )
    monkeypatch.setattr(
        "data_agent_langchain.run.runner._build_and_set_corpus_handles",
        lambda config, task: set_current_corpus_handles(object()),
    )
    monkeypatch.setattr(
        "data_agent_langchain.run.runner._llm_for_action_mode",
        MagicMock(side_effect=RuntimeError("gateway setup failed")),
    )
    monkeypatch.setattr(
        "data_agent_langchain.run.runner.MetricsCollector",
        lambda **kwargs: MagicMock(),
    )

    output_dir = cfg.run.output_dir / fake_task.task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    from data_agent_langchain.run.runner import _run_single_task_core

    with pytest.raises(RuntimeError, match="gateway setup failed"):
        _run_single_task_core(
            task_id=fake_task.task_id,
            config=cfg,
            task_output_dir=output_dir,
        )

    assert get_current_corpus_handles() is None


def test_runner_handles_empty_context_dir_gracefully(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``context_dir`` 为空（无文档）时 invoke 内 contextvar 应为 None，
    runner 仍正常完成（fail-closed），不抛错。"""
    pytest.importorskip("chromadb")

    from data_agent_langchain.runtime.context import (
        clear_current_corpus_handles,
        get_current_corpus_handles,
    )

    clear_current_corpus_handles()
    empty_ctx = tmp_path / "task_xxx" / "context"
    empty_ctx.mkdir(parents=True)

    seen_inside_invoke: list[Any] = []
    def _fake_invoke(state: Any, config: Any | None = None) -> Any:
        seen_inside_invoke.append(get_current_corpus_handles())
        return state

    fake_compiled = MagicMock()
    fake_compiled.invoke.side_effect = _fake_invoke

    cfg = _make_test_config(
        task_input_dir=empty_ctx,
        rag_enabled=True,
        embedder_backend="stub",
    )
    _run_runner_core(monkeypatch, fake_compiled, cfg, "task_xxx")

    assert seen_inside_invoke == [None], (
        f"无文档时 contextvar 应为 None（fail-closed）：{seen_inside_invoke}"
    )


# ---------------------------------------------------------------------------
# 工具：构造测试 cfg 与跑 runner core
# ---------------------------------------------------------------------------


def _make_test_config(
    *,
    task_input_dir: Path,
    rag_enabled: bool,
    embedder_backend: str = "stub",
) -> AppConfig:
    """构造一个最小可运行的 ``AppConfig``。"""
    return AppConfig(
        dataset=DatasetConfig(root_path=task_input_dir.parent.parent),
        run=RunConfig(
            output_dir=task_input_dir.parent.parent / "runs",
            run_id="test_run",
            max_workers=1,
            task_timeout_seconds=0,  # 同步路径
        ),
        memory=MemoryConfig(
            mode="read_only_dataset" if rag_enabled else "disabled",
            rag=CorpusRagConfig(
                enabled=rag_enabled,
                embedder_backend=embedder_backend,  # type: ignore[arg-type]
            ),
        ),
    )


def _run_runner_core(
    monkeypatch: pytest.MonkeyPatch,
    fake_compiled: MagicMock,
    cfg: AppConfig,
    task_id: str,
) -> None:
    """直接调用 runner 的 ``_run_single_task_core``，但替换两处真依赖：

      1. ``DABenchPublicDataset.get_task`` 返回 mock task（``context_dir``
         指向 fake fixture）。
      2. ``_build_compiled_graph`` 返回 ``fake_compiled``。
    """
    # 1) DABenchPublicDataset
    fake_task = MagicMock()
    fake_task.task_id = task_id
    fake_task.context_dir = (
        cfg.dataset.root_path / task_id / "context"
    )
    fake_dataset = MagicMock()
    fake_dataset.get_task.return_value = fake_task

    monkeypatch.setattr(
        "data_agent_langchain.run.runner.DABenchPublicDataset",
        lambda *a, **k: fake_dataset,
    )

    # 2) _build_compiled_graph
    monkeypatch.setattr(
        "data_agent_langchain.run.runner._build_compiled_graph",
        lambda mode: fake_compiled,
    )

    # 3) _llm_for_action_mode 不依赖 LLM gateway。
    monkeypatch.setattr(
        "data_agent_langchain.run.runner._llm_for_action_mode",
        lambda task, config, llm: None,
    )

    # 4) MetricsCollector / build_callbacks：不写盘。
    monkeypatch.setattr(
        "data_agent_langchain.run.runner.MetricsCollector",
        lambda **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "data_agent_langchain.run.runner.build_callbacks",
        lambda *a, **k: [],
    )

    # 5) build_run_result
    monkeypatch.setattr(
        "data_agent_langchain.run.runner.build_run_result",
        lambda task_id, state: MagicMock(to_dict=lambda: {}),
    )

    # 6) _initial_state_for_task
    monkeypatch.setattr(
        "data_agent_langchain.run.runner._initial_state_for_task",
        lambda task, cfg, mode: {},
    )

    output_dir = cfg.run.output_dir / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    from data_agent_langchain.run.runner import _run_single_task_core

    _run_single_task_core(
        task_id=task_id,
        config=cfg,
        task_output_dir=output_dir,
    )
