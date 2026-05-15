"""
LangGraph 后端的 task runner —— 单任务 / 批量任务执行入口。

模块职责（与 LANGCHAIN_MIGRATION_PLAN.md §13 / §15 / D1 / D4 / E13 对齐）：

  - :func:`run_single_task` —— 跑单个任务。可选用 ``multiprocessing`` 子
    进程隔离（``run.task_timeout_seconds > 0`` 时）；在子进程入口调用
    ``set_current_app_config`` 把配置注入 contextvar（D4 / §13.1）。
  - :func:`run_benchmark` —— 跑整套数据集任务，``run.max_workers > 1`` 时
    用 ThreadPoolExecutor 并发；最后聚合 ``metrics.json`` 为
    ``summary.json`` 写盘。
  - 工具 / 回调注入策略：``MetricsCollector`` 与 LangSmith
    ``LangChainTracer`` 都仅在 ``compiled.invoke`` 单点注入（D11 /
    §11.4.1），绝不挂在 ``ChatOpenAI`` 上。
  - 启动校验：``agent.action_mode == "tool_calling"`` 时强校验
    ``gateway_caps.yaml`` 存在且 ``tool_calling=true``，否则抛
    :class:`ConfigError`（v4 M3）。
"""
from __future__ import annotations

import csv
import json
import logging
import multiprocessing
import queue
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from data_agent_langchain.benchmark.dataset import DABenchPublicDataset
from data_agent_langchain.benchmark.schema import PublicTask
from data_agent_langchain.agents.finalize import build_run_result
from data_agent_langchain.agents.plan_solve_graph import build_plan_solve_graph
from data_agent_langchain.agents.react_graph import build_react_graph
from data_agent_langchain.config import AppConfig
from data_agent_langchain.exceptions import ConfigError
from data_agent_langchain.llm.factory import bind_tools_for_gateway
from data_agent_langchain.observability.events import (
    dispatch_observability_event,
    register_fallback_handler,
    unregister_fallback_handler,
)
from data_agent_langchain.observability.gateway_caps import GatewayCaps
from data_agent_langchain.observability.metrics import MetricsCollector
from data_agent_langchain.observability.reporter import aggregate_metrics
from data_agent_langchain.observability.tracer import build_callbacks
from data_agent_langchain.runtime.context import (
    clear_current_corpus_handles,
    set_current_app_config,
    set_current_corpus_handles,
)
from data_agent_langchain.runtime.rehydrate import build_runtime
from data_agent_langchain.runtime.state import RunState
from data_agent_langchain.tools.factory import create_all_tools

logger = logging.getLogger(__name__)

GraphMode = Literal["react", "plan_solve"]


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    """单任务跑完之后的产物清单。"""
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    """生成 UTC 时间戳形式的 run_id（如 ``20250507T154500Z``）。"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    """校验 / 生成 run_id：拒绝路径分隔符与空值；缺省自动生成时间戳。"""
    if run_id is None:
        return create_run_id()
    normalized = run_id.strip()
    if not normalized:
        raise ConfigError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ConfigError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    """在 *output_root* 下创建本次 run 的目录；返回 ``(run_id, dir)`` 对。"""
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def _initial_state_for_task(task: PublicTask, config: AppConfig, *, mode: GraphMode) -> RunState:
    """构造一个完整初始 ``RunState``，所有字段都是 picklable 基本类型。"""
    return {
        "task_id": task.task_id,
        "question": task.question,
        "difficulty": task.difficulty,
        "dataset_root": str(config.dataset.root_path),
        "task_dir": str(task.task_dir),
        "context_dir": str(task.context_dir),
        "mode": mode,
        "action_mode": config.agent.action_mode,
        "plan": [],
        "plan_index": 0,
        "replan_used": 0,
        "steps": [],
        "answer": None,
        "failure_reason": None,
        "discovery_done": False,
        "preview_done": False,
        "known_paths": [],
        "consecutive_gate_blocks": 0,
        "gate_decision": "pass",
        "skip_tool": False,
        "raw_response": "",
        "thought": "",
        "action": "",
        "action_input": {},
        "last_tool_ok": None,
        "last_tool_is_terminal": False,
        "last_error_kind": None,
        "subgraph_exit": "continue",
        "step_index": 0,
        "max_steps": config.agent.max_steps,
        "phase": "execution",
        "plan_progress": "",
        "plan_step_description": "",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """格式化写出 JSON（``ensure_ascii=False`` + 缩进 2，末尾留换行）。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    """写出 prediction.csv（UTF-8 + 标准 CSV 转义）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    """子进程超时 / 崩溃等场景下，给 task 写一份占位的失败结果。"""
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _build_compiled_graph(mode: GraphMode) -> Any:
    """根据 mode 选择 ReAct 或 Plan-and-Solve 图并编译。"""
    if mode == "plan_solve":
        return build_plan_solve_graph().compile()
    return build_react_graph().compile()


def _graph_recursion_limit(config: AppConfig, mode: GraphMode) -> int:
    """LangGraph ``recursion_limit`` 计算。

    内层子图每个 turn 至少 4-5 个节点，外加 plan_solve 的 replanner，
    给一个保守上界。最低 50 兜底，防止配置极小的 max_steps 让图无法
    完成最简单的循环。
    """
    max_steps = max(1, int(config.agent.max_steps))
    max_replans = max(0, int(config.agent.max_replans))
    limit = (max_steps * 6) + 20
    if mode == "plan_solve":
        limit += max_replans * 4
    return max(50, limit)


def _llm_for_action_mode(task: PublicTask, config: AppConfig, llm: Any | None) -> Any | None:
    """``tool_calling`` 模式下绑定工具到 LLM；其他模式直通。

    ``tool_calling`` 启动时强校验 gateway_caps.yaml 存在且
    ``tool_calling=true``（v4 M3）；不满足直接抛 :class:`ConfigError`。
    """
    if config.agent.action_mode != "tool_calling":
        return llm
    caps = GatewayCaps.from_yaml(config.observability.gateway_caps_path)
    if not caps.tool_calling:
        raise ConfigError(
            "agent.action_mode='tool_calling' requires gateway tool_calling support; "
            f"caps file reports tool_calling=false: {config.observability.gateway_caps_path}"
        )
    resolved_llm = llm
    if resolved_llm is None:
        from data_agent_langchain.llm.factory import build_chat_model

        resolved_llm = build_chat_model(config)
    runtime = build_runtime(task, config)
    tools = create_all_tools(task, runtime)
    return bind_tools_for_gateway(resolved_llm, tools, caps)


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    task_output_dir: Path,
    llm: Any | None = None,
    graph_mode: GraphMode = "plan_solve",
    show_progress: bool = False,
) -> dict[str, Any]:
    """核心执行流程：set contextvar → load task → 编译图 → invoke → build result。

    v3 corpus RAG（M4.4.4）：在 ``set_current_app_config`` 之后、
    ``compiled.invoke`` 之前构建 per-task corpus 索引并写入 contextvar；
    invoke 完成后无条件清理 contextvar 避免污染。
    """
    set_current_app_config(config)
    task = DABenchPublicDataset(config.dataset.root_path).get_task(task_id)

    # Bug 5 修复：``MetricsCollector`` 提前到 ``_build_and_set_corpus_handles``
    # 之前构造，并 register 到 events fallback。这样 corpus 构建阶段
    # （在 LangGraph runtime 之外，``dispatch_custom_event`` 抛 ``RuntimeError``）
    # dispatch 的 ``memory_rag_index_built`` / ``memory_rag_skipped`` 事件
    # 能经 fallback 路径送达 metrics，最终汇入 ``metrics.json.memory_rag``。
    metrics = MetricsCollector(task_id=task_id, output_dir=task_output_dir)
    register_fallback_handler(metrics.on_observability_event)
    try:
        try:
            # v3 corpus RAG（M4.4.4）：构建 per-task corpus handles。
            # 全部异常都被 fail-closed 吞掉：任何 RAG 失败都不应阻塞 task 主流程。
            _build_and_set_corpus_handles(config, task)

            resolved_llm = _llm_for_action_mode(task, config, llm)
            if show_progress:
                print(f"[dabench-lc] Loaded {task.task_id}; mode={graph_mode}; model={config.agent.model}", flush=True)
            compiled = _build_compiled_graph(graph_mode)
            callbacks = [metrics, *build_callbacks(config, task_id=task_id, mode=graph_mode)]
            runnable_config: dict[str, Any] = {
                "callbacks": callbacks,
                "recursion_limit": _graph_recursion_limit(config, graph_mode),
            }
            if resolved_llm is not None:
                runnable_config["configurable"] = {"llm": resolved_llm}
            final_state = compiled.invoke(
                _initial_state_for_task(task, config, mode=graph_mode),
                config=runnable_config,
            )
        finally:
            # 从 corpus handles 构建后到 invoke 完成之间，任何异常都要清理
            # contextvar，避免同进程路径污染后续 task。
            clear_current_corpus_handles()
    finally:
        # Bug 5 修复：无论成功 / 异常都 unregister fallback，避免跨 task 污染
        # （``run_benchmark`` 单进程并发时 register 列表是全局的）。
        unregister_fallback_handler(metrics.on_observability_event)
    return build_run_result(task_id, final_state).to_dict()


def _build_and_set_corpus_handles(config: AppConfig, task: Any) -> None:
    """为单个 task 构建 corpus handles 并写入 contextvar；fail-closed。

    守卫顺序（``CorpusRagConfig`` docstring 的 mode × rag 决策表）：

      1. ``memory.mode == "disabled"`` → 强制关闭 RAG，不构建索引（Bug 1）。
      2. ``memory.rag.enabled == False`` → 不构建索引。
      3. 任何异常 → fail-closed，contextvar 保持 ``None``，task 仍按 baseline
         路径运行（不抛、不阻塞）。
    """
    # Bug 1 守卫：``memory.mode=disabled`` 时无论 ``rag.enabled`` 如何都关闭 RAG，
    # 与 ``CorpusRagConfig`` docstring 的设计意图对齐。
    if config.memory.mode == "disabled":
        return
    if not config.memory.rag.enabled:
        return
    try:
        # 方法级延迟 import：避免在 ``rag.enabled=false`` 路径触发 factory
        # 链上的 chromadb / sentence-transformers import。
        from data_agent_langchain.memory.rag.factory import (
            build_embedder,
            build_task_corpus,
        )
    except Exception as exc:
        # Bug 2 修复：factory 链上的重依赖（chromadb / sentence-transformers /
        # rag 子包）import 失败时也必须落 trace，不再静默吞掉。
        dispatch_observability_event(
            "memory_rag_skipped",
            {
                "reason": "factory_import_failed",
                "task_id": str(task.task_id),
                "error": f"{type(exc).__name__}: {exc}",
            },
            config=None,
        )
        return

    try:
        embedder = build_embedder(config.memory.rag)
        if embedder is None:
            return
        handles = build_task_corpus(
            config.memory.rag,
            task_id=task.task_id,
            task_input_dir=task.context_dir,
            embedder=embedder,
        )
    except Exception as exc:
        # Bug 2 修复：兜底任何意外异常都落 trace。具体的 reason（如
        # embedder_load_failed / chroma_store_load_failed）由 factory 内部
        # dispatch；走到这里说明 factory 自己的 try/except 漏了某条路径，
        # 归类为 ``unexpected_error`` 兜底。
        dispatch_observability_event(
            "memory_rag_skipped",
            {
                "reason": "unexpected_error",
                "task_id": str(task.task_id),
                "error": f"{type(exc).__name__}: {exc}",
            },
            config=None,
        )
        return

    if handles is not None:
        set_current_corpus_handles(handles)


def _run_single_task_in_subprocess(
    task_id: str,
    config_payload: dict[str, Any],
    task_output_dir: Path,
    result_queue: multiprocessing.Queue[Any],
    graph_mode: GraphMode,
    show_progress: bool,
) -> None:
    """子进程入口函数：从纯 dict 重建 ``AppConfig`` 后调用 core。

    用 ``BaseException`` 兜底是因为子进程里 ``KeyboardInterrupt`` /
    ``SystemExit`` 不会自动传给父进程；我们需要把任何崩溃信息原封
    回传到父进程的 result_queue。
    """
    try:
        config = AppConfig.from_dict(config_payload)
        result_queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(
                    task_id=task_id,
                    config=config,
                    task_output_dir=task_output_dir,
                    graph_mode=graph_mode,
                    show_progress=show_progress,
                ),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        result_queue.put({"ok": False, "error": str(exc)})


def _reap_process(process: multiprocessing.Process, *, join_timeout: float = 5.0) -> None:
    """优雅回收子进程：先 join，再 terminate，最后 kill 兜底。"""
    process.join(timeout=join_timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()


def _run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
    task_output_dir: Path,
    graph_mode: GraphMode,
    show_progress: bool = False,
) -> dict[str, Any]:
    """带子进程隔离 + 任务级超时地跑一次任务。

    ``timeout_seconds <= 0`` 时直接 in-process 跑，避免子进程开销
    （主要供单测使用）。
    """
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(
            task_id=task_id,
            config=config,
            task_output_dir=task_output_dir,
            graph_mode=graph_mode,
            show_progress=show_progress,
        )
    result_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config.to_dict(), task_output_dir, result_queue, graph_mode, show_progress),
    )
    process.start()
    logger.debug("Spawned LangGraph subprocess PID=%s for task %s", process.pid, task_id)
    try:
        result: dict[str, Any] = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        logger.warning("Task %s timed out after %ds", task_id, timeout_seconds)
        _reap_process(process, join_timeout=1.0)
        return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")
    _reap_process(process)
    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")


def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    """落盘 trace.json / prediction.csv，并返回 :class:`TaskRunArtifacts`。"""
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)
    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )
    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    llm: Any | None = None,
    graph_mode: GraphMode = "plan_solve",
    show_progress: bool = False,
) -> TaskRunArtifacts:
    """跑单个任务并写出全部产物。

    显式传入 *llm*（如测试用 fake）时跳过子进程隔离 —— fake 对象不
    可 pickle，且单测希望同进程内更易断言。
    """
    started_at = perf_counter()
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    if llm is None:
        run_result = _run_single_task_with_timeout(
            task_id=task_id,
            config=config,
            task_output_dir=task_output_dir,
            graph_mode=graph_mode,
            show_progress=show_progress,
        )
    else:
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            task_output_dir=task_output_dir,
            llm=llm,
            graph_mode=graph_mode,
            show_progress=show_progress,
        )
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    llm: Any | None = None,
    limit: int | None = None,
    graph_mode: GraphMode = "plan_solve",
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    """跑整套数据集任务并写出 summary.json。

    *llm* 显式传入时强制 ``effective_workers=1``：fake LLM 不可在线程间
    安全共享。
    """
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)
    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]
    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if llm is not None:
        effective_workers = 1
    artifacts: list[TaskRunArtifacts] = []
    if effective_workers == 1:
        for task in tasks:
            artifact = run_single_task(
                task_id=task.task_id,
                config=config,
                run_output_dir=run_output_dir,
                llm=llm,
                graph_mode=graph_mode,
            )
            artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task.task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                    graph_mode=graph_mode,
                ): index
                for index, task in enumerate(tasks)
            }
            indexed: list[TaskRunArtifacts | None] = [None] * len(tasks)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            artifacts = [artifact for artifact in indexed if artifact is not None]
    metrics_summary = aggregate_metrics(run_output_dir)
    _write_json(
        run_output_dir / "summary.json",
        {
            "run_id": effective_run_id,
            "task_count": len(artifacts),
            "succeeded_task_count": sum(1 for artifact in artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in artifacts],
            "metrics": metrics_summary,
        },
    )
    return run_output_dir, artifacts


__all__ = [
    "GraphMode",
    "TaskRunArtifacts",
    "create_run_id",
    "create_run_output_dir",
    "resolve_run_id",
    "run_benchmark",
    "run_single_task",
]
