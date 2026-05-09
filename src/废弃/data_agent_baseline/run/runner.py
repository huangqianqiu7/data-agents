from __future__ import annotations

import csv
import json
import multiprocessing
import queue
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry

# =====================================================================
# 数据结构：单次任务运行的产物清单
# =====================================================================
@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str                        # 任务ID
    task_output_dir: Path               # 该任务专属输出文件夹
    prediction_csv_path: Path | None    # 预测出的csv 文件路径（如果失败则为None）
    trace_path: Path                    # 思考录像的JSON 文件路径
    succeeded: bool                     # 是否成功
    failure_reason: str | None          # 失败原因

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }

# =====================================================================
# 辅助函数：生成运行批次号和输出目录
# =====================================================================
def create_run_id() -> str:
    # 默认用当前的 UTC 时间戳作为批次号，比如 20260414T134153Z
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    # 根据配置初始化大模型接口
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
    )

# =====================================================================
# 辅助函数：文件读写（已注入 UTF-8 魔法）
# =====================================================================
def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)

# 快速生成失败任务的假成绩单
def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }

# =====================================================================
# 核心做题逻辑：初始化 Agent 并下达指令
# =====================================================================
def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    # 加载数据集中的指定任务
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)
    
    # 打印进度日志（这就是你在终端里看到的 Loading... 提示）
    if progress:
        print(
            f"[dabench] Loaded {task.task_id} (difficulty={task.difficulty}); "
            f"model={config.agent.model!s}; max_steps={config.agent.max_steps}",
            flush=True,
        )

    # 召唤大模型！
    agent = ReActAgent(
        model=model or build_model_adapter(config),
        tools=tools or create_default_tool_registry(),
        config=ReActAgentConfig(
            max_steps=config.agent.max_steps,
            # progress=progress,
            # require_data_preview_before_compute=config.agent.require_data_preview_before_compute,
        ),
    )
    # 开始执行 ReAct 循环，返回 AgentRunResult
    run_result = agent.run(task)
    return run_result.to_dict()

# =====================================================================
# 多进程包装器：把大模型关进小黑屋
# =====================================================================
def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    queue: multiprocessing.Queue[Any],
    progress: bool,
) -> None:
    """
    这个函数会在一个全新的 Python 子进程中独立运行。
    无论里面发生了多严重的内存泄露、死循环报错，都不会影响外面的主进程。
    """
    try:
        # 做题成功，把成绩单塞进管道
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(task_id=task_id, config=config, progress=progress),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        # 如果代码崩溃，把崩溃信息塞进管道
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )

# 收尸程序：彻底清理用完的子进程，防止 Windows 满屏僵尸进程
def _reap_process(process: multiprocessing.Process, *, join_timeout: float = 5.0) -> None:
    """Ensure the worker exits after queue IPC (avoids zombie workers on Windows)."""
    process.join(timeout=join_timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()

# =====================================================================
# 超时控制官：拿着秒表在外面的主进程
# =====================================================================
def _run_single_task_with_timeout(*, task_id: str, config: AppConfig, progress: bool = False) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    # 如果配置里把超时时间设为 0，就不开子进程，直接在主进程裸跑
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config, progress=progress)

    # 创建一个用来沟通的管道 (Queue)
    result_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    # 开启一个子进程 (小黑屋) 让它在里面做题
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, result_queue, progress),
    )
    process.start()
    # Must read from the queue *before* join(): a large pickled payload can block the child's
    # queue.put() until a reader drains the pipe; join() would wait forever (deadlock).
    
    # 🌟 修复死锁的关键改动 🌟
    # 必须先用 queue.get() 并设置超时时间。
    # 这样如果子进程的数据太大塞满管道，主进程能立刻抽走数据，防止双方互相卡死。
    try:
        result = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        # 如果等了 timeout_seconds 秒还没动静，说明里面卡死了。
        # 不留情面，直接连人带屋子一起强杀！
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
        return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")

    # 如果顺利拿到了结果，派人去收尸清理内存
    _reap_process(process)

    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")

# 把成绩单落盘写进文件
def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    # 写出录像
    _write_json(trace_path, run_result)

    # 如果有答案，写出 prediction.csv
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

# 跑单个任务的入口函数（带有计时功能
def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
    progress: bool = False,
) -> TaskRunArtifacts:
    started_at = perf_counter() # 掐表开始
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config, progress=progress)
    else:
        run_result = _run_single_task_core(
            task_id=task_id, config=config, model=model, tools=tools, progress=progress
        )
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)   # 掐表结束，计算耗时
    return _write_task_outputs(task_id, run_output_dir, run_result)

# =====================================================================
# 流水线模式：跑完整个数据集
# =====================================================================
def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    # 创建类似 20260414T... 的跑分总文件夹
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        # 单线程模式，一题一题顺着做
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        # 多线程并发模式 (Thread Pool)
        # 用线程池把上面的多进程 (Process) 任务分发出去。
        # 比如 max_workers=4，就是同时有 4 个监考老师，盯着 4 个在小黑屋里做题的 Agent。
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            
            # 按顺序收集结果
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    # 全部跑完后，汇总生成终极大礼包：summary.json
    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
