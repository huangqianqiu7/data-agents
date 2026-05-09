"""
Task runner — single-task and batch execution with timeout isolation.
任务运行器 —— 支持单任务和批量执行，并提供超时隔离机制。

Key fixes vs. the original:
  - Imports point to the correct ``agents`` package (was broken: ``agents.model``)
  - ``old_runner.py`` duplicate removed entirely
  - Full type annotations on all public functions
  - Logging added for subprocess lifecycle events

相比原始版本的主要修复：
  - 修正了导入路径，指向正确的 ``agents`` 包（原来的 ``agents.model`` 是错误的）
  - 完全移除了重复的 ``old_runner.py``
  - 为所有公开函数添加了完整的类型注解
  - 为子进程生命周期事件添加了日志记录
"""
from __future__ import annotations

import csv          # CSV 文件读写
import json         # JSON 序列化/反序列化
import logging      # 日志记录
import multiprocessing  # 多进程支持，用于子进程隔离执行任务
import queue        # 队列模块，用于捕获子进程超时异常 (queue.Empty)
from collections.abc import Callable  # 可调用对象类型注解
from concurrent.futures import ThreadPoolExecutor, as_completed  # 线程池，用于并行执行任务
from dataclasses import dataclass     # 数据类装饰器
from datetime import datetime, timezone  # 日期时间处理，用于生成运行 ID
from pathlib import Path              # 路径操作
from time import perf_counter         # 高精度计时器，用于统计执行耗时
from typing import Any                # 通用类型注解

from data_agent_refactored.agents.model import OpenAIModelAdapter          # OpenAI 模型适配器
from data_agent_refactored.agents.plan_solve_agent import PlanAndSolveAgent  # Plan-and-Solve 智能体
from data_agent_refactored.benchmark.dataset import DABenchPublicDataset     # DABench 公开数据集加载器
from data_agent_refactored.config import AppConfig, PlanAndSolveAgentConfig  # 应用配置 & 智能体配置
from data_agent_refactored.exceptions import InvalidRunIdError               # 自定义异常：无效的运行 ID
from data_agent_refactored.tools.registry import ToolRegistry, create_default_tool_registry  # 工具注册表

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure: per-task run artifacts
# 数据结构：单任务运行产物
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    """Files and metadata produced by running a single task.
    运行单个任务后产生的文件和元数据。

    使用 frozen=True 使实例不可变（安全、可哈希），
    使用 slots=True 减少内存占用。
    """
    task_id: str                        # 任务唯一标识符
    task_output_dir: Path               # 该任务的输出目录
    prediction_csv_path: Path | None    # 预测结果 CSV 文件路径（如果有的话）
    trace_path: Path                    # 执行轨迹 JSON 文件路径
    succeeded: bool                     # 任务是否成功完成
    failure_reason: str | None          # 失败原因（成功时为 None）

    def to_dict(self) -> dict[str, Any]:
        """将产物信息转换为字典格式，便于 JSON 序列化。"""
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": (
                str(self.prediction_csv_path) if self.prediction_csv_path else None
            ),
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


# ---------------------------------------------------------------------------
# Run-ID & output-dir helpers
# 运行 ID 与输出目录辅助函数
# ---------------------------------------------------------------------------

def create_run_id() -> str:
    """Generate a UTC-timestamped run ID like ``20260414T134153Z``.
    生成基于 UTC 时间戳的运行 ID，格式如 ``20260414T134153Z``。
    每次运行都会生成一个唯一的 ID，用于区分不同的运行批次。
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    """Validate and normalise a user-supplied run ID.
    验证并规范化用户提供的运行 ID。

    - 如果用户未提供 run_id（为 None），则自动生成一个新的时间戳 ID
    - 去除首尾空白后进行校验：不允许为空、不允许是路径（包含 / 或 \\）
    """
    if run_id is None:
        return create_run_id()  # 未提供则自动生成
    normalized = run_id.strip()
    if not normalized:
        raise InvalidRunIdError("run_id must not be empty.")
    # 防止用户传入路径而非单纯的目录名
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise InvalidRunIdError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(
    output_root: Path, *, run_id: str | None = None
) -> tuple[str, Path]:
    """Create and return ``(effective_run_id, run_output_dir)``.
    在 output_root 下创建以 run_id 命名的输出目录，并返回 (实际使用的 run_id, 输出目录路径)。

    注意：exist_ok=False 意味着如果目录已存在会抛出异常，确保每次运行的输出目录是全新的。
    """
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig) -> OpenAIModelAdapter:
    """Instantiate an :class:`OpenAIModelAdapter` from *config*.
    根据应用配置 config 实例化一个 OpenAI 模型适配器。

    从配置中提取模型名称、API 地址、API 密钥和温度参数。
    """
    return OpenAIModelAdapter(
        model=config.agent.model,           # 模型名称（如 gpt-4）
        api_base=config.agent.api_base,     # API 基础地址
        api_key=config.agent.api_key,       # API 密钥
        temperature=config.agent.temperature,  # 生成温度，控制输出随机性
    )


# ---------------------------------------------------------------------------
# File I/O helpers
# 文件读写辅助函数
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """将字典以美化格式写入 JSON 文件。
    ensure_ascii=False 保证中文等非 ASCII 字符能正常写入。
    """
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    """将列名和数据行写入 CSV 文件。
    会自动创建父目录（如果不存在的话）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)   # 写入表头
        for row in rows:
            writer.writerow(row)   # 逐行写入数据


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    """构造一个表示任务失败的标准结果字典。
    当任务超时或发生未捕获异常时使用此函数生成统一的失败结果。
    """
    return {
        "task_id": task_id,
        "answer": None,             # 失败时没有答案
        "steps": [],                # 失败时没有执行步骤
        "failure_reason": failure_reason,  # 失败原因描述
        "succeeded": False,         # 标记为失败
    }


# ---------------------------------------------------------------------------
# Core execution (in-process)
# 核心执行逻辑（在当前进程中运行）
# ---------------------------------------------------------------------------

def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model: OpenAIModelAdapter | None = None,
    tools: ToolRegistry | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Load a task and run the ReAct agent in the current process.
    加载指定任务并在当前进程中运行 Plan-and-Solve 智能体。

    参数:
        task_id: 要执行的任务 ID
        config: 应用全局配置
        model: 可选的模型适配器（不传则根据 config 自动创建）
        tools: 可选的工具注册表（不传则使用默认工具集）
        show_progress: 是否在控制台打印进度信息

    返回:
        包含执行结果的字典（answer、steps 等）
    """
    # 从数据集中加载指定任务
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    # 如果开启了进度显示，打印任务加载信息
    if show_progress:
        print(
            f"[dabench] Loaded {task.task_id} (difficulty={task.difficulty}); "
            f"model={config.agent.model!s}; max_steps={config.agent.max_steps}",
            flush=True,
        )

    # 创建 Plan-and-Solve 智能体：
    # - 如果调用方已提供 model/tools 则复用，否则新建
    agent = PlanAndSolveAgent(
        model=model or build_model_adapter(config),
        tools=tools or create_default_tool_registry(),
        config=PlanAndSolveAgentConfig(
            max_steps=config.agent.max_steps,
            progress=config.agent.progress,
        ),
    )
    # 执行任务并返回结果字典
    run_result = agent.run(task)
    return run_result.to_dict()


# ---------------------------------------------------------------------------
# Subprocess isolation
# 子进程隔离机制
# ---------------------------------------------------------------------------

def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    result_queue: multiprocessing.Queue[Any],
    show_progress: bool,
) -> None:
    """Worker target for the child process.
    子进程的工作函数。在独立进程中执行任务，通过队列将结果传回主进程。

    这样做的好处是：如果任务崩溃或超时，不会影响主进程的稳定性。
    """
    try:
        # 执行任务并将成功结果放入队列
        result_queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(
                    task_id=task_id, config=config, show_progress=show_progress
                ),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        # 捕获所有异常（包括 KeyboardInterrupt 等），将错误信息放入队列
        result_queue.put({"ok": False, "error": str(exc)})


def _reap_process(
    process: multiprocessing.Process, *, join_timeout: float = 5.0
) -> None:
    """Ensure the worker process is fully terminated (avoids zombies).
    确保子进程被完全终止，防止产生僵尸进程。

    采用逐步升级的终止策略：
    1. 先 join 等待进程自然结束
    2. 如果还活着，发送 SIGTERM (terminate)
    3. 如果仍然存活，发送 SIGKILL (kill) 强制杀死
    """
    process.join(timeout=join_timeout)
    if process.is_alive():
        process.terminate()        # 发送终止信号
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()         # 强制杀死
            process.join()


# ---------------------------------------------------------------------------
# Timeout-guarded single-task execution
# 带超时保护的单任务执行
# ---------------------------------------------------------------------------

def _run_single_task_with_timeout(
    *, task_id: str, config: AppConfig, show_progress: bool = False
) -> dict[str, Any]:
    """Run a task in a child process with a hard timeout.
    在子进程中运行任务，并设置硬超时限制。

    执行流程：
    1. 如果超时设为 <= 0，则直接在当前进程执行（不启用超时保护）
    2. 否则，启动一个子进程执行任务
    3. 主进程通过队列等待结果，超过超时时间则强制终止子进程
    """
    timeout_seconds = config.run.task_timeout_seconds
    # 超时 <= 0 表示不启用超时保护，直接在当前进程运行
    if timeout_seconds <= 0:
        return _run_single_task_core(
            task_id=task_id, config=config, show_progress=show_progress
        )

    # 创建进程间通信队列，用于接收子进程的执行结果
    result_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    # 启动子进程执行任务
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, result_queue, show_progress),
    )
    process.start()
    logger.debug("Spawned subprocess PID=%s for task %s", process.pid, task_id)

    try:
        # 等待子进程将结果放入队列，超时则抛出 queue.Empty
        result: dict[str, Any] = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        # 超时处理：记录警告日志，终止子进程，返回失败结果
        logger.warning("Task %s timed out after %ds — killing subprocess.", task_id, timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
        return _failure_run_result_payload(
            task_id, f"Task timed out after {timeout_seconds} seconds."
        )

    # 正常完成后，确保子进程被清理回收
    _reap_process(process)

    # 根据子进程返回的结果判断成功或失败
    if result.get("ok"):
        return dict(result["run_result"])  # 成功：返回执行结果
    return _failure_run_result_payload(
        task_id, f"Task failed with uncaught error: {result['error']}"  # 失败：包装错误信息
    )


# ---------------------------------------------------------------------------
# Write task outputs to disk
# 将任务输出写入磁盘
# ---------------------------------------------------------------------------

def _write_task_outputs(
    task_id: str, run_output_dir: Path, run_result: dict[str, Any]
) -> TaskRunArtifacts:
    """将单个任务的执行结果持久化到磁盘。

    输出结构：
        run_output_dir/
        └── {task_id}/
            ├── trace.json          # 完整的执行轨迹（始终生成）
            └── prediction.csv      # 预测结果表格（仅当 answer 是字典时生成）

    返回:
        TaskRunArtifacts 对象，包含所有输出文件路径和状态信息
    """
    # 为该任务创建专属输出目录
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # 始终保存执行轨迹 JSON 文件
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    # 如果 answer 是字典格式（包含 columns 和 rows），则额外输出 CSV 文件
    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),       # 列名列表
            [list(row) for row in answer.get("rows", [])],  # 数据行列表
        )

    # 返回封装好的任务产物对象
    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


# ---------------------------------------------------------------------------
# Public API: single task
# 公开接口：单任务执行
# ---------------------------------------------------------------------------

def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model: OpenAIModelAdapter | None = None,
    tools: ToolRegistry | None = None,
    show_progress: bool = False,
) -> TaskRunArtifacts:
    """Run a single task and persist outputs under *run_output_dir*.
    执行单个任务并将输出持久化到 run_output_dir 目录下。

    执行策略：
    - 如果未提供 model 和 tools（均为 None），使用子进程隔离 + 超时保护模式
    - 如果提供了自定义 model 或 tools，直接在当前进程执行（因为这些对象不可序列化到子进程）
    """
    started_at = perf_counter()  # 记录开始时间
    if model is None and tools is None:
        # 无自定义对象 → 使用子进程隔离（支持超时保护）
        run_result = _run_single_task_with_timeout(
            task_id=task_id, config=config, show_progress=show_progress
        )
    else:
        # 有自定义对象 → 直接在当前进程执行
        run_result = _run_single_task_core(
            task_id=task_id, config=config, model=model, tools=tools,
            show_progress=show_progress,
        )
    # 记录端到端耗时（秒），精确到毫秒
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    # 将结果写入磁盘并返回产物对象
    return _write_task_outputs(task_id, run_output_dir, run_result)


# ---------------------------------------------------------------------------
# Public API: full benchmark
# 公开接口：完整基准测试
# ---------------------------------------------------------------------------

def run_benchmark(
    *,
    config: AppConfig,
    model: OpenAIModelAdapter | None = None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    """Run the agent on the full dataset (or a limited subset).
    在完整数据集（或其子集）上运行智能体进行基准测试。

    参数:
        config: 应用全局配置
        model: 可选的自定义模型适配器
        tools: 可选的自定义工具注册表
        limit: 限制执行的任务数量（None 表示全部执行）
        progress_callback: 每完成一个任务后的回调函数（可用于进度条等）

    返回:
        (run_output_dir, task_artifacts) —— 输出目录路径和所有任务的产物列表
    """
    # 创建本次运行的输出目录
    effective_run_id, run_output_dir = create_run_output_dir(
        config.run.output_dir, run_id=config.run.run_id
    )

    # 加载数据集中的所有任务
    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]  # 按需截取子集

    # 确定并行工作线程数
    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    # 如果提供了自定义 model 或 tools，强制单线程（这些对象不适合跨线程共享）
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        # ===== 单线程串行执行 =====
        # 复用同一个 model 和 tools 实例，避免重复创建开销
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for tid in task_ids:
            artifact = run_single_task(
                task_id=tid,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            # 每完成一个任务，调用进度回调
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        # ===== 多线程并行执行 =====
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # 提交所有任务到线程池，并记录 future → 原始索引的映射
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=tid,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, tid in enumerate(task_ids)
            }
            # 预分配结果列表，保证最终结果与原始任务顺序一致
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            # 过滤掉 None（正常情况下不应出现）
            task_artifacts = [a for a in indexed_artifacts if a is not None]

    # 生成本次运行的汇总报告 summary.json
    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,                                        # 运行 ID
            "task_count": len(task_artifacts),                                 # 总任务数
            "succeeded_task_count": sum(1 for a in task_artifacts if a.succeeded),  # 成功任务数
            "max_workers": effective_workers,                                  # 使用的工作线程数
            "tasks": [a.to_dict() for a in task_artifacts],                   # 各任务详细产物
        },
    )
    return run_output_dir, task_artifacts
