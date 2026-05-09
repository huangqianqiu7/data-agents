"""KDD Cup 正式提交入口（v5 §二）。

容器内 ``ENTRYPOINT ["python", "-m", "data_agent_langchain.submission"]``
会执行 :func:`main`，按 v5 §四第 2 步的 13 步顺序完成：

  1. 校验 ``/logs`` 可写并初始化脱敏 ``logging.FileHandler``。
  2. 校验 ``/input`` 存在 + ``/output`` 可写。
  3. 主线程注册 SIGTERM handler。
  4. 清理 LangSmith 相关环境变量（防御性）。
  5. 从 ``MODEL_*`` env 构造 ``AppConfig``，其它字段走 ``config.py`` dataclass
     默认；固定 ``graph_mode="plan_solve"``。
  6. 校验 ``/app/gateway_caps.yaml`` 存在且 ``tool_calling=true``。
  7. 枚举 ``/input`` 下所有 ``task_*`` 目录。
  8. 每题先写占位 CSV ``result\\r\\n`` 到 ``/output/<task_id>/prediction.csv``。
  9. ThreadPoolExecutor 派发，每题 try/except 隔离。
  10. 成功 → 同目录 ``.prediction.csv.tmp`` + ``os.replace`` 覆盖；失败 → 保留占位。
  11. 失败追加一行到 ``/logs/failed_tasks.jsonl``（仅 task_id / 错误类型 /
      错误首行 / 耗时）。
  12. 派发循环每次提交前检查 ``_shutting_down``；收到 SIGTERM 即停。
  13. 全部任务结束后写 ``/logs/submission_summary.json``。

提交态严禁读取 YAML 配置或泄漏密钥；本模块不调用
``load_app_config``，不打印环境变量值，``/logs`` 仅写入脱敏摘要。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import yaml

from data_agent_langchain.benchmark.dataset import DABenchPublicDataset
from data_agent_langchain.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    ObservabilityConfig,
    RunConfig,
)
from data_agent_langchain.exceptions import DataAgentError
from data_agent_langchain.run.runner import TaskRunArtifacts, run_single_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 提交态固定常量（v5 §二.4 / §二.6）
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR: Path = Path("/input")
DEFAULT_OUTPUT_DIR: Path = Path("/output")
DEFAULT_LOGS_DIR: Path = Path("/logs")
DEFAULT_INTERNAL_RUNS_DIR: Path = Path("/tmp/dabench-runs")
DEFAULT_GATEWAY_CAPS_PATH: Path = Path("/app/gateway_caps.yaml")

EMPTY_API_KEY: str = "EMPTY"

PLACEHOLDER_CSV: bytes = b"result\r\n"
PREDICTION_FILE_NAME: str = "prediction.csv"
PREDICTION_TMP_NAME: str = ".prediction.csv.tmp"

SUBMISSION_GRAPH_MODE: Literal["plan_solve"] = "plan_solve"


# ---------------------------------------------------------------------------
# SIGTERM 主线程协作（v5 §二.5）
# ---------------------------------------------------------------------------

_shutting_down: threading.Event = threading.Event()


def _on_sigterm(signum: int, frame: Any) -> None:  # pragma: no cover - signal-driven
    """SIGTERM handler：只标志，不抛异常。"""
    logger.warning("Received SIGTERM (signum=%d); will stop dispatching new tasks.", signum)
    _shutting_down.set()


def _register_signal_handlers() -> None:
    """主线程注册 SIGTERM；Windows 下 SIGTERM 不会真触发但 register 不会抛。"""
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError, AttributeError) as exc:  # pragma: no cover
        # 子线程 register / 不支持的平台
        logger.warning("Failed to register SIGTERM handler: %s", type(exc).__name__)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class SubmissionConfigError(DataAgentError):
    """提交入口的配置 / 启动校验错误。

    继承 :class:`DataAgentError` 而不是 ``ConfigError``，避免与开发态
    ``load_app_config`` 的 ``ConfigError`` 混淆 —— 提交入口的失败模式
    （缺环境变量、缺 caps 文件等）需要单独的错误类来支持脱敏 logging。
    """


# ---------------------------------------------------------------------------
# §五. 配置构造（v5 §二.4）
# ---------------------------------------------------------------------------

def build_submission_config() -> AppConfig:
    """从 ``MODEL_*`` env 构造 ``AppConfig``，其它字段走 dataclass 默认，绝不读 YAML。

    必填 env（缺失任意一个都招 :class:`SubmissionConfigError`，错误信息
    只含 env 名，不透露任何 URL / key / model 值）：

      - ``MODEL_API_URL``
      - ``MODEL_NAME``。2026-05-09 v4 §4.1 D1：与 ``MODEL_API_URL`` 同级硬约束，
        避免容器静默 fallback 到 "qwen3.5-35b-a3b" 导致身份漂移。

    可选 env：
      - ``MODEL_API_KEY``：缺失回退 ``"EMPTY"``（网关 LLM 哨兵值）。

    容器特化路径（与本地 ``load_app_config`` 合法差异的 3 个字段）：
      - ``dataset.root_path = /input``
      - ``run.output_dir = /tmp/dabench-runs``（内部目录）
      - ``observability.gateway_caps_path = /app/gateway_caps.yaml``

    其它所有字段（``max_workers`` / ``task_timeout_seconds`` /
    ``action_mode`` / ``langsmith_enabled`` / ``reproducible`` / …）均走
    ``config.py`` dataclass 默认，本地与容器使用同一套参数。
    """
    api_url = os.environ.get("MODEL_API_URL")
    if not api_url:
        # 错误信息严格脱敏：只点名 env 名，不写任何字面值或回退提示
        raise SubmissionConfigError(
            "Missing required environment variable: MODEL_API_URL"
        )
    model_name = os.environ.get("MODEL_NAME")
    if not model_name:
        # 2026-05-09 v4 §4.1 D1：与 MODEL_API_URL 同级硬约束
        raise SubmissionConfigError(
            "Missing required environment variable: MODEL_NAME"
        )
    api_key = os.environ.get("MODEL_API_KEY") or EMPTY_API_KEY

    return AppConfig(
        dataset=DatasetConfig(root_path=DEFAULT_INPUT_DIR),
        agent=AgentConfig(
            model=model_name,
            api_base=api_url,
            api_key=api_key,
        ),
        run=RunConfig(output_dir=DEFAULT_INTERNAL_RUNS_DIR),
        observability=ObservabilityConfig(
            gateway_caps_path=DEFAULT_GATEWAY_CAPS_PATH,
        ),
    )


def _scrub_langsmith_env() -> None:
    """v5 §二.9：清理 LangSmith / LangChain tracing 相关环境变量，防御回归。"""
    for key in list(os.environ):
        if key.startswith("LANGSMITH_") or key in {
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_ENDPOINT",
            "LANGCHAIN_PROJECT",
        }:
            os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Gateway caps 启动校验（v5 §二.2）
# ---------------------------------------------------------------------------

def _verify_gateway_caps(path: Path) -> None:
    """要求文件存在且 ``tool_calling=true``，否则抛 SubmissionConfigError。"""
    if not path.exists():
        raise SubmissionConfigError(
            f"gateway_caps file missing: {path}. "
            "Container must ship /app/gateway_caps.yaml with tool_calling=true."
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SubmissionConfigError(
            f"gateway_caps file is not valid YAML: {path}"
        ) from exc
    caps = (payload or {}).get("gateway_caps") or {}
    if not bool(caps.get("tool_calling")):
        raise SubmissionConfigError(
            f"gateway_caps tool_calling must be true; got: {caps.get('tool_calling')!r}"
        )


# ---------------------------------------------------------------------------
# 占位 CSV / 原子覆盖（v5 §二.3）
# ---------------------------------------------------------------------------

def _write_placeholder_csv(target: Path) -> None:
    """写入 ``result\\r\\n`` 占位 CSV；存在时覆盖（同设备写）。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(PLACEHOLDER_CSV)


def _atomic_overwrite_with(source: Path, target: Path) -> None:
    """同目录临时文件 + ``os.replace`` 把 *source* 覆盖到 *target*。

    避免跨挂载点 ``EXDEV``：先在 target 同目录写 ``.prediction.csv.tmp``，
    再 ``os.replace`` 到 target。这一步是同设备原子重命名。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(PREDICTION_TMP_NAME)
    try:
        shutil.copy(source, tmp)
        os.replace(tmp, target)
    finally:
        # 异常路径时清理潜在残留临时文件
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 单任务执行（带异常隔离）
# ---------------------------------------------------------------------------

def _run_single_task_impl(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    llm: Any,
    graph_mode: Literal["plan_solve", "react"],
) -> TaskRunArtifacts:
    """单任务执行实现 —— 测试可 monkeypatch 此符号注入 fake artifact。"""
    return run_single_task(
        task_id=task_id,
        config=config,
        run_output_dir=run_output_dir,
        llm=llm,
        graph_mode=graph_mode,
    )


def _execute_one_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    output_dir: Path,
    logs_dir: Path,
    llm: Any,
    graph_mode: Literal["plan_solve", "react"],
) -> dict[str, Any]:
    """v5 §二.5 per-task try/except 包装。

    异常时只把 (task_id / 错误类型 / 错误首行 / 耗时) 写入
    ``failed_tasks.jsonl``，不写堆栈、observation、raw_response 或密钥。
    """
    started_at = perf_counter()
    placeholder_target = output_dir / task_id / PREDICTION_FILE_NAME
    record: dict[str, Any] = {
        "task_id": task_id,
        "succeeded": False,
        "elapsed_seconds": 0.0,
    }
    try:
        artifact = _run_single_task_impl(
            task_id=task_id,
            config=config,
            run_output_dir=run_output_dir,
            llm=llm,
            graph_mode=graph_mode,
        )
        record["succeeded"] = bool(artifact.succeeded)
        if artifact.prediction_csv_path is not None and artifact.prediction_csv_path.exists():
            _atomic_overwrite_with(artifact.prediction_csv_path, placeholder_target)
        # else: prediction_csv_path is None 或文件不存在 —— 保留占位 CSV，不动 target
    except BaseException as exc:  # noqa: BLE001 - 提交态严格隔离 + 全捕获
        first_line = str(exc).splitlines()[0] if str(exc) else ""
        elapsed = round(perf_counter() - started_at, 3)
        record.update(
            {
                "succeeded": False,
                "error_type": type(exc).__name__,
                "error_first_line": first_line,
                "elapsed_seconds": elapsed,
            }
        )
        _append_failed_task_record(logs_dir, record)
        logger.warning(
            "Task %s failed (%s); see /logs/failed_tasks.jsonl for redacted summary.",
            task_id,
            type(exc).__name__,
        )
        return record
    record["elapsed_seconds"] = round(perf_counter() - started_at, 3)
    if not record["succeeded"]:
        # 即使 run_single_task 没抛异常但内部判定失败，也要写 failed log
        record.setdefault("error_type", "TaskMarkedFailed")
        record.setdefault("error_first_line", "")
        _append_failed_task_record(logs_dir, record)
    return record


def _append_failed_task_record(logs_dir: Path, record: dict[str, Any]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_record = {
        "task_id": record.get("task_id", "?"),
        "error_type": record.get("error_type", ""),
        "error_first_line": record.get("error_first_line", ""),
        "elapsed_seconds": record.get("elapsed_seconds", 0.0),
    }
    line = json.dumps(safe_record, ensure_ascii=False) + "\n"
    with (logs_dir / "failed_tasks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line)


# ---------------------------------------------------------------------------
# 日志初始化（v5 §二.6）
# ---------------------------------------------------------------------------

def _setup_redacted_logging(logs_dir: Path) -> logging.Handler:
    """初始化 ``/logs/runtime.log`` FileHandler，root logger INFO。"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(logs_dir / "runtime.log", encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return handler


def _teardown_logging(handler: logging.Handler) -> None:
    handler.flush()
    handler.close()
    logging.getLogger().removeHandler(handler)


# ---------------------------------------------------------------------------
# §六. 顶层 ``run`` —— 测试 + main 共用
# ---------------------------------------------------------------------------

def run(
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    logs_dir: Path | None = None,
    internal_runs_dir: Path | None = None,
    gateway_caps_path: Path | None = None,
    register_signals: bool = True,
    llm: Any = None,
    max_workers: int | None = None,
    graph_mode: Literal["plan_solve", "react"] = SUBMISSION_GRAPH_MODE,
    action_mode_override: str | None = None,
) -> dict[str, Any]:
    """提交入口的可测试核心逻辑。

    所有路径默认指向容器内绝对路径；测试可注入 ``tmp_path`` 子目录。
    *register_signals* 默认 True；测试设 False 避免改主进程信号。

    *llm* / *action_mode_override* 仅供测试：传入 fake LLM 时
    ``run_single_task`` 走同进程，且 ``action_mode_override="json_action"``
    可绕过 tool_calling 的 LLM 绑定。生产路径都不传这两个。
    """
    input_dir = input_dir or DEFAULT_INPUT_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    logs_dir = logs_dir or DEFAULT_LOGS_DIR
    internal_runs_dir = internal_runs_dir or DEFAULT_INTERNAL_RUNS_DIR
    gateway_caps_path = gateway_caps_path or DEFAULT_GATEWAY_CAPS_PATH

    # 1. /logs 可写 + 脱敏 logging
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = _setup_redacted_logging(logs_dir)
    started_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    started_perf = perf_counter()

    try:
        # 2. /input 存在 + /output 可写
        if not input_dir.is_dir():
            raise SubmissionConfigError(f"Input dir not found: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 3. SIGTERM handler（主线程）
        if register_signals:
            _register_signal_handlers()

        # 4. 清理 LangSmith env
        _scrub_langsmith_env()

        # 5. 构造 AppConfig（生产路径），测试场景下覆盖关键路径
        config = build_submission_config()
        from dataclasses import replace  # 局部 import 避免污染顶层

        config = replace(
            config,
            dataset=replace(config.dataset, root_path=input_dir),
            run=replace(
                config.run,
                output_dir=internal_runs_dir,
                run_id="submission",
                max_workers=max_workers if max_workers is not None else config.run.max_workers,
                # 测试场景下走 in-process（FakeListChatModel 不可 pickle）
                task_timeout_seconds=0 if llm is not None else config.run.task_timeout_seconds,
            ),
            observability=replace(
                config.observability,
                gateway_caps_path=gateway_caps_path,
                langsmith_enabled=False,  # 双保险
            ),
            agent=replace(
                config.agent,
                action_mode=action_mode_override or config.agent.action_mode,
            ),
        )

        # 6. gateway_caps 启动校验
        _verify_gateway_caps(gateway_caps_path)

        # 7. 枚举 task_*
        dataset = DABenchPublicDataset(input_dir)
        task_ids = dataset.list_task_ids()
        logger.info("Discovered %d task(s) under input dir.", len(task_ids))

        # 8. 占位 CSV
        for task_id in task_ids:
            _write_placeholder_csv(output_dir / task_id / PREDICTION_FILE_NAME)

        # 内部 run_output_dir
        internal_runs_dir.mkdir(parents=True, exist_ok=True)
        run_output_dir = internal_runs_dir / "submission"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # 9-12. ThreadPoolExecutor 派发，per-task 异常隔离 + SIGTERM 中止
        effective_workers = max(1, config.run.max_workers)
        records: list[dict[str, Any]] = []

        def _submit_one(task_id: str) -> dict[str, Any]:
            return _execute_one_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                output_dir=output_dir,
                logs_dir=logs_dir,
                llm=llm,
                graph_mode=graph_mode,
            )

        if effective_workers == 1 or llm is not None:
            # 顺序执行：单 worker 或 fake llm（线程不安全）
            for task_id in task_ids:
                if _shutting_down.is_set():
                    logger.warning("Shutdown signaled; stopping dispatch.")
                    break
                records.append(_submit_one(task_id))
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures: list[tuple[str, Future[dict[str, Any]]]] = []
                for task_id in task_ids:
                    if _shutting_down.is_set():
                        logger.warning("Shutdown signaled; stopping dispatch.")
                        break
                    futures.append((task_id, executor.submit(_submit_one, task_id)))
                executor.shutdown(wait=True, cancel_futures=False)
                for _, future in futures:
                    try:
                        records.append(future.result())
                    except BaseException as exc:  # noqa: BLE001
                        logger.warning("Future raised unexpectedly: %s", type(exc).__name__)

        # 13. submission_summary.json
        finished_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        succeeded = sum(1 for r in records if r.get("succeeded"))
        summary = {
            "task_count": len(records),
            "succeeded_count": succeeded,
            "failed_count": len(records) - succeeded,
            "elapsed_seconds": round(perf_counter() - started_perf, 3),
            "started_at": started_iso,
            "finished_at": finished_iso,
            "max_workers": effective_workers,
            "task_timeout_seconds": config.run.task_timeout_seconds,
            "graph_mode": graph_mode,
        }
        (logs_dir / "submission_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Submission summary: %d/%d succeeded in %.1fs.",
            summary["succeeded_count"],
            summary["task_count"],
            summary["elapsed_seconds"],
        )
        return summary
    finally:
        _teardown_logging(handler)


# ---------------------------------------------------------------------------
# §七. main 入口（容器 ENTRYPOINT）
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """容器 ENTRYPOINT。argv 仅用于测试；生产路径不消费命令行参数。"""
    del argv  # 提交态不接受 CLI 参数；所有配置走环境变量
    try:
        run()
        return 0
    except SubmissionConfigError as exc:
        # 提交态启动失败：脱敏写一行到 stderr + runtime.log 已在 run() 内写
        sys.stderr.write(f"submission config error: {type(exc).__name__}: {exc}\n")
        return 2
    except BaseException as exc:  # noqa: BLE001
        sys.stderr.write(f"submission unexpected error: {type(exc).__name__}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))


__all__ = [
    "DEFAULT_GATEWAY_CAPS_PATH",
    "DEFAULT_INPUT_DIR",
    "DEFAULT_INTERNAL_RUNS_DIR",
    "DEFAULT_LOGS_DIR",
    "DEFAULT_OUTPUT_DIR",
    "EMPTY_API_KEY",
    "PLACEHOLDER_CSV",
    "PREDICTION_FILE_NAME",
    "PREDICTION_TMP_NAME",
    "SUBMISSION_GRAPH_MODE",
    "SubmissionConfigError",
    "build_submission_config",
    "main",
    "run",
]
