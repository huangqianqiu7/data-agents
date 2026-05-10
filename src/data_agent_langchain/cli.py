"""``dabench-lc`` 命令行入口（typer 框架）。

提供三条子命令：
  - ``run-task``：对单个任务跑一次（ReAct 或 Plan-and-Solve）。
  - ``run-benchmark``：对整套任务跑一遍并写入 summary.json；
    默认启用 baseline 风格的 ``rich`` 进度条，``--no-progress`` 可关闭。
  - ``gateway-smoke``：跑 Phase 0.5 网关能力探针并写 gateway_caps.yaml。
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated, Iterator, TYPE_CHECKING

import typer

from data_agent_langchain.config import load_app_config
from data_agent_langchain.observability.gateway_smoke import run_gateway_smoke
from data_agent_langchain.run.runner import (
    TaskRunArtifacts,
    create_run_output_dir,
    run_benchmark,
    run_single_task,
)

if TYPE_CHECKING:  # pragma: no cover - 仅用于静态类型检查
    pass

app = typer.Typer(help="Run DABench tasks with the LangGraph backend.")


# ---------------------------------------------------------------------------
# 进度条帮助函数：移植自 ``data_agent_baseline.cli`` （2026-05-08 v5 后续）。
# rich 仅供开发态 CLI 使用；容器提交态走 ``submission.py`` 不介入。
# ---------------------------------------------------------------------------


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    """返回适配 rich Progress 字段插槽的 dict。"""
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }


def _build_progress_columns(compact: bool = True) -> list:
    """构造 rich.Progress 列。compact=True 去 elapsed / eta 适配窄终端。"""
    from rich.progress import (
        BarColumn,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    columns: list = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
    ]
    if not compact:
        columns.extend([
            TextColumn("[dim]| elapsed[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]| eta[/dim]"),
            TimeRemainingColumn(),
        ])
    columns.extend([
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ])
    return columns


_NOISY_LOGGERS_DURING_PROGRESS: tuple[str, ...] = (
    # 频繁触发的重试警告（429 / timeout / 模型异常），会刷屏淹没进度条。
    # 进度条期间临时静音到 ERROR 级；ERROR/CRITICAL 仍能透出。
    "data_agent_langchain.agents.model_retry",
)


@contextmanager
def _install_rich_logging(console) -> Iterator:
    """临时给根 logger 装 ``RichHandler``，并屏蔽 noisy logger 的 WARNING。

    设计目标：
    - 进度条期间日志通过 rich.Live 整齐渲染，不打断进度条。
    - 高频重试 WARNING（如 ``model_retry`` 的 429/timeout）完全屏蔽，避免刷屏。
    - ERROR / CRITICAL 仍能透传（fatal 不被吞）。
    - 退出上下文严格还原原 logger 状态（root handlers/level + noisy logger level）。
    """
    from rich.logging import RichHandler

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    # 抑制 noisy logger：保存原 level，临时拉到 ERROR
    original_noisy_levels: dict[str, int] = {}
    for name in _NOISY_LOGGERS_DURING_PROGRESS:
        noisy = logging.getLogger(name)
        original_noisy_levels[name] = noisy.level
        noisy.setLevel(logging.ERROR)

    handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=False,
        rich_tracebacks=False,
    )
    handler.setLevel(logging.WARNING)
    root.handlers = [handler]
    if root.level == logging.NOTSET or root.level > logging.WARNING:
        root.setLevel(logging.WARNING)

    try:
        yield handler
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)
        for name, level in original_noisy_levels.items():
            logging.getLogger(name).setLevel(level)


@app.command("run-task")
def run_task_command(
    task_id: Annotated[str, typer.Argument(help="Task id such as task_1.")],
    config: Annotated[Path, typer.Option("--config", "-c", help="YAML config path.")],
    graph_mode: Annotated[str, typer.Option("--graph-mode", help="react or plan_solve.")] = "plan_solve",
) -> None:
    """对单个任务跑一次 LangGraph 后端。"""
    cfg = load_app_config(config)
    _, run_output_dir = create_run_output_dir(cfg.run.output_dir, run_id=cfg.run.run_id)
    artifact = run_single_task(
        task_id=task_id,
        config=cfg,
        run_output_dir=run_output_dir,
        graph_mode=graph_mode,  # type: ignore[arg-type]
        show_progress=cfg.agent.progress,
    )
    typer.echo(f"trace: {artifact.trace_path}")


@app.command("run-benchmark")
def run_benchmark_command(
    config: Annotated[Path, typer.Option("--config", "-c", help="YAML config path.")],
    limit: Annotated[int | None, typer.Option("--limit", help="Limit number of tasks.")] = None,
    graph_mode: Annotated[str, typer.Option("--graph-mode", help="react or plan_solve.")] = "plan_solve",
    progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="是否显示 baseline 风格的进度条（默认开）。",
        ),
    ] = True,
) -> None:
    """对整套任务跑一遍并产出批量 summary.json。默认进度条可用 ``--no-progress`` 关闭。"""
    cfg = load_app_config(config)

    # 估算 task_total：仅用于进度条；load_app_config 后这里能访问 dataset_root。
    if progress:
        from data_agent_langchain.benchmark.dataset import DABenchPublicDataset

        dataset = DABenchPublicDataset(cfg.dataset.root_path)
        task_total = len(dataset.iter_tasks())
        if limit is not None:
            task_total = min(task_total, limit)
        run_output_dir, artifacts = _run_benchmark_with_progress_bar(
            cfg=cfg,
            limit=limit,
            graph_mode=graph_mode,  # type: ignore[arg-type]
            task_total=task_total,
        )
    else:
        run_output_dir, artifacts = run_benchmark(
            config=cfg,
            limit=limit,
            graph_mode=graph_mode,  # type: ignore[arg-type]
        )

    typer.echo(f"run: {run_output_dir}")
    typer.echo(f"tasks: {len(artifacts)}")
    typer.echo(f"succeeded: {sum(1 for a in artifacts if a.succeeded)}")


def _run_benchmark_with_progress_bar(
    *,
    cfg,
    limit: int | None,
    graph_mode: str,
    task_total: int,
) -> tuple[Path, list[TaskRunArtifacts]]:
    """包装 ``run_benchmark`` 加 baseline 风格的 rich 进度条。

    rich 未安装时优雅降级到 ``typer.echo`` 逐题输出。
    """
    try:
        from rich.console import Console
        from rich.progress import Progress
    except ImportError:  # pragma: no cover - dataagent dev 环境总带 rich
        return _run_benchmark_with_text_progress(
            cfg=cfg, limit=limit, graph_mode=graph_mode, task_total=task_total
        )

    console = Console()
    effective_workers = cfg.run.max_workers
    progress_columns = _build_progress_columns(compact=True)

    completion_count = 0
    succeeded_count = 0
    failed_count = 0
    start_time = perf_counter()

    with _install_rich_logging(console), Progress(*progress_columns, console=console) as progress_bar:
        progress_task_id = progress_bar.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        def on_task_complete(artifact: TaskRunArtifacts) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress_bar.update(
                progress_task_id,
                completed=completion_count,
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        run_output_dir, artifacts = run_benchmark(
            config=cfg,
            limit=limit,
            graph_mode=graph_mode,  # type: ignore[arg-type]
            progress_callback=on_task_complete,
        )

    return run_output_dir, artifacts


def _run_benchmark_with_text_progress(
    *,
    cfg,
    limit: int | None,
    graph_mode: str,
    task_total: int,
) -> tuple[Path, list[TaskRunArtifacts]]:
    """rich 不可用时的降级路径：逐题打一行 「[N/M] task_x ok/fail」。"""
    completed = 0
    succeeded = 0
    failed = 0

    def on_task_complete(artifact: TaskRunArtifacts) -> None:
        nonlocal completed, succeeded, failed
        completed += 1
        if artifact.succeeded:
            succeeded += 1
        else:
            failed += 1
        status = "ok" if artifact.succeeded else "fail"
        typer.echo(
            f"[{completed}/{task_total}] {artifact.task_id} {status} "
            f"(ok={succeeded} fail={failed})"
        )

    return run_benchmark(
        config=cfg,
        limit=limit,
        graph_mode=graph_mode,  # type: ignore[arg-type]
        progress_callback=on_task_complete,
    )


@app.command("gateway-smoke")
def gateway_smoke_command(
    config: Annotated[Path, typer.Option("--config", "-c", help="YAML config path.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="gateway_caps.yaml output path."),
    ] = None,
) -> None:
    """跑 Phase 0.5 网关能力探针并写 gateway_caps.yaml。"""
    cfg = load_app_config(config)
    caps = run_gateway_smoke(cfg, output_path=output)
    written_path = output or cfg.observability.gateway_caps_path
    typer.echo(f"gateway_caps: {written_path}")
    typer.echo(f"tool_calling: {caps.tool_calling}")
    typer.echo(f"parallel_tool_calls: {caps.parallel_tool_calls}")
    typer.echo(f"seed_param: {caps.seed_param}")
    typer.echo(f"strict_mode: {caps.strict_mode}")
    if caps.tool_calling:
        typer.echo("default action_mode: tool_calling (gateway supports it)")
    else:
        typer.echo(
            "default action_mode: tool_calling, but gateway lacks tool_calling; "
            "set agent.action_mode=json_action in YAML to use legacy fallback"
        )


def main() -> None:
    # dev 便利：自动从 .env 加载 MODEL_* / SERPAPI_API_KEY 等环境变量。
    # 容器路径走 submission.py:main()，不经此函数；python-dotenv 不在 runtime
    # 依赖里，try-import 失败时静默跳过（§5.2 容器仅靠 env 协议不变）。
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]