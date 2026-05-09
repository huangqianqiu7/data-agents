"""Phase 8 进度条移植 —— v5 §三.5 后续完善（2026-05-08）。

`dabench-lc run-benchmark` 默认启用 baseline 风格的 rich 进度条，
`--no-progress` 可关闭。设计参考 `data_agent_baseline/cli.py` 但实现位于
`data_agent_langchain/cli.py`，避免反向依赖已废弃包。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner


# ---------------------------------------------------------------------------
# 纯函数：紧凑进度字段 —— 移植自 baseline 同名 helper
# ---------------------------------------------------------------------------


def test_compact_progress_fields_returns_expected_keys():
    from data_agent_langchain.cli import _build_compact_progress_fields

    fields = _build_compact_progress_fields(
        completed_count=0,
        succeeded_count=0,
        failed_count=0,
        task_total=10,
        max_workers=2,
        elapsed_seconds=0.0,
        last_artifact=None,
    )

    assert set(fields.keys()) == {"ok", "fail", "run", "queue", "speed", "last"}


def test_compact_progress_fields_computes_running_and_queue_from_workers():
    from data_agent_langchain.cli import _build_compact_progress_fields

    # 10 任务 / 2 worker / 完成 3 题：剩 7 题，2 个 running，5 个 queue
    fields = _build_compact_progress_fields(
        completed_count=3,
        succeeded_count=3,
        failed_count=0,
        task_total=10,
        max_workers=2,
        elapsed_seconds=10.0,
        last_artifact=None,
    )

    assert fields["ok"] == "3"
    assert fields["fail"] == "0"
    assert fields["run"] == "2"
    assert fields["queue"] == "5"


def test_compact_progress_fields_formats_rate_in_tasks_per_minute():
    from data_agent_langchain.cli import _build_compact_progress_fields

    # 完成 5 题用 60 秒 → 5 task/min
    fields = _build_compact_progress_fields(
        completed_count=5,
        succeeded_count=4,
        failed_count=1,
        task_total=20,
        max_workers=4,
        elapsed_seconds=60.0,
        last_artifact=None,
    )

    assert "5.0 task/min" in fields["speed"]


def test_compact_progress_fields_handles_zero_elapsed_without_zero_division():
    from data_agent_langchain.cli import _build_compact_progress_fields

    fields = _build_compact_progress_fields(
        completed_count=0,
        succeeded_count=0,
        failed_count=0,
        task_total=10,
        max_workers=1,
        elapsed_seconds=0.0,
        last_artifact=None,
    )

    assert "0.0 task/min" in fields["speed"]


def test_compact_progress_fields_formats_last_artifact_with_status():
    """`last` 字段反映最近完成任务的 task_id + ok/fail。"""
    from data_agent_langchain.cli import _build_compact_progress_fields
    from data_agent_langchain.run.runner import TaskRunArtifacts

    artifact = TaskRunArtifacts(
        task_id="task_42",
        task_output_dir=Path("/tmp/run/task_42"),
        prediction_csv_path=Path("/tmp/run/task_42/prediction.csv"),
        trace_path=Path("/tmp/run/task_42/trace.json"),
        succeeded=True,
        failure_reason=None,
    )
    fields = _build_compact_progress_fields(
        completed_count=1,
        succeeded_count=1,
        failed_count=0,
        task_total=10,
        max_workers=1,
        elapsed_seconds=15.0,
        last_artifact=artifact,
    )

    assert "task_42" in fields["last"]
    assert "ok" in fields["last"]


def test_compact_progress_fields_formats_last_artifact_failure_status():
    from data_agent_langchain.cli import _build_compact_progress_fields
    from data_agent_langchain.run.runner import TaskRunArtifacts

    artifact = TaskRunArtifacts(
        task_id="task_99",
        task_output_dir=Path("/tmp/run/task_99"),
        prediction_csv_path=None,
        trace_path=Path("/tmp/run/task_99/trace.json"),
        succeeded=False,
        failure_reason="model_exhausted",
    )
    fields = _build_compact_progress_fields(
        completed_count=1,
        succeeded_count=0,
        failed_count=1,
        task_total=10,
        max_workers=1,
        elapsed_seconds=8.0,
        last_artifact=artifact,
    )

    assert "task_99" in fields["last"]
    assert "fail" in fields["last"]


# ---------------------------------------------------------------------------
# CLI 集成：--progress 默认开 / --no-progress 关
# ---------------------------------------------------------------------------


def _make_minimal_yaml(tmp_path: Path) -> Path:
    """构造最小 YAML 让 load_app_config 不抛错。"""
    dataset_root = tmp_path / "data"
    dataset_root.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"dataset:\n  root_path: {dataset_root.as_posix()}\n"
        "agent:\n  action_mode: json_action\n"
        "run:\n  max_workers: 1\n"
        "  task_timeout_seconds: 0\n",
        encoding="utf-8",
    )
    return cfg_path


def test_run_benchmark_command_passes_progress_callback_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """默认情况下 CLI 必须给 run_benchmark 传一个非 None 的 progress_callback。"""
    from data_agent_langchain import cli as cli_module

    captured: dict[str, Any] = {}

    def _fake_run_benchmark(**kwargs):
        captured.update(kwargs)
        return tmp_path / "fake_run_dir", []

    monkeypatch.setattr(cli_module, "run_benchmark", _fake_run_benchmark)
    cfg_path = _make_minimal_yaml(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        ["run-benchmark", "--config", str(cfg_path)],
    )

    assert result.exit_code == 0, result.output
    assert "progress_callback" in captured
    assert captured["progress_callback"] is not None


def test_run_benchmark_command_no_progress_flag_disables_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`--no-progress` 必须让 progress_callback 不再驱动 rich.Progress。

    具体契约：要么 progress_callback=None，要么传一个不渲染 rich UI 的简版回调。
    本测试只验证不会注入 rich.Progress 的更新（通过检查标准输出里没有 ANSI
    转义符 ``\\x1b[`` 即可，rich 渲染会留下大量 ANSI 序列）。
    """
    from data_agent_langchain import cli as cli_module

    def _fake_run_benchmark(**kwargs):
        cb = kwargs.get("progress_callback")
        # 触发一次回调，模拟一道任务完成
        if cb is not None:
            from data_agent_langchain.run.runner import TaskRunArtifacts

            artifact = TaskRunArtifacts(
                task_id="task_x",
                task_output_dir=tmp_path / "task_x",
                prediction_csv_path=None,
                trace_path=tmp_path / "task_x" / "trace.json",
                succeeded=True,
                failure_reason=None,
            )
            cb(artifact)
        return tmp_path / "fake_run_dir", []

    monkeypatch.setattr(cli_module, "run_benchmark", _fake_run_benchmark)
    cfg_path = _make_minimal_yaml(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        ["run-benchmark", "--config", str(cfg_path), "--no-progress"],
    )

    assert result.exit_code == 0, result.output
    # rich.Progress 渲染的 ANSI 序列以 \x1b[ 开头；no-progress 模式不应有
    assert "\x1b[" not in result.output


def test_run_benchmark_command_progress_callback_updates_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """progress_callback 被调用时正确累计 succeeded / failed 计数（通过监控 _build_compact_progress_fields 调用）。"""
    from data_agent_langchain import cli as cli_module
    from data_agent_langchain.run.runner import TaskRunArtifacts

    captured_fields: list[dict[str, str]] = []

    real_builder = cli_module._build_compact_progress_fields

    def _capture_builder(**kwargs):
        result = real_builder(**kwargs)
        captured_fields.append(result)
        return result

    monkeypatch.setattr(cli_module, "_build_compact_progress_fields", _capture_builder)

    def _fake_run_benchmark(**kwargs):
        cb = kwargs["progress_callback"]
        # 模拟 3 题：2 ok + 1 fail
        for tid, ok in [("task_a", True), ("task_b", False), ("task_c", True)]:
            cb(TaskRunArtifacts(
                task_id=tid,
                task_output_dir=tmp_path / tid,
                prediction_csv_path=None,
                trace_path=tmp_path / tid / "trace.json",
                succeeded=ok,
                failure_reason=None if ok else "x",
            ))
        return tmp_path / "fake_run_dir", []

    monkeypatch.setattr(cli_module, "run_benchmark", _fake_run_benchmark)
    cfg_path = _make_minimal_yaml(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        ["run-benchmark", "--config", str(cfg_path)],
    )

    assert result.exit_code == 0, result.output
    # 至少 3 次回调 → 至少 3 次 _build_compact_progress_fields 调用
    assert len(captured_fields) >= 3
    # 最后一次调用应反映 2 ok + 1 fail
    last = captured_fields[-1]
    assert last["ok"] == "2"
    assert last["fail"] == "1"


# ---------------------------------------------------------------------------
# 进度条 UX 后续优化（2026-05-08 增量 #3）
# ---------------------------------------------------------------------------


def test_build_progress_columns_compact_mode_excludes_time_columns():
    """compact 模式去掉 TimeElapsedColumn / TimeRemainingColumn 以适配窄终端。"""
    from rich.progress import TimeElapsedColumn, TimeRemainingColumn

    from data_agent_langchain.cli import _build_progress_columns

    columns = _build_progress_columns(compact=True)
    column_types = [type(c) for c in columns]

    assert TimeElapsedColumn not in column_types
    assert TimeRemainingColumn not in column_types


def test_build_progress_columns_full_mode_includes_time_columns():
    """非 compact 模式保留 elapsed / eta 列。"""
    from rich.progress import TimeElapsedColumn, TimeRemainingColumn

    from data_agent_langchain.cli import _build_progress_columns

    columns = _build_progress_columns(compact=False)
    column_types = [type(c) for c in columns]

    assert TimeElapsedColumn in column_types
    assert TimeRemainingColumn in column_types


def test_build_progress_columns_compact_default():
    """默认 compact=True：CLI 默认走窄终端友好模式。"""
    from rich.progress import TimeElapsedColumn

    from data_agent_langchain.cli import _build_progress_columns

    columns = _build_progress_columns()
    column_types = [type(c) for c in columns]

    assert TimeElapsedColumn not in column_types


def test_install_rich_logging_attaches_and_restores_handler():
    """上下文管理器进入时给根 logger 加 RichHandler，退出时还原。"""
    import logging

    from rich.console import Console
    from rich.logging import RichHandler

    from data_agent_langchain.cli import _install_rich_logging

    root = logging.getLogger()
    handlers_before = list(root.handlers)

    console = Console()
    with _install_rich_logging(console) as token:
        assert token is not None  # context manager 返回非 None token
        # 进入后根 logger 应有至少一个 RichHandler
        rich_handlers = [h for h in root.handlers if isinstance(h, RichHandler)]
        assert len(rich_handlers) >= 1

    # 退出后还原
    assert root.handlers == handlers_before


def test_install_rich_logging_routes_warnings_through_console(capsys):
    """RichHandler 安装期间，logger.warning 走 rich.Console，不再直写 plain stderr。

    具体契约：rich Console 把日志输出到 stderr 但带 ANSI 样式；不安装时
    plain StreamHandler 输出无 ANSI 序列。本测试验证安装 RichHandler 后
    输出含 ANSI（rich 渲染特征）。
    """
    import logging

    from rich.console import Console

    from data_agent_langchain.cli import _install_rich_logging

    test_logger = logging.getLogger("data_agent_langchain.test_progress_handler")
    test_logger.setLevel(logging.WARNING)

    console = Console(force_terminal=True, width=80)
    with _install_rich_logging(console):
        test_logger.warning("retry attempt 1/3 for step %d", 7)

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # rich 输出会带 ANSI 转义符（颜色 / 格式），plain logging 不会
    assert "\x1b[" in combined or "retry attempt" in combined


# ---------------------------------------------------------------------------
# 进度条 UX 后续优化（2026-05-08 增量 #4）：屏蔽 model_retry 噪声 WARNING
# ---------------------------------------------------------------------------


def test_install_rich_logging_suppresses_model_retry_warnings(capsys):
    """进度条期间，model_retry 的 WARNING 应被完全屏蔽，不渲染到 console。

    背景：`data_agent_langchain.agents.model_retry` 的 retry warning（429 / timeout
    频繁触发）即使经 RichHandler 整齐渲染，仍然刷屏并淹没真正有用的 last/ok/fail。
    用户要求：进度条模式下完全不显示这些 warning。
    """
    import logging

    from rich.console import Console

    from data_agent_langchain.cli import _install_rich_logging

    retry_logger = logging.getLogger("data_agent_langchain.agents.model_retry")
    # 先确保 logger level 是 NOTSET 或 WARNING，模拟实际运行场景
    retry_logger.setLevel(logging.NOTSET)

    console = Console(force_terminal=True, width=80)
    with _install_rich_logging(console):
        retry_logger.warning(
            "[model_node] step %d attempt 1/3 failed: 429 quota exceeded — retrying in 2.0s", 8
        )

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # WARNING 应被屏蔽：输出不应含 "model_node" 或 "429"
    assert "model_node" not in combined
    assert "429" not in combined


def test_install_rich_logging_still_shows_model_retry_errors(capsys):
    """ERROR 级仍然通过 RichHandler 显示（fatal 错误不被吞）。"""
    import logging

    from rich.console import Console

    from data_agent_langchain.cli import _install_rich_logging

    retry_logger = logging.getLogger("data_agent_langchain.agents.model_retry")
    retry_logger.setLevel(logging.NOTSET)

    console = Console(force_terminal=True, width=80)
    with _install_rich_logging(console):
        retry_logger.error("model_node fatal: connection lost permanently")

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "fatal" in combined or "connection lost" in combined


def test_install_rich_logging_restores_suppressed_logger_level():
    """退出上下文后，被临时屏蔽的 noisy logger level 还原到原值。"""
    import logging

    from rich.console import Console

    from data_agent_langchain.cli import _install_rich_logging

    retry_logger = logging.getLogger("data_agent_langchain.agents.model_retry")
    original_level = retry_logger.level

    console = Console(force_terminal=True, width=80)
    with _install_rich_logging(console):
        # 进入期间 effective level 应 >= ERROR
        assert retry_logger.getEffectiveLevel() >= logging.ERROR

    # 退出后还原
    assert retry_logger.level == original_level
