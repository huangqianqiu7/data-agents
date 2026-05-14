from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from data_agent_langchain.config import default_app_config
from data_agent_langchain.cli import app
import data_agent_langchain.cli as cli_module


def test_run_benchmark_accepts_memory_mode_flag():
    result = CliRunner().invoke(app, ["run-benchmark", "--help"])

    assert result.exit_code == 0
    assert "--memory-mode" in result.output


def test_run_task_accepts_memory_mode_flag():
    result = CliRunner().invoke(app, ["run-task", "--help"])

    assert result.exit_code == 0
    assert "--memory-mode" in result.output


def test_run_task_memory_mode_flag_overrides_loaded_config(tmp_path: Path, monkeypatch):
    seen = {}

    monkeypatch.setattr(cli_module, "load_app_config", lambda config: default_app_config())
    monkeypatch.setattr(
        cli_module,
        "create_run_output_dir",
        lambda output_dir, *, run_id=None: ("run1", tmp_path / "run1"),
    )

    def fake_run_single_task(*, task_id, config, run_output_dir, graph_mode, show_progress):
        seen["memory_mode"] = config.memory.mode
        return SimpleNamespace(trace_path=tmp_path / "run1" / "task_1" / "trace.json")

    monkeypatch.setattr(cli_module, "run_single_task", fake_run_single_task)

    result = CliRunner().invoke(
        app,
        [
            "run-task",
            "task_1",
            "--config",
            str(tmp_path / "config.yaml"),
            "--memory-mode",
            "full",
        ],
    )

    assert result.exit_code == 0
    assert seen["memory_mode"] == "full"


def test_run_benchmark_memory_mode_flag_overrides_loaded_config(tmp_path: Path, monkeypatch):
    seen = {}

    monkeypatch.setattr(cli_module, "load_app_config", lambda config: default_app_config())

    def fake_run_benchmark(*, config, limit, graph_mode, progress_callback=None):
        seen["memory_mode"] = config.memory.mode
        return tmp_path / "run1", []

    monkeypatch.setattr(cli_module, "run_benchmark", fake_run_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "run-benchmark",
            "--config",
            str(tmp_path / "config.yaml"),
            "--memory-mode",
            "read_only_dataset",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0
    assert seen["memory_mode"] == "read_only_dataset"
