from __future__ import annotations

from dataclasses import replace
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


def test_run_task_accepts_memory_rag_flag():
    result = CliRunner().invoke(app, ["run-task", "--help"])

    assert result.exit_code == 0
    assert "--memory-rag" in result.output
    assert "--no-memory-rag" in result.output


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


def test_run_task_memory_rag_flag_enables_loaded_config(tmp_path: Path, monkeypatch):
    seen = {}

    monkeypatch.setattr(cli_module, "load_app_config", lambda config: default_app_config())
    monkeypatch.setattr(
        cli_module,
        "create_run_output_dir",
        lambda output_dir, *, run_id=None: ("run1", tmp_path / "run1"),
    )

    def fake_run_single_task(*, task_id, config, run_output_dir, graph_mode, show_progress):
        seen["rag_enabled"] = config.memory.rag.enabled
        return SimpleNamespace(trace_path=tmp_path / "run1" / "task_1" / "trace.json")

    monkeypatch.setattr(cli_module, "run_single_task", fake_run_single_task)

    result = CliRunner().invoke(
        app,
        [
            "run-task",
            "task_1",
            "--config",
            str(tmp_path / "config.yaml"),
            "--memory-rag",
        ],
    )

    assert result.exit_code == 0
    assert seen["rag_enabled"] is True


def test_run_task_no_memory_rag_flag_disables_loaded_config(tmp_path: Path, monkeypatch):
    seen = {}
    cfg = default_app_config()
    cfg = replace(
        cfg,
        memory=replace(
            cfg.memory,
            rag=replace(cfg.memory.rag, enabled=True),
        ),
    )

    monkeypatch.setattr(cli_module, "load_app_config", lambda config: cfg)
    monkeypatch.setattr(
        cli_module,
        "create_run_output_dir",
        lambda output_dir, *, run_id=None: ("run1", tmp_path / "run1"),
    )

    def fake_run_single_task(*, task_id, config, run_output_dir, graph_mode, show_progress):
        seen["rag_enabled"] = config.memory.rag.enabled
        return SimpleNamespace(trace_path=tmp_path / "run1" / "task_1" / "trace.json")

    monkeypatch.setattr(cli_module, "run_single_task", fake_run_single_task)

    result = CliRunner().invoke(
        app,
        [
            "run-task",
            "task_1",
            "--config",
            str(tmp_path / "config.yaml"),
            "--no-memory-rag",
        ],
    )

    assert result.exit_code == 0
    assert seen["rag_enabled"] is False


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


def test_run_task_memory_rag_sets_hf_offline_defaults(tmp_path: Path, monkeypatch):
    captured = {}
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(cli_module, "load_app_config", lambda path: default_app_config())
    monkeypatch.setattr(cli_module, "create_run_output_dir", lambda output_dir, run_id=None: ("run", tmp_path))

    def fake_run_single_task(*, task_id, config, run_output_dir, graph_mode, show_progress):
        captured["hf_hub_offline"] = __import__("os").environ.get("HF_HUB_OFFLINE")
        captured["transformers_offline"] = __import__("os").environ.get("TRANSFORMERS_OFFLINE")
        return SimpleNamespace(trace_path=tmp_path / "trace.json")

    monkeypatch.setattr(cli_module, "run_single_task", fake_run_single_task)

    result = CliRunner().invoke(
        app,
        [
            "run-task",
            "task_1",
            "--config",
            str(tmp_path / "config.yaml"),
            "--memory-mode",
            "read_only_dataset",
            "--memory-rag",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "hf_hub_offline": "1",
        "transformers_offline": "1",
    }


def test_run_benchmark_memory_rag_preserves_explicit_hf_offline_env(tmp_path: Path, monkeypatch):
    captured = {}
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setattr(cli_module, "load_app_config", lambda path: default_app_config())

    def fake_run_benchmark(*, config, limit, graph_mode):
        captured["hf_hub_offline"] = __import__("os").environ.get("HF_HUB_OFFLINE")
        captured["transformers_offline"] = __import__("os").environ.get("TRANSFORMERS_OFFLINE")
        return tmp_path, []

    monkeypatch.setattr(cli_module, "run_benchmark", fake_run_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "run-benchmark",
            "--config",
            str(tmp_path / "config.yaml"),
            "--memory-mode",
            "read_only_dataset",
            "--memory-rag",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "hf_hub_offline": "0",
        "transformers_offline": "0",
    }


def test_run_benchmark_disabled_memory_does_not_set_hf_offline_env(tmp_path: Path, monkeypatch):
    captured = {}
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(cli_module, "load_app_config", lambda path: default_app_config())

    def fake_run_benchmark(*, config, limit, graph_mode):
        captured["hf_hub_offline"] = __import__("os").environ.get("HF_HUB_OFFLINE")
        captured["transformers_offline"] = __import__("os").environ.get("TRANSFORMERS_OFFLINE")
        return tmp_path, []

    monkeypatch.setattr(cli_module, "run_benchmark", fake_run_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "run-benchmark",
            "--config",
            str(tmp_path / "config.yaml"),
            "--memory-mode",
            "disabled",
            "--memory-rag",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "hf_hub_offline": None,
        "transformers_offline": None,
    }


def test_run_benchmark_memory_rag_sets_hf_offline_defaults(tmp_path: Path, monkeypatch):
    captured = {}
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(cli_module, "load_app_config", lambda path: default_app_config())

    def fake_run_benchmark(*, config, limit, graph_mode):
        captured["hf_hub_offline"] = __import__("os").environ.get("HF_HUB_OFFLINE")
        captured["transformers_offline"] = __import__("os").environ.get("TRANSFORMERS_OFFLINE")
        return tmp_path, []

    monkeypatch.setattr(cli_module, "run_benchmark", fake_run_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "run-benchmark",
            "--config",
            str(tmp_path / "config.yaml"),
            "--memory-mode",
            "read_only_dataset",
            "--memory-rag",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "hf_hub_offline": "1",
        "transformers_offline": "1",
    }

