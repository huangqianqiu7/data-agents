from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

from langchain_core.language_models import FakeListChatModel

from data_agent_langchain.config import AgentConfig, AppConfig, DatasetConfig, ObservabilityConfig, RunConfig, ToolsConfig
from data_agent_langchain.run.runner import (
    _initial_state_for_task,
    create_run_output_dir,
    run_benchmark,
    run_single_task,
)

_LIST_CTX = '```json\n{"thought":"Inspect","action":"list_context","action_input":{"max_depth":4}}\n```'
_READ_CSV = '```json\n{"thought":"Read","action":"read_csv","action_input":{"path":"matches.csv","max_rows":5}}\n```'
_ANSWER = '```json\n{"thought":"Done","action":"answer","action_input":{"columns":["id"],"rows":[[1]]}}\n```'


def _make_task(root: Path, task_id: str) -> None:
    task_dir = root / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": task_id, "difficulty": "easy", "question": "Q?"}),
        encoding="utf-8",
    )
    with (context_dir / "matches.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["id"], [1]])


def _config(dataset_root: Path, output_dir: Path, run_id: str = "run1") -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=8, max_model_retries=1, model_retry_backoff=(0.0,), action_mode="json_action"),
        tools=ToolsConfig(),
        run=RunConfig(output_dir=output_dir, run_id=run_id, max_workers=1, task_timeout_seconds=0),
    )


def test_initial_state_contains_dataset_root_and_graph_defaults(tmp_path: Path):
    _make_task(tmp_path, "task_1")
    cfg = _config(tmp_path, tmp_path / "runs")
    from data_agent_common.benchmark.dataset import DABenchPublicDataset
    task = DABenchPublicDataset(tmp_path).get_task("task_1")
    state = _initial_state_for_task(task, cfg, mode="react")
    assert state["task_id"] == "task_1"
    assert state["dataset_root"] == str(tmp_path)
    assert state["context_dir"] == str(tmp_path / "task_1" / "context")
    assert state["mode"] == "react"
    assert state["action_mode"] == "json_action"
    assert state["max_steps"] == 8
    assert state["steps"] == []


def test_run_single_task_writes_trace_prediction_and_metrics(tmp_path: Path):
    _make_task(tmp_path, "task_1")
    cfg = _config(tmp_path, tmp_path / "runs")
    _, run_output_dir = create_run_output_dir(cfg.run.output_dir, run_id=cfg.run.run_id)
    fake_llm = FakeListChatModel(responses=[_LIST_CTX, _READ_CSV, _ANSWER])
    artifact = run_single_task(task_id="task_1", config=cfg, run_output_dir=run_output_dir, llm=fake_llm, graph_mode="react")
    assert artifact.succeeded is True
    assert artifact.trace_path.exists()
    assert artifact.prediction_csv_path is not None
    assert artifact.prediction_csv_path.exists()
    assert (artifact.task_output_dir / "metrics.json").exists()
    trace = json.loads(artifact.trace_path.read_text(encoding="utf-8"))
    assert trace["succeeded"] is True
    assert trace["answer"]["columns"] == ["id"]


def test_run_benchmark_writes_summary_with_metrics(tmp_path: Path):
    _make_task(tmp_path, "task_1")
    _make_task(tmp_path, "task_2")
    cfg = _config(tmp_path, tmp_path / "runs", run_id="batch1")
    fake_llm = FakeListChatModel(responses=[_LIST_CTX, _READ_CSV, _ANSWER, _LIST_CTX, _READ_CSV, _ANSWER])
    run_output_dir, artifacts = run_benchmark(config=cfg, llm=fake_llm, limit=2, graph_mode="react")
    assert len(artifacts) == 2
    summary = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "batch1"
    assert summary["task_count"] == 2
    assert summary["succeeded_task_count"] == 2
    assert summary["metrics"]["task_count"] == 2


def test_pyproject_exposes_dabench_lc_script():
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["scripts"]["dabench-lc"] == "data_agent_langchain.cli:main"


def test_run_single_task_core_injects_metrics_and_tracer_callbacks(tmp_path: Path, monkeypatch):
    from data_agent_common.benchmark.schema import AnswerTable
    import data_agent_langchain.run.runner as runner_module

    _make_task(tmp_path, "task_1")
    cfg = _config(tmp_path, tmp_path / "runs")
    seen = {}
    sentinel_callback = object()

    class FakeCompiledGraph:
        def invoke(self, state, config):
            seen["callbacks"] = config["callbacks"]
            return {
                **state,
                "answer": AnswerTable(columns=["id"], rows=[[1]]),
                "failure_reason": None,
            }

    monkeypatch.setattr(runner_module, "_build_compiled_graph", lambda mode: FakeCompiledGraph())
    monkeypatch.setattr(
        runner_module,
        "build_callbacks",
        lambda config, *, task_id, mode: [sentinel_callback],
        raising=False,
    )

    runner_module._run_single_task_core(
        task_id="task_1",
        config=cfg,
        task_output_dir=tmp_path / "out" / "task_1",
        graph_mode="react",
    )

    assert seen["callbacks"][0].__class__.__name__ == "MetricsCollector"
    assert seen["callbacks"][1] is sentinel_callback


def test_run_single_task_core_json_action_does_not_require_gateway_caps(tmp_path: Path):
    _make_task(tmp_path, "task_1")
    cfg = _config(tmp_path, tmp_path / "runs")
    fake_llm = FakeListChatModel(responses=[_LIST_CTX, _READ_CSV, _ANSWER])

    result = run_single_task(
        task_id="task_1",
        config=cfg,
        run_output_dir=tmp_path / "runs" / "run1",
        llm=fake_llm,
        graph_mode="react",
    )

    assert result.succeeded is True


def test_run_single_task_core_tool_calling_requires_gateway_caps(tmp_path: Path):
    import pytest
    from dataclasses import replace

    from data_agent_langchain.exceptions import GatewayCapsMissingError

    _make_task(tmp_path, "task_1")
    cfg = _config(tmp_path, tmp_path / "runs")
    cfg = replace(
        cfg,
        agent=replace(cfg.agent, action_mode="tool_calling"),
        observability=replace(cfg.observability, gateway_caps_path=tmp_path / "missing_gateway_caps.yaml"),
    )

    with pytest.raises(GatewayCapsMissingError):
        run_single_task(
            task_id="task_1",
            config=cfg,
            run_output_dir=tmp_path / "runs" / "run1",
            llm=FakeListChatModel(responses=[]),
            graph_mode="react",
        )


def test_run_single_task_core_tool_calling_binds_tools_with_caps(tmp_path: Path, monkeypatch):
    import yaml
    from dataclasses import replace

    import data_agent_langchain.run.runner as runner_module
    from data_agent_common.benchmark.schema import AnswerTable

    _make_task(tmp_path, "task_1")
    caps_path = tmp_path / "gateway_caps.yaml"
    caps_path.write_text(
        yaml.safe_dump({
            "gateway_caps": {
                "tool_calling": True,
                "parallel_tool_calls": False,
                "seed_param": False,
                "strict_mode": False,
            }
        }),
        encoding="utf-8",
    )
    cfg = _config(tmp_path, tmp_path / "runs")
    cfg = replace(
        cfg,
        agent=replace(cfg.agent, action_mode="tool_calling"),
        observability=replace(cfg.observability, gateway_caps_path=caps_path),
    )
    seen = {}

    class FakeCompiledGraph:
        def invoke(self, state, config):
            seen["llm"] = config["configurable"]["llm"]
            return {**state, "answer": AnswerTable(columns=["id"], rows=[[1]]), "failure_reason": None}

    class FakeToolCallingLLM:
        def bind_tools(self, tools, **kwargs):
            seen["tool_names"] = [tool.name for tool in tools]
            seen["bind_kwargs"] = kwargs
            return "bound-llm"

    monkeypatch.setattr(runner_module, "_build_compiled_graph", lambda mode: FakeCompiledGraph())

    runner_module._run_single_task_core(
        task_id="task_1",
        config=cfg,
        task_output_dir=tmp_path / "out" / "task_1",
        llm=FakeToolCallingLLM(),
        graph_mode="react",
    )

    assert seen["llm"] == "bound-llm"
    assert "answer" in seen["tool_names"]
    assert seen["bind_kwargs"] == {"parallel_tool_calls": False}


def test_run_single_task_default_tool_calling_requires_gateway_caps(tmp_path: Path):
    import pytest

    from data_agent_langchain.exceptions import GatewayCapsMissingError

    _make_task(tmp_path, "task_1")
    cfg = AppConfig(
        dataset=DatasetConfig(root_path=tmp_path),
        agent=AgentConfig(max_steps=8, max_model_retries=1, model_retry_backoff=(0.0,)),
        tools=ToolsConfig(),
        observability=ObservabilityConfig(gateway_caps_path=tmp_path / "missing_gateway_caps.yaml"),
        run=RunConfig(output_dir=tmp_path / "runs", run_id="run1", max_workers=1, task_timeout_seconds=0),
    )

    with pytest.raises(GatewayCapsMissingError):
        run_single_task(
            task_id="task_1",
            config=cfg,
            run_output_dir=tmp_path / "runs" / "run1",
            llm=FakeListChatModel(responses=[]),
            graph_mode="react",
        )




def test_run_single_task_core_sets_recursion_limit_from_step_budget(tmp_path: Path, monkeypatch):
    import data_agent_langchain.run.runner as runner_module
    from data_agent_common.benchmark.schema import AnswerTable

    _make_task(tmp_path, "task_1")
    cfg = AppConfig(
        dataset=DatasetConfig(root_path=tmp_path),
        agent=AgentConfig(max_steps=8, max_model_retries=1, model_retry_backoff=(0.0,), action_mode="json_action"),
        tools=ToolsConfig(),
        run=RunConfig(output_dir=tmp_path / "runs", run_id="run1", max_workers=1, task_timeout_seconds=0),
    )
    seen = {}

    class FakeCompiledGraph:
        def invoke(self, state, config):
            seen["recursion_limit"] = config["recursion_limit"]
            return {**state, "answer": AnswerTable(columns=["id"], rows=[[1]]), "failure_reason": None}

    monkeypatch.setattr(runner_module, "_build_compiled_graph", lambda mode: FakeCompiledGraph())

    runner_module._run_single_task_core(
        task_id="task_1",
        config=cfg,
        task_output_dir=tmp_path / "out" / "task_1",
        graph_mode="plan_solve",
    )

    assert seen["recursion_limit"] > 25
    assert seen["recursion_limit"] >= cfg.agent.max_steps * 5
