from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from data_agent_langchain.observability.metrics import MetricsCollector
from data_agent_langchain.observability.reporter import aggregate_metrics


def test_metrics_collector_counts_custom_events_and_ignores_unknown(tmp_path: Path):
    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    run_id = uuid4()
    collector.on_custom_event("gate_block", {"step_index": 1}, run_id=run_id)
    collector.on_custom_event("replan_triggered", {"step_index": 2}, run_id=run_id)
    collector.on_custom_event("replan_failed", {"step_index": 3}, run_id=run_id)
    collector.on_custom_event("parse_error", {"step_index": 4}, run_id=run_id)
    collector.on_custom_event("model_error", {"step_index": 5}, run_id=run_id)
    collector.on_custom_event("memory_recall", {"hit_ids": ["a"]}, run_id=run_id)
    collector.on_custom_event("tool_call", {"tool": "read_json"}, run_id=run_id)
    collector.on_custom_event("unknown", {"x": 1}, run_id=run_id)
    collector.on_chain_end({}, run_id=run_id)
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["task_id"] == "task_1"
    assert payload["gate_blocks"] == 1
    assert payload["replan_count"] == 2
    assert payload["parse_errors"] == 1
    assert payload["model_errors"] == 1
    assert payload["memory_recalls"] == [{"hit_ids": ["a"]}]
    assert payload["tool_calls"] == {"read_json": 1}


def test_metrics_collector_writes_only_for_outermost_chain(tmp_path: Path):
    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    collector.on_chain_end({}, run_id=uuid4(), parent_run_id=uuid4())
    assert not (tmp_path / "metrics.json").exists()


def test_aggregate_metrics_summarizes_task_metrics(tmp_path: Path):
    for task_id, succeeded, wall_clock in [("task_1", True, 1.0), ("task_2", False, 3.0)]:
        task_dir = tmp_path / task_id
        task_dir.mkdir()
        (task_dir / "metrics.json").write_text(
            json.dumps({
                "task_id": task_id,
                "succeeded": succeeded,
                "tokens": {"prompt": 2, "completion": 3, "total": 5},
                "tool_calls": {"list_context": 1},
                "wall_clock_s": wall_clock,
                "gate_blocks": 1,
                "replan_count": 0,
                "parse_errors": 0,
                "model_errors": 0,
            }),
            encoding="utf-8",
        )
    summary = aggregate_metrics(tmp_path)
    assert summary["task_count"] == 2
    assert summary["succeeded"] == 1
    assert summary["tokens_total"] == {"prompt": 4, "completion": 6, "total": 10}
    assert summary["tool_calls_total"] == {"list_context": 2}
    assert summary["wall_clock"]["p50"] == 3.0
    assert summary["wall_clock"]["max"] == 3.0


def test_metrics_collector_counts_tool_name_from_serialized_output(tmp_path: Path):
    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    run_id = uuid4()
    collector.on_tool_end('{"tool_name":"read_csv","ok":true}', run_id=run_id)
    collector.on_chain_end({}, run_id=run_id)
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["tool_calls"] == {"read_csv": 1}


def test_tool_node_dispatches_tool_call_custom_event(tmp_path: Path):
    from langchain_core.runnables import RunnableLambda

    from data_agent_langchain.agents.tool_node import tool_node
    from data_agent_langchain.config import default_app_config
    from data_agent_langchain.runtime.context import set_current_app_config

    root = tmp_path
    task_dir = root / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_test", "difficulty": "easy", "question": "Q?"}),
        encoding="utf-8",
    )
    state = {
        "dataset_root": str(root),
        "task_id": "task_test",
        "step_index": 1,
        "thought": "",
        "raw_response": "",
        "phase": "execution",
        "plan_progress": "",
        "plan_step_description": "",
        "action": "list_context",
        "action_input": {"max_depth": 4},
        "skip_tool": False,
    }

    set_current_app_config(default_app_config())
    collector = MetricsCollector(task_id="task_test", output_dir=tmp_path / "out")

    def invoke_tool(_, config):
        return tool_node(state, config=config)

    RunnableLambda(invoke_tool).invoke({}, {"callbacks": [collector]})
    collector.on_chain_end({}, run_id=uuid4())
    payload = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["tool_calls"] == {"list_context": 1}
def test_gate_node_dispatches_gate_block_custom_event(tmp_path: Path):
    from data_agent_langchain.agents.gate import gate_node
    from data_agent_langchain.config import default_app_config
    from data_agent_langchain.runtime.context import set_current_app_config

    config = default_app_config()
    set_current_app_config(config)
    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    state = {
        "task_id": "task_1",
        "task_dir": str(tmp_path / "task_1"),
        "context_dir": str(tmp_path / "task_1" / "context"),
        "dataset_root": str(tmp_path),
        "question": "q",
        "difficulty": None,
        "mode": "plan_solve",
        "phase": "execution",
        "action_mode": "json_action",
        "steps": [],
        "step_index": 1,
        "max_steps": 3,
        "thought": "t",
        "action": "execute_python",
        "action_input": {"code": "print(1)"},
        "raw_response": "{}",
        "plan": [],
        "plan_index": 0,
        "replan_used": 0,
        "consecutive_gate_blocks": 0,
    }

    from langchain_core.runnables import RunnableLambda

    def invoke_gate(_, config):
        return gate_node(state, config=config)

    RunnableLambda(invoke_gate).invoke({}, {"callbacks": [collector]})
    collector.on_chain_end({}, run_id=uuid4())
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["gate_blocks"] == 1




def test_parse_action_node_dispatches_parse_error_custom_event(tmp_path: Path):
    from langchain_core.runnables import RunnableLambda

    from data_agent_langchain.agents.parse_action import parse_action_node

    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    state = {
        "raw_response": "not json",
        "action_mode": "json_action",
        "step_index": 2,
        "phase": "execution",
    }

    def invoke_parse(_, config):
        return parse_action_node(state, config=config)

    RunnableLambda(invoke_parse).invoke({}, {"callbacks": [collector]})
    collector.on_chain_end({}, run_id=uuid4())
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["parse_errors"] == 1


def test_model_node_dispatches_model_error_custom_event(tmp_path: Path):
    from dataclasses import replace

    from langchain_core.runnables import RunnableLambda

    from data_agent_langchain.agents.model_node import model_node
    from data_agent_langchain.config import default_app_config
    from data_agent_langchain.runtime.context import set_current_app_config

    app_config = default_app_config()
    fast_agent_config = replace(
        app_config.agent,
        max_model_retries=1,
        model_retry_backoff=(0.0,),
        model_timeout_s=1.0,
    )
    set_current_app_config(replace(app_config, agent=fast_agent_config))
    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    failing_llm = RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    state = {
        "task_id": "task_1",
        "task_dir": str(tmp_path / "task_1"),
        "context_dir": str(tmp_path / "task_1" / "context"),
        "dataset_root": str(tmp_path),
        "question": "q",
        "difficulty": None,
        "mode": "react",
        "phase": "execution",
        "action_mode": "json_action",
        "steps": [],
        "step_index": 0,
        "max_steps": 3,
    }

    def invoke_model(_, config):
        return model_node(state, config={**config, "configurable": {"llm": failing_llm}})

    RunnableLambda(invoke_model).invoke({}, {"callbacks": [collector]})
    collector.on_chain_end({}, run_id=uuid4())
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["model_errors"] == 1


def test_replanner_node_dispatches_replan_custom_events(tmp_path: Path, monkeypatch):
    import importlib

    from langchain_core.runnables import RunnableLambda

    planner_module = importlib.import_module("data_agent_langchain.agents.planner_node")

    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    state = {"steps": [], "replan_used": 0}
    monkeypatch.setattr(planner_module, "_generate_plan_from_state", lambda *args, **kwargs: ["next"])

    def invoke_replan(_, config):
        return planner_module.replanner_node(state, config=config)

    RunnableLambda(invoke_replan).invoke({}, {"callbacks": [collector]})
    collector.on_chain_end({}, run_id=uuid4())
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["replan_count"] == 1


def test_replanner_node_dispatches_replan_failed_custom_event(tmp_path: Path, monkeypatch):
    import importlib

    from langchain_core.runnables import RunnableLambda

    planner_module = importlib.import_module("data_agent_langchain.agents.planner_node")

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    collector = MetricsCollector(task_id="task_1", output_dir=tmp_path)
    state = {"steps": [], "replan_used": 0}
    monkeypatch.setattr(planner_module, "_generate_plan_from_state", fail)

    def invoke_replan(_, config):
        return planner_module.replanner_node(state, config=config)

    RunnableLambda(invoke_replan).invoke({}, {"callbacks": [collector]})
    collector.on_chain_end({}, run_id=uuid4())
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["replan_count"] == 1


def test_build_callbacks_returns_langsmith_tracer_when_enabled(monkeypatch):
    from dataclasses import replace

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    from data_agent_langchain.config import default_app_config
    from data_agent_langchain.observability.tracer import build_callbacks

    cfg = default_app_config()
    cfg = replace(
        cfg,
        observability=replace(cfg.observability, langsmith_enabled=True),
    )
    callbacks = build_callbacks(cfg, task_id="task_1", mode="react")
    assert [callback.__class__.__name__ for callback in callbacks] == ["LangChainTracer"]




def test_build_callbacks_degrades_without_langsmith_api_key(monkeypatch):
    from dataclasses import replace

    from data_agent_langchain.config import default_app_config
    from data_agent_langchain.observability.tracer import build_callbacks

    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    cfg = default_app_config()
    cfg = replace(
        cfg,
        observability=replace(cfg.observability, langsmith_enabled=True),
    )
    assert build_callbacks(cfg, task_id="task_1", mode="react") == []



