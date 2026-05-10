from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from data_agent_langchain.agents.planner_node import planner_node
from data_agent_langchain.config import AgentConfig, AppConfig
from data_agent_langchain.runtime.context import _APP_CONFIG


@pytest.fixture()
def synthetic_task(tmp_path: Path) -> dict:
    root = tmp_path
    task_dir = root / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_test", "difficulty": "easy", "question": "Q?"}),
        encoding="utf-8",
    )
    with (context_dir / "x.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["a"], [1]])
    return {"dataset_root": str(root), "task_id": "task_test"}


@pytest.fixture()
def fast_app_config():
    cfg = AppConfig(agent=AgentConfig(model_timeout_s=5.0))
    token = _APP_CONFIG.set(cfg)
    try:
        yield cfg
    finally:
        _APP_CONFIG.reset(token)


def _base_state(task_state: dict, **extra) -> dict:
    state = {
        **task_state,
        "mode": "plan_solve",
        "action_mode": "json_action",
        "steps": [],
        "plan_index": 0,
        "replan_used": 0,
        "step_index": 0,
        "max_steps": 20,
    }
    state.update(extra)
    return state


def test_planner_node_records_successful_plan(synthetic_task, fast_app_config):
    llm = FakeListChatModel(responses=['```json\n{"plan":["List files","Read data"]}\n```'])
    update = planner_node(_base_state(synthetic_task), config={"configurable": {"llm": llm}})
    assert update["plan"] == ["List files", "Read data"]
    assert update["plan_index"] == 0
    assert update["replan_used"] == 0
    [step] = update["steps"]
    assert step.step_index == 0
    assert step.action == "__plan_generation__"
    assert step.phase == "planning"
    assert step.ok is True
    assert step.observation == {"ok": True, "plan": ["List files", "Read data"]}


def test_planner_node_uses_fallback_plan_on_model_or_parse_failure(synthetic_task, fast_app_config):
    llm = RunnableLambda(lambda _: "not json")
    update = planner_node(_base_state(synthetic_task), config={"configurable": {"llm": llm}})
    assert update["plan"] == ["List context files", "Inspect data", "Solve and call answer"]
    [step] = update["steps"]
    assert step.action == "__plan_generation__"
    assert step.ok is False
    assert step.observation == {
        "ok": False,
        "plan": ["List context files", "Inspect data", "Solve and call answer"],
    }

from data_agent_common.agents.runtime import StepRecord
from data_agent_langchain.agents.planner_node import replanner_node


def test_replanner_node_builds_history_hint_and_resets_plan(synthetic_task, fast_app_config):
    captured = []

    def planner(messages):
        captured.append(messages)
        return '```json\n{"plan":["Use alternate data","Answer"]}\n```'

    state = _base_state(
        synthetic_task,
        steps=[
            StepRecord(
                step_index=1,
                thought="ok",
                action="read_csv",
                action_input={"path": "x.csv"},
                raw_response="{}",
                observation={"ok": True, "content": {"rows": 1}},
                ok=True,
                phase="execution",
            ),
            StepRecord(
                step_index=2,
                thought="fail",
                action="execute_python",
                action_input={"code": "raise RuntimeError()"},
                raw_response="{}",
                observation={"ok": False, "error": "boom"},
                ok=False,
                phase="execution",
            ),
        ],
        plan=["Old"],
        plan_index=0,
        replan_used=0,
    )
    update = replanner_node(state, config={"configurable": {"llm": RunnableLambda(planner)}})
    assert update["plan"] == ["Use alternate data", "Answer"]
    assert update["plan_index"] == 0
    assert update["replan_used"] == 1
    human_text = captured[0][1].content
    assert "Already completed (do NOT repeat):" in human_text
    assert "read_csv" in human_text
    assert "Last failed action:" in human_text
    assert "execute_python" in human_text
    assert "IMPORTANT: Your new plan must skip steps that have already succeeded." in human_text


def test_replanner_node_records_sentinel_when_replan_fails(synthetic_task, fast_app_config):
    state = _base_state(
        synthetic_task,
        steps=[],
        plan=["Keep current"],
        plan_index=0,
        replan_used=1,
    )
    update = replanner_node(state, config={"configurable": {"llm": RunnableLambda(lambda _: "not json")}})
    assert "plan" not in update
    assert "plan_index" not in update
    assert update["replan_used"] == 2
    [step] = update["steps"]
    assert step.step_index == -1
    assert step.action == "__replan_failed__"
    assert step.ok is False
    assert step.phase == "execution"
    assert "Re-plan failed:" in step.observation["error"]


# --- Callback propagation (token-usage bug regression) ----------------

class _ConfigCapturingLLM:
    """Captures the ``config=`` kwarg passed to ``invoke`` for assertions.

    See ``tests/test_phase3_model.py`` for the full bug context: the
    metrics.json tokens=0 bug stemmed from ``model_node`` and ``planner_node``
    calling ``llm.invoke(messages)`` without forwarding the LangGraph
    ``RunnableConfig``, breaking the ``MetricsCollector.on_llm_end`` chain.
    """

    def __init__(self, plan_response: str = '```json\n{"plan":["List","Solve"]}\n```') -> None:
        self.captured_config: Any | None = None
        self.captured_messages: Any | None = None
        self._plan_response = plan_response

    def invoke(self, messages, config=None, **_kwargs):
        self.captured_config = config
        self.captured_messages = messages
        return self._plan_response


def test_planner_node_propagates_runnable_config_to_llm_invoke(synthetic_task, fast_app_config):
    """Regression: ``planner_node`` must forward its ``config`` to ``llm.invoke``.

    Same root cause as the model_node regression: without forwarding the
    LangGraph ``RunnableConfig``, ``MetricsCollector`` never records the LLM
    call performed during plan generation.
    """
    llm = _ConfigCapturingLLM()
    sentinel = object()
    runtime_config = {
        "configurable": {"llm": llm},
        "callbacks": [sentinel],
    }
    planner_node(_base_state(synthetic_task), config=runtime_config)

    assert llm.captured_config is not None, (
        "planner_node did not forward config to llm.invoke; callbacks will "
        "not propagate and MetricsCollector will record tokens=0."
    )
    forwarded_callbacks = llm.captured_config.get("callbacks") if isinstance(
        llm.captured_config, dict
    ) else None
    assert forwarded_callbacks == [sentinel], (
        f"Expected callbacks=[<sentinel>] forwarded to llm.invoke, "
        f"got {forwarded_callbacks!r}"
    )
