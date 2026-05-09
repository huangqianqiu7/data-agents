"""
Phase 3 — End-to-end ReAct graph test using ``FakeListChatModel``.

This is the integration test for the whole Phase 3 deliverable: the
compiled ReAct graph drives a synthetic task to a successful ``answer``
without any network calls.

The fake LLM produces three responses in order:
  1. list_context  — discovers files
  2. read_csv      — previews the matches.csv
  3. answer        — submits the final table

Each response is a properly-fenced JSON action so ``parse_action_node``
parses them. After the third action, ``advance_node`` sees
``last_tool_is_terminal=True`` and routes to ``finalize`` via the outer
graph's edge.

The test also covers:
  - The full RunState terminal shape (answer present, failure_reason None).
  - StepRecord trace records the 3 executed steps in order with
    ``step_index`` advancing 1,2,3 (D2: model_node is the only writer).
  - ``build_run_result`` packs state into a successful AgentRunResult.
"""
from __future__ import annotations

import csv
import json
import operator
from pathlib import Path

import pytest
from langchain_core.language_models import FakeListChatModel

from data_agent_langchain.agents.finalize import build_run_result
from data_agent_langchain.agents.react_graph import build_react_graph
from data_agent_langchain.config import AgentConfig, AppConfig, ToolsConfig
from data_agent_langchain.runtime.context import _APP_CONFIG


# Three pre-canned LLM responses that drive the ReAct loop to success.
_LIST_CTX = '```json\n{"thought":"Inspect","action":"list_context","action_input":{"max_depth":4}}\n```'
_READ_CSV = '```json\n{"thought":"Read","action":"read_csv","action_input":{"path":"matches.csv","max_rows":5}}\n```'
_ANSWER  = '```json\n{"thought":"Done","action":"answer","action_input":{"columns":["id"],"rows":[[1]]}}\n```'


@pytest.fixture()
def task_root(tmp_path: Path) -> Path:
    """Create a minimal ``task_test`` tree under tmp_path and return root."""
    task_dir = tmp_path / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_test", "difficulty": "easy", "question": "Q?"}),
        encoding="utf-8",
    )
    csv_path = context_dir / "matches.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id"])
        w.writerow([1])
    return tmp_path


@pytest.fixture()
def app_cfg():
    cfg = AppConfig(
        agent=AgentConfig(
            max_steps=8,
            max_model_retries=1,
            model_retry_backoff=(0.0,),
            require_data_preview_before_compute=True,
        ),
        tools=ToolsConfig(),
    )
    token = _APP_CONFIG.set(cfg)
    try:
        yield cfg
    finally:
        _APP_CONFIG.reset(token)


def test_react_graph_e2e_three_step_success(task_root, app_cfg):
    """list_context -> read_csv -> answer drives the graph to success."""
    fake_llm = FakeListChatModel(responses=[_LIST_CTX, _READ_CSV, _ANSWER])
    graph = build_react_graph().compile()

    initial_state = {
        "task_id": "task_test",
        "question": "Q?",
        "difficulty": "easy",
        "dataset_root": str(task_root),
        "task_dir": str(task_root / "task_test"),
        "context_dir": str(task_root / "task_test" / "context"),
        "mode": "react",
        "action_mode": "json_action",
        "plan": [],
        "plan_index": 0,
        "replan_used": 0,
        "steps": [],
        "answer": None,
        "failure_reason": None,
        "discovery_done": False,
        "preview_done": False,
        "known_paths": [],
        "consecutive_gate_blocks": 0,
        "step_index": 0,
        "max_steps": 8,
    }

    final = graph.invoke(initial_state, config={"configurable": {"llm": fake_llm}})

    # ----- Terminal state -----
    assert final["answer"] is not None
    assert final["answer"].columns == ["id"]
    assert final["answer"].rows == [[1]]
    assert final["failure_reason"] is None

    # ----- Trace shape -----
    steps = final["steps"]
    actions = [s.action for s in steps]
    assert actions == ["list_context", "read_csv", "answer"]
    # D2: model_node is the only writer of step_index, so each step
    # carries the index assigned at the model turn that produced it.
    assert [s.step_index for s in steps] == [1, 2, 3]
    assert all(s.ok for s in steps)


def test_react_graph_runner_packs_result(task_root, app_cfg):
    """Confirm ``build_run_result`` translates the terminal state."""
    fake_llm = FakeListChatModel(responses=[_LIST_CTX, _READ_CSV, _ANSWER])
    compiled = build_react_graph().compile()

    initial_state = {
        "task_id": "task_test",
        "question": "Q?",
        "dataset_root": str(task_root),
        "mode": "react",
        "action_mode": "json_action",
        "step_index": 0,
        "max_steps": 8,
        "steps": [],
        "answer": None,
        "failure_reason": None,
    }
    final = compiled.invoke(initial_state, config={"configurable": {"llm": fake_llm}})
    result = build_run_result("task_test", final)
    assert result.succeeded is True
    assert result.answer is final["answer"]
    assert len(result.steps) == 3


def test_react_graph_max_steps_failure(task_root, app_cfg):
    """A model that loops forever on list_context must finalise with the D14 message."""
    fake_llm = FakeListChatModel(responses=[_LIST_CTX] * 12)
    compiled = build_react_graph().compile()

    initial_state = {
        "task_id": "task_test",
        "dataset_root": str(task_root),
        "mode": "react",
        "action_mode": "json_action",
        "step_index": 0,
        "max_steps": 4,
        "steps": [],
        "answer": None,
        "failure_reason": None,
    }
    final = compiled.invoke(initial_state, config={"configurable": {"llm": fake_llm}})
    assert final["answer"] is None
    assert final["failure_reason"] == "Agent did not submit an answer within max_steps."
