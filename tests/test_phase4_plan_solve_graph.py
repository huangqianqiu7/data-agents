from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from langchain_core.language_models import FakeListChatModel

from data_agent_langchain.agents.plan_solve_graph import (
    _route_after_execution,
    build_plan_solve_graph,
)
from data_agent_langchain.config import AgentConfig, AppConfig, ToolsConfig
from data_agent_langchain.runtime.context import _APP_CONFIG


_PLAN = '```json\n{"plan":["List files","Read data","Answer"]}\n```'
_LIST_CTX = '```json\n{"thought":"Inspect","action":"list_context","action_input":{"max_depth":4}}\n```'
_READ_CSV = '```json\n{"thought":"Read","action":"read_csv","action_input":{"path":"matches.csv","max_rows":5}}\n```'
_ANSWER = '```json\n{"thought":"Done","action":"answer","action_input":{"columns":["id"],"rows":[[1]]}}\n```'


@pytest.fixture()
def task_root(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_test", "difficulty": "easy", "question": "Q?"}),
        encoding="utf-8",
    )
    with (context_dir / "matches.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["id"], [1]])
    return tmp_path


@pytest.fixture()
def app_cfg():
    cfg = AppConfig(
        agent=AgentConfig(
            max_steps=8,
            max_replans=1,
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


def _initial_state(task_root: Path) -> dict:
    return {
        "task_id": "task_test",
        "question": "Q?",
        "difficulty": "easy",
        "dataset_root": str(task_root),
        "task_dir": str(task_root / "task_test"),
        "context_dir": str(task_root / "task_test" / "context"),
        "mode": "plan_solve",
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


def test_route_after_execution_finalizes_done(app_cfg):
    assert _route_after_execution({"subgraph_exit": "done"}) == "finalize"


def test_route_after_execution_replans_only_with_budget(app_cfg):
    assert _route_after_execution({"subgraph_exit": "replan_required", "replan_used": 0}) == "replan"
    assert _route_after_execution({"subgraph_exit": "replan_required", "replan_used": 1}) == "finalize"


def test_plan_solve_graph_e2e_success(task_root, app_cfg):
    fake_llm = FakeListChatModel(responses=[_PLAN, _LIST_CTX, _READ_CSV, _ANSWER])
    graph = build_plan_solve_graph().compile()
    final = graph.invoke(_initial_state(task_root), config={"configurable": {"llm": fake_llm}})
    assert final["answer"] is not None
    assert final["answer"].columns == ["id"]
    assert final["answer"].rows == [[1]]
    assert final["failure_reason"] is None
    assert final["plan"] == ["List files", "Read data", "Answer"]
    assert final["plan_index"] == 2
    assert [s.action for s in final["steps"]] == ["__plan_generation__", "list_context", "read_csv", "answer"]
    assert final["steps"][0].phase == "planning"
    assert [s.step_index for s in final["steps"][1:]] == [1, 2, 3]
