import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from data_agent_langchain.agents.model_node import _build_messages_for_state
from data_agent_langchain.config import default_app_config
from data_agent_langchain.memory.types import MemoryHit


def _task_state(tmp_path: Path) -> dict:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "difficulty": "easy",
                "question": "Which file should be inspected?",
            }
        ),
        encoding="utf-8",
    )
    return {
        "dataset_root": str(tmp_path),
        "task_id": "task_1",
        "step_index": 0,
        "max_steps": 16,
        "mode": "react",
        "action_mode": "json_action",
        "phase": "execution",
        "plan_progress": "",
        "plan_step_description": "",
        "steps": [],
    }


def _joined_content(messages) -> str:
    return "\n".join(str(message.content) for message in messages)


def test_build_messages_appends_dataset_facts_for_react_memory_hits(tmp_path):
    state = _task_state(tmp_path)
    state["memory_hits"] = [
        MemoryHit(
            record_id="record-a",
            namespace="dataset:task_1",
            score=1.0,
            summary="File: a.csv  Kind: csv  Columns: ['a', 'b']",
        )
    ]

    messages = _build_messages_for_state(state, default_app_config())

    assert isinstance(messages[-1], HumanMessage)
    assert "Dataset facts" in str(messages[-1].content)
    assert "a.csv" in _joined_content(messages)


def test_build_messages_omits_dataset_facts_when_memory_hits_missing(tmp_path):
    messages = _build_messages_for_state(_task_state(tmp_path), default_app_config())

    assert "Dataset facts" not in _joined_content(messages)


def test_build_messages_appends_dataset_facts_for_plan_solve_memory_hits(tmp_path):
    state = _task_state(tmp_path)
    state.update(
        {
            "mode": "plan_solve",
            "plan": ["Inspect available files"],
            "plan_index": 0,
            "memory_hits": [
                MemoryHit(
                    record_id="record-b",
                    namespace="dataset:task_1",
                    score=0.8,
                    summary="File: b.csv  Kind: csv  Columns: ['x']",
                )
            ],
        }
    )

    messages = _build_messages_for_state(state, default_app_config())

    assert isinstance(messages[-1], HumanMessage)
    assert "Dataset facts" in str(messages[-1].content)
    assert "b.csv" in _joined_content(messages)
