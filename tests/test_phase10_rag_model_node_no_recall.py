"""A2 护栏：``model_node`` 不主动触发 corpus recall（M4.5.5）。"""
from __future__ import annotations

import json
from pathlib import Path

from data_agent_langchain.agents import model_node as model_module
from data_agent_langchain.config import default_app_config
from data_agent_langchain.memory.types import MemoryHit


def _task_state(tmp_path: Path) -> dict:
    """构造 ``model_node._build_messages_for_state`` 可用的最小 state。"""
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
        "max_steps": 60,
        "mode": "react",
        "action_mode": "json_action",
        "phase": "execution",
        "plan_progress": "",
        "plan_step_description": "",
        "steps": [],
        "memory_hits": [
            MemoryHit(
                record_id="doc-a#0000",
                namespace="corpus_task:task_1",
                score=0.9,
                summary="[markdown] README.md: use data.csv",
            )
        ],
    }


def test_model_node_build_messages_does_not_call_corpus_recall(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """即使循环构造 60 次 prompt，``model_node`` 也只渲染既有 hits。"""
    calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("model_node must not call recall_corpus_snippets")

    # 若未来有人在 model_node 中引入同名全局函数，本 monkeypatch 会立即拦截。
    monkeypatch.setattr(
        model_module,
        "recall_corpus_snippets",
        fail_if_called,
        raising=False,
    )
    state = _task_state(tmp_path)
    app_config = default_app_config()

    for _ in range(60):
        messages = model_module._build_messages_for_state(state, app_config)
        assert "Reference snippets" in str(messages[-1].content)
        assert len(state["memory_hits"]) == 1

    assert calls == 0

