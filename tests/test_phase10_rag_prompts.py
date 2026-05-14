"""corpus snippets prompt 渲染与 ``model_node`` 注入测试（M4.5.2）。"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import HumanMessage

from data_agent_langchain.agents.model_node import _build_messages_for_state
from data_agent_langchain.agents.prompts import render_corpus_snippets
from data_agent_langchain.config import default_app_config
from data_agent_langchain.memory.types import MemoryHit


def _task_state(tmp_path: Path) -> dict:
    """构造可被 ``rehydrate_task`` 使用的最小 RunState。"""
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


def _corpus_hit(summary: str = "[markdown] README.md: use data.csv") -> MemoryHit:
    """构造 corpus namespace 的 ``MemoryHit``。"""
    return MemoryHit(
        record_id="doc-a#0000",
        namespace="corpus_task:task_1",
        score=0.91,
        summary=summary,
    )


def test_render_corpus_snippets_empty_returns_empty_string() -> None:
    """空 hits 不渲染任何 prompt 片段。"""
    assert render_corpus_snippets([], budget_chars=1800) == ""


def test_render_corpus_snippets_uses_only_summary() -> None:
    """corpus prompt 只允许 ``summary`` 进入，不暴露 id、namespace、score。"""
    text = render_corpus_snippets([_corpus_hit()], budget_chars=1800)

    assert "## Reference snippets (from task documentation)" in text
    assert "[markdown] README.md: use data.csv" in text
    assert "doc-a#0000" not in text
    assert "corpus_task:task_1" not in text
    assert "0.91" not in text
    assert "MemoryHit" not in text


def test_render_corpus_snippets_respects_budget_chars() -> None:
    """渲染结果总长度不超过 ``budget_chars``。"""
    text = render_corpus_snippets(
        [_corpus_hit("[markdown] README.md: " + "x" * 200)],
        budget_chars=90,
    )

    assert len(text) <= 90
    assert text.endswith("...")


def test_model_node_appends_dataset_facts_then_corpus_snippets(tmp_path: Path) -> None:
    """``model_node`` 应先注入 dataset facts，再注入 corpus snippets。"""
    state = _task_state(tmp_path)
    state["memory_hits"] = [
        MemoryHit(
            record_id="dataset-a",
            namespace="dataset:task_1",
            score=1.0,
            summary="File: a.csv  Kind: csv  Columns: x",
        ),
        _corpus_hit(),
    ]

    messages = _build_messages_for_state(state, default_app_config())

    assert isinstance(messages[-1], HumanMessage)
    content = str(messages[-1].content)
    assert "## Dataset facts" in content
    assert "File: a.csv" in content
    assert "## Reference snippets (from task documentation)" in content
    assert "[markdown] README.md: use data.csv" in content
    assert content.index("## Dataset facts") < content.index("## Reference snippets")


def test_model_node_uses_rag_prompt_budget_for_corpus_hits(tmp_path: Path) -> None:
    """``model_node`` 应使用 ``app_config.memory.rag.prompt_budget_chars`` 截断 corpus。"""
    state = _task_state(tmp_path)
    state["memory_hits"] = [_corpus_hit("[markdown] README.md: " + "x" * 200)]

    base = default_app_config()
    app_config = replace(
        base,
        memory=replace(
            base.memory,
            rag=replace(base.memory.rag, prompt_budget_chars=100),
        ),
    )

    messages = _build_messages_for_state(state, app_config)

    assert isinstance(messages[-1], HumanMessage)
    content = str(messages[-1].content)
    assert "Dataset facts" not in content
    assert "Reference snippets" in content
    assert len(content) <= 100

