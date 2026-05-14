"""plan-solve ``planner_node`` 接入 corpus recall 的测试（M4.5.3）。"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from langchain_core.language_models import FakeListChatModel

import data_agent_langchain.agents.planner_node as planner_module
from data_agent_langchain.config import CorpusRagConfig, MemoryConfig, default_app_config
from data_agent_langchain.memory.types import MemoryHit


def _make_task(dataset_root: Path) -> dict:
    """构造 plan-solve planner 可读取的最小 task。"""
    task_dir = dataset_root / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": "task_test",
                "difficulty": "easy",
                "question": "question from task.json",
            }
        ),
        encoding="utf-8",
    )
    return {
        "dataset_root": str(dataset_root),
        "task_id": "task_test",
        "question": "question from state",
    }


def _base_state(task_state: dict) -> dict:
    """构造 planner_node 所需 RunState 子集。"""
    return {
        **task_state,
        "mode": "plan_solve",
        "action_mode": "json_action",
        "steps": [],
        "plan_index": 0,
        "replan_used": 0,
        "step_index": 0,
        "max_steps": 20,
    }


def _planner_config():
    """返回稳定产出 plan 的 fake LLM 配置。"""
    return {
        "configurable": {
            "llm": FakeListChatModel(
                responses=['```json\n{"plan":["List files","Read data"]}\n```']
            )
        }
    }


def test_planner_node_recalls_dataset_then_corpus_and_merges_hits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """RAG 开启时 planner 应先召回 dataset，再召回 corpus，并合并 hits。"""
    calls: list[tuple[str, dict]] = []
    dataset_hit = MemoryHit(
        record_id="dataset-a",
        namespace="dataset:ds",
        score=1.0,
        summary="File: a.csv",
    )
    corpus_hit = MemoryHit(
        record_id="doc-a#0000",
        namespace="corpus_task:task_test",
        score=0.9,
        summary="[markdown] README.md: use a.csv",
    )
    app_config = replace(
        default_app_config(),
        memory=MemoryConfig(
            mode="read_only_dataset",
            path=tmp_path,
            rag=CorpusRagConfig(enabled=True),
        ),
    )
    monkeypatch.setattr(planner_module, "_safe_get_app_config", lambda: app_config)

    def fake_recall_dataset_facts(memory_cfg, dataset, node, config):
        calls.append(
            {
                "kind": "dataset",
                "memory_cfg": memory_cfg,
                "dataset": dataset,
                "node": node,
                "config": config,
            }
        )
        return [dataset_hit]

    def fake_recall_corpus_snippets(memory_cfg, *, task_id, query, node, config):
        calls.append(
            {
                "kind": "corpus",
                "memory_cfg": memory_cfg,
                "task_id": task_id,
                "query": query,
                "node": node,
                "config": config,
            }
        )
        return [corpus_hit]

    monkeypatch.setattr(planner_module, "recall_dataset_facts", fake_recall_dataset_facts)
    monkeypatch.setattr(planner_module, "recall_corpus_snippets", fake_recall_corpus_snippets)
    runtime_config = _planner_config()

    output = planner_module.planner_node(
        _base_state(_make_task(tmp_path / "ds")),
        config=runtime_config,
    )

    assert [call["kind"] for call in calls] == ["dataset", "corpus"]
    assert calls[0]["dataset"] == "ds"
    assert calls[0]["node"] == "planner_node"
    assert calls[0]["config"] is runtime_config
    # Bug 1 修复后 planner 传完整 ``MemoryConfig``（含 mode）而不是 子 cfg ``rag``，
    # 以便 helper 内部同时按 ``memory.mode`` 守卫。
    assert calls[1]["memory_cfg"] is app_config.memory
    assert calls[1]["task_id"] == "task_test"
    assert calls[1]["query"] == "question from state"
    assert calls[1]["node"] == "planner_node"
    assert calls[1]["config"] is runtime_config
    assert output["memory_hits"] == [dataset_hit, corpus_hit]


def test_planner_node_keeps_dataset_only_when_corpus_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """corpus 未命中时 planner 应保持 v2 dataset facts 行为。"""
    dataset_hit = MemoryHit(
        record_id="dataset-a",
        namespace="dataset:ds",
        score=1.0,
        summary="File: a.csv",
    )
    app_config = replace(
        default_app_config(),
        memory=MemoryConfig(
            mode="read_only_dataset",
            path=tmp_path,
            rag=CorpusRagConfig(enabled=False),
        ),
    )
    monkeypatch.setattr(planner_module, "_safe_get_app_config", lambda: app_config)
    monkeypatch.setattr(
        planner_module,
        "recall_dataset_facts",
        lambda memory_cfg, dataset, node, config: [dataset_hit],
    )
    monkeypatch.setattr(
        planner_module,
        "recall_corpus_snippets",
        lambda memory_cfg, *, task_id, query, node, config: [],
    )

    output = planner_module.planner_node(
        _base_state(_make_task(tmp_path / "ds")),
        config=_planner_config(),
    )

    assert output["memory_hits"] == [dataset_hit]

