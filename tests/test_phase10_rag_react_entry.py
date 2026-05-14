"""ReAct task entry recall 节点测试（M4.5.4）。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from data_agent_langchain.config import CorpusRagConfig, MemoryConfig, default_app_config
from data_agent_langchain.memory.types import MemoryHit
from data_agent_langchain.runtime.state import RunState


def _state(tmp_path: Path) -> dict:
    """构造 task_entry_node / ReAct graph 可用的最小 state。"""
    return {
        "task_id": "task_test",
        "question": "question from state",
        "dataset_root": str(tmp_path / "ds"),
        "mode": "react",
        "action_mode": "json_action",
        "step_index": 0,
        "max_steps": 8,
        "steps": [],
        "answer": None,
        "failure_reason": None,
    }


def _dataset_hit() -> MemoryHit:
    """构造 dataset memory hit。"""
    return MemoryHit(
        record_id="dataset-a",
        namespace="dataset:ds",
        score=1.0,
        summary="File: a.csv",
    )


def _corpus_hit() -> MemoryHit:
    """构造 corpus memory hit。"""
    return MemoryHit(
        record_id="doc-a#0000",
        namespace="corpus_task:task_test",
        score=0.9,
        summary="[markdown] README.md: use a.csv",
    )


def test_task_entry_node_returns_empty_when_both_recalls_empty(tmp_path: Path, monkeypatch):
    """memory disabled + rag disabled 时不写 ``memory_hits``。"""
    import data_agent_langchain.agents.task_entry_node as entry_module

    app_config = replace(
        default_app_config(),
        memory=MemoryConfig(
            mode="disabled",
            path=tmp_path,
            rag=CorpusRagConfig(enabled=False),
        ),
    )
    monkeypatch.setattr(entry_module, "_safe_get_app_config", lambda: app_config)
    monkeypatch.setattr(
        entry_module,
        "recall_dataset_facts",
        lambda memory_cfg, dataset, node, config: [],
    )
    monkeypatch.setattr(
        entry_module,
        "recall_corpus_snippets",
        lambda cfg, *, task_id, query, node, config: [],
    )

    assert entry_module.task_entry_node(_state(tmp_path), config=None) == {}


def test_task_entry_node_keeps_dataset_only_when_rag_disabled(tmp_path: Path, monkeypatch):
    """rag 关闭但 dataset memory 开启时只写 dataset hits。"""
    import data_agent_langchain.agents.task_entry_node as entry_module

    app_config = replace(
        default_app_config(),
        memory=MemoryConfig(
            mode="read_only_dataset",
            path=tmp_path,
            rag=CorpusRagConfig(enabled=False),
        ),
    )
    monkeypatch.setattr(entry_module, "_safe_get_app_config", lambda: app_config)
    monkeypatch.setattr(
        entry_module,
        "recall_dataset_facts",
        lambda memory_cfg, dataset, node, config: [_dataset_hit()],
    )
    monkeypatch.setattr(
        entry_module,
        "recall_corpus_snippets",
        lambda cfg, *, task_id, query, node, config: [],
    )

    output = entry_module.task_entry_node(_state(tmp_path), config=None)

    assert output == {"memory_hits": [_dataset_hit()]}


def test_task_entry_node_merges_dataset_then_corpus_hits(tmp_path: Path, monkeypatch):
    """rag 开启且两类召回均命中时按 dataset → corpus 顺序合并。"""
    import data_agent_langchain.agents.task_entry_node as entry_module

    calls: list[dict] = []
    app_config = replace(
        default_app_config(),
        memory=MemoryConfig(
            mode="read_only_dataset",
            path=tmp_path,
            rag=CorpusRagConfig(enabled=True),
        ),
    )
    monkeypatch.setattr(entry_module, "_safe_get_app_config", lambda: app_config)

    def fake_dataset(memory_cfg, dataset, node, config):
        calls.append({"kind": "dataset", "dataset": dataset, "node": node, "config": config})
        return [_dataset_hit()]

    def fake_corpus(cfg, *, task_id, query, node, config):
        calls.append(
            {
                "kind": "corpus",
                "task_id": task_id,
                "query": query,
                "node": node,
                "config": config,
            }
        )
        return [_corpus_hit()]

    monkeypatch.setattr(entry_module, "recall_dataset_facts", fake_dataset)
    monkeypatch.setattr(entry_module, "recall_corpus_snippets", fake_corpus)
    runtime_config = {"callbacks": []}

    output = entry_module.task_entry_node(_state(tmp_path), config=runtime_config)

    assert [call["kind"] for call in calls] == ["dataset", "corpus"]
    assert calls[0]["dataset"] == "ds"
    assert calls[0]["node"] == "task_entry_node"
    assert calls[1]["task_id"] == "task_test"
    assert calls[1]["query"] == "question from state"
    assert calls[1]["node"] == "task_entry_node"
    assert output == {"memory_hits": [_dataset_hit(), _corpus_hit()]}


def test_react_graph_runs_task_entry_before_execution(tmp_path: Path, monkeypatch):
    """ReAct 进入 execution 前，state 应已经带有 task_entry 写入的 memory hits。"""
    import data_agent_langchain.agents.react_graph as react_graph
    import data_agent_langchain.agents.task_entry_node as entry_module

    app_config = replace(
        default_app_config(),
        memory=MemoryConfig(
            mode="read_only_dataset",
            path=tmp_path,
            rag=CorpusRagConfig(enabled=True),
        ),
    )
    monkeypatch.setattr(entry_module, "_safe_get_app_config", lambda: app_config)
    monkeypatch.setattr(
        entry_module,
        "recall_dataset_facts",
        lambda memory_cfg, dataset, node, config: [],
    )
    monkeypatch.setattr(
        entry_module,
        "recall_corpus_snippets",
        lambda cfg, *, task_id, query, node, config: [_corpus_hit()],
    )

    def fake_build_execution_subgraph() -> StateGraph:
        g: StateGraph = StateGraph(RunState)

        def capture_memory_count(state: RunState, config=None) -> dict:
            return {
                "failure_reason": f"memory_hits={len(state.get('memory_hits') or [])}",
                "subgraph_exit": "done",
            }

        g.add_node("capture_memory_count", capture_memory_count)
        g.add_edge(START, "capture_memory_count")
        g.add_edge("capture_memory_count", END)
        return g

    monkeypatch.setattr(
        react_graph,
        "build_execution_subgraph",
        fake_build_execution_subgraph,
    )

    final = react_graph.build_react_graph().compile().invoke(_state(tmp_path))

    assert final["failure_reason"] == "memory_hits=1"

