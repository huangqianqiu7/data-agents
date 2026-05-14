"""ReAct task 入口召回节点（M4.5.4）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from data_agent_langchain.agents.corpus_recall import recall_corpus_snippets
from data_agent_langchain.agents.memory_recall import recall_dataset_facts
from data_agent_langchain.config import AppConfig, default_app_config
from data_agent_langchain.runtime.context import get_current_app_config
from data_agent_langchain.runtime.state import RunState


def task_entry_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    """ReAct 图入口：对当前 task 做一次 dataset + corpus 召回。"""
    app_config = _safe_get_app_config()
    dataset_name = Path(state.get("dataset_root", "") or ".").name or "default"
    task_id = str(state.get("task_id") or "")
    query = str(state.get("question") or "")

    dataset_hits = recall_dataset_facts(
        memory_cfg=app_config.memory,
        dataset=dataset_name,
        node="task_entry_node",
        config=config,
    )
    corpus_hits = recall_corpus_snippets(
        app_config.memory.rag,
        task_id=task_id,
        query=query,
        node="task_entry_node",
        config=config,
    )
    memory_hits = list(dataset_hits) + list(corpus_hits)
    if not memory_hits:
        return {}
    return {"memory_hits": memory_hits}


def _safe_get_app_config() -> AppConfig:
    """读取当前 AppConfig；直接跑单测时回退到默认配置。"""
    try:
        return get_current_app_config()
    except RuntimeError:
        return default_app_config()


__all__ = ["task_entry_node"]

