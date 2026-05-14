"""
ReAct 外层图（§9.1 / D7）。

外层图刻意保持极简：挂载共享的 ``execution_subgraph`` + ``finalize_node``。

  START → execution → finalize → END

ReAct 没有 plan，没有 replanner，所以子图返回 ``replan_required`` 时
直接路由到 ``finalize`` —— 与 ``done`` 行为完全相同。``advance_node``
rule 5 触发 replan 的唯一前提是 "工具失败且无 replan 预算"，在 ReAct
模式下属于终止失败。

通过 :func:`build_react_graph` 按需构造（未编译）；runner / 测试自行
``.compile(...)`` 并挂载所需 checkpointer（Phase 5 接 MemorySaver /
SqliteSaver）。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from data_agent_langchain.agents.execution_subgraph import build_execution_subgraph
from data_agent_langchain.agents.finalize import finalize_node
from data_agent_langchain.agents.task_entry_node import task_entry_node
from data_agent_langchain.runtime.state import RunState


def build_react_graph() -> StateGraph:
    """构造（未编译的）ReAct 外层图。"""
    inner = build_execution_subgraph().compile()

    g: StateGraph = StateGraph(RunState)
    g.add_node("task_entry", task_entry_node)
    g.add_node("execution", inner)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "task_entry")
    g.add_edge("task_entry", "execution")
    g.add_edge("execution", "finalize")
    g.add_edge("finalize", END)
    return g


__all__ = ["build_react_graph"]
