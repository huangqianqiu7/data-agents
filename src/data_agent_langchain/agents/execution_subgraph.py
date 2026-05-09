"""
``execution_subgraph`` —— ReAct 与 Plan-and-Solve 共享的内层 T-A-O 循环
（§9.0 / D7）。

拓扑::

  START → model_node → parse_action_node → gate_node ─┐
                                                       │  skip_tool? → advance_node
                                                       └─ no  → tool_node → advance_node
                                          advance_node → model_node      (continue)
                                                       → END             (done)
                                                       → END             (replan_required)

外层图（``react_graph`` 或 ``plan_solve_graph``）通过读取
``state["subgraph_exit"]`` 决定子图退出后是 ``finalize`` 还是
``replanner``。

图按需懒构造，仅在调用 :func:`build_execution_subgraph` 时才触发
LangGraph 编译机器，让 import 阶段保持轻量。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from data_agent_langchain.agents.advance_node import advance_node
from data_agent_langchain.agents.gate import gate_node
from data_agent_langchain.agents.model_node import model_node
from data_agent_langchain.agents.parse_action import parse_action_node
from data_agent_langchain.agents.tool_node import tool_node
from data_agent_langchain.runtime.state import RunState


def _route_after_gate(state: RunState) -> str:
    """``gate_node`` 后的条件边。

    pass / forced_inject（``skip_tool=False``）→ 执行原 action。
    block（``skip_tool=True``）→ 跳过 tool_node 直接到 advance。
    parse_error 透传：gate_node 在该情况下也会写 ``skip_tool=True``，
    同样走这条分支。
    """
    return "advance_node" if state.get("skip_tool") else "tool_node"


def _route_after_advance(state: RunState) -> str:
    """``advance_node`` 后的条件边。

    把 ``subgraph_exit`` 枚举映射到子图节点名。任何意外值都会落到
    ``END``，让外层图通过附加在子图节点上的条件边接管。
    """
    exit_kind = state.get("subgraph_exit") or "continue"
    if exit_kind == "continue":
        return "model_node"
    # done / replan_required 都让子图退出，外层图根据
    # state.subgraph_exit 决定下一步。
    return END


def build_execution_subgraph() -> StateGraph:
    """构造内层 T-A-O 循环；调用方负责 ``compile()``。"""
    g: StateGraph = StateGraph(RunState)
    g.add_node("model_node", model_node)
    g.add_node("parse_action_node", parse_action_node)
    g.add_node("gate_node", gate_node)
    g.add_node("tool_node", tool_node)
    g.add_node("advance_node", advance_node)

    g.add_edge(START, "model_node")
    g.add_edge("model_node", "parse_action_node")
    g.add_edge("parse_action_node", "gate_node")
    g.add_conditional_edges(
        "gate_node",
        _route_after_gate,
        {
            "tool_node": "tool_node",
            "advance_node": "advance_node",
        },
    )
    g.add_edge("tool_node", "advance_node")
    g.add_conditional_edges(
        "advance_node",
        _route_after_advance,
        {
            "model_node": "model_node",
            END: END,
        },
    )
    return g


__all__ = ["build_execution_subgraph"]