"""
Plan-and-Solve 外层图（§10）。

  START → planner → execution
                    ├─ done            → finalize → END
                    └─ replan_required ─┐
                                        ├─ replan budget remains  → replanner → execution
                                        └─ budget exhausted       → finalize  → END

planner / replanner 节点写 ``state["plan"]`` 与 ``plan_index``；
execution 节点把内层 ``execution_subgraph`` 整体当一个节点用，
returns 的 partial state 经过 *trim* 后只保留本轮新增的 steps，
避免外层 reducer 把内层 ``StepRecord`` 重复追加。
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from data_agent_langchain.agents.advance_node import _can_replan
from data_agent_langchain.agents.execution_subgraph import build_execution_subgraph
from data_agent_langchain.agents.finalize import finalize_node
from data_agent_langchain.agents.planner_node import planner_node, replanner_node
from data_agent_langchain.runtime.state import RunState


def build_plan_solve_graph() -> StateGraph:
    """构造（未编译的）Plan-and-Solve 外层图。"""
    g: StateGraph = StateGraph(RunState)
    g.add_node("planner", planner_node)
    g.add_node("execution", _execution_node)
    g.add_node("replanner", replanner_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "execution")
    g.add_conditional_edges(
        "execution",
        _route_after_execution,
        {
            "finalize": "finalize",
            "replan": "replanner",
        },
    )
    g.add_edge("replanner", "execution")
    g.add_edge("finalize", END)
    return g


def _execution_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    """把内层 ``execution_subgraph`` 当作单节点跑，并裁剪重复的 steps。"""
    compiled = build_execution_subgraph().compile()
    before_count = len(state.get("steps") or [])
    result = dict(compiled.invoke(state, config=config))
    if "steps" in result:
        # 内层子图返回的 steps 是包含 *所有* 历史步骤的合并视图；外层 reducer
        # 还会再 ``operator.add``，会出现重复。这里只保留本轮新增的子集。
        result["steps"] = list(result["steps"])[before_count:]
    return result


def _route_after_execution(state: RunState) -> str:
    """子图退出后的条件边：done 终止；replan_required 视预算决定 replan / 终止。"""
    exit_kind = state.get("subgraph_exit") or "continue"
    if exit_kind == "done":
        return "finalize"
    if exit_kind == "replan_required":
        return "replan" if _can_replan(state) else "finalize"
    return "finalize"


__all__ = ["build_plan_solve_graph"]