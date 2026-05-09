"""
``advance_node`` —— 纯 state 决策节点，写入 ``subgraph_exit``。

§10.3（v3 D5 + v4 M5）规则优先级（与 legacy ``BaseAgent`` 一致）：

  1. ``answer is not None`` 或 ``last_tool_is_terminal`` → ``done``。
  2. ``step_index >= max_steps``                        → ``done``。
  3. ``last_error_kind == "gate_block"``                → ``continue``
     （gate L1 / L2 已消费一步；让循环重试）。
  4. （仅 Plan-and-Solve）成功时 ``plan_index += 1``，计划耗尽时追加
     ``FALLBACK_STEP_PROMPT``（D5）。
  5. ``last_tool_ok is False`` 且
     ``last_error_kind in _REPLAN_TRIGGER_ERROR_KINDS``  → ``replan_required``
     （v4 M5：只对 tool_error / tool_timeout / tool_validation 三态触发；
     parse / model / unknown_tool 不会，让 LLM 下一轮自行纠错，与 baseline
     一致）。
  6. 其他                                                → ``continue``。

规则 4 在这里实现是为了 Phase 4 前向兼容；ReAct 外层图（Phase 3）
不会触发 replan —— 决策由 ``execution_subgraph`` 在 ReAct / Plan-and-Solve
之间共享（D7）。如果 ``state["mode"]`` 缺失或为 ``"react"``，规则 4 是
no-op，规则 5 仍然产 replan_required —— 该信号会被 ReAct 外层图当作
``done`` 处理（无 replanner 存在）。

本节点绝不动 ``step_index``（归 ``model_node`` 拥有，D2），也不追加 ``steps``
（``StepRecord`` 由产生可观察副作用的节点写入）。
"""
from __future__ import annotations

from typing import Any

from data_agent_langchain.constants import FALLBACK_STEP_PROMPT
from data_agent_langchain.config import AppConfig, default_app_config
from data_agent_langchain.runtime.context import get_current_app_config
from data_agent_langchain.runtime.state import RunState

# v4 M5：只有这几种 error_kind 会触发 replan。仅限 ``tool_*`` 族（E15），
# 避免 parse_error / model_error 误触 replan。
_REPLAN_TRIGGER_ERROR_KINDS: frozenset[str] = frozenset({
    "tool_error",
    "tool_timeout",
    "tool_validation",
})


# ---------------------------------------------------------------------------
# 公共节点
# ---------------------------------------------------------------------------

def advance_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    """根据 *state* 计算下一个 ``subgraph_exit`` 值。

    遵循 LangGraph reducer 协议返回 partial state；只写 ``subgraph_exit``，
    Plan-and-Solve 模式下额外写 ``plan`` / ``plan_index``。
    """
    update: dict[str, Any] = {}

    # 1) 终止 answer 或终止 tool 结果 → done
    if state.get("answer") is not None or state.get("last_tool_is_terminal"):
        update["subgraph_exit"] = "done"
        return update

    # 2) 步骤预算耗尽 → done
    step_index = int(state.get("step_index", 0) or 0)
    max_steps = int(state.get("max_steps") or _default_max_steps())
    if step_index >= max_steps:
        update["subgraph_exit"] = "done"
        return update

    # 3) gate block / parse error / model error：继续（model_node 会重试）
    if state.get("last_error_kind") == "gate_block":
        update["subgraph_exit"] = "continue"
        return update

    # 4) Plan-and-Solve：成功时推进 plan_index；计划耗尽时追加 FALLBACK_STEP
    if state.get("mode") == "plan_solve":
        plan = list(state.get("plan") or [])
        plan_index = int(state.get("plan_index", 0) or 0)
        if state.get("last_tool_ok"):
            plan_index += 1
            update["plan_index"] = plan_index
        # 重新读取（可能已更新过的）plan 长度做边界检查。
        if plan_index >= len(plan):
            if not plan or plan[-1] != FALLBACK_STEP_PROMPT:
                plan = plan + [FALLBACK_STEP_PROMPT]
                update["plan"] = plan
                update["plan_index"] = max(0, len(plan) - 1)
            else:
                update["plan_index"] = max(0, len(plan) - 1)

    # 5) 工具失败 + 仍有 replan 预算 → replan_required（v4 M5 白名单）
    if (
        state.get("last_tool_ok") is False
        and state.get("last_error_kind") in _REPLAN_TRIGGER_ERROR_KINDS
        and _can_replan(state)
    ):
        update["subgraph_exit"] = "replan_required"
        return update

    # 6) 默认：继续循环
    update["subgraph_exit"] = "continue"
    return update


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _can_replan(state: RunState) -> bool:
    """``replan_used < max_replans``（v3 D5）。

    优先取 ``state`` 上的值（让测试可固定预算），缺失时回退到 AppConfig 默认。
    """
    cfg = _safe_get_app_config()
    used = int(state.get("replan_used", 0) or 0)
    max_replans = int(getattr(cfg.agent, "max_replans", 2))
    return used < max_replans


def _safe_get_app_config() -> AppConfig:
    try:
        return get_current_app_config()
    except RuntimeError:
        return default_app_config()


def _default_max_steps() -> int:
    return int(getattr(_safe_get_app_config().agent, "max_steps", 20))


__all__ = ["advance_node", "_REPLAN_TRIGGER_ERROR_KINDS"]