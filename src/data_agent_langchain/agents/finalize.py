"""
``finalize_node`` —— 把终态 :class:`RunState` 转为 :class:`AgentRunResult`。

v3 D14 / §9.3：

  - ``state["answer"] is None`` 且 ``state["failure_reason"] is None``
    时，必须把 ``failure_reason`` 设置为 byte-for-byte 的字面量
    ``"Agent did not submit an answer within max_steps."``。任何措辞
    偏差都会破坏 legacy parity 测试（参见
    ``data_agent_refactored/agents/base_agent.py:216-217``）。

  - 本节点只写 ``failure_reason``；``trace.json`` / ``prediction.csv``
    由 runner 基于同一份 state 单独写出。

节点本身刻意做得极简：无 LLM 调用、无 tool I/O、无日志输出，仅作为图
显式的终止节点存在，让 Phase 5 的 metrics / summary 可以挂在这里。
"""
from __future__ import annotations

from typing import Any

from data_agent_langchain.agents.runtime import AgentRunResult, StepRecord
from data_agent_langchain.benchmark.schema import AnswerTable
from data_agent_langchain.runtime.state import RunState

# 真值来源（也是 parity 参考）：legacy ``BaseAgent._finalize``。
# 不要润色这条字符串 —— 一字之差就会破坏 parity（D14）。
_MAX_STEPS_FAILURE_MSG: str = "Agent did not submit an answer within max_steps."


def finalize_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    """没有产出 answer 时把 ``failure_reason`` 写好。

    形参 *config* 接受任意 LangGraph ``RunnableConfig``，仅为了与其他
    图节点签名对称；本函数内部不读取它。
    """
    answer = state.get("answer")
    failure_reason = state.get("failure_reason")
    if answer is None and failure_reason is None:
        failure_reason = _MAX_STEPS_FAILURE_MSG
    return {"failure_reason": failure_reason}


def build_run_result(task_id: str, state: RunState) -> AgentRunResult:
    """把终态 ``RunState`` 打包成 :class:`AgentRunResult`。

    Phase 5 runner 子进程在 ``compiled.invoke`` 返回后调用本函数。
    放在这里（而非 ``run/runner.py``）是为了让直接驱动图的单元测试
    也能拼装结果，无需引入 runner 机器。
    """
    answer: AnswerTable | None = state.get("answer")
    steps: list[StepRecord] = list(state.get("steps") or [])
    failure_reason = state.get("failure_reason")
    if answer is None and failure_reason is None:
        failure_reason = _MAX_STEPS_FAILURE_MSG
    return AgentRunResult(
        task_id=task_id,
        answer=answer,
        steps=steps,
        failure_reason=failure_reason,
    )


__all__ = [
    "_MAX_STEPS_FAILURE_MSG",
    "build_run_result",
    "finalize_node",
]