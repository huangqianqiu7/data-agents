"""
数据预览门控机制。

强制大模型在执行代码或提交答案之前，必须先"看过数据"。
从 react.py 和 plan_and_solve.py 中提取的共享逻辑。

常量：
  - DATA_PREVIEW_ACTIONS: 数据预览类工具集合
  - GATED_ACTIONS:        受门控限制的工具集合
  - GATE_REMINDER:        门控拦截提示消息

函数：
  - has_data_preview(state): 检查历史中是否已成功预览过数据
"""
from __future__ import annotations

from data_agent_baseline.agents_v2.runtime import AgentRuntimeState


# 数据预览类工具（"审题"动作）
DATA_PREVIEW_ACTIONS = frozenset({
    "read_csv",
    "read_json",
    "read_doc",
    "inspect_sqlite_schema",
})

# 受门控限制的工具（必须先预览数据才能使用）
GATED_ACTIONS = frozenset({
    "execute_python",
    "execute_context_sql",
    "answer",
})

# 门控拦截时返回给大模型的提示消息
GATE_REMINDER = (
    "Blocked: inspect data first. Call read_csv / read_json / read_doc / "
    "inspect_sqlite_schema before using execute_python, execute_context_sql, or answer."
)


def has_data_preview(state: AgentRuntimeState) -> bool:
    """历史中是否至少成功执行过一次数据预览工具。"""
    return any(s.ok and s.action in DATA_PREVIEW_ACTIONS for s in state.steps)
