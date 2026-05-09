"""
LangGraph ``RunState`` TypedDict 定义。

每个图节点都会读 / 写其中一部分字段。实现要点：

  - ``steps`` 必须用 ``Annotated[list, operator.add]`` 声明，使 reducer
    把并发更新拼接而不是覆盖（§5.1 / C1）。

  - 所有字段都限定为 **可 pickle 的基本类型**，让 checkpointer
    （``MemorySaver`` / ``SqliteSaver``）能序列化。``Path`` / ``BaseTool``
    实例 / ``AppConfig`` 等不能放进 state —— 节点入口现场重建（§5.3 /
    C4 / D4）。

  - ``step_index`` 仅由 ``model_node`` 递增；其他节点只追加
    ``StepRecord``，不动计数器（D2 / §5.4）。

  - ``gate_decision`` / ``skip_tool`` / ``last_tool_*`` 是显式路由字段；
    条件边直接读它们而不是去看 ``steps[-1]``（C11）。

  - ``subgraph_exit`` 让内层 ``execution_subgraph`` 通知外层是继续、
    finalize 还是触发 replan（D7 / E14）。

  - ``last_error_kind`` 把 ``tool_validation``（pydantic schema / 输入
    形态错误）与 ``tool_error``（工具体内运行时异常）分开，便于
    ``advance_node`` rule 5 应用 v4 M5 白名单（§5.1 / §6.4 / E15）。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

# 这些 import 不放在 ``TYPE_CHECKING`` 守护下：LangGraph 与测试会调
# ``typing.get_type_hints(RunState, include_extras=True)`` 在运行时解析
# ``steps`` 的 reducer 注解，缺名字会抛 ``NameError``。
from data_agent_langchain.agents.runtime import StepRecord
from data_agent_langchain.benchmark.schema import AnswerTable


# ---------------------------------------------------------------------------
# 路由用的 Literal 类型（保留模块级以便测试复用）
# ---------------------------------------------------------------------------

GateDecision = Literal["pass", "block", "forced_inject"]

# v4 E15：``tool_validation`` 与 ``tool_error`` 是不同的枚举值。
# advance_node rule 5（v4 M5）的白名单据此判断是否触发 replan，
# 合并它们会让 parse error 误触 replan。
LastErrorKind = Literal[
    "parse_error",
    "unknown_tool",
    "tool_timeout",
    "tool_validation",   # pydantic schema / 入参问题
    "tool_error",        # 工具体内运行时异常
    "model_error",
    "gate_block",
    "max_steps_exceeded",
]

# v3 D7 / v4 E14：由 ``advance_node`` 写入的子图退出信号。
SubgraphExit = Literal["continue", "done", "replan_required"]

AgentMode = Literal["react", "plan_solve"]
ActionMode = Literal["tool_calling", "json_action"]


class RunState(TypedDict, total=False):
    """所有 LangGraph 节点共享的唯一 state 对象。

    所有字段都用 ``total=False`` 声明为可选；节点只写自己关心的子集，
    ``StateGraph`` 把 partial update 合并到当前 state。
    """

    # ----- 任务标识（全部基本类型，可 pickle） -----
    task_id: str
    question: str
    difficulty: str
    dataset_root: str              # ``rehydrate_task`` 使用（D17 / E8）
    task_dir: str                  # 用 str 存路径以保 picklable
    context_dir: str

    # ----- 模式 -----
    mode: AgentMode
    action_mode: ActionMode

    # ----- Plan-and-Solve 计划追踪 -----
    plan: list[str]
    plan_index: int
    replan_used: int

    # ----- 步骤累积（必须配 reducer） -----
    # 用 ``operator.add`` 把并发 update 拼接而不是覆盖（C1）。
    # 多节点 fake graph 测试会验证此行为。
    steps: Annotated[list[StepRecord], operator.add]

    # ----- 终止信号 -----
    answer: AnswerTable | None
    failure_reason: str | None

    # ----- gate 状态 -----
    discovery_done: bool
    preview_done: bool
    known_paths: list[str]
    consecutive_gate_blocks: int
    gate_decision: GateDecision
    skip_tool: bool

    # ----- 单轮缓存：model_node 写，parse / tool 消费 -----
    raw_response: str
    thought: str
    action: str
    action_input: dict[str, Any]

    # ----- 上一动作的结果（显式路由字段，C11） -----
    last_tool_ok: bool | None
    last_tool_is_terminal: bool
    last_error_kind: LastErrorKind | None

    # ----- 子图退出信号（D7 / E14） -----
    subgraph_exit: SubgraphExit

    # ----- 步骤计数 -----
    step_index: int
    max_steps: int

    # ----- trace 记录用的阶段字段 -----
    phase: str
    plan_progress: str
    plan_step_description: str


__all__ = [
    "ActionMode",
    "AgentMode",
    "GateDecision",
    "LastErrorKind",
    "RunState",
    "SubgraphExit",
]