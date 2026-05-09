"""
智能体运行时数据结构定义。

本模块承载贯穿 LangGraph 各节点的核心 dataclass：

  - :class:`StepRecord` —— 单步执行的不可变记录（trace.json 行格式）。
  - :class:`AgentRuntimeState` —— legacy 风格的可变运行时状态；
    LangGraph 后端用 ``RunState`` TypedDict 取代它，保留这个类是为了与
    legacy 共享 ``StepRecord`` 形态，trace 文件 schema 完全兼容。
  - :class:`AgentRunResult` —— 任务结束后的不可变结果摘要（trace.json 顶层）。
  - :class:`ModelMessage` —— legacy ModelAdapter 接口与 ``parse_model_step``
    使用的简单 role/content 消息体。
  - :class:`ModelStep` —— 解析自 LLM 输出的 (thought, action, action_input)。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_langchain.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    """单步智能体执行的不可变记录。"""
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool
    # ----- 阶段 / 计划进度 -----
    phase: str = ""                    # "planning" 或 "execution"
    plan_progress: str = ""            # 例如 "2/5"
    plan_step_description: str = ""    # 当前 plan 步骤的文字描述

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRuntimeState:
    """运行过程中累积的可变状态（兼容 legacy 接口）。"""
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """任务结束后由 finalize_node 产出的不可变报告。"""
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None

    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.failure_reason,
            "succeeded": self.succeeded,
        }


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """LLM 对话中的单条消息（role + content）。

    在 ``json_action`` 模式下用于拼装提示词；``tool_calling`` 模式直接
    使用 LangChain ``BaseMessage`` 家族。
    """
    role: str       # "system" / "user" / "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    """从 LLM 输出解析出的结构化动作。"""
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


__all__ = [
    "AgentRunResult",
    "AgentRuntimeState",
    "ModelMessage",
    "ModelStep",
    "StepRecord",
]