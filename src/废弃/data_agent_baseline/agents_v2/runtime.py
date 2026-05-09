"""
运行时状态与结果记录。

包含：
  - StepRecord:        单步执行记录（不可变）
  - AgentRuntimeState: Agent 运行时的"短期记忆"（可变）
  - AgentRunResult:    最终运行报告（不可变）
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


# =====================================================================
# 步骤记录
# =====================================================================
@dataclass(frozen=True, slots=True)
class StepRecord:
    """单步执行的完整记录（不可变）。"""
    step_index: int                 # 步骤编号
    thought: str                    # 大模型思考过程
    action: str                     # 工具名称
    action_input: dict[str, Any]    # 工具参数
    raw_response: str               # 大模型原始输出
    observation: dict[str, Any]     # 工具返回结果
    ok: bool                        # 是否执行成功

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# 运行时状态
# =====================================================================
@dataclass(slots=True)
class AgentRuntimeState:
    """Agent 运行时的可变状态。"""
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None


# =====================================================================
# 最终运行报告
# =====================================================================
@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """任务完成后的最终报告（不可变）。"""
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
