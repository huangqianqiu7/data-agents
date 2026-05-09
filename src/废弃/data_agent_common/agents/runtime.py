"""
Runtime state and result recording for agent execution.

Contains:
  - :class:`StepRecord`: immutable record of one agent step (trace schema).
  - :class:`AgentRuntimeState`: mutable working memory used by the legacy
    hand-written loop. The new LangGraph backend uses ``RunState`` (TypedDict)
    instead, but reuses :class:`StepRecord` so ``trace.json`` schema stays
    backwards compatible.
  - :class:`AgentRunResult`: immutable final report (``trace.json`` payload).
  - :class:`ModelMessage`: a single role/content message used by the legacy
    ``ModelAdapter`` interface and by ``parse_model_step``.
  - :class:`ModelStep`: parsed (thought, action, action_input) value object.

These classes are the canonical source. ``data_agent_refactored`` re-exports
them for backwards compatibility (LANGCHAIN_MIGRATION_PLAN.md v4 E1/E2).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_common.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Immutable record of a single agent step."""
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool
    # --- Phase / plan progress ---
    phase: str = ""                    # "planning" | "execution"
    plan_progress: str = ""            # e.g. "2/5"
    plan_step_description: str = ""    # current plan step description

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRuntimeState:
    """Mutable working memory accumulated during an agent run.

    Used by the legacy ``BaseAgent`` loop. The LangGraph backend keeps the
    same schema (``StepRecord`` / ``answer`` / ``failure_reason``) inside its
    ``RunState`` TypedDict instead of this dataclass.
    """
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Immutable final report produced after an agent run completes."""
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
    """A single message in an LLM conversation.

    Used by the legacy ``ModelAdapter`` protocol and by the JSON action
    fallback path of the new LangGraph backend (when assembling prompts in
    ``json_action`` mode). The ``tool_calling`` path uses LangChain's
    ``BaseMessage`` family instead.
    """
    role: str       # "system" / "user" / "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    """Structured representation of one agent action parsed from LLM output.

    Returned by ``parse_model_step`` and consumed by both backends:
      - Legacy: as the input to ``BaseAgent._validate_and_execute_tool``.
      - LangGraph: ``parse_action_node`` reads ``thought / action / action_input``
        directly into ``RunState`` (LANGCHAIN_MIGRATION_PLAN.md §9.2.1).
    """
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
