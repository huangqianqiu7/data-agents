"""
ReAct Agent — observe-then-act loop without a global plan.

Core loop: Thought → Action → Observation → repeat until the ``answer``
tool is called or ``max_steps`` is exhausted.
"""
from __future__ import annotations

import logging
from typing import Any

from data_agent_refactored.config import ReActAgentConfig
from data_agent_refactored.agents.base_agent import BaseAgent
from data_agent_refactored.agents.model import ModelAdapter, ModelMessage
from data_agent_refactored.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_refactored.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_refactored.agents.json_parser import parse_model_step
from data_agent_refactored.agents.data_preview_gate import GATE_REMINDER
from data_agent_refactored.agents.text_helpers import progress, preview_json
from data_agent_refactored.benchmark.schema import PublicTask
from data_agent_refactored.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ReActAgent(BaseAgent):
    """ReAct-style agent: Thought → Action → Observation loop."""

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(model=model, tools=tools)
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    # -- message assembly --------------------------------------------------

    def _build_messages(
        self, task: PublicTask, state: AgentRuntimeState
    ) -> list[ModelMessage]:
        """Assemble the full conversation: system + task + history replay."""
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages: list[ModelMessage] = [
            ModelMessage(role="system", content=system_content),
            ModelMessage(role="user", content=build_task_prompt(task)),
        ]
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    # -- main loop ---------------------------------------------------------

    def run(self, task: PublicTask) -> AgentRunResult:
        """Execute the ReAct loop for *task*."""
        state = AgentRuntimeState()
        cfg = self.config
        show_progress = cfg.progress
        max_steps = cfg.max_steps

        for step_index in range(1, max_steps + 1):
            if show_progress:
                progress(f"[react] Step {step_index}/{max_steps}: calling model ...")

            # 1) Model call (with retry)
            raw = self._call_model_with_retry(
                self._build_messages(task, state),
                step_index,
                state,
                show_progress=show_progress,
                timeout_seconds=cfg.model_timeout_s,
                max_retries=cfg.max_model_retries,
                retry_backoff=cfg.model_retry_backoff,
                tag="react",
            )
            if raw is None:
                continue

            try:
                # 2) Parse model output
                model_step = parse_model_step(raw)
                if show_progress:
                    progress(
                        f"[react] Step {step_index}/{max_steps}: "
                        f"action={model_step.action!r} "
                        f"input={preview_json(model_step.action_input)}"
                    )

                # 3) Gate check
                if self._check_gate(
                    model_step, state, cfg.require_data_preview_before_compute
                ):
                    observation: dict[str, Any] = {
                        "ok": False,
                        "tool": model_step.action,
                        "error": GATE_REMINDER,
                    }
                    state.steps.append(StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw,
                        observation=observation,
                        ok=False,
                    ))
                    if show_progress:
                        progress(
                            f"[react] Step {step_index}/{max_steps}: blocked (preview gate)"
                        )
                    continue

                # 4) Tool validation + execution
                tool_result = self._validate_and_execute_tool(
                    task, model_step, raw, state, step_index,
                    show_progress=show_progress,
                    tool_timeout_seconds=cfg.tool_timeout_s,
                    tag="react",
                )
                if tool_result is None:
                    continue

                # 5) Record result
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw,
                    observation=observation,
                    ok=tool_result.ok,
                ))
                if show_progress:
                    term = " (terminal)" if tool_result.is_terminal else ""
                    progress(
                        f"[react] Step {step_index}/{max_steps}: ok={tool_result.ok}{term}"
                    )

                # 6) Terminal tool → submit answer
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    if show_progress:
                        progress("[react] Submitted final answer.")
                    break

            except Exception as exc:
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="",
                    action="__error__",
                    action_input={},
                    raw_response=raw,
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                ))
                logger.warning("[react] Step %d: error: %s", step_index, exc)
                if show_progress:
                    progress(f"[react] Step {step_index}/{max_steps}: error: {exc}")

        return self._finalize(task, state, show_progress, tag="react")
