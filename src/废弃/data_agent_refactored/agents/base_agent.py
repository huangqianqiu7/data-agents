"""
Abstract base class for all agent types.

Encapsulates shared logic used by both ReAct and Plan-and-Solve agents:
  - Model call with exponential-backoff retry
  - Tool name pre-validation + timeout-guarded execution
  - Data-preview gate check
  - Finalisation and result packaging
"""
from __future__ import annotations

import abc
import logging
import time
from typing import Any

from data_agent_refactored.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_refactored.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_refactored.agents.text_helpers import progress
from data_agent_refactored.agents.timeout import call_with_timeout
from data_agent_refactored.agents.data_preview_gate import GATED_ACTIONS, has_data_preview
from data_agent_refactored.benchmark.schema import PublicTask
from data_agent_refactored.tools.registry import ToolExecutionResult, ToolRegistry

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Base class for agents.  Subclasses implement :meth:`run`."""

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
    ) -> None:
        self.model = model
        self.tools = tools

    @abc.abstractmethod
    def run(self, task: PublicTask) -> AgentRunResult:
        """Execute the agent loop for *task* and return the result."""
        ...

    # ----------------------------------------------------------------
    # Shared: model call with exponential-backoff retry
    # ----------------------------------------------------------------
    def _call_model_with_retry(
        self,
        messages: list[ModelMessage],
        step_index: int,
        state: AgentRuntimeState,
        *,
        show_progress: bool,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff: tuple[float, ...],
        tag: str = "agent",
        phase: str = "",
        plan_progress: str = "",
        plan_step_description: str = "",
    ) -> str | None:
        """Call the model API with retry.  Returns ``None`` on exhaustion."""
        for attempt in range(max_retries):
            try:
                return call_with_timeout(
                    self.model.complete, (messages,), timeout_seconds,
                )
            except (TimeoutError, RuntimeError) as exc:
                if attempt < max_retries - 1:
                    backoff = retry_backoff[min(attempt, len(retry_backoff) - 1)]
                    logger.warning(
                        "[%s] Step %d: model call attempt %d/%d failed: %s — "
                        "retrying in %.1fs",
                        tag, step_index, attempt + 1, max_retries, exc, backoff,
                    )
                    if show_progress:
                        progress(
                            f"[{tag}] Step {step_index}: model call failed "
                            f"(attempt {attempt + 1}/{max_retries}): {exc}, "
                            f"retrying in {backoff}s ..."
                        )
                    time.sleep(backoff)
                else:
                    state.steps.append(StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response="",
                        observation={
                            "ok": False,
                            "error": f"Model call failed after {max_retries} attempts: {exc}",
                        },
                        ok=False,
                        phase=phase,
                        plan_progress=plan_progress,
                        plan_step_description=plan_step_description,
                    ))
                    logger.error(
                        "[%s] Step %d: model call failed after all %d retries",
                        tag, step_index, max_retries,
                    )
                    if show_progress:
                        progress(f"[{tag}] Step {step_index}: model call failed after all retries")
        return None

    # ----------------------------------------------------------------
    # Shared: tool name validation + timeout-guarded execution
    # ----------------------------------------------------------------
    def _validate_and_execute_tool(
        self,
        task: PublicTask,
        model_step: ModelStep,
        raw: str,
        state: AgentRuntimeState,
        step_index: int,
        *,
        show_progress: bool,
        tool_timeout_seconds: float,
        tag: str = "agent",
        phase: str = "",
        plan_progress: str = "",
        plan_step_description: str = "",
    ) -> ToolExecutionResult | None:
        """Validate the tool name and execute it.

        Returns:
            A :class:`ToolExecutionResult` on success, or ``None`` when
            validation / timeout fails (state is updated internally).
        """
        available_tools: set[str] = set(self.tools.handlers.keys())
        if model_step.action not in available_tools:
            observation: dict[str, Any] = {
                "ok": False,
                "tool": model_step.action,
                "error": (
                    f"Unknown tool '{model_step.action}'. "
                    f"Available tools: {', '.join(sorted(available_tools))}"
                ),
            }
            state.steps.append(StepRecord(
                step_index=step_index,
                thought=model_step.thought,
                action=model_step.action,
                action_input=model_step.action_input,
                raw_response=raw,
                observation=observation,
                ok=False,
                phase=phase,
                plan_progress=plan_progress,
                plan_step_description=plan_step_description,
            ))
            if show_progress:
                progress(f"[{tag}] Step {step_index}: unknown tool '{model_step.action}'")
            return None

        try:
            return call_with_timeout(
                self.tools.execute,
                (task, model_step.action, model_step.action_input),
                tool_timeout_seconds,
            )
        except TimeoutError:
            observation = {
                "ok": False,
                "tool": model_step.action,
                "error": (
                    f"Tool '{model_step.action}' timed out after {tool_timeout_seconds}s. "
                    f"Try simplifying your code or query."
                ),
            }
            state.steps.append(StepRecord(
                step_index=step_index,
                thought=model_step.thought,
                action=model_step.action,
                action_input=model_step.action_input,
                raw_response=raw,
                observation=observation,
                ok=False,
                phase=phase,
                plan_progress=plan_progress,
                plan_step_description=plan_step_description,
            ))
            if show_progress:
                progress(f"[{tag}] Step {step_index}: tool timed out")
            return None

    # ----------------------------------------------------------------
    # Shared: data-preview gate check
    # ----------------------------------------------------------------
    @staticmethod
    def _check_gate(
        model_step: ModelStep,
        state: AgentRuntimeState,
        require_preview: bool,
    ) -> bool:
        """Return ``True`` if the gate should block the current action."""
        return (
            require_preview
            and model_step.action in GATED_ACTIONS
            and not has_data_preview(state)
        )

    # ----------------------------------------------------------------
    # Shared: finalisation
    # ----------------------------------------------------------------
    @staticmethod
    def _finalize(
        task: PublicTask,
        state: AgentRuntimeState,
        show_progress: bool,
        tag: str = "agent",
    ) -> AgentRunResult:
        """Package the state into a final :class:`AgentRunResult`."""
        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."
        if show_progress and state.answer is None:
            progress(f"[{tag}] Stopped: {state.failure_reason}")
        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
