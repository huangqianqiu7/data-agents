"""
Agent 抽象基类。

封装两种 Agent（ReAct / Plan-and-Solve）共享的核心逻辑：
  - 模型调用（带指数退避重试）     ← 修复审计隐患 4
  - 工具名预校验 + 带超时执行      ← 修复审计隐患 7
  - 简单门控检查
  - 收尾与结果封装
"""
from __future__ import annotations

import abc
import time

from data_agent_baseline.agents_v2.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents_v2.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.agents_v2.text_helpers import progress
from data_agent_baseline.agents_v2.timeout import call_with_timeout
from data_agent_baseline.agents_v2.gate import GATED_ACTIONS, has_data_preview
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


class BaseAgent(abc.ABC):
    """智能体基类，子类只需实现 run() 主循环。"""

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
        """子类实现主循环。"""
        ...

    # ================================================================
    # 共享方法：模型调用（带指数退避重试）
    # ================================================================
    def _call_model_with_retry(
        self,
        messages: list[ModelMessage],
        step_index: int,
        state: AgentRuntimeState,
        *,
        prog: bool,
        timeout_s: float,
        max_retries: int,
        retry_backoff: tuple[float, ...],
        tag: str = "agent",
    ) -> str | None:
        """调用模型 API，失败时进行指数退避重试。

        返回:
            模型原始输出文本；若所有重试耗尽则返回 None 并将错误记录到 state。
        """
        for attempt in range(max_retries):
            try:
                return call_with_timeout(
                    self.model.complete, (messages,), timeout_s,
                )
            except (TimeoutError, RuntimeError) as exc:
                # RuntimeError 包含 429 限流、502 网关等可恢复的 API 错误
                if attempt < max_retries - 1:
                    backoff = retry_backoff[min(attempt, len(retry_backoff) - 1)]
                    if prog:
                        progress(
                            f"[{tag}] Step {step_index}: model call failed "
                            f"(attempt {attempt+1}/{max_retries}): {exc}, "
                            f"retrying in {backoff}s ..."
                        )
                    time.sleep(backoff)
                else:
                    # 所有重试耗尽
                    state.steps.append(StepRecord(
                        step_index=step_index, thought="",
                        action="__error__", action_input={}, raw_response="",
                        observation={
                            "ok": False,
                            "error": f"Model call failed after {max_retries} attempts: {exc}",
                        },
                        ok=False,
                    ))
                    if prog:
                        progress(f"[{tag}] Step {step_index}: model call failed after all retries")
        return None

    # ================================================================
    # 共享方法：工具名校验 + 带超时执行
    # ================================================================
    def _validate_and_execute_tool(
        self,
        task: PublicTask,
        model_step: ModelStep,
        raw: str,
        state: AgentRuntimeState,
        step_index: int,
        *,
        prog: bool,
        tool_timeout_s: float,
        tag: str = "agent",
    ) -> object | None:
        """校验工具名并执行工具调用。

        返回:
            ToolExecutionResult 对象（成功或工具自身报错）；
            None 表示校验失败或超时，已自行写入 state，调用方应 continue。
        """
        # --- 工具名预校验（防止幻觉工具名浪费步数） ---
        available_tools = set(self.tools.handlers.keys())
        if model_step.action not in available_tools:
            obs = {
                "ok": False,
                "tool": model_step.action,
                "error": f"Unknown tool '{model_step.action}'. "
                         f"Available tools: {', '.join(sorted(available_tools))}",
            }
            state.steps.append(StepRecord(
                step_index=step_index,
                thought=model_step.thought, action=model_step.action,
                action_input=model_step.action_input, raw_response=raw,
                observation=obs, ok=False,
            ))
            if prog:
                progress(f"[{tag}] Step {step_index}: unknown tool '{model_step.action}'")
            return None

        # --- 执行工具（带超时保护） ---
        try:
            return call_with_timeout(
                self.tools.execute,
                (task, model_step.action, model_step.action_input),
                tool_timeout_s,
            )
        except TimeoutError:
            obs = {
                "ok": False,
                "tool": model_step.action,
                "error": f"Tool '{model_step.action}' timed out after {tool_timeout_s}s. "
                         f"Try simplifying your code or query.",
            }
            state.steps.append(StepRecord(
                step_index=step_index,
                thought=model_step.thought, action=model_step.action,
                action_input=model_step.action_input, raw_response=raw,
                observation=obs, ok=False,
            ))
            if prog:
                progress(f"[{tag}] Step {step_index}: tool timed out")
            return None

    # ================================================================
    # 共享方法：简单门控检查
    # ================================================================
    @staticmethod
    def _check_gate(
        model_step: ModelStep,
        state: AgentRuntimeState,
        require_preview: bool,
    ) -> bool:
        """检查是否需要门控拦截。返回 True 表示需要拦截。"""
        return (
            require_preview
            and model_step.action in GATED_ACTIONS
            and not has_data_preview(state)
        )

    # ================================================================
    # 共享方法：收尾与结果封装
    # ================================================================
    @staticmethod
    def _finalize(
        task: PublicTask,
        state: AgentRuntimeState,
        prog: bool,
        tag: str = "agent",
    ) -> AgentRunResult:
        """主循环结束后收尾：设置失败原因（若未交卷）并封装结果。"""
        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."
        if prog and state.answer is None:
            progress(f"[{tag}] Stopped: {state.failure_reason}")
        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
