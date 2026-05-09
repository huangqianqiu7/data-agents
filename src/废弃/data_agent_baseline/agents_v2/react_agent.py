"""
ReAct Agent — 走一步看一步的智能体。

核心循环：Thought → Action → Observation → 重复
每步都是即兴决策，没有全局计划。

相比原版 react.py 的改进：
  - 继承 BaseAgent，复用模型重试、工具校验等共享逻辑
  - JSON 解析使用统一的 json_parser 模块
  - 新增模型调用重试（指数退避）
  - 新增工具名预校验
"""
from __future__ import annotations

from dataclasses import dataclass

from data_agent_baseline.agents_v2.base_agent import BaseAgent
from data_agent_baseline.agents_v2.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents_v2.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents_v2.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.agents_v2.json_parser import parse_model_step
from data_agent_baseline.agents_v2.gate import GATE_REMINDER
from data_agent_baseline.agents_v2.text_helpers import progress, preview_json
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


# =====================================================================
# 配置类
# =====================================================================
@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16                                # 最大工具调用次数
    progress: bool = False                             # 是否打印运行进度
    require_data_preview_before_compute: bool = True   # 强制先预览数据
    model_timeout_s: float = 120.0                     # 单次模型调用超时（秒）
    tool_timeout_s: float = 180.0                      # 单次工具执行超时（秒）
    max_model_retries: int = 3                         # 模型 API 瞬态错误最大重试次数
    model_retry_backoff: tuple[float, ...] = (2.0, 5.0, 10.0)  # 重试退避秒数序列


# =====================================================================
# ReAct Agent
# =====================================================================
class ReActAgent(BaseAgent):
    """ReAct 架构智能体：Thought → Action → Observation 循环。"""

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

    # ----------------------------------------------------------------
    # 消息构建
    # ----------------------------------------------------------------
    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        """组装历史对话消息：系统设定 + 任务 + 历史步骤回放。"""
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    # ----------------------------------------------------------------
    # 主循环
    # ----------------------------------------------------------------
    def run(self, task: PublicTask) -> AgentRunResult:
        """ReAct 主循环。"""
        state = AgentRuntimeState()
        cfg = self.config
        prog = cfg.progress
        max_steps = cfg.max_steps

        for step_index in range(1, max_steps + 1):
            if prog:
                progress(f"[react] Step {step_index}/{max_steps}: calling model ...")

            # 1) 模型调用（带重试）
            raw = self._call_model_with_retry(
                self._build_messages(task, state), step_index, state,
                prog=prog, timeout_s=cfg.model_timeout_s,
                max_retries=cfg.max_model_retries,
                retry_backoff=cfg.model_retry_backoff,
                tag="react",
            )
            if raw is None:
                continue

            try:
                # 2) 解析模型输出
                model_step = parse_model_step(raw)
                if prog:
                    progress(
                        f"[react] Step {step_index}/{max_steps}: "
                        f"action={model_step.action!r} "
                        f"input={preview_json(model_step.action_input)}"
                    )

                # 3) 门控检查
                if self._check_gate(model_step, state, cfg.require_data_preview_before_compute):
                    observation = {
                        "ok": False,
                        "tool": model_step.action,
                        "error": GATE_REMINDER,
                    }
                    state.steps.append(StepRecord(
                        step_index=step_index,
                        thought=model_step.thought, action=model_step.action,
                        action_input=model_step.action_input, raw_response=raw,
                        observation=observation, ok=False,
                    ))
                    if prog:
                        progress(f"[react] Step {step_index}/{max_steps}: blocked (preview gate)")
                    continue

                # 4) 工具校验 + 执行
                tool_result = self._validate_and_execute_tool(
                    task, model_step, raw, state, step_index,
                    prog=prog, tool_timeout_s=cfg.tool_timeout_s, tag="react",
                )
                if tool_result is None:
                    continue

                # 5) 记录执行结果
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought=model_step.thought, action=model_step.action,
                    action_input=model_step.action_input, raw_response=raw,
                    observation=observation, ok=tool_result.ok,
                ))
                if prog:
                    term = " (terminal)" if tool_result.is_terminal else ""
                    progress(f"[react] Step {step_index}/{max_steps}: ok={tool_result.ok}{term}")

                # 6) 终止型工具 → 交卷
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    if prog:
                        progress("[react] Submitted final answer.")
                    break

            except Exception as exc:
                # 容错：解析/工具异常 → 记录错误，让大模型下轮自行修正
                state.steps.append(StepRecord(
                    step_index=step_index, thought="",
                    action="__error__", action_input={},
                    raw_response=raw,
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                ))
                if prog:
                    progress(f"[react] Step {step_index}/{max_steps}: error: {exc}")

        return self._finalize(task, state, prog, tag="react")
