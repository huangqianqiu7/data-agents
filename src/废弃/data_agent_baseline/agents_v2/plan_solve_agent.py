"""
Plan-and-Solve Agent — 先规划全局，再按计划执行的智能体。

两阶段架构：
  Phase 1: 让大模型生成分步计划
  Phase 2: 按计划逐步执行工具调用，支持动态重规划

相比原版 plan_and_solve.py (819 行单体) 的改进：
  - 继承 BaseAgent，复用模型重试、工具校验等共享逻辑
  - 使用 context_manager 模块处理历史消息（钉住 + 滑动窗口）
  - 门控分级升级：提醒 → 改写计划 → 强制注入 list_context
  - 重规划失败用 step_index=-1 避免 ID 冲突（修复审计隐患 2）
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from data_agent_baseline.agents_v2.base_agent import BaseAgent
from data_agent_baseline.agents_v2.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents_v2.prompt import (
    PLAN_AND_SOLVE_SYSTEM_PROMPT,
    PLANNING_INSTRUCTION,
    EXECUTION_INSTRUCTION,
    build_task_prompt,
)
from data_agent_baseline.agents_v2.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.agents_v2.json_parser import parse_model_step, parse_plan
from data_agent_baseline.agents_v2.gate import GATED_ACTIONS, GATE_REMINDER, has_data_preview
from data_agent_baseline.agents_v2.context_manager import build_history_messages
from data_agent_baseline.agents_v2.text_helpers import progress, preview_json
from data_agent_baseline.agents_v2.timeout import call_with_timeout
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


# =====================================================================
# 配置类
# =====================================================================
@dataclass(frozen=True, slots=True)
class PlanAndSolveAgentConfig:
    max_steps: int = 20                               # 执行阶段最大工具调用次数
    max_replans: int = 2                               # 最多允许重新规划几次
    progress: bool = False                             # 是否打印运行进度
    require_data_preview_before_compute: bool = True   # 强制先预览数据
    max_obs_chars: int = 3000                          # 单条 observation content 最大字符数
    max_context_tokens: int = 24000                    # 历史消息区的 token 预算上限
    model_timeout_s: float = 120.0                     # 单次模型调用超时（秒）
    tool_timeout_s: float = 180.0                      # 单次工具执行超时（秒）
    max_gate_retries: int = 2                          # 门控连续拦截次数上限
    max_model_retries: int = 3                         # 模型 API 瞬态错误最大重试次数
    model_retry_backoff: tuple[float, ...] = (2.0, 5.0, 10.0)  # 重试退避秒数序列


# =====================================================================
# Plan-and-Solve Agent
# =====================================================================
class PlanAndSolveAgent(BaseAgent):
    """两阶段智能体：先规划全局，再按计划执行，执行中可动态重规划。"""

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: PlanAndSolveAgentConfig | None = None,
    ) -> None:
        super().__init__(model=model, tools=tools)
        self.config = config or PlanAndSolveAgentConfig()

    # ================================================================
    # Phase 1: 规划
    # ================================================================
    def _generate_plan(
        self, task: PublicTask, tool_desc: str, history_hint: str = "",
    ) -> list[str]:
        """调用大模型，根据任务和已有信息生成分步计划。"""
        system = (
            f"{PLAN_AND_SOLVE_SYSTEM_PROMPT}\n\n"
            f"Available tools:\n{tool_desc}\n\n"
            "Return a single ```json fenced block."
        )
        user = build_task_prompt(task) + "\n\n"
        if history_hint:
            user += f"Previous observations:\n{history_hint}\n\n"
        user += PLANNING_INSTRUCTION

        raw = call_with_timeout(
            self.model.complete,
            ([ModelMessage(role="system", content=system),
              ModelMessage(role="user", content=user)],),
            self.config.model_timeout_s,
        )
        return parse_plan(raw)

    # ================================================================
    # Phase 2: 构建执行消息
    # ================================================================
    def _build_exec_messages(
        self,
        task: PublicTask,
        tool_desc: str,
        plan: list[str],
        plan_idx: int,
        state: AgentRuntimeState,
    ) -> list[ModelMessage]:
        """组装执行阶段的对话：系统设定 + 任务 + 计划进度 + 历史观测。"""
        numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan))
        exec_block = EXECUTION_INSTRUCTION.format(
            numbered_plan=numbered,
            current=plan_idx + 1,
            total=len(plan),
            step_description=plan[plan_idx],
        )

        system = (
            f"{PLAN_AND_SOLVE_SYSTEM_PROMPT}\n\n"
            f"Available tools:\n{tool_desc}\n\n"
            "Return a single ```json fenced block with keys "
            "`thought`, `action`, `action_input`."
        )

        base_messages: list[ModelMessage] = [
            ModelMessage(role="system", content=system),
            ModelMessage(role="user", content=f"{build_task_prompt(task)}\n\n{exec_block}"),
        ]

        # 使用 context_manager 构建带双层截断的历史消息
        return build_history_messages(
            state, base_messages,
            max_obs_chars=self.config.max_obs_chars,
            max_context_tokens=self.config.max_context_tokens,
        )

    # ================================================================
    # 门控判定与分级升级
    # ================================================================
    def _handle_gate_block(
        self, task: PublicTask, model_step, raw: str,
        plan: list[str], plan_idx: int,
        state: AgentRuntimeState, step_index: int,
        consecutive_gate_blocks: int, prog: bool,
    ) -> int:
        """处理数据预览门控拦截，返回更新后的 consecutive_gate_blocks。

        返回值 > 0 时调用方应 continue 跳过当前步。
        返回 0 表示已通过门控（不需要拦截）。
        """
        cfg = self.config

        # 不需要门控检查的情况：直接放行
        if not (
            cfg.require_data_preview_before_compute
            and model_step.action in GATED_ACTIONS
            and not has_data_preview(state)
        ):
            return 0  # 放行信号

        # ---- 触发门控拦截 ----
        consecutive_gate_blocks += 1
        obs = {"ok": False, "tool": model_step.action, "error": GATE_REMINDER}
        state.steps.append(StepRecord(
            step_index=step_index,
            thought=model_step.thought, action=model_step.action,
            action_input=model_step.action_input, raw_response=raw,
            observation=obs, ok=False,
        ))

        # 分级处理
        if consecutive_gate_blocks >= cfg.max_gate_retries + 2:
            # 升级措施：强制注入 list_context 绕过模型
            if prog:
                progress(
                    f"[plan-solve] Step {step_index}: gate escalation — "
                    f"force-injecting list_context"
                )
            try:
                forced_result = call_with_timeout(
                    self.tools.execute,
                    (task, "list_context", {"max_depth": 4}),
                    cfg.tool_timeout_s,
                )
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="[auto-injected by gate escalation]",
                    action="list_context", action_input={"max_depth": 4},
                    raw_response="",
                    observation={
                        "ok": forced_result.ok, "tool": "list_context",
                        "content": forced_result.content,
                    },
                    ok=forced_result.ok,
                ))
            except Exception as forced_exc:
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="[auto-injected by gate escalation]",
                    action="list_context", action_input={"max_depth": 4},
                    raw_response="",
                    observation={"ok": False, "error": str(forced_exc)},
                    ok=False,
                ))
            consecutive_gate_blocks = 0  # 重置计数器
        elif consecutive_gate_blocks >= cfg.max_gate_retries:
            # 中级措施：改写当前计划步骤为强制数据预览
            plan[plan_idx] = (
                "MANDATORY: Inspect data files first by calling "
                "read_csv, read_json, read_doc, or inspect_sqlite_schema "
                "before any compute or answer action."
            )
            if prog:
                progress(
                    f"[plan-solve] Step {step_index}: gate blocked "
                    f"{consecutive_gate_blocks}x, plan step overridden"
                )
        elif prog:
            progress(f"[plan-solve] Step {step_index}: blocked (preview gate)")

        return consecutive_gate_blocks

    # ================================================================
    # 重规划逻辑
    # ================================================================
    def _handle_replan(
        self, task: PublicTask, tool_desc: str,
        state: AgentRuntimeState, step_index: int,
        replan_used: int, prog: bool,
    ) -> tuple[list[str], int] | None:
        """工具失败后尝试生成新计划。

        返回:
            (new_plan, 0) 表示重规划成功（plan_idx 重置为 0）；
            None 表示重规划失败，调用方应继续使用原计划。
        """
        if prog:
            progress("[plan-solve] Tool failed, re-planning ...")

        # 构建已完成动作摘要，防止新计划重复已成功的步骤
        completed = []
        failed_last = []
        for s in state.steps:
            pv = json.dumps(s.observation, ensure_ascii=False)[:200]
            if s.ok:
                completed.append(f"  ✓ {s.action}({preview_json(s.action_input, 80)}) → {pv}")
            elif s == state.steps[-1]:
                failed_last.append(f"  ✗ {s.action}({preview_json(s.action_input, 80)}) → {pv}")

        history_parts = []
        if completed:
            history_parts.append("Already completed (do NOT repeat):\n" + "\n".join(completed))
        if failed_last:
            history_parts.append("Last failed action:\n" + "\n".join(failed_last))
        history_parts.append(
            "IMPORTANT: Your new plan must skip steps that have already succeeded. "
            "Start from the next logical action."
        )
        history = "\n\n".join(history_parts)

        try:
            new_plan = self._generate_plan(task, tool_desc, history_hint=history)
            if prog:
                progress("[plan-solve] New plan:")
                for i, s in enumerate(new_plan, 1):
                    progress(f"[plan-solve]   {i}. {s}")
            return (new_plan, 0)
        except Exception as replan_exc:
            logging.warning(
                "[plan-solve] Re-plan failed (attempt %d/%d): %s",
                replan_used, self.config.max_replans, replan_exc,
            )
            if prog:
                progress(
                    f"[plan-solve] Re-plan failed: {replan_exc!r}, "
                    f"continuing with current plan."
                )
            # 使用虚拟 step_index=-1，避免与同轮工具失败记录的 step_index 冲突
            state.steps.append(StepRecord(
                step_index=-1, thought="",
                action="__replan_failed__", action_input={}, raw_response="",
                observation={"ok": False, "error": f"Re-plan failed: {replan_exc}"},
                ok=False,
            ))
            return None

    # ================================================================
    # 主循环
    # ================================================================
    def run(self, task: PublicTask) -> AgentRunResult:
        """Plan-and-Solve 主循环：规划 → 执行 → 交卷。"""
        state = AgentRuntimeState()
        cfg = self.config
        prog = cfg.progress
        tool_desc = self.tools.describe_for_prompt()

        # ---- Phase 1: 生成计划 ----
        if prog:
            progress("[plan-solve] Phase 1: generating plan ...")
        try:
            plan = self._generate_plan(task, tool_desc)
        except Exception as exc:
            if prog:
                progress(f"[plan-solve] Plan generation failed: {exc}, using fallback.")
            plan = ["List context files", "Inspect data", "Solve and call answer"]

        if prog:
            for i, s in enumerate(plan, 1):
                progress(f"[plan-solve]   {i}. {s}")

        # ---- Phase 2: 按计划执行 ----
        plan_idx = 0
        replan_used = 0
        consecutive_gate_blocks = 0
        _FALLBACK_STEP = "Call the answer tool with the final result table."

        for step_index in range(1, cfg.max_steps + 1):
            # 计划步骤越界兜底
            if plan_idx >= len(plan):
                if plan[-1] != _FALLBACK_STEP:
                    plan.append(_FALLBACK_STEP)
                plan_idx = len(plan) - 1

            if prog:
                progress(
                    f"[plan-solve] Step {step_index}/{cfg.max_steps} "
                    f"(plan {plan_idx+1}/{len(plan)}): calling model ..."
                )

            messages = self._build_exec_messages(task, tool_desc, plan, plan_idx, state)

            # 1) 模型调用（带重试）
            raw = self._call_model_with_retry(
                messages, step_index, state,
                prog=prog, timeout_s=cfg.model_timeout_s,
                max_retries=cfg.max_model_retries,
                retry_backoff=cfg.model_retry_backoff,
                tag="plan-solve",
            )
            if raw is None:
                continue

            # 2) 解析模型输出 + 执行
            try:
                model_step = parse_model_step(raw)
                if prog:
                    progress(
                        f"[plan-solve] Step {step_index}: "
                        f"action={model_step.action!r} "
                        f"input={preview_json(model_step.action_input)}"
                    )

                # 3) 门控检查
                gate_result = self._handle_gate_block(
                    task, model_step, raw, plan, plan_idx,
                    state, step_index, consecutive_gate_blocks, prog,
                )
                if gate_result > 0:
                    consecutive_gate_blocks = gate_result
                    continue
                consecutive_gate_blocks = 0

                # 4) 工具校验 + 执行
                tool_result = self._validate_and_execute_tool(
                    task, model_step, raw, state, step_index,
                    prog=prog, tool_timeout_s=cfg.tool_timeout_s, tag="plan-solve",
                )
                if tool_result is None:
                    continue

                # 5) 记录工具执行结果
                obs = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought=model_step.thought, action=model_step.action,
                    action_input=model_step.action_input, raw_response=raw,
                    observation=obs, ok=tool_result.ok,
                ))
                if prog:
                    term = " (terminal)" if tool_result.is_terminal else ""
                    progress(f"[plan-solve] Step {step_index}: ok={tool_result.ok}{term}")

                # 6) 终止型工具 → 交卷
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    if prog:
                        progress("[plan-solve] Submitted final answer.")
                    break

                # 7) 工具成功 → 推进计划
                if tool_result.ok:
                    plan_idx += 1

                # 8) 工具失败 → 尝试重规划
                if not tool_result.ok and replan_used < cfg.max_replans:
                    replan_used += 1
                    replan_result = self._handle_replan(
                        task, tool_desc, state, step_index, replan_used, prog,
                    )
                    if replan_result is not None:
                        plan, plan_idx = replan_result

            except Exception as exc:
                # 解析/工具异常 → 记录错误，让大模型下轮自行修正
                state.steps.append(StepRecord(
                    step_index=step_index, thought="",
                    action="__error__", action_input={},
                    raw_response=raw,
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                ))
                if prog:
                    progress(f"[plan-solve] Step {step_index}: error: {exc}")

        return self._finalize(task, state, prog, tag="plan-solve")
