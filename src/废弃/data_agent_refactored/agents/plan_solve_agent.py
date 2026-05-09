"""
Plan-and-Solve Agent — 两阶段智能体：先规划，后执行。

整体工作流程：
  阶段 1（规划）：通过 LLM 生成编号的分步计划，例如
      ["List context files", "Read CSV data", "Compute metric", "Call answer"]
  阶段 2（执行）：按照计划顺序逐步调用工具，每步都通过 LLM 决策调用哪个工具。
      失败时支持动态重规划（最多 max_replans 次）。

核心特性：
  - 继承自 BaseAgent，复用模型重试、工具校验/超时执行、结果打包等公共逻辑
  - 数据预览门控：强制要求 LLM 在执行代码/提交答案前先查看数据文件
  - 三级门控升级策略：提醒 → 改写计划步骤 → 强制注入 list_context
  - 上下文窗口管理：通过 context_manager 截断历史消息，避免超出 token 限制
"""
from __future__ import annotations  # 允许在类型注解中使用 "X | Y" 语法（Python 3.9 兼容）

import json
import logging
from typing import Any

# ---------- 共享常量（与 LangGraph backend 字符级一致） ----------
from data_agent_common.constants import FALLBACK_STEP_PROMPT

# ---------- 配置类 ----------
from data_agent_refactored.config import PlanAndSolveAgentConfig  # Agent 可调参数配置

# ---------- 基类与模型层 ----------
from data_agent_refactored.agents.base_agent import BaseAgent         # 抽象基类，提供共享的重试/工具执行/门控/打包逻辑
from data_agent_refactored.agents.model import ModelAdapter, ModelMessage  # LLM 适配器协议与消息数据结构

# ---------- 提示词模板 ----------
from data_agent_refactored.agents.prompt import (
    PLAN_AND_SOLVE_SYSTEM_PROMPT,  # Plan-and-Solve 智能体的系统提示词
    PLANNING_INSTRUCTION,          # 规划阶段的指令模板（告诉 LLM 如何输出计划）
    EXECUTION_INSTRUCTION,         # 执行阶段的指令模板（告诉 LLM 当前执行到哪一步）
    build_task_prompt,             # 构建任务描述的 user 消息
)

# ---------- 运行时状态与记录 ----------
from data_agent_refactored.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
# AgentRuntimeState: 运行过程中的可变状态（步骤列表、答案、失败原因）
# StepRecord:        单步不可变记录（思考、动作、观察结果等）
# AgentRunResult:    最终不可变结果报告

# ---------- JSON 解析 ----------
from data_agent_refactored.agents.json_parser import parse_model_step, parse_plan
# parse_model_step: 将 LLM 输出解析为 ModelStep(thought, action, action_input)
# parse_plan:       将 LLM 输出解析为计划步骤列表 list[str]

# ---------- 数据预览门控 ----------
from data_agent_refactored.agents.data_preview_gate import (
    GATED_ACTIONS,     # 受门控的工具集合：execute_python, execute_context_sql, answer
    GATE_REMINDER,     # 门控阻断时返回给 LLM 的提醒消息
    has_data_preview,  # 检查历史步骤中是否已有成功的数据预览
)

# ---------- 上下文窗口管理 ----------
from data_agent_refactored.agents.context_manager import build_history_messages
# 根据 token 预算截断历史消息，保留关键数据预览步骤，淘汰旧的非关键步骤

# ---------- 工具函数 ----------
from data_agent_refactored.agents.text_helpers import progress, preview_json
# progress:     实时打印进度信息（带 flush）
# preview_json: 将 dict 截断为简短的 JSON 字符串，用于日志输出

from data_agent_refactored.agents.timeout import call_with_timeout  # 同步调用超时包装器（守护线程实现）

# ---------- 外部模块 ----------
from data_agent_refactored.benchmark.schema import PublicTask   # 任务数据结构（task_id, question, context_dir）
from data_agent_refactored.tools.registry import ToolRegistry    # 工具注册表，提供工具描述和执行能力

logger = logging.getLogger(__name__)  # 模块级日志记录器


# ---------------------------------------------------------------------------
# 智能体
# ---------------------------------------------------------------------------

class PlanAndSolveAgent(BaseAgent):
    """两阶段智能体：全局规划，逐步执行。

    工作流程：
        1. _generate_plan()   → 生成计划列表 list[str]
        2. for 每一步:
           a. _build_execution_messages() → 组装包含当前计划步骤的消息
           b. _call_model_with_retry()    → 调用 LLM（继承自 BaseAgent）
           c. parse_model_step()          → 解析 JSON 得到 thought/action/action_input
           d. _handle_gate_block()        → 检查数据预览门控
           e. _validate_and_execute_tool() → 执行工具（继承自 BaseAgent）
           f. 根据结果决定：提交答案 / 推进计划 / 重规划
    """

    def __init__(
        self,
        *,
        model: ModelAdapter,   # LLM 适配器实例（如 OpenAIModelAdapter）
        tools: ToolRegistry,   # 工具注册表（提供可用工具的描述和执行）
        config: PlanAndSolveAgentConfig | None = None,  # 可选配置，为 None 时使用默认值
    ) -> None:
        super().__init__(model=model, tools=tools)  # 调用 BaseAgent.__init__，保存 model 和 tools
        self.config = config or PlanAndSolveAgentConfig()  # 未提供配置时使用默认参数

    # ================================================================
    # 阶段 1：规划
    # ================================================================

    def _generate_plan(
        self,
        task: PublicTask,          # 待解决的任务
        tool_description: str,     # 所有可用工具的文本描述（用于告知 LLM）
        history_hint: str = "",    # 重规划时传入的历史提示（告诉 LLM 哪些步骤已完成/失败）
    ) -> list[str]:
        """调用 LLM 为 *task* 生成编号计划。

        返回值：
            计划步骤描述列表，例如 ["List context files", "Read CSV", "Compute", "Answer"]

        异常：
            TimeoutError: LLM 调用超时
            ModelResponseParseError: LLM 输出无法解析为有效计划
        """
        # 构建 system 消息：角色设定 + 可用工具列表 + 输出格式要求
        system = (
            f"{PLAN_AND_SOLVE_SYSTEM_PROMPT}\n\n"
            f"Available tools:\n{tool_description}\n\n"
            "Return a single ```json fenced block."
        )
        # 构建 user 消息：任务描述 + 可选的历史提示 + 规划指令
        user = build_task_prompt(task) + "\n\n"
        if history_hint:
            # 重规划场景：将已完成/失败的步骤信息传给 LLM，避免重复执行
            user += f"Previous observations:\n{history_hint}\n\n"
        user += PLANNING_INSTRUCTION  # 附加规划格式要求（要求返回 {"plan": [...]} 格式）

        # 带超时保护地调用 LLM
        raw = call_with_timeout(
            self.model.complete,
            ([  # 消息列表：[system, user]
                ModelMessage(role="system", content=system),
                ModelMessage(role="user", content=user),
            ],),
            self.config.model_timeout_s,  # 超时秒数（默认 120s）
        )
        return parse_plan(raw)  # 解析 LLM 输出为计划列表（多级容错）

    # ================================================================
    # 阶段 2：执行阶段消息组装
    # ================================================================

    def _build_execution_messages(
        self,
        task: PublicTask,            # 当前任务
        tool_description: str,       # 可用工具的文本描述
        plan: list[str],             # 当前计划列表
        plan_index: int,             # 当前执行到计划的第几步（从 0 开始）
        state: AgentRuntimeState,    # 当前运行时状态（包含历史步骤）
    ) -> list[ModelMessage]:
        """组装执行阶段的消息列表（含截断历史）。

        消息结构：
            [system] 角色设定 + 工具列表 + 格式要求
            [user]   任务描述 + 当前计划步骤信息
            [assistant/user 交替]  截断后的历史对话（由 build_history_messages 生成）

        返回值：
            完整的消息列表，可直接传给 LLM
        """
        # 将计划列表格式化为编号文本，例如：
        #   1. List context files
        #   2. Read CSV data
        #   3. Compute metric
        numbered = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan))

        # 填充执行指令模板，告诉 LLM 当前在执行计划的第几步、该步的描述
        exec_block = EXECUTION_INSTRUCTION.format(
            numbered_plan=numbered,              # 完整计划文本
            current=plan_index + 1,              # 当前步骤编号（1-based）
            total=len(plan),                     # 总步骤数
            step_description=plan[plan_index],   # 当前步骤描述
        )

        # system 消息：角色设定 + 可用工具 + 输出格式要求
        system = (
            f"{PLAN_AND_SOLVE_SYSTEM_PROMPT}\n\n"
            f"Available tools:\n{tool_description}\n\n"
            "Return a single ```json fenced block with keys "
            "`thought`, `action`, `action_input`."
        )

        # 基础消息：system + user（任务 + 执行指令）
        base_messages: list[ModelMessage] = [
            ModelMessage(role="system", content=system),
            ModelMessage(
                role="user",
                content=f"{build_task_prompt(task)}\n\n{exec_block}",
            ),
        ]

        # 在基础消息后附加截断后的历史对话（assistant/user 交替）
        # build_history_messages 会根据 token 预算进行两层截断：
        #   L1: 单条观察内容截断至 max_obs_chars
        #   L2: 窗口淘汰，保留数据预览步骤，FIFO 淘汰旧的非关键步骤
        return build_history_messages(
            state,
            base_messages,
            max_obs_chars=self.config.max_obs_chars,          # 单条观察字符上限（默认 3000）
            max_context_tokens=self.config.max_context_tokens, # 历史窗口 token 预算（默认 24000）
        )

    # ================================================================
    # 门控：分层升级
    # ================================================================

    def _handle_gate_block(
        self,
        task: PublicTask,                  # 当前任务
        model_step: Any,                   # LLM 解析出的动作（含 thought/action/action_input）
        raw: str,                          # LLM 的原始输出文本
        plan: list[str],                   # 当前计划列表（可能被修改）
        plan_index: int,                   # 当前计划步骤索引
        state: AgentRuntimeState,          # 运行时状态
        step_index: int,                   # 当前执行步编号（1-based）
        consecutive_gate_blocks: int,      # 连续被门控阻断的次数
        show_progress: bool,               # 是否打印进度
        plan_progress: str = "",           # 执行进度，如 "2/5"
        plan_step_description: str = "",   # 当前计划步骤描述
    ) -> int:
        """处理数据预览门控阻断。

        门控机制的目的：防止 LLM 在未查看数据的情况下直接执行代码或提交答案。
        当 LLM 试图调用受门控工具（execute_python/execute_context_sql/answer）
        但尚未成功执行过任何数据预览工具时，触发门控。

        三级升级策略（随连续阻断次数递增）：
            L1 (低级):  返回提醒消息 GATE_REMINDER，让 LLM 自行调整
            L2 (中级):  直接改写当前计划步骤为“强制检查数据”
            L3 (高级):  绕过 LLM，直接强制执行 list_context 工具

        返回值：
            更新后的连续阻断计数器。
            返回 0 表示未被阻断，调用方应继续执行工具。
            返回 >0 表示被阻断，调用方应 continue 跳过当前步。
        """
        cfg = self.config

        # 判断是否需要门控阻断：三个条件同时满足才触发
        #   1. 配置中启用了门控 (require_data_preview_before_compute=True)
        #   2. LLM 要调用的工具属于受门控工具 (GATED_ACTIONS)
        #   3. 历史步骤中还没有成功的数据预览
        if not (
            cfg.require_data_preview_before_compute
            and model_step.action in GATED_ACTIONS
            and not has_data_preview(state)
        ):
            return 0  # 未被阻断，可以继续执行工具

        # --- 触发门控阻断 ---
        consecutive_gate_blocks += 1

        # 构建阻断观察结果，将提醒消息作为错误返回给 LLM
        observation: dict[str, Any] = {
            "ok": False,
            "tool": model_step.action,
            "error": GATE_REMINDER,  # 提醒 LLM 先调用数据检查工具
        }
        # 记录这次被阻断的步骤
        state.steps.append(StepRecord(
            step_index=step_index,
            thought=model_step.thought,
            action=model_step.action,
            action_input=model_step.action_input,
            raw_response=raw,
            observation=observation,
            ok=False,  # 门控阻断视为失败
            phase="execution",
            plan_progress=plan_progress,
            plan_step_description=plan_step_description,
        ))

        # ===== 三级分层升级 =====
        if consecutive_gate_blocks >= cfg.max_gate_retries + 2:
            # --- L3 最高级别：强制注入 list_context ---
            # LLM 多次被提醒仍拒绝查看数据，直接绕过 LLM 执行工具
            if show_progress:
                progress(
                    f"[plan-solve] Step {step_index}: gate escalation — "
                    f"force-injecting list_context"
                )
            try:
                # 直接调用 list_context 工具，不经过 LLM 决策
                forced_result = call_with_timeout(
                    self.tools.execute,
                    (task, "list_context", {"max_depth": 4}),
                    cfg.tool_timeout_s,
                )
                # 记录强制注入的步骤，让 LLM 在后续对话中能看到文件列表
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="[auto-injected by gate escalation]",
                    action="list_context",
                    action_input={"max_depth": 4},
                    raw_response="",
                    observation={
                        "ok": forced_result.ok,
                        "tool": "list_context",
                        "content": forced_result.content,
                    },
                    ok=forced_result.ok,
                    phase="execution",
                    plan_progress=plan_progress,
                    plan_step_description=plan_step_description,
                ))
            except Exception as forced_exc:
                # 强制注入也失败了，记录错误但不中止流程
                logger.warning(
                    "[plan-solve] Force-injected list_context failed: %s", forced_exc
                )
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="[auto-injected by gate escalation]",
                    action="list_context",
                    action_input={"max_depth": 4},
                    raw_response="",
                    observation={"ok": False, "error": str(forced_exc)},
                    ok=False,
                    phase="execution",
                    plan_progress=plan_progress,
                    plan_step_description=plan_step_description,
                ))
            consecutive_gate_blocks = 0  # 重置计数器，避免无限循环

        elif consecutive_gate_blocks >= cfg.max_gate_retries:
            # --- L2 中级：改写当前计划步骤 ---
            # 将当前计划步骤替换为强制数据检查指令，引导 LLM 优先查看数据
            plan[plan_index] = (
                "MANDATORY: Inspect data files first by calling "
                "read_csv, read_json, read_doc, or inspect_sqlite_schema "
                "before any compute or answer action."
            )
            if show_progress:
                progress(
                    f"[plan-solve] Step {step_index}: gate blocked "
                    f"{consecutive_gate_blocks}x, plan step overridden"
                )
        elif show_progress:
            # --- L1 低级：仅打印提醒 ---
            # GATE_REMINDER 已经作为观察结果返回给 LLM，等待其自行调整
            progress(f"[plan-solve] Step {step_index}: blocked (preview gate)")

        return consecutive_gate_blocks  # 返回更新后的计数器，供主循环跟踪

    # ================================================================
    # 重规划
    # ================================================================

    def _handle_replan(
        self,
        task: PublicTask,              # 当前任务
        tool_description: str,         # 可用工具描述文本
        state: AgentRuntimeState,      # 当前运行时状态（包含所有历史步骤）
        step_index: int,               # 当前执行步编号
        replan_used: int,              # 已使用的重规划次数（用于日志）
        show_progress: bool,           # 是否打印进度
    ) -> tuple[list[str], int] | None:
        """工具执行失败后尝试重规划。

        重规划流程：
            1. 汇总已完成的步骤和最后失败的步骤，作为历史提示
            2. 调用 _generate_plan() 让 LLM 生成新计划（跳过已完成步骤）
            3. 若重规划成功，返回新计划并从第 0 步开始执行

        返回值：
            成功：(new_plan, 0) — 新计划列表和重置后的 plan_index
            失败：None — 重规划失败，继续使用原计划
        """
        if show_progress:
            progress("[plan-solve] Tool failed, re-planning ...")

        # --- 构建历史提示，告诉 LLM 哪些步骤已完成、哪个步骤失败了 ---
        completed: list[str] = []     # 已成功的步骤摘要
        failed_last: list[str] = []   # 最后一个失败步骤的摘要
        for step in state.steps:
            # 截取观察结果的前 200 字符作为摘要
            preview = json.dumps(step.observation, ensure_ascii=False)[:200]
            if step.ok:
                # 成功步骤：标记为 ✓，包含工具名、输入参数摘要、结果摘要
                completed.append(
                    f"  ✓ {step.action}({preview_json(step.action_input, 80)}) → {preview}"
                )
            elif step == state.steps[-1]:
                # 仅记录最后一个失败步骤（即触发重规划的那个）
                failed_last.append(
                    f"  ✗ {step.action}({preview_json(step.action_input, 80)}) → {preview}"
                )

        # 组装历史提示文本
        history_parts: list[str] = []
        if completed:
            # 告诉 LLM 这些步骤已经完成，不要重复
            history_parts.append(
                "Already completed (do NOT repeat):\n" + "\n".join(completed)
            )
        if failed_last:
            # 告诉 LLM 最后哪个步骤失败了
            history_parts.append("Last failed action:\n" + "\n".join(failed_last))
        # 强调新计划必须跳过已成功步骤
        history_parts.append(
            "IMPORTANT: Your new plan must skip steps that have already succeeded. "
            "Start from the next logical action."
        )
        history_hint = "\n\n".join(history_parts)

        # --- 调用 _generate_plan 生成新计划 ---
        try:
            new_plan = self._generate_plan(task, tool_description, history_hint=history_hint)
            if show_progress:
                progress("[plan-solve] New plan:")
                for i, step_desc in enumerate(new_plan, 1):
                    progress(f"[plan-solve]   {i}. {step_desc}")
            return (new_plan, 0)  # 返回新计划，plan_index 重置为 0
        except Exception as replan_exc:
            # 重规划失败（如 LLM 超时或解析失败），继续使用原计划
            logger.warning(
                "[plan-solve] Re-plan failed (attempt %d/%d): %s",
                replan_used, self.config.max_replans, replan_exc,
            )
            if show_progress:
                progress(
                    f"[plan-solve] Re-plan failed: {replan_exc!r}, "
                    f"continuing with current plan."
                )
            # 记录重规划失败的步骤（step_index=-1 表示这不是常规执行步）
            state.steps.append(StepRecord(
                step_index=-1,
                thought="",
                action="__replan_failed__",  # 特殊标记，context_manager 会识别并清理
                action_input={},
                raw_response="",
                observation={"ok": False, "error": f"Re-plan failed: {replan_exc}"},
                ok=False,
                phase="execution",
            ))
            return None

    # ================================================================
    # 主循环
    # ================================================================

    def run(self, task: PublicTask) -> AgentRunResult:
        """执行 Plan-and-Solve 循环以完成 *task*。

        完整流程：
            Phase 1: 调用 LLM 生成计划 → 失败则使用兆底计划
            Phase 2: 按计划逐步执行，每步都经过：
                模型调用 → JSON 解析 → 门控检查 → 工具执行 → 结果处理

        参数：
            task: 待解决的任务（包含 task_id, question, context_dir）

        返回值：
            AgentRunResult: 包含答案、执行步骤、失败原因的最终结果
        """
        state = AgentRuntimeState()  # 初始化运行时状态（空步骤列表、无答案、无失败原因）
        cfg = self.config
        show_progress = cfg.progress            # 是否打印实时进度
        tool_description = self.tools.describe_for_prompt()  # 获取所有工具的文本描述

        # ===============================================================
        # 阶段 1：生成计划
        # 调用 LLM 生成一个分步计划。如果失败（超时/解析错误），
        # 使用硬编码的兆底计划以保证流程不中断。
        # ===============================================================
        if show_progress:
            progress("[plan-solve] Phase 1: generating plan ...")
        plan_ok = True
        try:
            plan = self._generate_plan(task, tool_description)
        except Exception as exc:
            logger.warning("[plan-solve] Plan generation failed: %s — using fallback.", exc)
            if show_progress:
                progress(f"[plan-solve] Plan generation failed: {exc}, using fallback.")
            # 兆底计划：最基本的三步流程
            plan = ["List context files", "Inspect data", "Solve and call answer"]
            plan_ok = False

        # 记录规划阶段结果
        state.steps.append(StepRecord(
            step_index=0,
            thought="",
            action="__plan_generation__",
            action_input={},
            raw_response="",
            observation={
                "ok": plan_ok,
                "plan": plan,
            },
            ok=plan_ok,
            phase="planning",
        ))

        # 打印生成的计划
        if show_progress:
            for i, step_desc in enumerate(plan, 1):
                progress(f"[plan-solve]   {i}. {step_desc}")

        # ===============================================================
        # 阶段 2：执行计划
        # 按照计划顺序逐步执行，每步通过 LLM 决定调用哪个工具。
        # 工具成功则推进 plan_index，失败则尝试重规划。
        # ===============================================================
        plan_index = 0               # 当前执行到计划的第几步
        replan_used = 0              # 已使用的重规划次数（上限为 cfg.max_replans）
        consecutive_gate_blocks = 0  # 连续门控阻断计数器（用于分层升级）

        # ---------- 主执行循环 ----------
        for step_index in range(1, cfg.max_steps + 1):
            # 如果 plan_index 已超出计划长度，追加兆底步骤并停留在最后一步
            # 这确保即使计划执行完了，Agent 仍会尝试提交答案
            if plan_index >= len(plan):
                if plan[-1] != FALLBACK_STEP_PROMPT:
                    plan.append(FALLBACK_STEP_PROMPT)
                plan_index = len(plan) - 1

            # --- 计算当前执行进度与计划步骤描述 ---
            cur_progress = f"{plan_index + 1}/{len(plan)}"
            cur_step_desc = plan[plan_index]

            if show_progress:
                progress(
                    f"[plan-solve] Step {step_index}/{cfg.max_steps} "
                    f"(plan {cur_progress}): calling model ..."
                )

            # 组装消息：system + task + 当前计划步骤 + 截断历史
            messages = self._build_execution_messages(
                task, tool_description, plan, plan_index, state
            )

            # 1) 模型调用（带指数退避重试）
            # 返回 LLM 的原始文本输出，失败返回 None
            raw = self._call_model_with_retry(
                messages, step_index, state,
                show_progress=show_progress,
                timeout_seconds=cfg.model_timeout_s,     # 单次调用超时（默认 120s）
                max_retries=cfg.max_model_retries,       # 最大重试次数（默认 3）
                retry_backoff=cfg.model_retry_backoff,   # 退避间隔序列（默认 2s, 5s, 10s）
                tag="plan-solve",                        # 日志标签
                phase="execution",
                plan_progress=cur_progress,
                plan_step_description=cur_step_desc,
            )
            if raw is None:
                # 所有重试均失败，跳过这一步，进入下一步循环
                continue

            # 2) 解析 LLM 输出 + 执行工具
            try:
                # 将 LLM 的 JSON 输出解析为 ModelStep(thought, action, action_input)
                # 支持多级容错：严格 JSON → 括号修复 → json_repair 自动修复
                model_step = parse_model_step(raw)
                if show_progress:
                    progress(
                        f"[plan-solve] Step {step_index}: "
                        f"action={model_step.action!r} "
                        f"input={preview_json(model_step.action_input)}"
                    )

                # 3) 门控检查
                # 检查 LLM 是否在未预览数据的情况下试图执行受门控工具
                # 返回 0 = 未阻断，>0 = 被阻断（计数器值）
                gate_result = self._handle_gate_block(
                    task, model_step, raw, plan, plan_index,
                    state, step_index, consecutive_gate_blocks, show_progress,
                    plan_progress=cur_progress,
                    plan_step_description=cur_step_desc,
                )
                if gate_result > 0:
                    # 被门控阻断，更新连续阻断计数器，跳过这一步
                    consecutive_gate_blocks = gate_result
                    continue
                consecutive_gate_blocks = 0  # 未被阻断，重置计数器

                # 4) 工具校验 + 执行
                # 先校验工具名是否存在，然后带超时保护执行工具
                # 校验失败或超时返回 None（已内部记录错误 StepRecord）
                tool_result = self._validate_and_execute_tool(
                    task, model_step, raw, state, step_index,
                    show_progress=show_progress,
                    tool_timeout_seconds=cfg.tool_timeout_s,  # 工具执行超时（默认 180s）
                    tag="plan-solve",
                    phase="execution",
                    plan_progress=cur_progress,
                    plan_step_description=cur_step_desc,
                )
                if tool_result is None:
                    # 工具名无效或执行超时，跳过这一步
                    continue

                # 5) 记录工具执行结果
                observation: dict[str, Any] = {
                    "ok": tool_result.ok,           # 工具执行是否成功
                    "tool": model_step.action,       # 工具名称
                    "content": tool_result.content,   # 工具返回内容
                }
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw,
                    observation=observation,
                    ok=tool_result.ok,
                    phase="execution",
                    plan_progress=cur_progress,
                    plan_step_description=cur_step_desc,
                ))
                if show_progress:
                    term = " (terminal)" if tool_result.is_terminal else ""
                    progress(f"[plan-solve] Step {step_index}: ok={tool_result.ok}{term}")

                # 6) 终止工具 → 提交答案
                # 当 LLM 调用 answer 工具时，tool_result.is_terminal=True
                # 此时将答案保存到 state 并结束循环
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    if show_progress:
                        progress("[plan-solve] Submitted final answer.")
                    break

                # 7) 成功 → 推进计划
                # 工具执行成功，前进到计划的下一步
                if tool_result.ok:
                    plan_index += 1

                # 8) 失败 → 尝试重规划
                # 工具执行失败且还有重规划配额时，调用 _handle_replan
                # 生成新计划（跳过已完成步骤）
                if not tool_result.ok and replan_used < cfg.max_replans:
                    replan_used += 1
                    replan_result = self._handle_replan(
                        task, tool_description, state, step_index, replan_used,
                        show_progress,
                    )
                    if replan_result is not None:
                        # 重规划成功，替换计划并重置 plan_index
                        plan, plan_index = replan_result

            except Exception as exc:
                # 全局异常捕获：JSON 解析失败等未预期错误
                # 记录错误但不中止流程，继续下一步循环
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="",
                    action="__error__",      # 特殊标记，context_manager 会识别并清理
                    action_input={},
                    raw_response=raw,
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                    phase="execution",
                    plan_progress=cur_progress,
                    plan_step_description=cur_step_desc,
                ))
                logger.warning("[plan-solve] Step %d: error: %s", step_index, exc)
                if show_progress:
                    progress(f"[plan-solve] Step {step_index}: error: {exc}")

        # 打包最终结果：如果循环结束时仍未提交答案，_finalize 会设置 failure_reason
        return self._finalize(task, state, show_progress, tag="plan-solve")
