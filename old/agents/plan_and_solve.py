from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass

import json_repair

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import build_observation_prompt, build_task_prompt
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


# =====================================================================
# Plan-and-Solve 提示词
# =====================================================================
# 与 ReAct 的核心区别：
#   ReAct  = 走一步看一步（Thought → Action → Observation 循环）
#   P&S    = 先规划全局，再按计划执行，执行中可动态重规划

PLAN_AND_SOLVE_SYSTEM_PROMPT = """
You are a Plan-and-Solve data agent that works in two phases:

Phase 1 – Planning: Analyze the task and devise a numbered step-by-step plan.
Phase 2 – Execution: Carry out each step by calling the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information observed through provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object wrapped in a ```json fenced block.
6. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()

# 规划阶段指令：要求大模型输出步骤列表
_PLANNING_INSTRUCTION = """
Create a step-by-step plan to solve this task.
Return a JSON object with a single key "plan" — a list of concise step descriptions.
The plan should start with inspecting data files and end with calling the `answer` tool.

Example:
```json
{"plan": ["List context files", "Read the CSV data to understand schema", "Compute the required metric with Python", "Call answer tool with the result table"]}
```
""".strip()

# 执行阶段指令模板：在每轮对话中注入当前计划和进度
_EXECUTION_INSTRUCTION = """
=== Your Plan ===
{numbered_plan}

You are on step {current}/{total}: "{step_description}"

Return a JSON with keys `thought`, `action`, `action_input`.
If this step's goal is achieved, say so in your thought and proceed to the next step's action.

```json
{{"thought":"...","action":"tool_name","action_input":{{...}}}}
```
""".strip()


# =====================================================================
# 数据预览门控（与 ReAct 版相同）
# =====================================================================
_DATA_PREVIEW_ACTIONS = frozenset({"read_csv", "read_json", "read_doc", "inspect_sqlite_schema"})
_GATED_ACTIONS = frozenset({"execute_python", "execute_context_sql", "answer"})
_GATE_REMINDER = (
    "Blocked: inspect data first. Call read_csv / read_json / read_doc / "
    "inspect_sqlite_schema before using execute_python, execute_context_sql, or answer."
)

# __error__ 步骤的消毒替换模板：格式正确的 JSON，引导模型自纠正
_ERROR_SANITIZED_RESPONSE = json.dumps(
    {
        "thought": "My previous response had a format error. "
                   "I must return a valid JSON with keys: thought, action, action_input.",
        "action": "__error__",
        "action_input": {},
    },
    ensure_ascii=False,
)


def _has_data_preview(state: AgentRuntimeState) -> bool:
    """历史中是否至少成功执行过一次数据预览工具。"""
    return any(s.ok and s.action in _DATA_PREVIEW_ACTIONS for s in state.steps)


# =====================================================================
# 辅助函数
# =====================================================================
def _progress(msg: str) -> None:
    """实时打印进度（flush 防止多进程延迟）。"""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 超时调用包装器（基于独立 daemon 线程，避免线程池资源耗尽导致死锁）
# ---------------------------------------------------------------------------
def _call_with_timeout(fn, args: tuple, timeout_s: float):
    """带超时控制的同步调用包装器。

    为每次调用创建一个独立的 daemon 线程，避免使用固定大小的线程池。
    超时后主调方立即抛出 TimeoutError，底层 daemon 线程会在后台自然结束
    或在进程退出时被回收，不会阻塞其他调用。
    """
    result_container: list = [None]
    exception_container: list[BaseException | None] = [None]

    def _wrapper():
        try:
            result_container[0] = fn(*args)
        except Exception as e:
            exception_container[0] = e

    t = threading.Thread(target=_wrapper, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise TimeoutError(
            f"{getattr(fn, '__qualname__', str(fn))} timed out after {timeout_s}s"
        )
    if exception_container[0] is not None:
        raise exception_container[0]
    return result_container[0]


def _estimate_tokens(text: str) -> int:
    """粗估 token 数。中英混合场景下 1 token ≈ 3 字符，偏保守以留安全余量。"""
    return len(text) // 3 + 1


def _truncate_observation(observation: dict, max_chars: int) -> dict:
    """对 observation 中的 content 字段做硬截断，防止单条 observation 过大。

    截断逻辑：
      - 仅截断 'content' 字段（CSV 预览、SQL 结果等大文本的载体）
      - 截断后追加提示标记，让大模型知道数据被裁剪了
      - ok / tool / error 等元字段保持原样
    """
    content = observation.get("content")
    if not isinstance(content, str) or len(content) <= max_chars:
        return observation
    truncated = dict(observation)  # 浅拷贝，避免污染原始 state
    truncated["content"] = (
        content[:max_chars]
        + f"\n... [truncated, showing first {max_chars} chars of {len(content)}]"
    )
    return truncated


def _preview_json(obj: object, max_len: int = 140) -> str:
    """将对象转为截断的 JSON 预览字符串。"""
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        text = str(obj)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


# =====================================================================
# JSON 解析工具（多层降级：严格 → 修复括号 → json_repair）
# =====================================================================
def _strip_json_fence(raw: str) -> str:
    """去除 Markdown ```json ... ``` 代码块标记。"""
    text = raw.strip()
    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


def _fix_trailing_bracket(text: str) -> str:
    """修复 JSON 结尾 `}]}` → `}}`。"""
    return re.sub(r"\}\s*\]\s*\}\s*$", "}}", text.strip(), count=1)


def _try_strict_json(text: str) -> dict | None:
    """严格解析 JSON，失败返回 None。"""
    try:
        payload, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    remainder = re.sub(r"(?:\\[nrt])+", "", text[end:].strip()).strip()
    if remainder:
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_object(text: str) -> dict:
    """多层降级 JSON 解析。"""
    text = text.strip()
    for variant in (text, _fix_trailing_bracket(text)):
        result = _try_strict_json(variant)
        if result is not None:
            return result
    last_exc: BaseException | None = None
    for variant in (text, _fix_trailing_bracket(text)):
        try:
            repaired = json_repair.loads(variant)
        except Exception as exc:
            last_exc = exc
            continue
        if isinstance(repaired, dict):
            return repaired
    raise ValueError("Model response is not valid JSON.") from last_exc


# =====================================================================
# 模型输出解析
# =====================================================================
def _parse_step(raw: str) -> ModelStep:
    """解析大模型输出为 ModelStep（thought / action / action_input）。"""
    payload = _load_json_object(_strip_json_fence(raw))
    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")
    return ModelStep(thought=thought, action=action,
                     action_input=action_input, raw_response=raw)


def _parse_plan(raw: str) -> list[str]:
    """解析大模型输出的计划列表。"""
    payload = _load_json_object(_strip_json_fence(raw))
    plan = payload.get("plan", [])
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must be a non-empty list of step descriptions.")
    return [str(s) for s in plan]


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
    max_gate_retries: int = 2                          # 门控连续拦截次数上限，超过后强制改写计划步骤
    max_model_retries: int = 3                          # 模型 API 瞬态错误最大重试次数
    model_retry_backoff: tuple[float, ...] = (2.0, 5.0, 10.0)  # 重试退避秒数序列


# =====================================================================
# Plan-and-Solve Agent 核心类
# =====================================================================
class PlanAndSolveAgent:
    """
    两阶段智能体：
      Phase 1 — 让大模型根据任务生成一份分步计划
      Phase 2 — 按计划逐步执行工具调用，每步都能看到完整计划和当前进度

    与 ReAct 的核心区别：
      ReAct 每一步都是即兴的；PlanAndSolve 有全局计划作为执行指引，
      大模型始终知道"接下来该做什么"，减少迷路和重复探索。
    """

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: PlanAndSolveAgentConfig | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or PlanAndSolveAgentConfig()

    # -------------------- Phase 1: 规划 --------------------
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
        user += _PLANNING_INSTRUCTION

        raw = _call_with_timeout(
            self.model.complete,
            ([ModelMessage(role="system", content=system),
              ModelMessage(role="user", content=user)],),
            self.config.model_timeout_s,
        )
        return _parse_plan(raw)

    # -------------------- Phase 2: 构建执行消息 --------------------
    def _build_exec_messages(
        self,
        task: PublicTask,
        tool_desc: str,
        plan: list[str],
        plan_idx: int,
        state: AgentRuntimeState,
    ) -> list[ModelMessage]:
        """组装执行阶段的对话：系统设定 + 任务 + 计划进度 + 历史观测。"""
        # 格式化计划列表
        numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan))
        exec_block = _EXECUTION_INSTRUCTION.format(
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

        messages: list[ModelMessage] = [
            ModelMessage(role="system", content=system),
            ModelMessage(role="user", content=f"{build_task_prompt(task)}\n\n{exec_block}"),
        ]
        # ----- 回放历史 observation（加入双层截断防护） -----
        cfg = self.config

        # L1: 预处理每一步的消息对（assistant + user），同时：
        #     - 截断过大的 observation
        #     - 对 __error__ / __replan_failed__ 步骤做消毒替换，防止非法 JSON 污染模型推理
        _SKIP_ACTIONS = frozenset({"__error__", "__replan_failed__"})
        history_pairs: list[tuple[str, str]] = []
        for step in state.steps:
            obs_safe = _truncate_observation(step.observation, cfg.max_obs_chars)
            # 消毒：__error__ / __replan_failed__ 步骤的 raw_response 是非法 JSON 或空字符串，替换为合法占位符
            if step.action in _SKIP_ACTIONS:
                assistant_text = _ERROR_SANITIZED_RESPONSE
            else:
                assistant_text = step.raw_response
            user_text = build_observation_prompt(obs_safe)
            history_pairs.append((assistant_text, user_text))

        # L2: 钉住关键步骤 + 滑动窗口淘汰
        # 将历史步骤分为"钉住"（pinned）和"可淘汰"（evictable）两组：
        #   - 成功的数据预览步骤（schema/数据结构信息）永远保留，防止模型"失忆"
        #   - 其余步骤按 FIFO 淘汰
        base_tokens = sum(_estimate_tokens(m.content) for m in messages)
        budget = cfg.max_context_tokens

        # 隐患5防护: 当 base prompt 自身就超预算时，记录警告并留出最小历史余量
        if base_tokens >= budget:
            logging.warning(
                "[plan-solve] Base prompt (%d tokens) exceeds budget (%d). "
                "Allowing minimal history headroom.",
                base_tokens, budget,
            )
            budget = base_tokens + cfg.max_obs_chars  # 至少留出一条 observation 的空间

        pair_tokens = [
            _estimate_tokens(a) + _estimate_tokens(u) for a, u in history_pairs
        ]

        # 分区：pinned（成功的数据预览）vs evictable（其余步骤）
        pinned_indices: list[int] = []
        evictable_indices: list[int] = []
        for i, step in enumerate(state.steps):
            if i >= len(history_pairs):
                break
            if step.ok and step.action in _DATA_PREVIEW_ACTIONS:
                pinned_indices.append(i)
            else:
                evictable_indices.append(i)

        pinned_tokens = sum(pair_tokens[i] for i in pinned_indices)
        remaining_budget = budget - base_tokens - pinned_tokens

        # 对 evictable 步骤做 FIFO 淘汰（从最旧的开始丢弃）
        evictable_total = sum(pair_tokens[i] for i in evictable_indices)
        evict_start = 0
        while evict_start < len(evictable_indices) and evictable_total > remaining_budget:
            evictable_total -= pair_tokens[evictable_indices[evict_start]]
            evict_start += 1

        kept_evictable = set(evictable_indices[evict_start:])
        kept_pinned = set(pinned_indices)
        n_omitted = evict_start  # 被淘汰的步骤数

        # 如果有步骤被淘汰，插入一条摘要占位消息
        if n_omitted > 0:
            omitted_summary = (
                f"[Note: {n_omitted} earlier step(s) omitted to fit context window. "
                f"Key data-preview steps are preserved. "
                f"Recent {len(kept_evictable) + len(kept_pinned)} step(s) are shown below.]"
            )
            messages.append(ModelMessage(role="user", content=omitted_summary))

        # 按原始顺序拼接保留的步骤（pinned + 保留的 evictable），维持时序一致性
        for i, (assistant_text, user_text) in enumerate(history_pairs):
            if i in kept_pinned or i in kept_evictable:
                messages.append(ModelMessage(role="assistant", content=assistant_text))
                messages.append(ModelMessage(role="user", content=user_text))

        return messages

    # ================================================================
    # 子方法：模型调用（带指数退避重试）
    # ================================================================
    def _call_model_with_retry(
        self, messages: list[ModelMessage], step_index: int,
        state: AgentRuntimeState, prog: bool,
    ) -> str | None:
        """调用模型 API，失败时进行指数退避重试。

        返回:
            模型原始输出文本；若所有重试耗尽则返回 None 并将错误记录到 state。
        """
        cfg = self.config
        for attempt in range(cfg.max_model_retries):
            try:
                return _call_with_timeout(
                    self.model.complete, (messages,), cfg.model_timeout_s,
                )
            except (TimeoutError, RuntimeError) as exc:
                # RuntimeError 包含 429 限流、502 网关等可恢复的 API 错误
                if attempt < cfg.max_model_retries - 1:
                    backoff = cfg.model_retry_backoff[min(attempt, len(cfg.model_retry_backoff) - 1)]
                    if prog:
                        _progress(
                            f"[plan-solve] Step {step_index}: model call failed "
                            f"(attempt {attempt+1}/{cfg.max_model_retries}): {exc}, "
                            f"retrying in {backoff}s ..."
                        )
                    time.sleep(backoff)
                else:
                    # 所有重试耗尽，记录错误步骤
                    state.steps.append(StepRecord(
                        step_index=step_index, thought="",
                        action="__error__", action_input={}, raw_response="",
                        observation={"ok": False, "error": f"Model call failed after {cfg.max_model_retries} attempts: {exc}"},
                        ok=False,
                    ))
                    if prog:
                        _progress(f"[plan-solve] Step {step_index}: model call failed after all retries")
        return None

    # ================================================================
    # 子方法：门控判定与分级升级
    # ================================================================
    def _handle_gate_block(
        self, task: PublicTask, model_step: ModelStep, raw: str,
        plan: list[str], plan_idx: int,
        state: AgentRuntimeState, step_index: int,
        consecutive_gate_blocks: int, prog: bool,
    ) -> int:
        """处理数据预览门控拦截，返回更新后的 consecutive_gate_blocks。

        调用方在返回值 > 0 时应 continue 跳过当前步。
        返回 0 表示已通过门控（不需要拦截）。
        """
        cfg = self.config

        # 不需要门控检查的情况：直接放行
        if not (
            cfg.require_data_preview_before_compute
            and model_step.action in _GATED_ACTIONS
            and not _has_data_preview(state)
        ):
            return 0  # 放行信号

        # ---- 触发门控拦截 ----
        consecutive_gate_blocks += 1
        obs = {"ok": False, "tool": model_step.action, "error": _GATE_REMINDER}
        state.steps.append(StepRecord(
            step_index=step_index,
            thought=model_step.thought, action=model_step.action,
            action_input=model_step.action_input, raw_response=raw,
            observation=obs, ok=False,
        ))

        # 分级处理
        if consecutive_gate_blocks >= cfg.max_gate_retries + 2:
            # 升级措施：模型持续忽视指令，直接强制注入 list_context 绕过模型
            if prog:
                _progress(
                    f"[plan-solve] Step {step_index}: gate escalation — "
                    f"force-injecting list_context"
                )
            try:
                forced_result = _call_with_timeout(
                    self.tools.execute,
                    (task, "list_context", {"max_depth": 4}),
                    cfg.tool_timeout_s,
                )
                state.steps.append(StepRecord(
                    step_index=step_index,
                    thought="[auto-injected by gate escalation]",
                    action="list_context", action_input={"max_depth": 4},
                    raw_response="",
                    observation={"ok": forced_result.ok, "tool": "list_context", "content": forced_result.content},
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
            plan[plan_idx] = (
                "MANDATORY: Inspect data files first by calling "
                "read_csv, read_json, read_doc, or inspect_sqlite_schema "
                "before any compute or answer action."
            )
            if prog:
                _progress(
                    f"[plan-solve] Step {step_index}: gate blocked "
                    f"{consecutive_gate_blocks}x, plan step overridden to force data preview"
                )
        elif prog:
            _progress(f"[plan-solve] Step {step_index}: blocked (preview gate)")

        return consecutive_gate_blocks

    # ================================================================
    # 子方法：工具名校验 + 带超时执行
    # ================================================================
    def _validate_and_execute_tool(
        self, task: PublicTask, model_step: ModelStep, raw: str,
        state: AgentRuntimeState, step_index: int, prog: bool,
    ) -> object | None:
        """校验工具名并执行工具调用。

        返回:
            ToolExecutionResult 对象（成功或工具自身报错）；
            None 表示校验失败或超时，已自行写入 state，调用方应 continue。
        """
        cfg = self.config

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
                _progress(f"[plan-solve] Step {step_index}: unknown tool '{model_step.action}'")
            return None

        # --- 执行工具（带超时保护） ---
        try:
            return _call_with_timeout(
                self.tools.execute,
                (task, model_step.action, model_step.action_input),
                cfg.tool_timeout_s,
            )
        except TimeoutError:
            obs = {
                "ok": False,
                "tool": model_step.action,
                "error": f"Tool '{model_step.action}' timed out after {cfg.tool_timeout_s}s. "
                         f"Try simplifying your code or query.",
            }
            state.steps.append(StepRecord(
                step_index=step_index,
                thought=model_step.thought, action=model_step.action,
                action_input=model_step.action_input, raw_response=raw,
                observation=obs, ok=False,
            ))
            if prog:
                _progress(f"[plan-solve] Step {step_index}: tool timed out")
            return None

    # ================================================================
    # 子方法：重规划逻辑
    # ================================================================
    def _handle_replan(
        self, task: PublicTask, tool_desc: str,
        state: AgentRuntimeState, step_index: int,
        replan_used: int, prog: bool,
    ) -> tuple[list[str], int] | None:
        """工具失败后尝试生成新计划。

        返回:
            (new_plan, 0) 表示重规划成功（plan_idx 重置为 0）；
            None 表示重规划失败，已记录到 state，调用方应继续使用原计划。
        """
        if prog:
            _progress("[plan-solve] Tool failed, re-planning ...")

        # 构建已完成动作摘要，防止新计划重复已成功的步骤
        completed = []
        failed_last = []
        for s in state.steps:
            preview = json.dumps(s.observation, ensure_ascii=False)[:200]
            if s.ok:
                completed.append(f"  ✓ {s.action}({_preview_json(s.action_input, 80)}) → {preview}")
            elif s == state.steps[-1]:
                failed_last.append(f"  ✗ {s.action}({_preview_json(s.action_input, 80)}) → {preview}")

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
                _progress("[plan-solve] New plan:")
                for i, s in enumerate(new_plan, 1):
                    _progress(f"[plan-solve]   {i}. {s}")
            return (new_plan, 0)
        except Exception as replan_exc:
            # 重规划失败：记录日志并写入 state，保留可观测性，然后继续用原计划
            logging.warning(
                "[plan-solve] Re-plan failed (attempt %d/%d): %s",
                replan_used, self.config.max_replans, replan_exc,
            )
            if prog:
                _progress(
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
    # 子方法：收尾与结果封装
    # ================================================================
    @staticmethod
    def _finalize(
        task: PublicTask, state: AgentRuntimeState, prog: bool,
    ) -> AgentRunResult:
        """主循环结束后收尾：设置失败原因（若未交卷）并封装结果。"""
        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."
        if prog and state.answer is None:
            _progress(f"[plan-solve] Stopped: {state.failure_reason}")
        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )

    # ================================================================
    # 主循环（瘦身后的顶层编排器）
    # ================================================================
    def run(self, task: PublicTask) -> AgentRunResult:
        """
        Plan-and-Solve 主循环：
          1. 生成计划（Phase 1）
          2. 按计划逐步执行，工具成功则推进到下一计划步骤（Phase 2）
          3. 遇到终止型工具（answer）则交卷退出
        """
        state = AgentRuntimeState()
        cfg = self.config
        prog = cfg.progress
        tool_desc = self.tools.describe_for_prompt()

        # ---- Phase 1: 生成计划 ----
        if prog:
            _progress("[plan-solve] Phase 1: generating plan ...")
        try:
            plan = self._generate_plan(task, tool_desc)
        except Exception as exc:
            if prog:
                _progress(f"[plan-solve] Plan generation failed: {exc}, using fallback.")
            plan = ["List context files", "Inspect data", "Solve and call answer"]

        if prog:
            for i, s in enumerate(plan, 1):
                _progress(f"[plan-solve]   {i}. {s}")

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
                _progress(
                    f"[plan-solve] Step {step_index}/{cfg.max_steps} "
                    f"(plan {plan_idx+1}/{len(plan)}): calling model ..."
                )

            messages = self._build_exec_messages(task, tool_desc, plan, plan_idx, state)

            # 1) 模型调用（带重试）
            raw = self._call_model_with_retry(messages, step_index, state, prog)
            if raw is None:
                continue

            # 2) 解析模型输出 + 执行
            try:
                model_step = _parse_step(raw)
                if prog:
                    _progress(
                        f"[plan-solve] Step {step_index}: "
                        f"action={model_step.action!r} "
                        f"input={_preview_json(model_step.action_input)}"
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
                    task, model_step, raw, state, step_index, prog,
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
                    _progress(f"[plan-solve] Step {step_index}: ok={tool_result.ok}{term}")

                # 6) 终止型工具 → 交卷
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    if prog:
                        _progress("[plan-solve] Submitted final answer.")
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
                    _progress(f"[plan-solve] Step {step_index}: error: {exc}")

        return self._finalize(task, state, prog)
