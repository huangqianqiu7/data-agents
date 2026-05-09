"""
工作记忆 —— 单 run 内的 scratchpad 渲染器（LangGraph 后端）。

对应 legacy ``data_agent_refactored.agents.context_manager.build_history_messages``，
但做了三件事：

  - 输入从 legacy ``AgentRuntimeState`` 改为 :class:`StepRecord` 列表
    （即 LangGraph 的 ``RunState["steps"]``）。
  - 输出 ``langchain_core.messages.BaseMessage``，可以直接喂给
    LangChain prompt 拼装，而不是 legacy 的 ``ModelMessage`` 字符串。
  - 同时支持两种 ``action_mode``（§11.3 / C9）：

      * ``json_action`` —— 把每一步回放成
        ``AIMessage(content=raw_response)`` + ``HumanMessage(content="Observation:...")``，
        与 legacy 行为一致。
      * ``tool_calling`` —— 把 ``raw_response`` 反序列化为 ``tool_calls`` JSON，
        回放成 ``AIMessage(tool_calls=...)`` + ``ToolMessage(content=..., tool_call_id=...)``。

  - 保留两层截断契约：

      L1：每一步 ``truncate_observation``（从 ``agents.text_helpers``
          re-export，更小的 import 表面积）。
      L2：把成功的数据预览步骤 *钉住*，剩下的按 token 预算 FIFO 淘汰。

  - 替换 ``__error__`` / ``__replan_failed__``（即 ``SANITIZE_ACTIONS``）
    步骤的 ``raw_response`` 为 ``ERROR_SANITIZED_RESPONSE`` 占位符，
    避免畸形 raw_response 污染下一轮 prompt。

本模块放在 ``memory/`` 而不是 ``agents/`` 下：working memory 是 §8 规
划中将来的 memory hierarchy（dataset-knowledge / tool-playbook / corpus
等模块）的第一个成员；后者已抽到独立提案 ``MEMORY_MODULE_PROPOSAL.md``。
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from data_agent_langchain.agents.gate import DATA_PREVIEW_ACTIONS
from data_agent_langchain.agents.observation_prompt import build_observation_prompt
from data_agent_langchain.agents.runtime import StepRecord
from data_agent_langchain.agents.sanitize import (
    ERROR_SANITIZED_RESPONSE,
    SANITIZE_ACTIONS,
)
from data_agent_langchain.agents.text_helpers import (
    estimate_tokens,
    truncate_observation as _truncate_observation_dict,
)


# 直接 re-export 出去，让 ``memory.working.truncate_observation`` 可用，
# 调用方不必再额外 import ``text_helpers``。
truncate_observation = _truncate_observation_dict


# ---------------------------------------------------------------------------
# 单步 → BaseMessage 列表
# ---------------------------------------------------------------------------

def render_step_messages(
    step: StepRecord,
    *,
    action_mode: str = "json_action",
    max_obs_chars: int = 3000,
) -> list[BaseMessage]:
    """把一条 ``StepRecord`` 渲染成 (assistant, user/tool) 消息对。

    Args:
        step: 要回放的步骤。
        action_mode: ``"json_action"``（legacy fenced-JSON）或
            ``"tool_calling"``（LangChain ``tool_calls``）；详见 §11.3。
        max_obs_chars: observation 单步字符截断预算。

    Returns:
        恰好两条消息 —— 一条 assistant + 一条后续消息，保留 legacy
        backend 的 round-trip 契约。
    """
    obs_safe = truncate_observation(step.observation, max_obs_chars)
    raw_response = (
        ERROR_SANITIZED_RESPONSE if step.action in SANITIZE_ACTIONS
        else step.raw_response
    )

    if action_mode == "tool_calling":
        return _render_tool_calling(step, raw_response, obs_safe)
    return _render_json_action(raw_response, obs_safe)


def _render_json_action(raw_response: str, observation: dict[str, Any]) -> list[BaseMessage]:
    """``json_action`` 模式：直接把 raw_response 当作 assistant 消息回放。"""
    return [
        AIMessage(content=raw_response),
        HumanMessage(content=build_observation_prompt(observation)),
    ]


def _render_tool_calling(
    step: StepRecord, raw_response: str, observation: dict[str, Any],
) -> list[BaseMessage]:
    """``tool_calling`` 模式：从 raw_response 反序列化 ``tool_calls``。

    走 sanitize 的步骤（``__error__`` / ``__replan_failed__``）里
    raw_response 是占位 JSON，没有 ``tool_calls`` 形态；这种情况下回退到
    plain ``AIMessage`` + ``HumanMessage`` 的双消息结构，保证下一轮的
    transcript 仍然语义连贯。
    """
    tool_call_id = "unknown"
    tool_calls: list[dict[str, Any]] = []
    if step.action not in SANITIZE_ACTIONS and raw_response:
        try:
            decoded = json.loads(raw_response)
        except Exception:
            decoded = []
        if isinstance(decoded, list):
            tool_calls = [tc for tc in decoded if isinstance(tc, dict)]
        elif isinstance(decoded, dict):
            # 某些 provider 把单个 tool call 序列化为 dict；当作一元列表处理。
            tool_calls = [decoded]

    if not tool_calls:
        # 反序列化不出 tool_calls —— 退化到 json_action 形态。
        return _render_json_action(raw_response, observation)

    if isinstance(tool_calls[0].get("id"), str):
        tool_call_id = tool_calls[0]["id"]

    return [
        AIMessage(content="", tool_calls=tool_calls),
        ToolMessage(
            content=json.dumps(observation, ensure_ascii=False),
            tool_call_id=tool_call_id,
        ),
    ]


# ---------------------------------------------------------------------------
# L2 选取：钉住数据预览 + FIFO 淘汰其余
# ---------------------------------------------------------------------------

def _step_pair_text(step: StepRecord, *, max_obs_chars: int, action_mode: str) -> tuple[str, str]:
    """产出 (assistant, follow-up) 字符串对，用于 token 预算估算。

    这两个字符串 **不是** 实际消息；它们是 token-count 的代理，与 legacy
    ``estimate_tokens(a) + estimate_tokens(u)`` 的算账方式一致。
    """
    obs_safe = truncate_observation(step.observation, max_obs_chars)
    if step.action in SANITIZE_ACTIONS:
        assistant_text = ERROR_SANITIZED_RESPONSE
    else:
        assistant_text = step.raw_response
    user_text = build_observation_prompt(obs_safe)
    if action_mode == "tool_calling":
        # tool-calling 消息长度大致与 observation 序列化相当；继续复用
        # ``user_text`` 作为代理，让两种模式的预算可比。
        pass
    return assistant_text, user_text


def select_steps_for_context(
    steps: Sequence[StepRecord],
    *,
    base_tokens: int,
    max_context_tokens: int,
    max_obs_chars: int,
    action_mode: str = "json_action",
) -> tuple[list[int], int]:
    """决定哪些 step 索引在预算内被保留。

    返回值 ``(kept_indices_sorted, n_omitted)``：保留的步骤索引按原顺序
    排列；``n_omitted`` 是被 FIFO 淘汰的步骤数。

    逻辑严格对齐 ``context_manager.build_history_messages``：

        1. 每条 ``ok and action in DATA_PREVIEW_ACTIONS`` 步骤被 *钉住*。
        2. 从 **非钉住** 列表的前端 FIFO 淘汰，直到剩余 token 适配预算；
           钉住步骤永不淘汰，即使因此预算被超过。
    """
    if not steps:
        return [], 0

    pair_tokens: list[int] = []
    for step in steps:
        a, u = _step_pair_text(step, max_obs_chars=max_obs_chars, action_mode=action_mode)
        pair_tokens.append(estimate_tokens(a) + estimate_tokens(u))

    pinned_indices: list[int] = []
    evictable_indices: list[int] = []
    for i, step in enumerate(steps):
        if step.ok and step.action in DATA_PREVIEW_ACTIONS:
            pinned_indices.append(i)
        else:
            evictable_indices.append(i)

    # 允许 base prompt 超预算；给历史保留至少 ``max_obs_chars``（一条
    # observation 的容量）的头空间，确保至少一条消息仍能流转。
    budget = max_context_tokens
    if base_tokens >= budget:
        budget = base_tokens + max_obs_chars

    pinned_tokens = sum(pair_tokens[i] for i in pinned_indices)
    remaining_budget = budget - base_tokens - pinned_tokens

    evictable_total = sum(pair_tokens[i] for i in evictable_indices)
    evict_start = 0
    while evict_start < len(evictable_indices) and evictable_total > remaining_budget:
        evictable_total -= pair_tokens[evictable_indices[evict_start]]
        evict_start += 1

    kept = sorted(set(pinned_indices) | set(evictable_indices[evict_start:]))
    return kept, evict_start


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------

def build_scratchpad_messages(
    steps: Iterable[StepRecord],
    base_messages: list[BaseMessage],
    *,
    action_mode: str = "json_action",
    max_obs_chars: int = 3000,
    max_context_tokens: int = 24000,
) -> list[BaseMessage]:
    """在 *base_messages* 末尾追加截断后的历史并返回新列表。

    Args:
        steps: ``StepRecord`` 序列（一般来自 ``RunState["steps"]``）。
        base_messages: 已有的 prompt 前缀（system / user 等）；返回列表
            的开头与之相同（不会被原地修改）。
        action_mode: ``"json_action"`` 或 ``"tool_calling"``。
        max_obs_chars: 单条 observation 的字符上限。
        max_context_tokens: base + history 总 token 预算。

    Returns:
        新的消息列表。``base_messages`` 不会被改动。
    """
    out: list[BaseMessage] = list(base_messages)
    materialised = list(steps)
    if not materialised:
        return out

    base_tokens = sum(estimate_tokens(_message_text(m)) for m in out)
    kept_indices, n_omitted = select_steps_for_context(
        materialised,
        base_tokens=base_tokens,
        max_context_tokens=max_context_tokens,
        max_obs_chars=max_obs_chars,
        action_mode=action_mode,
    )

    if n_omitted > 0:
        out.append(HumanMessage(content=(
            f"[Note: {n_omitted} earlier step(s) omitted to fit context window. "
            f"Key data-preview steps are preserved. "
            f"Recent {len(kept_indices)} step(s) are shown below.]"
        )))

    for i in kept_indices:
        out.extend(render_step_messages(
            materialised[i],
            action_mode=action_mode,
            max_obs_chars=max_obs_chars,
        ))
    return out


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _message_text(message: BaseMessage) -> str:
    """尽力提取消息文本用于 token 估算。

    LangChain ``BaseMessage`` 的 ``content`` 既可以是 ``str``，也可以是
    multi-modal 块的 ``list[dict]``。token 预算只关心粗略长度，所以列表
    形态直接 ``json.dumps`` 一下 —— 这恰好就是 LangChain 自己在序列化
    OpenAI 请求时的做法。
    """
    content = message.content
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


__all__ = [
    "build_scratchpad_messages",
    "render_step_messages",
    "select_steps_for_context",
    "truncate_observation",
]