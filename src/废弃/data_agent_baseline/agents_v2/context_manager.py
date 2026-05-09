"""
历史上下文管理器。

负责将 Agent 的历史步骤组装为对话消息，同时实现双层截断防护：
  L1: 单条 observation 硬截断
  L2: 钉住关键步骤（数据预览）+ 滑动窗口淘汰非关键步骤

修复审计隐患 3（早期关键步骤被 FIFO 丢弃）和隐患 5（base_tokens 溢出无保护）。
"""
from __future__ import annotations

import json
import logging

from data_agent_baseline.agents_v2.model import ModelMessage
from data_agent_baseline.agents_v2.prompt import build_observation_prompt
from data_agent_baseline.agents_v2.runtime import AgentRuntimeState
from data_agent_baseline.agents_v2.gate import DATA_PREVIEW_ACTIONS
from data_agent_baseline.agents_v2.text_helpers import estimate_tokens, truncate_observation


# __error__ / __replan_failed__ 步骤的消毒替换模板
# 这些步骤的 raw_response 是非法 JSON 或空字符串，替换为合法占位符防止污染模型推理
ERROR_SANITIZED_RESPONSE = json.dumps(
    {
        "thought": "My previous response had a format error. "
                   "I must return a valid JSON with keys: thought, action, action_input.",
        "action": "__error__",
        "action_input": {},
    },
    ensure_ascii=False,
)

# 需要消毒替换的 action 类型
_SKIP_ACTIONS = frozenset({"__error__", "__replan_failed__"})


def build_history_messages(
    state: AgentRuntimeState,
    base_messages: list[ModelMessage],
    max_obs_chars: int,
    max_context_tokens: int,
) -> list[ModelMessage]:
    """构建包含历史对话的完整消息列表（带双层截断防护）。

    Args:
        state: 当前运行时状态
        base_messages: 已有的基础消息（system + user），不会被修改
        max_obs_chars: 单条 observation content 最大字符数
        max_context_tokens: 历史消息区的 token 预算上限

    Returns:
        追加了历史消息的完整消息列表（新列表，不污染 base_messages）
    """
    messages = list(base_messages)  # 浅拷贝

    # ---- L1: 预处理每一步的消息对（assistant + user） ----
    history_pairs: list[tuple[str, str]] = []
    for step in state.steps:
        obs_safe = truncate_observation(step.observation, max_obs_chars)
        # 消毒：错误/重规划失败步骤的 raw_response 替换为合法占位符
        if step.action in _SKIP_ACTIONS:
            assistant_text = ERROR_SANITIZED_RESPONSE
        else:
            assistant_text = step.raw_response
        user_text = build_observation_prompt(obs_safe)
        history_pairs.append((assistant_text, user_text))

    if not history_pairs:
        return messages

    # ---- L2: 钉住关键步骤 + 滑动窗口淘汰 ----
    base_tokens = sum(estimate_tokens(m.content) for m in messages)
    budget = max_context_tokens

    # 隐患5防护: 当 base prompt 自身就超预算时，留出最小历史余量
    if base_tokens >= budget:
        logging.warning(
            "[context] Base prompt (%d tokens) exceeds budget (%d). "
            "Allowing minimal history headroom.",
            base_tokens, budget,
        )
        budget = base_tokens + max_obs_chars

    pair_tokens = [
        estimate_tokens(a) + estimate_tokens(u) for a, u in history_pairs
    ]

    # 分区：pinned（成功的数据预览）vs evictable（其余步骤）
    pinned_indices: list[int] = []
    evictable_indices: list[int] = []
    for i, step in enumerate(state.steps):
        if i >= len(history_pairs):
            break
        if step.ok and step.action in DATA_PREVIEW_ACTIONS:
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
    n_omitted = evict_start

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
