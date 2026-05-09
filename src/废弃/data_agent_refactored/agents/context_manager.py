"""
History context manager for building conversation messages.

Assembles past agent steps into LLM messages with two-layer truncation:
  L1: hard-truncate individual observation content
  L2: pin critical steps (data previews) + sliding-window eviction of the rest

The sanitize protocol constants (``ERROR_SANITIZED_RESPONSE`` /
``SANITIZE_ACTIONS``) are now sourced from ``data_agent_common.agents.sanitize``
so the new LangGraph backend uses byte-for-byte identical strings (see
LANGCHAIN_MIGRATION_PLAN.md v4 E2). The legacy private alias
``_SANITIZE_ACTIONS`` is retained for backwards compatibility.
"""
from __future__ import annotations

import logging

from data_agent_common.agents.sanitize import (
    ERROR_SANITIZED_RESPONSE,
    SANITIZE_ACTIONS,
)
from data_agent_refactored.agents.model import ModelMessage
from data_agent_refactored.agents.prompt import build_observation_prompt
from data_agent_refactored.agents.runtime import AgentRuntimeState
from data_agent_refactored.agents.data_preview_gate import DATA_PREVIEW_ACTIONS
from data_agent_refactored.agents.text_helpers import estimate_tokens, truncate_observation

logger = logging.getLogger(__name__)

# Legacy private alias retained for backwards compatibility — the canonical
# name is ``data_agent_common.agents.sanitize.SANITIZE_ACTIONS``.
_SANITIZE_ACTIONS: frozenset[str] = SANITIZE_ACTIONS


def build_history_messages(
    state: AgentRuntimeState,
    base_messages: list[ModelMessage],
    max_obs_chars: int,
    max_context_tokens: int,
) -> list[ModelMessage]:
    """Build the full message list including truncated history.

    Args:
        state: Current agent runtime state.
        base_messages: Base messages (system + user). Not modified.
        max_obs_chars: Per-observation character limit.
        max_context_tokens: Token budget for the history window.

    Returns:
        New list with history messages appended.
    """
    messages: list[ModelMessage] = list(base_messages)

    # ---- L1: pre-process each step into (assistant, user) text pairs ----
    history_pairs: list[tuple[str, str]] = []
    for step in state.steps:
        obs_safe: dict[str, Any] = truncate_observation(step.observation, max_obs_chars)
        if step.action in _SANITIZE_ACTIONS:
            assistant_text = ERROR_SANITIZED_RESPONSE
        else:
            assistant_text = step.raw_response
        user_text = build_observation_prompt(obs_safe)
        history_pairs.append((assistant_text, user_text))

    if not history_pairs:
        return messages

    # ---- L2: pin critical steps + sliding-window eviction ----
    base_tokens = sum(estimate_tokens(m.content) for m in messages)
    budget = max_context_tokens

    if base_tokens >= budget:
        logger.warning(
            "[context] Base prompt (%d tokens) exceeds budget (%d). "
            "Allowing minimal history headroom.",
            base_tokens,
            budget,
        )
        budget = base_tokens + max_obs_chars

    pair_tokens = [
        estimate_tokens(a) + estimate_tokens(u) for a, u in history_pairs
    ]

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

    # FIFO eviction from oldest evictable steps.
    evictable_total = sum(pair_tokens[i] for i in evictable_indices)
    evict_start = 0
    while evict_start < len(evictable_indices) and evictable_total > remaining_budget:
        evictable_total -= pair_tokens[evictable_indices[evict_start]]
        evict_start += 1

    kept_evictable: set[int] = set(evictable_indices[evict_start:])
    kept_pinned: set[int] = set(pinned_indices)
    n_omitted = evict_start

    if n_omitted > 0:
        omitted_summary = (
            f"[Note: {n_omitted} earlier step(s) omitted to fit context window. "
            f"Key data-preview steps are preserved. "
            f"Recent {len(kept_evictable) + len(kept_pinned)} step(s) are shown below.]"
        )
        messages.append(ModelMessage(role="user", content=omitted_summary))

    for i, (assistant_text, user_text) in enumerate(history_pairs):
        if i in kept_pinned or i in kept_evictable:
            messages.append(ModelMessage(role="assistant", content=assistant_text))
            messages.append(ModelMessage(role="user", content=user_text))

    return messages
