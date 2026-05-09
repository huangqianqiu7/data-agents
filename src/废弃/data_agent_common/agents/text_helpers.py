"""
Shared text-processing utilities used by both backends.

Both ``data_agent_refactored.agents.context_manager.build_history_messages``
and the new ``data_agent_langchain.memory.working.build_scratchpad_messages``
rely on the same token estimator and observation truncator; placing them
here is what lets the two backends produce byte-for-byte identical
prompts in ``json_action`` mode.

Public functions:
  - :func:`progress`            — flush-safe progress printing.
  - :func:`preview_json`        — truncated JSON preview for logs.
  - :func:`estimate_tokens`     — rough token count estimate.
  - :func:`truncate_observation` — hard-truncate an observation's
    ``content`` field to ``max_chars`` characters.
"""
from __future__ import annotations

import json
from typing import Any


def progress(msg: str) -> None:
    """Print a progress message with immediate flush (multiprocess-safe)."""
    print(msg, flush=True)


def preview_json(obj: Any, max_len: int = 140) -> str:
    """Convert *obj* to a truncated JSON string for logging."""
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        text = str(obj)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def estimate_tokens(text: str) -> int:
    """Rough token estimate (≈ 1 token per 3 chars, biased conservative)."""
    return len(text) // 3 + 1


def truncate_observation(observation: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Hard-truncate the ``content`` field of an observation dict.

    Other metadata fields (``ok``, ``tool``, ``error``) are preserved as-is.
    Returns a shallow copy when truncation is applied.
    """
    content = observation.get("content")
    if not isinstance(content, str) or len(content) <= max_chars:
        return observation
    truncated = dict(observation)
    truncated["content"] = (
        content[:max_chars]
        + f"\n... [truncated, showing first {max_chars} chars of {len(content)}]"
    )
    return truncated


__all__ = [
    "estimate_tokens",
    "preview_json",
    "progress",
    "truncate_observation",
]
