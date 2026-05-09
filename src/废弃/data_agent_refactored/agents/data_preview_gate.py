"""
Backwards-compat shim — gate constants moved to
``data_agent_common.agents.gate``.

The legacy ``has_data_preview`` accepted an ``AgentRuntimeState``; the new
common helper accepts any iterable of ``StepRecord``. We keep a thin local
wrapper here so call sites that pass the whole ``state`` keep working.
"""
from __future__ import annotations

from data_agent_common.agents.gate import (
    DATA_PREVIEW_ACTIONS,
    GATED_ACTIONS,
    GATE_REMINDER,
)
from data_agent_common.agents.gate import has_data_preview as _has_data_preview_iter
from data_agent_common.agents.runtime import AgentRuntimeState


def has_data_preview(state: AgentRuntimeState) -> bool:
    """Return ``True`` if at least one data-preview tool succeeded.

    Legacy signature accepting the full ``AgentRuntimeState``; delegates to
    the common iterable-based helper.
    """
    return _has_data_preview_iter(state.steps)


__all__ = [
    "DATA_PREVIEW_ACTIONS",
    "GATED_ACTIONS",
    "GATE_REMINDER",
    "has_data_preview",
]
