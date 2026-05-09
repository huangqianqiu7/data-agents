"""
Backwards-compat shim — schema moved to ``data_agent_common.benchmark.schema``.

Both backends share the same task / answer schema; the canonical home is
``data_agent_common.benchmark.schema`` (LANGCHAIN_MIGRATION_PLAN.md §3.1 / C7).
"""
from __future__ import annotations

from data_agent_common.benchmark.schema import (
    AnswerTable,
    PublicTask,
    TaskAssets,
    TaskRecord,
)

__all__ = ["AnswerTable", "PublicTask", "TaskAssets", "TaskRecord"]
