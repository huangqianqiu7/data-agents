"""
Backwards-compat shim — dataset loader moved to
``data_agent_common.benchmark.dataset``.
"""
from __future__ import annotations

from data_agent_common.benchmark.dataset import (
    TASK_DIR_PREFIX,
    DABenchPublicDataset,
)

__all__ = ["DABenchPublicDataset", "TASK_DIR_PREFIX"]
