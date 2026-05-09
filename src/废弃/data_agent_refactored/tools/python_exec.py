"""
Backwards-compat shim — Python sandbox moved to
``data_agent_common.tools.python_exec``.
"""
from __future__ import annotations

from data_agent_common.tools.python_exec import (
    EXECUTE_PYTHON_TIMEOUT_SECONDS,
    execute_python_code,
)

__all__ = ["EXECUTE_PYTHON_TIMEOUT_SECONDS", "execute_python_code"]
