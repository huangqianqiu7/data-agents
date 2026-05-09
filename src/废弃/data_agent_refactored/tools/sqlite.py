"""
Backwards-compat shim — SQLite helpers moved to
``data_agent_common.tools.sqlite``.
"""
from __future__ import annotations

from data_agent_common.tools.sqlite import (
    execute_read_only_sql,
    inspect_sqlite_schema,
)

__all__ = ["execute_read_only_sql", "inspect_sqlite_schema"]
