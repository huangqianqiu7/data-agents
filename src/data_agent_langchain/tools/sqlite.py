"""
只读 SQLite 工具：schema 检视与只读 SQL 查询。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from data_agent_langchain.exceptions import ReadOnlySQLViolationError


def _connect_read_only(path: Path) -> sqlite3.Connection:
    """通过 URI 模式打开只读 SQLite 连接。"""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def inspect_sqlite_schema(path: Path) -> dict[str, Any]:
    """返回数据库文件中所有用户表的 name + CREATE SQL。"""
    with _connect_read_only(path) as conn:
        rows = conn.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        tables: list[dict[str, Any]] = []
        for name, create_sql in rows:
            tables.append(
                {
                    "name": name,
                    "create_sql": create_sql,
                }
            )
    return {
        "path": str(path),
        "tables": tables,
    }


def execute_read_only_sql(
    path: Path, sql: str, *, limit: int = 200
) -> dict[str, Any]:
    """执行一条只读 SQL 并返回最多 *limit* 行结果。"""
    normalized_sql = sql.lstrip().lower()
    if not normalized_sql.startswith(("select", "with", "pragma")):
        raise ReadOnlySQLViolationError("Only read-only SQL statements are allowed.")

    with _connect_read_only(path) as conn:
        cursor = conn.execute(sql)
        column_names = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    return {
        "path": str(path),
        "columns": column_names,
        "rows": [list(row) for row in limited_rows],
        "row_count": len(limited_rows),
        "truncated": truncated,
    }


__all__ = ["execute_read_only_sql", "inspect_sqlite_schema"]