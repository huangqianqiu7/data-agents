"""
读取任务 ``context/`` 目录下文件的底层工具函数。

所有路径操作都强制 sandbox 边界：相对路径必须落在 ``context_dir`` 内，
不允许 ``..`` 之类的逃逸；找不到文件抛 :class:`ContextAssetNotFoundError`。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from data_agent_langchain.benchmark.schema import PublicTask
from data_agent_langchain.exceptions import (
    ContextAssetNotFoundError,
    ContextPathEscapeError,
)


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    """在任务上下文中解析 *relative_path*，拒绝路径穿越攻击。"""
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ContextPathEscapeError(relative_path)
    if not candidate.exists():
        raise ContextAssetNotFoundError(relative_path)
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, Any]:
    """递归列出任务 ``context/`` 目录下的所有文件 / 子目录。"""
    entries: list[dict[str, Any]] = []

    def _walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                _walk(child, depth + 1)

    _walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }


def read_csv_preview(
    task: PublicTask, relative_path: str, *, max_rows: int = 20
) -> dict[str, Any]:
    """返回 CSV 文件前 *max_rows* 行的预览结果。"""
    path = resolve_context_path(task, relative_path)
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return {
            "path": relative_path,
            "columns": [],
            "rows": [],
            "row_count": 0,
        }

    header = rows[0]
    data_rows = rows[1:]
    return {
        "path": relative_path,
        "columns": header,
        "rows": data_rows[:max_rows],
        "row_count": len(data_rows),
    }


def read_json_preview(
    task: PublicTask, relative_path: str, *, max_chars: int = 4000
) -> dict[str, Any]:
    """返回 JSON 文件经字符长度截断后的预览。"""
    path = resolve_context_path(task, relative_path)
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    preview = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "path": relative_path,
        "preview": preview[:max_chars],
        "truncated": len(preview) > max_chars,
    }


def read_doc_preview(
    task: PublicTask, relative_path: str, *, max_chars: int = 4000
) -> dict[str, Any]:
    """返回文本文档经字符长度截断后的预览。"""
    path = resolve_context_path(task, relative_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": relative_path,
        "preview": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


__all__ = [
    "list_context_tree",
    "read_csv_preview",
    "read_doc_preview",
    "read_json_preview",
    "resolve_context_path",
]