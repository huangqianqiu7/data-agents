"""Cross-task structured memory record types.

Fields are audited and deliberately exclude free-text fields such as question,
answer, approach, hint, and summary. Extensions must modify this file and pass
review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


FileKind = Literal["csv", "json", "doc", "sqlite", "other"]


@dataclass(frozen=True, slots=True)
class DatasetKnowledgeRecord:
    file_path: str
    file_kind: FileKind
    schema: dict[str, str]
    row_count_estimate: int | None
    encoding: str | None = None
    sample_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolPlaybookRecord:
    tool_name: str
    input_template: dict[str, Any]
    preconditions: list[str]
    typical_failures: list[str] = field(default_factory=list)


__all__ = ["DatasetKnowledgeRecord", "FileKind", "ToolPlaybookRecord"]
