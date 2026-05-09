"""
DABench 任务 schema 的不可变数据结构定义。

涵盖：
  - :class:`TaskRecord` —— ``task.json`` 解析出的元数据。
  - :class:`TaskAssets` —— 任务相关的文件系统路径。
  - :class:`PublicTask` —— record + assets 合体；图节点和工具消费此对象。
  - :class:`AnswerTable` —— 智能体最终提交给评测系统的结果表。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """从 ``task.json`` 抽取的任务元数据。"""
    task_id: str
    difficulty: str
    question: str


@dataclass(frozen=True, slots=True)
class TaskAssets:
    """单个任务相关的文件系统路径。"""
    task_dir: Path
    context_dir: Path


@dataclass(frozen=True, slots=True)
class PublicTask:
    """合并元数据与文件系统资产的完整任务对象。"""
    record: TaskRecord
    assets: TaskAssets

    @property
    def task_id(self) -> str:
        return self.record.task_id

    @property
    def difficulty(self) -> str:
        return self.record.difficulty

    @property
    def question(self) -> str:
        return self.record.question

    @property
    def task_dir(self) -> Path:
        return self.assets.task_dir

    @property
    def context_dir(self) -> Path:
        return self.assets.context_dir


@dataclass(frozen=True, slots=True)
class AnswerTable:
    """智能体提交的结构化答案表。"""
    columns: list[str]
    rows: list[list[Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }