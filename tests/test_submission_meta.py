"""v5 §四第 1 步「元测试 / dataset」4 条测试。

守护两件事：

1. ``pyproject.toml`` 主依赖瘦身、entry point 清理（§二.8）。
2. ``task.json`` schema 放宽：必含三键，多余字段忽略，缺字段抛
   ``DatasetError``（§三.1）。
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from data_agent_langchain.benchmark.dataset import DABenchPublicDataset
from data_agent_langchain.exceptions import DatasetError


# ---------------------------------------------------------------------------
# pyproject.toml 元测试（2 条）
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_pyproject_main_dependencies_excludes_baseline_only_packages():
    """v5 §二.8.1：主依赖不再含 datasets / duckdb / huggingface-hub /
    polars / pyarrow / openpyxl / rich 这些 baseline 专属大包。"""
    payload = tomllib.loads((_project_root() / "pyproject.toml").read_text("utf-8"))
    forbidden = {"datasets", "duckdb", "huggingface-hub", "polars",
                 "pyarrow", "openpyxl", "rich"}
    listed = {
        dep.split(">=")[0].split("<")[0].split("==")[0].split("~=")[0].strip().lower()
        for dep in payload["project"]["dependencies"]
    }
    overlap = listed & forbidden
    assert not overlap, f"forbidden deps in main: {sorted(overlap)}"


def test_pyproject_drops_dabench_keeps_dabench_lc_script():
    """v5 §二.8.4：删除过期 ``dabench`` entry point，保留 ``dabench-lc``。"""
    payload = tomllib.loads((_project_root() / "pyproject.toml").read_text("utf-8"))
    scripts = payload["project"]["scripts"]
    assert "dabench" not in scripts, "stale dabench entry point must be removed"
    assert scripts.get("dabench-lc") == "data_agent_langchain.cli:main"


# ---------------------------------------------------------------------------
# task.json schema 放宽（2 条）
# ---------------------------------------------------------------------------

def _write_task(input_dir: Path, task_id: str, payload: dict) -> Path:
    task_dir = input_dir / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    task_json = task_dir / "task.json"
    task_json.write_text(json.dumps(payload), encoding="utf-8")
    return task_dir


def test_task_json_with_extra_fields_still_loads(tmp_path: Path):
    """v5 §三.1：``task.json`` 含官方未来扩展字段时仍可正常加载，仅取必需三键。"""
    _write_task(
        tmp_path,
        "task_1",
        {
            "task_id": "task_1",
            "difficulty": "easy",
            "question": "Q?",
            # 官方未来可能新增的字段
            "category": "table_qa",
            "tags": ["sql", "easy"],
            "metadata": {"version": 2},
        },
    )

    dataset = DABenchPublicDataset(tmp_path)
    task = dataset.get_task("task_1")

    assert task.task_id == "task_1"
    assert task.difficulty == "easy"
    assert task.question == "Q?"


def test_task_json_missing_required_key_raises_dataset_error(tmp_path: Path):
    """v5 §三.1：缺三键之一时抛 ``DatasetError``。"""
    _write_task(
        tmp_path,
        "task_1",
        {
            "task_id": "task_1",
            "difficulty": "easy",
            # 故意删去 "question"
        },
    )

    dataset = DABenchPublicDataset(tmp_path)
    with pytest.raises(DatasetError):
        dataset.get_task("task_1")
