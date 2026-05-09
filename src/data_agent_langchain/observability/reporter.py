"""
批量 metrics 聚合器 —— 把每任务 ``metrics.json`` 合并为
``summary.json``（§20.5）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def aggregate_metrics(run_output_dir: Path) -> dict[str, Any]:
    """遍历每任务 ``metrics.json`` 并产出汇总字典。"""
    metrics: list[dict[str, Any]] = []
    for path in sorted(run_output_dir.glob("*/metrics.json")):
        metrics.append(json.loads(path.read_text(encoding="utf-8")))
    tokens_total = {"prompt": 0, "completion": 0, "total": 0}
    tool_calls_total: dict[str, int] = {}
    wall_clocks: list[float] = []
    for item in metrics:
        tokens = item.get("tokens", {}) or {}
        for key in tokens_total:
            tokens_total[key] += int(tokens.get(key, 0) or 0)
        for name, count in (item.get("tool_calls", {}) or {}).items():
            tool_calls_total[name] = tool_calls_total.get(name, 0) + int(count)
        wall_clocks.append(float(item.get("wall_clock_s", 0.0) or 0.0))
    summary = {
        "task_count": len(metrics),
        "succeeded": sum(1 for item in metrics if item.get("succeeded")),
        "tokens_total": tokens_total,
        "tool_calls_total": tool_calls_total,
        "wall_clock": _wall_clock_summary(wall_clocks),
    }
    return summary


def _wall_clock_summary(values: list[float]) -> dict[str, float]:
    """计算 wall_clock 的 p50 / p95 / max。"""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    values = sorted(values)
    return {
        "p50": values[len(values) // 2],
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
        "max": values[-1],
    }


__all__ = ["aggregate_metrics"]