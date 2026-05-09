"""
``MetricsCollector`` —— 每任务级离线指标 callback，输出 ``metrics.json``。

LangChain ``BaseCallbackHandler`` 子类，订阅：
  - ``on_llm_end``：累计 prompt / completion token。
  - ``on_tool_end``：累计每工具调用次数（兼容 LangChain 内置 ``ToolNode`` 的回调）。
  - ``on_custom_event``（v4 M4）：订阅 6 类业务事件
    （``gate_block`` / ``replan_triggered`` / ``replan_failed`` /
    ``parse_error`` / ``model_error`` / ``memory_recall`` / ``tool_call``）。
  - ``on_chain_end``：在最外层 chain end 时把汇总结果写入 ``metrics.json``。

业务节点不直接调用 ``MetricsCollector``，全部走 ``dispatch_custom_event``
事件链路（§11.5），这样 metrics 收集器与节点完全解耦。
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler


class MetricsCollector(BaseCallbackHandler):
    """每任务级 callback 收集器，最终写出 ``metrics.json``。"""

    def __init__(self, *, task_id: str, output_dir: Path) -> None:
        super().__init__()
        self._task_id = task_id
        self._output_dir = output_dir
        self._start_t = perf_counter()
        self._tokens_prompt = 0
        self._tokens_completion = 0
        self._tool_counts: dict[str, int] = {}
        self._gate_blocks = 0
        self._replan_count = 0
        self._parse_errors = 0
        self._model_errors = 0
        self._memory_recalls: list[dict] = []

    def on_llm_end(self, response, **kwargs) -> None:
        usage = getattr(response, "llm_output", {}) or {}
        token_usage = usage.get("token_usage", {}) or {}
        self._tokens_prompt += int(token_usage.get("prompt_tokens", 0) or 0)
        self._tokens_completion += int(token_usage.get("completion_tokens", 0) or 0)

    def on_tool_end(self, output, **kwargs) -> None:
        name = str(kwargs.get("name") or _tool_name_from_output(output) or "unknown")
        self._tool_counts[name] = self._tool_counts.get(name, 0) + 1

    def on_custom_event(self, name: str, data: dict, *, run_id, **kwargs) -> None:
        if name == "gate_block":
            self._gate_blocks += 1
        elif name == "replan_triggered":
            self._replan_count += 1
        elif name == "replan_failed":
            self._replan_count += 1
        elif name == "parse_error":
            self._parse_errors += 1
        elif name == "model_error":
            self._model_errors += 1
        elif name == "memory_recall":
            self._memory_recalls.append(dict(data))
        elif name == "tool_call":
            tool = str(data.get("tool") or "unknown")
            self._tool_counts[tool] = self._tool_counts.get(tool, 0) + 1

    def on_chain_end(self, outputs, **kwargs) -> None:
        # 嵌套子图的 chain_end 也会走到这里；通过 ``parent_run_id`` 判断
        # 是否最外层，避免重复写 metrics.json。
        if kwargs.get("parent_run_id") is not None:
            return
        total = self._tokens_prompt + self._tokens_completion
        payload = {
            "task_id": self._task_id,
            "succeeded": bool(outputs.get("failure_reason") is None) if isinstance(outputs, dict) else False,
            "tokens": {
                "prompt": self._tokens_prompt,
                "completion": self._tokens_completion,
                "total": total,
            },
            "tool_calls": dict(self._tool_counts),
            "gate_blocks": self._gate_blocks,
            "replan_count": self._replan_count,
            "parse_errors": self._parse_errors,
            "model_errors": self._model_errors,
            "memory_recalls": list(self._memory_recalls),
            "wall_clock_s": round(perf_counter() - self._start_t, 6),
        }
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _tool_name_from_output(output) -> str | None:
    """从 ``on_tool_end`` 的 output 里反向猜出工具名（兼容 legacy 行为）。"""
    if not isinstance(output, str):
        return None
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw_name = payload.get("tool_name") or payload.get("action")
    return str(raw_name) if raw_name else None


__all__ = ["MetricsCollector"]