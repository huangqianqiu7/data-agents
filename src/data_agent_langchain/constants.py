"""
跨模块的字符串常量集合。

新加常量请统一从本模块导入，避免在多个调用点重复字面量。
"""
from __future__ import annotations

# Plan-and-Solve 计划耗尽但仍未提交答案时的回退提示语；图会把这条文本追加
# 到 plan 末尾，并停留在最后一个 index，直到模型最终调用 ``answer``。
FALLBACK_STEP_PROMPT: str = "Call the answer tool with the final result table."


__all__ = ["FALLBACK_STEP_PROMPT"]