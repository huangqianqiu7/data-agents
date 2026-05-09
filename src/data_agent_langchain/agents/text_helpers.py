"""
LangGraph 后端共用的文本处理小工具。

公开函数：
  - :func:`progress`            —— 立即 flush 的进度打印（多进程安全）。
  - :func:`preview_json`        —— 把任意对象渲染成截断后的 JSON 预览串。
  - :func:`estimate_tokens`     —— 按字符数粗估 token 数。
  - :func:`truncate_observation` —— 把 observation 字典里的 ``content``
    字段硬截断到 ``max_chars`` 字符。
"""
from __future__ import annotations

import json
from typing import Any


def progress(msg: str) -> None:
    """立即 flush 地打印进度消息（多进程下不会被缓冲吞掉）。"""
    print(msg, flush=True)


def preview_json(obj: Any, max_len: int = 140) -> str:
    """把 *obj* 转成 JSON 字符串并截断到 *max_len*，超出加 ``...`` 标记。"""
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        text = str(obj)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def estimate_tokens(text: str) -> int:
    """token 数粗估（≈ 每 3 个字符 1 token，偏保守）。"""
    return len(text) // 3 + 1


def truncate_observation(observation: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """对 observation 字典的 ``content`` 字段做硬截断。

    其他字段（``ok`` / ``tool`` / ``error`` 等）保持不变；触发截断时返回
    一份浅拷贝，原对象不被修改。
    """
    content = observation.get("content")
    if not isinstance(content, str) or len(content) <= max_chars:
        return observation
    truncated = dict(observation)
    truncated["content"] = (
        content[:max_chars]
        + f"\n... [truncated, showing first {max_chars} chars of {len(content)}]"
    )
    return truncated


__all__ = [
    "estimate_tokens",
    "preview_json",
    "progress",
    "truncate_observation",
]