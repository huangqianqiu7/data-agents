"""
工具 observation 渲染成 user 消息字符串。

:func:`build_observation_prompt` 把 tool observation dict 序列化成
带 ``"Observation:"`` 前缀的 user 消息体。格式与 legacy hand-written
backend 字符级一致，保证 ``json_action`` 模式下两个 backend 的消息回放
完全相同。
"""
from __future__ import annotations

import json
from typing import Any


def build_observation_prompt(observation: dict[str, Any]) -> str:
    """把 *observation* 渲染成 user 消息体。"""
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"


__all__ = ["build_observation_prompt"]