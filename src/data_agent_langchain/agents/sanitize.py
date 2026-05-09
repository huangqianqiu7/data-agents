"""
trace replay 的 sanitize 协议。

智能体追加的 ``__error__`` (parse / model 失败) 或 ``__replan_failed__``
(re-plan 失败) 步骤里，原始 ``raw_response`` 可能是非法 JSON 或空串。
回放给 LLM 时若直接用，会污染下一轮（format-greedy decoding / JSON-mode
混乱）。

LangGraph backend 把这两类步骤的 ``raw_response`` 替换为
``ERROR_SANITIZED_RESPONSE`` 占位 JSON，让模型在下一轮自我纠错。
"""
from __future__ import annotations

import json


# 需要走 sanitize 替换的 action 名集合。frozen 集合让调用方可以放心 ``in``。
SANITIZE_ACTIONS: frozenset[str] = frozenset({"__error__", "__replan_failed__"})


# 替换 raw_response 用的占位 JSON。措辞经过权衡，让模型下一轮自动修复格式。
ERROR_SANITIZED_RESPONSE: str = json.dumps(
    {
        "thought": "My previous response had a format error. "
                   "I must return a valid JSON with keys: thought, action, action_input.",
        "action": "__error__",
        "action_input": {},
    },
    ensure_ascii=False,
)


__all__ = ["ERROR_SANITIZED_RESPONSE", "SANITIZE_ACTIONS"]