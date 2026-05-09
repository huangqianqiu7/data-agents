"""
LLM 输出的多层 JSON 解析。

解析顺序：严格 JSON → 修复缺括号 → ``json_repair`` 自动修复。

公开函数：
  - :func:`parse_model_step` —— 把原始响应解析成 :class:`ModelStep`。
  - :func:`parse_plan` —— 把原始响应解析成 plan（步骤描述列表）。
"""
from __future__ import annotations

import json
import re
from typing import Any

import json_repair

from data_agent_langchain.agents.runtime import ModelStep
from data_agent_langchain.exceptions import ModelResponseParseError


# ---------------------------------------------------------------------------
# 底层辅助函数
# ---------------------------------------------------------------------------

def strip_json_fence(raw: str) -> str:
    """移除 Markdown 的 ```` ```json ... ``` ```` 围栏（如果有）。"""
    text = raw.strip()
    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def fix_trailing_bracket(text: str) -> str:
    """修复 LLM 常见错误：``}]}`` -> ``}}``。"""
    return re.sub(r"\}\s*\]\s*\}\s*$", "}}", text.strip(), count=1)


def try_strict_json(text: str) -> dict[str, Any] | None:
    """尝试严格 JSON 解码，失败返回 ``None``。"""
    try:
        payload, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    remainder = re.sub(r"(?:\\[nrt])+", "", text[end:].strip()).strip()
    if remainder:
        return None
    return payload if isinstance(payload, dict) else None


def load_json_object(text: str) -> dict[str, Any]:
    """多层 fallback 解析 *text* 为 dict。

    Tier 1：严格解析（原文 + 括号修复版本）。
    Tier 2：``json_repair`` 自动修复（原文 + 括号修复版本）。
    全部失败时抛 :class:`ModelResponseParseError`。
    """
    text = text.strip()
    # Tier 1
    for variant in (text, fix_trailing_bracket(text)):
        result = try_strict_json(variant)
        if result is not None:
            return result
    # Tier 2
    last_exc: BaseException | None = None
    for variant in (text, fix_trailing_bracket(text)):
        try:
            repaired = json_repair.loads(variant)
        except Exception as exc:
            last_exc = exc
            continue
        if isinstance(repaired, dict):
            return repaired
    raise ModelResponseParseError("Model response is not valid JSON.") from last_exc


# ---------------------------------------------------------------------------
# 高层解析函数
# ---------------------------------------------------------------------------

def parse_model_step(raw: str) -> ModelStep:
    """把 LLM 原始响应解析成 :class:`ModelStep`。"""
    payload = load_json_object(strip_json_fence(raw))
    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ModelResponseParseError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ModelResponseParseError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ModelResponseParseError("action_input must be a JSON object.")
    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw,
    )


def parse_plan(raw: str) -> list[str]:
    """把 LLM 原始响应解析成 plan 步骤描述列表。"""
    payload = load_json_object(strip_json_fence(raw))
    plan = payload.get("plan", [])
    if not isinstance(plan, list) or not plan:
        raise ModelResponseParseError("plan must be a non-empty list of step descriptions.")
    return [str(s) for s in plan]


__all__ = [
    "fix_trailing_bracket",
    "load_json_object",
    "parse_model_step",
    "parse_plan",
    "strip_json_fence",
    "try_strict_json",
]