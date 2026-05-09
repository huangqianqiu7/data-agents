"""
JSON 解析工具（统一版）。

消除 react.py 和 plan_and_solve.py 中的重复实现，提供统一的多层降级解析：
  严格解析 → 修复常见括号错误 → json_repair 自动修复

公开函数：
  - parse_model_step(raw) → ModelStep
  - parse_plan(raw)       → list[str]
"""
from __future__ import annotations

import json
import re

import json_repair

from data_agent_baseline.agents_v2.model import ModelStep


# =====================================================================
# 底层解析工具
# =====================================================================
def strip_json_fence(raw: str) -> str:
    """去除 Markdown ```json ... ``` 代码块标记。"""
    text = raw.strip()
    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


def fix_trailing_bracket(text: str) -> str:
    """修复 JSON 结尾 `}]}` → `}}`。"""
    return re.sub(r"\}\s*\]\s*\}\s*$", "}}", text.strip(), count=1)


def try_strict_json(text: str) -> dict | None:
    """严格解析 JSON，失败返回 None。"""
    try:
        payload, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    remainder = re.sub(r"(?:\\[nrt])+", "", text[end:].strip()).strip()
    if remainder:
        return None
    return payload if isinstance(payload, dict) else None


def load_json_object(text: str) -> dict:
    """多层降级 JSON 解析：严格 → 修复括号 → json_repair。"""
    text = text.strip()
    # 第一阶段：严格解析（原版 + 修复括号版）
    for variant in (text, fix_trailing_bracket(text)):
        result = try_strict_json(variant)
        if result is not None:
            return result
    # 第二阶段：json_repair 自动修复
    last_exc: BaseException | None = None
    for variant in (text, fix_trailing_bracket(text)):
        try:
            repaired = json_repair.loads(variant)
        except Exception as exc:
            last_exc = exc
            continue
        if isinstance(repaired, dict):
            return repaired
    raise ValueError("Model response is not valid JSON.") from last_exc


# =====================================================================
# 高层解析函数
# =====================================================================
def parse_model_step(raw: str) -> ModelStep:
    """解析大模型输出为 ModelStep（thought / action / action_input）。"""
    payload = load_json_object(strip_json_fence(raw))
    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")
    return ModelStep(
        thought=thought, action=action,
        action_input=action_input, raw_response=raw,
    )


def parse_plan(raw: str) -> list[str]:
    """解析大模型输出的计划列表。"""
    payload = load_json_object(strip_json_fence(raw))
    plan = payload.get("plan", [])
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must be a non-empty list of step descriptions.")
    return [str(s) for s in plan]
