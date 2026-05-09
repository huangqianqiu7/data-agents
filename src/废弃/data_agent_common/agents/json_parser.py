"""
Unified JSON parsing for LLM outputs.

Provides multi-tier fallback parsing:
  strict JSON → bracket fix → ``json_repair`` auto-repair

Public functions:
  - :func:`parse_model_step` — parse a raw LLM response into a :class:`ModelStep`.
  - :func:`parse_plan` — parse a raw LLM response into a plan (list of step descriptions).

Used by both backends:
  - Legacy ``data_agent_refactored.agents.{plan_solve_agent,react_agent}``.
  - LangGraph ``data_agent_langchain.agents.parse_action_node`` (json_action mode,
    see LANGCHAIN_MIGRATION_PLAN.md §9.2.1).
"""
from __future__ import annotations

import json
import re
from typing import Any

import json_repair

from data_agent_common.agents.runtime import ModelStep
from data_agent_common.exceptions import ModelResponseParseError


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def strip_json_fence(raw: str) -> str:
    """Remove Markdown ````` ```json ... ``` ````` fencing if present."""
    text = raw.strip()
    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def fix_trailing_bracket(text: str) -> str:
    """Fix a common LLM mistake: ``}]}`` → ``}}``."""
    return re.sub(r"\}\s*\]\s*\}\s*$", "}}", text.strip(), count=1)


def try_strict_json(text: str) -> dict[str, Any] | None:
    """Attempt strict JSON decoding.  Returns ``None`` on failure."""
    try:
        payload, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    remainder = re.sub(r"(?:\\[nrt])+", "", text[end:].strip()).strip()
    if remainder:
        return None
    return payload if isinstance(payload, dict) else None


def load_json_object(text: str) -> dict[str, Any]:
    """Parse *text* into a JSON dict with multi-tier fallback."""
    text = text.strip()
    # Tier 1: strict parse (original + bracket-fixed variant)
    for variant in (text, fix_trailing_bracket(text)):
        result = try_strict_json(variant)
        if result is not None:
            return result
    # Tier 2: json_repair auto-fix
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
# High-level parsers
# ---------------------------------------------------------------------------

def parse_model_step(raw: str) -> ModelStep:
    """Parse a raw LLM response into a :class:`ModelStep`."""
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
    """Parse a raw LLM response into a list of plan step descriptions."""
    payload = load_json_object(strip_json_fence(raw))
    plan = payload.get("plan", [])
    if not isinstance(plan, list) or not plan:
        raise ModelResponseParseError("plan must be a non-empty list of step descriptions.")
    return [str(s) for s in plan]
