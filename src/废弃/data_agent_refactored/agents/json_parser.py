"""
Backwards-compat shim — JSON parser moved to
``data_agent_common.agents.json_parser``.
"""
from __future__ import annotations

from data_agent_common.agents.json_parser import (
    fix_trailing_bracket,
    load_json_object,
    parse_model_step,
    parse_plan,
    strip_json_fence,
    try_strict_json,
)

__all__ = [
    "fix_trailing_bracket",
    "load_json_object",
    "parse_model_step",
    "parse_plan",
    "strip_json_fence",
    "try_strict_json",
]
