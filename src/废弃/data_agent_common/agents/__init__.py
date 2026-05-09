"""Cross-backend agent primitives (runtime objects, parser, sanitize, gate, helpers)."""
from data_agent_common.agents.gate import (
    DATA_PREVIEW_ACTIONS,
    GATED_ACTIONS,
    GATE_REMINDER,
    has_data_preview,
)
from data_agent_common.agents.json_parser import (
    fix_trailing_bracket,
    load_json_object,
    parse_model_step,
    parse_plan,
    strip_json_fence,
    try_strict_json,
)
from data_agent_common.agents.prompts import build_observation_prompt
from data_agent_common.agents.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    ModelMessage,
    ModelStep,
    StepRecord,
)
from data_agent_common.agents.sanitize import (
    ERROR_SANITIZED_RESPONSE,
    SANITIZE_ACTIONS,
)
from data_agent_common.agents.text_helpers import (
    estimate_tokens,
    preview_json,
    progress,
    truncate_observation,
)

__all__ = [
    # Runtime
    "AgentRunResult",
    "AgentRuntimeState",
    "ModelMessage",
    "ModelStep",
    "StepRecord",
    # JSON parsing
    "fix_trailing_bracket",
    "load_json_object",
    "parse_model_step",
    "parse_plan",
    "strip_json_fence",
    "try_strict_json",
    # Sanitize protocol
    "ERROR_SANITIZED_RESPONSE",
    "SANITIZE_ACTIONS",
    # Gate constants & predicate
    "DATA_PREVIEW_ACTIONS",
    "GATED_ACTIONS",
    "GATE_REMINDER",
    "has_data_preview",
    # Prompt / text helpers
    "build_observation_prompt",
    "estimate_tokens",
    "preview_json",
    "progress",
    "truncate_observation",
]
