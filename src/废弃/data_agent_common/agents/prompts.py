"""
Cross-backend prompt builders.

Holds only the prompt helpers that **both** backends call so the prompts
sent to the LLM stay byte-for-byte identical in ``json_action`` mode. The
big system-prompt / few-shot constants stay in the legacy
``data_agent_refactored.agents.prompt`` module because the new LangGraph
backend re-renders them via ``ChatPromptTemplate`` rather than concatenating
strings.

Public functions:
  - :func:`build_observation_prompt` — render a tool observation dict as
    a user-message string prefixed with ``"Observation:"``.
"""
from __future__ import annotations

import json
from typing import Any


def build_observation_prompt(observation: dict[str, Any]) -> str:
    """Render *observation* as a ``user`` message body.

    The format intentionally matches the legacy version
    (``data_agent_refactored.agents.prompt.build_observation_prompt``)
    character-for-character so message replays are identical between
    backends in ``json_action`` mode.
    """
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"


__all__ = ["build_observation_prompt"]
