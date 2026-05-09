"""
Sanitization protocol for trace replay.

When the agent appends a step with action ``__error__`` (parse / model
failure) or ``__replan_failed__`` (re-planning attempt failed), the original
``raw_response`` may be malformed JSON or empty. Replaying such a step back
to the LLM as an assistant message would corrupt the next round's output
(format-greedy decoding, JSON-mode confusion).

To avoid that, both backends substitute the raw response with the
``ERROR_SANITIZED_RESPONSE`` placeholder when re-rendering history. Both
constants must stay byte-for-byte identical between backends, so they live
here as the single source of truth (LANGCHAIN_MIGRATION_PLAN.md v4 E2).
"""
from __future__ import annotations

import json


# Action names whose raw_response should be replaced by the sanitized
# placeholder when re-rendering history. Frozen set so callers can compare
# membership without mutation.
SANITIZE_ACTIONS: frozenset[str] = frozenset({"__error__", "__replan_failed__"})


# Sanitized placeholder used in place of malformed / empty raw_response when
# replaying a sanitize-action step back to the LLM. The wording is chosen so
# the model self-corrects on the next turn.
ERROR_SANITIZED_RESPONSE: str = json.dumps(
    {
        "thought": "My previous response had a format error. "
                   "I must return a valid JSON with keys: thought, action, action_input.",
        "action": "__error__",
        "action_input": {},
    },
    ensure_ascii=False,
)
