"""
Model adapter layer — standard data structures and protocols for LLM interaction.

Contains:
  - :class:`ModelMessage` (re-exported from ``data_agent_common.agents.runtime``)
  - :class:`ModelStep`    (re-exported from ``data_agent_common.agents.runtime``)
  - :class:`ModelAdapter`: structural typing protocol for all adapters
  - :class:`OpenAIModelAdapter`: production adapter (OpenAI-compatible APIs)
  - :class:`ScriptedModelAdapter`: deterministic adapter for unit tests

``ModelMessage`` and ``ModelStep`` are shared with the new LangGraph backend
(``data_agent_langchain``); see LANGCHAIN_MIGRATION_PLAN.md §3.1.
"""
from __future__ import annotations

from typing import Protocol

from openai import APIError, OpenAI

# Re-exported from data_agent_common so the legacy import path
# `from data_agent_refactored.agents.model import ModelMessage, ModelStep`
# still resolves while the canonical home is the shared package.
from data_agent_common.agents.runtime import ModelMessage, ModelStep
from data_agent_refactored.exceptions import ModelCallError


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class ModelAdapter(Protocol):
    """Structural protocol that all model adapters must satisfy."""

    def complete(self, messages: list[ModelMessage]) -> str:
        """Send *messages* to the LLM and return its text response."""
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter
# ---------------------------------------------------------------------------

class OpenAIModelAdapter:
    """Adapter for any service implementing the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        if not api_key:
            raise ModelCallError(0, "Missing model API key in config.agent.api_key.")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def complete(self, messages: list[ModelMessage]) -> str:
        """Send a chat completion request and return the text content."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=self.temperature,
            )
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

        # Defensive: gateway may return a plain string instead of JSON.
        if isinstance(response, str):
            snippet = response.strip().replace("\n", " ")[:400]
            raise RuntimeError(
                "The HTTP API returned a plain string instead of an OpenAI-style chat completion JSON. "
                "Usually this means api_base is wrong or the gateway is not OpenAI-compatible. "
                f"Body begins with: {snippet!r}"
            )

        if not hasattr(response, "choices"):
            raise RuntimeError(
                f"Unexpected chat completion type: {type(response).__name__!r} "
                "(expected an object with `.choices`)."
            )

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content


# ---------------------------------------------------------------------------
# Testing adapter
# ---------------------------------------------------------------------------

class ScriptedModelAdapter:
    """Return pre-set responses in order — used exclusively for testing."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
