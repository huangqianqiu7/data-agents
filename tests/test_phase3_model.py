"""
Phase 3 — unit tests for ``model_node`` (E4 retry-and-record + D2 step_index).

Covers:
  - Successful invocation increments ``step_index`` exactly once.
  - Successful response writes ``raw_response`` (json_action mode).
  - Tool-calling mode: AIMessage with one tool_call serialised as JSON
    list (C9), and parse_action_node accepts it on the next turn.
  - Failure path: all retries exhausted -> ``last_error_kind=model_error``,
    ``__error__`` step appended, ``step_index`` still increments (D2).
  - Step budget exhausted -> ``last_error_kind=max_steps_exceeded`` and
    no LLM call.

The model is injected via ``RunnableConfig.configurable['llm']`` so we
don't depend on the Phase 5 ``build_chat_model`` factory.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from data_agent_langchain.agents.model_node import (
    ModelExhaustedError,
    _call_model_with_retry,
    _extract_raw_response,
    model_node,
)
from data_agent_langchain.config import AgentConfig, AppConfig, ToolsConfig
from data_agent_langchain.runtime.context import _APP_CONFIG


@pytest.fixture()
def synthetic_task(tmp_path: Path) -> dict:
    root = tmp_path
    task_dir = root / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_test", "difficulty": "easy", "question": "Q?"}),
        encoding="utf-8",
    )
    csv_path = context_dir / "x.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["a"], [1]])
    return {"dataset_root": str(root), "task_id": "task_test"}


@pytest.fixture()
def fast_app_config():
    """AppConfig with retry backoff = (0,) so failure tests don't sleep."""
    cfg = AppConfig(
        agent=AgentConfig(
            max_model_retries=2,
            model_retry_backoff=(0.0,),
            model_timeout_s=5.0,
        ),
        tools=ToolsConfig(),
    )
    token = _APP_CONFIG.set(cfg)
    try:
        yield cfg
    finally:
        _APP_CONFIG.reset(token)


def _base_state(task_state: dict, **extra) -> dict:
    base = {
        **task_state,
        "step_index": 0,
        "max_steps": 16,
        "mode": "react",
        "action_mode": "json_action",
        "phase": "execution",
        "plan_progress": "",
        "plan_step_description": "",
        "steps": [],
    }
    base.update(extra)
    return base


# --- Happy path -----------------------------------------------------

def test_model_node_increments_step_index_and_writes_raw_response(synthetic_task, fast_app_config):
    fake_response = '```json\n{"thought":"t","action":"list_context","action_input":{}}\n```'
    llm = FakeListChatModel(responses=[fake_response])
    update = model_node(_base_state(synthetic_task), config={"configurable": {"llm": llm}})
    assert update["step_index"] == 1
    assert "list_context" in update["raw_response"]
    # Per-turn fields reset:
    assert update["thought"] == ""
    assert update["action"] == ""
    assert update["action_input"] == {}
    assert update["last_error_kind"] is None
    assert update["skip_tool"] is False


# --- tool_calling mode ----------------------------------------------

def test_model_node_tool_calling_serialises_tool_calls(synthetic_task, fast_app_config):
    """In tool_calling mode the AIMessage tool_calls list is JSON-serialised (C9)."""
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "list_context", "args": {"max_depth": 4}, "id": "x"}],
    )
    llm = RunnableLambda(lambda _: msg)
    state = _base_state(synthetic_task, action_mode="tool_calling")
    update = model_node(state, config={"configurable": {"llm": llm}})
    decoded = json.loads(update["raw_response"])
    assert isinstance(decoded, list)
    assert decoded[0]["name"] == "list_context"


def test_model_node_tool_calling_prompt_requests_tool_calls(synthetic_task, fast_app_config):
    captured = []

    def capture(messages):
        captured.extend(messages)
        return AIMessage(content="", tool_calls=[{"name": "list_context", "args": {}, "id": "x"}])

    state = _base_state(synthetic_task, action_mode="tool_calling")
    model_node(state, config={"configurable": {"llm": RunnableLambda(capture)}})

    system_text = str(captured[0].content)
    prompt_text = "\n".join(str(message.content) for message in captured)
    assert "tool call" in system_text.lower()
    assert "```json" not in system_text
    assert "Return a JSON with keys `thought`, `action`, `action_input`" not in prompt_text
def test_model_node_tool_calling_plan_solve_prompt_requests_tool_calls(synthetic_task, fast_app_config):
    captured = []

    def capture(messages):
        captured.extend(messages)
        return AIMessage(content="", tool_calls=[{"name": "list_context", "args": {}, "id": "x"}])

    state = _base_state(
        synthetic_task,
        action_mode="tool_calling",
        mode="plan_solve",
        plan=["List context files"],
        plan_index=0,
    )
    model_node(state, config={"configurable": {"llm": RunnableLambda(capture)}})

    system_text = str(captured[0].content)
    prompt_text = "\n".join(str(message.content) for message in captured)
    assert "tool call" in system_text.lower()
    assert "```json" not in system_text
    assert "Return a JSON with keys `thought`, `action`, `action_input`" not in prompt_text
def test_extract_raw_response_handles_string_and_aimessage():
    assert _extract_raw_response("hello", action_mode="json_action") == "hello"
    assert _extract_raw_response(AIMessage(content="x"), action_mode="json_action") == "x"
    msg = AIMessage(content="", tool_calls=[{"name": "answer", "args": {}, "id": "1"}])
    out = _extract_raw_response(msg, action_mode="tool_calling")
    decoded = json.loads(out)
    assert decoded[0]["name"] == "answer"


# --- Retry exhausted ------------------------------------------------

def test_model_node_returns_model_error_on_exhaustion(synthetic_task, fast_app_config):
    def boom(_):
        raise RuntimeError("boom")
    llm = RunnableLambda(boom)
    update = model_node(_base_state(synthetic_task), config={"configurable": {"llm": llm}})
    # step_index still advances on failure (D2).
    assert update["step_index"] == 1
    assert update["raw_response"] == ""
    assert update["last_error_kind"] == "model_error"
    assert update["last_tool_ok"] is False
    [step] = update["steps"]
    assert step.action == "__error__"
    assert step.ok is False
    assert "Model call failed" in step.observation["error"]


def test_call_model_with_retry_retries_and_records():
    calls = []
    def flaky(messages):
        calls.append(messages)
        if len(calls) < 2:
            raise RuntimeError("flaky")
        return AIMessage(content="ok")
    raw = _call_model_with_retry(
        flaky, [], step_index=1,
        max_retries=3, retry_backoff=(0.0,), timeout_seconds=5.0,
        action_mode="json_action",
    )
    assert raw == "ok"
    assert len(calls) == 2


def test_call_model_with_retry_raises_when_exhausted():
    def boom(_):
        raise RuntimeError("boom")
    with pytest.raises(ModelExhaustedError):
        _call_model_with_retry(
            boom, [], step_index=1,
            max_retries=2, retry_backoff=(0.0,), timeout_seconds=5.0,
            action_mode="json_action",
        )


# --- Step budget exhausted -----------------------------------------

def test_model_node_step_budget_exhausted_returns_max_steps_kind(synthetic_task, fast_app_config):
    state = _base_state(synthetic_task, step_index=16, max_steps=16)
    # No LLM should be called -> use a model that would explode if invoked.
    def boom(_):  # pragma: no cover - should not run
        raise AssertionError("LLM should not have been called")
    update = model_node(state, config={"configurable": {"llm": RunnableLambda(boom)}})
    assert update["step_index"] == 17
    assert update["last_error_kind"] == "max_steps_exceeded"


