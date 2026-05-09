"""
Phase 3 — unit tests for ``finalize_node`` and ``build_run_result``.

Covers:
  - D14 byte-for-byte ``_MAX_STEPS_FAILURE_MSG``.
  - Answer already present → ``failure_reason`` stays ``None``.
  - Failure reason already set → ``finalize_node`` does not overwrite.
  - ``build_run_result`` returns a populated ``AgentRunResult``.
"""
from __future__ import annotations

from data_agent_common.benchmark.schema import AnswerTable
from data_agent_common.agents.runtime import StepRecord
from data_agent_langchain.agents.finalize import (
    _MAX_STEPS_FAILURE_MSG,
    build_run_result,
    finalize_node,
)


def test_finalize_sets_failure_when_no_answer_and_no_reason():
    state: dict = {"answer": None, "failure_reason": None}
    update = finalize_node(state)
    assert update["failure_reason"] == _MAX_STEPS_FAILURE_MSG


def test_finalize_keeps_existing_failure_reason():
    state = {"answer": None, "failure_reason": "custom"}
    update = finalize_node(state)
    assert update["failure_reason"] == "custom"


def test_finalize_noop_when_answer_present():
    at = AnswerTable(columns=["x"], rows=[[1]])
    state = {"answer": at, "failure_reason": None}
    update = finalize_node(state)
    assert update["failure_reason"] is None


def test_max_steps_failure_msg_is_byte_for_byte():
    assert _MAX_STEPS_FAILURE_MSG == "Agent did not submit an answer within max_steps."


def test_build_run_result_fills_failure_reason():
    result = build_run_result("task_1", {"answer": None, "failure_reason": None, "steps": []})
    assert result.task_id == "task_1"
    assert result.answer is None
    assert result.failure_reason == _MAX_STEPS_FAILURE_MSG
    assert result.steps == []


def test_build_run_result_with_answer_and_steps():
    at = AnswerTable(columns=["x"], rows=[[1]])
    step = StepRecord(
        step_index=1, thought="", action="answer", action_input={},
        raw_response="", observation={"ok": True}, ok=True,
    )
    result = build_run_result("task_2", {"answer": at, "failure_reason": None, "steps": [step]})
    assert result.answer is at
    assert result.steps == [step]
    assert result.failure_reason is None
    assert result.succeeded is True
