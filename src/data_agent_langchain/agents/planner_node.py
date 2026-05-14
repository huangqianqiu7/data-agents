"""
``planner_node`` 与 ``replanner_node`` ——  Plan-and-Solve 的规划入口。

planner_node：
  - 任务首次进入时调用 LLM 生成 plan（``list[str]``）。
  - LLM 失败时退化为 :data:`FALLBACK_PLAN`，让流程仍可继续推进。
  - 把生成结果作为一条 ``__plan_generation__`` StepRecord 记录到 trace。

replanner_node：
  - tool 失败且仍有 replan 预算时被外层图调用。
  - 用 ``_build_replan_history_hint`` 把 "已成功步骤 + 最后失败动作"
    塞给 LLM，让新计划跳过已完成步骤。
  - 失败时不抛异常，写一条 ``__replan_failed__`` StepRecord 让 sanitize
    协议在下一轮替换它。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable

from data_agent_langchain.agents.corpus_recall import recall_corpus_snippets
from data_agent_langchain.agents.json_parser import parse_plan
from data_agent_langchain.agents.memory_recall import recall_dataset_facts
from data_agent_langchain.agents.model_retry import extract_raw_response
from data_agent_langchain.agents.prompts import build_planning_messages
from data_agent_langchain.agents.runtime import StepRecord
from data_agent_langchain.agents.text_helpers import preview_json
from data_agent_langchain.config import AppConfig, default_app_config
from data_agent_langchain.observability.events import dispatch_observability_event
from data_agent_langchain.runtime.context import get_current_app_config
from data_agent_langchain.runtime.rehydrate import rehydrate_task
from data_agent_langchain.runtime.state import RunState
from data_agent_langchain.tools.timeout import call_with_timeout

# LLM 不可用时的最小可用计划，保证图至少能走到 finalize。
FALLBACK_PLAN = ["List context files", "Inspect data", "Solve and call answer"]


def planner_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    """生成首份 plan；失败时使用 FALLBACK_PLAN。"""
    try:
        plan = _generate_plan_from_state(state, config=config)
        plan_ok = True
    except Exception:
        plan = list(FALLBACK_PLAN)
        plan_ok = False
    output = {
        "plan": plan,
        "plan_index": 0,
        "replan_used": int(state.get("replan_used", 0) or 0),
        "steps": [_planning_step(plan, ok=plan_ok)],
    }
    app_config = _safe_get_app_config()
    dataset_name = Path(state.get("dataset_root", "") or ".").name or "default"
    dataset_hits = recall_dataset_facts(
        memory_cfg=app_config.memory,
        dataset=dataset_name,
        node="planner_node",
        config=config,
    )
    corpus_hits = recall_corpus_snippets(
        app_config.memory.rag,
        task_id=str(state.get("task_id") or ""),
        query=str(state.get("question") or ""),
        node="planner_node",
        config=config,
    )
    memory_hits = list(dataset_hits) + list(corpus_hits)
    if memory_hits:
        output["memory_hits"] = memory_hits
    return output


def replanner_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    """重新生成 plan；失败时写 ``__replan_failed__`` step。"""
    try:
        plan = _generate_plan_from_state(
            state,
            config=config,
            history_hint=_build_replan_history_hint(state),
        )
    except Exception as exc:
        dispatch_observability_event(
            "replan_failed",
            {"step_index": int(state.get("step_index", 0) or 0), "error": str(exc)},
            config,
        )
        return {
            "steps": [
                StepRecord(
                    step_index=-1,
                    thought="",
                    action="__replan_failed__",
                    action_input={},
                    raw_response="",
                    observation={"ok": False, "error": f"Re-plan failed: {exc}"},
                    ok=False,
                    phase="execution",
                )
            ],
            "replan_used": int(state.get("replan_used", 0) or 0) + 1,
        }
    dispatch_observability_event(
        "replan_triggered",
        {
            "step_index": int(state.get("step_index", 0) or 0),
            "replan_used": int(state.get("replan_used", 0) or 0) + 1,
            "reason": str(state.get("last_error_kind") or ""),
        },
        config,
    )
    return {
        "plan": plan,
        "plan_index": 0,
        "replan_used": int(state.get("replan_used", 0) or 0) + 1,
    }


def _generate_plan_from_state(
    state: RunState,
    config: Any | None = None,
    *,
    history_hint: str = "",
) -> list[str]:
    """构造 planning 提示词、调 LLM、解析 plan。"""
    app_config = _safe_get_app_config()
    llm = _resolve_llm(config, app_config)
    task = rehydrate_task(state)
    messages = build_planning_messages(task, history_hint=history_hint)
    # 把 LangGraph 透传给本节点的 RunnableConfig 继续向下游 LLM 转发，否则
    # callbacks（含 MetricsCollector）链路会断，plan 阶段的 LLM token usage
    # 不会进 metrics.json（详见 ``test_planner_node_propagates_runnable_
    # config_to_llm_invoke``）。``config=None`` 时退化为 legacy 行为。
    invoke_kwargs: dict[str, Any] | None = (
        {"config": config} if config is not None else None
    )
    response = call_with_timeout(
        llm.invoke,
        (messages,),
        float(getattr(app_config.agent, "model_timeout_s", 120.0)),
        kwargs=invoke_kwargs,
    )
    raw = extract_raw_response(response, action_mode="json_action")
    return parse_plan(raw)


def _build_replan_history_hint(state: RunState) -> str:
    """把成功步骤 + 最后失败动作总结给 LLM，引导新 plan 跳过已做的部分。"""
    completed: list[str] = []
    failed_last: list[str] = []
    steps = list(state.get("steps") or [])
    for step in steps:
        preview = preview_json(step.observation, 200)
        if step.ok:
            completed.append(
                f"  ✓ {step.action}({preview_json(step.action_input, 80)}) → {preview}"
            )
        elif step == steps[-1]:
            failed_last.append(
                f"  ✗ {step.action}({preview_json(step.action_input, 80)}) → {preview}"
            )
    parts: list[str] = []
    if completed:
        parts.append("Already completed (do NOT repeat):\n" + "\n".join(completed))
    if failed_last:
        parts.append("Last failed action:\n" + "\n".join(failed_last))
    parts.append(
        "IMPORTANT: Your new plan must skip steps that have already succeeded. "
        "Start from the next logical action."
    )
    return "\n\n".join(parts)


def _planning_step(plan: list[str], *, ok: bool) -> StepRecord:
    """把生成的 plan 包成一条 ``__plan_generation__`` StepRecord 写入 trace。"""
    return StepRecord(
        step_index=0,
        thought="",
        action="__plan_generation__",
        action_input={},
        raw_response="",
        observation={"ok": ok, "plan": plan},
        ok=ok,
        phase="planning",
    )


def _safe_get_app_config() -> AppConfig:
    try:
        return get_current_app_config()
    except RuntimeError:
        return default_app_config()


def _resolve_llm(config: Any | None, app_config: AppConfig) -> Runnable:
    """从 ``RunnableConfig`` 取 LLM；未注入时退化到工厂构造。"""
    if isinstance(config, dict):
        configurable = config.get("configurable") or {}
        candidate = configurable.get("llm")
        if candidate is not None:
            return candidate
    from data_agent_langchain.llm.factory import build_chat_model
    return build_chat_model(app_config)


__all__ = [
    "FALLBACK_PLAN",
    "planner_node",
    "replanner_node",
]
