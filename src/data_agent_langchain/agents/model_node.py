"""
``model_node`` —— 唯一允许递增 ``step_index`` 的节点（v3 D2）。

LANGCHAIN_MIGRATION_PLAN.md §11.1.1：

  - 根据当前 state 拼装提示词（ReAct 或 Plan-and-Solve 两种）。
  - 调用 LLM，重试 + 响应抽取逻辑下沉到 :mod:`agents.model_retry`。
  - 无论成败都把 ``step_index`` 加 1 写回 partial state。
  - 重试耗尽时追加 ``__error__`` StepRecord 并把
    ``last_error_kind = "model_error"``，使条件边把这一步算作"已消费"
    且不会触发 replan（v4 M5：``model_error`` 不在
    ``_REPLAN_TRIGGER_ERROR_KINDS`` 白名单中）。

LLM 注入方式：

  - 单元测试通过 ``RunnableConfig.configurable["llm"]`` 注入
    ``FakeListChatModel`` 等测试用 Runnable。
  - 生产 runner（Phase 5）注入真实 ``ChatOpenAI``。
  - 都未注入时，本模块懒加载
    ``data_agent_langchain.llm.factory.build_chat_model``。

成功路径返回的 partial state：

  ``{"step_index": ..., "raw_response": "...", "thought": "", "action": "", "action_input": {}}``

失败路径返回的 partial state：

  ``{"step_index": ..., "raw_response": "", "steps": [error_step], "last_tool_ok": False, "last_error_kind": "model_error"}``

注意：成功时把 ``thought`` / ``action`` / ``action_input`` 重置为空，让
``parse_action_node`` 来填；不重置会让上一轮的旧值污染本轮 gate 决策。
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from data_agent_langchain.agents.model_retry import (
    ModelExhaustedError,
    call_model_with_retry,
    extract_raw_response,
)
from data_agent_langchain.agents.prompts import (
    build_plan_solve_execution_messages,
    build_react_messages,
)
from data_agent_langchain.agents.runtime import StepRecord
from data_agent_langchain.config import AppConfig, default_app_config
from data_agent_langchain.observability.events import dispatch_observability_event
from data_agent_langchain.runtime.context import get_current_app_config
from data_agent_langchain.runtime.rehydrate import rehydrate_task
from data_agent_langchain.runtime.state import RunState

# 别名导出：保持向后兼容历史调用点 ``from agents.model_node import _extract_raw_response``
# 与 ``_call_model_with_retry``。新代码请直接 ``from agents.model_retry import ...``。
_extract_raw_response = extract_raw_response
_call_model_with_retry = call_model_with_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 公共节点
# ---------------------------------------------------------------------------

def model_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    """调用 LLM、递增 step_index、写入 raw_response。"""
    app_config = _safe_get_app_config()
    cur_step = int(state.get("step_index", 0) or 0) + 1

    # 步骤预算已耗尽：通过 advance_node 短路到 ``done``。这里仍然把
    # step_index 加 1 写入 trace，以保留 attempt 记录（与 legacy 行为一致）。
    max_steps = int(state.get("max_steps") or app_config.agent.max_steps)
    if cur_step > max_steps:
        return {
            "step_index": cur_step,
            "last_error_kind": "max_steps_exceeded",
        }

    llm = _resolve_llm(config, app_config)

    # 根据 state 构造消息。ReAct 与 Plan-and-Solve 仅在 prompt 拼装环节
    # 有差异；execution_subgraph 共用本节点（D7）。
    try:
        messages = _build_messages_for_state(state, app_config)
    except Exception as exc:
        # prompt 拼装失败按 model_error 处理，让循环可以恢复。
        logger.exception("[model_node] prompt assembly failed: %s", exc)
        return _model_error_update(state, cur_step, exc, config)

    # 调用 LLM。``raw_response`` 语义：
    #   - tool_calling: ``json.dumps(ai_message.tool_calls)``（C9）
    #   - json_action:  ``ai_message.content``
    # 优先取 RunState.action_mode（Phase 5 runner 启动时把 AppConfig 值复制
    # 到 state，E13），允许 per-run override。
    try:
        raw_response = call_model_with_retry(
            llm.invoke,
            messages,
            step_index=cur_step,
            max_retries=int(getattr(app_config.agent, "max_model_retries", 3)),
            retry_backoff=tuple(getattr(app_config.agent, "model_retry_backoff", (2.0, 5.0, 10.0))),
            timeout_seconds=float(getattr(app_config.agent, "model_timeout_s", 120.0)),
            action_mode=str(state.get("action_mode") or app_config.agent.action_mode),
            # 把 LangGraph 透传给本节点的 RunnableConfig 继续向下游 LLM 转发，
            # 否则 callbacks（含 MetricsCollector）链路会断，导致 metrics.tokens=0。
            config=config,
        )
    except ModelExhaustedError as exc:
        return _model_error_update(state, cur_step, exc, config)

    # 重置本轮的临时字段，让 parse_action_node 来填。
    return {
        "step_index": cur_step,
        "raw_response": raw_response,
        "thought": "",
        "action": "",
        "action_input": {},
        # 同时清掉上一轮的路由字段，避免污染本轮 gate / advance 决策。
        "last_tool_ok": None,
        "last_tool_is_terminal": False,
        "last_error_kind": None,
        "skip_tool": False,
    }


# ---------------------------------------------------------------------------
# 提示词组装（按 state["mode"] 选择 builder）
# ---------------------------------------------------------------------------

def _build_messages_for_state(state: RunState, app_config: AppConfig) -> list[BaseMessage]:
    """按 ``state["mode"]`` 选择对应的 prompt builder。"""
    task = rehydrate_task(state)
    steps = list(state.get("steps") or [])
    action_mode = str(state.get("action_mode") or app_config.agent.action_mode)
    max_obs_chars = int(getattr(app_config.agent, "max_obs_chars", 3000))
    max_context_tokens = int(getattr(app_config.agent, "max_context_tokens", 24000))

    mode = state.get("mode") or "react"
    if mode == "plan_solve":
        plan = list(state.get("plan") or [])
        plan_index = int(state.get("plan_index", 0) or 0)
        return build_plan_solve_execution_messages(
            task,
            steps,
            plan=plan,
            plan_index=plan_index,
            action_mode=action_mode,
            max_obs_chars=max_obs_chars,
            max_context_tokens=max_context_tokens,
        )

    return build_react_messages(
        task,
        steps,
        action_mode=action_mode,
        max_obs_chars=max_obs_chars,
        max_context_tokens=max_context_tokens,
    )


# ---------------------------------------------------------------------------
# 失败路径
# ---------------------------------------------------------------------------

def _model_error_update(
    state: RunState,
    step_index: int,
    exc: BaseException,
    config: Any | None = None,
) -> dict[str, Any]:
    """追加 model_error StepRecord 并设置后续路由字段。"""
    dispatch_observability_event(
        "model_error",
        {"step_index": step_index, "attempts": 1, "error": str(exc)},
        config,
    )
    return {
        "step_index": step_index,
        "raw_response": "",
        "thought": "",
        "action": "",
        "action_input": {},
        "steps": [
            StepRecord(
                step_index=step_index,
                thought="",
                action="__error__",
                action_input={},
                raw_response="",
                observation={
                    "ok": False,
                    "error": f"Model call failed: {exc}",
                },
                ok=False,
                phase=state.get("phase", "") or "",
                plan_progress=state.get("plan_progress", "") or "",
                plan_step_description=state.get("plan_step_description", "") or "",
            )
        ],
        "last_tool_ok": False,
        "last_tool_is_terminal": False,
        "last_error_kind": "model_error",
        "skip_tool": True,  # 没东西可执行，跳过下游 gate / tool。
    }


# ---------------------------------------------------------------------------
# 配置 & LLM 解析辅助
# ---------------------------------------------------------------------------

def _safe_get_app_config() -> AppConfig:
    """读取 contextvar 中的 AppConfig，未初始化时回退到默认配置（用于直跑测试）。"""
    try:
        return get_current_app_config()
    except RuntimeError:
        return default_app_config()


def _resolve_llm(config: Any | None, app_config: AppConfig) -> Runnable:
    """从 ``RunnableConfig`` 解析 LLM Runnable，未注入时退化到工厂构造。

    单测约定：``config={'configurable': {'llm': fake_runnable}}``。
    Phase 5 runner 用同样方式注入真实 ``ChatOpenAI``。
    """
    if isinstance(config, dict):
        configurable = config.get("configurable") or {}
        candidate = configurable.get("llm")
        if candidate is not None:
            return candidate

    # Phase 5 钩子：若 llm/factory 还没实现，给出明确错误信息。
    try:
        from data_agent_langchain.llm.factory import build_chat_model
    except ImportError as exc:  # pragma: no cover - depends on Phase 5 plumbing
        raise RuntimeError(
            "model_node could not resolve an LLM. Either pass "
            "config={'configurable': {'llm': <Runnable>}} or implement "
            "data_agent_langchain.llm.factory.build_chat_model (Phase 5)."
        ) from exc
    return build_chat_model(app_config)


__all__ = ["ModelExhaustedError", "model_node"]