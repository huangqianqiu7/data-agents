"""
``parse_action_node`` —— 把 ``model_node`` 写入的 ``raw_response`` 转换为
下游 gate / tool / advance 节点期待的 ``thought`` / ``action`` /
``action_input`` 字段。

§9.2.1（v4 M2）的完整 I/O 契约：

  输入（来自 RunState）：
    - ``raw_response``      ``model_node`` 保证写入。
    - ``action_mode``       ``"json_action"`` 或 ``"tool_calling"``。
    - ``step_index``        parse_error step 复用此计数器。
    - ``phase``/``plan_*``  parse_error step 透传，让 trace.json 保留
                            阶段 / plan 来源信息。

  成功输出：
    - ``thought``        字符串（tool_calling 下可为 ``""``）
    - ``action``         工具名
    - ``action_input``   dict

  失败输出（parse_error 路径）：
    - ``steps`` += [parse_error StepRecord]   （由 reducer 追加）
    - ``last_tool_ok = False``
    - ``last_error_kind = "parse_error"``

本节点 **绝不能** 抛异常；pydantic / json 错误都被捕获并转成 parse_error
step。``step_index`` 不在这里递增 —— 仅 ``model_node`` 拥有该权限（D2）。

C15 多 tool_calls 拒绝：tool-calling 模式下若返回多于一条 ``tool_calls``，
本节点立即拒绝并要求模型重试，绝不静默取第一条。空 ``tool_calls`` 同样
走 parse_error 路径。
"""
from __future__ import annotations

import json
from typing import Any

from data_agent_langchain.agents.json_parser import parse_model_step
from data_agent_langchain.agents.runtime import StepRecord
from data_agent_langchain.observability.events import dispatch_observability_event
from data_agent_langchain.runtime.state import RunState


# ---------------------------------------------------------------------------
# 公共节点
# ---------------------------------------------------------------------------

def parse_action_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    """把 ``state["raw_response"]`` 解析成 action / action_input 字段。

    完整契约见模块 docstring。本节点是纯 Python；LLM / tool I/O 在邻居
    节点里发生。
    """
    raw = state.get("raw_response", "") or ""
    mode = state.get("action_mode", "json_action") or "json_action"

    if mode == "tool_calling":
        return _parse_tool_calling(state, raw, config)
    return _parse_json_action(state, raw, config)


# ---------------------------------------------------------------------------
# 模式专属解析器
# ---------------------------------------------------------------------------

def _parse_tool_calling(state: RunState, raw: str, config: Any | None = None) -> dict[str, Any]:
    """解码并校验一段 ``tool_calls`` JSON 列表。

    ``model_node`` 在 tool-calling 模式下写入
    ``raw_response = json.dumps(ai_message.tool_calls)``（§11.3 / C9），
    所以这里 *raw* 总是一个 JSON 数组（或空串）。
    """
    if not raw.strip():
        return _emit_parse_error(state, "Model returned no tool_calls.", config)

    try:
        decoded = json.loads(raw)
    except Exception as exc:
        return _emit_parse_error(state, f"Parse failed: {exc}", config)

    if isinstance(decoded, dict):
        # 某些 provider 把单个 tool call 序列化为 dict；当作一元列表处理。
        decoded = [decoded]
    if not isinstance(decoded, list) or not decoded:
        return _emit_parse_error(state, "Model returned no tool_calls.", config)
    if len(decoded) > 1:
        # C15：绝不静默取第一条；强制模型重试。
        return _emit_parse_error(
            state,
            "Multiple tool_calls rejected (only one allowed).",
            config,
        )

    call = decoded[0]
    if not isinstance(call, dict):
        return _emit_parse_error(state, f"Invalid tool_call format: {call!r}", config)
    name = call.get("name")
    args = call.get("args", {})
    if not isinstance(name, str) or not name:
        return _emit_parse_error(state, f"tool_call missing 'name': {call!r}", config)
    if not isinstance(args, dict):
        return _emit_parse_error(state, f"tool_call 'args' must be a dict: {args!r}", config)

    return {
        "thought": "",
        "action": name,
        "action_input": args,
    }


def _parse_json_action(state: RunState, raw: str, config: Any | None = None) -> dict[str, Any]:
    """解码 ```` ```json {thought, action, action_input} ``` ```` payload。"""
    try:
        ms = parse_model_step(raw)
    except Exception as exc:
        return _emit_parse_error(state, f"Parse failed: {exc}", config)
    return {
        "thought": ms.thought,
        "action": ms.action,
        "action_input": ms.action_input,
    }


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------

def _emit_parse_error(state: RunState, msg: str, config: Any | None = None) -> dict[str, Any]:
    """追加一条 parse_error StepRecord 并设置后续路由字段。

    本步骤共用 ``state["step_index"]``（D2：仅 model_node 推进它），并把
    phase / plan_* 字段透传过来，让 ``trace.json`` 保留 ReAct vs
    Plan-and-Solve 来源信息。
    """
    step_index = int(state.get("step_index", 0) or 0)
    dispatch_observability_event(
        "parse_error",
        {
            "step_index": step_index,
            "mode": str(state.get("action_mode", "json_action") or "json_action"),
            "error": msg,
        },
        config,
    )
    return {
        "steps": [
            StepRecord(
                step_index=step_index,
                thought="",
                action="__error__",
                action_input={},
                raw_response=state.get("raw_response", "") or "",
                observation={"ok": False, "error": msg},
                ok=False,
                phase=state.get("phase", "") or "",
                plan_progress=state.get("plan_progress", "") or "",
                plan_step_description=state.get("plan_step_description", "") or "",
            )
        ],
        "last_tool_ok": False,
        "last_error_kind": "parse_error",
    }


__all__ = ["parse_action_node"]