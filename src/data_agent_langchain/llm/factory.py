"""
``ChatOpenAI`` 构造与 gateway 感知的 ``bind_tools`` 包装。

实现要点：
  - :func:`build_chat_model` 把 :class:`AppConfig.agent` 上的字段映射到
    ``ChatOpenAI`` 构造参数；``max_retries=0`` 让重试逻辑完全由项目级
    ``model_retry`` 承担（§11.1.1 / E4）。
  - :func:`bind_tools_for_gateway` 根据 :class:`GatewayCaps`（来自
    Phase 0.5 smoke 产物）决定是否调用 ``bind_tools``；不支持
    tool_calling 的网关下直接返回原 LLM。
"""
from __future__ import annotations

from typing import Any, Sequence

from data_agent_langchain.config import AppConfig
from data_agent_langchain.observability.gateway_caps import GatewayCaps


def build_chat_model(config: AppConfig) -> Any:
    """根据 :class:`AppConfig` 构造一个 ``ChatOpenAI``。"""
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": config.agent.model,
        "base_url": config.agent.api_base,
        "api_key": config.agent.api_key,
        "temperature": config.agent.temperature,
        "request_timeout": float(config.agent.model_timeout_s),
        # 项目级重试由 ``model_retry`` 实现（§11.1.1），底层这里禁用。
        "max_retries": 0,
    }
    if config.agent.seed is not None:
        kwargs["seed"] = config.agent.seed
    return ChatOpenAI(**kwargs)


def bind_tools_for_gateway(llm: Any, tools: Sequence[Any], caps: GatewayCaps) -> Any:
    """按 :class:`GatewayCaps` 决定是否给 *llm* 绑定工具。"""
    if not caps.tool_calling:
        return llm
    kwargs: dict[str, Any] = {}
    if caps.parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = caps.parallel_tool_calls
    return llm.bind_tools(list(tools), **kwargs)


__all__ = ["bind_tools_for_gateway", "build_chat_model"]