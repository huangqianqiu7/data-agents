"""
``run_gateway_smoke`` —— Phase 0.5 网关能力探针。

按 4 个维度逐个 ``bind_tools`` / ``invoke`` 探测真实网关响应，把结果
写到 yaml 供 runner 启动校验使用。每条探针都用 ``try/except`` 兜底，
不让单一探针失败影响其他维度。

  - ``tool_calling``：``llm.bind_tools([probe]).invoke(...)`` 是否真有
    ``tool_calls``。
  - ``parallel_tool_calls``：是否接受 ``parallel_tool_calls=False`` 参数
    且仍能调出 tool。
  - ``seed_param``：构造时传 ``seed=42`` 是否正常。
  - ``strict_mode``：``strict=True`` 绑定后是否仍能调出 tool。
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import yaml
from langchain_core.tools import tool

from data_agent_langchain.config import AppConfig
from data_agent_langchain.observability.gateway_caps import GatewayCaps

ChatModelFactory = Callable[..., Any]


@tool
def probe_tool() -> str:
    """gateway tool-calling 探针；模型应该调它并返回 ``"ok"``。"""
    return "ok"


def run_gateway_smoke(
    config: AppConfig,
    *,
    output_path: Path | None = None,
    chat_model_factory: ChatModelFactory | None = None,
) -> GatewayCaps:
    """跑全部 4 项 smoke 探针并把结果写到 yaml。"""
    factory = chat_model_factory or _default_chat_model_factory
    resolved_config = _config_with_resolved_api_key(config)
    target_path = output_path or resolved_config.observability.gateway_caps_path

    tool_calling = _probe_tool_calling(factory, resolved_config)
    parallel_tool_calls = (
        _probe_parallel_tool_calls(factory, resolved_config) if tool_calling else None
    )
    seed_param = _probe_seed_param(factory, resolved_config)
    strict_mode = _probe_strict_mode(factory, resolved_config) if tool_calling else False

    caps = GatewayCaps(
        tool_calling=tool_calling,
        parallel_tool_calls=parallel_tool_calls,
        seed_param=seed_param,
        strict_mode=strict_mode,
    )
    write_gateway_caps(caps, target_path)
    return caps


def write_gateway_caps(caps: GatewayCaps, path: Path) -> None:
    """把 :class:`GatewayCaps` dump 到 yaml；目录不存在时自动创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gateway_caps": {
            "tool_calling": caps.tool_calling,
            "parallel_tool_calls": caps.parallel_tool_calls,
            "seed_param": caps.seed_param,
            "strict_mode": caps.strict_mode,
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _config_with_resolved_api_key(config: AppConfig) -> AppConfig:
    """``agent.api_key`` 为空时回退到 ``OPENAI_API_KEY`` 环境变量。"""
    if config.agent.api_key:
        return config
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_key:
        return config
    return replace(config, agent=replace(config.agent, api_key=env_key))


def _default_chat_model_factory(**kwargs: Any) -> Any:
    """默认工厂：直接构造 ``ChatOpenAI``。测试可注入 fake factory。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**kwargs)


def _model_kwargs(config: AppConfig, *, seed: int | None = None) -> dict[str, Any]:
    """组装 ``ChatOpenAI`` 构造参数。"""
    kwargs: dict[str, Any] = {
        "model": config.agent.model,
        "base_url": config.agent.api_base,
        "api_key": config.agent.api_key,
        "temperature": config.agent.temperature,
        "request_timeout": float(config.agent.model_timeout_s),
        "max_retries": 0,
    }
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs


def _probe_tool_calling(factory: ChatModelFactory, config: AppConfig) -> bool:
    try:
        llm = factory(**_model_kwargs(config))
        response = llm.bind_tools([probe_tool]).invoke("Call the probe tool.")
        return bool(getattr(response, "tool_calls", None))
    except Exception:
        return False


def _probe_parallel_tool_calls(factory: ChatModelFactory, config: AppConfig) -> bool:
    try:
        llm = factory(**_model_kwargs(config))
        response = llm.bind_tools([probe_tool], parallel_tool_calls=False).invoke(
            "Call the probe tool."
        )
        return bool(getattr(response, "tool_calls", None))
    except Exception:
        return False


def _probe_seed_param(factory: ChatModelFactory, config: AppConfig) -> bool:
    try:
        factory(**_model_kwargs(config, seed=42)).invoke("Reply ok.")
        return True
    except Exception:
        return False


def _probe_strict_mode(factory: ChatModelFactory, config: AppConfig) -> bool:
    try:
        llm = factory(**_model_kwargs(config))
        response = llm.bind_tools([probe_tool], strict=True).invoke("Call the probe tool.")
        return bool(getattr(response, "tool_calls", None))
    except Exception:
        return False


__all__ = ["run_gateway_smoke", "write_gateway_caps"]