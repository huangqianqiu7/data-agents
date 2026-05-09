from __future__ import annotations

from data_agent_langchain.config import AgentConfig, AppConfig
from data_agent_langchain.llm.factory import bind_tools_for_gateway, build_chat_model
from data_agent_langchain.observability.gateway_caps import GatewayCaps


class DummyLLM:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, **kwargs):
        self.calls.append((tools, kwargs))
        return {"tools": tools, "kwargs": kwargs}


def test_build_chat_model_uses_config_fields():
    cfg = AppConfig(agent=AgentConfig(model="m", api_base="https://example.test/v1", api_key="k", temperature=0.25, seed=7))
    llm = build_chat_model(cfg)
    assert llm.model_name == "m"
    assert str(llm.openai_api_base).rstrip("/") == "https://example.test/v1"
    assert llm.openai_api_key.get_secret_value() == "k"
    assert llm.temperature == 0.25


def test_bind_tools_for_gateway_skips_when_tool_calling_disabled():
    llm = DummyLLM()
    caps = GatewayCaps(tool_calling=False, parallel_tool_calls=False, seed_param=False, strict_mode=False)
    assert bind_tools_for_gateway(llm, ["tool"], caps) is llm
    assert llm.calls == []


def test_bind_tools_for_gateway_passes_parallel_flag_only_when_known():
    llm = DummyLLM()
    caps = GatewayCaps(tool_calling=True, parallel_tool_calls=False, seed_param=False, strict_mode=False)
    result = bind_tools_for_gateway(llm, ["tool"], caps)
    assert result["tools"] == ["tool"]
    assert result["kwargs"] == {"parallel_tool_calls": False}


def test_bind_tools_for_gateway_omits_parallel_flag_when_unknown():
    llm = DummyLLM()
    caps = GatewayCaps(tool_calling=True, parallel_tool_calls=None, seed_param=False, strict_mode=False)
    result = bind_tools_for_gateway(llm, ["tool"], caps)
    assert result["kwargs"] == {}
