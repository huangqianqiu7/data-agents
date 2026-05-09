"""
模型适配器层：定义大模型对话的标准数据结构和接口。

包含：
  - ModelMessage: 对话消息结构体
  - ModelStep:    Agent 单步执行记录
  - ModelAdapter: 模型接口协议
  - OpenAIModelAdapter: 基于 OpenAI SDK 的通用适配器（兼容阿里云、DeepSeek 等）
  - ScriptedModelAdapter: 测试专用的脚本化适配器
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI


# =====================================================================
# 数据结构
# =====================================================================
@dataclass(frozen=True, slots=True)
class ModelMessage:
    """大模型对话消息的标准结构体。"""
    role: str       # "system" / "user" / "assistant"
    content: str    # 文本内容


@dataclass(frozen=True, slots=True)
class ModelStep:
    """Agent 单步执行记录（大模型输出解析后的结构化表示）。"""
    thought: str                    # 大模型的思考过程
    action: str                     # 决定调用的工具名称
    action_input: dict[str, Any]    # 工具参数
    raw_response: str               # 大模型原始输出文本


# =====================================================================
# 接口协议
# =====================================================================
class ModelAdapter(Protocol):
    """模型适配器协议，所有适配器必须实现 complete 方法。"""
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


# =====================================================================
# OpenAI 兼容适配器
# =====================================================================
class OpenAIModelAdapter:
    """
    基于 OpenAI SDK 的模型适配器。
    兼容所有实现 OpenAI Chat Completions 接口的服务商。
    """
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        if not api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def complete(self, messages: list[ModelMessage]) -> str:
        """向大模型发送对话请求，返回文本回答。"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=self.temperature,
            )
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

        # 防御性检查：防止网关返回 HTML 页面
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


# =====================================================================
# 测试专用适配器
# =====================================================================
class ScriptedModelAdapter:
    """脚本化适配器，按顺序返回预设回答，用于单元测试。"""
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
