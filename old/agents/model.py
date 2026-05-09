from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """
    大模型对话消息的标准结构体。
    大模型的接口通常需要一个列表，里面装着一条条这样的消息。
    - role: 角色，通常是 "system" (系统设定), "user" (用户输入), "assistant" (模型回答)
    - content: 具体的文本内容
    """
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    """
    Agent 运行过程中的“一步”的完整记录。
    在 ReAct 循环中，大模型的每一次返回都会被解析成这个结构。
    - thought: 大模型的思考过程
    - action: 大模型决定调用的工具名称
    - action_input: 传给该工具的具体参数 (字典格式)
    - raw_response: 大模型最初返回的原始文本（也就是未经解析的 JSON 字符串）
    """
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str

# =====================================================================
# 接口 (Protocol) 定义区
# =====================================================================
class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError

# =====================================================================
# 核心实现：基于 OpenAI SDK 的模型适配器
# =====================================================================
class OpenAIModelAdapter:
    """
    真正干活的类。
    因为阿里云、DeepSeek 等绝大多数主流厂商都兼容了 OpenAI 的接口格式，
    所以只需要这一个类，配合修改 `api_base`，就能打通市面上 90% 的大模型。
    """
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        # 基础校验：没有 API Key 绝对跑不起来，提前拦截避免运行时才报错
        if not api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")
        self.model = model
        # .rstrip("/") 的作用是防止你在配置文件里多写了一个斜杠，导致后续拼接 URL 报错
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        # 一次性创建 OpenAI 客户端并复用，避免每次 complete() 都重建连接池
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    def complete(self, messages: list[ModelMessage]) -> str:
        """
        向大模型发送对话请求，并获取文本回答。
        """
        # 发送网络请求
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                # 用列表推导式，把系统内部的 ModelMessage 对象转换成 OpenAI 要求的字典格式
                messages=[{"role": message.role, "content": message.content} for message in messages],
                temperature=self.temperature,
            )
        except APIError as exc:
            # 用列表推导式，把系统内部的 ModelMessage 对象转换成 OpenAI 要求的字典格式
            raise RuntimeError(f"Model request failed: {exc}") from exc

        # 4. 极致的防御性编程：防止网关返回 HTML 报错页面
        # 有时候你填错了 api_base，请求发到了一个普通的网页服务器上，
        # 它可能会给你返回一段 "<html>404 Not Found</html>" 的纯字符串，而不是 JSON 对象。
        # 这里专门做拦截，给出清晰的排错提示。
        if isinstance(response, str):
            snippet = response.strip().replace("\n", " ")[:400]
            raise RuntimeError(
                "The HTTP API returned a plain string instead of an OpenAI-style chat completion JSON. "
                "Usually this means api_base is wrong or the gateway is not OpenAI-compatible on this URL. "
                "Try setting api_base to the provider's documented root, often ending with `/v1` "
                "(e.g. `https://your-host/v1`). "
                f"Body begins with: {snippet!r}"
            )
        # 5. 校验返回对象的格式：必须包含 choices 字段
        if not hasattr(response, "choices"):
            raise RuntimeError(
                f"Unexpected chat completion type: {type(response).__name__!r} (expected an object with "
                "`.choices`). Check that the provider implements OpenAI-compatible POST .../chat/completions "
                "and that agent.api_base matches their docs."
            )

        # 6. 安全地剥离出最终的文本结果
        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content

# =====================================================================
# 测试专用：脚本化模型适配器
# =====================================================================
class ScriptedModelAdapter:
    """
    这是一个“假”的模型适配器，专门用来做本地单元测试 (Unit Testing) 的。
    在测试框架逻辑时，我们不想每次都花钱去调用真大模型，还要等它生成。
    所以我们提前写好几段假回答 (responses)，每次调 complete，它就按顺序吐出下一段剧本。
    """
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        # del 明确表示我们根本不关心传进来的 messages 是什么
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        
        # pop(0) 每次弹出列表的第一个回答，模拟真实的连续对话
        return self._responses.pop(0)
