"""
LLM 工厂层 —— 包装 LangChain ``ChatOpenAI``，注入项目级策略
（重试 / 超时 / 由 Phase 0.5 caps 驱动的 ``bind_tools_for_gateway``）。

具体实现在 :mod:`llm.factory`：
  - :func:`build_chat_model`：从 :class:`AppConfig` 构造 ``ChatOpenAI``。
  - :func:`bind_tools_for_gateway`：按 :class:`GatewayCaps` 决定是否
    及如何 ``bind_tools``（v4 M3 caps 驱动）。
"""
__all__: list[str] = []