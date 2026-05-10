"""
模型重试与响应抽取的纯逻辑工具集。

从 ``agents/model_node.py`` 中拆分出来，让 ``model_node`` 只关心节点编排
（state 读写、step_index 推进、调用回调），重试逻辑与 LangChain
``AIMessage`` 解析逻辑独立成本模块。

LANGCHAIN_MIGRATION_PLAN.md §11.1.1 / v4 E4 约束：
  - 项目级 retry-and-record（``max_retries=3`` / ``backoff=(2, 5, 10)`` /
    单次 ``timeout=120s``），与 legacy ``BaseAgent._call_model_with_retry``
    行为对齐。
  - 重试耗尽抛 :class:`ModelExhaustedError`，由调用方写入 ``__error__``
    StepRecord。

公共 API：
  - :class:`ModelExhaustedError` —— 重试耗尽信号。
  - :func:`call_model_with_retry` —— 带 timeout / backoff 的重试循环。
  - :func:`extract_raw_response` —— 把 LangChain ``AIMessage`` 转换为
    ``raw_response`` 字符串（兼容 ``tool_calling`` / ``json_action``
    两种动作模式，§11.3 / C9）。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from langchain_core.messages import BaseMessage

from data_agent_langchain.tools.timeout import call_with_timeout

logger = logging.getLogger(__name__)


class ModelExhaustedError(RuntimeError):
    """重试全部耗尽 —— 调用方应记录 ``__error__`` 步骤。"""


# ---------------------------------------------------------------------------
# 重试循环
# ---------------------------------------------------------------------------

def call_model_with_retry(
    invoke: Callable[..., Any],
    messages: list[BaseMessage],
    *,
    step_index: int,
    max_retries: int,
    retry_backoff: tuple[float, ...],
    timeout_seconds: float,
    action_mode: str,
    config: Any | None = None,
) -> str:
    """带边界重试地执行 ``invoke(messages, config=config)``，与 legacy 行为对齐。

    ``config`` 是 LangGraph 透传给本节点的 ``RunnableConfig``；本函数把它直接
    转发给 ``invoke``，让 ``MetricsCollector.on_llm_end`` 等 callback 能正常
    触发（详见 ``test_phase3_model.py::test_model_node_propagates_runnable_
    config_to_llm_invoke``）。``config=None`` 时退化为 legacy ``invoke(messages)``。

    返回 ``raw_response`` 字符串，可直接交给 ``parse_action_node`` 消费。
    重试全部失败时抛 :class:`ModelExhaustedError`，调用方负责将其转为
    ``__error__`` StepRecord。
    """
    invoke_kwargs: dict[str, Any] | None = (
        {"config": config} if config is not None else None
    )
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            response = call_with_timeout(
                invoke,
                (messages,),
                timeout_seconds,
                kwargs=invoke_kwargs,
            )
            return extract_raw_response(response, action_mode=action_mode)
        except (TimeoutError, RuntimeError, Exception) as exc:  # noqa: BLE001
            # 把 ``Exception`` 放在最后捕获，让 OpenAI / 网络错误也算可重试；
            # ``KeyboardInterrupt`` / ``SystemExit`` 仍会向上抛。
            last_exc = exc
            if attempt < max_retries - 1:
                backoff = retry_backoff[min(attempt, len(retry_backoff) - 1)]
                logger.warning(
                    "[model_node] step %d attempt %d/%d failed: %s — retrying in %.1fs",
                    step_index, attempt + 1, max_retries, exc, backoff,
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "[model_node] step %d failed after all %d retries",
                    step_index, max_retries,
                )
    raise ModelExhaustedError(
        f"Model call failed after {max_retries} attempts: {last_exc}"
    )


# ---------------------------------------------------------------------------
# AIMessage → raw_response 抽取
# ---------------------------------------------------------------------------

def extract_raw_response(response: Any, *, action_mode: str) -> str:
    """从一次 LLM 调用的返回值里抽出 ``raw_response`` 字符串。

    LangChain ``BaseChatModel.invoke`` 通常返回 ``AIMessage``；某些 fake /
    legacy adapter 直接返回字符串。本函数兼容三种情况：

      - ``AIMessage`` 带 ``tool_calls`` 且 ``action_mode == "tool_calling"``：
        把 tool_calls 列表序列化为 JSON（§11.3 / C9）。
      - ``AIMessage`` 不带 tool_calls：直接返回 ``message.content``。
      - 字符串：原样返回。
    """
    if isinstance(response, str):
        return response
    tool_calls = getattr(response, "tool_calls", None) or []
    if action_mode == "tool_calling" and tool_calls:
        # 过滤掉非 dict 形态的条目，避免 ``json.dumps`` 在 provider-specific
        # 对象上抛异常；``parse_action_node`` 已能容错残缺 payload。
        cleaned = [tc for tc in tool_calls if isinstance(tc, dict)]
        return json.dumps(cleaned, ensure_ascii=False)
    content = getattr(response, "content", "")
    if isinstance(content, list):
        # multi-part 消息：拼接所有文本段。
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                parts.append(str(text))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content or "")


__all__ = [
    "ModelExhaustedError",
    "call_model_with_retry",
    "extract_raw_response",
]
