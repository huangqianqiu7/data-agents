"""
LangSmith ``LangChainTracer`` 回调工厂。

§11.4.1（D1：仅在 ``compiled.invoke`` 单点注入，绝不挂在 ``ChatOpenAI``
上）：当 ``observability.langsmith_enabled`` 为真且环境里有
``LANGSMITH_API_KEY`` / ``LANGCHAIN_API_KEY`` 时，返回一个
``LangChainTracer``；否则返回 ``[]`` 让 runner 离线运行（C5 / §11.4.3）。
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from data_agent_langchain.config import AppConfig


def build_callbacks(
    config: "AppConfig", *, task_id: str, mode: str
) -> list[Any]:
    """返回 ``compiled.invoke`` 用的 LangChain callback 列表。

    满足以下条件之一即返回 ``[]``：
      - ``observability.langsmith_enabled`` 为 False。
      - 环境变量 ``LANGSMITH_API_KEY`` / ``LANGCHAIN_API_KEY`` 都没设置。
      - 实际构造 ``LangChainTracer`` 抛异常（离线 / 未安装等情况）。
    """
    if not bool(config.observability.langsmith_enabled):
        return []
    if not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")):
        return []
    try:
        from langchain_core.tracers.langchain import LangChainTracer
        return [
            LangChainTracer(
                project_name="dabench-lc",
                tags=[f"task_id:{task_id}", f"mode:{mode}"],
            )
        ]
    except Exception:
        return []


__all__ = ["build_callbacks"]