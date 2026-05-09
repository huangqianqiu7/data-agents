"""业务节点向 ``MetricsCollector`` 发送自定义观测事件的统一入口。

走 LangGraph 0.4 ``dispatch_custom_event``；当 contextvar 链路未初始化
（直接驱动节点的单元测试场景）时静默 no-op。
"""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks.manager import dispatch_custom_event


def dispatch_observability_event(name: str, data: dict[str, Any], config: Any | None = None) -> None:
    """向 callbacks 发送一个自定义事件；contextvar 缺失时直接吞掉。"""
    try:
        dispatch_custom_event(name, data, config=config)
    except RuntimeError:
        return


__all__ = ["dispatch_observability_event"]