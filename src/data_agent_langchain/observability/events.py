"""业务节点向 ``MetricsCollector`` 发送自定义观测事件的统一入口。

主路径：LangGraph 0.4 ``dispatch_custom_event`` —— 业务节点在 graph 运行时
内 dispatch 的事件经此路径送达 LangChain ``callbacks`` 链上的所有 handler。

Bug 5 回退路径：``dispatch_custom_event`` 需要一个 callback manager 在
contextvar 里才能工作；``runner._build_and_set_corpus_handles`` 在
``compiled.invoke`` 之前调用，此阶段没有 callback manager，``dispatch_custom_event``
会抛 ``RuntimeError``。原实现把这种情况静默吞掉，导致 ``memory_rag_*`` 系列事件
（来自 ``factory.build_task_corpus``）从未到达 ``MetricsCollector``。

修复：维护一个模块级 ``_FALLBACK_HANDLERS`` 列表；``RuntimeError`` 时遍历
fallback 调用每个 handler。``MetricsCollector`` 在构造时 ``register_fallback_handler(self.on_observability_event)``，task 结束 finally 块 ``unregister_fallback_handler(...)``。
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.callbacks.manager import dispatch_custom_event

_logger = logging.getLogger(__name__)


# 模块级 fallback handler 列表；线程模型：runner 子进程单线程，无需加锁。
_FALLBACK_HANDLERS: list[Callable[[str, dict[str, Any]], None]] = []


def dispatch_observability_event(
    name: str, data: dict[str, Any], config: Any | None = None
) -> None:
    """向 callbacks 发送一个自定义事件。

    主路径：LangGraph ``dispatch_custom_event``。
    回退：``RuntimeError`` 时依次调用 ``_FALLBACK_HANDLERS`` 中的每个 handler。
    其中任一 handler 抛异常都被吞掉且记日志，不影响后续 handler 与调用方。
    """
    try:
        dispatch_custom_event(name, data, config=config)
        return
    except RuntimeError:
        # 主路径失败：在 LangGraph runtime 之外或 callback manager 缺失。
        # 退路到 fallback handlers。
        pass

    for handler in list(_FALLBACK_HANDLERS):
        try:
            handler(name, data)
        except Exception:  # noqa: BLE001 - fail-safe：fallback 抛错不影响调用方
            _logger.exception(
                "observability fallback handler raised; event=%r data=%r",
                name,
                data,
            )


def register_fallback_handler(
    handler: Callable[[str, dict[str, Any]], None],
) -> None:
    """注册一个 fallback handler；同一个对象重复注册被忽略（幂等）。"""
    if handler in _FALLBACK_HANDLERS:
        return
    _FALLBACK_HANDLERS.append(handler)


def unregister_fallback_handler(
    handler: Callable[[str, dict[str, Any]], None],
) -> None:
    """注销 fallback handler；未注册过的 handler 静默忽略（fail-safe）。"""
    try:
        _FALLBACK_HANDLERS.remove(handler)
    except ValueError:
        return


__all__ = [
    "dispatch_observability_event",
    "register_fallback_handler",
    "unregister_fallback_handler",
]