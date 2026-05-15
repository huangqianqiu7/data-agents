"""Bug 5 回归：``dispatch_observability_event`` 在 LangGraph runtime 之外
（即 ``dispatch_custom_event`` 抛 ``RuntimeError`` 时）应路由到已注册的
fallback handlers，而不是静默吞掉。

背景：
  - ``runner._build_and_set_corpus_handles`` 在 ``compiled.invoke`` 之前执行；
    此阶段没有 LangGraph callback manager 在 contextvar 里 → ``dispatch_custom_event``
    抛 ``RuntimeError`` → 原实现把所有 ``memory_rag_*`` 事件全部吞掉，
    ``MetricsCollector`` 永远收不到。
  - 修复策略：``observability.events`` 模块新增模块级 fallback handler list；
    ``runner`` 在 ``MetricsCollector`` 构造时 register、task 结束 finally 块 unregister。
"""
from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _clear_fallback_handlers():
    """每个 test 自动清理 module-level fallback list，避免互相干扰。"""
    from data_agent_langchain.observability import events as events_mod

    # 测试前先清空（防御性，确保各 test 独立）。
    events_mod._FALLBACK_HANDLERS.clear()
    yield
    events_mod._FALLBACK_HANDLERS.clear()


def test_dispatch_outside_langgraph_runtime_silently_drops_when_no_fallback() -> None:
    """无 fallback 注册时 dispatch 在 LangGraph runtime 之外不抛异常（保持原 fail-safe 行为）。"""
    from data_agent_langchain.observability.events import dispatch_observability_event

    # 不应抛异常。
    dispatch_observability_event("memory_rag_skipped", {"reason": "no_documents"})


def test_dispatch_outside_langgraph_runtime_routes_to_registered_fallback() -> None:
    """注册 fallback handler 后，LangGraph runtime 外的 dispatch 应路由给它。"""
    from data_agent_langchain.observability.events import (
        dispatch_observability_event,
        register_fallback_handler,
    )

    captured: list[tuple[str, dict[str, Any]]] = []

    def handler(name: str, data: dict[str, Any]) -> None:
        captured.append((name, data))

    register_fallback_handler(handler)

    dispatch_observability_event("memory_rag_index_built", {"task_id": "task_x", "doc_count": 2})
    dispatch_observability_event("memory_rag_skipped", {"reason": "index_timeout"})

    assert captured == [
        ("memory_rag_index_built", {"task_id": "task_x", "doc_count": 2}),
        ("memory_rag_skipped", {"reason": "index_timeout"}),
    ]


def test_register_fallback_handler_is_idempotent_for_same_handler() -> None:
    """同一个 handler 重复 register 不应重复触发。"""
    from data_agent_langchain.observability.events import (
        dispatch_observability_event,
        register_fallback_handler,
    )

    captured: list[str] = []

    def handler(name: str, data: dict[str, Any]) -> None:
        captured.append(name)

    register_fallback_handler(handler)
    register_fallback_handler(handler)  # 二次注册应被忽略
    register_fallback_handler(handler)  # 第三次也是

    dispatch_observability_event("event_a", {})

    assert captured == ["event_a"], "handler should be called exactly once even if registered multiple times"


def test_unregister_fallback_handler_stops_delivery() -> None:
    """unregister 后 dispatch 不应再调用该 handler。"""
    from data_agent_langchain.observability.events import (
        dispatch_observability_event,
        register_fallback_handler,
        unregister_fallback_handler,
    )

    captured: list[str] = []

    def handler(name: str, data: dict[str, Any]) -> None:
        captured.append(name)

    register_fallback_handler(handler)
    dispatch_observability_event("event_a", {})
    unregister_fallback_handler(handler)
    dispatch_observability_event("event_b", {})

    assert captured == ["event_a"], "handler should only see events before unregister"


def test_unregister_unknown_handler_is_noop() -> None:
    """unregister 一个未注册的 handler 不应抛异常（fail-safe）。"""
    from data_agent_langchain.observability.events import unregister_fallback_handler

    def stranger(name: str, data: dict[str, Any]) -> None:
        pass

    # 不应抛异常。
    unregister_fallback_handler(stranger)


def test_multiple_fallback_handlers_all_called_in_registration_order() -> None:
    """多个 fallback 都应被调，按注册顺序。"""
    from data_agent_langchain.observability.events import (
        dispatch_observability_event,
        register_fallback_handler,
    )

    captured: list[str] = []

    def h1(name: str, data: dict[str, Any]) -> None:
        captured.append(f"h1:{name}")

    def h2(name: str, data: dict[str, Any]) -> None:
        captured.append(f"h2:{name}")

    register_fallback_handler(h1)
    register_fallback_handler(h2)

    dispatch_observability_event("event_a", {})

    assert captured == ["h1:event_a", "h2:event_a"]


def test_fallback_handler_exception_does_not_block_other_handlers() -> None:
    """一个 fallback handler 抛异常不应阻塞后续 handler 被调（fail-safe）。"""
    from data_agent_langchain.observability.events import (
        dispatch_observability_event,
        register_fallback_handler,
    )

    captured: list[str] = []

    def h_raising(name: str, data: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    def h_good(name: str, data: dict[str, Any]) -> None:
        captured.append(name)

    register_fallback_handler(h_raising)
    register_fallback_handler(h_good)

    # 不应抛异常，且 h_good 仍被调。
    dispatch_observability_event("event_a", {})

    assert captured == ["event_a"]


def test_dispatch_inside_langgraph_runtime_does_not_call_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """在 LangGraph runtime 内 ``dispatch_custom_event`` 成功时，
    fallback 不应被调（避免重复传递）。"""
    from data_agent_langchain.observability import events as events_mod

    successful_dispatches: list[tuple[str, dict[str, Any]]] = []

    def fake_dispatch_custom_event(name, data, config=None):
        successful_dispatches.append((name, data))

    monkeypatch.setattr(events_mod, "dispatch_custom_event", fake_dispatch_custom_event)

    fallback_captured: list[tuple[str, dict[str, Any]]] = []

    def fallback(name: str, data: dict[str, Any]) -> None:
        fallback_captured.append((name, data))

    events_mod.register_fallback_handler(fallback)
    events_mod.dispatch_observability_event("event_a", {"k": "v"})

    assert successful_dispatches == [("event_a", {"k": "v"})]
    assert fallback_captured == [], "fallback should not be called when dispatch_custom_event succeeds"
