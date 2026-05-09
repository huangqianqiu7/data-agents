"""
同步工具调用的硬超时包装器。

LANGCHAIN_MIGRATION_PLAN.md §4.4 要求三层独立超时（model 请求、单工具、
单任务）。本模块负责单工具这一层。

设计要点：

  - 用 daemon 线程 + ``thread.join(timeout)`` 实现；超时后线程仍在跑，
    但不阻塞主线程，与 legacy ``call_with_timeout`` 实现一致（Windows
    ``spawn`` 语义下行为可预测，§4.6：仅同步）。
  - 超时不会让 ``_run`` 把整张图卡死；调用方拿到的是 ``ToolRuntimeResult
    (ok=False, error_kind="timeout", ...)``，``tool_node`` 会进一步把它
    映射成 ``last_error_kind="tool_timeout"``。
"""
from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

from data_agent_langchain.tools.tool_runtime import ToolRuntimeResult


T = TypeVar("T")


def call_with_timeout(
    fn: Callable[..., T],
    args: tuple[Any, ...],
    timeout_seconds: float,
) -> T:
    """带硬超时地调用 ``fn(*args)``。与 legacy helper 同契约。"""
    result_container: list[T | None] = [None]
    exception_container: list[BaseException | None] = [None]

    def _wrapper() -> None:
        try:
            result_container[0] = fn(*args)
        except Exception as exc:
            exception_container[0] = exc

    thread = threading.Thread(target=_wrapper, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(
            f"{getattr(fn, '__qualname__', str(fn))} timed out after "
            f"{timeout_seconds}s"
        )
    if exception_container[0] is not None:
        raise exception_container[0]
    return result_container[0]  # type: ignore[return-value]


def call_tool_with_timeout(
    tool: Any,                         # langchain_core.tools.BaseTool subclass
    action_input: dict[str, Any],
    timeout_seconds: float,
) -> ToolRuntimeResult:
    """带硬超时地调用 ``tool.invoke(action_input)``。

    返回值：

      - 成功：工具自己返回的 ``ToolRuntimeResult``。
      - 超时：``ToolRuntimeResult(ok=False, error_kind="timeout", ...)``。
      - pydantic 校验失败：``error_kind="validation"``。
      - 其他异常：``error_kind="runtime"``。

    ``validation`` 与 ``runtime`` 的区分是 v4 M5 的关键：``advance_node``
    rule 5 仅对 ``tool_*`` 族 ``error_kind`` 触发 replan。
    """
    tool_name = getattr(tool, "name", str(tool))
    try:
        return call_with_timeout(tool.invoke, (action_input,), timeout_seconds)
    except TimeoutError:
        return ToolRuntimeResult(
            ok=False,
            content={
                "tool": tool_name,
                "error": (
                    f"Tool '{tool_name}' timed out after {timeout_seconds}s. "
                    "Try simplifying your code or query."
                ),
            },
            error_kind="timeout",
        )
    except Exception as exc:
        # pydantic ``ValidationError`` 视为 validation；其他全部当 runtime。
        # ``BaseTool.invoke`` 在 schema 不匹配时会抛 pydantic ValidationError。
        try:
            from pydantic import ValidationError as _PydanticValidationError
        except Exception:  # pragma: no cover — pydantic always present
            _PydanticValidationError = ValueError  # type: ignore[assignment]

        if isinstance(exc, _PydanticValidationError):
            return ToolRuntimeResult(
                ok=False,
                content={"tool": tool_name, "error": str(exc)},
                error_kind="validation",
            )
        return ToolRuntimeResult(
            ok=False,
            content={"tool": tool_name, "error": str(exc)},
            error_kind="runtime",
        )


__all__ = ["call_tool_with_timeout", "call_with_timeout"]