"""
Timeout wrapper for synchronous function calls.

Uses a dedicated daemon thread per call.  On timeout the caller receives
:exc:`TimeoutError` immediately; the background thread is left to finish
naturally or is reaped at process exit.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def call_with_timeout(
    fn: Callable[..., T],
    args: tuple[Any, ...],
    timeout_seconds: float,
) -> T:
    """Invoke ``fn(*args)`` with a hard timeout.

    Args:
        fn: Callable to execute.
        args: Positional arguments forwarded to *fn*.
        timeout_seconds: Maximum wall-clock seconds to wait.

    Returns:
        The return value of ``fn(*args)``.

    Raises:
        TimeoutError: If *fn* does not complete within the budget.
        Exception: Any exception raised inside *fn* is re-raised.
    """
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
            f"{getattr(fn, '__qualname__', str(fn))} timed out after {timeout_seconds}s"
        )
    if exception_container[0] is not None:
        raise exception_container[0]
    return result_container[0]  # type: ignore[return-value]
