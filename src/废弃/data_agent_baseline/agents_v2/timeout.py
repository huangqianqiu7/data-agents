"""
超时调用包装器。

使用独立 daemon 线程实现超时控制，修复原版线程池 (ThreadPoolExecutor) 的死锁隐患：
  - 原版使用 max_workers=2 的固定线程池，超时后僵尸任务占满 worker → 全局死锁
  - 新版每次调用创建独立 daemon 线程，超时后主调方立即抛出 TimeoutError，
    底层线程在后台自然结束或在进程退出时被回收
"""
from __future__ import annotations

import threading


def call_with_timeout(fn, args: tuple, timeout_s: float):
    """带超时控制的同步调用包装器。

    Args:
        fn: 要调用的函数
        args: 传给 fn 的参数元组
        timeout_s: 超时秒数

    Returns:
        fn(*args) 的返回值

    Raises:
        TimeoutError: 超时
        Exception: fn 内部抛出的任何异常
    """
    result_container: list = [None]
    exception_container: list[BaseException | None] = [None]

    def _wrapper():
        try:
            result_container[0] = fn(*args)
        except Exception as e:
            exception_container[0] = e

    t = threading.Thread(target=_wrapper, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        raise TimeoutError(
            f"{getattr(fn, '__qualname__', str(fn))} timed out after {timeout_s}s"
        )
    if exception_container[0] is not None:
        raise exception_container[0]
    return result_container[0]
