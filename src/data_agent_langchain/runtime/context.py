"""
通过 ``contextvars`` 注入 ``AppConfig``。

LANGCHAIN_MIGRATION_PLAN.md v3 D4：``AppConfig`` **不** 通过
``RunnableConfig.configurable`` 传递——那条路径会 pickle 配置对象，
而 ``AppConfig`` 可能含 ``Path`` / 嵌套 dataclass / lambda 等
``multiprocessing.Process`` 难以正确 round-trip 的字段。

替代方案：runner 在子进程入口用 ``AppConfig.from_dict`` 重建配置，
然后塞进模块级 ``contextvars.ContextVar``。每个需要配置的节点都通过
:func:`get_current_app_config` 同步取用。

这样 ``RunState`` 维持最小且可 pickle，节点又能在不重新解析 YAML 的
情况下随手访问配置。
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — 避免 import-time 强依赖 config 模块
    from data_agent_langchain.config import AppConfig


_APP_CONFIG: ContextVar["AppConfig | None"] = ContextVar(
    "data_agent_langchain_app_config", default=None
)


def set_current_app_config(cfg: "AppConfig") -> None:
    """把 *cfg* 写入当前 context，供后续节点取用。

    runner 子进程入口在 ``compiled.invoke`` 之前必须先调用本函数
    （LANGCHAIN_MIGRATION_PLAN.md §5.3 / §13.2）。
    """
    _APP_CONFIG.set(cfg)


def get_current_app_config() -> "AppConfig":
    """读取当前 context 上的 ``AppConfig``。

    contextvar 未设置时抛 ``RuntimeError`` —— 这属于编程错误：任何触发
    ``compiled.invoke`` 的入口都必须先初始化 context。
    """
    cfg = _APP_CONFIG.get()
    if cfg is None:
        raise RuntimeError(
            "AppConfig not initialised; subprocess entry must call "
            "set_current_app_config() before invoking the compiled graph."
        )
    return cfg


__all__ = ["get_current_app_config", "set_current_app_config"]