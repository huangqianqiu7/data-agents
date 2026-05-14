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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — 避免 import-time 强依赖 config 模块
    from data_agent_langchain.config import AppConfig


_APP_CONFIG: ContextVar["AppConfig | None"] = ContextVar(
    "data_agent_langchain_app_config", default=None
)

# v3 corpus RAG（M4.4.4）：per-task corpus handles 通过 contextvar 注入到图节点。
# 与 ``_APP_CONFIG`` 不同的语义：
#   - ``AppConfig`` 是必填，没设置 → 编程错误（``get_current_app_config`` 抛错）。
#   - corpus handles 是可选，没设置 → 视为"该 task 没启用 RAG"（fail-closed），
#     ``get_current_corpus_handles`` 返回 ``None``，调用方决定降级路径。
#
# handle 类型故意标注为 ``Any``，避免本模块顶层 import
# ``data_agent_langchain.memory.rag.factory`` —— 后者会传递 import
# 到 chromadb 等重依赖（虽然走方法级延迟 import，但减少耦合）。
_CORPUS_HANDLES: ContextVar["Any"] = ContextVar(
    "data_agent_langchain_corpus_handles", default=None
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


def set_current_corpus_handles(handles: Any) -> None:
    """v3 corpus RAG（M4.4.4）：把 ``TaskCorpusHandles`` 写入 context。

    runner 子进程入口在构建 task corpus 索引成功后调用；图节点通过
    :func:`get_current_corpus_handles` 同步读取。
    """
    _CORPUS_HANDLES.set(handles)


def get_current_corpus_handles() -> Any:
    """v3 corpus RAG（M4.4.4）：读取当前 context 上的 corpus handles。

    未设置时返回 ``None``（**不**抛错）：corpus handles 是可选项，
    ``rag.enabled=false`` 或索引构建失败的场景下都会保持 ``None``。
    调用方（``recall_corpus_snippets`` 等）应优雅降级。
    """
    return _CORPUS_HANDLES.get()


def clear_current_corpus_handles() -> None:
    """清理当前 context 上的 corpus handles。

    runner 在 ``compiled.invoke`` 之后必须调用本函数，避免污染同进程下
    后续 task 的 contextvar 状态（``ContextVar.set`` 在子线程 / 池场景中
    不会自动回退）。
    """
    _CORPUS_HANDLES.set(None)


__all__ = [
    "clear_current_corpus_handles",
    "get_current_app_config",
    "get_current_corpus_handles",
    "set_current_app_config",
    "set_current_corpus_handles",
]