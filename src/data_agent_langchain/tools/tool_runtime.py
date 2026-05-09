"""
``ToolRuntime`` 与 ``ToolRuntimeResult`` 值对象定义。

LANGCHAIN_MIGRATION_PLAN.md v3 D3 / D12 / v4 E10 / E15：

  - ``ToolRuntime`` 是任务级的 **可 pickle** 运行时上下文（不含闭包、
    ``Path`` 或 callable 字段）。``tool_node`` 每次进入时从 ``RunState``
    + ``AppConfig`` 重建。``allow_path_traversal`` 字段已在 v4 E10 删除：
    路径安全完全交给 ``tools/filesystem.py`` 处理。

  - ``ToolRuntimeResult.error_kind`` 由工具 / wrapper 在错误源头设置，
    ``tool_node`` 不需要二次猜测（v3 D12）。三态：
      ``"validation"`` —— pydantic schema / 输入 shape 拒绝。
      ``"runtime"``    —— 工具体内运行时异常。
      ``"timeout"``    —— ``call_tool_with_timeout`` 触发超时。
    区分意义：``advance_node`` rule 5（v4 M5）仅对 ``tool_*`` 族错误
    触发 replan，parse / model 错误不会。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from data_agent_langchain.benchmark.schema import AnswerTable


# ---------------------------------------------------------------------------
# 运行时上下文
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToolRuntime:
    """任务内所有工具共享的可 pickle 运行时上下文。

    所有字段都是基本类型；不存 ``Path`` / 不存 callable / 不存闭包。
    构造成本接近 0；图节点每轮入口都重建一次，避免在 ``RunState`` 中
    搬运它（LANGCHAIN_MIGRATION_PLAN.md §5.3 / §6.1.4）。
    """

    task_dir: str                  # 普通字符串；节点用时再 ``Path()`` 包装
    context_dir: str               # 普通字符串；节点用时再 ``Path()`` 包装
    python_timeout_s: float        # execute_python 超时上界
    sql_row_limit: int             # execute_context_sql 默认行数上限
    max_obs_chars: int             # observation 字符截断阈值


# ---------------------------------------------------------------------------
# 工具统一返回值
# ---------------------------------------------------------------------------

# v4 E15：``validation`` 与 ``runtime`` 在此处保持独立，``tool_node``
# 才能无歧义地映射到 ``last_error_kind={tool_validation, tool_error}``。
ToolErrorKind = Literal["timeout", "validation", "runtime"]


@dataclass(frozen=True, slots=True)
class ToolRuntimeResult:
    """新的 ``BaseTool`` 子类的统一返回值。

    工具 **绝不能** 抛异常；失败时返回 ``ok=False`` + ``error_kind`` +
    ``content`` 字典（必带 ``error`` 字段与 tool 名），保证 LLM 可见的
    observation 信息完整（LANGCHAIN_MIGRATION_PLAN.md §6.4 / C2）。
    """

    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None
    # ``ok=True`` 时为 ``None``；失败时为以上三态之一。
    error_kind: ToolErrorKind | None = None


__all__ = [
    "ToolErrorKind",
    "ToolRuntime",
    "ToolRuntimeResult",
]