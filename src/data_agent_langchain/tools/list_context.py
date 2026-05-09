"""
``ListContextTool`` —— 列出任务 ``context/`` 目录下的文件与子目录。

这是规则上的 *第一动作*：所有提示词都要求智能体先调用 ``list_context``
再去猜任何路径。Discovery Gate（启用时）在调用成功后会把
``state["discovery_done"] = True``（§7.1）。
"""
from __future__ import annotations

from typing import Any, ClassVar, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from data_agent_langchain.benchmark.schema import PublicTask
from data_agent_langchain.tools.filesystem import list_context_tree
from data_agent_langchain.tools.descriptions import render_legacy_description
from data_agent_langchain.tools.tool_runtime import ToolRuntime, ToolRuntimeResult


class ListContextInput(BaseModel):
    """Pydantic v2 输入 schema；拒绝未知字段。"""
    model_config = ConfigDict(extra="forbid")
    max_depth: int = Field(
        default=4, ge=1, le=10,
        description="Maximum recursion depth when walking the context tree.",
    )


class ListContextTool(BaseTool):
    name: str = "list_context"
    description: str = render_legacy_description("list_context")
    args_schema: Type[BaseModel] = ListContextInput
    return_direct: ClassVar[bool] = False  # 显式置为 False（C2）

    # v4 E9：``PrivateAttr`` 是 pydantic v2 给 ``BaseModel`` 子类挂非
    # model 状态的官方做法；直接 ``object.__setattr__`` 会被 frozen
    # tool model 拒绝抛 ValidationError。
    _task: PublicTask = PrivateAttr()
    _runtime: ToolRuntime = PrivateAttr()

    def __init__(self, *, task: PublicTask, runtime: ToolRuntime, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._task = task
        self._runtime = runtime

    # 基类 ``_run`` 接受 ``run_manager``；声明 ``**_kwargs`` 防止未来版本
    # 无条件路由它时报 TypeError。
    def _run(self, max_depth: int = 4, **_kwargs: Any) -> ToolRuntimeResult:
        try:
            tree = list_context_tree(self._task, max_depth=max_depth)
        except Exception as exc:
            return ToolRuntimeResult(
                ok=False,
                content={"tool": self.name, "error": str(exc)},
                error_kind="runtime",
            )
        return ToolRuntimeResult(ok=True, content=tree)


__all__ = ["ListContextInput", "ListContextTool"]