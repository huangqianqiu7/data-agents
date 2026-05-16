"""``ReadJsonTool`` —— 预览任务 ``context/`` 目录下 JSON 文件（按字符长度截断）。"""
from __future__ import annotations

from typing import Any, ClassVar, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from data_agent_langchain.benchmark.schema import PublicTask
from data_agent_langchain.exceptions import ContextAssetNotFoundError
from data_agent_langchain.tools.filesystem import read_json_preview
from data_agent_langchain.tools.descriptions import render_legacy_description
from data_agent_langchain.tools.tool_runtime import ToolRuntime, ToolRuntimeResult


class ReadJsonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Relative path to a JSON file inside `context/`.")
    max_chars: int = Field(
        default=4000, ge=1,
        description="Maximum characters to include in the rendered preview.",
    )


class ReadJsonTool(BaseTool):
    name: str = "read_json"
    description: str = render_legacy_description("read_json")
    args_schema: Type[BaseModel] = ReadJsonInput
    return_direct: ClassVar[bool] = False

    _task: PublicTask = PrivateAttr()
    _runtime: ToolRuntime = PrivateAttr()

    def __init__(self, *, task: PublicTask, runtime: ToolRuntime, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._task = task
        self._runtime = runtime

    def _run(self, path: str, max_chars: int = 4000, **_kwargs: Any) -> ToolRuntimeResult:
        try:
            preview = read_json_preview(self._task, path, max_chars=max_chars)
        except ContextAssetNotFoundError as exc:
            return ToolRuntimeResult(
                ok=False,
                content={"tool": self.name, "error": str(exc)},
                error_kind="validation",
            )
        except Exception as exc:
            return ToolRuntimeResult(
                ok=False,
                content={"tool": self.name, "error": str(exc)},
                error_kind="runtime",
            )
        return ToolRuntimeResult(ok=True, content=preview)


__all__ = ["ReadJsonInput", "ReadJsonTool"]