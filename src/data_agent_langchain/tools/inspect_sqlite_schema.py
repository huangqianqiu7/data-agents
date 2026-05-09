"""``InspectSqliteSchemaTool`` —— 查看 sqlite/db 文件的表与 CREATE SQL。"""
from __future__ import annotations

from typing import Any, ClassVar, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from data_agent_langchain.benchmark.schema import PublicTask
from data_agent_langchain.tools.filesystem import resolve_context_path
from data_agent_langchain.tools.sqlite import inspect_sqlite_schema
from data_agent_langchain.tools.descriptions import render_legacy_description
from data_agent_langchain.tools.tool_runtime import ToolRuntime, ToolRuntimeResult


class InspectSqliteSchemaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Relative path to a sqlite/db file inside `context/`.")


class InspectSqliteSchemaTool(BaseTool):
    name: str = "inspect_sqlite_schema"
    description: str = render_legacy_description("inspect_sqlite_schema")
    args_schema: Type[BaseModel] = InspectSqliteSchemaInput
    return_direct: ClassVar[bool] = False

    _task: PublicTask = PrivateAttr()
    _runtime: ToolRuntime = PrivateAttr()

    def __init__(self, *, task: PublicTask, runtime: ToolRuntime, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._task = task
        self._runtime = runtime

    def _run(self, path: str, **_kwargs: Any) -> ToolRuntimeResult:
        try:
            resolved = resolve_context_path(self._task, path)
            schema = inspect_sqlite_schema(resolved)
        except Exception as exc:
            return ToolRuntimeResult(
                ok=False,
                content={"tool": self.name, "error": str(exc)},
                error_kind="runtime",
            )
        return ToolRuntimeResult(ok=True, content=schema)


__all__ = ["InspectSqliteSchemaInput", "InspectSqliteSchemaTool"]