"""``ExecuteContextSqlTool`` —— 在任务 context 中的 sqlite/db 上跑只读 SQL。"""
from __future__ import annotations

from typing import Any, ClassVar, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from data_agent_langchain.benchmark.schema import PublicTask
from data_agent_langchain.exceptions import ContextAssetNotFoundError
from data_agent_langchain.tools.filesystem import resolve_context_path
from data_agent_langchain.tools.sqlite import execute_read_only_sql
from data_agent_langchain.tools.descriptions import render_legacy_description
from data_agent_langchain.tools.tool_runtime import ToolRuntime, ToolRuntimeResult


class ExecuteContextSqlInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Relative path to a sqlite/db file inside `context/`.")
    sql: str = Field(description="Read-only SQL (SELECT / WITH / PRAGMA only).")
    limit: int | None = Field(
        default=None, ge=1,
        description=(
            "Optional row cap; defaults to the runtime ``sql_row_limit`` "
            "configured by ``AppConfig``."
        ),
    )


class ExecuteContextSqlTool(BaseTool):
    name: str = "execute_context_sql"
    description: str = render_legacy_description("execute_context_sql")
    args_schema: Type[BaseModel] = ExecuteContextSqlInput
    return_direct: ClassVar[bool] = False

    _task: PublicTask = PrivateAttr()
    _runtime: ToolRuntime = PrivateAttr()

    def __init__(self, *, task: PublicTask, runtime: ToolRuntime, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._task = task
        self._runtime = runtime

    def _run(
        self,
        path: str,
        sql: str,
        limit: int | None = None,
        **_kwargs: Any,
    ) -> ToolRuntimeResult:
        effective_limit = limit if limit is not None else self._runtime.sql_row_limit
        try:
            resolved = resolve_context_path(self._task, path)
            result = execute_read_only_sql(resolved, sql, limit=effective_limit)
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
        return ToolRuntimeResult(ok=True, content=result)


__all__ = ["ExecuteContextSqlInput", "ExecuteContextSqlTool"]