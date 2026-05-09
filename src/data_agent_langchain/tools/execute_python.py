"""``ExecutePythonTool`` —— 在任务 context 沙箱里执行 Python 代码。

实际子进程执行逻辑在 ``tools/python_exec.py`` 的 ``execute_python_code``；
本 ``BaseTool`` 子类只是把异常翻译为 ``ToolRuntimeResult``。超时来自
``ToolRuntime.python_timeout_s``，与 description 常量
（``EXECUTE_PYTHON_TIMEOUT_SECONDS``）分开存放，方便未来把超时
提升到配置项。

Phase 2 / parity 约束：``ToolRuntime.python_timeout_s`` 必须等于
``EXECUTE_PYTHON_TIMEOUT_SECONDS``（30s），factory 默认值已经保证这点，
所以 description 里写的数和实际沙箱设置永远一致。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from data_agent_langchain.benchmark.schema import PublicTask
from data_agent_langchain.tools.python_exec import execute_python_code
from data_agent_langchain.tools.descriptions import render_legacy_description
from data_agent_langchain.tools.tool_runtime import ToolRuntime, ToolRuntimeResult


class ExecutePythonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(
        description=(
            "Python source code to execute. The working directory is the task "
            "context root; ``context_root`` and ``Path`` are pre-injected into "
            "globals."
        ),
    )


class ExecutePythonTool(BaseTool):
    name: str = "execute_python"
    description: str = render_legacy_description("execute_python")
    args_schema: Type[BaseModel] = ExecutePythonInput
    return_direct: ClassVar[bool] = False

    _task: PublicTask = PrivateAttr()
    _runtime: ToolRuntime = PrivateAttr()

    def __init__(self, *, task: PublicTask, runtime: ToolRuntime, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._task = task
        self._runtime = runtime

    def _run(self, code: str, **_kwargs: Any) -> ToolRuntimeResult:
        try:
            result = execute_python_code(
                Path(self._runtime.context_dir),
                code,
                timeout_seconds=int(self._runtime.python_timeout_s),
            )
        except Exception as exc:
            return ToolRuntimeResult(
                ok=False,
                content={"tool": self.name, "error": str(exc)},
                error_kind="runtime",
            )
        # 沙箱已经用 ``success`` 区分成功 / 失败；直接映射成 ``ok``，
        # 不让 ``tool_node`` 二次解读。
        return ToolRuntimeResult(
            ok=bool(result.get("success")),
            content=result,
            # 用 ``runtime``（沙箱崩溃 / 子进程报告的超时），而不是
            # ``timeout`` —— ``timeout`` 只允许由外层 ``call_tool_with_timeout``
            # 设置（§6.4）。
            error_kind=None if result.get("success") else "runtime",
        )


__all__ = ["ExecutePythonInput", "ExecutePythonTool"]