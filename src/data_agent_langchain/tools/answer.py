"""
``AnswerTool`` —— 提交最终答案表，是唯一合法的终止动作。

§6.3 / C2：本工具 **绝不能** 用 ``return_direct=True``。
``AgentExecutor`` 的短路语义不会被自定义 ``tool_node`` 尊重；改为：
返回 ``ToolRuntimeResult(is_terminal=True, answer=AnswerTable(...))``，
``tool_node`` 显式把 ``state["answer"]`` 写好，条件边据此路由到
``finalize_node`` 而不丢失结构化结果。

Validation 与 legacy ``handle_answer`` 字符级一致：
  - ``columns`` 必须是非空字符串列表。
  - ``rows`` 必须是列表。
  - 每行必须是列表，长度等于 ``len(columns)``。

任何校验失败都返回 ``ok=False`` + ``error_kind="validation"``，
不抛异常，让 ``tool_node`` 写一条干净的 ``StepRecord`` 让 LLM 下一轮
自行纠错（§6.4）。
"""
from __future__ import annotations

from typing import Any, ClassVar, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from data_agent_langchain.benchmark.schema import AnswerTable, PublicTask
from data_agent_langchain.tools.descriptions import render_legacy_description
from data_agent_langchain.tools.tool_runtime import ToolRuntime, ToolRuntimeResult


class AnswerInput(BaseModel):
    """Pydantic 输入 schema。

    ``rows`` 类型放宽到 ``list[list[Any]]``，因为答案 payload 可能含
    数字 / 字符串 / ``None`` 等；行长度一致校验放在 ``_run`` 里做，
    错误信息可以指明具体哪一行出错。
    """

    model_config = ConfigDict(extra="forbid")
    columns: list[str] = Field(
        description="Column names of the final answer table; must be non-empty.",
    )
    rows: list[list[Any]] = Field(
        description="Row values; each row must have the same length as ``columns``.",
    )


class AnswerTool(BaseTool):
    name: str = "answer"
    description: str = render_legacy_description("answer")
    args_schema: Type[BaseModel] = AnswerInput
    return_direct: ClassVar[bool] = False  # 必须是 False（C2）

    _task: PublicTask = PrivateAttr()
    _runtime: ToolRuntime = PrivateAttr()

    def __init__(self, *, task: PublicTask, runtime: ToolRuntime, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._task = task
        self._runtime = runtime

    def _run(
        self,
        columns: list[str],
        rows: list[list[Any]],
        **_kwargs: Any,
    ) -> ToolRuntimeResult:
        # ----- columns 校验（与 legacy handle_answer 字符级一致） -----
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(item, str) for item in columns)
        ):
            return ToolRuntimeResult(
                ok=False,
                content={
                    "tool": self.name,
                    "error": "answer.columns must be a non-empty list of strings.",
                },
                error_kind="validation",
            )

        # ----- rows 校验 -----
        if not isinstance(rows, list):
            return ToolRuntimeResult(
                ok=False,
                content={"tool": self.name, "error": "answer.rows must be a list."},
                error_kind="validation",
            )

        normalized_rows: list[list[Any]] = []
        for row in rows:
            if not isinstance(row, list):
                return ToolRuntimeResult(
                    ok=False,
                    content={
                        "tool": self.name,
                        "error": "Each answer row must be a list.",
                    },
                    error_kind="validation",
                )
            if len(row) != len(columns):
                return ToolRuntimeResult(
                    ok=False,
                    content={
                        "tool": self.name,
                        "error": "Each answer row must match the number of columns.",
                    },
                    error_kind="validation",
                )
            normalized_rows.append(list(row))

        # ----- 成功路径：产出 AnswerTable + 终止标志 -----
        answer = AnswerTable(columns=list(columns), rows=normalized_rows)
        return ToolRuntimeResult(
            ok=True,
            content={
                "status": "submitted",
                "column_count": len(columns),
                "row_count": len(normalized_rows),
            },
            is_terminal=True,
            answer=answer,
        )


__all__ = ["AnswerInput", "AnswerTool"]