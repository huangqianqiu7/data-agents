"""
工具集工厂。

``create_all_tools(task, runtime)`` 在每个需要调工具的节点入口都被调用一次
（§6.1.3）。刻意每次都重建一遍是为了：

  - ``RunState`` 不需要承载非 picklable 的 ``BaseTool`` 实例（C4）。
  - ``_task`` 与 ``_runtime`` PrivateAttr 永远是新鲜的，避免跨任务泄漏。
  - 子进程 worker 可以独立构造工具集，不继承父进程状态。

构造成本是微秒级（无 I/O / 无 DB 连接），与 LLM 调用延迟比可忽略。
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from data_agent_langchain.benchmark.schema import PublicTask
from data_agent_langchain.tools.answer import AnswerTool
from data_agent_langchain.tools.execute_context_sql import ExecuteContextSqlTool
from data_agent_langchain.tools.execute_python import ExecutePythonTool
from data_agent_langchain.tools.inspect_sqlite_schema import InspectSqliteSchemaTool
from data_agent_langchain.tools.list_context import ListContextTool
from data_agent_langchain.tools.read_csv import ReadCsvTool
from data_agent_langchain.tools.read_doc import ReadDocTool
from data_agent_langchain.tools.read_json import ReadJsonTool
from data_agent_langchain.tools.tool_runtime import ToolRuntime


def create_all_tools(task: PublicTask, runtime: ToolRuntime) -> list[BaseTool]:
    """构造一份任务级新工具集。

    返回顺序固定，让 ``bind_tools`` 每次都把 OpenAI ``functions`` 数组
    序列化成同样的字节流；这对网关层的 prompt 缓存命中很重要。
    """
    return [
        AnswerTool(task=task, runtime=runtime),
        ExecuteContextSqlTool(task=task, runtime=runtime),
        ExecutePythonTool(task=task, runtime=runtime),
        InspectSqliteSchemaTool(task=task, runtime=runtime),
        ListContextTool(task=task, runtime=runtime),
        ReadCsvTool(task=task, runtime=runtime),
        ReadDocTool(task=task, runtime=runtime),
        ReadJsonTool(task=task, runtime=runtime),
    ]


__all__ = ["create_all_tools"]