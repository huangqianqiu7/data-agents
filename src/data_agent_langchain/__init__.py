"""
data_agent_langchain —— DABench 数据智能体的 LangGraph 后端实现。

架构对应 ``src/LangChain/LANGCHAIN_MIGRATION_PLAN.md`` v4：

  - LangChain 提供 ``BaseChatModel`` / ``BaseTool`` / ``ChatPromptTemplate``。
  - LangGraph 承载 ReAct 与 Plan-and-Solve 两套状态机。
  - 与 DABench 兼容的运行时对象（``StepRecord`` / ``AgentRunResult`` /
    ``AnswerTable`` 等）现已在本包内自包含定义（Phase 7 完成），不再
    依赖 ``data_agent_common``。
  - 子进程隔离、批量并发与 trace 落盘逻辑集中在 ``run/runner.py``（Phase 5）。

本包是当前的默认 backend；老的 hand-written ``data_agent_refactored`` 与
``data_agent_baseline`` 仅作为 parity / 历史回归参考归档于 ``src/废弃/``，
不再是运行时依赖。``tests/test_langchain_self_contained.py`` 守护这一不变
量：本包源码内不允许出现来自 sibling 包的 ``import``。
"""
__all__ = ["__version__"]

__version__ = "0.1.0"
