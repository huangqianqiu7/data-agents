"""
LangGraph-side tool layer.

Each built-in tool is implemented as its own ``BaseTool`` subclass file
(LANGCHAIN_MIGRATION_PLAN.md v3 D8) so schema / description / runtime
behaviour stay co-located. ``descriptions.py`` ensures the prompt block
fed to the LLM in ``json_action`` mode is byte-for-byte identical to the
legacy ``ToolRegistry.describe_for_prompt()`` output (v3 D8 / §6.5).

This package intentionally does NOT eagerly import its submodules: doing
so would force every ``from data_agent_langchain.tools.X import Y``
consumer to load the full tools stack (descriptions, factory, all tools),
which re-enters ``agents.gate`` for gate constants and creates circular
imports. Import the specific submodule you need explicitly, e.g.::

    from data_agent_langchain.tools.tool_runtime import ToolRuntime, ToolRuntimeResult
    from data_agent_langchain.tools.factory import create_all_tools
    from data_agent_langchain.tools.descriptions import render_legacy_description
    from data_agent_langchain.tools.timeout import call_tool_with_timeout, call_with_timeout
"""
