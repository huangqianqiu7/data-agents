"""
Runtime objects shared across LangGraph nodes.

This package intentionally does NOT eagerly import its submodules: doing so
would force every ``from data_agent_langchain.runtime.X import Y`` consumer
to load the full runtime stack (rehydrate, context, state, result), which
re-enters ``tools.*`` and ``agents.*`` and creates circular imports. Import
the specific submodule you need explicitly, e.g.::

    from data_agent_langchain.runtime.state import RunState
    from data_agent_langchain.runtime.context import get_current_app_config
    from data_agent_langchain.runtime.rehydrate import build_runtime, rehydrate_task
    from data_agent_langchain.runtime.result import AgentRunResult, StepRecord
"""
