"""
LangGraph agent layer — graph node implementations.

This package intentionally does NOT eagerly import its submodules: doing so
would force every consumer (including light-weight modules like
``tools/descriptions.py``) to load the full graph stack just to reach a
single constant, which previously created import cycles. Import the
specific submodule you need explicitly, e.g.::

    from data_agent_langchain.agents.gate import gate_node
    from data_agent_langchain.agents.model_node import model_node
"""
