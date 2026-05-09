"""
data_agent_common — shared package between data_agent_refactored (legacy
hand-written agent loop) and data_agent_langchain (LangGraph backend).

Holds the parts that must NOT be duplicated between the two backends:
  - DABench dataset loader and task / answer schema
  - Path-safe filesystem, sandboxed python, read-only sqlite tool primitives
  - StepRecord / AgentRunResult / ModelMessage / ModelStep runtime objects
  - Multi-tier JSON parsing (parse_model_step / parse_plan)
  - Sanitization protocol (__error__ / __replan_failed__ placeholders)
  - Cross-backend constants such as FALLBACK_STEP_PROMPT

See ``src/LangChain/LANGCHAIN_MIGRATION_PLAN.md`` §3.1 for the full layout.
"""
__all__ = ["__version__"]

__version__ = "0.1.0"
