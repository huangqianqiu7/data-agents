"""
agents — core agent module (ReAct and Plan-and-Solve paradigms).

Usage::

    from data_agent_refactored.agents import (
        PlanAndSolveAgent, PlanAndSolveAgentConfig,
        ReActAgent, ReActAgentConfig,
    )
"""
from data_agent_refactored.agents.model import (
    ModelAdapter,
    ModelMessage,
    ModelStep,
    OpenAIModelAdapter,
    ScriptedModelAdapter,
)
from data_agent_refactored.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    PLAN_AND_SOLVE_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_refactored.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_refactored.agents.json_parser import parse_model_step, parse_plan
from data_agent_refactored.config import ReActAgentConfig, PlanAndSolveAgentConfig
from data_agent_refactored.agents.react_agent import ReActAgent
from data_agent_refactored.agents.plan_solve_agent import PlanAndSolveAgent

__all__ = [
    # Model
    "ModelAdapter",
    "ModelMessage",
    "ModelStep",
    "OpenAIModelAdapter",
    "ScriptedModelAdapter",
    # Prompts
    "REACT_SYSTEM_PROMPT",
    "PLAN_AND_SOLVE_SYSTEM_PROMPT",
    "build_observation_prompt",
    "build_system_prompt",
    "build_task_prompt",
    # Runtime
    "AgentRunResult",
    "AgentRuntimeState",
    "StepRecord",
    # Parsing
    "parse_model_step",
    "parse_plan",
    # Agents
    "ReActAgent",
    "ReActAgentConfig",
    "PlanAndSolveAgent",
    "PlanAndSolveAgentConfig",
]
