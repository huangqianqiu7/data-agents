"""
agents_v2 — 重构后的 Agent 模块。

基于原 agents 包的 plan_and_solve.py 重构，将 819 行单体拆分为模块化架构。
原始文件保持不变，本包为独立的新实现。

使用示例：
    from data_agent_baseline.agents_v2 import (
        PlanAndSolveAgent, PlanAndSolveAgentConfig,
        ReActAgent, ReActAgentConfig,
    )
"""
from data_agent_baseline.agents_v2.model import (
    ModelAdapter,
    ModelMessage,
    ModelStep,
    OpenAIModelAdapter,
    ScriptedModelAdapter,
)
from data_agent_baseline.agents_v2.prompt import (
    REACT_SYSTEM_PROMPT,
    PLAN_AND_SOLVE_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents_v2.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.agents_v2.json_parser import parse_model_step, parse_plan
from data_agent_baseline.agents_v2.react_agent import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents_v2.plan_solve_agent import PlanAndSolveAgent, PlanAndSolveAgentConfig

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
