"""
统一提示词模块。

合并 ReAct 和 Plan-and-Solve 的所有提示词定义与构建函数。

常量：
  - REACT_SYSTEM_PROMPT:          ReAct 系统提示词
  - PLAN_AND_SOLVE_SYSTEM_PROMPT: Plan-and-Solve 系统提示词
  - RESPONSE_EXAMPLES:            输出格式样例
  - PLANNING_INSTRUCTION:         Plan-and-Solve 规划阶段指令
  - EXECUTION_INSTRUCTION:        Plan-and-Solve 执行阶段指令模板

函数：
  - build_system_prompt():     构建完整系统提示词
  - build_task_prompt():       构建任务提示词
  - build_observation_prompt(): 构建观测结果提示词
"""
from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


# =====================================================================
# ReAct 系统提示词
# =====================================================================
REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()


# =====================================================================
# Plan-and-Solve 系统提示词
# =====================================================================
PLAN_AND_SOLVE_SYSTEM_PROMPT = """
You are a Plan-and-Solve data agent that works in two phases:

Phase 1 – Planning: Analyze the task and devise a numbered step-by-step plan.
Phase 2 – Execution: Carry out each step by calling the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information observed through provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object wrapped in a ```json fenced block.
6. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()


# =====================================================================
# 输出格式样例（Few-shot Prompting）
# =====================================================================
RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


# =====================================================================
# Plan-and-Solve 规划阶段指令
# =====================================================================
PLANNING_INSTRUCTION = """
Create a step-by-step plan to solve this task.
Return a JSON object with a single key "plan" — a list of concise step descriptions.
The plan should start with inspecting data files and end with calling the `answer` tool.

Example:
```json
{"plan": ["List context files", "Read the CSV data to understand schema", "Compute the required metric with Python", "Call answer tool with the result table"]}
```
""".strip()


# =====================================================================
# Plan-and-Solve 执行阶段指令模板
# =====================================================================
EXECUTION_INSTRUCTION = """
=== Your Plan ===
{numbered_plan}

You are on step {current}/{total}: "{step_description}"

Return a JSON with keys `thought`, `action`, `action_input`.
If this step's goal is achieved, say so in your thought and proceed to the next step's action.

```json
{{"thought":"...","action":"tool_name","action_input":{{...}}}}
```
""".strip()


# =====================================================================
# 提示词构建函数
# =====================================================================
def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    """构建完整的系统提示词（人设 + 工具说明 + 输出样例 + 格式约束）。"""
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    """构建任务提示词。"""
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    """构建观测结果提示词。"""
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
