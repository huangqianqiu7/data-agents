"""
LangGraph 后端的 prompt 构建器集合。

职责：
  - 引入 :mod:`agents.prompt_strings` 中保存的 LLM 字符串常量。
  - 拼装 ``BaseMessage`` 列表（system / user / scratchpad）供 ``ChatOpenAI``
    或 LangGraph ``model_node`` 使用。
  - 兼容 ``json_action`` 与 ``tool_calling`` 两种动作模式（v4 §11.3 / C9）。

不在本模块定义 LLM 提示词字符串本身——那些常量集中在
``agents/prompt_strings.py``，便于 review 与 parity 比对。
"""
from __future__ import annotations

from typing import Iterable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from data_agent_langchain.agents.prompt_strings import (
    EXECUTION_INSTRUCTION,
    PLAN_AND_SOLVE_SYSTEM_PROMPT,
    PLANNING_INSTRUCTION,
    REACT_SYSTEM_PROMPT,
    RESPONSE_EXAMPLES,
)
from data_agent_langchain.agents.runtime import StepRecord
from data_agent_langchain.benchmark.schema import PublicTask
from data_agent_langchain.memory.types import MemoryHit
from data_agent_langchain.memory.working import build_scratchpad_messages
from data_agent_langchain.tools.descriptions import render_legacy_prompt_block


# ---------------------------------------------------------------------------
# 任务提示词
# ---------------------------------------------------------------------------

def build_task_prompt(task: PublicTask) -> str:
    """构造单个任务的 user-message 主体。"""
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "IMPORTANT: You must call `list_context` first to discover available files. "
        "Do NOT guess file paths — only use paths returned by `list_context`. "
        "When you have the final table, call the `answer` tool."
    )


# ---------------------------------------------------------------------------
# 系统提示词组装
# ---------------------------------------------------------------------------

def _strip_json_action_rules(text: str) -> str:
    """从系统提示词里删掉所有强制 JSON-fenced 输出的规则行。

    在 ``tool_calling`` 模式下模型应该直接发起 tool call，而不是在 message
    content 里再写一层 JSON；保留 json_action 的强制规则会让模型困惑。
    """
    blocked = (
        "always return exactly one json object",
        "always wrap that json object",
        "wrapped in a ```json fenced block",
        "do not output any text before or after",
        # 任何提到 ```json 字面量的规则行 → fenced JSON 输出约束，tool_calling 无需。
        # 兜底捕获新写法（如 "Do not place the JSON object ... must not start with anything other than ```json"）。
        "```json",
    )
    return "\n".join(
        line for line in text.splitlines()
        if not any(token in line.lower() for token in blocked)
    ).strip()


def _tool_calling_instruction() -> str:
    """``tool_calling`` 模式的执行约束（替代 fenced JSON 的强制规则）。"""
    return (
        "Use the bound tools by making exactly one tool call for each action. "
        "Do not write a JSON action object in message content. "
        "Call the `answer` tool when you have the final table."
    )


def _build_react_system_text() -> str:
    """组装 ReAct 系统消息正文（json_action 模式，与 legacy parity）。"""
    return (
        f"{REACT_SYSTEM_PROMPT}\n\n"
        "Available tools:\n"
        f"{render_legacy_prompt_block()}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def _build_react_tool_calling_system_text() -> str:
    """组装 ReAct 系统消息正文（tool_calling 模式）。"""
    base = _strip_json_action_rules(REACT_SYSTEM_PROMPT)
    return (
        f"{base}\n\n"
        f"Available tools:\n{render_legacy_prompt_block()}\n\n"
        f"{_tool_calling_instruction()}"
    )


def _build_plan_solve_system_text() -> str:
    """组装 Plan-and-Solve 系统消息正文（json_action 模式）。"""
    return (
        f"{PLAN_AND_SOLVE_SYSTEM_PROMPT}\n\n"
        f"Available tools:\n{render_legacy_prompt_block()}\n\n"
        "Return a single ```json fenced block with keys "
        "`thought`, `action`, `action_input`."
    )


def _build_plan_solve_tool_calling_system_text() -> str:
    """组装 Plan-and-Solve 系统消息正文（tool_calling 模式）。"""
    base = _strip_json_action_rules(PLAN_AND_SOLVE_SYSTEM_PROMPT)
    return (
        f"{base}\n\n"
        f"Available tools:\n{render_legacy_prompt_block()}\n\n"
        f"{_tool_calling_instruction()}"
    )


def _build_planning_system_text() -> str:
    """组装 *规划阶段* 的系统消息（仅 Plan-and-Solve 使用）。"""
    return (
        f"{PLAN_AND_SOLVE_SYSTEM_PROMPT}\n\n"
        f"Available tools:\n{render_legacy_prompt_block()}\n\n"
        "Return a single ```json fenced block."
    )


# ---------------------------------------------------------------------------
# 公共构建器
# ---------------------------------------------------------------------------

def build_react_messages(
    task: PublicTask,
    steps: Iterable[StepRecord],
    *,
    action_mode: str = "json_action",
    max_obs_chars: int = 3000,
    max_context_tokens: int = 24000,
) -> list[BaseMessage]:
    """组装 ReAct 提示词：system + task + scratchpad。"""
    system_text = (
        _build_react_tool_calling_system_text()
        if action_mode == "tool_calling"
        else _build_react_system_text()
    )
    base: list[BaseMessage] = [
        SystemMessage(content=system_text),
        HumanMessage(content=build_task_prompt(task)),
    ]
    return build_scratchpad_messages(
        steps,
        base,
        action_mode=action_mode,
        max_obs_chars=max_obs_chars,
        max_context_tokens=max_context_tokens,
    )


def build_plan_solve_execution_messages(
    task: PublicTask,
    steps: Iterable[StepRecord],
    *,
    plan: list[str],
    plan_index: int,
    action_mode: str = "json_action",
    max_obs_chars: int = 3000,
    max_context_tokens: int = 24000,
) -> list[BaseMessage]:
    """组装 Plan-and-Solve 执行阶段提示词：system + task + 当前步骤指令 + scratchpad。"""
    numbered = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan))
    step_description = plan[plan_index] if plan else ""
    if action_mode == "tool_calling":
        instruction = (
            "Execute the following plan using the bound tools.\n\n"
            f"{numbered}\n\n"
            f"You are on step {plan_index + 1}/{len(plan)}: \"{step_description}\"\n\n"
            "Choose the next action by making exactly one tool call."
        )
    else:
        instruction = EXECUTION_INSTRUCTION.format(
            numbered_plan=numbered,
            current=plan_index + 1,
            total=len(plan),
            step_description=step_description,
        )
    system_text = (
        _build_plan_solve_tool_calling_system_text()
        if action_mode == "tool_calling"
        else _build_plan_solve_system_text()
    )
    base: list[BaseMessage] = [
        SystemMessage(content=system_text),
        HumanMessage(content=f"{build_task_prompt(task)}\n\n{instruction}"),
    ]
    return build_scratchpad_messages(
        steps,
        base,
        action_mode=action_mode,
        max_obs_chars=max_obs_chars,
        max_context_tokens=max_context_tokens,
    )


def build_planning_messages(
    task: PublicTask,
    *,
    history_hint: str = "",
) -> list[BaseMessage]:
    """组装 Plan-and-Solve 规划阶段的提示词消息（让 LLM 产出 plan）。"""
    user = build_task_prompt(task) + "\n\n"
    if history_hint:
        user += f"Previous observations:\n{history_hint}\n\n"
    user += PLANNING_INSTRUCTION
    return [
        SystemMessage(content=_build_planning_system_text()),
        HumanMessage(content=user),
    ]


# 历史 re-export：保留这些常量名以兼容现有 ``from agents.prompts import REACT_SYSTEM_PROMPT``
# 风格的调用点；新代码请直接 ``from agents.prompt_strings import ...``。
_DATASET_FACTS_POLICY = (
    "These facts are recalled from prior runs and may be stale. "
    "They are not the current task file inventory. "
    "Do not use paths from this section unless they are verified by the current task's "
    "list_context output. Current tool observations override these facts."
)

_CORPUS_SNIPPETS_POLICY = (
    "These snippets are task documentation references. "
    "They are not verified file inventory or schema. "
    "Use current list_context and tool observations as the source of truth; "
    "current tool observations override these snippets."
)


def render_dataset_facts(hits: list[MemoryHit]) -> str:
    """Render MemoryHit summaries into a whitelisted prompt fragment.

    Only the ``summary`` field is used so unauthorized payload fields cannot enter
    the prompt through this path.
    """
    if not hits:
        return ""
    lines = [
        "## Dataset facts (from prior runs, informational only)",
        _DATASET_FACTS_POLICY,
    ]
    for hit in hits:
        lines.append(f"- {hit.summary}")
    return "\n".join(lines)


def render_corpus_snippets(hits: list[MemoryHit], *, budget_chars: int) -> str:
    """渲染 task corpus snippets；只使用 ``summary`` 字段并严格遵守字符预算。"""
    if not hits or budget_chars <= 0:
        return ""

    header = "## Reference snippets (from task documentation)"
    base = f"{header}\n{_CORPUS_SNIPPETS_POLICY}"
    if len(base) >= budget_chars:
        return _truncate_prompt_fragment(base, budget_chars)

    text = base
    for hit in hits:
        line = f"- {hit.summary}"
        remaining = budget_chars - len(text) - 1
        if remaining <= 0:
            break
        if len(line) > remaining:
            line = _truncate_prompt_fragment(line, remaining)
        if not line:
            break
        text = f"{text}\n{line}"
        if len(text) >= budget_chars:
            break
    return text


def _truncate_prompt_fragment(text: str, limit: int) -> str:
    """截断 prompt 片段并保证返回长度不超过 ``limit``。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


__all__ = [
    "EXECUTION_INSTRUCTION",
    "PLAN_AND_SOLVE_SYSTEM_PROMPT",
    "PLANNING_INSTRUCTION",
    "REACT_SYSTEM_PROMPT",
    "RESPONSE_EXAMPLES",
    "build_planning_messages",
    "build_plan_solve_execution_messages",
    "build_react_messages",
    "build_task_prompt",
    "render_corpus_snippets",
    "render_dataset_facts",
]
