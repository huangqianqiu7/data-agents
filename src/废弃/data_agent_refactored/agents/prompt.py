"""
统一的提示词模板与构建函数，供 ReAct 和 Plan-and-Solve 两种智能体使用。

本模块集中管理所有与大语言模型 (LLM) 交互时使用的系统提示词 (system prompt)、
少样本示例 (few-shot examples) 以及动态拼接提示词的辅助函数。
将提示词与业务逻辑解耦，便于统一维护和迭代。

常量 (Constants):
  - ``REACT_SYSTEM_PROMPT``      — ReAct 智能体的系统提示词
  - ``PLAN_AND_SOLVE_SYSTEM_PROMPT`` — Plan-and-Solve 智能体的系统提示词
  - ``RESPONSE_EXAMPLES``        — 供模型参考的少样本响应示例
  - ``PLANNING_INSTRUCTION``     — Plan-and-Solve 智能体 "规划阶段" 的指令模板
  - ``EXECUTION_INSTRUCTION``    — Plan-and-Solve 智能体 "执行阶段" 的指令模板

函数 (Functions):
  - :func:`build_system_prompt`      — 拼接完整的系统提示词（角色 + 工具 + 示例 + 格式约束）
  - :func:`build_task_prompt`        — 根据任务对象构建用户侧提示词
  - :func:`build_observation_prompt` — 将工具返回的观测结果序列化为用户消息
"""
from __future__ import annotations

# build_observation_prompt 现由 data_agent_common 统一定义，确保两个 backend
# 渲染出来的 user 消息字符串完全一致（LANGCHAIN_MIGRATION_PLAN.md §3.1）。
from data_agent_common.agents.prompts import build_observation_prompt

# PublicTask 是对公开数据集任务的结构化描述（包含 question 等字段）
from data_agent_refactored.benchmark.schema import PublicTask


# ---------------------------------------------------------------------------
# ReAct 智能体系统提示词
# ---------------------------------------------------------------------------
# ReAct (Reasoning + Acting) 模式：模型在每一步先进行 *思考 (thought)*，
# 再决定调用哪个 *工具 (action)*，并传入对应参数 (action_input)。
# 该提示词告知模型其角色定位和必须遵守的输出格式规范。
REACT_SYSTEM_PROMPT: str = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Workflow (follow this order strictly; the runtime will block heavy actions until you complete step 2):
1. Discover: call `list_context` to see which files exist.
2. Read knowledge.md FIRST (mandatory if present): call `read_doc` on `knowledge.md` before touching any data file. This document defines the exact meaning of business terms, field names, and calculation rules. If `truncated: true`, call again with a larger `max_chars`. Extract the answers to these questions from it:
   - What does each key term in the question mean in this dataset? (e.g. "cost" = which field? "members" = which table and filter?)
   - Are there domain-specific thresholds or reference ranges? (e.g. "abnormal WBC" = WBC > 11.0)
   - Which field or column should be used for aggregation or filtering?
3. Inspect data structure: call `read_csv` (max_rows 5–8), `read_json` (max_items 5–8), or `inspect_sqlite_schema` on **ALL** relevant files identified in steps 1–2. For every key term in the question, list ALL candidate columns across all tables that could semantically match it — not just columns with the same name, but any column whose meaning could plausibly correspond to the term (e.g. "track number" could be `round`, `position`, `circuitId`, or `number`). Compare sample values of these candidates to understand how they differ.
4. Map question terms to fields: for each key term, enumerate the candidate columns identified in step 3 and explicitly reason which **table.column** is the correct match. Consider the context of the sentence — e.g. "he was in track number X" describes a person's status/rank, so it maps to a ranking field like `position`, not a sequential field like `round`. Ground your choice in knowledge.md definitions, example SQL patterns, and sample data. If knowledge.md examples use JOINs, your query should follow the same JOIN pattern.
5. Compute: use `execute_python` or `execute_context_sql` to aggregate, join, or scan full files using the correct fields identified in step 4.
6. Verify: cross-check your result using a different method or tool. For example, if you used Python in step 5, verify with SQL, or vice versa. If only one tool is applicable, re-compute with a different approach (e.g., a simpler aggregation, or print intermediate row counts) to confirm the result is correct.
7. Submit: call `answer` with the final table. Include only columns explicitly requested in the question — do not add extra columns.

Rules:
1. Base your answer only on information you can observe through the provided tools.
2. The task is complete only when you call the `answer` tool.
3. The `answer` tool must receive a table with `columns` and `rows`.
4. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
5. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
6. Do not output any text before or after the fenced JSON block.
7. Never guess field meanings from column names alone — always ground your field selection in knowledge.md definitions or observed sample data. If knowledge.md has an "Ambiguity Resolution" section that distinguishes similar fields (e.g. `rank` vs `positionOrder`), those definitions are **authoritative** and must override your intuition.
8. Submit only the columns the question asks for. Extra columns are penalized.
9. Do NOT filter out or exclude zero values unless the question or knowledge.md explicitly instructs you to. Treat 0 as a valid data point — SQL AVG/SUM/COUNT include zeros, and your Python code must behave the same way. However, NULL (or missing/empty values) should remain NULL — do NOT convert them to 0 with COALESCE or fillna. SQL AVG naturally skips NULLs; your code should do the same.
10. Before submitting, perform a sanity check in your thought: verify the result is reasonable given the data scale and question context. If something looks off, re-examine your computation logic.
11. When a question term could map to multiple columns across tables (whether same-named or differently-named), you MUST: (a) list all candidates, (b) write a short Python/SQL snippet to print sample values from EACH candidate column for the same records, so you can see the actual difference, (c) then choose the correct one based on the printed evidence and sentence context. Pay attention to sentence structure: "his number" → entity attribute (e.g. `drivers.number`); "he was in [X] number" → ranking/status (e.g. `standings.position`). Do not default to the first plausible column — always print and compare before deciding.
12. Follow the JOIN patterns shown in knowledge.md examples. If knowledge.md shows `qualifying JOIN drivers ON driverId`, your computation must do the same — do not take shortcuts by using only one table when the correct answer requires a JOIN.
13. When the question specifies a value at lower precision than the data (e.g. question says "0:01:54" but data contains "1:54.455", "1:54.960"), match ALL records that fall within that precision range — do not pick only one. Use prefix or range matching (e.g. `LIKE '1:54%'` or `startswith('1:54')`) and return every matching row.
14. If your verification step discovers results that differ from or extend your initial computation (e.g. more matching rows), you MUST update your answer accordingly before submitting. Never ignore verification output.
15. A database file may contain only a sample of the full data (e.g. filename contains `1k`, `sample`, or its date range is much narrower than the question asks). When the DB lacks rows for the requested time period, do NOT immediately return empty. Instead: (a) check whether the DB still provides useful **mapping relationships** (e.g. CustomerID → GasStationID) that you can combine with other data sources (CSV/JSON) that DO cover the requested period, (b) use Python to join across sources — e.g. find relevant CustomerIDs from the CSV, look up their GasStationIDs from the DB, then get attributes from JSON. Never return an empty result without first trying all cross-source join paths.

Keep reasoning concise and grounded in the observed data.
""".strip()
# 注意：.strip() 去除首尾空白行，确保拼接时不会引入多余换行。
#
# 【中文翻译】
# 你是一个 ReAct 风格的数据智能体。
#
# 你正在解决一个来自公开数据集的任务。你只能通过提供的工具查看任务 `context/` 目录中的文件。
#
# 工作流程（严格按此顺序执行；在你完成第 2 步之前，运行时会阻止重型操作）：
# 1. 发现：调用 `list_context` 查看有哪些文件。
# 2. 首先读取 knowledge.md（如果存在则为强制要求）：在接触任何数据文件之前，对 `knowledge.md` 调用 `read_doc`。该文档定义了业务术语、字段名和计算规则的确切含义。如果返回 `truncated: true`，则使用更大的 `max_chars` 再次调用。从中提取以下问题的答案：
#    - 问题中的每个关键术语在该数据集中是什么意思？（例如，"cost" = 哪个字段？"members" = 哪张表以及哪个过滤条件？）
#    - 是否存在领域特定的阈值或参考范围？（例如，"abnormal WBC" = WBC > 11.0）
#    - 应使用哪个字段或列进行聚合或过滤？
# 3. 检查数据结构：对第 1–2 步识别出的所有相关文件调用 `read_csv`（max_rows 5–8）、`read_json`（max_items 5–8）或 `inspect_sqlite_schema`。对于问题中的每个关键术语，列出所有表中所有可能在语义上匹配它的候选列——不仅是同名列，还包括任何含义上可能对应该术语的列（例如，"track number" 可能是 `round`、`position`、`circuitId` 或 `number`）。比较这些候选列的样例值，以理解它们的差异。
# 4. 将问题术语映射到字段：对每个关键术语，枚举第 3 步识别出的候选列，并明确推理哪个 **table.column** 才是正确匹配。考虑句子的上下文——例如，"he was in track number X" 描述的是某人的状态/排名，因此应映射到像 `position` 这样的排名字段，而不是像 `round` 这样的顺序字段。你的选择必须基于 knowledge.md 定义、SQL 示例模式和样例数据。如果 knowledge.md 的示例使用 JOIN，你的查询也应遵循相同的 JOIN 模式。
# 5. 计算：使用 `execute_python` 或 `execute_context_sql`，基于第 4 步识别出的正确字段进行聚合、连接或全文件扫描。
# 6. 验证：使用另一种方法或工具交叉检查结果。例如，如果第 5 步使用了 Python，则用 SQL 验证；反之亦然。如果只有一种工具适用，则用另一种方式重新计算（例如，更简单的聚合，或打印中间行数）来确认结果正确。
# 7. 提交：调用 `answer` 返回最终表。只包含问题明确要求的列——不要添加额外列。
#
# 规则：
# 1. 你的答案只能基于通过提供的工具观察到的信息。
# 2. 只有当你调用 `answer` 工具时，任务才算完成。
# 3. `answer` 工具必须接收一个包含 `columns`（列名）和 `rows`（行数据）的表格。
# 4. 始终返回恰好一个包含 `thought`、`action` 和 `action_input` 键的 JSON 对象。
# 5. 始终将该 JSON 对象包裹在恰好一个以 ```json 开头、以 ``` 结尾的围栏代码块中。
# 6. 不要在围栏 JSON 代码块的前后输出任何文本。
# 7. 绝不要仅凭列名猜测字段含义——字段选择必须始终基于 knowledge.md 定义或观察到的样例数据。如果 knowledge.md 中有区分类似字段（例如 `rank` 与 `positionOrder`）的 "Ambiguity Resolution" 部分，这些定义就是**权威依据**，必须覆盖你的直觉。
# 8. 只提交问题要求的列。额外列会被惩罚。
# 9. 除非问题或 knowledge.md 明确要求，否则不要过滤或排除零值。将 0 视为有效数据点——SQL AVG/SUM/COUNT 会包含零，你的 Python 代码也必须同样处理。但是，NULL（或缺失/空值）应保持为 NULL——不要用 COALESCE 或 fillna 将它们转换为 0。SQL AVG 会自然跳过 NULL；你的代码也应如此。
# 10. 提交前，在 thought 中执行合理性检查：根据数据规模和问题上下文验证结果是否合理。如果看起来不对，重新检查你的计算逻辑。
# 11. 当问题术语可能映射到多张表中的多个列（无论同名还是不同名）时，你必须：(a) 列出所有候选列，(b) 编写简短的 Python/SQL 片段，为相同记录打印每个候选列的样例值，以便看到实际差异，(c) 然后根据打印出的证据和句子上下文选择正确列。注意句子结构："his number" → 实体属性（例如 `drivers.number`）；"he was in [X] number" → 排名/状态（例如 `standings.position`）。不要默认选择第一个看似合理的列——决策前必须打印并比较。
# 12. 遵循 knowledge.md 示例中展示的 JOIN 模式。如果 knowledge.md 显示 `qualifying JOIN drivers ON driverId`，你的计算也必须这样做——当正确答案需要 JOIN 时，不要只使用单表走捷径。
# 13. 当问题给出的值精度低于数据精度时（例如问题写 "0:01:54"，但数据包含 "1:54.455"、"1:54.960"），应匹配落在该精度范围内的所有记录——不要只选一条。使用前缀或范围匹配（例如 `LIKE '1:54%'` 或 `startswith('1:54')`），并返回每一条匹配行。
# 14. 如果验证步骤发现结果与初始计算不同或扩展了初始结果（例如更多匹配行），提交前必须相应更新答案。绝不要忽略验证输出。
# 15. 数据库文件可能只包含完整数据的一部分样本（例如文件名包含 `1k`、`sample`，或其日期范围远窄于问题要求）。当数据库缺少请求时间段的行时，不要立刻返回空结果。而是：(a) 检查数据库是否仍提供有用的**映射关系**（例如 CustomerID → GasStationID），可与覆盖请求时间段的其他数据源（CSV/JSON）结合，(b) 使用 Python 跨数据源连接——例如，从 CSV 找到相关 CustomerID，在数据库中查找其 GasStationID，再从 JSON 获取属性。不要在尝试所有跨源连接路径之前返回空结果。
#
# 保持推理简洁，并以观察到的数据为依据。


# ---------------------------------------------------------------------------
# Plan-and-Solve 智能体系统提示词
# ---------------------------------------------------------------------------
# Plan-and-Solve 模式将任务拆分为两个阶段：
#   阶段 1 — 规划 (Planning)：分析任务并生成编号步骤计划。
#   阶段 2 — 执行 (Execution)：按计划逐步调用工具完成任务。
# 相比 ReAct 的逐步推理，Plan-and-Solve 更擅长处理需要全局规划的复杂任务。
PLAN_AND_SOLVE_SYSTEM_PROMPT: str = """
You are a Plan-and-Solve data agent that works in two phases:

Phase 1 – Planning: Analyze the task and devise a numbered step-by-step plan.
Phase 2 – Execution: Carry out each step by calling the provided tools.

Rules:
1. Your FIRST action MUST be `list_context` to discover what files actually exist. NEVER guess or assume file paths.
2. Only use file paths that were returned by `list_context`. Do NOT fabricate paths like "doc/knowledge.md".
3. Base your answer only on information observed through provided tools.
4. The task is complete only when you call the `answer` tool.
5. The `answer` tool must receive a table with `columns` and `rows`.
6. Always return exactly one JSON object wrapped in a ```json fenced block.
7. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()
#
# 【中文翻译】
# 你是一个 Plan-and-Solve（计划与执行）数据智能体，分两个阶段工作：
#
# 阶段 1 — 规划：分析任务并制定一个带编号的逐步计划。
# 阶段 2 — 执行：通过调用提供的工具来完成每一步。
#
# 规则：
# 1. 在回答之前，先使用工具查看可用的上下文。
# 2. 你的回答只能基于通过工具观察到的信息。
# 3. 只有当你调用 `answer` 工具时，任务才算完成。
# 4. `answer` 工具必须接收一个包含 `columns`（列名）和 `rows`（行数据）的表格。
# 5. 始终返回恰好一个包裹在 ```json 围栏代码块中的 JSON 对象。
# 6. 不要在围栏 JSON 代码块的前后输出任何文本。
#
# 保持推理简洁，并以观察到的数据为依据。


# ---------------------------------------------------------------------------
# 少样本示例 (Few-shot examples)
# ---------------------------------------------------------------------------
# 为模型提供两种典型场景的示例响应：
#   1. 需要查看上下文文件时 → 调用 list_context 工具
#   2. 已获得最终答案时     → 调用 answer 工具
# 少样本示例帮助模型理解期望的 JSON 输出格式，减少格式错误。
RESPONSE_EXAMPLES: str = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()
#
# 【中文翻译】
# 当你需要查看上下文时的示例响应：
# ```json
# {"thought":"我应该先查看可用的文件。","action":"list_context","action_input":{"max_depth":4}}
# ```
#
# 当你已经得到最终答案时的示例响应：
# ```json
# {"thought":"我已经得到了最终结果表。","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
# ```


# ---------------------------------------------------------------------------
# Plan-and-Solve 规划阶段指令
# ---------------------------------------------------------------------------
# 在规划阶段发送给模型的指令，要求其返回一个包含 "plan" 键的 JSON 对象，
# 值为一个字符串列表，每个元素是计划中的一个步骤描述。
# 计划应以 "检查数据文件" 开头，以 "调用 answer 工具" 结尾。
PLANNING_INSTRUCTION: str = """
Create a step-by-step plan to solve this task.
Return a JSON object with a single key "plan" — a list of concise step descriptions.
The plan should start with inspecting data files and end with calling the `answer` tool.

Example:
```json
{"plan": ["List context files", "Read the CSV data to understand schema", "Compute the required metric with Python", "Call answer tool with the result table"]}
```
""".strip()
#
# 【中文翻译】
# 创建一个逐步计划来解决此任务。
# 返回一个只有单个键 "plan" 的 JSON 对象 — 值为一个简洁的步骤描述列表。
# 计划应以检查数据文件开头，以调用 `answer` 工具结尾。
#
# 示例：
# ```json
# {"plan": ["列出上下文文件", "读取 CSV 数据以了解其结构", "用 Python 计算所需指标", "调用 answer 工具返回结果表"]}
# ```


# ---------------------------------------------------------------------------
# Plan-and-Solve 执行阶段指令模板
# ---------------------------------------------------------------------------
# 执行阶段的动态模板，使用 Python str.format() 占位符：
#   {numbered_plan}    — 带编号的完整计划文本
#   {current}          — 当前正在执行的步骤序号
#   {total}            — 计划总步骤数
#   {step_description} — 当前步骤的文字描述
# 注意：模板中 {{ 和 }} 是对花括号的转义，最终输出为 { 和 }，
# 用来给模型展示 JSON 示例格式。
EXECUTION_INSTRUCTION: str = """
=== Your Plan ===
{numbered_plan}

You are on step {current}/{total}: "{step_description}"

Return a JSON with keys `thought`, `action`, `action_input`.
If this step's goal is achieved, say so in your thought and proceed to the next step's action.

```json
{{"thought":"...","action":"tool_name","action_input":{{...}}}}
```
""".strip()
#
# 【中文翻译】
# === 你的计划 ===
# {numbered_plan}                          （带编号的完整计划）
#
# 你正在执行第 {current}/{total} 步："{step_description}"
#
# 返回一个包含 `thought`、`action`、`action_input` 键的 JSON。
# 如果当前步骤的目标已达成，请在 thought 中说明，并继续执行下一步的动作。
#
# ```json
# {"thought":"...","action":"工具名称","action_input":{...}}
# ```


# ---------------------------------------------------------------------------
# 提示词构建函数 (Prompt Builders)
# ---------------------------------------------------------------------------

def build_system_prompt(
    tool_descriptions: str,
    system_prompt: str | None = None,
) -> str:
    """
    构建完整的系统提示词，供发送给 LLM 的 system message 使用。

    拼接顺序：
      1. 基础系统提示词（角色定义 + 规则）
      2. 可用工具列表（由调用方传入的文本描述）
      3. 少样本响应示例
      4. 输出格式的强制约束语句

    参数:
        tool_descriptions: 所有可用工具的文本描述，通常由 ToolRegistry 生成。
        system_prompt:     自定义的基础系统提示词；为 None 时默认使用 REACT_SYSTEM_PROMPT。

    返回:
        拼接完成的系统提示词字符串。
    """
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        # "可用工具：\n"（工具描述列表）
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        # "你必须始终返回一个 ```json 围栏代码块，其中包含一个 JSON 对象，
        #  键为 `thought`、`action` 和 `action_input`，不能有其他多余文本。"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    """
    根据任务对象构建用户侧提示词（即作为 user message 发送给 LLM 的内容）。

    包含：
      - 任务的问题文本 (task.question)
      - 提示模型所有文件路径相对于任务上下文目录
      - 提醒模型得到最终结果后调用 answer 工具

    参数:
        task: PublicTask 实例，包含任务的元数据与问题描述。

    返回:
        格式化后的任务提示词字符串。
    """
    return (
        # "问题：{task.question}"
        f"Question: {task.question}\n"
        # "所有工具的文件路径都相对于任务上下文目录。当你得到最终表格后，调用 `answer` 工具。"
        "All tool file paths are relative to the task context directory. "
        "IMPORTANT: You must call `list_context` first to discover available files. "
        "Do NOT guess file paths — only use paths returned by `list_context`. "
        "When you have the final table, call the `answer` tool."
    )


# build_observation_prompt is imported from data_agent_common.agents.prompts
# at the top of this module.
