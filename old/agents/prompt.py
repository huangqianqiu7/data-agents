# 从 __future__ 模块导入 annotations，允许在当前代码中使用未来版本的类型提示语法
# 例如在定义类时，可以提前引用尚未完全定义的类型，让代码更简洁
from __future__ import annotations

# 导入 Python 内置的 json 模块，用于处理 JSON 数据的序列化和反序列化
import json

# 从当前项目的数据结构模块中导入 PublicTask 类，它代表了一个具体的评测任务（包含问题、难度等信息）
from data_agent_baseline.benchmark.schema import PublicTask

# 定义 ReAct Agent 的核心系统提示词 (System Prompt)。
# 它的作用是给大模型设定“人设”，并下达绝对不能违背的“军规”。
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
# .strip() 用于去除多余的头尾换行符，保持提示词整洁。
# 这里的规则 5、6、7 极其关键，它们强制大模型只输出纯净的 JSON 格式，
# 这是为了防止大模型“说废话”，从而让代码能够稳定地解析大模型的输出。

# 定义输出样例 (Few-shot Prompting)。
# 给大模型看两个正确的回复模板，告诉它“当你想思考时该怎么回复”以及“当你想交卷时该怎么回复”。
# 这种提供样例的方式能大幅提升大模型输出格式的稳定性。
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


# 构建系统提示词 (System Prompt) 的工厂函数。
# 它把"基础人设 + 工具说明 + 输出样例 + 格式强制指令"拼接成一条完整的系统消息，
# 发送给大模型后，大模型就知道自己该扮演什么角色、能用什么工具、该怎么回复。
#
# 参数:
#   tool_descriptions: 所有可用工具的名称和参数说明（由 tool 注册模块自动生成的字符串）
#   system_prompt:     可选的自定义系统提示词；如果不传，则使用默认的 REACT_SYSTEM_PROMPT
# 返回:
#   拼接好的完整系统提示词字符串
def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    # 如果调用者没有传入自定义 system_prompt，就使用上面定义的默认 REACT_SYSTEM_PROMPT
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    # 将各部分按顺序拼接：基础提示词 → 可用工具列表 → 输出样例 → 格式强制指令
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        # 最后再次强调输出格式要求，防止大模型在对话后期"遗忘"格式约束
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


# 构建任务提示词 (User Prompt) 的函数。
# 每次向大模型发送新一轮对话时，都会用这个函数把具体的评测问题包装成一条用户消息。
#
# 参数:
#   task: 一个 PublicTask 对象，包含 question（问题文本）等字段
# 返回:
#   格式化后的任务提示词字符串
def build_task_prompt(task: PublicTask) -> str:
    return (
        # 将问题内容嵌入提示词
        f"Question: {task.question}\n"
        # 提醒大模型：工具操作的文件路径都是相对于任务上下文目录的
        "All tool file paths are relative to the task context directory. "
        # 告知大模型：当得出最终结果表格后，必须调用 answer 工具来提交答案
        "When you have the final table, call the `answer` tool."
    )


# 构建观测结果提示词的函数。
# 每当 Agent 执行完一个工具调用后，工具会返回一个结果（observation），
# 这个函数将结果格式化为 JSON 字符串，再作为一条消息反馈给大模型，
# 让大模型基于这个观测结果继续推理下一步动作。
#
# 参数:
#   observation: 工具执行后返回的结果字典（如文件列表、数据内容等）
# 返回:
#   格式化后的观测提示词字符串，形如 "Observation:\n{JSON内容}"
def build_observation_prompt(observation: dict[str, object]) -> str:
    # 将 observation 字典序列化为格式化的 JSON 字符串
    # ensure_ascii=False: 保留中文等非 ASCII 字符的原样输出
    # indent=2: 缩进 2 个空格，便于大模型阅读理解
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
