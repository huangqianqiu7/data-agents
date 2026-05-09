from __future__ import annotations

import json
import re
from dataclasses import dataclass

# =====================================================================
# 新增依赖：json_repair 库
# 大模型经常输出格式有瑕疵的 JSON（比如少个引号、多个逗号等）。
# 原生的 json.loads() 遇到这种情况直接报错，而 json_repair 能尝试自动修复。
# 使用前需要安装：pip install json-repair
# =====================================================================
import json_repair

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry

# =====================================================================
# 新增功能：数据预览门控机制 (Data Preview Gatekeeping)
# =====================================================================
# 这一组常量定义了一个"安全规范"：
# 在大模型真正动手写代码或回答问题之前，必须先"看一眼"数据长什么样。
# 这就像考试时，老师要求你必须先审题再答题，不能上来就瞎写。

# frozenset 是 Python 的"冻结集合"，创建后不可修改，适合用来定义常量集合。
# 这些是"数据预览"类的工具 —— 相当于"审题"动作
_DATA_PREVIEW_ACTIONS = frozenset({"read_csv", "read_json", "read_doc", "inspect_sqlite_schema"})

# 这些是"正式答题"类的工具 —— 必须先审题才能用
_GATED_AFTER_PREVIEW_ACTIONS = frozenset({"execute_python", "execute_context_sql", "answer"})

# 当大模型试图跳过审题直接答题时，系统会返回这段提示信息，
# 告诉它："你还没看过数据呢，先去看看再来！"
_GATE_REMINDER = (
    "Blocked: inspect data shape first. After list_context, successfully call read_csv and/or read_json "
    "on each tabular/JSON file you will use, read_doc on knowledge.md if present, or "
    "inspect_sqlite_schema before SQL — then you may use execute_python, execute_context_sql, or answer."
)


def _has_successful_data_preview(state: AgentRuntimeState) -> bool:
    """
    检查代理的历史步骤中，是否至少成功执行过一次"数据预览"工具。
    遍历所有历史步骤，只要找到一个 ok=True 且 action 属于预览类工具的记录，就返回 True。
    """
    for step in state.steps:
        if not step.ok:
            continue
        if step.action in _DATA_PREVIEW_ACTIONS:
            return True
    return False


# =====================================================================
# 新增功能：运行进度输出辅助函数
# =====================================================================
def _progress_line(message: str) -> None:
    """
    在终端实时打印一行进度信息。
    flush=True 确保信息立即输出，不会被缓冲区延迟。
    这在多进程 (multiprocessing) 运行时尤其重要，否则输出可能会错乱或延迟。
    """
    print(message, flush=True)


def _preview_json(obj: object, max_len: int = 140) -> str:
    """
    将一个 Python 对象转为 JSON 字符串的"预览版"。
    如果字符串太长（超过 max_len），就截断并加上省略号 "..."。
    主要用于在终端日志中显示工具调用的参数，避免刷屏。
    """
    try:
        text = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        text = str(obj)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


# =====================================================================
# 代理配置类
# =====================================================================
@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    # 限制大模型最多只能"思考+行动"多少次。防止它陷入死循环
    max_steps: int = 16

    # 新增配置项：是否在终端打印每一步的执行进度
    # 设为 True 后，终端会实时显示类似 "[dabench] Step 1/16: calling model ..." 的信息
    # 在命令行运行或多进程调试时非常有用，让你知道 Agent 跑到哪一步了
    progress: bool = False

    # 新增配置项：是否强制要求先预览数据再执行代码/回答
    # 设为 True 时，大模型必须先成功调用 read_csv / read_json / read_doc / inspect_sqlite_schema
    # 之后才被允许调用 execute_python / execute_context_sql / answer
    # 目的是防止大模型"盲写"代码，提高回答质量
    require_data_preview_before_compute: bool = True

# =====================================================================
# LLM 输出解析辅助函数：扒掉 Markdown 外衣
# =====================================================================
def _strip_json_fence(raw_response: str) -> str:
    """
    大模型经常喜欢在 JSON 外面白白加上 ```json 和 ``` 的 Markdown 代码块标记。
    这个函数的作用就是把这些外衣脱掉，只提取里面纯净的 JSON 字符串。
    """
    text = raw_response.strip() # strip()去除字符串两头的空白字符
    # re.search(正则表达式规则，要搜索的文本， 特殊标志) re.IGNORECASE (忽略大小写)， re.DOTALL (点号通配一切)
    # 尝试匹配 ```json ... ```，re.DOTALL 表示 . 可以匹配换行符
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    # 如果没写 json，只是写了 ``` ... ``` 也兼容处理
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    # 如果大模型很乖，直接输出了裸的 JSON，就原样返回
    return text


# =====================================================================
# 新增辅助函数：修复大模型常见的结尾括号拼写错误
# =====================================================================
def _fix_trailing_bracket_typo(text: str) -> str:
    """
    大模型有时候会在 JSON 结尾多输出一个 `]`，
    比如输出 {"action_input": {"key": "value"}}]} 这种奇怪的格式。
    这个函数用正则表达式把结尾的 `}]}`  修正为 `}}`。
    re.sub(pattern, replacement, text, count=1) 的 count=1 表示只替换第一个匹配。
    """
    return re.sub(r"\}\s*\]\s*\}\s*$", r"}}", text.strip(), count=1)


# =====================================================================
# 新增辅助函数：严格模式的 JSON 解析（不使用 json_repair）
# =====================================================================
def _try_strict_single_json_object(text: str) -> dict[str, object] | None:
    """
    尝试用最严格的方式解析 JSON。
    如果成功，返回解析出的字典；如果失败或格式不对，返回 None（而不是抛异常）。
    这样上层可以优雅地尝试多种修复策略。
    """
    try:
        # raw_decode 会返回解析出的对象和解析停止的索引位置 (end)
        payload, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        # 原生解析器都无法解析，直接返回 None，让上层换别的方式试试
        return None
    # 检查 JSON 解析完之后，后面还有没有多余的字符串
    remainder = text[end:].strip()
    if remainder:
        # 去掉换行和制表符等空白字符
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            # 后面还有多余内容，表示这不是一个干净的单 JSON 对象
            return None
    # python内置函数isinstance(obj, class) 判断变量obj 是否属于特定的类型class
    if isinstance(payload, dict):
        return payload
    return None


# =====================================================================
# 核心解析函数（增强版）：多层降级策略的 JSON 解析
# =====================================================================
def _load_single_json_object(text: str) -> dict[str, object]:
    """
    将文本解析成一个 JSON 字典对象。采用"三级降级"策略：
    1. 严格模式解析（原始文本）
    2. 严格模式解析（修复结尾括号后的文本）
    3. 使用 json_repair 库进行自动修复解析

    这种多层降级的设计理念是：
    - 优先用最严格的方式解析，保证数据准确性
    - 如果严格模式失败，说明大模型输出有小瑕疵，尝试常见问题修复
    - 如果修复也不行，最后才动用 json_repair 这个"万能胶水"
    """
    text = text.strip()

    # 第一阶段：严格解析（原版 + 修复括号版）
    for variant in (text, _fix_trailing_bracket_typo(text)):
        got = _try_strict_single_json_object(variant)
        if got is not None:
            return got

    # 第二阶段：使用 json_repair 库进行自动修复
    # json_repair.loads() 能够处理各种常见的 JSON 格式错误，
    # 比如缺少引号、多余逗号、单引号代替双引号等
    last_exc: BaseException | None = None
    for variant in (text, _fix_trailing_bracket_typo(text)):
        try:
            repaired = json_repair.loads(variant)
        except Exception as exc:
            last_exc = exc
            continue
        if isinstance(repaired, dict):
            return repaired

    # 如果所有方式都失败了，才抛出异常，让大模型知道它的输出格式有严重问题
    # `from last_exc` 会把最后一个捕获的异常链接到新异常上，方便调试追踪
    raise ValueError(
        "Model response is not valid JSON (strict parse and json_repair both failed)."
    ) from last_exc


# =====================================================================
# 核心解析函数：将文本转化为大模型的"思维步骤"
# =====================================================================
def parse_model_step(raw_response: str) -> ModelStep:
    """
    将大模型返回的文本，经过脱衣、提取、校验，最终变成一个规范的 ModelStep 对象。
    """
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    # 提取三大核心要素：思考过程，工具名称、工具参数
    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    
    # 类型安全校验（防御性编程：永远不要完全信任大模型的输出格式）
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )

# =====================================================================
# ReAct Agent 核心类：控制整个智能体的大脑运作
# =====================================================================
class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,    # 大模型接口
        tools: ToolRegistry,    # 工具箱
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        """
        每次大模型发请求前，都需要把过去的"记忆"重新组装一遍。
        这就是所谓的"上下文 (Context)"。大模型没有真正的记忆，全靠我们每次把历史记录发给它。
        """
        # 1. 塞入系统人设和工具说明书
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        
        # 2. 塞入当前的考题
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        
        # 3. 循环回放历史步骤（经典的 ReAct 历史组装）
        for step in state.steps:
            # 助理(大模型)之前的回复
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            # 用户(系统)返回的工具执行结果或报错
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        """
        🌟 Agent 的主循环 (Main Loop) 🌟
        """
        state = AgentRuntimeState()

        # 从配置中提取常用变量，避免在循环中反复通过 self.config 访问
        prog = self.config.progress       # 是否打印进度
        max_steps = self.config.max_steps  # 最大步数
        
        # 限制循环次数，最多跑 max_steps 轮
        for step_index in range(1, max_steps + 1):
            # 新增：打印进度 —— 告诉用户"正在调用大模型..."
            if prog:
                _progress_line(f"[dabench] Step {step_index}/{max_steps}: calling model ...")

            # 第一步：组装历史并向大模型发请求
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                # 第二步：解析大模型的意图
                model_step = parse_model_step(raw_response)

                # 新增：打印进度 —— 显示大模型决定调用什么工具以及参数预览
                if prog:
                    _progress_line(
                        f"[dabench] Step {step_index}/{max_steps}: action={model_step.action!r} "
                        f"input={_preview_json(model_step.action_input)}"
                    )

                # =====================================================
                # 新增核心逻辑：数据预览门控检查
                # =====================================================
                # 如果开启了"强制先预览数据"的配置，
                # 并且大模型当前想要执行的工具是"需要先审题才能用"的工具，
                # 并且历史记录中还没有成功执行过任何数据预览工具，
                # 那么 —— 拦截！不让它执行，直接返回错误提示。
                if (
                    self.config.require_data_preview_before_compute
                    and model_step.action in _GATED_AFTER_PREVIEW_ACTIONS
                    and not _has_successful_data_preview(state)
                ):
                    observation = {
                        "ok": False,
                        "tool": model_step.action,
                        "error": _GATE_REMINDER,
                    }
                    step_record = StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                    state.steps.append(step_record)
                    # 新增：打印进度 —— 告诉用户这一步被门控拦截了
                    if prog:
                        _progress_line(
                            f"[dabench] Step {step_index}/{max_steps}: tool ok=False (preview gate)"
                        )
                    # continue 跳过本轮循环的剩余部分，直接进入下一轮
                    # 这样大模型会在下一轮收到错误提示，被迫先去"审题"
                    continue

                # 第三步：执行物理世界的动作（调用对应的 Python 函数或 SQL 查询）
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                
                # 第四步：收集执行结果作为"观察反馈"
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                
                
                # 记录这一步的档案
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)

                # 新增：打印进度 —— 显示工具执行结果和是否为终止型工具
                if prog:
                    term = " (terminal)" if tool_result.is_terminal else ""
                    _progress_line(
                        f"[dabench] Step {step_index}/{max_steps}: tool ok={tool_result.ok}{term}"
                    )
                
                # 第五步：如果大模型调用的是终止型工具（比如 answer 工具），直接交卷退出循环！
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    # 新增：打印进度 —— 告诉用户最终答案已提交
                    if prog:
                        _progress_line("[dabench] Submitted final answer.")
                    break
            except Exception as exc:
                # 容错机制：如果大模型乱输出（比如拼错参数名，写错 JSON 格式）
                # 此时代码不能崩溃！而是要把报错信息包装成 observation，
                # 在下一轮循环发给大模型看，逼它自己修 Bug。
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__", # 标记这是一个由解析错误导致的步骤
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )
                # 新增：打印进度 —— 显示解析/工具调用时发生的错误
                if prog:
                    _progress_line(f"[dabench] Step {step_index}/{max_steps}: parse/tool error: {exc}")

        # 如果循环全跑完了，它还没交卷（没调用 answer 工具），判定为超时失败
        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        # 新增：打印进度 —— 如果最终没有提交答案，打印失败原因
        if prog and state.answer is None:
            _progress_line(f"[dabench] Stopped: {state.failure_reason}")

        # 返回最终报告，这就是你后来看到的 summary.json 和 trace.json 里的内容
        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
