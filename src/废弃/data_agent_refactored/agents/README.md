# agents 模块说明文档

## 模块概述

`agents` 是 `data_agent_refactored` 的核心模块，实现了两种主流 LLM Agent 范式——**ReAct**（Reasoning + Acting）和 **Plan-and-Solve**（先规划后执行）。模块采用模板方法设计模式：抽象基类 `BaseAgent` 封装模型调用重试、工具校验与超时执行、数据预览门控、结果打包等共享逻辑；两个具体子类各自实现差异化的推理循环。辅助子模块提供模型适配、提示词构建、JSON 容错解析、上下文窗口管理等基础能力。

---

## 架构总览

### 类继承关系

```
BaseAgent (abc.ABC)                  # 抽象基类：共享重试/工具执行/门控/打包
├── ReActAgent                       # Thought → Action → Observation 循环
└── PlanAndSolveAgent                # Phase 1 规划 → Phase 2 逐步执行
```

### 模块间依赖

```
agents/
├── base_agent.py ─────┬── model.py          (ModelAdapter, ModelMessage, ModelStep)
│                      ├── runtime.py        (AgentRuntimeState, StepRecord, AgentRunResult)
│                      ├── text_helpers.py   (progress)
│                      ├── timeout.py        (call_with_timeout)
│                      ├── data_preview_gate.py (GATED_ACTIONS, has_data_preview)
│                      ├── benchmark.schema  (PublicTask)          [外部]
│                      └── tools.registry    (ToolRegistry)        [外部]
│
├── react_agent.py ────┬── base_agent.py
│                      ├── config.py         (ReActAgentConfig)            [外部]
│                      ├── prompt.py         (REACT_SYSTEM_PROMPT, build_*)
│                      └── json_parser.py    (parse_model_step)
│
├── plan_solve_agent.py┬── base_agent.py
│                      ├── config.py         (PlanAndSolveAgentConfig)     [外部]
│                      ├── prompt.py         (PLAN_AND_SOLVE_SYSTEM_PROMPT, PLANNING/EXECUTION_INSTRUCTION)
│                      ├── json_parser.py    (parse_model_step, parse_plan)
│                      └── context_manager.py(build_history_messages)
│
├── context_manager.py ┬── model.py
│                      ├── prompt.py
│                      ├── runtime.py
│                      ├── data_preview_gate.py
│                      └── text_helpers.py
│
└── json_parser.py ────┬── model.py
                       └── json_repair       [第三方库]
```

### 外部依赖

| 依赖 | 用途 |
|---|---|
| `openai` | `OpenAIModelAdapter` 调用 OpenAI-compatible Chat Completions API |
| `json_repair` | `json_parser.py` 中 JSON 自动修复（Tier 2 容错） |
| `data_agent_refactored.benchmark.schema` | `PublicTask`、`AnswerTable` 数据结构 |
| `data_agent_refactored.tools.registry` | `ToolRegistry`、`ToolExecutionResult` 工具注册与执行 |
| `data_agent_refactored.config` | `ReActAgentConfig`、`PlanAndSolveAgentConfig` Agent 配置类 |
| `data_agent_refactored.exceptions` | `ModelCallError`、`ModelResponseParseError` 异常类 |

---

## 文件说明

| 文件 | 职责 | 关键导出 |
|---|---|---|
| `__init__.py` | 模块公开 API 汇总，统一 re-export | 见 `__all__` 列表 |
| `base_agent.py` | 抽象基类，封装所有 Agent 共享逻辑 | `BaseAgent` |
| `react_agent.py` | ReAct 范式 Agent 实现 | `ReActAgent`（`ReActAgentConfig` 已迁移至 `config.py`） |
| `plan_solve_agent.py` | Plan-and-Solve 范式 Agent 实现 | `PlanAndSolveAgent`（`PlanAndSolveAgentConfig` 已迁移至 `config.py`） |
| `model.py` | 模型适配层：消息/步骤数据结构、适配器协议与实现 | `ModelMessage`, `ModelStep`, `ModelAdapter`, `OpenAIModelAdapter`, `ScriptedModelAdapter` |
| `prompt.py` | 提示词常量与构建函数 | `REACT_SYSTEM_PROMPT`, `PLAN_AND_SOLVE_SYSTEM_PROMPT`, `build_system_prompt`, `build_task_prompt`, `build_observation_prompt` |
| `json_parser.py` | LLM 输出 JSON 多级容错解析 | `parse_model_step`, `parse_plan` |
| `runtime.py` | 运行时状态与结果记录 | `StepRecord`, `AgentRuntimeState`, `AgentRunResult` |
| `context_manager.py` | 历史上下文构建与窗口管理 | `build_history_messages` |
| `data_preview_gate.py` | 数据预览门控机制 | `DATA_PREVIEW_ACTIONS`, `GATED_ACTIONS`, `GATE_REMINDER`, `has_data_preview` |
| `text_helpers.py` | 文本工具函数集 | `progress`, `preview_json`, `estimate_tokens`, `truncate_observation` |
| `timeout.py` | 同步调用超时包装器 | `call_with_timeout` |

---

## 核心类 API 速查

### `BaseAgent`（`base_agent.py`）

抽象基类，子类必须实现 `run` 方法。

| 方法 | 签名 | 说明 |
|---|---|---|
| `__init__` | `(*, model: ModelAdapter, tools: ToolRegistry)` | 注入模型适配器和工具注册表 |
| `run` | `(task: PublicTask) -> AgentRunResult` | **抽象方法**，执行 Agent 循环 |
| `_call_model_with_retry` | `(messages, step_index, state, *, show_progress, timeout_seconds, max_retries, retry_backoff, tag) -> str \| None` | 带指数退避重试的模型调用，失败返回 `None` |
| `_validate_and_execute_tool` | `(task, model_step, raw, state, step_index, *, show_progress, tool_timeout_seconds, tag) -> ToolExecutionResult \| None` | 工具名校验 + 超时保护执行，失败返回 `None` |
| `_check_gate` | `(model_step, state, require_preview) -> bool` | 静态方法，检查数据预览门控是否应阻断当前动作 |
| `_finalize` | `(task, state, show_progress, tag) -> AgentRunResult` | 静态方法，将运行时状态打包为最终结果 |

### `ReActAgent`（`react_agent.py`）

| 方法 | 签名 | 说明 |
|---|---|---|
| `__init__` | `(*, model, tools, config: ReActAgentConfig \| None, system_prompt: str \| None)` | 可选注入自定义配置和系统提示词 |
| `run` | `(task: PublicTask) -> AgentRunResult` | 执行 ReAct 循环 |
| `_build_messages` | `(task, state) -> list[ModelMessage]` | 组装完整对话：system + task + 历史回放 |

### `PlanAndSolveAgent`（`plan_solve_agent.py`）

| 方法 | 签名 | 说明 |
|---|---|---|
| `__init__` | `(*, model, tools, config: PlanAndSolveAgentConfig \| None)` | 可选注入自定义配置 |
| `run` | `(task: PublicTask) -> AgentRunResult` | 执行 Plan-and-Solve 循环 |
| `_generate_plan` | `(task, tool_description, history_hint) -> list[str]` | Phase 1：调用 LLM 生成编号计划 |
| `_build_execution_messages` | `(task, tool_description, plan, plan_index, state) -> list[ModelMessage]` | Phase 2：组装含截断历史的执行消息 |
| `_handle_gate_block` | `(task, model_step, raw, plan, plan_index, state, step_index, consecutive_gate_blocks, show_progress) -> int` | 分层门控处理（提醒 → 改写计划步骤 → 强制注入 `list_context`） |
| `_handle_replan` | `(task, tool_description, state, step_index, replan_used, show_progress) -> tuple[list[str], int] \| None` | 工具失败后动态重规划 |

---

## 配置参数

### `ReActAgentConfig`（定义于 `config.py`，通过 `agents/__init__.py` re-export）

| 字段 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `max_steps` | `int` | `16` | 最大推理步数 |
| `progress` | `bool` | `False` | 是否打印实时进度 |
| `require_data_preview_before_compute` | `bool` | `True` | 是否启用数据预览门控 |
| `model_timeout_s` | `float` | `120.0` | 单次模型调用超时（秒） |
| `tool_timeout_s` | `float` | `180.0` | 单次工具执行超时（秒） |
| `max_model_retries` | `int` | `3` | 模型调用最大重试次数 |
| `model_retry_backoff` | `tuple[float, ...]` | `(2.0, 5.0, 10.0)` | 重试退避间隔序列（秒） |

### `PlanAndSolveAgentConfig`（定义于 `config.py`，通过 `agents/__init__.py` re-export）

| 字段 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `max_steps` | `int` | `20` | 最大执行步数 |
| `max_replans` | `int` | `2` | 最大重规划次数 |
| `progress` | `bool` | `False` | 是否打印实时进度 |
| `require_data_preview_before_compute` | `bool` | `True` | 是否启用数据预览门控 |
| `max_obs_chars` | `int` | `3000` | 单条观察内容截断字符数 |
| `max_context_tokens` | `int` | `24000` | 历史上下文窗口 token 预算 |
| `model_timeout_s` | `float` | `120.0` | 单次模型调用超时（秒） |
| `tool_timeout_s` | `float` | `180.0` | 单次工具执行超时（秒） |
| `max_gate_retries` | `int` | `2` | 门控阻断后升级前的容忍次数 |
| `max_model_retries` | `int` | `3` | 模型调用最大重试次数 |
| `model_retry_backoff` | `tuple[float, ...]` | `(2.0, 5.0, 10.0)` | 重试退避间隔序列（秒） |

---

## 运行流程

### ReAct Agent

```
1. 初始化 AgentRuntimeState
2. for step_index in 1..max_steps:
   a. 组装消息 (_build_messages): system + task + 全量历史回放
   b. 调用模型 (_call_model_with_retry): 指数退避重试
   c. 解析输出 (parse_model_step): JSON 多级容错
   d. 门控检查 (_check_gate): 未预览数据 → 阻断并注入提醒
   e. 工具执行 (_validate_and_execute_tool): 名称校验 + 超时保护
   f. 记录 StepRecord → 若工具为 terminal (answer) → 提交答案并退出
3. 打包结果 (_finalize)
```

### Plan-and-Solve Agent

```
Phase 1 — 规划:
  1. 调用 _generate_plan: LLM 生成编号步骤列表
  2. 失败则使用兜底计划 ["List context files", "Inspect data", "Solve and call answer"]

Phase 2 — 执行:
  1. for step_index in 1..max_steps:
     a. 组装消息 (_build_execution_messages): system + task + 当前计划步骤 + 截断历史
     b. 调用模型 (_call_model_with_retry)
     c. 解析输出 (parse_model_step)
     d. 门控检查 (_handle_gate_block): 三级升级策略
        - L1: 注入提醒消息
        - L2: 改写当前计划步骤为强制数据检查
        - L3: 强制注入 list_context 工具调用
     e. 工具执行 (_validate_and_execute_tool)
     f. 记录 StepRecord
     g. terminal → 提交答案并退出
     h. 成功 → plan_index 前进
     i. 失败且剩余重规划次数 > 0 → _handle_replan 动态重规划
  2. 打包结果 (_finalize)
```

---

## 关键机制说明

### 数据预览门控（`data_preview_gate.py`）

防止 LLM 在未查看数据的情况下直接执行代码或提交答案。

- **解锁工具**（`DATA_PREVIEW_ACTIONS`）：`read_csv`、`read_json`、`read_doc`、`inspect_sqlite_schema`
- **受控工具**（`GATED_ACTIONS`）：`execute_python`、`execute_context_sql`、`answer`
- **判定逻辑**：`has_data_preview(state)` 检查历史步骤中是否存在至少一个成功的数据预览动作
- **Plan-and-Solve 升级策略**：连续阻断次数递增时，从提醒 → 改写计划步骤 → 强制注入 `list_context`

### 模型调用重试（`base_agent.py`）

`_call_model_with_retry` 实现指数退避重试：

- 捕获 `TimeoutError` 和 `RuntimeError`
- 退避间隔从 `model_retry_backoff` 元组中按 `min(attempt, len-1)` 取值
- 所有重试耗尽后记录错误 `StepRecord`（action=`__error__`）

### 上下文窗口管理（`context_manager.py`）

`build_history_messages` 实现两层截断策略：

- **L1（单条截断）**：每条观察内容硬截断至 `max_obs_chars` 字符；错误/重规划失败步骤的 `raw_response` 替换为标准化占位 JSON
- **L2（窗口淘汰）**：
  - **钉住**（pin）：成功的数据预览步骤始终保留
  - **淘汰**（evict）：其余步骤按 FIFO 从最早开始淘汰，直到总 token 量不超过 `max_context_tokens` 预算
  - 淘汰发生时，在消息中插入摘要提示

### JSON 多级容错解析（`json_parser.py`）

LLM 输出经 `strip_json_fence` 去除 Markdown 围栏后，进入三级解析：

| 级别 | 方法 | 说明 |
|---|---|---|
| Tier 1a | `try_strict_json(text)` | 标准 `json.JSONDecoder().raw_decode` |
| Tier 1b | `try_strict_json(fix_trailing_bracket(text))` | 修复 `}]}` → `}}` 后再尝试 |
| Tier 2 | `json_repair.loads(text)` | 第三方库自动修复 |

解析结果通过 `parse_model_step` 转为 `ModelStep`（需包含 `thought`、`action`、`action_input`），或通过 `parse_plan` 转为计划步骤列表。

### 超时保护（`timeout.py`）

`call_with_timeout(fn, args, timeout_seconds)` 使用守护线程执行目标函数：

- 主线程通过 `thread.join(timeout=...)` 等待
- 超时则立即抛出 `TimeoutError`，后台线程作为 daemon 在进程退出时自动回收
- 目标函数内部异常会被捕获并在主线程重新抛出

---

## 快速上手示例

### 使用 ReActAgent

> **注**：`ReActAgentConfig` / `PlanAndSolveAgentConfig` 实际定义于 `data_agent_refactored.config`，但通过 `agents/__init__.py` re-export，以下导入路径仍然有效。

```python
from data_agent_refactored.agents import (
    ReActAgent, ReActAgentConfig, OpenAIModelAdapter,
)
from data_agent_refactored.tools.registry import ToolRegistry
from data_agent_refactored.benchmark.schema import PublicTask

# 1. 创建模型适配器
model = OpenAIModelAdapter(
    model="gpt-4o",
    api_base="https://api.openai.com/v1",
    api_key="sk-...",
    temperature=0.0,
)

# 2. 创建工具注册表（需根据项目实际注册工具）
tools = ToolRegistry()

# 3. 配置并创建 Agent
config = ReActAgentConfig(max_steps=16, progress=True)
agent = ReActAgent(model=model, tools=tools, config=config)

# 4. 执行任务
task = PublicTask(task_id="example-001", question="...", context_dir="path/to/context")
result = agent.run(task)

print(f"成功: {result.succeeded}")
print(f"答案: {result.answer}")
print(f"步骤数: {len(result.steps)}")
```

### 使用 PlanAndSolveAgent

```python
from data_agent_refactored.agents import (
    PlanAndSolveAgent, PlanAndSolveAgentConfig, OpenAIModelAdapter,
)
from data_agent_refactored.tools.registry import ToolRegistry
from data_agent_refactored.benchmark.schema import PublicTask

model = OpenAIModelAdapter(
    model="gpt-4o",
    api_base="https://api.openai.com/v1",
    api_key="sk-...",
    temperature=0.0,
)

tools = ToolRegistry()

config = PlanAndSolveAgentConfig(
    max_steps=20,
    max_replans=2,
    progress=True,
    max_context_tokens=24000,
)
agent = PlanAndSolveAgent(model=model, tools=tools, config=config)

task = PublicTask(task_id="example-002", question="...", context_dir="path/to/context")
result = agent.run(task)

print(f"成功: {result.succeeded}")
if result.failure_reason:
    print(f"失败原因: {result.failure_reason}")
```

### 使用 ScriptedModelAdapter 进行测试

```python
from data_agent_refactored.agents import ScriptedModelAdapter, ReActAgent
from data_agent_refactored.tools.registry import ToolRegistry

# 预设模型返回序列，用于确定性测试
scripted = ScriptedModelAdapter(responses=[
    '```json\n{"thought":"Inspect files","action":"list_context","action_input":{"max_depth":4}}\n```',
    '```json\n{"thought":"Read data","action":"read_csv","action_input":{"file_path":"data.csv"}}\n```',
    '```json\n{"thought":"Submit answer","action":"answer","action_input":{"columns":["col"],"rows":[["val"]]}}\n```',
])

agent = ReActAgent(model=scripted, tools=ToolRegistry())
```
