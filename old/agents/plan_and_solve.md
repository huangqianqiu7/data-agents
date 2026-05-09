# Plan-and-Solve Agent 说明文档

## 一、概述

`plan_and_solve.py` 实现了一个 **Plan-and-Solve（先规划再执行）** 架构的智能体（Agent）。

它与经典的 **ReAct** 架构的核心区别在于：

| 维度 | ReAct | Plan-and-Solve |
|------|-------|----------------|
| 决策方式 | 每步即兴决定下一步做什么 | **先生成全局计划**，再按计划逐步执行 |
| 上下文 | 只有历史 Observation | 完整计划 + 当前进度 + 历史 Observation |
| 容错机制 | 靠大模型自行纠错 | 工具失败时可触发**动态重规划** |
| 适用场景 | 简单、少步骤任务 | 复杂、多步骤的数据分析任务 |

---

## 二、整体架构

```
┌─────────────────────────────────────────────────┐
│               PlanAndSolveAgent.run()            │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Phase 1: 规划 (_generate_plan)          │    │
│  │  ─ 输入：任务描述 + 可用工具列表          │    │
│  │  ─ 输出：["步骤1", "步骤2", ..., "提交"]  │    │
│  └──────────────┬───────────────────────────┘    │
│                 ▼                                 │
│  ┌──────────────────────────────────────────┐    │
│  │  Phase 2: 按计划执行 (主循环)             │    │
│  │                                           │    │
│  │  for step_index in range(1, max_steps+1): │    │
│  │    ┌─────────────────────────────┐        │    │
│  │    │ 构建消息 (计划+进度+历史)    │        │    │
│  │    └──────────┬──────────────────┘        │    │
│  │               ▼                           │    │
│  │    ┌─────────────────────────────┐        │    │
│  │    │ 调用大模型 → 解析 JSON 输出  │        │    │
│  │    └──────────┬──────────────────┘        │    │
│  │               ▼                           │    │
│  │    ┌─────────────────────────────┐        │    │
│  │    │ 数据预览门控检查             │        │    │
│  │    └──────────┬──────────────────┘        │    │
│  │               ▼                           │    │
│  │    ┌─────────────────────────────┐        │    │
│  │    │ 执行工具 → 收集 Observation  │        │    │
│  │    └──────────┬──────────────────┘        │    │
│  │               ▼                           │    │
│  │    成功 → 推进计划步骤                     │    │
│  │    失败 → 触发重规划（如果配额允许）        │    │
│  │    终止型工具(answer) → 交卷退出           │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  返回 AgentRunResult                             │
└─────────────────────────────────────────────────┘
```

---

## 三、文件结构速览

文件按功能自上而下划分为 **6 个区域**：

### 1. 提示词定义（第 16–65 行）

| 常量 | 用途 |
|------|------|
| `PLAN_AND_SOLVE_SYSTEM_PROMPT` | 系统级提示词，定义 Agent 的角色和输出格式规则 |
| `_PLANNING_INSTRUCTION` | Phase 1 指令，要求大模型输出 `{"plan": [...]}` 格式的计划 |
| `_EXECUTION_INSTRUCTION` | Phase 2 指令模板，每轮对话注入完整计划和当前步骤进度 |

### 2. 数据预览门控（第 68–81 行）

强制大模型在执行代码或提交答案之前，必须先"看过数据"。

- **预览类工具**：`read_csv`、`read_json`、`read_doc`、`inspect_sqlite_schema`
- **受限工具**：`execute_python`、`execute_context_sql`、`answer`
- 如果大模型试图跳过预览直接用受限工具，会被拦截并收到 `_GATE_REMINDER` 提示

### 3. 辅助函数（第 84–98 行）

| 函数 | 作用 |
|------|------|
| `_progress(msg)` | 终端实时打印进度信息 |
| `_preview_json(obj)` | 将对象转为截断的 JSON 预览（日志用） |

### 4. JSON 解析工具（第 101–175 行）

大模型输出的 JSON 经常有格式瑕疵，这里采用 **三级降级策略** 来解析：

```
原始文本
  │
  ▼
_strip_json_fence()      ← 去掉 Markdown ```json ``` 标记
  │
  ▼
_load_json_object()      ← 三级降级解析
  ├── 1. _try_strict_json()         严格解析
  ├── 2. _fix_trailing_bracket()    修复常见括号错误后再严格解析
  └── 3. json_repair.loads()        自动修复各种 JSON 格式问题
```

解析出的 JSON 会被进一步转化为：
- `_parse_step(raw)` → `ModelStep` 对象（执行阶段：thought / action / action_input）
- `_parse_plan(raw)` → `list[str]`（规划阶段：步骤描述列表）

### 5. 配置类 `PlanAndSolveAgentConfig`（第 178–186 行）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_steps` | `int` | `20` | 执行阶段最大工具调用次数 |
| `max_replans` | `int` | `2` | 最多允许动态重规划几次 |
| `progress` | `bool` | `False` | 是否在终端打印运行进度 |
| `require_data_preview_before_compute` | `bool` | `True` | 是否强制先预览数据 |

### 6. 核心类 `PlanAndSolveAgent`（第 192–425 行）

| 方法 | 职责 |
|------|------|
| `__init__()` | 初始化：注入模型、工具箱、配置 |
| `_generate_plan()` | **Phase 1**：调用大模型生成分步计划 |
| `_build_exec_messages()` | **Phase 2**：组装执行阶段的对话消息（系统设定 + 任务 + 计划进度 + 历史观测） |
| `run()` | **主入口**：先规划、再执行、处理门控/容错/重规划/交卷 |

---

## 四、核心流程详解

### Phase 1：规划（`_generate_plan`）

```python
# 输入：任务描述 + 工具列表 + (可选)历史观测摘要
# 输出：["List context files", "Read CSV data", ..., "Call answer tool"]
plan = agent._generate_plan(task, tool_desc)
```

- 向大模型发送 **1 次** 请求，要求返回 `{"plan": [...]}`
- 如果规划失败（网络错误、格式错误等），自动降级为 3 步兜底计划

### Phase 2：按计划执行（`run` 主循环）

每一轮循环中：

1. **构建消息**：将完整计划、当前步骤编号、历史 observation 打包成对话
2. **调用大模型**：获取 `{"thought": "...", "action": "...", "action_input": {...}}`
3. **门控检查**：如果要执行受限工具但还没预览过数据 → 拦截
4. **执行工具**：调用 `tools.execute()` 获取结果
5. **推进/重规划**：
   - 工具成功 → `plan_idx += 1`，推进到下一个计划步骤
   - 工具失败且有重规划配额 → 重新生成计划，`plan_idx = 0`
   - 终止型工具（`answer`）→ 提交答案，退出循环

### 容错机制

- **JSON 解析错误**：记录为 `action="__error__"` 的步骤，下轮让大模型看到错误信息自行修正
- **计划步骤耗尽**：自动追加一步"提交答案"兜底
- **超时未交卷**：`failure_reason = "Agent did not submit an answer within max_steps."`

---

## 五、使用示例

```python
from data_agent_baseline.agents.plan_and_solve import PlanAndSolveAgent, PlanAndSolveAgentConfig

agent = PlanAndSolveAgent(
    model=my_model_adapter,
    tools=my_tool_registry,
    config=PlanAndSolveAgentConfig(
        max_steps=20,
        max_replans=2,
        progress=True,                          # 终端打印运行进度
        require_data_preview_before_compute=True,
    ),
)

result = agent.run(task)
# result.answer       → 最终答案表格（AnswerTable 或 None）
# result.steps        → 所有步骤的完整记录（list[StepRecord]）
# result.succeeded    → 是否成功
# result.failure_reason → 失败原因（成功时为 None）
```

---

## 六、依赖关系

```
plan_and_solve.py
  ├── model.py          → ModelAdapter, ModelMessage, ModelStep
  ├── prompt.py         → build_observation_prompt, build_task_prompt
  ├── runtime.py        → AgentRunResult, AgentRuntimeState, StepRecord
  ├── benchmark/schema  → PublicTask
  ├── tools/registry    → ToolRegistry
  └── json_repair (外部库) → 自动修复格式有瑕疵的 JSON
```
