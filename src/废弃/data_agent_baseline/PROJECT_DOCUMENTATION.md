# KDD Cup 2026 Data Agent Baseline — 项目架构说明文档

## 一、项目概述

本项目是 **KDD Cup 2026 Data Agents** 竞赛的基线（baseline）解决方案。其核心目标是：构建一个由大语言模型（LLM）驱动的**数据分析智能体（Data Agent）**，能够自动阅读任务描述、检查数据文件、编写并执行代码、最终提交结构化的表格答案。

整个系统由 4 个核心模块组成：

| 模块 | 路径 | 职责 |
|------|------|------|
| **agents_v2** | `agents_v2/` | 智能体核心逻辑（推理循环、模型调用、上下文管理） |
| **benchmark** | `benchmark/` | 数据集与任务的数据结构定义 |
| **tools** | `tools/` | 工具注册中心 + 8 个可调用工具的实现 |
| **run** | `run/` | 运行器：单任务执行、多任务并发、超时控制、结果持久化 |

---

## 二、整体架构与数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                         run/runner.py                           │
│  （运行入口：加载配置 → 遍历任务 → 启动 Agent → 保存结果）         │
└───────────────┬─────────────────────────────────────────────────┘
                │ 对每个任务创建 Agent 并调用 agent.run(task)
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      agents_v2/                                  │
│                                                                  │
│   ┌──────────────┐    ┌────────────────────┐                     │
│   │  ReActAgent   │    │ PlanAndSolveAgent  │                     │
│   │ (走一步看一步) │    │ (先规划再执行)      │                     │
│   └──────┬───────┘    └────────┬───────────┘                     │
│          │                     │                                  │
│          └─────────┬───────────┘                                  │
│                    │ 继承                                         │
│            ┌───────▼────────┐                                     │
│            │   BaseAgent    │ 共享：模型重试、工具校验、门控、收尾   │
│            └───────┬────────┘                                     │
│                    │ 调用                                         │
│    ┌───────────────┼──────────────────┐                           │
│    ▼               ▼                  ▼                           │
│  model.py     json_parser.py    context_manager.py               │
│  (LLM 适配)  (JSON 解析)        (历史上下文截断)                  │
│                                                                  │
│  prompt.py    gate.py           text_helpers.py   timeout.py     │
│  (提示词)     (数据预览门控)     (文本工具)         (超时控制)      │
└───────────────┬──────────────────────────────────────────────────┘
                │ Agent 循环中调用 tools.execute(task, action, input)
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        tools/                                    │
│                                                                  │
│   registry.py  ← 工具注册中心（ToolSpec + ToolHandler 映射）      │
│       │                                                          │
│       ├── list_context      （列出文件目录树）                     │
│       ├── read_csv          （预览 CSV 文件）                     │
│       ├── read_json         （预览 JSON 文件）                    │
│       ├── read_doc          （读取文本文档）                       │
│       ├── inspect_sqlite_schema （查看 SQLite 表结构）            │
│       ├── execute_context_sql   （执行只读 SQL）                  │
│       ├── execute_python        （执行 Python 代码）              │
│       └── answer ⭐             （提交最终答案，终止循环）          │
│                                                                  │
│   filesystem.py  ← 文件系统操作（路径安全校验 + 文件预览）          │
│   python_exec.py ← Python 代码沙箱执行（独立子进程 + 超时）        │
│   sqlite.py      ← SQLite 只读查询                               │
└───────────────┬──────────────────────────────────────────────────┘
                │ 工具操作的数据来源
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      benchmark/                                  │
│                                                                  │
│   schema.py   ← 核心数据结构：TaskRecord, PublicTask, AnswerTable │
│   dataset.py  ← 数据集加载器：DABenchPublicDataset                │
│                  遍历 task_xxx/ 目录，解析 task.json + context/    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、模块详细说明

### 3.1 `agents_v2/` — 智能体核心

这是从原始 819 行单体文件重构而来的**模块化 Agent 架构**，包含 12 个文件。

#### 3.1.1 类继承关系

```
BaseAgent (抽象基类)
  ├── ReActAgent          — 走一步看一步（Thought → Action → Observation 循环）
  └── PlanAndSolveAgent   — 先规划全局计划，再按计划逐步执行
```

#### 3.1.2 核心文件说明

| 文件 | 职责 | 关键类/函数 |
|------|------|------------|
| `base_agent.py` | 智能体抽象基类，封装共享逻辑 | `BaseAgent` — 提供 `_call_model_with_retry()`（模型调用+指数退避重试）、`_validate_and_execute_tool()`（工具名预校验+超时执行）、`_check_gate()`（门控检查）、`_finalize()`（收尾封装） |
| `react_agent.py` | ReAct 架构智能体 | `ReActAgent` + `ReActAgentConfig` — 每步即兴决策，无全局计划。主循环：调用模型 → 解析 JSON → 门控检查 → 执行工具 → 记录结果 → 判断终止 |
| `plan_solve_agent.py` | Plan-and-Solve 架构智能体 | `PlanAndSolveAgent` + `PlanAndSolveAgentConfig` — Phase 1 生成分步计划，Phase 2 按计划执行。支持动态重规划（最多 `max_replans` 次）、三级门控升级（提醒→改写计划→强制注入 list_context） |
| `model.py` | 大模型适配器层 | `ModelMessage`（消息结构体）、`ModelStep`（单步解析结果）、`ModelAdapter`（协议接口）、`OpenAIModelAdapter`（兼容 OpenAI/阿里云/DeepSeek 等）、`ScriptedModelAdapter`（测试用脚本适配器） |
| `prompt.py` | 统一提示词模块 | `REACT_SYSTEM_PROMPT` / `PLAN_AND_SOLVE_SYSTEM_PROMPT`（系统提示词）、`RESPONSE_EXAMPLES`（Few-shot 样例）、`PLANNING_INSTRUCTION` / `EXECUTION_INSTRUCTION`（规划/执行指令模板）、`build_system_prompt()` / `build_task_prompt()` / `build_observation_prompt()`（构建函数） |
| `json_parser.py` | JSON 多层降级解析 | `parse_model_step(raw)` → `ModelStep`、`parse_plan(raw)` → `list[str]`。解析策略：严格解析 → 修复括号错误 → `json_repair` 自动修复 |
| `context_manager.py` | 历史上下文管理（双层截断） | `build_history_messages()` — **L1**：单条 observation 硬截断（`max_obs_chars`）；**L2**：钉住关键步骤（数据预览）+ 滑动窗口淘汰非关键步骤（FIFO） |
| `gate.py` | 数据预览门控机制 | 强制 Agent 在执行代码或提交答案前必须先"看过数据"。`DATA_PREVIEW_ACTIONS`（预览工具集）、`GATED_ACTIONS`（受限工具集）、`has_data_preview()` |
| `runtime.py` | 运行时状态与结果记录 | `StepRecord`（单步记录，不可变）、`AgentRuntimeState`（运行时状态，可变）、`AgentRunResult`（最终报告，不可变） |
| `text_helpers.py` | 文本处理工具函数 | `progress()`（进度打印）、`preview_json()`（截断预览）、`estimate_tokens()`（粗估 token 数）、`truncate_observation()`（截断大 observation） |
| `timeout.py` | 超时调用包装器 | `call_with_timeout(fn, args, timeout_s)` — 使用独立 daemon 线程实现超时控制，避免线程池死锁 |

#### 3.1.3 ReAct Agent 执行流程

```
1. 初始化 AgentRuntimeState（空状态）
2. for step_index in range(1, max_steps + 1):
   a. 组装消息：system prompt + task prompt + 历史步骤回放
   b. 调用模型（带指数退避重试，最多 3 次）
   c. 解析模型 JSON 输出为 ModelStep（thought / action / action_input）
   d. 门控检查：如果 action 是 execute_python / answer 等，但尚未预览过数据 → 拦截
   e. 工具名预校验：检查 action 是否存在于已注册工具列表
   f. 执行工具（带超时保护）
   g. 记录 StepRecord 到 state
   h. 若工具返回 is_terminal=True → 提取 answer，break 退出循环
3. 封装 AgentRunResult 返回
```

#### 3.1.4 Plan-and-Solve Agent 执行流程

```
Phase 1 — 规划：
  1. 发送任务描述 + 工具列表给大模型
  2. 大模型返回分步计划（JSON 数组），如失败则使用兜底计划

Phase 2 — 执行：
  1. for step_index in range(1, max_steps + 1):
     a. 组装消息：system + task + 当前计划进度 + 带双层截断的历史
     b. 调用模型（带重试）
     c. 解析输出为 ModelStep
     d. 三级门控升级：
        - 第 1 次拦截：返回提醒消息
        - 第 N 次拦截（≥ max_gate_retries）：改写当前计划步骤为强制数据预览
        - 第 N+2 次拦截：强制注入 list_context 工具调用，绕过模型
     e. 工具校验 + 执行
     f. 工具成功 → plan_idx++（推进计划）
     g. 工具失败 → 尝试重规划（最多 max_replans 次）
     h. 终止型工具 → 提交答案，break
  2. 封装结果返回
```

#### 3.1.5 已修复的审计隐患

| 编号 | 隐患描述 | 修复方式 |
|------|---------|---------|
| 2 | 重规划失败记录的 step_index 与同轮工具失败记录冲突 | 重规划失败用 `step_index=-1` |
| 3 | 早期关键步骤（数据预览）被 FIFO 丢弃 | `context_manager.py` 钉住数据预览步骤 |
| 4 | 模型调用失败无重试 | `base_agent.py` 指数退避重试 |
| 5 | base_tokens 溢出无保护 | `context_manager.py` 溢出时留最小余量 |
| 7 | 幻觉工具名浪费步数 | `base_agent.py` 工具名预校验 |

---

### 3.2 `benchmark/` — 数据集与任务结构

本模块定义了 DABench 公开数据集的数据结构和加载逻辑。

#### 3.2.1 数据目录结构（磁盘上的任务布局）

```
dataset_root/
├── task_001/
│   ├── task.json          ← {"task_id": "task_001", "difficulty": "easy", "question": "..."}
│   └── context/           ← 任务附带的数据文件（CSV、JSON、SQLite 等）
│       ├── data.csv
│       └── info.json
├── task_002/
│   ├── task.json
│   └── context/
│       └── database.sqlite
└── ...
```

#### 3.2.2 核心数据结构 (`schema.py`)

| 类 | 说明 | 关键字段 |
|----|------|---------|
| `TaskRecord` | 任务元信息（来自 `task.json`） | `task_id`, `difficulty`, `question` |
| `TaskAssets` | 任务文件路径 | `task_dir`（任务根目录）, `context_dir`（数据目录） |
| `PublicTask` | 完整任务对象（Record + Assets） | 提供 `task_id`, `question`, `context_dir` 等便捷属性 |
| `AnswerTable` | Agent 提交的答案表格 | `columns: list[str]`, `rows: list[list[Any]]` |

#### 3.2.3 数据集加载器 (`dataset.py`)

`DABenchPublicDataset` 类负责扫描数据集根目录，提供以下功能：

- **`list_task_ids()`** — 返回所有任务 ID 列表（按数字排序）
- **`get_task(task_id)`** — 加载单个任务（解析 `task.json` + 校验 `context/` 目录存在）
- **`iter_tasks(task_ids=..., difficulty=...)`** — 按条件筛选并迭代任务
- **`task_counts()`** — 按难度统计任务数量

---

### 3.3 `tools/` — 工具注册中心

本模块实现了 Agent 可调用的全部工具，采用**注册表模式**统一管理。

#### 3.3.1 核心设计（`registry.py`）

```
ToolSpec（菜单）    →  告诉大模型"有什么工具、怎么用"
ToolHandler（厨师） →  真正执行工具逻辑的 Python 函数
ToolRegistry（服务员）→ 根据大模型的选择，派发给对应的 Handler 执行
```

**`ToolRegistry` 类**：
- `specs: dict[str, ToolSpec]` — 工具说明书字典
- `handlers: dict[str, ToolHandler]` — 工具执行函数字典
- `describe_for_prompt()` — 将所有工具格式化为文本，嵌入 System Prompt
- `execute(task, action, action_input)` — 根据工具名调用对应的 Handler

**`create_default_tool_registry()`** — 工厂函数，一键组装包含 8 个工具的默认工具箱。

#### 3.3.2 工具清单

| 工具名 | 类型 | 功能 | 关键参数 | 实现文件 |
|--------|------|------|---------|---------|
| `list_context` | 侦查型 | 列出任务 context 目录的文件树 | `max_depth`(默认 4) | `filesystem.py` |
| `read_csv` | 预览型 | 预览 CSV 文件前 N 行 | `path`, `max_rows`(默认 20) | `filesystem.py` |
| `read_json` | 预览型 | 预览 JSON 文件内容 | `path`, `max_chars`(默认 4000) | `filesystem.py` |
| `read_doc` | 预览型 | 读取文本文档内容 | `path`, `max_chars`(默认 4000) | `filesystem.py` |
| `inspect_sqlite_schema` | 预览型 | 查看 SQLite 数据库表结构 | `path` | `sqlite.py` |
| `execute_context_sql` | 计算型 | 在 SQLite 上执行只读 SQL | `path`, `sql`, `limit`(默认 200) | `sqlite.py` |
| `execute_python` | 计算型 | 执行任意 Python 代码 | `code` | `python_exec.py` |
| `answer` ⭐ | **终止型** | 提交最终答案表格 | `columns`, `rows` | `registry.py` |

#### 3.3.3 安全机制

- **路径穿越防护** (`filesystem.py`): `resolve_context_path()` 校验所有文件路径不会逃逸出 `context/` 目录
- **SQL 只读限制** (`sqlite.py`): 仅允许 `SELECT` / `WITH` / `PRAGMA` 开头的语句；使用 `?mode=ro` 只读连接
- **Python 执行沙箱** (`python_exec.py`): 在独立子进程中执行代码，30 秒超时强制终止；stdout/stderr 流重定向捕获
- **答案校验** (`registry.py`): `_answer()` 严格校验 columns 和 rows 的格式一致性

---

### 3.4 `run/` — 运行器

本模块负责将 Agent、工具、数据集串联起来，提供单任务和批量运行能力。

#### 3.4.1 文件说明

| 文件 | 说明 |
|------|------|
| `runner.py` | **当前版本**的运行器，修复了 `old_runner.py` 中的死锁问题 |
| `old_runner.py` | 原版运行器（保留作为对比参考） |

#### 3.4.2 `runner.py` 核心函数

| 函数 | 职责 |
|------|------|
| `run_single_task()` | **单任务入口**：计时 → 执行 → 写出结果文件（`trace.json` + `prediction.csv`） |
| `run_benchmark()` | **批量运行入口**：遍历数据集 → 单线程或多线程并发 → 写出 `summary.json` |
| `_run_single_task_core()` | 核心逻辑：加载任务 → 创建 ReActAgent → 调用 `agent.run(task)` |
| `_run_single_task_with_timeout()` | 超时控制：在子进程中运行任务，主进程用 `queue.get(timeout=...)` 等待 |
| `_run_single_task_in_subprocess()` | 子进程执行体：隔离崩溃/内存泄漏 |
| `build_model_adapter()` | 根据 `AppConfig` 创建 `OpenAIModelAdapter` |

#### 3.4.3 执行模式

```
单任务模式：
  run_single_task(task_id, config, run_output_dir)
    → 若未注入 model/tools → 启动子进程 + 超时控制
    → 若已注入 model/tools → 主进程直接执行（方便调试）

批量模式：
  run_benchmark(config, limit=N)
    → 单线程(max_workers=1): 顺序执行，复用同一 model/tools 实例
    → 多线程(max_workers>1): ThreadPoolExecutor 分发，每个任务独立子进程
```

#### 3.4.4 输出目录结构

```
output_root/
└── 20260414T134153Z/          ← run_id（UTC 时间戳）
    ├── summary.json            ← 全局汇总报告
    ├── task_001/
    │   ├── trace.json          ← Agent 完整执行录像（每步的 thought/action/observation）
    │   └── prediction.csv      ← 最终答案（如果成功提交）
    ├── task_002/
    │   ├── trace.json
    │   └── prediction.csv
    └── ...
```

#### 3.4.5 `runner.py` vs `old_runner.py` 的改进

| 改进点 | old_runner.py | runner.py |
|--------|---------------|-----------|
| 超时实现 | `process.join(timeout)` 后检查 `queue.empty()` — 可能死锁（大数据塞满管道） | 先 `queue.get(timeout=...)` 再 `_reap_process()` — 避免管道阻塞死锁 |
| 进程清理 | 简单的 `terminate + join` | 三级清理：`join → terminate → kill`（`_reap_process()`） |
| 进度打印 | 无 | 支持 `progress=True` 参数打印加载日志 |
| 文件编码 | 未指定编码 | `_write_json` / `_write_csv` 强制 UTF-8 编码 |

---

## 四、关键配置参数速查

### ReActAgentConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_steps` | 16 | 最大工具调用次数 |
| `progress` | False | 是否打印运行进度 |
| `require_data_preview_before_compute` | True | 强制先预览数据 |
| `model_timeout_s` | 120.0 | 单次模型调用超时（秒） |
| `tool_timeout_s` | 180.0 | 单次工具执行超时（秒） |
| `max_model_retries` | 3 | 模型 API 最大重试次数 |
| `model_retry_backoff` | (2.0, 5.0, 10.0) | 重试退避秒数 |

### PlanAndSolveAgentConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_steps` | 20 | 执行阶段最大工具调用次数 |
| `max_replans` | 2 | 最多重新规划次数 |
| `max_obs_chars` | 3000 | 单条 observation 最大字符数 |
| `max_context_tokens` | 24000 | 历史消息区 token 预算上限 |
| `max_gate_retries` | 2 | 门控连续拦截次数上限 |
| 其余参数 | 同 ReAct | — |

---

## 五、典型使用示例

```python
from data_agent_baseline.agents_v2 import (
    PlanAndSolveAgent, PlanAndSolveAgentConfig,
    ReActAgent, ReActAgentConfig,
    OpenAIModelAdapter,
)
from data_agent_baseline.tools import create_default_tool_registry
from data_agent_baseline.benchmark import DABenchPublicDataset

# 1. 初始化模型
model = OpenAIModelAdapter(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-xxx",
    temperature=0.0,
)

# 2. 创建工具箱
tools = create_default_tool_registry()

# 3. 创建智能体
agent = PlanAndSolveAgent(
    model=model,
    tools=tools,
    config=PlanAndSolveAgentConfig(max_steps=20, progress=True),
)

# 4. 加载任务并执行
dataset = DABenchPublicDataset(root_dir=Path("path/to/dataset"))
task = dataset.get_task("task_001")
result = agent.run(task)

# 5. 查看结果
print(f"成功: {result.succeeded}")
print(f"答案: {result.answer}")
print(f"步骤数: {len(result.steps)}")
```

---

## 六、模块依赖关系图

```
run/runner.py
  ├── benchmark/dataset.py    (加载数据集)
  ├── benchmark/schema.py     (PublicTask 等数据结构)
  ├── tools/registry.py       (创建工具箱)
  └── agents_v2/              (创建并运行 Agent)
        ├── base_agent.py
        │     ├── model.py          (调用 LLM)
        │     ├── runtime.py        (状态记录)
        │     ├── text_helpers.py   (文本工具)
        │     ├── timeout.py        (超时控制)
        │     └── gate.py           (门控检查)
        ├── react_agent.py          (继承 BaseAgent)
        │     ├── json_parser.py    (解析 LLM 输出)
        │     └── prompt.py         (构建提示词)
        └── plan_solve_agent.py     (继承 BaseAgent)
              ├── json_parser.py
              ├── prompt.py
              └── context_manager.py (历史截断)
```
