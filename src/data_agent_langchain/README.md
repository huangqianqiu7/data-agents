# `data_agent_langchain` —— LangGraph 后端实现

`data_agent_langchain` 是 DABench 数据智能体的 **LangGraph 后端**，也是当前
仓库的默认 backend。整个包**自包含、零 sibling 依赖**：源码内不允许出现来自
`data_agent_common` / `data_agent_refactored` / `data_agent_baseline` 的
import（这三个 sibling 包已归档到 `src/废弃/`，仅作 parity / 历史回归参考）。

> 本文档定位是 **包内速查 + 开发者上手手册**，不重复 Phase 历史叙事。
> 阶段进度、设计动机、跨包决策以
> [`../LangChain/MIGRATION_STATUS.md`](../LangChain/MIGRATION_STATUS.md)
> 与 [`../LangChain/LANGCHAIN_MIGRATION_PLAN.md`](../LangChain/LANGCHAIN_MIGRATION_PLAN.md)
> 为 single source of truth。

---

## 目录

- [整体架构](#整体架构)
- [包结构](#包结构)
- [状态机拓扑](#状态机拓扑)
- [`gateway_caps` 驱动 `bind_tools`](#gateway_caps-驱动-bind_tools)
- [快速上手](#快速上手)
- [配置（`config.py`）](#配置configpy)
- [常见 extension point](#常见-extension-point)
- [运行与调试](#运行与调试)
- [运行测试](#运行测试)
- [Phase 状态](#phase-状态)
- [使用约定 / 不变量](#使用约定--不变量)

---

## 整体架构

包整体由三层组合而成：

| 层 | 来源 | 在本包中的角色 |
| --- | --- | --- |
| LangChain | `langchain-core` / `langchain-openai` | 提供 `BaseChatModel`（`ChatOpenAI`）、`BaseTool`、`ChatPromptTemplate` 等基础原语 |
| LangGraph | `langgraph` 0.4+ | 承载 ReAct 与 Plan-and-Solve 两套 `StateGraph`，所有节点共享同一份 `RunState` |
| 项目自有运行时 | 本包 `runtime/` `agents/runtime` `benchmark/schema` | `StepRecord` / `AgentRunResult` / `AgentRuntimeState` / `AnswerTable` / `PublicTask` 等 DABench 兼容对象，全部在本包内定义 |

子进程隔离、批量并发、trace / metrics 落盘逻辑集中在 `run/runner.py`，与图本身解耦。

---

## 包结构

| 子包 / 模块 | 文件数 | 一句话职责 |
| --- | --- | --- |
| `__init__.py` | 1 | 暴露 `__version__`，不 eager import 子模块 |
| `cli.py` | 1 | `dabench-lc` typer 入口：`run-task` / `run-benchmark` / `gateway-smoke` 三个子命令 |
| `config.py` | 1 | 6 个 frozen dataclass + `AppConfig` + YAML 加载 + 评测约束校验 |
| `constants.py` | 1 | 跨模块字符串常量（如 `FALLBACK_STEP_PROMPT`） |
| `exceptions.py` | 1 | 异常层次（根类 `DataAgentError`） |
| `agents/` | 19 个 `.py` | LangGraph 节点实现：gate / model / parse_action / tool / advance / finalize / planner / replanner，加 ReAct / Plan-and-Solve 顶层图与 execution_subgraph |
| `tools/` | 16 个 `.py` | 8 个 `BaseTool` 子类 + factory + descriptions + tool_runtime + timeout |
| `runtime/` | 4 个 `.py` | `RunState` TypedDict、`AppConfig` contextvar 注入、`rehydrate_task` 子进程入口辅助 |
| `benchmark/` | 3 个 `.py` | `DABenchPublicDataset` 加载器 + `PublicTask` / `TaskAssets` / `AnswerTable` schema |
| `llm/` | 2 个 `.py` | `build_chat_model` + caps 驱动的 `bind_tools_for_gateway` |
| `memory/` | 2 个 `.py` | scratchpad / pinned-preview / 上下文裁剪（`memory/working.py`） |
| `observability/` | 7 个 `.py` | 自定义事件 + LangSmith 回调 + `MetricsCollector` + Phase 0.5 smoke + 批量 reporter |
| `run/` | 2 个 `.py` | 子进程隔离 runner（单任务 / 批量 + 输出目录管理） |

> 子包文件计数含各自的 `__init__.py`。各子包的 `__init__.py` **故意不
> eager import 子模块**，避免循环导入。请显式
> `from data_agent_langchain.<sub>.<mod> import <name>` 拉取所需符号。

---

## 状态机拓扑

### ReAct 外层图（`agents/react_graph.py`）

```text
START → execution → finalize → END
```

外层图刻意保持极简：`execution` 是已编译的 `execution_subgraph`，
ReAct 没有 plan / replanner，子图返回 `replan_required` 时直接走 finalize。

### Plan-and-Solve 外层图（`agents/plan_solve_graph.py`）

```text
START → planner → execution
                  ├─ done            → finalize → END
                  └─ replan_required ─┐
                                      ├─ replan budget remains  → replanner → execution
                                      └─ budget exhausted       → finalize  → END
```

`planner` / `replanner` 写 `state["plan"]` 与 `plan_index`；`execution`
节点把内层 subgraph 整体当一个节点用，并裁剪重复 step，避免外层 reducer
把内层 `StepRecord` 二次追加。

### 共享内层 T-A-O 循环（`agents/execution_subgraph.py`）

```text
START → model_node → parse_action_node → gate_node ─┐
                                                    ├─ skip_tool? → advance_node
                                                    └─ no         → tool_node → advance_node
                                       advance_node ─ continue       → model_node
                                                    ─ done           → END
                                                    ─ replan_required → END
```

外层图通过读取 `state["subgraph_exit"]` 决定子图退出后是 `finalize`
还是 `replanner`。

### `RunState` 主要字段（`runtime/state.py`）

`RunState` 是一个 `TypedDict(total=False)`，所有字段都限定为可 pickle
的基本类型。下面按用途分组：

| 分组 | 字段 |
| --- | --- |
| 任务标识 | `task_id` / `question` / `difficulty` / `dataset_root` / `task_dir` / `context_dir` |
| 模式 | `mode`（`react` / `plan_solve`）、`action_mode`（`tool_calling` / `json_action`） |
| 计划追踪 | `plan` / `plan_index` / `replan_used` |
| 步骤累积 | `steps`（带 `operator.add` reducer，C1） |
| 终止信号 | `answer` / `failure_reason` |
| Gate 状态 | `discovery_done` / `preview_done` / `known_paths` / `consecutive_gate_blocks` / `gate_decision` / `skip_tool` |
| 单轮缓存 | `raw_response` / `thought` / `action` / `action_input` |
| 上一步结果 | `last_tool_ok` / `last_tool_is_terminal` / `last_error_kind` |
| 子图退出 | `subgraph_exit`（`continue` / `done` / `replan_required`，D7 / E14） |
| 步骤计数 | `step_index`（仅 `model_node` 递增，D2） / `max_steps` |
| Trace 用 | `phase` / `plan_progress` / `plan_step_description` |

> 任何不能 pickle 的对象（`Path` / `BaseTool` 实例 / `AppConfig`）都不进 state
> —— 节点入口处现场重建（§5.3 / C4 / D4）。

---

## `gateway_caps` 驱动 `bind_tools`

LangGraph 后端坚持"先探测 gateway 能力，再决定是否绑定工具"的硬约束（v4 M3）：

1. 跑一次 `dabench-lc gateway-smoke --config <YAML>`，写出
   `artifacts/gateway_caps.yaml`，记录 `tool_calling` / `parallel_tool_calls` /
   `seed_param` / `strict_mode` 四个能力位。
2. `run/runner.py` 启动时强校验 `gateway_caps.yaml` 存在；缺失会抛
   `GatewayCapsMissingError`，避免在不支持 tool-calling 的网关上"假装"支持。
3. `llm/factory.bind_tools_for_gateway(llm, tools, caps)`：
   - `caps.tool_calling=False` → 直接返回原 LLM，走 `json_action` fallback。
   - `caps.tool_calling=True` → `llm.bind_tools(tools, parallel_tool_calls=...)`。
4. `agent.action_mode` 默认 `tool_calling`（Phase 6 切换）；显式 YAML 仍可
   设回 `json_action` 走 legacy fallback / 离线 fake-model 测试。

---

## 快速上手

### 1. 准备环境

整个项目运行在 conda 虚拟环境 `dataagent` 上：

```bash
conda activate dataagent
```

或直接使用环境内 Python：

```bash
C:\Users\18155\anaconda3\envs\dataagent\python.exe -m data_agent_langchain.cli ...
```

### 2. 配置 LLM 凭据（环境变量）

按 `rules.zh.md` §5.2 / §5.4 的硬约束，**LLM 服务地址与 API Key 不应硬编码进
YAML / 镜像**。开发态 `load_app_config` 已支持 env-var 兜底，与提交态
`submission.py` 共用同一份 `MODEL_*` 协议：

```powershell
$env:MODEL_API_URL = "<你的网关 URL>"
$env:MODEL_API_KEY = "<你的 key；本地无 auth 写 EMPTY>"
$env:MODEL_NAME    = "<你的 model 名，例如 qwen3.5-35b-a3b>"
```

YAML 中 `agent.model` / `api_base` / `api_key` 留空 / 不写时，自动从对应
`MODEL_NAME` / `MODEL_API_URL` / `MODEL_API_KEY` env vars 兜底；
**YAML 显式非空值则胜出**（用于本地 mock gateway 调试时覆盖）。详见
`src/完善计划/2026-05-08-env-var-fallback-design.md`。

### 3. 准备 `gateway_caps.yaml`

第一次跑前先做一次能力探针（写 `artifacts/gateway_caps.yaml`）：

```bash
dabench-lc gateway-smoke --config configs/react_baseline.example.yaml
```

> `configs/react_baseline.example.yaml` 含有 API 密钥，**仅作示例路径引用**。
> 实际使用时请用你自己的、不入库的 YAML 副本（推荐放 `configs/local.yaml`，
> 已被 `.gitignore` 保护）；YAML 中不要写 secret，让 env vars 兜底即可。

### 4. 跑单个任务

```bash
dabench-lc run-task task_1 --config configs/react_baseline.example.yaml --graph-mode plan_solve
```

或：

```bash
python -m data_agent_langchain.cli run-task task_1 --config configs/react_baseline.example.yaml
```

### 5. 跑整套 benchmark

```bash
dabench-lc run-benchmark --config configs/react_baseline.example.yaml --limit 10
```

输出在 `artifacts/runs/<run_id>/`：
- 每任务：`<task_id>/trace.json` / `prediction.csv` / `metrics.json`
- 批量级：`summary.json`（由 `observability/reporter.py` 聚合）

`dabench-lc` 入口在 `pyproject.toml` 注册为
`dabench-lc = "data_agent_langchain.cli:main"`，与 baseline 的 `dabench`
入口共存。

---

## 配置（`config.py`）

`AppConfig` 由 6 个 frozen dataclass 组成，全部 `frozen=True, slots=True`，
可 pickle、可在子进程间安全传递：

| 子配置 | 关键字段 | 说明 |
| --- | --- | --- |
| `DatasetConfig` | `root_path` | 数据集根路径；图节点只看到字符串副本 |
| `ToolsConfig` | `python_timeout_s=30.0` / `sql_row_limit=200` | `execute_python` 沙箱超时、SQL 默认行数上限 |
| `AgentConfig` | `model` / `api_base` / `api_key` / `temperature` / `max_steps=20` / `max_replans=2` / `max_gate_retries=2` / `action_mode="tool_calling"` / `model_timeout_s=120` / `tool_timeout_s=180` / `max_obs_chars=3000` / `max_context_tokens=24000` / `seed` 等 | 主循环、超时、上下文预算、决定性 |
| `RunConfig` | `output_dir` / `run_id` / `max_workers=4` / `task_timeout_seconds=600` | runner 与子进程参数 |
| `ObservabilityConfig` | `langsmith_enabled=False` / `gateway_caps_path` | LangSmith 与 caps 文件位置 |
| `EvaluationConfig` | `reproducible=False` | `reproducible=true` 时强制 `agent.seed` 已设并禁 LangSmith（v4 E5） |

子进程序列化由 `AppConfig.to_dict` / `AppConfig.from_dict` 完成（v4 E13），
避免直接 pickle 嵌套 dataclass 时的兼容性陷阱。`load_app_config(path)` 完成
"读 YAML → 路径相对化 → 校验" 的全过程，失败抛 `ConfigError` /
`ReproducibilityViolationError`。

---

## 常见 extension point

### 加一个新的 `BaseTool`

1. 在 `tools/` 下新建 `<your_tool>.py`，定义一个 `BaseTool` 子类（参考
   `tools/answer.py` 或 `tools/read_csv.py`）。schema / description / 实现
   留在同一个文件里。
2. 把它注册到 `tools/factory.create_all_tools(task, runtime)` 的返回列表中
   ——**顺序敏感**，会影响 `bind_tools` 序列化的字节流和网关 prompt 缓存
   命中。
3. 在 `tools/descriptions.py` 添加同步的 parity 描述，确保 `json_action`
   prompt 块的 byte-level 输出与 legacy 一致（v3 D8 / §6.5）。
4. 加测试。本包测试约定：每个工具至少配一个 happy path 与一个错误分支测试。

### 改 prompt

prompt 拆分成两个文件（Phase 8 重构产物）：

- `agents/prompts.py` —— 模板编排、`ChatPromptTemplate` 构造、消息拼接
  逻辑。
- `agents/prompt_strings.py` —— 5 个纯字符串常量（system prompt、observation
  框架、gate reminder 等），不含逻辑。

只改文案不动结构，编辑 `prompt_strings.py` 即可；改消息流则到 `prompts.py`。

### 换底层 LLM

`llm/factory.build_chat_model(config)` 是 **唯一** LLM 构造入口。换 provider
时只改这一个函数（例如把 `ChatOpenAI` 换成 `ChatAnthropic`），上游所有节点
都通过 `BaseChatModel` 抽象消费它。注意 `max_retries=0`：项目级重试由
`agents/model_retry.py` 承担（§11.1.1 / E4），不要把重试逻辑下放回 SDK。

### 换记忆 / 上下文裁剪策略

`memory/working.py` 提供 4 个钩子：

- `truncate_observation` —— 单条 observation 字符串裁剪。
- `select_steps_for_context` —— 选择哪些历史 step 进入 prompt（pinned
  preview + 滑窗）。
- `render_step_messages` —— 把 `StepRecord` 渲染成 LangChain `BaseMessage`。
- `build_scratchpad_messages` —— 整体拼装 scratchpad 消息列表。

替换记忆策略只需重写这 4 个函数（或写成 strategy class 后在调用点切换）。
跨任务的更高级记忆已抽到独立提案
[`../LangChain/MEMORY_MODULE_PROPOSAL.md`](../LangChain/MEMORY_MODULE_PROPOSAL.md)，
本包暂不承载。

### 改 gate 行为

`agents/gate.py` 暴露三个集合 + 一个判定函数：

- `DATA_PREVIEW_ACTIONS` —— 哪些 action 算"数据预览"。
- `GATED_ACTIONS` —— 哪些 action 必须先看过预览才放行。
- `GATE_REMINDER` —— L2 / L3 阻塞时的 reminder 文本。
- `has_data_preview(steps)` —— 判定历史 steps 是否已经预览过数据。

修改这四样即可调整 gate 策略；`gate_node` 本身的状态机不需要改。

---

## 运行与调试

### 输出目录布局

```text
artifacts/
├── gateway_caps.yaml                    # Phase 0.5 smoke 产物
└── runs/
    └── <run_id>/
        ├── summary.json                 # 批量级聚合（reporter）
        └── <task_id>/
            ├── trace.json               # 完整 step 序列
            ├── prediction.csv           # 提交用预测
            └── metrics.json             # 单任务指标（MetricsCollector）
```

`run_id` 由 `run/runner.create_run_id` 生成（时间戳）；`--config` 里
`run.run_id` 可覆盖以复跑同一目录。

### 子进程隔离机制

`run/runner.run_single_task` 把每个任务塞进一个独立的
`multiprocessing.Process`，超时由 `RunConfig.task_timeout_seconds` 控制。
子进程内通过 `runtime/context.set_current_app_config` 把 `AppConfig` 注入
contextvar（v4 D4 / §13.1），避免直接把 `AppConfig` 塞进 `RunnableConfig`
的序列化流程。

### LangSmith 回调

`observability/tracer.build_callbacks(config, task_id, mode)` 返回 LangSmith
回调列表。**仅在 `compiled.invoke` 单点注入**（v4 D1 / §11.4.1），不要在
节点内部再挂回调，否则会重复记录。
`observability.langsmith_enabled=False` 时返回空列表。

### `MetricsCollector`

每任务级 callback，写 `metrics.json`。同样在 `compiled.invoke` 单点注入
（v4 D11 / §11.4.1），节点逻辑里不需要感知它存在。

### 自定义事件

业务节点要发自定义可观测事件时，统一调
`observability/events.dispatch_observability_event(name, data, config)`
（v4 M4 / §11.5），底层走 LangGraph 0.4 的 `dispatch_custom_event`。

### Phase 0.5 smoke

`observability/gateway_smoke.run_gateway_smoke(config, output_path)` 是
Phase 0.5 探针实现；CLI 子命令 `gateway-smoke` 是它的 typer 包装。

---

## 运行测试

整个仓库用 `pytest`：

```bash
pytest tests/ -q
```

当前基线为 **162 passed**（与
[`../LangChain/MIGRATION_STATUS.md`](../LangChain/MIGRATION_STATUS.md) §〇
保持一致）。其中
`tests/test_langchain_self_contained.py` 是一条**防倒退护栏**：扫描本包源码
导入图，禁止再出现来自 sibling 包（`data_agent_common` /
`data_agent_refactored` / `data_agent_baseline`）的 `import`。任何 PR 触发
这条测试失败都说明你引入了违反 Phase 7 边界的依赖。

`tests/conftest.py` 同时把 `src/废弃/` 加到 `sys.path`，让 legacy parity 测试
仍可跳；这与"运行时不依赖 sibling 包"并不冲突——本包源码不 import 它们即可。

---

## Phase 状态

| Phase | 状态 |
| --- | --- |
| 0 golden trace baseline | ⏸ 跳过（留待真实 API 可用时） |
| 0.5 gateway smoke | ✅ 真实网关通过 |
| 1 / 1.5 common 抽取 + 跨 backend 共享辅助 | ✅ |
| 2 / 2.5 工具 BaseTool 化 + working memory 抽取 | ✅ |
| 3 ReAct MVP | ✅ |
| 4 Plan-and-Solve graph | ✅ |
| 5 Runner + CLI + Metrics | ✅ |
| 6 默认切 `tool_calling` | ✅ |
| 7 langchain 自包含（解耦 sibling 包） | ✅ |
| 8 清理 / 拆分 / 中文化 | ✅ |

详见：

- [`../LangChain/MIGRATION_STATUS.md`](../LangChain/MIGRATION_STATUS.md) —— **single source of truth**，所有阶段细节、文件清单、测试 delta 都在此处。
- [`../LangChain/LANGCHAIN_MIGRATION_PLAN.md`](../LangChain/LANGCHAIN_MIGRATION_PLAN.md) —— v4 完整迁移方案。
- [`../LangChain/MEMORY_MODULE_PROPOSAL.md`](../LangChain/MEMORY_MODULE_PROPOSAL.md) —— 跨任务 memory 独立提案（不在主方案）。
- [`../LangChain/PHASE_05_GATEWAY_SMOKE_PLAN.md`](../LangChain/PHASE_05_GATEWAY_SMOKE_PLAN.md) —— gateway 能力探测方案。
- [`../LangChain/PHASE_6_TOOL_CALLING_DEFAULT_PLAN.md`](../LangChain/PHASE_6_TOOL_CALLING_DEFAULT_PLAN.md) —— 默认动作模式切换记录。

---

## 使用约定 / 不变量

下面这些是**硬约束**，违反它们的改动会被相应 guard 测试或 review 流程拦下：

1. **零 sibling 依赖。** 本包源码不允许 `import data_agent_common` /
   `data_agent_refactored` / `data_agent_baseline`。`tests/test_langchain_self_contained.py`
   守护这一点。
2. **子包 `__init__.py` 不 eager import 子模块。** 想拿某个符号时显式写出
   完整模块路径，例如：

   ```python
   from data_agent_langchain.agents.gate import gate_node
   from data_agent_langchain.tools.factory import create_all_tools
   from data_agent_langchain.runtime.state import RunState
   ```

3. **`RunState` 中只放可 pickle 的基本类型。** `Path` / `BaseTool` 实例 /
   `AppConfig` 等用 contextvar / 节点入口现场重建（§5.3 / C4 / D4）。
4. **回调单点注入。** LangSmith 回调与 `MetricsCollector` 仅在
   `compiled.invoke` 处挂一次，节点内部不要再挂。
5. **`step_index` 仅由 `model_node` 递增。** 其他节点只追加 `StepRecord`，
   不动计数器（D2 / §5.4）。
6. **`configs/react_baseline.example.yaml` 含 API 密钥。** 不要打开 / 提交 /
   把内容贴到任何地方（trace / 日志 / commit message）。准备自己不入库的
   YAML 副本。
7. **Phase 状态以 `MIGRATION_STATUS.md` 为准。** 本 README 的 "Phase 状态"
   小表只是镜像；细节、测试 delta、阶段决策都到那里查。
