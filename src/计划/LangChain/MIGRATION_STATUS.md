# LangChain 迁移完成情况 — 截至 2026-05-07 (Phase 0.5 通过 + Phase 6 默认切换 + Phase 7 自包含 + **Phase 8 清理 / 拆分 / 中文化**)

> 配套文档：[`LANGCHAIN_MIGRATION_PLAN.md`](./LANGCHAIN_MIGRATION_PLAN.md) v4
> 范围：Phase 0.5 gateway smoke + Phase 1 + 1.5 + 2 + 2.5 + 3 + 4 + 5 + 6 默认 tool_calling + Phase 7 自包含 + **Phase 8 代码中文化 / 冷代码删除 / 大文件拆分**
> 验证：`C:\Users\18155\anaconda3\envs\dataagent\python.exe -m pytest tests/ -q` → **162 passed**
> sibling 包归档：`data_agent_common` / `data_agent_refactored` / `data_agent_baseline` 已移到 `src/废弃/`，不再在 sys.path 顺主路径上；`tests/conftest.py` 额外把归档目录加回 path，legacy parity 测试仍可跳

---

## 〇、一句话总结

`data_agent_langchain` 现为 **零外部包依赖的独立包**（不再 import `data_agent_common` / `data_agent_refactored` / `data_agent_baseline`）；全量 docstring + 行注释已中文化，两个冷代码文件已删除，两个大文件（`prompts.py` 16KB、`model_node.py` 13KB）拆出了 `prompt_strings.py` 与 `model_retry.py`。`data_agent_common` / `data_agent_refactored` / `data_agent_baseline` 三个姊妹包源文件已归档到 `src/废弃/`，不再在运行时被引用；parity 测试仍能跳得通。完整 ReAct + Plan-and-Solve + Runner / CLI / Observability + 默认 `tool_calling` + 真实 gateway smoke 全部就绪。

---

## 一、阶段对照表

| Phase | 计划章节 | 状态 | 主要产出 |
|---|---|---|---|
| **0** golden trace baseline | §15 / Phase 0 | ⏸ 跳过（用户选项 C 改 1） | 留待真实 API 可用时执行 |
| **0.5** gateway smoke test | §15 / Phase 0.5 | ✅ **真实网关通过** | `tool_calling=true` / `parallel_tool_calls=true` / `seed_param=true` / `strict_mode=true` |
| **1** common 抽取 + langchain 骨架 | §3.1 / §15 / Phase 1 | ✅ **完成** | `data_agent_common/` + `data_agent_langchain/` 骨架（细节见下） |
| **1.5** 跨 backend 共享辅助 | 巾间步骤（plan §3.1 / §8 隐含） | ✅ **完成** | `text_helpers` / `prompts` / `gate` 提到 common |
| **2** 工具层 BaseTool 子类化 | §6 / §15 / Phase 2 | ✅ **完成** | 8 个工具 + `descriptions.py` parity + `tool_runtime` + `factory` |
| **2.5** working memory 抽取 | §8 / §15 / Phase 2.5 | ✅ **完成** | `memory/working.py` + `agents/scratchpad.py` |
| **3** ReAct MVP（execution_subgraph） | §9 / Phase 3 | ✅ **完成** | 7 个节点 + minimal `config.py` + `runtime/rehydrate.py`，60 个新测试 |
| **4** Plan-and-Solve graph | §10 / Phase 4 | ✅ **完成** | `planner_node` / `replanner_node` / `plan_solve_graph`，7 个新测试 |
| **4.5 / 4.6** memory 跨 task | v4 A1（已抽到独立提案） | 🚫 **不在主方案** | 见 `MEMORY_MODULE_PROPOSAL.md` |
| **5** Runner + CLI 双入口 + Metrics | §13 / §20 / Phase 5 | ✅ **完成** | config round-trip / LLM factory / metrics + reporter / subprocess runner / Typer CLI |
| **6** 清理 + 默认 tool_calling 切换 | §15 / Phase 6 | ✅ **默认切换完成；legacy 清理待后续** | `AgentConfig.action_mode` 默认 `tool_calling`；`json_action` 显式 fallback 保留 |
| **7** langchain 自包含（common/refactored 解耦） | 计划文件外，2026-05-07 增加 | ✅ **完成** | langchain 迁移 common 全部内容到本包，取消外部 import；新增 `tests/test_langchain_self_contained.py` 作为防倒退护栏 |
| **8** 清理 / 拆分 / 中文化 | 计划文件外，2026-05-07 增加 | ✅ **完成** | 全量 docstring 中文化 + 删除 2 个冷代码 re-export + 2 个大文件拆分；sibling 包归档到 `src/废弃/`；test_phase25_working_memory 重建后 × 16 测试（覆盖上一版都多） |

---

## 二、Phase 1 详情

### 2.1 `data_agent_common/` 新建文件清单

```text
src/data_agent_common/
├── __init__.py
├── constants.py                # FALLBACK_STEP_PROMPT（v4 E1）
├── exceptions.py               # 完整异常层次（C7）
├── benchmark/
│   ├── __init__.py
│   ├── dataset.py              # DABenchPublicDataset
│   └── schema.py               # PublicTask / TaskAssets / TaskRecord / AnswerTable
├── tools/
│   ├── __init__.py
│   ├── filesystem.py           # resolve_context_path / list_context_tree / read_*
│   ├── python_exec.py          # EXECUTE_PYTHON_TIMEOUT_SECONDS=30 + execute_python_code
│   └── sqlite.py               # inspect_sqlite_schema / execute_read_only_sql
└── agents/
    ├── __init__.py
    ├── runtime.py              # StepRecord / AgentRunResult / AgentRuntimeState / ModelMessage / ModelStep
    ├── json_parser.py          # parse_model_step / parse_plan + helpers
    ├── sanitize.py             # SANITIZE_ACTIONS / ERROR_SANITIZED_RESPONSE（v4 E2）
    ├── gate.py                 # DATA_PREVIEW_ACTIONS / GATED_ACTIONS / GATE_REMINDER / has_data_preview（Phase 1.5 新增）
    ├── prompts.py              # build_observation_prompt（Phase 1.5 新增）
    └── text_helpers.py         # progress / preview_json / estimate_tokens / truncate_observation（Phase 1.5 新增）
```

### 2.2 `data_agent_refactored/` 改 thin re-export 文件清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `exceptions.py` | re-export | 完全转发 common |
| `benchmark/schema.py` | re-export | |
| `benchmark/dataset.py` | re-export | |
| `tools/filesystem.py` | re-export | |
| `tools/python_exec.py` | re-export | |
| `tools/sqlite.py` | re-export | |
| `agents/runtime.py` | re-export | |
| `agents/json_parser.py` | re-export | |
| `agents/text_helpers.py` | re-export | Phase 1.5 |
| `agents/data_preview_gate.py` | re-export + thin wrapper | `has_data_preview(state)` 包装 common 的 `has_data_preview(steps)` |
| `agents/model.py` | re-import | `ModelMessage` / `ModelStep` 从 common 导入；`ModelAdapter` / `OpenAIModelAdapter` / `ScriptedModelAdapter` 仍在本地（仅 legacy 用） |
| `agents/context_manager.py` | re-import | `ERROR_SANITIZED_RESPONSE` / `_SANITIZE_ACTIONS` 从 common 导入；`build_history_messages` 仍在本地 |
| `agents/prompt.py` | re-import | `build_observation_prompt` 从 common 导入 |
| `agents/plan_solve_agent.py` | 局部修改 | `_FALLBACK_STEP` 替换为 `from data_agent_common.constants import FALLBACK_STEP_PROMPT` |
| `tools/handlers.py` | 改导入 | 全部 import 改成 `from data_agent_common...`，删除本地 `EXECUTE_PYTHON_TIMEOUT_SECONDS` 重复 |
| `tools/registry.py` | 改导入 | benchmark/schema、exceptions 从 common 导入 |

### 2.3 `data_agent_langchain/` 新建骨架

```text
src/data_agent_langchain/
├── __init__.py
├── exceptions.py               # GatewayCapsMissingError / ReproducibilityViolationError
├── runtime/
│   ├── __init__.py
│   ├── result.py               # 从 common 再导出 StepRecord / AgentRunResult / ModelMessage / ModelStep
│   ├── state.py                # RunState TypedDict + GateDecision / LastErrorKind / SubgraphExit / AgentMode / ActionMode（含 reducer）
│   └── context.py              # set_current_app_config / get_current_app_config（contextvar；v3 D4）
├── observability/
│   ├── __init__.py
│   ├── tracer.py               # build_callbacks 占位
│   ├── metrics.py              # MetricsCollector 占位（Phase 5 实现 on_custom_event）
│   ├── gateway_caps.py         # GatewayCaps + from_yaml（v4 M3）
│   └── reporter.py             # aggregate_metrics 占位
├── llm/
│   └── __init__.py             # Phase 5 占位
├── agents/
│   ├── __init__.py             # Phase 3+ 占位
│   └── scratchpad.py           # 转发 memory/working
├── memory/
│   ├── __init__.py
│   └── working.py              # Phase 2.5 真实实现（见 §四）
├── tools/                      # Phase 2 真实实现（见 §三）
│   ├── __init__.py
│   ├── tool_runtime.py
│   ├── timeout.py
│   ├── descriptions.py
│   ├── factory.py
│   ├── answer.py
│   ├── list_context.py
│   ├── read_csv.py
│   ├── read_json.py
│   ├── read_doc.py
│   ├── inspect_sqlite_schema.py
│   ├── execute_context_sql.py
│   └── execute_python.py
└── run/
    └── __init__.py             # Phase 5 占位
```

### 2.4 `RunState` 关键字段（§5.1）

| 字段 | 类型 | 说明 / 计划点 |
|---|---|---|
| `steps` | `Annotated[list[StepRecord], operator.add]` | C1 reducer，多节点并发 append 自动合并 |
| `gate_decision` | `Literal["pass", "block", "forced_inject"]` | C11 显式路由字段 |
| `skip_tool` | `bool` | gate 决策与 tool_node 路由解耦 |
| `last_tool_ok` / `last_tool_is_terminal` | `bool` | 路由依据，禁止读 `steps[-1]` |
| `last_error_kind` | 8-state Literal | v4 E15：`tool_validation` 与 `tool_error` 已分离 |
| `subgraph_exit` | `Literal["continue", "done", "replan_required"]` | v3 D7 / v4 E14 子图退出信号 |
| `dataset_root` | `str` | v4 E8 单路径 task 重建必备 |
| `step_index`, `max_steps` | `int` | model_node 单点递增（v3 D2 待 Phase 3 实现） |

---

## 三、Phase 2 详情：BaseTool 子类（v3 D8）

### 3.1 工具文件 → BaseTool 子类映射

| 工具名 | 文件 | 类名 | Input Schema |
|---|---|---|---|
| `answer` | `tools/answer.py` | `AnswerTool` | `AnswerInput(columns: list[str], rows: list[list[Any]])` |
| `execute_context_sql` | `tools/execute_context_sql.py` | `ExecuteContextSqlTool` | `ExecuteContextSqlInput(path, sql, limit?)` |
| `execute_python` | `tools/execute_python.py` | `ExecutePythonTool` | `ExecutePythonInput(code)` |
| `inspect_sqlite_schema` | `tools/inspect_sqlite_schema.py` | `InspectSqliteSchemaTool` | `InspectSqliteSchemaInput(path)` |
| `list_context` | `tools/list_context.py` | `ListContextTool` | `ListContextInput(max_depth=4)` |
| `read_csv` | `tools/read_csv.py` | `ReadCsvTool` | `ReadCsvInput(path, max_rows=20)` |
| `read_doc` | `tools/read_doc.py` | `ReadDocTool` | `ReadDocInput(path, max_chars=4000)` |
| `read_json` | `tools/read_json.py` | `ReadJsonTool` | `ReadJsonInput(path, max_chars=4000)` |

### 3.2 `descriptions.py`（v3 D8 / §6.5）

- `_LEGACY_DESCRIPTIONS` 字典存 8 个工具的 description + input_schema
- `EXECUTE_PYTHON_TIMEOUT_SECONDS` 从 `data_agent_common.tools.python_exec` 导入（v4 E11）
- `DATA_PREVIEW_ACTIONS` / `GATED_ACTIONS` 从 `data_agent_common.agents.gate` re-export（同一对象）
- `render_legacy_prompt_block(...)` 输出与 legacy `ToolRegistry.describe_for_prompt()` 字符级一致

### 3.3 `tool_runtime.py`（v3 D3 / D12 / v4 E10 / E15）

- `ToolRuntime`：`task_dir / context_dir / python_timeout_s / sql_row_limit / max_obs_chars`，全部基本类型可 pickle
- v4 E10：**已删除** `allow_path_traversal` 字段（路径安全交给 common.tools.filesystem）
- `ToolRuntimeResult`：`ok / content / is_terminal / answer / error_kind`
- v4 E15：`error_kind: Literal["timeout", "validation", "runtime"]` 三态分离

### 3.4 `timeout.py`

- `call_tool_with_timeout(tool, action_input, timeout_seconds)`：
  - 超时 → `error_kind="timeout"`
  - pydantic `ValidationError` → `error_kind="validation"`
  - 其他异常 → `error_kind="runtime"`
- 复用 daemon-thread join 模式，与 legacy `call_with_timeout` 一致

### 3.5 v4 E9 修订实施

每个 BaseTool 子类的 `_task` / `_runtime` 用 pydantic v2 `PrivateAttr()`，在 `__init__` 中通过 `super().__init__()` 后赋值（非 `object.__setattr__`）。

---

## 四、Phase 2.5 详情：working memory（§8）

### 4.1 文件

| 文件 | 角色 |
|---|---|
| `data_agent_langchain/memory/working.py` | 真实实现 |
| `data_agent_langchain/memory/__init__.py` | 顶层 re-export |
| `data_agent_langchain/agents/scratchpad.py` | 「§8 兼容」薄壳：`from ...memory.working import ...` |

### 4.2 公共 API

| 名称 | 签名 | 行为 |
|---|---|---|
| `truncate_observation` | re-export | 来自 `data_agent_common.agents.text_helpers` |
| `render_step_messages(step, *, action_mode, max_obs_chars)` | → `list[BaseMessage]` | 单步渲染。`json_action`：`AIMessage(raw)` + `HumanMessage("Observation:...")`；`tool_calling`：`AIMessage(tool_calls)` + `ToolMessage(content, tool_call_id)`（C9 / §11.3） |
| `select_steps_for_context(steps, *, base_tokens, max_context_tokens, max_obs_chars, action_mode)` | → `(kept_indices, n_omitted)` | L2 选择：pin `DATA_PREVIEW_ACTIONS` + FIFO evict 其余 |
| `build_scratchpad_messages(steps, base_messages, *, action_mode, max_obs_chars, max_context_tokens)` | → `list[BaseMessage]` | 顶层入口，自动拼上 omission summary |

### 4.3 sanitize 协议

`step.action ∈ {"__error__", "__replan_failed__"}` 时 `raw_response` 替换为 `ERROR_SANITIZED_RESPONSE`，避免格式坏的输出污染下一轮。

### 4.4 `tool_calling` 模式特殊点

- `raw_response` 反序列化为 `tool_calls` 列表
- 解析失败 / sanitize 步骤 → 自动 fall-back 为 `json_action` 形状（`AIMessage(content) + HumanMessage`）
- `tool_call_id` 取自 `tool_calls[0]["id"]`，缺失则用 `"unknown"`

---

## 五、Phase 3 详情：ReAct MVP（§9 / 本次会话新增）

### 5.0 新建文件清单

```text
src/data_agent_langchain/
├── config.py                           # 最小 AppConfig（DatasetConfig / AgentConfig / ToolsConfig + default_app_config）
├── runtime/rehydrate.py                # rehydrate_task / build_runtime（v3 D17 / v4 E8）
└── agents/
    ├── parse_action.py                 # parse_action_node（v4 M2 完整契约）
    ├── gate.py                         # gate_node L1/L2/L3 + v4 M1 fall-through
    ├── tool_node.py                    # 自定义 tool_node（§6.3 / C2）+ v4 E15 error_kind 映射
    ├── advance_node.py                 # subgraph_exit + v4 M5 _REPLAN_TRIGGER_ERROR_KINDS 白名单
    ├── model_node.py                   # _call_model_with_retry + step_index 单点递增（D2）
    ├── prompts.py                      # build_react_messages / build_plan_solve_execution_messages
    ├── execution_subgraph.py           # 内层 T-A-O 子图（共享给 ReAct/Plan-and-Solve）
    └── react_graph.py                  # ReAct 外层包装
```

### 5.1 节点契约要点

| 节点 | I/O 摘要 | 关键设计点 |
|---|---|---|
| **model_node** | 读 state → 写 `step_index += 1`、`raw_response`、重置 per-turn 字段 | D2: 唯一递增 step_index 的节点；E4: 项目级重试-记录（max_retries=3, backoff=(2,5,10)）；C9: tool_calling 模式将 `ai_message.tool_calls` 序列化为 JSON 列表；max_steps_exceeded 哨兵 |
| **parse_action_node** | 读 `raw_response` → 写 `thought`/`action`/`action_input` 或 parse_error step | M2: json_action / tool_calling 双模式；C15: 多 tool_calls 拒绝；不递增 step_index |
| **gate_node** | 读 `action` + `consecutive_gate_blocks` → 路由 `gate_decision` + `skip_tool` | L1 block / L2 plan rewrite / L3 forced_inject + skip_tool=False（v4 M1 fall-through，与 legacy `plan_solve_agent.py:577-597` 行为一致） |
| **tool_node** | 读 `action`/`action_input` → 调用 `create_all_tools`（重建一次） + `call_tool_with_timeout` | C2: 不用 prebuilt ToolNode；E15: `timeout`/`validation`/`runtime` → `tool_timeout`/`tool_validation`/`tool_error`；自动写 `discovery_done`/`preview_done`；terminal answer 写 state.answer |
| **advance_node** | 纯 RunState → `subgraph_exit` ∈ {continue, done, replan_required} | M5: 仅 `tool_*` 错误触发 replan，parse/model/unknown_tool 走 continue；plan-and-solve 模式自动推进 plan_index + 追加 FALLBACK_STEP_PROMPT |
| **finalize_node** | 读 state.answer/failure_reason → 设置 D14 字符级失败文案 | `"Agent did not submit an answer within max_steps."`（与 legacy `BaseAgent._finalize:216-217` 一致） |

### 5.2 子图拓扑

```
START → model_node → parse_action_node → gate_node ─┐
                                                     │  skip_tool? → advance_node
                                                     └─ no → tool_node → advance_node
                                          advance_node → model_node     (continue)
                                                       → END            (done / replan_required)
```

外层 `react_graph`：`START → execution → finalize → END`。

### 5.3 LLM 注入方式

为保持 Phase 3 可独立测试（避免依赖 Phase 5 的 `build_chat_model`），`model_node` 通过 `RunnableConfig.configurable["llm"]` 接受 LLM 注入：

- **测试**：`graph.invoke(state, config={"configurable": {"llm": FakeListChatModel(...)}})`
- **生产** (Phase 5)：runner 注入 `ChatOpenAI`（或对应 gateway）；如未注入，model_node 懒加载 `data_agent_langchain.llm.factory.build_chat_model(app_config)`

### 5.4 AppConfig 状态

Phase 3 最初实现的是**最小 AppConfig**，只含运行 graph 所需字段（max_steps / max_replans / max_gate_retries / model_retry_backoff / tool_timeout_s / max_obs_chars / action_mode / require_data_preview_before_compute / python_timeout_s / sql_row_limit）。Phase 5 已补齐：

- `to_dict()` / `from_dict()` 子进程序列化（v4 E13）
- `validate_eval_config()`（v4 E5）
- `evaluation.reproducible` / `observability.langsmith_enabled` / `agent.backend` / `agent.seed`
- GatewayCaps 驱动的 `bind_tools_for_gateway`（v4 M3）

---

## 六、Phase 4 详情：Plan-and-Solve Graph（§10 / 本次会话新增）

### 6.0 新建 / 修改文件清单

```text
src/data_agent_langchain/agents/
├── planner_node.py                 # planner_node + replanner_node + shared plan generation helper
├── plan_solve_graph.py             # 外层 Plan-and-Solve graph：planner → execution → replanner/finalize
└── __init__.py                      # 导出 build_plan_solve_graph / planner_node / replanner_node
```

### 6.1 节点与图契约

| 组件 | I/O 摘要 | 关键设计点 |
|---|---|---|
| **planner_node** | 读 task state + LLM → 写 `plan` / `plan_index=0` / planning `StepRecord` | 复用 `build_planning_messages` + common `parse_plan`；失败时使用 legacy fallback plan：`["List context files", "Inspect data", "Solve and call answer"]`；记录 `action="__plan_generation__"`、`phase="planning"`、`step_index=0` |
| **replanner_node** | 读历史 `steps` → 生成 `history_hint` → 新计划或 sentinel step | 成功时替换 `plan`、重置 `plan_index=0`、`replan_used += 1`；失败时追加 `action="__replan_failed__"`、`phase="execution"`、`step_index=-1`，不覆盖原 plan |
| **plan_solve_graph** | `START → planner → execution → (replanner → execution)* → finalize → END` | 外层只读 `subgraph_exit` 路由：`done → finalize`、`replan_required + budget → replanner`、预算耗尽 → `finalize`；执行阶段完全复用 Phase 3 `execution_subgraph` |

### 6.2 复用 Phase 3 的约束

- `model_node` 已在 `mode="plan_solve"` 时读取 `state["plan"]` / `state["plan_index"]` 拼接当前计划步骤 prompt。
- `advance_node` 已实现 Plan-and-Solve 的 `plan_index` 推进、`FALLBACK_STEP_PROMPT` 追加，以及 v4 M5 `_REPLAN_TRIGGER_ERROR_KINDS` 白名单。
- `plan_solve_graph` 的 execution wrapper 会过滤嵌套 subgraph 返回的既有 `steps`，避免外层 reducer 重复追加 planning step。

---

## 七、Phase 5 详情：Runner / CLI / Observability（§13 / §20 / 本次会话新增）

### 7.0 新建 / 修改文件清单

```text
src/data_agent_langchain/
├── config.py                         # RunConfig / ObservabilityConfig / EvaluationConfig + YAML loader + round-trip
├── llm/factory.py                    # build_chat_model + bind_tools_for_gateway
├── observability/
│   ├── events.py                     # dispatch_observability_event 兼容 helper
│   ├── metrics.py                    # MetricsCollector callback，写 metrics.json
│   ├── reporter.py                   # aggregate_metrics 汇总 summary metrics
│   └── tracer.py                     # LangSmith LangChainTracer factory（无 API key 时降级 []）
├── run/
│   ├── runner.py                     # 单 task / benchmark runner，子进程隔离，写 trace/prediction/metrics/summary
│   └── __init__.py                   # 导出 runner public API
└── cli.py                            # Typer CLI: run-task / run-benchmark
pyproject.toml                        # dabench-lc = data_agent_langchain.cli:main
```

### 7.1 运行与观测契约

| 组件 | 契约 |
|---|---|
| **config** | `AppConfig.to_dict()` / `from_dict()` 支持 Path / tuple round-trip；`load_app_config()` 支持 YAML 与路径解析；`validate_eval_config()` 在 `evaluation.reproducible=true` 时强制 `agent.seed` 且禁止 LangSmith tracing |
| **LLM factory** | `build_chat_model(config)` 从 `AgentConfig` 构造 `ChatOpenAI`；`bind_tools_for_gateway(llm, tools, caps)` 仅在 gateway caps 支持 tool calling 时绑定工具，并按 caps 控制 `parallel_tool_calls` |
| **MetricsCollector** | 统计 token usage、tool calls、gate blocks、replan、parse/model errors、memory recalls；外层 chain end 写 `metrics.json` |
| **dispatch events** | `gate_node` / `parse_action_node` / `model_node` / `replanner_node` 通过 `dispatch_observability_event` 发业务事件；直接单元调用无 parent run id 时静默跳过，不影响节点行为 |
| **tracer** | `build_callbacks()` 只在 `observability.langsmith_enabled=true` 且存在 `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY` 时返回 `LangChainTracer`；否则降级 `[]` |
| **runner** | 子进程内 `set_current_app_config`、加载 task、编译 ReAct 或 Plan-and-Solve graph、单点注入 `MetricsCollector + build_callbacks()`、写 `trace.json` / `prediction.csv` / `metrics.json` |
| **benchmark** | 支持 `max_workers` 并发、`limit` 截断、汇总每 task metrics 到 `summary.json` |
| **CLI** | 新增 `dabench-lc run-task` / `dabench-lc run-benchmark`；legacy `dabench` 不变 |

---

## 八、Phase 0.5 详情：Gateway Smoke Command（本次补齐）

### 8.0 新建 / 修改文件清单

```text
src/data_agent_langchain/observability/gateway_smoke.py   # gateway capability probes + gateway_caps.yaml writer
src/data_agent_langchain/cli.py                           # dabench-lc gateway-smoke
src/data_agent_langchain/run/runner.py                    # explicit tool_calling runs require GatewayCaps
src/LangChain/PHASE_05_GATEWAY_SMOKE_DESIGN.md            # approved design
src/LangChain/PHASE_05_GATEWAY_SMOKE_PLAN.md              # implementation plan
```

### 8.1 运行契约

| 组件 | 契约 |
|---|---|
| **gateway-smoke CLI** | `dabench-lc gateway-smoke --config <yaml> [--output <gateway_caps.yaml>]`；只新增在 LangGraph CLI，不修改 legacy `dabench` |
| **凭据读取** | `agent.api_key` 优先；为空时 fallback 到 `OPENAI_API_KEY` |
| **输出** | 写 `gateway_caps: {tool_calling, parallel_tool_calls, seed_param, strict_mode}` |
| **runner 启动检查** | 仅当 `agent.action_mode == "tool_calling"` 时要求 `gateway_caps.yaml` 存在且 `tool_calling=true`；`json_action` 不要求 caps |
| **默认模式** | 默认已切到 `tool_calling`；显式 `json_action` fallback 保留 |

---

## 八半、Phase 7 详情：langchain 自包含迁移（2026-05-07）

### 8半.0 目标

让 `data_agent_langchain` 可以脱离 `data_agent_common` / `data_agent_refactored` / `data_agent_baseline` 独立运行；删掉这三个包后 `dabench-lc` 仍可启动。

### 8半.1 新建 / 改写文件清单

```text
src/data_agent_langchain/
├── constants.py                      ← 新建：FALLBACK_STEP_PROMPT
├── exceptions.py                     ← 重写：从 thin re-export 改为完整异常层次
├── benchmark/
│   ├── __init__.py                   ← 新建
│   ├── schema.py                     ← 新建：PublicTask / TaskAssets / TaskRecord / AnswerTable
│   └── dataset.py                    ← 新建：DABenchPublicDataset / TASK_DIR_PREFIX
├── tools/
│   ├── filesystem.py                 ← 新建：list_context_tree / read_csv_preview / read_json_preview / read_doc_preview / resolve_context_path
│   ├── python_exec.py                ← 新建：execute_python_code / EXECUTE_PYTHON_TIMEOUT_SECONDS=30
│   └── sqlite.py                     ← 新建：inspect_sqlite_schema / execute_read_only_sql
└── agents/
    ├── __init__.py                   ← 改写：去除 eager imports（避免循环）
    ├── runtime.py                    ← 新建：StepRecord / AgentRunResult / AgentRuntimeState / ModelMessage / ModelStep
    ├── json_parser.py                ← 新建：parse_model_step / parse_plan / 多层 JSON 修复
    ├── text_helpers.py               ← 新建：progress / preview_json / estimate_tokens / truncate_observation
    ├── sanitize.py                   ← 新建：SANITIZE_ACTIONS / ERROR_SANITIZED_RESPONSE
    ├── observation_prompt.py         ← 新建：build_observation_prompt（独立文件避免与 prompts.py 循环）
    ├── gate.py                       ← 改写：内联 DATA_PREVIEW_ACTIONS / GATED_ACTIONS / GATE_REMINDER / has_data_preview，tools.factory 改为函数局部惰性导入
    └── prompts.py                    ← 改写：内联 REACT_SYSTEM_PROMPT / PLAN_AND_SOLVE_SYSTEM_PROMPT / RESPONSE_EXAMPLES / PLANNING_INSTRUCTION / EXECUTION_INSTRUCTION / build_task_prompt（原本从 data_agent_refactored.agents.prompt 懒导入）
```

其余 ~20 个 langchain 源文件的 `from data_agent_common.X import ...` import 路径替换为 `from data_agent_langchain.X import ...`，纯路径替换，无逻辑改动。

### 8半.2 关键技术决策

| 决策 | 原因 |
|---|---|
| `agents/observation_prompt.py` 独立文件 | `agents/prompts.py` 已被 `memory/working.py` 反向 import；如把 `build_observation_prompt` 也放在 `prompts.py`，会因 prompts.py → working.py → prompts.py 形成循环 |
| `agents/gate.py` 把 `tools.factory` / `tools.timeout` 改为函数局部惰性导入 | `tools/descriptions.py` 需要从 `agents/gate.py` 取 `DATA_PREVIEW_ACTIONS` 等常量；如果 `gate.py` 在 top-level import `tools.factory`，则 tools 子树和 agents 子树会通过 `tools/answer.py → tools/descriptions.py → agents/gate.py → tools/factory.py → tools/answer.py` 循环 |
| `agents/__init__.py` 去除 eager imports | 之前的 `agents/__init__.py` 在加载任何 agents 子模块时都强制 import 整张图，进一步加剧循环；改为空 docstring 后不影响功能（项目内全部使用 `from data_agent_langchain.agents.X import Y` 而非 `from data_agent_langchain.agents import Y`） |

### 8半.3 测试影响

- `tests/test_phase25_working_memory.py::test_data_preview_actions_identity_across_packages` 由"三个对象同一性"放宽为"common+refactored 仍同一对象 + langchain 字符级相等"。
- `tests/test_phase2_tools_functional.py::test_answer_tool_success_terminal` 的 `AnswerTable` import 从 `data_agent_common.benchmark.schema` 改到 `data_agent_langchain.benchmark.schema`，因为 langchain 工具现在返回 langchain 自己的 `AnswerTable` 类。
- 新增 `tests/test_langchain_self_contained.py`（1 个测试），AST 扫描 `src/data_agent_langchain/**/*.py`，断言零外部包导入。这是防倒退的硬护栏。

### 8半.4 三个姊妹包当前状态（Phase 7 底）

| 包 | 改动 | `python -m XXX.cli` 可运行性 | 注释 |
|---|---|---|---|
| `data_agent_common` | 源文件零变更 | n/a（不是入口） | 仍被 `data_agent_refactored` 使用 |
| `data_agent_refactored` | 源文件零变更 | ✅ 可运行（`python -m data_agent_refactored.cli`） | 依赖 common，不依赖 langchain |
| `data_agent_baseline` | 源文件零变更 | ✅ 可运行（`dabench` 命令） | 自包含，与本次迁移无关 |
| `data_agent_langchain` | 12 个新文件 + ~20 个 import 改写 + 3 个文件结构改写 | ✅ 可运行（`dabench-lc` 命令） | 零外部包 import |

---

## 八¾、Phase 8 详情：清理 / 拆分 / 中文化（2026-05-07）

### 8¾.0 目标

三个子目标并行推进：

  1. **干净整洁**：删除 `data_agent_langchain` 中不再被使用的 thin re-export 冷代码。
  2. **大文件拆分**：让单个文件不再超 18KB，逻辑与大体量常量分居。
  3. **全量中文化**：所有 docstring + `#` 行注释转中文；LLM prompt、logger / exception / CLI 文案保留英文。

### 8¾.1 冷代码删除

| 文件 | 原始定位 | 处理 |
|---|---|---|
| `agents/scratchpad.py` | 纯 re-export `memory/working`，src 零消费者 | **删除**；同步删除 `tests/test_phase25_working_memory.py::test_scratchpad_reexport_resolves_to_working` |
| `runtime/result.py` | 纯 re-export `agents/runtime`，src 零消费者 | **删除**；同步修改 `tests/test_phase1_imports.py:174` 的 import 路径 |

### 8¾.2 大文件拆分

| 原文件 | 原大小 | 拆出去 | 现在大小 |
|---|---|---|---|
| `agents/prompts.py` | 16,543 bytes | `agents/prompt_strings.py`（5 个 LLM 字符串常量） | 8,257 bytes（-50%） |
| `agents/model_node.py` | 13,182 bytes | `agents/model_retry.py`（`call_model_with_retry` + `extract_raw_response`） | 9,634 bytes（-27%） |

各拆出文件都保留了带下划线的向后兼容别名（例如 `_extract_raw_response` / `_call_model_with_retry`），老试 / 外部调用点不需同时改动。

### 8¾.3 sibling 包归档

`data_agent_common` / `data_agent_refactored` / `data_agent_baseline` 被移到 `src/废弃/`。后果：

  - **运行时**：sys.path 上仅有 `src/`，langchain 包 import 不到 sibling 包（这是 Phase 7 设计的边界，仍然成立）。
  - **测试时**：`tests/conftest.py` 额外把 `src/废弃/` 也加入 sys.path，legacy parity 测试仍能跳。

### 8¾.4 test_phase25_working_memory.py 重建

中间有次事故：PowerShell 处理脚本意外把这个测试文件清零为 0 字节。项目不走 git，无可恢复。重建后覆盖：

  - `truncate_observation` × 2
  - `render_step_messages` × 5（json_action × 2 + tool_calling × 3）
  - `select_steps_for_context` × 3（pinning + 空列表 + base 超预算后仍保留 1 条）
  - `build_scratchpad_messages` × 3（append 演示 + 空输入 + 警告消息插入）
  - cross-backend parity × 3（build_observation_prompt / DATA_PREVIEW_ACTIONS / SANITIZE_ACTIONS）

总计 16 个测试，覆盖面不低于原件（3 重记 + 13 个原始测试 = 16）。

### 8¾.5 中文化范围

按用户选项 A（保守）执行：

  - **翻译**：所有模块 / 类 / 函数 docstring + `#` 行注释。
  - **保留英文**：LLM prompt 字符串（`REACT_SYSTEM_PROMPT` 等 5 个会被 parity 测试 byte-for-byte 记忆）、`logger.warning/error/info` 消息、`raise XxxError(...)` 消息、`typer.echo` / typer.help 文案、API 名 / 类名 / 字段名。
  - **保留原引用**：parity 编号（`v4 M1` / `D14` / `§7.4` / `LANGCHAIN_MIGRATION_PLAN.md` 等）、legacy 文件路径。

涵盖 41 个 `.py` 文件（包括 2 个 Phase 8 拆分新增的 `prompt_strings.py` 与 `model_retry.py`）。

### 8¾.6 文件大小分布（Phase 8 后）

最大 8 个文件：

```text
  17,834 bytes  run/runner.py             # Phase 5 逻辑多，中文后略增
  11,548 bytes  agents/gate.py            # 3 层 gate L1/L2/L3 逻辑·本身就不小
  11,426 bytes  memory/working.py         # scratchpad + token 预算 + sanitize
  10,561 bytes  agents/prompt_strings.py  # 5 个 LLM 字符串常量
  10,392 bytes  config.py                 # 6 个 dataclass
   9,634 bytes  agents/model_node.py      # 拆后（13KB → 9.6KB）
   8,946 bytes  agents/tool_node.py       # tool_node + 3 条错误路径
   8,257 bytes  agents/prompts.py         # 拆后（16KB → 8KB）
```

未拆分但主动保留：`memory/working.py` 是一个高内聚单元（三块逻辑全部围绕 scratchpad 处理），拆为多个文件会起不了高质量调用名。

---

## 九、测试矩阵

`C:\Users\18155\anaconda3\envs\dataagent\python.exe -m pytest tests/ -q` → **162 passed** in ~3.4s

### 9.1 Phase 1/1.5/2/2.5 既有测试（54）

| 测试文件 | 数量 | 覆盖 |
|---|---|---|
| `tests/test_phase1_imports.py` | 4 | 共享包符号、legacy re-export、对象同一性、langchain skeleton import |
| `tests/test_phase1_parity_constants.py` | 7 | `FALLBACK_STEP_PROMPT` byte-for-byte（v4 E1）、`SANITIZE_ACTIONS` byte-for-byte（v4 E2）、`ERROR_SANITIZED_RESPONSE` byte-for-byte、`_SANITIZE_ACTIONS` 与 common 同对象、`EXECUTE_PYTHON_TIMEOUT_SECONDS == 30` |
| `tests/test_phase1_runstate.py` | 6 | `RunState` 字段完备、`Annotated[list, operator.add]` reducer、`LastErrorKind` 含 `tool_validation`/`tool_error`（v4 E15）、`SubgraphExit` 三态、`GateDecision` 三态、contextvar 未初始化时清晰报错 |
| `tests/test_phase2_descriptions_parity.py` | 9 | description block 与 `ToolRegistry.describe_for_prompt()` 字符级一致、工具名集合 parity（v4 E12）、`DATA_PREVIEW_ACTIONS` / `GATED_ACTIONS` 同一对象、`EXECUTE_PYTHON_TIMEOUT_SECONDS` 在描述中、`ToolRuntime` 可 pickle、无 `allow_path_traversal` 字段（v4 E10）、`error_kind` 三态构造 |
| `tests/test_phase2_tools_functional.py` | 15 | 8 个 BaseTool 真实数据 E2E（合成 task 含 CSV / JSON / MD / SQLite）、`AnswerTool` 4 种验证错误路径、`ExecuteContextSqlTool` 拒绝写操作、`call_tool_with_timeout` timeout/validation 映射、factory 决定性顺序 + PrivateAttr 注入正确 |
| `tests/test_phase25_working_memory.py` | 13 | json_action / tool_calling 双模式渲染、tool_calling fall-back、sanitize 替换 `__error__` 与 `__replan_failed__`、pin 数据预览 step、failed 预览不 pin、空 steps 处理、omission summary 文案、scratchpad re-export 同一函数对象、`build_observation_prompt` 跨 backend 同函数、`DATA_PREVIEW_ACTIONS` 三处 import 同一对象 |

### 9.2 Phase 3 新增测试（60）

| 测试文件 | 数量 | 覆盖 |
|---|---|---|
| `tests/test_phase3_finalize.py` | 6 | `_MAX_STEPS_FAILURE_MSG` byte-for-byte（D14）、answer 已存在时不覆盖 failure_reason、自定义 failure 不被覆盖、`build_run_result` 成功/失败两路 |
| `tests/test_phase3_parse_action.py` | 9 | json_action 成功/格式错/缺 action、tool_calling 单 tool_call 成功、dict payload 单 call、空数组、多 tool_calls 拒绝（C15）、缺 name、空 raw |
| `tests/test_phase3_advance.py` | 16 | 6 条规则 × 多场景：rule1 答案/terminal、rule2 步数耗尽、rule3 gate_block continue、rule5 白名单内 3 种 + 白名单外 3 种 + 预算耗尽、rule4 plan-and-solve 推进 + FALLBACK_STEP 不重复追加、`_REPLAN_TRIGGER_ERROR_KINDS` 集合断言 |
| `tests/test_phase3_tool_node.py` | 9 | skip_tool / parse_error 透传、list_context 设置 discovery_done、read_csv 设置 preview_done、answer terminal 写入 state、unknown_tool 错误、validation/runtime 错误映射、rehydrate 失败容错 |
| `tests/test_phase3_gate.py` | 8 | 未 gate 直通、preview 已完成直通、parse_error 透传、L1 block 步骤记录与 GATE_REMINDER 文本、L2 plan rewrite "MANDATORY: Inspect data files first..."、L3 forced_inject + skip_tool=False（v4 M1）+ consecutive_gate_blocks 重置、`require_data_preview_before_compute=False` 关闭 gating |
| `tests/test_phase3_model.py` | 9 | step_index 递增、json_action 写 raw_response、tool_calling 序列化 tool_calls 列表（C9）、tool_calling ReAct/Plan-Solve prompt 使用真实 tool calls 而非 fenced JSON、`_extract_raw_response` 兼容 string/AIMessage/multi-part content、retry 耗尽返回 model_error step、`_call_model_with_retry` 重试逻辑、step 预算耗尽不调 LLM |
| `tests/test_phase3_react_e2e.py` | 3 | 完整 ReAct graph E2E（list_context → read_csv → answer 三步成功）、`build_run_result` 翻译终态、max_steps 失败时 finalize 写 D14 文案 |

### 9.3 Phase 4 新增测试（7）

| 测试文件 | 数量 | 覆盖 |
|---|---|---|
| `tests/test_phase4_planner_node.py` | 4 | planner 成功生成计划 + planning step、planner fallback plan、replanner history_hint（已成功步骤 + 最后失败动作）、replanner 失败时 `__replan_failed__` sentinel |
| `tests/test_phase4_plan_solve_graph.py` | 3 | `_route_after_execution` done/replan/预算耗尽路由、完整 Plan-and-Solve graph E2E（plan → list_context → read_csv → answer） |

### 9.4 Phase 5 / Phase 6 新增测试（37）

| 测试文件 | 数量 | 覆盖 |
|---|---|---|
| `tests/test_phase5_config.py` | 6 | `AppConfig` round-trip、YAML load + path resolve、reproducible 约束、默认 `tool_calling`、显式 `json_action` fallback |
| `tests/test_phase5_llm_factory.py` | 4 | `build_chat_model` 参数映射、GatewayCaps tool binding、parallel flag 控制 |
| `tests/test_phase5_observability.py` | 12 | custom events 计数、outermost chain 写盘、metrics aggregation、tool name 恢复、gate/parse/model/replan/tool_call dispatch、LangSmith tracer factory 降级 |
| `tests/test_phase5_runner_cli.py` | 10 | initial state、single task 输出、benchmark summary、`dabench-lc` entry、runner callbacks 单点注入、默认 `tool_calling` caps 启动检查与工具绑定、显式 `json_action` fallback、LangGraph `recursion_limit` 随 `max_steps` 注入 |
| `tests/test_phase05_gateway_smoke.py` | 5 | gateway_caps.yaml 写入、tool-calling failure 降级、`OPENAI_API_KEY` fallback、`dabench-lc gateway-smoke` CLI 注册与 output 传递 |

### 9.5 Phase 7 新增测试（1）

| 测试文件 | 数量 | 覆盖 |
|---|---|---|
| `tests/test_langchain_self_contained.py` | 1 | AST 扫 `src/data_agent_langchain/**/*.py`，断言零 `from data_agent_common` / `from data_agent_refactored` / `from data_agent_baseline` 导入 |

### 9.6 Phase 8 测试变动（+3）

- `tests/test_phase25_working_memory.py` 重建后从 13 个测试 → 16 个测试（+3）。
- `tests/test_phase1_imports.py` 中一行 import 路径从 `runtime.result` 改为 `agents.runtime`（后者是前者删除后的真正定义位置）。
- 被删除的测试：`test_scratchpad_reexport_resolves_to_working`（验证 thin re-export 的套话合同，后者已删除）。

### 9.7 测试约定

测试在 conda 环境 `dataagent` 中运行；`tests/` 目录被 `.gitignore` 排除（项目约定），通过 PowerShell `Set-Content` / `Add-Content` 创建。**Phase 3 / Phase 4 / Phase 5 全部测试不依赖网络**：使用 `langchain_core.language_models.FakeListChatModel` / `RunnableLambda` 注入 LLM。

---

## 十、计划修订点（v4 / v3 / v2）实施情况

### 10.1 v4 关键修复 / 调整

| 编号 | 内容 | 实施位置 | 状态 |
|---|---|---|---|
| **M1** Gate L3 fall-through 执行原 action | §7.4 / §16.1 | `data_agent_langchain/agents/gate.py` | ✅ Phase 3 |
| **M2** parse_action_node 完整 I/O 契约 | §9.2.1 | `data_agent_langchain/agents/parse_action.py` | ✅ Phase 3 |
| **M3** GatewayCaps 配置驱动 bind_tools | §11.1 | `observability/gateway_caps.py` + `llm/factory.py` | ✅ Phase 5 |
| **M4** dispatch_custom_event / on_custom_event | §11.5 / §20.4 | `observability/events.py` + `observability/metrics.py` | ✅ Phase 5 |
| **M5** advance_node `_REPLAN_TRIGGER_ERROR_KINDS` 白名单 | §10.3 | `data_agent_langchain/agents/advance_node.py` | ✅ Phase 3 + Phase 4 |
| **A1** memory 跨 task 抽到独立提案 | §3.2 / §15 | 仅 `memory/working.py` 在主方案 | ✅ |
| **A2** parity 期双 CLI 入口 | §13.4 / §15 | `pyproject.toml` `[project.scripts]` | ✅ Phase 5 |
| **A3** dispatch_custom_event 评测约束 | §4.5 / §11.5 | `observability/events.py` | ✅ Phase 5 |
| **E1** `FALLBACK_STEP_PROMPT` 提到 common.constants | §3.1 / §10.3 | `data_agent_common.constants` | ✅ |
| **E2** sanitize 提到 common.agents.sanitize | §3.1 / §10.4 | `data_agent_common.agents.sanitize` | ✅ |
| **E4** `_call_model_with_retry` 骨架 | §11.1.1 | `data_agent_langchain/agents/model_node.py` | ✅ Phase 3 |
| **E5** `validate_eval_config` 调用时机 | §11.4.5 | `data_agent_langchain/config.py` | ✅ Phase 5 |
| **E6** Phase 0 golden trace fixture 规范 | Phase 0 | `tests/golden/` | ⏸ 跳过 |
| **E7** Phase 依赖图 | §15.0 | 仅在文档 | ✅（plan 内） |
| **E8** Runner `_build_initial_state` 必含 dataset_root | §13.1 / §15 | `data_agent_langchain/run/runner.py` | ✅ Phase 5 |
| **E9** BaseTool `PrivateAttr() + __init__` | §6.1.2 | 8 个工具 | ✅ |
| **E10** `ToolRuntime` 删 `allow_path_traversal` | §6.1.1 | `tools/tool_runtime.py` | ✅ |
| **E11** `EXECUTE_PYTHON_TIMEOUT_SECONDS` 从 common 导入 | §6.5.2 | `tools/descriptions.py` | ✅ |
| **E12** 工具名集合断言 | §16.1 | `test_tool_name_set_parity_v4_E12` | ✅ |
| **E13** `AppConfig.to_dict() / from_dict()` round-trip | §13.1 / §15 | `data_agent_langchain/config.py` | ✅ Phase 5 |
| **E14** `subgraph_exit` 字段定义统一 §5.1 | §5.1 / §9.0 | `runtime/state.py` | ✅ |
| **E15** `tool_validation` 与 `tool_error` 拆分 | §5.1 / §6.4 | `RunState.LastErrorKind` + `ToolErrorKind` | ✅ |

### 10.2 v3 关键修复

| 编号 | 内容 | 状态 |
|---|---|---|
| **D1** LangSmith callback 单点注入 | ✅ Phase 5（`runner.py` 仅在 `compiled.invoke` 注入 callbacks） |
| **D2** `step_index` 由 model_node 单点递增 | ✅ Phase 3 |
| **D3** `ToolRuntime` 显式 dataclass | ✅ |
| **D4** `AppConfig` 通过 contextvar 注入 | ✅ Phase 3 contextvar + Phase 5 runner/config round-trip |
| **D5** `_can_replan` + `_FALLBACK_STEP` 兜底 | ✅ Phase 3 + Phase 4（`_can_replan` / FALLBACK_STEP / replanner_node 均已实现） |
| **D6** §4.5 评测确定性约束 | ✅ Phase 5 |
| **D7** `execution_subgraph` 抽出 | ✅ Phase 3 |
| **D8** `BaseTool` 子类 + 描述 parity | ✅ |
| **D9 / D10** memory 模块按数据语义重命名 | 🚫 抽到独立提案（v4 A1） |
| **D11** `MetricsCollector` callback | ✅ Phase 5 |
| **D12** `ToolRuntimeResult.error_kind` 工具源头打标 | ✅ |
| **D13** `bind_tools_safely` | ✅ Phase 5（v4 M3 重构为 `bind_tools_for_gateway(caps)`） |
| **D14** `finalize_node` 失败文案 byte-for-byte | ✅ Phase 3 |
| **D15** CLI 单入口 | ✅ Phase 5（v4 A2 改双入口：`dabench` + `dabench-lc`） |
| **D16** 删除 `strip_langsmith_env` | ✅ 不存在该模块 |
| **D17** `_rehydrate_task` 单路径 | ✅ Phase 3（`runtime/rehydrate.py`） |

---

## 十一、当前可用功能

```python
# 1) legacy backend 完全不受影响（旧 import 仍生效）
from data_agent_refactored.cli import main
from data_agent_refactored.run.runner import run_single_task, run_benchmark

# 2) 共享包可被新代码使用
from data_agent_common.benchmark.dataset import DABenchPublicDataset
from data_agent_common.agents.runtime import StepRecord, AgentRunResult
from data_agent_common.agents.json_parser import parse_model_step
from data_agent_common.agents.gate import DATA_PREVIEW_ACTIONS, has_data_preview
from data_agent_common.constants import FALLBACK_STEP_PROMPT

# 3) 新 langchain 工具集可单独使用（不依赖 graph）
from data_agent_langchain.tools import (
    create_all_tools, ToolRuntime, call_tool_with_timeout,
    render_legacy_prompt_block,
)

# 4) 完整 ReAct LangGraph 可运行（Phase 3 新增）
from langchain_core.language_models import FakeListChatModel
from data_agent_langchain.agents import build_react_graph, build_run_result
from data_agent_langchain.config import default_app_config
from data_agent_langchain.runtime.context import set_current_app_config

set_current_app_config(default_app_config())
graph = build_react_graph().compile()
fake_llm = FakeListChatModel(responses=[
    '```json\n{"thought":"","action":"list_context","action_input":{}}\n```',
    '```json\n{"thought":"","action":"answer","action_input":{"columns":["x"],"rows":[[1]]}}\n```',
])
final = graph.invoke(
    {
        "task_id": "task_001", "dataset_root": "<dataset_root>",
        "mode": "react", "action_mode": "json_action",
        "step_index": 0, "max_steps": 8, "steps": [],
    },
    config={"configurable": {"llm": fake_llm}},
)
result = build_run_result("task_001", final)
# result.answer / result.steps / result.failure_reason 是 legacy AgentRunResult

# 5) Plan-and-Solve LangGraph 可运行（Phase 4 新增）
from data_agent_langchain.agents import build_plan_solve_graph

plan_graph = build_plan_solve_graph().compile()
plan_final = plan_graph.invoke(
    {
        "task_id": "task_001", "dataset_root": "<dataset_root>",
        "mode": "plan_solve", "action_mode": "json_action",
        "plan": [], "plan_index": 0, "replan_used": 0,
        "step_index": 0, "max_steps": 20, "steps": [],
    },
    config={"configurable": {"llm": fake_llm}},
)

# 6) Phase 5 runner / metrics / config 可用
from data_agent_langchain.config import load_app_config, AppConfig
from data_agent_langchain.run.runner import run_single_task, run_benchmark
from data_agent_langchain.observability.metrics import MetricsCollector
from data_agent_langchain.observability.reporter import aggregate_metrics

# CLI: dabench-lc run-task / dabench-lc run-benchmark / dabench-lc gateway-smoke

# 7) Phase 0.5 gateway smoke 可生成 caps（真实运行需要 API key）
from data_agent_langchain.observability.gateway_smoke import run_gateway_smoke
from data_agent_langchain.observability.gateway_caps import GatewayCaps

# 8) RunState / working memory 仍可单独使用
from data_agent_langchain.runtime.state import RunState
from data_agent_langchain.memory.working import build_scratchpad_messages
```

---

## 十二、下一步（真实单题 smoke / legacy 清理）

Phase 0.5 gateway smoke 已通过，Phase 6 默认 `tool_calling` 切换已完成；后续工作如下：

### 12.1 可选真实单题 smoke test

- 用户已手动跑通 `task_1` 写出 trace；`task_11` 暴露 LangGraph 默认 `recursion_limit=25` 不足，runner 已改为按 `agent.max_steps` 注入更高 `recursion_limit`。
- `task_11` 二次 smoke 已越过 recursion limit，但暴露默认 `tool_calling` prompt 仍要求 fenced JSON；已修为 tool_calling 执行阶段要求真实 tool calls。
- `task_11` 三次 smoke 已成功：`succeeded=true`，输出 `prediction.csv` 三行（ID/SEX/Diagnosis），trace 含 `list_context` / `read_doc` / `read_json` / `execute_python` / `answer` 真实 tool calls。
- 已补 custom `tool_node` 的 `tool_call` metrics 事件；重跑 `task_11` 后 `metrics.json.tool_calls` 已正确记录 `list_context=1`、`read_doc=1`、`read_json=2`、`execute_python=1`、`answer=1`。
- Cascade 不读取或修改 `configs/react_baseline.example.yaml`。

### 12.2 Phase 6 后续清理

- 删除 legacy backend 中已被调用者迁移走的 wrapper（需 Phase 0 golden trace 验证 parity）
- 补 Phase 0 golden trace parity 后再决定是否收敛 legacy wrapper / CI 策略。

---

## 十三、验证脚本

```powershell
# 全量测试
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/ -q

# 仅单 phase
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase1_imports.py tests/test_phase1_parity_constants.py tests/test_phase1_runstate.py -v
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase2_descriptions_parity.py tests/test_phase2_tools_functional.py -v
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase25_working_memory.py -v
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase3_finalize.py tests/test_phase3_parse_action.py tests/test_phase3_advance.py tests/test_phase3_tool_node.py tests/test_phase3_gate.py tests/test_phase3_model.py tests/test_phase3_react_e2e.py -v
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase4_planner_node.py tests/test_phase4_plan_solve_graph.py -v
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase05_gateway_smoke.py tests/test_phase5_config.py tests/test_phase5_llm_factory.py tests/test_phase5_observability.py tests/test_phase5_runner_cli.py -v

# 旧 baseline 仍可运行（实际跑 task 需真实 API key）
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -c "from data_agent_refactored.agents import PlanAndSolveAgent, ReActAgent; print('legacy OK')"
```

---

## 十四、本次会话引用文档

- 执行依据：[`LANGCHAIN_MIGRATION_PLAN.md`](./LANGCHAIN_MIGRATION_PLAN.md) v4
- 跨 task memory 单独提案：[`MEMORY_MODULE_PROPOSAL.md`](./MEMORY_MODULE_PROPOSAL.md)（暂不落地）
- 用户全局规则：`C:/Users/18155/.codeium/windsurf/memories/global_rules.md`（Superpowers TDD 工作流）
- 工程纪律：本次严格遵循 v4 修订点；测试覆盖中只有「golden trace 等价性」（D14 / 验收 16.2 第 1 项）与真实 gateway smoke 跳过——需要真实 API key + Phase 0 baseline 才能跑通，留待真实模型可用时补齐。
