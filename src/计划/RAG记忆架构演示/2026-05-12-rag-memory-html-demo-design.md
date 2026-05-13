# RAG 与记忆模块架构 HTML 演示设计

## 目标

为 `data_agent_langchain` 项目创建一个交互学习型 HTML 页面，用三阶段演进方式讲清：

1. 当前 LangGraph 数据智能体系统架构。
2. 如何先加入结构化 Memory MVP。
3. 如何再演进到 Corpus RAG 完整版。

## 用户选择

- 展示形式：交互学习型。
- 讲解方式：对比演进优先。
- 文件位置：`src/计划/RAG记忆架构演示/`。

## 项目架构依据

本演示基于当前 `src/data_agent_langchain` 代码结构：

- `submission.py`：容器提交入口，构造提交态配置并执行批量任务。
- `cli.py`：本地 `dabench-lc` 命令入口。
- `run/runner.py`：单任务、批量任务、并发、子进程隔离、trace/metrics 落盘。
- `agents/react_graph.py`：ReAct 外层图。
- `agents/plan_solve_graph.py`：Plan-and-Solve 外层图。
- `agents/execution_subgraph.py`：共享 T-A-O 执行子图。
- `agents/model_node.py`：构造 prompt 并调用 LLM。
- `agents/parse_action.py`：解析 tool calling 或 JSON action。
- `agents/gate.py`：数据预览门控。
- `agents/tool_node.py`：重建任务运行时并执行工具。
- `agents/advance_node.py`：决定继续、完成或触发 replan。
- `agents/finalize.py`：构造最终运行结果。
- `runtime/state.py`：`RunState` TypedDict，要求字段可 pickle。
- `memory/working.py`：当前 run 内 scratchpad / pinned preview / 上下文裁剪，不是跨 task 长期记忆。
- `src/计划/记忆模块/MEMORY_MODULE_PROPOSAL.md`：已存在的结构化记忆模块提案。

## 页面结构

HTML 页面包含五个主要区域：

1. 顶部摘要：项目定位、当前系统与扩展方向。
2. 三阶段演进路线：现状、Memory MVP、RAG 完整版。
3. 当前系统架构图：入口层、运行层、图层、节点层、工具层、状态与观测层。
4. Memory/RAG 接入点图：明确读写位置、数据边界和风险控制。
5. 实施路线与测试清单：按阶段列出建议改动和验证点。

## 交互设计

页面通过原生 HTML/CSS/JavaScript 实现，不依赖外网资源。用户可以：

- 点击阶段卡片，查看该阶段目标、文件变化、收益和风险。
- 点击架构节点，查看职责、关联文件、当前不变量、Memory/RAG 接入方式。
- 点击方案按钮，在“架构图 / 接入点 / 测试策略”之间切换。

## RAG 与 Memory 推荐方案

推荐路线是：

1. 先做结构化 Memory MVP：`dataset_knowledge` 与 `tool_playbook`。
2. 保持 `MemoryStore`、`Retriever`、`MemoryWriter` 解耦。
3. `tool_node` 只在工具成功后写结构化记录。
4. `planner_node` / `model_node` 读取召回结果。
5. `prompts.py` 做白名单字段渲染。
6. 不把 store、retriever、writer 句柄放进 `RunState`。
7. 再独立扩展 `memory/rag/` 做 corpus ingest 与检索。

## 安全与边界

- 不修改项目运行时代码。
- 不读取或修改敏感配置文件。
- 不把 API Key、模型 URL 或真实评测数据写入 HTML。
- 不建议跨 task 存储 question、answer、approach、hint、summary 等自由文本。
- 保留 `memory.mode=disabled` 作为 baseline parity 路径。

## 验收标准

- HTML 文件能在浏览器直接打开。
- 页面能解释当前系统架构。
- 页面能说明 Memory 与 RAG 的推荐接入点。
- 页面能说明为什么先做结构化 Memory，再做 RAG。
- 页面包含风险与测试策略。
