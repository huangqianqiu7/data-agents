# RAG Memory v2 功能分支改动记录

本文记录 `feature/rag-memory-v2` 相对本地 `main` 的主要功能改动、验证方式与运行边界，便于后续回顾、合并与评审。

## 目标

本分支实现 `2026-05-13-rag-memory-architecture-plan-v2.md` 中的结构化记忆 MVP：

- 在跨任务范围内保存安全的结构化 dataset facts。
- 在 planner / model 路径中召回并注入只读事实。
- 保持 `memory.mode=disabled` 的 baseline parity。
- 为后续 corpus RAG 留出独立边界，但不在本分支实现完整 RAG。

## 核心模块

### Memory 数据结构

新增 `src/data_agent_langchain/memory/` 子包：

- `base.py`：定义 `MemoryRecord`、`RetrievalResult`、`MemoryStore`、`Retriever`、`MemoryWriter` 协议。
- `records.py`：定义白名单 dataclass：`DatasetKnowledgeRecord` 与 `ToolPlaybookRecord`。
- `types.py`：定义写入 `RunState` 的轻量 `MemoryHit`。
- `stores/jsonl.py`：追加式 JSONL store，按 namespace 编码到独立文件，支持 tombstone 删除。
- `retrievers/exact.py`：精确 namespace retriever，返回最新记录。
- `writers/store_backed.py`：按 memory mode 控制 dataset facts 与 tool playbook 写入。
- `factory.py`：集中构建 store / retriever / writer。
- `rag/__init__.py`：保留 Phase M4 corpus RAG 边界。

### 配置

`MemoryConfig` 加入顶层 `AppConfig`：

- `mode`: `disabled | read_only_dataset | full`
- `store_backend`: 当前为 `jsonl`
- `path`: memory 存储路径，YAML 相对路径按配置文件目录解析
- `retriever_type`: 当前为 `exact`
- `retrieval_max_results`

CLI 增加 `--memory-mode` 覆盖：

- `run-task --memory-mode ...`
- `run-benchmark --memory-mode ...`

提交态 `build_submission_config()` 默认使用 `memory.mode=read_only_dataset`，避免评测路径进入写入模式。

## 写入路径

`tool_node` 在成功执行数据预览工具后尝试写入 `DatasetKnowledgeRecord`：

- 支持 `read_csv`
- 支持 `read_json`
- 支持 `read_doc`
- 支持 `inspect_sqlite_schema`

写入特点：

- 仅 dataset-preview 类 action 会初始化 memory store / writer。
- `memory.mode=disabled` 时不写入。
- schema 缺失、空 schema、路径缺失时安全跳过。
- memory 子系统或 writer 失败不会影响原工具执行结果。
- SQLite schema 输出中的 `tables/create_sql` 会转换为安全的结构化 schema 摘要。

## 召回与 Prompt 注入

新增 `agents/memory_recall.py`：

- `recall_dataset_facts(memory_cfg, dataset, node, config)` 从 `dataset:{dataset}` namespace 召回记录。
- 只生成 `MemoryHit`，不把 store / retriever / writer 放入 `RunState`。
- 只使用白名单 payload 字段渲染 summary。
- 分发 `memory_recall` 观测事件，包含 node、namespace、k、hit_ids、reason。
- 对未知 memory mode fail closed；负数 `retrieval_max_results` clamp 到 0。

`planner_node`：

- 生成或 fallback plan 后召回 dataset facts。
- 仅当命中非空时把 `memory_hits` 写入 partial state。
- disabled mode 下不产生 `memory_hits`。

`model_node`：

- 不做任何 memory 检索。
- 只读取 state 中已有 `memory_hits`。
- 通过 `render_dataset_facts()` 追加只读 facts 段到 prompt 末尾。
- ReAct 与 Plan-and-Solve prompt 都覆盖。

## Prompt 白名单

`agents.prompts.render_dataset_facts()` 只渲染 `MemoryHit.summary`：

- 不渲染 `record_id`
- 不渲染 `namespace`
- 不渲染 `score`
- 不从原始 payload 透传 question / answer / approach / hint / summary 等自由文本

## 测试覆盖

新增 Phase 9 测试覆盖：

- memory base / record / hit 类型
- JSONL store 行为、删除 tombstone、namespace 安全编码
- factory / exact retriever / writer mode
- tool_node 成功写入、跳过、异常隔离、SQLite schema 写入
- recall helper、memory_recall 事件、fail-closed 配置
- planner recall、fallback recall、disabled parity
- model prompt facts 注入
- CLI `--memory-mode`
- submission 默认 read-only memory
- RAG placeholder 边界

最终验证命令：

```powershell
$files = Get-ChildItem -LiteralPath tests -Filter 'test_phase9_*.py' | ForEach-Object { $_.FullName }; pytest @files -q
pytest tests/test_phase1_runstate.py tests/test_phase3_tool_node.py tests/test_phase3_model.py tests/test_phase4_planner_node.py tests/test_phase5_config.py tests/test_submission_config.py -q
```

最新结果：

- Phase 9：`74 passed, 26 warnings`
- 既有相关回归：`52 passed, 1 warning`

说明：Windows 环境下 pytest 结束后偶发 `pytest-current` 临时目录清理 `PermissionError`，出现在测试 pass summary 之后；测试进程退出码为 0。

## 安全边界

- 默认本地配置仍保持 `memory.mode=disabled`。
- 提交态默认 `read_only_dataset`，只读召回，不写入。
- 仅 `full` 模式允许 tool playbook 写入。
- `RunState` 只保存可 pickle 的 `MemoryHit`，不保存运行时句柄。
- 本分支不实现 corpus ingest、BM25、向量检索或完整 RAG 检索链路。

## 后续建议

- M4 单独设计 corpus RAG：ingest、chunking、index、retriever、prompt budget。
- 如需线上启用写入，建议先限定 memory path 与 namespace 清理策略。
- 可以补充一份运维说明，记录 memory artifacts 的保留、清理与迁移规则。
