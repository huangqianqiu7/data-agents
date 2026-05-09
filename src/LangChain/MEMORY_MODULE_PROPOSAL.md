# 记忆模块独立提案（从 LANGCHAIN_MIGRATION_PLAN.md §19 抽出）

## 版本说明

- **抽出时间**：v4 修订（在 v3 D9 / D10 基础上）
- **抽出原因**：现有 `data_agent_refactored` 无任何 memory 模块，§19 是从零起新增功能，不是迁移目标。混入 LangGraph 迁移会：
  1. 污染 parity 测试——跨 task 知识注入会改 LLM prompt，同 task 新旧版结果不同，无法分离「LangGraph bug」与「memory 注入影响」。
  2. 放大评测公平性争议——即使双层防御到位，只要能力存在就要 audit `artifacts/memory/` 内容。
  3. 推高 phase 数量 —— 原 Phase 4.5（1.5 天）+ 4.6（0.5 天）≈ 2 天不是迁移必要路径。
- **与主方案的关系**：LangGraph 迁移 parity 通过并稳定运行一段时间后，再独立评审本提案。主方案保留 `memory/working.py`（仅作为 `agents/context_manager.py` 的内部 refactor，不新增跨 task 能力）。
- **仅保留**：`memory/working.py`（working memory，单 run 内 scratchpad 重建 / pin 数据预览 / sanitize，跨 task 不共享）。

---

## 一、设计变更要点

v2 沿用了 MemGPT 风格的「working / episodic / semantic」三层划分，但对 **data agent** 不合身：
- v2 的 `episodic` 装「上一次 task 怎么答对了」，会把题目的解题思路与具体答案放进跨 task 命名空间，**评测公平性高度敏感**（用 namespace + 文本裁减绕过，但仍可泄露思路）。
- v2 的 `MemoryFacade` 把 working / episodic / semantic 三个角色塞进一个对象，违反 ISP，每个 graph node 都拿到了不需要的能力。
- v2 的 `MemoryStore.search` 强行要求所有后端实现 BM25/向量/namespace 过滤三合一，BM25 在 JSONL 上需要每次重建倒排，是错配。

**本提案的核心原则**：
1. **按数据语义命名**，不按时间生命周期：`working`（run 内）/ `dataset_knowledge`（数据集 schema 等结构化事实）/ `tool_playbook`（工具调用模板）/ `corpus`（外部语料 RAG）。
2. **写入层结构化强约束**：`dataset_knowledge` / `tool_playbook` 只能写预定义的 dataclass 字段，**禁止写自由文本字段（`question` / `approach` / `hint` / `summary`）**。
3. **运行时三态开关**：`memory.mode = disabled | read_only_dataset | full`，评测默认 `read_only_dataset`。
4. **`MemoryStore`（KV）与 `Retriever`（检索）解耦**：BM25 / 向量 / recency 是独立 `Retriever` 子类，订阅 store 写入维护索引；store 后端只负责 KV。
5. **不要 Facade**：按需注入 `Retriever` / `MemoryWriter`，每个 graph node 只看到自己需要的接口。
6. **不预埋 RAG 接口**：RAG 接口 emerge from need。

---

## 二、模块定位

`agents/context_manager.py` 的现有「工作记忆」职责保留（已在主方案 Phase 2.5 抽到 `memory/working.py`），跨 task 复用能力从「记忆 Facade」收紧为「数据集知识 + 工具 playbook」两类**结构化事实**：

| 子模块 | 装什么 | 跨 task 共享？ | 评测公平性影响 |
|---|---|---|---|
| `memory/working.py`（已在主方案） | 单 run 内的 scratchpad 重建 / pin 数据预览 / sanitize | 否 | 无 |
| `memory/dataset_knowledge.py` | 文件 schema（列名 / dtype）、行数估计、文件路径树（结构化字段） | 是（同 dataset） | 低（不存答案 / 思路） |
| `memory/tool_playbook.py` | 工具调用参数模板（不含具体行值），譬如「`read_csv` 在 dataset X 上 max_rows=20 即可看到表头」 | 是（同 dataset/工具） | 低（与具体题目解耦） |
| `memory/rag/`（占位） | 外部 corpus（人工录入文档片段） | 是 | 评估侧人工管控 |

**取消 v2 `episodic`**：跨 task 的「上次答案 + 解题思路」整体不进入记忆模块。如确需用于 dev 复盘，由 `artifacts/runs/{run_id}/{task_id}/trace.json` 自行整理，不进入运行时召回路径。

---

## 三、三层记忆模型

| 层 | namespace 模板 | 装什么 | 读写者 |
|---|---|---|---|
| **Run-Working**（已在主方案） | `run:{run_id}/task:{task_id}` | 当前 run 的 `StepRecord` 渲染、pinned previews | `model_node` / `parse_action_node` / `gate_node` |
| **Dataset-Knowledge** | `dataset:{name}` | `DatasetKnowledgeRecord`（结构化字段：file_path / schema / row_count_estimate / column_dtypes） | `tool_node` 写（仅在工具成功时）；`planner_node` / `model_node` 读 |
| **Tool-Playbook** | `dataset:{name}/tool:{tool_name}` | `ToolPlaybookRecord`（结构化字段：input_template / preconditions / typical_failures） | `tool_node` 写（仅在工具成功时）；`model_node` 读 |
| **Corpus**（占位） | `corpus:{name}` | 外部语料切片 | 离线 ingest；RAG 检索时读（不在本提案落地） |

**与 LangGraph Checkpointer 的区别**：

- Checkpointer（`SqliteSaver` / `MemorySaver`）是**状态恢复**，装的是完整 `RunState`，生命周期与 thread 绑定。
- Memory 是**语义检索载体**，装的是面向 LLM 重用的结构化记录，跨 thread 使用。两者互补，不互代。

---

## 四、核心抽象（Store / Retriever / Writer 解耦）

```python
# memory/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """通用 KV 信封；具体 record 类型由 records.py 提供。"""
    id: str
    namespace: str
    kind: Literal["working", "dataset_knowledge", "tool_playbook", "corpus"]
    payload: dict[str, Any]              # 结构化字段（dataclass.asdict 结果）
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    record: MemoryRecord
    score: float
    reason: str                          # 'bm25' / 'recency' / 'exact_namespace'


# ---- Store 与 Retriever 解耦 ----

@runtime_checkable
class MemoryStore(Protocol):
    """纯 KV 存储；不提供检索。"""
    def put(self, record: MemoryRecord) -> None: ...
    def get(self, namespace: str, record_id: str) -> MemoryRecord | None: ...
    def list(self, namespace: str, *, limit: int = 100) -> list[MemoryRecord]: ...
    def delete(self, namespace: str, record_id: str) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    """检索抽象；可订阅 store 的写入事件维护索引。"""
    def retrieve(
        self,
        query: str,
        *,
        namespace: str,
        k: int = 5,
    ) -> list[RetrievalResult]: ...


@runtime_checkable
class MemoryWriter(Protocol):
    """结构化写入器；约束写入字段，禁止写自由文本。"""
    def write_dataset_knowledge(
        self, dataset: str, record: "DatasetKnowledgeRecord"
    ) -> None: ...
    def write_tool_playbook(
        self, dataset: str, tool_name: str, record: "ToolPlaybookRecord"
    ) -> None: ...
```

**结构化 record 类型**（写入层强约束）：

```python
# memory/records.py
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class DatasetKnowledgeRecord:
    """允许写入的字段清单。
    DELIBERATELY excludes: question, approach, answer, hint, summary, freetext.
    """
    file_path: str                              # context 内相对路径
    file_kind: Literal["csv", "json", "doc", "sqlite", "other"]
    schema: dict[str, str]                       # column -> dtype 字符串
    row_count_estimate: int | None
    encoding: str | None = None
    sample_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolPlaybookRecord:
    """工具调用模板。"""
    tool_name: str
    input_template: dict[str, Any]               # 参数模板（譬如 {"max_rows": 20}）
    preconditions: list[str]                     # 可机器解析的条件，譬如 ["preview_done"]
    typical_failures: list[str] = field(default_factory=list)
    # DELIBERATELY excludes: example_question, example_answer
```

**`MemoryWriter` 实现**只接受这两类 dataclass：

```python
# memory/writers.py
class StoreBackedMemoryWriter(MemoryWriter):
    def __init__(self, store: MemoryStore, mode: Literal["disabled", "read_only_dataset", "full"]):
        self._store = store
        self._mode = mode

    def write_dataset_knowledge(self, dataset: str, record: DatasetKnowledgeRecord) -> None:
        if self._mode == "disabled":
            return
        # read_only_dataset 也允许写 dataset_knowledge（结构化、不含答案）
        self._store.put(MemoryRecord(
            id=f"dk:{dataset}:{record.file_path}",
            namespace=f"dataset:{dataset}",
            kind="dataset_knowledge",
            payload=dataclasses.asdict(record),
        ))

    def write_tool_playbook(self, dataset: str, tool_name: str, record: ToolPlaybookRecord) -> None:
        if self._mode == "disabled":
            return
        # read_only_dataset 默认禁止写 tool_playbook（避免 dev 时无意污染评测命名空间）
        if self._mode == "read_only_dataset":
            return
        self._store.put(MemoryRecord(
            id=f"tp:{dataset}:{tool_name}",
            namespace=f"dataset:{dataset}/tool:{tool_name}",
            kind="tool_playbook",
            payload=dataclasses.asdict(record),
        ))
```

**关键约束**：`MemoryWriter` 的接口**没有 `write_run_outcome` / `write_episodic`**。这是设计层面拒绝跨 task 答案 / 思路写入。

---

## 五、三态开关

```yaml
memory:
  mode: read_only_dataset             # disabled | read_only_dataset | full
  store_backend: jsonl                # jsonl | sqlite
  path: artifacts/memory
  retriever:
    type: exact                       # exact | recency （bm25 推迟到后续提案）
  retrieval_max_results: 5
```

| `mode` | 读 dataset_knowledge | 读 tool_playbook | 写 dataset_knowledge | 写 tool_playbook | 适用场景 |
|---|---|---|---|---|---|
| `disabled` | 否 | 否 | 否 | 否 | 完全无记忆 baseline |
| `read_only_dataset` | 是 | 是（如果有） | 是 | **否** | **评测默认**：只读 dataset 级知识，禁止写新 playbook 防污染 |
| `full` | 是 | 是 | 是 | 是 | 开发 / 数据探索期 |

**评测协议**：
- 默认 `mode: read_only_dataset`。
- 评测 run 启动前必须能 audit 现有 `artifacts/memory/` 内容；评测组织方决定是否清空。
- `--memory-mode disabled` CLI 开关用于完全消除记忆影响，对 baseline parity 测试必备。

---

## 六、`MemoryStore` 后端

只保留两种后端（从 v3 三种里删除 `LangGraphStoreAdapter` —— 标「实验性」等于现在不做）：

| 后端 | 文件 | 适用 |
|---|---|---|
| `JsonlMemoryStore` | `memory/stores/jsonl.py` | 默认；纯 KV append-only；运维零成本 |
| `SqliteMemoryStore` | `memory/stores/sqlite.py` | 共用 `langgraph-checkpoint-sqlite` 同款连接；批量评测可选 |

**关键约束**：所有后端都**只实现 KV 接口**，不实现搜索；搜索由 `Retriever` 子类提供。

---

## 七、`Retriever` 实现

MVP 只做 `ExactNamespaceRetriever`。BM25 推迟到后续提案（先证明 exact 不够用再升级）。

```python
# memory/retrievers/exact.py
class ExactNamespaceRetriever(Retriever):
    """完整列出 namespace 下所有 record，按 created_at 倒序。"""
    def __init__(self, store: MemoryStore):
        self._store = store

    def retrieve(self, query: str, *, namespace: str, k: int = 5) -> list[RetrievalResult]:
        records = self._store.list(namespace, limit=k)
        return [
            RetrievalResult(record=r, score=1.0, reason="exact_namespace")
            for r in records
        ]
```

**索引构建**仅在子进程入口 `build_retriever()` 中执行一次，不参与 LangGraph 节点循环。

---

## 八、节点接入点

每个 graph node 只接收自己需要的接口：

```python
# planner_node
def planner_node(state: RunState, config: RunnableConfig) -> dict[str, Any]:
    app_config = get_current_app_config()
    if app_config.memory.mode == "disabled":
        return _generate_plan_without_memory(state, app_config)

    retriever: Retriever = build_retriever(app_config)
    hits = retriever.retrieve(
        state["question"],
        namespace=f"dataset:{app_config.dataset.name}",
        k=app_config.memory.retrieval_max_results,
    )
    return _generate_plan_with_dataset_knowledge(state, hits, app_config)


# tool_node 在工具成功后写 dataset_knowledge
def tool_node(state, config):
    ...
    if result.ok and state["action"] == "read_csv":
        writer: MemoryWriter = build_writer(app_config)
        knowledge = _extract_dataset_knowledge_from_csv(result)
        writer.write_dataset_knowledge(app_config.dataset.name, knowledge)
    ...
```

**没有 `recall_for_execution`、`write_run_outcome` 这种通用接口**：每个写入点都明确知道自己写什么类型的 record。

---

## 九、跨 task 隔离与答案泄露（双层防御）

**写入层（不可绕过）**：
- `MemoryWriter` 只接受 `DatasetKnowledgeRecord` / `ToolPlaybookRecord` 两种 dataclass。
- 这两个 dataclass 的字段经过审计，不含 `question` / `answer` / `approach` / `hint` / 自由文本。
- 即使将来想加自由文本字段，也必须**修改 records.py**（被 PR review 看到）。

**运行时层**：
- 评测默认 `mode: read_only_dataset`，禁止写 `tool_playbook`。
- `recall` 仅返回结构化 payload，由 `prompts.py` 决定如何拼到 system prompt。
- 每次 `retriever.retrieve()` 调用都记录到 trace，通过主方案 §11.5 的 `dispatch_custom_event("memory_recall", ...)` 上报，评测方可离线 audit。

**审计示例**：

```json
// trace.json 或 metrics.json
"memory_recalls": [
  {
    "node": "planner_node",
    "namespace": "dataset:dabench-public",
    "k": 5,
    "hit_ids": ["dk:dabench-public:transactions.csv", ...],
    "reason": "exact_namespace"
  }
]
```

---

## 十、与 RunState 的集成

**不要**将 `MemoryStore` / `Retriever` / `MemoryWriter` 句柄放进 `RunState`（反主方案 §5.3 Picklability 约束）。

仅在 `RunState` 保留轻量引用：

```python
class RunState(TypedDict, total=False):
    ...
    memory_hits: Annotated[list[MemoryHit], operator.add]   # planner / model 召回结果（仅 summary）
```

```python
# memory/types.py
@dataclass(frozen=True, slots=True)
class MemoryHit:
    record_id: str
    namespace: str
    score: float
    summary: str          # 由 prompt 注入函数从 payload 渲染；不直接含原 dataclass
```

`MemoryStore` / `Retriever` / `MemoryWriter` 句柄由 `factory.build_*(app_config)` 在子进程入口现场构造（与主方案 §13.1 子进程 compile 同机制）。

---

## 十一、Prompt 注入约定

`agents/prompts.py` 提供**结构化模板**注入，不直接拼自由文本：

```text
SYSTEM
  ...
  ## Dataset facts (from prior runs)
  - File: transactions.csv  Columns: ['date', 'amount', 'merchant']  Rows~: 12450
  - File: customers.csv     Columns: ['id', 'name', 'tier']         Rows~: 3200
  ## Tool playbook (when applicable)
  - read_csv: typical max_rows=20 sufficient for header inspection
```

**严格要求模型「仅作参考」**，并与现有 `PLAN_AND_SOLVE_SYSTEM_PROMPT` 合并；prompt 注入函数对 `memory_hits` 做白名单字段渲染，绝不拼接 `payload` 中未列出的字段。

---

## 十二、测试点

| 测试 | 目标 |
|---|---|
| `JsonlMemoryStore.put/list` | KV 操作正确；namespace 隔离 |
| `ExactNamespaceRetriever.retrieve` | 按 created_at 倒序；limit 生效；空 namespace 返 `[]` |
| `MemoryWriter.write_dataset_knowledge` | 接受 dataclass；写入后可读 |
| `MemoryWriter` 字段约束 | 试图写 `{"question": "..."}` 应被 `TypeError` 拒绝（dataclass 强类型） |
| `mode=read_only_dataset` 评测协议 | 拒写 `tool_playbook`；允许写 `dataset_knowledge` |
| `mode=disabled` 评测协议 | 任何 `write_*` 调用 no-op；`retrieve` 返 `[]` |
| Picklability | `MemoryStore` / `Retriever` 句柄不会进入 `RunState` |
| memory_recall 事件 | 每次 `retrieve` 都通过 `dispatch_custom_event` 上报给 MetricsCollector |
| Prompt 字段白名单 | `payload` 中加入未列出字段，prompt 注入函数应忽略 |

---

## 十三、实施排期（本提案独立落地）

仅在本提案独立评审通过后启动：

- **Phase M1（1.5 天）**：实现 `JsonlMemoryStore` + `MemoryWriter` + `ExactNamespaceRetriever`；`tool_node` 在 `read_csv` / `read_json` / `inspect_sqlite_schema` 成功后写 `dataset_knowledge`；`planner_node` 入口召回。
- **Phase M2（0.5 天）**：`SqliteMemoryStore` 后端；yaml 加载；CLI `--memory-mode` 开关。
- **Phase M3（独立再提案）**：`BM25Retriever`（依赖 `rank_bm25`，先证明 exact 不够用才做）。
- **Phase M4（独立再提案，1-2 周）**：RAG 拓展。

**前置条件**：主方案 v4 LangGraph 迁移 parity 测试已稳定通过，且评测组织方书面同意启用跨 task 记忆能力。

---

## 十四、风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 跨 task 答案泄露 | 评测作弊 | **双层防御**：写入 dataclass 字段白名单 + `mode=read_only_dataset` 默认 + 每次 recall 通过 `dispatch_custom_event` 写 trace |
| 记忆质量老化 | 后期召回质量下降 | `policies.py` 加 TTL；`max_records_per_namespace` 倒序淘汰 |
| 评测场景要求零记忆 | 复现不一致 | `--memory-mode disabled` CLI 开关 |
| Picklability 被破坏 | SqliteSaver 任务丢 state | `MemoryStore` / `Retriever` / `MemoryWriter` 严禁进 `RunState`；`memory_hits` 仅存 dataclass |
| Dataset 路径冲突 | 多 dataset 共用 store | 强制 namespace `dataset:{name}` 前缀；`MemoryWriter` API 必传 `dataset` 参数 |
| v2 episodic 字段被引用（兼容性） | 旧脚本读不到 episodic | 升级 notes 显式说明 episodic 已删除；保留只读迁移工具读 v2 jsonl |
