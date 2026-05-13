# RAG 与记忆模块架构设计 v2（技术落地版）

> 版本：v2，2026-05-13
> 前置版本：`2026-05-12-rag-memory-html-demo-design.md`（HTML 演示页摘要版）
> 参考提案：`src/计划/记忆模块/MEMORY_MODULE_PROPOSAL.md`
> 适用代码基：`src/data_agent_langchain/`

## 一、背景与目标

### 1.1 为什么需要 v2

v1 文档（`2026-05-12-rag-memory-html-demo-design.md`）的定位是「为 HTML 演示页提供摘要型说明」，覆盖了三阶段路线和接入点的概念，但缺少：

- 模块边界、文件路径、数据结构的可执行规格。
- 与现有 `data_agent_langchain` 源码（`config.py`、`runtime/state.py`、`agents/*`、`memory/working.py`、`observability/events.py`）的精确对接关系。
- 阶段化的实施排期、测试矩阵、验收标准。
- 跨 task 答案泄露、`RunState` picklability、并发写入这些可执行的风险控制项。

v2 把 v1 升级为「可直接据此开发」的技术落地规格。HTML 演示页继续作为传达载体，但不再是设计的主出口。

### 1.2 目标

1. 在不破坏现有 LangGraph 后端 parity 的前提下，落地结构化 Memory MVP（`dataset_knowledge` + `tool_playbook`）。
2. 为后续 Corpus RAG 预留独立子包边界（`memory/rag/`），但**本期不实现**。
3. 设计层面强约束：禁止把 `question` / `answer` / `approach` / `hint` / `summary` 等自由文本写入跨 task 记忆。
4. 评测默认路径 `memory.mode=disabled` 与 `memory.mode=read_only_dataset` 必须可审计、可关闭、可复现。

### 1.3 非目标（YAGNI）

- 不实现 BM25 / 向量检索（延后到单独提案）。
- 不实现 episodic memory（v2 提案已设计层面拒绝）。
- 不让 LangGraph checkpointer 与 Memory 互相替代。
- 不修改 `RunState` 字段以外的图节点路由逻辑。

---

## 二、当前系统事实（设计依据）

本节列出 v2 设计所依赖的现有源码事实，避免后续实现与文档漂移。

### 2.1 入口与运行编排

- `src/data_agent_langchain/cli.py`：本地 `dabench-lc` CLI；读 YAML，进入 `runner`。
- `src/data_agent_langchain/submission.py`：容器提交入口；只读 `MODEL_*` env，构造提交态 `AppConfig`。
- `src/data_agent_langchain/run/runner.py`：单任务/批量/并发/子进程隔离/trace/metrics 落盘。
- `src/data_agent_langchain/config.py`：`AppConfig`（`dataset` / `agent` / `tools` / `run` / `observability` / `evaluation`），全部 `frozen=True, slots=True`，并通过 `to_dict` / `from_dict` 子进程序列化。
- `src/data_agent_langchain/llm/factory.py`：`build_chat_model(app_config)`，根据 gateway caps 绑定工具。

### 2.2 图与节点

- `agents/react_graph.py`：`START → execution → finalize`。
- `agents/plan_solve_graph.py`：`planner → execution → replanner? → finalize`。
- `agents/execution_subgraph.py`：共享 T-A-O 子图。
- 核心节点：`planner_node`、`model_node`、`parse_action`、`gate`、`tool_node`、`advance_node`、`finalize`。
- `runtime/state.py`：`RunState` `TypedDict(total=False)`；`steps` 用 `Annotated[list, operator.add]`；**字段必须可 pickle**；当前**没有** `memory_hits` 字段。
- `agents/prompts.py`：`build_react_messages` / `build_plan_solve_execution_messages` / `build_planning_messages`，把 `task` + `steps` 拼成 `BaseMessage` 列表；目前没有 memory 注入位点。
- `memory/working.py`：单 run scratchpad，按 token 预算裁剪、钉住数据预览。**这是 `memory/` 目录下目前唯一的实现**。

### 2.3 观测

- `observability/events.py`：`dispatch_observability_event(name, data, config)`，底层走 LangGraph `dispatch_custom_event`；contextvar 缺失时静默 no-op。MetricsCollector 通过 callbacks 链路接收。

### 2.4 现有约束

- `RunState` picklability 由 SqliteSaver / 子进程并发依赖；任何复杂对象（`Path`、`BaseTool`、`AppConfig` 实例）都**不能**放进 state。
- `runner` 在每个子进程入口现场构造工具与 LLM；Memory 句柄必须遵守同一协议。
- `gate.py` 强制 `read_csv` / `read_json` / `read_doc` / `inspect_sqlite_schema` 必须在计算前执行；Memory 命中**不应**绕过 gate。

---

## 三、设计原则

1. **按数据语义命名**：`working`（run 内）/ `dataset_knowledge`（数据集结构化事实）/ `tool_playbook`（工具调用模板）/ `corpus`（外部语料，占位）。
2. **写入层 dataclass 字段白名单**：禁止自由文本字段；新增字段必须改 `memory/records.py` 才能写入，PR review 兜底。
3. **store / retriever / writer 三层解耦**：`MemoryStore` 只做 KV，不做检索；`Retriever` 子类自行维护索引；`MemoryWriter` 只接受预定义 dataclass。
4. **三态运行开关**：`memory.mode = disabled | read_only_dataset | full`；评测默认 `read_only_dataset`。
5. **不放对象进 `RunState`**：句柄由 `factory.build_*(app_config)` 在节点入口现场构造；state 仅保留轻量 `memory_hits` summary。
6. **每次召回都进 audit**：复用 `dispatch_observability_event("memory_recall", ...)`，写入 trace / metrics。
7. **RAG 边界先画后做**：`memory/rag/` 子包目录保留，但 MVP 不落地；接口仅定义到 `Retriever` Protocol 即可。

---

## 四、目标架构

### 4.1 目录与新增文件

```
src/data_agent_langchain/memory/
  __init__.py                  # 现有；新增导出
  working.py                   # 现有；不动
  base.py                      # 新增：Protocol / MemoryRecord / RetrievalResult
  records.py                   # 新增：DatasetKnowledgeRecord / ToolPlaybookRecord
  types.py                     # 新增：MemoryHit（写入 RunState 用的轻量摘要）
  factory.py                   # 新增：build_store / build_retriever / build_writer
  stores/
    __init__.py
    jsonl.py                   # 新增：JsonlMemoryStore（默认）
    sqlite.py                  # 后续阶段
  retrievers/
    __init__.py
    exact.py                   # 新增：ExactNamespaceRetriever
  writers/
    __init__.py
    store_backed.py            # 新增：StoreBackedMemoryWriter
  rag/                         # 占位，本期不实现
    __init__.py                # 仅声明边界 + NotImplementedError 占位
```

`memory/working.py` **保持原样**；它只服务于单 run scratchpad，与跨 task 记忆模块分层。

### 4.2 `MemoryConfig` 接入 `AppConfig`

在 `src/data_agent_langchain/config.py` 新增一个 frozen dataclass，与现有 6 个子配置同级，保持 `to_dict` / `from_dict` 协议：

```python
@dataclass(frozen=True, slots=True)
class MemoryConfig:
    mode: str = "disabled"                     # disabled | read_only_dataset | full
    store_backend: str = "jsonl"               # jsonl | sqlite（sqlite 后续阶段）
    path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "memory"
    )
    retriever_type: str = "exact"              # exact（MVP 唯一选项）
    retrieval_max_results: int = 5
```

`AppConfig` 增加字段 `memory: MemoryConfig = field(default_factory=MemoryConfig)`，并在 `to_dict` / `from_dict` 中加入对应分支。所有字段保持 picklable。

评测路径默认值（在 `submission.build_submission_config` 中显式设置而非依赖 dataclass 默认）：

- 评测：`mode = "read_only_dataset"`（或 `disabled`，由评测方决定）。
- 本地开发：YAML 显式开 `full`。

### 4.3 核心抽象（`memory/base.py`）

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    namespace: str
    kind: Literal["working", "dataset_knowledge", "tool_playbook", "corpus"]
    payload: dict[str, Any]                 # dataclasses.asdict 结果，纯基本类型
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    record: MemoryRecord
    score: float
    reason: str                              # "exact_namespace" | "recency" | "bm25" | ...


@runtime_checkable
class MemoryStore(Protocol):
    def put(self, record: MemoryRecord) -> None: ...
    def get(self, namespace: str, record_id: str) -> MemoryRecord | None: ...
    def list(self, namespace: str, *, limit: int = 100) -> list[MemoryRecord]: ...
    def delete(self, namespace: str, record_id: str) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self, query: str, *, namespace: str, k: int = 5
    ) -> list[RetrievalResult]: ...


@runtime_checkable
class MemoryWriter(Protocol):
    def write_dataset_knowledge(
        self, dataset: str, record: "DatasetKnowledgeRecord"
    ) -> None: ...
    def write_tool_playbook(
        self, dataset: str, tool_name: str, record: "ToolPlaybookRecord"
    ) -> None: ...
```

**关键约束**：`MemoryWriter` 接口**没有** `write_run_outcome` / `write_episodic` / `write_freetext`。任何把 question/answer/思路写入跨 task 命名空间的尝试，在接口层就被拒绝。

### 4.4 结构化 record 类型（`memory/records.py`）

```python
@dataclass(frozen=True, slots=True)
class DatasetKnowledgeRecord:
    """允许写入的字段清单。
    DELIBERATELY excludes: question, approach, answer, hint, summary, freetext.
    """
    file_path: str                                          # context 内相对路径
    file_kind: Literal["csv", "json", "doc", "sqlite", "other"]
    schema: dict[str, str]                                  # column -> dtype 字符串
    row_count_estimate: int | None
    encoding: str | None = None
    sample_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolPlaybookRecord:
    tool_name: str
    input_template: dict[str, Any]                          # 譬如 {"max_rows": 20}
    preconditions: list[str]                                # 譬如 ["preview_done"]
    typical_failures: list[str] = field(default_factory=list)
    # DELIBERATELY excludes: example_question, example_answer
```

### 4.5 `JsonlMemoryStore`（`memory/stores/jsonl.py`）

- append-only JSONL；按 namespace 分文件：`{path}/{namespace_safe}.jsonl`。
- `namespace_safe` 把 `:` / `/` 等替换为 `__`，避免路径越界。
- `put` 用 `open(..., "a", encoding="utf-8")` + `fcntl`/`msvcrt` 单进程锁；并发由 runner 子进程串行写入 + 文件级 advisory lock 保障。
- `list(namespace, limit)`：逐行解码，按 `created_at` 倒序取前 `limit` 条；空文件返回 `[]`。
- 删除：标记 tombstone 行（`{"_tombstone": id}`），读时过滤。

记录序列化由 `dataclasses.asdict` + `json.dumps(..., ensure_ascii=False, default=str)` 完成（`datetime` 用 `isoformat`）。

### 4.6 `ExactNamespaceRetriever`（`memory/retrievers/exact.py`）

```python
class ExactNamespaceRetriever:
    def __init__(self, store: MemoryStore):
        self._store = store

    def retrieve(self, query: str, *, namespace: str, k: int = 5) -> list[RetrievalResult]:
        records = self._store.list(namespace, limit=k)
        return [
            RetrievalResult(record=r, score=1.0, reason="exact_namespace")
            for r in records
        ]
```

- `query` 参数保留是为对齐 `Retriever` Protocol；MVP 不使用。
- BM25 / 向量召回延后到独立提案。

### 4.7 `StoreBackedMemoryWriter`（`memory/writers/store_backed.py`）

```python
class StoreBackedMemoryWriter:
    def __init__(self, store: MemoryStore, mode: str):
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
        if self._mode in {"disabled", "read_only_dataset"}:
            return
        self._store.put(MemoryRecord(
            id=f"tp:{dataset}:{tool_name}",
            namespace=f"dataset:{dataset}/tool:{tool_name}",
            kind="tool_playbook",
            payload=dataclasses.asdict(record),
        ))
```

### 4.8 工厂（`memory/factory.py`）

```python
def build_store(cfg: MemoryConfig) -> MemoryStore: ...
def build_retriever(cfg: MemoryConfig, store: MemoryStore) -> Retriever: ...
def build_writer(cfg: MemoryConfig, store: MemoryStore) -> MemoryWriter: ...
```

- 工厂只在子进程入口（`runner` 构建图之后、节点执行之前）或节点入口现场调用。
- 工厂不缓存到 `RunState`；可以在 `runtime/context.py` 用 contextvar 提供 per-run 单例（与现有 `get_current_app_config` 同机制）。

### 4.9 `RunState` 扩展（`runtime/state.py`）

仅新增一个轻量字段，**不放任何句柄**：

```python
class RunState(TypedDict, total=False):
    ...
    memory_hits: Annotated[list[MemoryHit], operator.add]
```

```python
# memory/types.py
@dataclass(frozen=True, slots=True)
class MemoryHit:
    record_id: str
    namespace: str
    score: float
    summary: str          # 由 prompt 渲染函数从 payload 白名单生成
```

`MemoryHit` 仅供 trace audit 与可选 prompt 渲染；不直接含原始 payload。

### 4.10 Prompt 注入（`agents/prompts.py`）

新增一个白名单渲染函数，**严格基于字段名枚举**：

```python
def render_dataset_facts(hits: list[MemoryHit]) -> str:
    """白名单渲染：只输出允许字段，忽略 payload 中任何未列出的字段。"""
    allowed = ("file_path", "file_kind", "schema", "row_count_estimate", "sample_columns")
    lines: list[str] = []
    for hit in hits:
        ...  # 仅取 allowed 字段做单行渲染
    return "\n".join(lines)
```

`build_planning_messages` / `build_plan_solve_execution_messages` / `build_react_messages` 在已有 system + user 消息后追加一段 `## Dataset facts (from prior runs)`，文本上明确「仅作参考，不替代真实数据预览」。

`render_dataset_facts` 与 `MemoryHit` 是唯一允许把 memory 内容拼进 prompt 的路径；任何其它直接读 `payload` 拼字符串都视为违规。

### 4.11 RAG 边界（仅画界，本期不实现）

- `memory/rag/__init__.py` 抛 `NotImplementedError("Corpus RAG is reserved for Phase M4")`。
- 接口仍复用同一组 Protocol（`MemoryStore` / `Retriever`）；corpus 使用独立 namespace `corpus:{name}`。
- 索引构建必须在离线 ingest 或子进程入口完成，**禁止**进入 graph node 循环。

---

## 五、读写数据流

### 5.1 写入路径（`tool_node`）

只在工具成功后写，且写入点严格限定：

| Action | 来源字段 | 写入 record |
|---|---|---|
| `read_csv` ok | `result.content`（columns / dtype / row_count_estimate） | `DatasetKnowledgeRecord(file_kind="csv")` |
| `read_json` ok | 顶层 schema 推断 | `DatasetKnowledgeRecord(file_kind="json")` |
| `inspect_sqlite_schema` ok | 表名 + 列定义 | `DatasetKnowledgeRecord(file_kind="sqlite")` |
| `read_doc` ok | 文件元数据 | `DatasetKnowledgeRecord(file_kind="doc")` |
| 任意工具 ok（仅 `full` 模式） | 成功参数模板 | `ToolPlaybookRecord` |

实现要点：

- 仅当 `result.ok and not skip_tool and last_error_kind is None` 才进入写入分支。
- `tool_node` 不直接构造 store；调用 `factory.build_writer(app_config.memory, store=factory.build_store(...))`，或通过 contextvar 拿到 per-run writer。
- `disabled` 模式下 writer 全部 no-op，不依赖调用方判断。
- 解析失败不抛异常进入主路径；用 try/except 包住并发 `dispatch_observability_event("memory_write_skipped", {...})` 上报。

### 5.2 召回路径（`planner_node` / `model_node`）

- 入口判断 `app_config.memory.mode`：`disabled` → 直接退化为现有行为，不调 retriever。
- 非 disabled：构造 retriever，按 `namespace=f"dataset:{dataset_name}"` 召回 ≤ `retrieval_max_results` 条。
- 渲染 `MemoryHit` 列表后追加到 prompt；同时把 `memory_hits` 写回 partial state，供 audit 使用。
- 每次召回触发 `dispatch_observability_event("memory_recall", {"node": ..., "namespace": ..., "k": ..., "hit_ids": [...], "reason": ...}, config)`。

### 5.3 与 gate 的关系

- Memory 命中**不**置 `preview_done = True`。
- gate 的 `discovery_done` / `preview_done` 仍由真实 `list_context` / 数据预览工具更新。
- Memory 只用于减少重复探索的成本，不可作为预览替代。

### 5.4 与 LangGraph checkpointer 的关系

- Checkpointer 序列化完整 `RunState`，绑定 thread；Memory 服务跨 thread 的语义复用。
- 两者各自生命周期，**不互相替代**；测试中要确认 `SqliteSaver` 恢复后 `memory_hits` 字段能被 reducer 合并（list + `operator.add`）。

---

## 六、配置示例

```yaml
# configs/dev.yaml
memory:
  mode: full
  store_backend: jsonl
  path: artifacts/memory
  retriever_type: exact
  retrieval_max_results: 5
```

```yaml
# configs/eval.yaml
memory:
  mode: read_only_dataset
  store_backend: jsonl
  path: artifacts/memory
  retriever_type: exact
  retrieval_max_results: 5
```

CLI 开关：`dabench-lc --memory-mode disabled` 用于 baseline parity 测试，覆盖 YAML。

---

## 七、实施阶段与排期

| 阶段 | 工作量 | 交付 | 前置 |
|---|---|---|---|
| **M0：协议落地** | 0.5 天 | `memory/base.py` + `memory/records.py` + `memory/types.py` 与单元测试 | 无 |
| **M1：JSONL 后端 + Writer + Retriever** | 1 天 | `stores/jsonl.py`、`writers/store_backed.py`、`retrievers/exact.py`、`factory.py`、`MemoryConfig` 接入 `AppConfig` | M0 |
| **M2：节点接入** | 1 天 | `tool_node` 写、`planner_node` / `model_node` 召回、`prompts.py` 白名单渲染、`memory_recall` 事件 | M1 |
| **M3：评测协议 & CLI** | 0.5 天 | `submission.py` 默认值、`cli.py --memory-mode`、`disabled` 全链路冒烟 | M2 |
| **M4（独立提案）** | TBD | `memory/rag/` corpus ingest、`SqliteMemoryStore`、BM25 / 向量检索 | M3 稳定后再单独评审 |

总 MVP 工作量约 **3 天**，不含 M4。

前置条件：

- LangGraph 后端 parity 测试稳定通过。
- 评测组织方对「跨 task 结构化记忆」的可审计性书面同意。

---

## 八、测试与验收

### 8.1 单元测试

| 测试 | 文件 | 验证点 |
|---|---|---|
| `test_jsonl_store_put_list` | `tests/memory/test_jsonl_store.py` | 写后可读；namespace 隔离；`limit` 生效；空 namespace 返 `[]` |
| `test_jsonl_store_concurrent_append` | 同上 | 多进程 append 不损坏 JSONL 行 |
| `test_exact_retriever_order_limit` | `tests/memory/test_retriever_exact.py` | 倒序；`k` 生效；reason 字符串正确 |
| `test_writer_dataclass_only` | `tests/memory/test_writer.py` | 传非 dataclass 拒收（TypeError）；自由文本字段在 dataclass 定义层缺失 |
| `test_writer_mode_disabled` | 同上 | 任何 write 调用 no-op |
| `test_writer_mode_read_only_dataset` | 同上 | 允许写 `dataset_knowledge`，拒写 `tool_playbook` |
| `test_memory_config_round_trip` | `tests/test_config.py` | `to_dict` / `from_dict` 后字段一致；评测约束不破坏 |
| `test_runstate_picklable_with_memory_hits` | `tests/runtime/test_state.py` | `RunState` 含 `memory_hits` 仍可 pickle |
| `test_prompts_dataset_facts_whitelist` | `tests/agents/test_prompts.py` | payload 含非白名单字段时被忽略；不出现在最终 prompt |

### 8.2 集成测试

- `test_tool_node_writes_dataset_knowledge`：fake `read_csv` 工具成功后，writer 收到 `DatasetKnowledgeRecord`。
- `test_planner_recalls_dataset_facts`：fake store 预置一条记录，planner 提示词出现对应 `file_path`。
- `test_memory_recall_event_dispatched`：MetricsCollector callback 收到 `memory_recall` 自定义事件，字段齐全。
- `test_disabled_mode_parity`：相同任务在 `disabled` 与禁用 memory 代码路径下产生**逐字节相同** trace（reproducible 验证）。

### 8.3 验收标准

- v1 文档列出的「HTML 演示页可在浏览器打开」继续成立；演示页内容与 v2 的模块命名一致。
- `memory.mode=disabled` 在评测路径下是 zero-cost、zero-risk 的默认开关。
- `read_only_dataset` 模式下：可读 `dataset_knowledge`；写 `tool_playbook` 被运行时拒绝；trace 中所有 `memory_recall` 事件可离线导出。
- `RunState` picklability 测试覆盖 `memory_hits` 字段。
- 所有写入操作可通过审计 trace 重建：`(namespace, record_id, kind, dataset)` 四元组必现。

---

## 九、风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 跨 task 答案泄露 | 评测作弊 | 写入 dataclass 字段白名单 + `mode=read_only_dataset` 默认 + 每次 recall 走 `dispatch_observability_event` |
| `RunState` picklability 被破坏 | SqliteSaver / 子进程丢 state | `MemoryStore` / `Retriever` / `MemoryWriter` 禁入 state；只允 `MemoryHit` |
| 召回污染 prompt | 模型被错误事实误导 | `render_dataset_facts` 仅白名单渲染；prompt 文案声明「仅作参考」 |
| Memory 替代 gate | 模型未读数据就直接计算 | gate 字段仅由真实数据预览工具更新；memory 命中**不**置 `preview_done` |
| 并发写入 JSONL 损坏 | 行半写 / 编码错误 | 文件级 advisory lock + append-only 模式 + 写后 fsync；测试覆盖多进程 append |
| Memory 质量老化 | 召回质量下降 | 后续提案中加 TTL / `max_records_per_namespace` 倒序淘汰策略 |
| RAG 边界被误用 | 用户尝试在 MVP 期写 corpus | `memory/rag/__init__.py` 抛 `NotImplementedError`，并在 PR 模板提醒 |
| 评测要求零记忆 | 复现不一致 | `--memory-mode disabled` CLI 开关；评测默认即可关闭 |

---

## 十、与 v1 文档的差异

| 维度 | v1（2026-05-12） | v2（2026-05-13） |
|---|---|---|
| 定位 | HTML 演示页摘要 | 可实施的技术规格 |
| 目录结构 | 未给出 | 给出 `memory/` 完整目录与新增文件 |
| 配置 | 仅口头提及 `memory.mode` | 完整 `MemoryConfig` dataclass 接入 `AppConfig` |
| Record 类型 | 提及名称 | 给出 dataclass 字段与白名单 |
| `RunState` | 不动 | 新增 `memory_hits: Annotated[list[MemoryHit], operator.add]` |
| Prompt 注入 | 提及白名单 | 给出 `render_dataset_facts` 函数边界 |
| 测试 | 列概要 | 给出文件路径与验证点矩阵 |
| 排期 | 路线图 | M0–M3 工作量明确；M4 单独提案 |
| RAG | 与 Memory 同层叙述 | 仅画接口边界 + `NotImplementedError` 占位 |

---

## 十一、后续动作

1. 你审阅本设计；如同意，下一步通过 `writing-plans` 技能产出 `2026-05-13-rag-memory-architecture-plan-v2.md`，把 M0–M3 拆成 2–5 分钟级 task。
2. HTML 演示页（`architecture-rag-memory-demo.html`）的模块说明与本文件保持术语一致；下一次更新 HTML 时按 v2 命名对齐。
3. M4（Corpus RAG）单独提案，前置条件是 M3 稳定 + 评测方书面同意。
