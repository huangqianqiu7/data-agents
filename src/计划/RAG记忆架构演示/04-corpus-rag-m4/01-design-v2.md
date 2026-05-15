# Corpus RAG 架构设计 v3.1（M4 落地版，v2 修订）

> 版本：v3.1，2026-05-14
> 前置版本：`01-design.md`（v3 / 2026-05-14）
> 修订动机：v3 在与代码基交叉核对时发现多处事实性偏差与设计选择歧义。
> 本版按 A2 + B2 + C2 + D2 决策落定：
> - **A2**：corpus 只在 task 入口召回**一次**（plan-solve 走 `planner_node`，ReAct 走新增入口节点），不每步召回；
> - **B2**：corpus store 走独立 `CorpusStore` mini Protocol，不强行兼容 v2 `MemoryStore.put/get/list/delete`；
> - **C2**：`sentence-transformers` / `chromadb` 仅进 `[project.optional-dependencies].rag`，不污染最小提交镜像；
> - **D2**：`shared_corpus` 完整工作（持久化、ingest CLI、镜像打包）拆出 `05-shared-corpus-design.md` 独立提案；本版仅保留 dataclass 字段占位 + fail-closed 行为。

适用代码基：`src/data_agent_langchain/`。

---

## 〇、与 v1 设计的差异（ADR）

| ID | v1 主张 | v2 改为 | 依据 |
|---|---|---|---|
| D1 | corpus 在 `planner_node` 与 `model_node` **都**召回 | 仅在 task 入口召回**一次**：plan-solve 走 `planner_node`，ReAct 在 `execution` 子图前加 `task_entry_node` | `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/model_node.py:165-169` 显示 v2 dataset_facts 仅在 model_node **消费**已写入的 hits；与 v1 描述不符。每步召回会让 `memory_hits` 单调膨胀并撞 `agent.max_context_tokens` |
| D2 | `ChromaCorpusStore` 同时实现 v2 `MemoryStore.put/get/list/delete` + corpus 专属接口 | `CorpusStore` 独立 Protocol（仅 `upsert_chunks` / `query_by_vector` / `close`）；`VectorCorpusRetriever` 仍实现 v2 `Retriever` Protocol | 写路径单一；不再把 chroma 存储模型挤进 KV 抽象。`MemoryRecord.kind="corpus"` 仍是 retriever 返回值的类型标识（已存在于 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/memory/base.py:9`） |
| D3 | `sentence-transformers` / `chromadb` 进 `[project.dependencies]` | 进 `[project.optional-dependencies].rag` | `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/pyproject.toml:16-35` 注释明确 `[project.dependencies]` 是"实际运行最小集"。所有 rag 重依赖在 rag 子包内**方法级延迟 import** |
| D4 | `shared_corpus` + `ingest_cli` 作为 M4.6 在主提案内 | 拆出独立提案 `05-shared-corpus-design.md`；本版 `CorpusRagConfig.shared_corpus=True` 时 fail-closed | 评测打包路径与 chromadb 版本兼容是单独问题，不应阻塞核心 ephemeral 路径 |
| D5 | 改 `build_planning_messages` / `build_*_messages` 三个 builder 注入 corpus snippets | **不动 builder**；在 `model_node._build_messages_for_state` 内 `render_dataset_facts(hits) + render_corpus_snippets(hits)` 合并成一段 HumanMessage 拼到 builder 输出后 | 与 v2 dataset_facts 注入路径（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/model_node.py:165-169`）保持一致；减少 diff |
| D6 | §4.6 `HarrierEmbedder` 示例 | 修复：`self._model_id = cfg.embedder_model_id` 必须赋值；`model_kwargs={"dtype": dtype}` 始终传，包括 `"auto"`（HF model card 推荐用法） | HF 实测：`SentenceTransformer("microsoft/harrier-oss-v1-270m", model_kwargs={"dtype": "auto"})` 是官方写法 |
| D7 | §4.3 `MemoryConfig.path = field(...)` 省略 default_factory | 完整保留 `default_factory=lambda: PROJECT_ROOT / "artifacts" / "memory"` | 防止抄袭误把 path 改成必填 |
| D8 | `task_input_dir` 取值未明确 | 明确：`task_input_dir = task.context_dir`（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/benchmark/schema.py` `PublicTask.context_dir`），**不**扫 `task_dir`（含 `expected_output.json`） | 多层防御：Redactor 之外再加目录边界 |
| D9 | §4.2 `_dataclass_from_dict` 仅说 `typing.get_type_hints` | 增加对 `tuple[str, ...]` / `Optional[Path]` / `Literal[...]` 等 v3 实际字段类型的处理与单测 | `CorpusRagConfig.shared_collections: tuple[str, ...]`、`embedder_cache_dir: Path \| None`、`embedder_device: Literal[...]` 都需要覆盖 |
| D10 | `namespace_safe = ns.replace(":", "__").replace("/", "__")` | 改为 `hashlib.sha1(ns.encode()).hexdigest()[:16]` + 前缀 | 避免 `a:b` / `a__b` collision；chroma collection name 字符集限制更严，sha1 保险 |
| D11 | 缺"启动期 import 边界"约束 | 明确：`memory/rag/__init__.py` 与 rag 子模块**顶层** import 不允许出现 `torch` / `chromadb` / `sentence_transformers`；用回归测试守护 | 让 `rag.enabled=false` / `RAG extra` 未安装时启动 0 副作用 |

---

## 一、背景与目标

### 1.1 为什么要做 M4

v2 的 Memory MVP（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/memory/`）已经能复用 dataset 字段、schema 这类**结构化事实**，但无法覆盖：

- 长篇 README、数据字典、领域知识这类**非结构化文本**；
- 评测时只读、需要语义检索的**任务文档**。

v2 §4.11 已留 `memory/rag/` 占位（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/memory/rag/__init__.py:1-10`，抛 `NotImplementedError`）。M4 在该边界内落地 **离线、向量化、本地嵌入** 的 corpus RAG。

### 1.2 目标

1. 在不破坏 `memory.mode=disabled` baseline parity 的前提下，落地基于 **向量嵌入 + 向量数据库** 的 corpus RAG。
2. 沿用 v2 的 `Retriever` Protocol、`MemoryConfig`、`MemoryHit`、`memory_recall` 事件、`render_*_facts` 白名单注入。
3. 嵌入模型本地推理：**`microsoft/harrier-oss-v1-270m`**，与评测使用同一份权重，**不依赖任何远程 embedding API**。
4. 向量库使用 **`chromadb`** 本地 ephemeral 客户端（M4 阶段不引入 persistent；persistent + shared 由独立提案处理）。
5. **task 入口召回一次**，索引构建在 runner 子进程入口完成；图节点循环内只允许消费已有 hits。
6. 严格遵守 v2 已有的安全边界：禁止 question / answer / approach / hint 进入任何持久化结构。

### 1.3 非目标（YAGNI）

- 不实现 episodic memory、跨 task answer 记忆。
- 不在评测路径写入 corpus 数据。
- 不引入除 `chromadb` / `sentence-transformers` / `transformers` / `torch` 外的新向量库 / embedding 框架。
- 不做 reranker、cross-encoder、LLM-based rerank。
- 不做混合检索（BM25 + vector）。
- **不**实现 shared_corpus 的 ingest CLI、persistent backend 与镜像打包（拆到独立提案）。
- **不**每步召回（A2 决策）。

---

## 二、当前系统事实（v2 已落地）

### 2.1 v2 已有抽象（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/memory/`）

- `base.py`：`MemoryRecord(kind=Literal["working","dataset_knowledge","tool_playbook","corpus"])`、`RetrievalResult`、`MemoryStore` / `Retriever` / `MemoryWriter` Protocol。`"corpus"` literal **已存在**，但当前没有任何 store 写入 `kind="corpus"`。
- `records.py`：`DatasetKnowledgeRecord`、`ToolPlaybookRecord`（白名单字段）。
- `types.py`：`MemoryHit`（写入 `RunState`）。
- `stores/jsonl.py`：`JsonlMemoryStore`，append-only JSONL，按 namespace 分文件。
- `retrievers/exact.py`：`ExactNamespaceRetriever`。
- `writers/store_backed.py`：`StoreBackedMemoryWriter`，按 mode 控制。
- `factory.py`：集中构建 store / retriever / writer。
- `working.py`：单 run scratchpad，**不动**。
- `rag/__init__.py`：占位，抛 `NotImplementedError("Corpus RAG is reserved for Phase M4")`。

### 2.2 v2 已有配置（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/config.py:126-135`）

```python
@dataclass(frozen=True, slots=True)
class MemoryConfig:
    mode: str = "disabled"
    store_backend: str = "jsonl"
    path: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "memory")
    retriever_type: str = "exact"
    retrieval_max_results: int = 5
```

`AppConfig.from_dict` 用扁平的 `_dataclass_from_dict`（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/config.py:318-330`），**不递归**进嵌套 dataclass。M4 必须先解决（见 §4.2）。

### 2.3 v2 已有写入与召回路径

- `tool_node` 成功后写 `DatasetKnowledgeRecord` / `ToolPlaybookRecord`。
- `agents/memory_recall.py` 的 `recall_dataset_facts` **仅**在 `planner_node` 入口召回（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/planner_node.py:54-64`）。
- `model_node` **不**主动召回；它只消费 `state["memory_hits"]` 并通过 `render_dataset_facts(hits)` 拼成 HumanMessage 拼到 builder 输出末尾（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/model_node.py:165-169`）。
- ReAct 模式没有 planner（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/react_graph.py`：`START → execution → finalize`），所以**目前 v2 在 ReAct 模式下根本不召回 dataset facts**。这是 v2 的一个 gap，v3 一并修复。
- `observability.events.dispatch_observability_event("memory_recall", ...)` 审计每次召回。
- `RunState.memory_hits: Annotated[list[MemoryHit], operator.add]`，仅含 `summary`（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/runtime/state.py:96-98`）。
- `submission.build_submission_config` 默认 `memory.mode=read_only_dataset`（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/submission.py:166`）。

### 2.4 v2 已有图结构

- `react_graph.py`：`START → execution(subgraph: model→parse→gate→tool→advance) → finalize`。
- `plan_solve_graph.py`：`START → planner → execution → ... → finalize`。
- 两图共用 `execution_subgraph`（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/execution_subgraph.py`）。

### 2.5 v2 已有提交镜像最小集（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/pyproject.toml:20-35`）

```toml
dependencies = [
    "langchain-core>=0.3.84,<0.4",
    "langchain-openai>=0.3.35,<0.4",
    "langgraph>=0.4.10,<0.5",
    "json-repair>=0.30,<1",
    "openai>=2.31,<3",
    "pandas>=3.0,<4",
    "pydantic>=2.12,<3",
    "pyyaml>=6.0.3,<7",
    "typer>=0.24,<1",
    "numpy>=2.4,<3",
]
```

明确注释："任何 baseline 专属大包都已下放到 `[project.optional-dependencies].baseline`"。M4 沿用此 convention 把 rag 大包下放到 `.rag`。

---

## 三、设计原则

1. **复用 v2 抽象**：`Retriever` Protocol 不变；`MemoryRecord(kind="corpus")` 仅作为 retriever 返回值的类型标识，不强制 corpus 存储走 v2 `MemoryStore.put/get/list/delete`。
2. **embedding 与 store 严格解耦**：`Embedder` 是独立 Protocol，`ChromaCorpusStore` 只持有客户端句柄，不知道嵌入模型是谁。
3. **单次召回，单次注入**（A2）：task 入口召回一次，写满 `state["memory_hits"]`；图节点循环内只消费。
4. **索引前置、检索内联**：所有 ingest / chunking / embedding 在子进程入口完成；图节点只读检索。
5. **白名单内容过滤**：v2 的写入禁令同样适用 corpus 文档；`Redactor` 在 chunking 之前过滤。
6. **fail closed**：`memory.mode=disabled` 时 RAG 强制关闭；`shared_corpus=True` 但 §独立提案未实现 → `memory_rag_skipped(reason="shared_corpus_not_implemented")`；任何 RAG 异常都退化到无 RAG 行为。
7. **picklability 不变**：`RunState` 只增加 `memory_hits` 已有字段中的 corpus hit；不放 store / retriever / embedder 句柄。
8. **启动期 import 边界**（新增）：`memory/rag/__init__.py` 与 rag 子模块顶层 import **不得**出现 `torch` / `chromadb` / `sentence_transformers`；任何重依赖都走方法级延迟 import。
9. **依赖最小化**（C2）：`sentence-transformers` / `chromadb` 仅进 `[project.optional-dependencies].rag`；提交镜像不带 RAG 时体积零增量。
10. **观测复用 v2**：所有召回都走 `memory_recall` 事件；新增 `memory_rag_index_built` / `memory_rag_skipped` 两个事件用于运维审计。
11. **代码注释中文化**（新增）：所有实施代码的 docstring 与行内 `#` 注释一律用中文，与现有 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/config.py` 等模块的风格一致。专有名词（`HF` / `cosine` / `ephemeral` / `chromadb` 等）保留原文。详细规则见 `02-implementation-plan-v2.md` 的 Conventions 段。

---

## 四、目标架构

### 4.1 目录与新增文件

在现有 `memory/` 下展开 `rag/` 子包：

```
src/data_agent_langchain/memory/
  rag/
    __init__.py                  # 由占位改为正式导出（仅纯类型，不 import 重依赖）
    documents.py                 # 新增：CorpusDocument / CorpusChunk
    loader.py                    # 新增：从 task context_dir 扫描 + 解析
    chunker.py                   # 新增：char-window 切片
    redactor.py                  # 新增：内容过滤
    base.py                      # 新增：CorpusStore Protocol（独立于 v2 MemoryStore）
    embedders/
      __init__.py
      base.py                    # 新增：Embedder Protocol
      sentence_transformer.py    # 新增：HarrierEmbedder（方法级延迟 import）
      stub.py                    # 新增：DeterministicStubEmbedder（测试用）
    stores/
      __init__.py
      chroma.py                  # 新增：ChromaCorpusStore.ephemeral（方法级延迟 import）
    retrievers/
      __init__.py
      vector.py                  # 新增：VectorCorpusRetriever
    factory.py                   # 新增：build_embedder / build_task_corpus
```

**不新增**（D2 决策）：

- `stores/chroma.py` 中的 `persistent_readonly` / `persistent_writable`
- `ingest_cli.py`
- `assets/corpus_shared/`

这些拆到 `05-shared-corpus-design.md`。

新增模块约定：

- 所有 dataclass 保持 `frozen=True, slots=True`，picklable。
- 所有 Protocol 用 `@runtime_checkable`，与 v2 一致。
- 不引入对 `langchain` 高层 RAG 抽象的依赖；只用 `langchain-core` 的 message / config 类型。
- **重依赖（torch / chromadb / sentence_transformers）只在 `embedders/sentence_transformer.py` 与 `stores/chroma.py` 的方法体内 import**，模块顶层不出现。

### 4.2 前置任务：让 `_dataclass_from_dict` 支持嵌套（含 Optional / tuple / Literal）

v3 的 `MemoryConfig` 需要嵌套 `CorpusRagConfig`，但当前 `_dataclass_from_dict`（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/config.py:318-330`）不递归。M4.0 必须先：

```python
# config.py
import typing
from typing import get_args, get_origin

def _dataclass_from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    type_hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in payload:
            continue
        value = payload[field_info.name]
        hint = type_hints[field_info.name]
        kwargs[field_info.name] = _coerce_field(field_info.name, hint, value)
    return cls(**kwargs)


def _coerce_field(name: str, hint: Any, value: Any) -> Any:
    # 1) 显式 Path 字段
    if name in {"root_path", "output_dir", "gateway_caps_path", "path", "shared_path", "embedder_cache_dir"}:
        return Path(value) if value is not None else None

    # 2) 显式 tuple[float, ...] 字段
    if name == "model_retry_backoff":
        return tuple(float(item) for item in value)

    # 3) 嵌套 dataclass
    if is_dataclass(hint) and isinstance(value, dict):
        return _dataclass_from_dict(hint, value)

    # 4) tuple[str, ...] / tuple[T, ...] 通用支持（CorpusRagConfig.shared_collections / redact_patterns / redact_filenames）
    if get_origin(hint) is tuple and isinstance(value, (list, tuple)):
        return tuple(value)

    # 5) Literal / Optional / 基本类型：直接返回
    return value
```

**注意**：

- `typing.get_type_hints(cls)` 自动处理 `from __future__ import annotations` 下字符串类型注解。
- `Optional[Path]` 会被 `get_type_hints` 解析为 `Path | None`；上面 P-Path 字段集已显式列出 `embedder_cache_dir`，None 值原样返回。
- 不强行处理 `Literal[...]` —— YAML 解析器返回的 str 直接传给 dataclass 构造，dataclass 不强校验 Literal 值；如果需要校验，单独写 `validate_eval_config` 风格的 validator。

M4.0 验收：`tests/test_phase5_config.py` 全绿 + 新增 `tests/test_phase10_rag_config_nested.py` 覆盖 `tuple[str, ...]`、`Optional[Path]`、嵌套 dataclass round-trip。

### 4.3 `MemoryConfig` 扩展

```python
from typing import Literal

@dataclass(frozen=True, slots=True)
class CorpusRagConfig:
    """v3 corpus RAG 配置。"""

    # ----- 总开关 -----
    enabled: bool = False                   # 全局开关；mode=disabled 时强制 false
    task_corpus: bool = True                # L1：当前 task 的文档（M4 唯一实现）
    shared_corpus: bool = False             # L2：维护方策划的只读语料（M4 占位，独立提案落地）
    shared_collections: tuple[str, ...] = ()
    shared_path: Path | None = None         # M4 不使用；保留供独立提案

    # ----- 索引 -----
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    max_chunks_per_doc: int = 200
    max_docs_per_task: int = 100
    task_corpus_index_timeout_s: float = 30.0

    # ----- Embedding -----
    embedder_backend: Literal["sentence_transformer", "stub"] = "sentence_transformer"
    embedder_model_id: str = "microsoft/harrier-oss-v1-270m"
    embedder_device: Literal["cpu", "cuda", "auto"] = "cpu"
    embedder_dtype: Literal["float32", "float16", "auto"] = "auto"
    embedder_query_prompt_name: str = "web_search_query"
    embedder_max_seq_len: int = 1024
    embedder_batch_size: int = 8
    embedder_cache_dir: Path | None = None  # None → 走 HF 默认；评测镜像指向 /opt/models/hf

    # ----- 向量库 -----
    vector_backend: Literal["chroma"] = "chroma"
    vector_distance: Literal["cosine", "ip", "l2"] = "cosine"

    # ----- 检索 -----
    retrieval_k: int = 4
    prompt_budget_chars: int = 1800

    # ----- 内容过滤 -----
    redact_patterns: tuple[str, ...] = (
        r"(?i)\banswer\b",
        r"(?i)\bhint\b",
        r"(?i)\bapproach\b",
        r"(?i)\bsolution\b",
    )
    redact_filenames: tuple[str, ...] = (
        "expected_output.json",
        "ground_truth*",
        "*label*",
        "*solution*",
    )
```

挂到 `MemoryConfig`（注意 `path` 的 default_factory 必须完整保留，**勿**简写为 `field(...)`）：

```python
@dataclass(frozen=True, slots=True)
class MemoryConfig:
    mode: str = "disabled"
    store_backend: str = "jsonl"
    path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "memory"
    )
    retriever_type: str = "exact"
    retrieval_max_results: int = 5
    rag: CorpusRagConfig = field(default_factory=CorpusRagConfig)
```

#### 三态运行开关与 RAG 的交叉

| `memory.mode` | `rag.enabled` | 实际行为 |
|---|---|---|
| `disabled` | * | RAG 全关；`recall_corpus_*` 直接 no-op |
| `read_only_dataset` | `false` | dataset facts 召回，corpus 全关 |
| `read_only_dataset` | `true` | dataset facts + corpus 召回（task_corpus；shared_corpus=true 时 fail-closed） |
| `full` | `false` | dataset facts + tool playbook 写入；corpus 全关 |
| `full` | `true` | 加上 corpus 召回 |

### 4.4 文档与 Chunk Dataclass（`memory/rag/documents.py`）

```python
@dataclass(frozen=True, slots=True)
class CorpusDocument:
    doc_id: str                                          # sha1(source_path + size + mtime)[:16]
    source_path: str                                     # task context_dir 内相对路径
    doc_kind: Literal["markdown", "text", "doc"]
    bytes_size: int
    char_count: int
    collection: str                                      # "task_corpus" 或 "shared:<name>"


@dataclass(frozen=True, slots=True)
class CorpusChunk:
    chunk_id: str                                        # f"{doc_id}#{ord:04d}"
    doc_id: str
    ord: int
    text: str
    char_offset: int
    char_length: int
```

- 故意没有：`question` / `answer` / `hint` / `approach` / `predicted_label` 字段。
- `text` 由 `Redactor` 过滤后才会进入 `CorpusChunk`，详见 §4.7。
- `chunk_id` 全局唯一且可由 `doc_id + ord` 重建。

### 4.5 `Embedder` Protocol（`memory/rag/embedders/base.py`）

```python
@runtime_checkable
class Embedder(Protocol):
    """文本 → L2 归一化向量。"""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
```

要点：

1. **doc / query 分离**。Harrier 系列 query 端必须加 instruction prompt（`web_search_query`），doc 端不加。
2. **返回 `list[list[float]]`**。简化 picklability、跨进程传输。
3. **L2 归一化由 Embedder 实现负责**。

### 4.6 `HarrierEmbedder`（`memory/rag/embedders/sentence_transformer.py`）

修复 v1 示例中两个 bug（D6）：`self._model_id` 必须赋值；`dtype="auto"` 也要传。

```python
class HarrierEmbedder:
    """sentence-transformers 后端，默认承载 microsoft/harrier-oss-v1-270m。"""

    def __init__(self, cfg: CorpusRagConfig) -> None:
        # 方法级延迟 import，rag.enabled=false 路径不付出 torch 加载代价
        from sentence_transformers import SentenceTransformer

        self._model_id = cfg.embedder_model_id
        self._query_prompt_name = cfg.embedder_query_prompt_name
        self._batch_size = cfg.embedder_batch_size
        device = self._resolve_device(cfg.embedder_device)

        self._model = SentenceTransformer(
            cfg.embedder_model_id,
            device=device,
            cache_folder=str(cfg.embedder_cache_dir) if cfg.embedder_cache_dir else None,
            # HF model card 推荐写法：始终传 model_kwargs={"dtype": ...}，包括 "auto"
            model_kwargs={"dtype": cfg.embedder_dtype},
        )
        self._model.max_seq_length = min(self._model.max_seq_length, cfg.embedder_max_seq_len)
        self._dim = self._model.get_sentence_embedding_dimension()

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        # "auto" 时尝试 cuda；失败安静回退 cpu（不抛）
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [text],
            prompt_name=self._query_prompt_name,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()
```

要点：

- **方法级延迟 import**：`sentence_transformers` / `torch` 在 `__init__` 内 import；模块顶层只 import 标准库与 `Embedder` Protocol。这样 rag extra 未安装时，`from data_agent_langchain.memory.rag.embedders.sentence_transformer import HarrierEmbedder` 这一行**本身**不抛（class 定义只引用类型注解中的字符串名）；只有真的 `HarrierEmbedder(cfg)` 才会触发 import error。
- **离线模式**：`HF_HOME` / `TRANSFORMERS_OFFLINE=1` 由容器入口设置；`cache_folder` 走 `cfg.embedder_cache_dir`。
- **CUDA 不可用容错**：`_resolve_device("auto")` 自动回退 cpu。CUDA OOM 在 `embed_documents` 内部不再特别处理（M4 评测态默认 cpu，OOM 几率极低）；若以后需要，单独提案。

### 4.7 `Redactor` 与 `Loader`（`memory/rag/redactor.py` / `loader.py`）

`Redactor` 在 chunk 化之前过滤：

```python
class Redactor:
    def __init__(self, cfg: CorpusRagConfig) -> None:
        self._patterns = tuple(re.compile(p) for p in cfg.redact_patterns)
        self._filename_globs = cfg.redact_filenames

    def is_safe_filename(self, name: str) -> bool: ...
    def filter_text(self, text: str) -> str: ...        # 整段命中即丢弃，不做 mask
```

`Loader` 职责：

1. 扫描 **`task.context_dir`**（D8）—— `task_dir` 含 `expected_output.json`，永不扫。
2. 跳过 `is_safe_filename=False` 的文件。
3. 解析 markdown / text / doc（``.json`` 已移除——大 JSON 数据文件导致 CPU 索引超时，agent 用 ``read_json`` / ``execute_python`` 处理更高效）。
4. 截断超过 `max_docs_per_task` 的文件数。
5. 返回 `list[CorpusDocument]`。

不做：

- 不解析二进制（pdf / docx / xlsx）—— 容器内不一定有解析器。
- 不调用网络 / 子进程。

### 4.8 `Chunker`（`memory/rag/chunker.py`）

```python
class CharWindowChunker:
    def __init__(self, cfg: CorpusRagConfig) -> None:
        self._size = cfg.chunk_size_chars
        self._overlap = cfg.chunk_overlap_chars
        self._max_chunks = cfg.max_chunks_per_doc

    def chunk(self, doc: CorpusDocument, text: str) -> list[CorpusChunk]: ...
```

规则：

- 按字符窗口切；窗口大小 `chunk_size_chars`，相邻窗口重叠 `chunk_overlap_chars`。
- 不破坏 markdown 段落语义优先（如果当前位置正好在 `\n\n` 分段点，优先在此处切）。
- 每个 doc 最多产出 `max_chunks_per_doc`，超过截断并 dispatch 一个 `memory_rag_skipped(reason="max_chunks_truncated")`。

### 4.9 `CorpusStore` Protocol 与 `ChromaCorpusStore`（B2 决策）

#### `memory/rag/base.py`（独立 Protocol，不复用 v2 `MemoryStore`）

```python
@runtime_checkable
class CorpusStore(Protocol):
    """corpus 专属 store 接口，与 v2 MemoryStore 解耦。"""

    @property
    def namespace(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def upsert_chunks(self, chunks: Sequence[CorpusChunk]) -> None: ...

    def query_by_vector(self, vector: Sequence[float], *, k: int) -> list[RetrievalResult]: ...

    def close(self) -> None: ...
```

`query_by_vector` 返回 `list[RetrievalResult]`（v2 类型）；其中 `record = MemoryRecord(kind="corpus", payload={...})`，让上层 `VectorCorpusRetriever` 与 v2 `Retriever` Protocol 兼容。

#### `memory/rag/stores/chroma.py`

```python
class ChromaCorpusStore:
    """实现 CorpusStore 协议；仅 ephemeral 模式（D2：persistent 拆到独立提案）。"""

    @classmethod
    def ephemeral(
        cls,
        namespace: str,
        embedder: Embedder,
        *,
        distance: str = "cosine",
    ) -> "ChromaCorpusStore":
        # 方法级延迟 import
        import chromadb
        from chromadb.config import Settings

        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        return cls(client, namespace=namespace, embedder=embedder, distance=distance)

    def __init__(self, client, *, namespace: str, embedder: Embedder, distance: str) -> None:
        self._client = client
        self._namespace = namespace
        self._embedder = embedder
        self._distance = distance
        # 用 sha1 编码 namespace，避免 collision 与 chroma collection name 限制
        import hashlib
        safe = hashlib.sha1(namespace.encode()).hexdigest()[:16]
        collection_name = f"memrag_{safe}"
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance, "ns": namespace},
            embedding_function=None,  # 显式禁用 chroma 内置 embedding（避免 OpenAI 等外部 API）
        )

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    def upsert_chunks(self, chunks: Sequence[CorpusChunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self._embedder.embed_documents(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "ord": c.ord,
                "char_offset": c.char_offset,
                "char_length": c.char_length,
            }
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    def query_by_vector(self, vector: Sequence[float], *, k: int) -> list[RetrievalResult]:
        if k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievalResult] = []
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            record = MemoryRecord(
                id=chunk_id,
                namespace=self._namespace,
                kind="corpus",
                payload={
                    "text": text,
                    "doc_id": meta["doc_id"],
                    "ord": meta["ord"],
                    "char_offset": meta["char_offset"],
                    "char_length": meta["char_length"],
                },
                metadata={},
            )
            # cosine distance ∈ [0, 2]; score = 1 - dist/2 ∈ [-1, 1]，越大越相关
            score = 1.0 - float(dist) / 2.0 if self._distance == "cosine" else -float(dist)
            out.append(RetrievalResult(record=record, score=score, reason="vector_cosine"))
        return out

    def close(self) -> None:
        # EphemeralClient 没有显式 close；这里仅丢弃引用以便 GC
        self._client = None
```

要点：

- **collection 命名**：`memrag_{sha1(namespace)[:16]}`（D10）。
- **embedding function 显式禁用**：`embedding_function=None` + 手动 `upsert(embeddings=...)`，避免触发外部网络。
- **元数据**：每个 chunk 在 Chroma 中保存 `doc_id`、`ord`、`char_offset`、`char_length`。**不保存** `source_path` 等内容外字段；source_path 由调用方在召回后渲染 summary 时从 `doc_id → source_path` 反查（factory 在构建索引时同时返回一份 `doc_id → CorpusDocument` 映射，存进 retriever）。
- **不实现 `put / get / list / delete`**（B2）。任何想用 v2 `MemoryStore` Protocol 的代码不能误把 chroma store 当 KV 用。
- **不实现 `persistent_readonly`** / `persistent_writable`（D2）。

### 4.10 `VectorCorpusRetriever`（`memory/rag/retrievers/vector.py`）

```python
class VectorCorpusRetriever:
    """实现 v2 Retriever 协议（适配 corpus 语义）。"""

    def __init__(
        self,
        store: CorpusStore,
        embedder: Embedder,
        doc_index: Mapping[str, CorpusDocument],
        *,
        k: int,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._doc_index = dict(doc_index)
        self._k = k

    def retrieve(self, query: str, *, namespace: str, k: int | None = None) -> list[RetrievalResult]:
        actual_k = self._k if k is None else k
        if actual_k <= 0:
            return []
        try:
            vector = self._embedder.embed_query(query)
        except Exception:
            return []  # 由 recall_corpus_snippets 负责事件 dispatch
        try:
            raw = self._store.query_by_vector(vector, k=actual_k)
        except Exception:
            return []
        # 把 doc 元数据补回 payload（source_path / doc_kind）
        results: list[RetrievalResult] = []
        for r in raw:
            doc_id = r.record.payload.get("doc_id")
            doc = self._doc_index.get(doc_id)
            payload = dict(r.record.payload)
            if doc is not None:
                payload["source_path"] = doc.source_path
                payload["doc_kind"] = doc.doc_kind
            results.append(RetrievalResult(
                record=MemoryRecord(
                    id=r.record.id,
                    namespace=r.record.namespace,
                    kind="corpus",
                    payload=payload,
                    metadata=r.record.metadata,
                ),
                score=r.score,
                reason="vector_cosine",
            ))
        return results
```

- 召回返回的 `RetrievalResult.reason="vector_cosine"`。
- 召回时**不再对 query 文本做 redact 过滤**，因为 query 不会进入持久化 store；但调用方（`recall_corpus_snippets`）必须保证 query 仅来自当前 task 内合法字段（`task.question` 或 `last_thought`）。
- `doc_index` 保留 `source_path` / `doc_kind`，避免把 source_path 写进 chroma metadata 占空间。

### 4.11 `factory`（`memory/rag/factory.py`）

```python
@dataclass(frozen=True, slots=True)
class TaskCorpusHandles:
    """子进程入口构建出的 task corpus 句柄；通过 contextvar 注入。"""
    embedder: Embedder
    store: CorpusStore
    retriever: VectorCorpusRetriever


def build_embedder(cfg: CorpusRagConfig) -> Embedder | None:
    """rag 关闭返回 None；rag 开启返回 stub 或 HarrierEmbedder。"""

def build_task_corpus(
    cfg: CorpusRagConfig,
    *,
    task_id: str,
    task_input_dir: Path,                  # 总是 task.context_dir（D8）
    embedder: Embedder,
    config: RunnableConfig | None = None,  # 用于 dispatch 事件
) -> TaskCorpusHandles | None:
    """子进程入口构建 per-task corpus；rag 关闭、目录不存在、空文档、超时等情况返回 None。"""
```

工厂返回的 handles **通过 contextvar 暴露**给 graph node，不进 `RunState`。沿用 v2 `runtime/context.py` 的 contextvar 模式。

`build_task_corpus` 内部流程：

1. 检查 `cfg.enabled and cfg.task_corpus`；否则返回 None。
2. `loader.scan(task_input_dir)` → `list[CorpusDocument]`；空列表返回 None + `memory_rag_skipped(reason="no_documents")`。
3. `redactor` 过滤 → 读取每个 doc 内容 → 命中 redact_patterns 的整段 doc 丢弃。
4. `chunker.chunk(doc, text)` → `list[CorpusChunk]`。
5. `ChromaCorpusStore.ephemeral(namespace=f"corpus_task:{task_id}", embedder=embedder)`。
6. `store.upsert_chunks(all_chunks)`（含 embed_documents）。
7. 全程包在 `task_corpus_index_timeout_s` 超时内；超时返回 None + `memory_rag_skipped(reason="index_timeout")`。
8. 成功 dispatch `memory_rag_index_built(task_id, doc_count, chunk_count, elapsed_ms, model_id, dimension)`。
9. 返回 `TaskCorpusHandles(embedder, store, retriever)`。

`shared_corpus=True` 但独立提案未实现 → `memory_rag_skipped(reason="shared_corpus_not_implemented")`，task 继续。

### 4.12 `RunState` 与 prompt 注入

#### 不修改 `RunState`

v2 已有 `memory_hits: Annotated[list[MemoryHit], operator.add]`（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/runtime/state.py:96-98`）。corpus 召回的 hits 同样写到这个字段，仅靠 `MemoryHit.summary` 与 `MemoryHit.namespace` 区分来源。**不新增**字段。

##### `MemoryHit.summary` 渲染规则（corpus 专属）

```
[<doc_kind>] <source_path>: <text 摘要 ≤ 240 字符>
```

例：

```
[markdown] task_344/README.md: transaction_date 字段以 YYYYMMDD 整数存储，应当解析为 UTC 日期...
```

- `text` 摘要由 `chunk.text[:240]` + `...`（截断时）构成。
- 不渲染 `chunk_id` / `doc_id` / `score` / `embedding` 等内部字段。

#### `agents/prompts.py` 新增渲染函数（D5 路线）

```python
def render_corpus_snippets(hits: list[MemoryHit], budget_chars: int) -> str:
    """白名单：仅渲染 hit.summary；按 budget 截断；保持稳定排序。"""
```

**注入位置**：与 `render_dataset_facts` 完全平行 —— 在 `model_node._build_messages_for_state` 内（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/model_node.py:165-169`）：

```python
hits = list(state.get("memory_hits") or [])
dataset_hits = [h for h in hits if not h.namespace.startswith("corpus_")]
corpus_hits = [h for h in hits if h.namespace.startswith("corpus_")]
facts_text = render_dataset_facts(dataset_hits)
snippets_text = render_corpus_snippets(corpus_hits, budget_chars=app_config.memory.rag.prompt_budget_chars)
extra = "\n\n".join(s for s in (facts_text, snippets_text) if s)
if extra:
    messages = list(messages) + [HumanMessage(content=extra)]
```

**不动 `build_planning_messages` / `build_plan_solve_execution_messages` / `build_react_messages`** —— v2 的 dataset facts 注入就在 builders 外，v3 corpus snippets 沿用此风格。

### 4.13 `agents/corpus_recall.py` 与 task 入口召回（A2 决策）

#### `recall_corpus_snippets`

与 v2 `agents/memory_recall.py` 并列：

```python
def recall_corpus_snippets(
    cfg: CorpusRagConfig,
    *,
    task_id: str,
    query: str,
    node: str,
    config: RunnableConfig | None,
) -> list[MemoryHit]: ...
```

实现：

1. `cfg.enabled` 关 / contextvar 无 corpus handles → 返回 `[]`。
2. 拿到 retriever 后召回 `retrieval_k` 条。
3. 按 `prompt_budget_chars` 软截断（累计 summary 字符数不超）。
4. dispatch `memory_recall(kind="corpus_task", query_digest=sha1(query)[:8], hit_chunk_ids=[...], model_id=embedder.model_id, reason="vector_cosine")`。
5. 任何异常 → 返回 `[]` + dispatch `memory_rag_skipped(reason="retrieve_failed")`。
6. 返回 `list[MemoryHit]`。

#### 召回时机（A2）

| 模式 | 召回点 | 调用顺序 |
|---|---|---|
| `plan_solve` | `planner_node` 入口 | `recall_dataset_facts` → `recall_corpus_snippets`；两组 hits 通过 `output["memory_hits"]` 一次性写回（reducer = `operator.add` 自然合并） |
| `react` | 新增 `task_entry_node`（`START → task_entry → execution → finalize`） | 同上 |

`task_entry_node` 的伪代码：

```python
def task_entry_node(state: RunState, config: Any | None = None) -> dict[str, Any]:
    app_config = _safe_get_app_config()
    dataset_name = Path(state.get("dataset_root", "") or ".").name or "default"
    task_id = state.get("task_id", "")
    question = state.get("question", "")

    dataset_hits = recall_dataset_facts(
        memory_cfg=app_config.memory,
        dataset=dataset_name,
        node="task_entry_node",
        config=config,
    )
    corpus_hits = recall_corpus_snippets(
        cfg=app_config.memory.rag,
        task_id=task_id,
        query=question,
        node="task_entry_node",
        config=config,
    )
    output: dict[str, Any] = {}
    hits = list(dataset_hits) + list(corpus_hits)
    if hits:
        output["memory_hits"] = hits
    return output
```

修改：

- `react_graph.build_react_graph`：`START → task_entry → execution → finalize`。
- `plan_solve_graph.build_plan_solve_graph`：planner_node 内部追加 `recall_corpus_snippets` 调用并把 hits 合并到 `output["memory_hits"]`。**不**新增节点（v2 已有 planner 充当入口）。
- `model_node`：**完全不动召回逻辑**，仍只消费 `state["memory_hits"]`。

**收益**：

- prompt 不再随步数膨胀；
- ReAct 模式首次具备 dataset facts 注入能力（修复 v2 隐藏 gap）；
- 每 task 仅 1 次 query embed + 1 次 chroma query；
- `state["memory_hits"]` 也只在入口写 1 次，reducer 累加 0 次（vs. v1 设计每步 +N）。

### 4.14 RAG 边界（v2 4.11 占位 → v3.1 落地）

- `memory/rag/__init__.py` 不再抛 `NotImplementedError`，正式导出 `CorpusDocument` / `CorpusChunk` / `Embedder` / `CorpusStore` / `VectorCorpusRetriever` / `build_*`；但**不导入** `HarrierEmbedder` / `ChromaCorpusStore`（避免触发重依赖顶层 import）。这两类用 `from data_agent_langchain.memory.rag.embedders.sentence_transformer import HarrierEmbedder` 显式 import，且仅在 factory 内部使用。
- corpus 不写入跨 task 自由文本；写入路径仅在 task 子进程入口的 `build_task_corpus` 中存在，且 `EphemeralClient` 子进程退出即释放。
- Phase M5 引入混合检索（BM25 + vector + RRF）只需新增 `retrievers/hybrid.py`，不改图层。
- shared corpus 由 `05-shared-corpus-design.md` 独立提案落地（D2）。

---

## 五、读写数据流

### 5.1 离线 ingest（推迟）

shared corpus 的离线构建拆到独立提案。M4 阶段，corpus 数据**只**在 task 子进程入口的内存中构建一次，task 退出即销毁。

### 5.2 子进程入口（`runner` 启动 task 子进程时）

伪代码：

```python
runner._run_task_in_subprocess(task_id, app_config_dict):
    app_config = AppConfig.from_dict(app_config_dict)
    set_current_app_config(app_config)
    ...
    if app_config.memory.mode != "disabled" and app_config.memory.rag.enabled:
        embedder = factory.build_embedder(app_config.memory.rag)
        if embedder is not None and app_config.memory.rag.task_corpus:
            handles = factory.build_task_corpus(
                app_config.memory.rag,
                task_id=task_id,
                task_input_dir=task.context_dir,
                embedder=embedder,
                config=runnable_config,
            )
            if handles is not None:
                set_current_corpus_handles(handles)
        if app_config.memory.rag.shared_corpus:
            dispatch_observability_event(
                "memory_rag_skipped",
                {"scope": "shared", "reason": "shared_corpus_not_implemented"},
                config=runnable_config,
            )
    graph.invoke(state)
```

事件：`memory_rag_index_built(task_id, doc_count, chunk_count, elapsed_ms, model_id, dimension)`。

### 5.3 召回路径（task 入口）

**Plan-Solve**：在 `planner_node` 入口已有的 `recall_dataset_facts` 调用之后**立即**追加 `recall_corpus_snippets`，把两个 hits list 合并到 `output["memory_hits"]`。

**ReAct**：新增 `task_entry_node`，挂在 `START` 与 `execution` 之间，做与 plan-solve planner 等价的两次召回。

`query` 选取：固定使用 `task.question`（不用 `last_thought` —— 入口时 thought 还不存在；这也避免了 v1 设计中 off-by-one 的歧义）。

### 5.4 Prompt 注入

`model_node._build_messages_for_state` 拼装：

```
<builder 输出 messages>
└── HumanMessage(
        ## Dataset facts (from prior runs, informational only)
        - ...

        ## Reference snippets (from task documentation)
        - [markdown] task_344/README.md: ...
        - [text] task_344/data_schema.md: ...
    )
```

dataset facts 段在前、corpus snippets 段在后；两段共用 prompt 末尾、独立各自的字符预算（dataset facts 不截断，受 `retrieval_max_results=5` 自然控长；corpus snippets 受 `prompt_budget_chars=1800` 软截断）。

明确文案：「仅供参考，不替代 `read_csv` / `inspect_sqlite_schema` 等真实工具。」

### 5.5 Prompt budget 算账

- `agent.max_context_tokens=24000`（约 96 K 字符上限）。
- dataset facts：5 条 × ~80 字符 ≈ 400 字符。
- corpus snippets：受 `prompt_budget_chars=1800` 硬上限。
- 两段合计 ≤ 2200 字符 ≈ 550 tokens，占 `max_context_tokens` 约 2.3%。
- 由于 A2 决策（仅入口 1 次召回），prompt 增量不会随步数累加；ReAct 跑 60 步与 1 步的注入开销一致。

### 5.6 与 gate / dataset_facts 的关系

- corpus 召回**不**置 `preview_done` / `discovery_done`。
- corpus snippets 不替代 dataset_knowledge；它们各自走各自 namespace、各自渲染。
- 同一段 prompt 内：`dataset facts` 在前、`corpus snippets` 在后；总长度受 §5.5 算账约束。

---

## 六、安全边界

### 6.1 数据流禁令

- **跨 task 不持久化任何 `CorpusChunk`**：L1 索引随 task 子进程结束释放（`EphemeralClient`）。
- **M4 没有 corpus 写入路径**：`MemoryWriter` Protocol 不新增 `write_corpus_*`；shared corpus 离线构建由独立提案处理。
- **corpus 召回 query 不进入任何持久化结构**：仅 `query_digest = sha1(query)[:8]` 进入 `memory_recall` 事件。

### 6.2 内容过滤

`Redactor`：

- 命中 `redact_patterns` 的整段（chunk 化前的原 doc 文本）直接丢弃。
- 命中 `redact_filenames`（glob）的文件直接 skip。
- 任务输入目录中的 `expected_output.json`、`*label*`、`*solution*`、`ground_truth*` 一律 skip。
- **`task.task_dir` 永不扫**（D8）—— 即使 redactor patterns 失效，也不会读到 `expected_output.json`。

### 6.3 picklability

- Embedder / Store / Retriever 通过 contextvar 持有；不进 `RunState`。
- `MemoryHit.summary` 字段保持 `str`，不含任何句柄或 numpy。
- `CorpusRagConfig` 全部基本类型 + `Path | None`，picklable。

### 6.4 Fail closed

- `memory.mode=disabled` → RAG 强制关闭。
- `rag.enabled=false` → factory 不构建任何索引。
- Embedder 加载失败 → `memory_rag_skipped(reason="embedder_load_failed")`，task 继续。
- 索引构建失败 → 同上 `reason="index_failed"`。
- 索引超时 → `reason="index_timeout"`。
- Chroma query 异常 → 召回返回 `[]`，dispatch `reason="retrieve_failed"`，task 继续。
- query embed 失败 → 同上。
- `shared_corpus=true` 但独立提案未实现 → `reason="shared_corpus_not_implemented"`，task 继续。

### 6.5 启动期 import 边界（D11，新增）

- `memory/rag/__init__.py` 顶层只允许 import 标准库与 `documents` / `loader` / `chunker` / `redactor` / `base`（CorpusStore Protocol） / `embedders.base` / `retrievers.vector`。
- **顶层禁止 import**：`embedders.sentence_transformer`（拉 torch）、`stores.chroma`（拉 chromadb）。
- `embedders.sentence_transformer` 与 `stores.chroma` 自身模块顶层也**禁止 import** `torch` / `sentence_transformers` / `chromadb`；只能在方法体 / `@classmethod` 内 import。
- 守护：新增回归测试 `tests/test_phase10_rag_import_boundary.py`：用 `subprocess` 启动 `python -c "import data_agent_langchain"` 后断言 `sys.modules` 不含 `torch` / `chromadb` / `sentence_transformers`。

### 6.6 Resource budget

- `embedder_max_seq_len` 限制 token 数。
- `embedder_batch_size` 限制单次 forward 的 batch。
- `max_chunks_per_doc` / `max_docs_per_task` 限制单 task 索引规模。
- 索引构建超过 `task_corpus_index_timeout_s`（默认 30s）→ skip。

---

## 七、观测与可审计

### 7.1 复用 `memory_recall`

`MetricsCollector.on_custom_event("memory_recall", ...)` 现状（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/observability/metrics.py:62-63`）只 `dict(data)`，对加字段透明。新增字段：

- `kind`：`dataset_knowledge | corpus_task | corpus_shared`
- `query_digest`：`sha1(query)[:8]`，**不存原文**
- `hit_doc_ids`、`hit_chunk_ids`、`scores`
- `model_id`：召回时使用的 embedder model id（仅 corpus 类）

兼容性：扫 `tests/test_phase9_memory_*.py`，确认 `kind` 字段不破坏现有断言（v2 测试没用 `kind` 字段，应当无影响 —— 实施前用 `grep -r "memory_recall.*kind" tests/` 验证）。

### 7.2 新增事件

- `memory_rag_index_built`：`{ "scope": "task", "namespace": ..., "doc_count": ..., "chunk_count": ..., "elapsed_ms": ..., "model_id": ..., "dimension": ... }`
- `memory_rag_skipped`：`{ "scope": ..., "reason": ..., "elapsed_ms": ... }`

`reason` 枚举：`embedder_load_failed | index_failed | index_timeout | no_documents | max_docs_truncated | max_chunks_truncated | retrieve_failed | query_embed_failed | shared_corpus_not_implemented`。

`MetricsCollector` 需要把这两个事件聚合进 `metrics.json.memory_rag`（见 §7.3）。

### 7.3 新增 metrics 字段

`metrics.json` 增加（仅 RAG enabled 时出现，避免 baseline metrics 漂移）：

```json
"memory_rag": {
  "task_index_built": true,
  "task_doc_count": 12,
  "task_chunk_count": 48,
  "shared_collections_loaded": 0,
  "recall_count": {"task_entry": 1},
  "skipped": []
}
```

`recall_count` 在 A2 决策下只会有一个 key（`task_entry` 或 `planner`），永不出现 `model`。

---

## 八、容器与离线打包

### 8.1 依赖打包（C2 决策）

`pyproject.toml`：

```toml
[project.optional-dependencies]
rag = [
    "sentence-transformers>=3,<4",
    "chromadb>=0.5,<1",
    # torch 由 sentence-transformers 间接拉；如果评测镜像需要严格 pin，再单独加
]
```

- 本地 dev：`uv sync` / `pip install -e ".[dev,rag]"`。
- 提交镜像默认：`pip install .` —— 体积零增量。
- 启用 RAG 的提交镜像：Dockerfile 改为 `pip install ".[rag]"` + COPY 预下载的 HF cache。

### 8.2 Submission 镜像构建步骤

**前提**：评测镜像构建期不联网（这是常见的 KDD Cup 约束；如果允许联网，构建期可改用 `RUN python -c "SentenceTransformer(...)"`）。

```dockerfile
# 1. 启用 RAG 路径才走这一阶段
ARG ENABLE_RAG=0
RUN if [ "$ENABLE_RAG" = "1" ]; then pip install ".[rag]"; fi

# 2. 把宿主预下载好的 HF cache COPY 进镜像
ENV HF_HOME=/opt/models/hf
ENV TRANSFORMERS_OFFLINE=1
COPY models/harrier-270m /opt/models/hf/hub/models--microsoft--harrier-oss-v1-270m
```

**宿主预下载**（dev 机器联网时执行一次）：

```bash
HF_HOME=./models python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('microsoft/harrier-oss-v1-270m')"
```

评测态 yaml 显式指定：

```yaml
memory:
  mode: read_only_dataset
  rag:
    enabled: true
    embedder_cache_dir: /opt/models/hf
    embedder_device: cpu
    embedder_dtype: float32  # CPU 上 fp16 反而慢
```

### 8.3 体积与依赖估算（修正 v1）

| 项 | 增量 |
|---|---|
| `sentence-transformers` | ~30 MB |
| `transformers` + `tokenizers` | ~200 MB |
| `torch` CPU wheel | **~750 MB**（v1 低估） |
| `chromadb` + 其传递依赖（onnxruntime, posthog, opentelemetry, ...） | ~250 MB |
| Harrier 270m fp32 权重 | ~1.1 GB |
| Harrier 270m fp16 权重 | ~540 MB |

合计：CPU fp32 路线下 + **~2.3 GB**；CPU fp16 路线下 + ~1.8 GB。需要在 M4.0 完成后用 `docker build --build-arg ENABLE_RAG=1` 实测镜像大小，确认竞赛镜像上限。

### 8.4 评测态默认值（`submission.build_submission_config`）

- `memory.mode = "read_only_dataset"`（不变）。
- `memory.rag.enabled = false`（M4 上线初期保守关闭，A/B 测试通过后再开）。
- 通过 env `DATA_AGENT_RAG=1` 一键开启。
- 当 `DATA_AGENT_RAG=1` 但镜像未装 RAG extra → `build_embedder` 触发 `ImportError` → catch 后 dispatch `memory_rag_skipped(reason="embedder_load_failed")`，task fall back 到 baseline 路径。

---

## 九、风险与待决问题

### 9.1 已识别风险

1. **CPU 上 270m decoder-only 编码速度**：未实测。M4.4 用 task_344 实测，目标：单 task 索引 ≤ 5s，单次 query embed ≤ 100ms。
2. **chroma 离线模式**：`Settings(anonymized_telemetry=False)` 已强制；M4.3 单测验证不触发任何外部网络。
3. **chromadb Windows 安装**：开发机偶发 SQLite 锁。M4.3 必须 Windows + ephemeral 路径验证无 issue。
4. **镜像体积超限**：CPU fp32 增量 ~2.3 GB。M4.4 完成后实测；若超限，降级 fp16 或换 `bge-small-zh-v1.5`（仅改 `embedder_model_id`，代码不变）。
5. **启动期 import 边界破坏**：任何后续 PR 在 rag 子模块顶层加 `import torch` 都会触发回归测试失败；CI 必须把这条测试纳入 phase 10 必跑集合。
6. **A2 召回精度不足**：仅入口召回 1 次，可能错过 ReAct 后期 thought 演进出的细粒度查询。M4 完成后用 A/B 实测：若 task_344 上 corpus 召回完全没贡献，再评估是否升级到 A3（动态条件触发）。

### 9.2 待决问题（M4 完成前需回答）

- Q1：评测容器是否允许使用 GPU？默认假设 **CPU only**。
- Q2：评测镜像构建期是否允许联网？若不允许，必须走 §8.2 的 COPY 方案。
- Q3：评测镜像体积上限？影响是否走 fp32 / fp16 / `bge-small` 降级。

---

## 十、与 v1（v3）的差异回顾

| 维度 | v1 主张 | v2（本版） |
|---|---|---|
| 召回时机 | planner + model_node 每步 | **task 入口一次**（plan-solve 走 planner，ReAct 走新增 task_entry_node） |
| Store Protocol | corpus store 同时实现 v2 `MemoryStore.put/get/list/delete` + corpus 接口 | **独立 `CorpusStore`** Protocol，仅 `upsert_chunks / query_by_vector / close` |
| 依赖位置 | `[project.dependencies]` | **`[project.optional-dependencies].rag`** |
| shared_corpus | M4.6 在主提案内 | **拆出独立提案** `05-shared-corpus-design.md`；本版仅占位 + fail-closed |
| Prompt 注入 | 改三个 builder | **不动 builder**；在 `model_node` 内串行渲染 dataset_facts + corpus_snippets，与 v2 现状一致 |
| HarrierEmbedder 示例 | `self._model_id` 未赋值；`dtype="auto"` 走空 model_kwargs | 修复：赋值 + `model_kwargs={"dtype": cfg.embedder_dtype}` 始终传 |
| `MemoryConfig.path` 示例 | `path: Path = field(...)`（截断） | 完整 `default_factory` |
| `task_input_dir` | 未明确 | **`task.context_dir`**（不扫 `task_dir`） |
| 嵌套 dataclass | 只提 `typing.get_type_hints` | 增加 `tuple[T, ...]` / `Optional[Path]` 显式处理 |
| namespace 编码 | `replace(":", "__")` | `sha1(ns)[:16]` |
| 启动期 import 边界 | 未提 | **新增**回归测试，禁止 rag 子模块顶层 import torch / chromadb |

v2 不破坏 v2（v2 = 当前主干）任何已有契约，是纯增量扩展；与 v1（v3）相比是行为范围收缩 + 事实修正。
