# Corpus RAG 架构设计 v3（M4 落地版）

> 版本：v3，2026-05-14
> 前置版本：`02-v2-technical-design/01-technical-design.md`（v2 结构化记忆设计）
> 实施依赖：v2 已合入主干，`memory/rag/` 子包当前为占位（`NotImplementedError`）。
> 适用代码基：`src/data_agent_langchain/`

---

## 一、背景与目标

### 1.1 为什么要做 v3 / M4

v2 的 Memory MVP 已经落地结构化 dataset_knowledge 与 tool_playbook，能解决「数据集字段、schema 这种**结构化事实**的跨任务复用」。但仍然不能覆盖：

- 长篇 README、说明文档、数据字典等**非结构化文本**。
- 比赛规则、工具用法、领域知识这种**评测时只读、需要语义检索的语料**。
- 当用户问到「这个字段含义是什么」「这种数据格式怎么解析」之类问题时，仅靠 schema 召回明显信息不足。

v2 设计 4.11 已经画出 `memory/rag/` 占位边界。v3 在该边界内落地 **离线、向量化、本地嵌入** 的 corpus RAG。

### 1.2 目标

1. 在不破坏 `memory.mode=disabled` baseline parity 的前提下，落地基于 **向量嵌入 + 向量数据库** 的 corpus RAG。
2. 沿用 v2 的 `Store / Retriever / Writer` Protocol、`MemoryConfig`、`MemoryHit`、`memory_recall` 事件、`render_*_facts` 白名单注入。
3. 嵌入模型本地推理：**`microsoft/harrier-oss-v1-270m`**，本地与评测使用同一份权重，**不依赖任何远程 embedding API**。
4. 向量库使用 **`chromadb`** 本地客户端：per-task 用 ephemeral，shared corpus 用 persistent + 只读。
5. 所有 RAG 索引构建必须在 **runner 子进程入口** 完成；图节点循环内只允许做 retrieve，不允许做 ingest / index。
6. 严格遵守 v2 已有的安全边界：禁止 question / answer / approach / hint 进入任何持久化结构。

### 1.3 非目标（YAGNI）

- 不实现 episodic memory、跨 task answer 记忆。
- 不在评测路径写入 corpus 数据；shared corpus 只能通过离线 ingest CLI 构建。
- 不引入除 `chromadb` / `sentence-transformers` / `transformers` 外的新向量库 / embedding 框架。
- 不做 reranker、cross-encoder、LLM-based rerank（M4 完成后另立提案）。
- 不做混合检索（BM25 + vector）；接口预留，但默认纯向量。
- 不替代 `gate.py` 的真实数据预览。

---

## 二、当前系统事实（v2 已落地）

v3 设计基于以下 v2 已合入主干的事实，避免重复定义。

### 2.1 v2 已有抽象（`src/data_agent_langchain/memory/`）

- `base.py`：`MemoryRecord`、`RetrievalResult`、`MemoryStore` / `Retriever` / `MemoryWriter` Protocol。
- `records.py`：`DatasetKnowledgeRecord`、`ToolPlaybookRecord`（白名单字段）。
- `types.py`：`MemoryHit`（写入 `RunState`）。
- `stores/jsonl.py`：`JsonlMemoryStore`，append-only JSONL，按 namespace 分文件。
- `retrievers/exact.py`：`ExactNamespaceRetriever`。
- `writers/store_backed.py`：`StoreBackedMemoryWriter`，按 mode 控制。
- `factory.py`：集中构建 store / retriever / writer。
- `working.py`：单 run scratchpad，**不动**。
- `rag/__init__.py`：占位，抛 `NotImplementedError("Corpus RAG is reserved for Phase M4")`。

### 2.2 v2 已有配置（`config.py:126-135`）

```python
@dataclass(frozen=True, slots=True)
class MemoryConfig:
    mode: str = "disabled"                          # disabled | read_only_dataset | full
    store_backend: str = "jsonl"                    # jsonl | sqlite (future)
    path: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "memory")
    retriever_type: str = "exact"
    retrieval_max_results: int = 5
```

`AppConfig.from_dict` 用扁平的 `_dataclass_from_dict`，**不递归**进嵌套 dataclass。这是 v3 必须先解决的前置约束（详见 §4.2）。

### 2.3 v2 已有写入与召回路径

- `tool_node` 成功后写 `DatasetKnowledgeRecord` / `ToolPlaybookRecord`。
- `agents/memory_recall.py` 的 `recall_dataset_facts` 在 `planner_node` / `model_node` 入口召回。
- `agents/prompts.py` 的 `render_dataset_facts` 把 `MemoryHit.summary` 列表渲染为 prompt 段。
- `observability.events.dispatch_observability_event("memory_recall", ...)` 审计每次召回。
- `RunState.memory_hits: Annotated[list[MemoryHit], operator.add]`，仅含 `summary`。
- `submission.build_submission_config` 默认 `memory.mode=read_only_dataset`。

---

## 三、设计原则

1. **复用 v2 抽象，不另起 Protocol**：corpus 视作一类特殊的 `MemoryRecord(kind="corpus")`；`MemoryStore` / `Retriever` 子类承担向量索引职责。
2. **embedding 与 store 严格解耦**：`Embedder` 是独立 Protocol，`ChromaCorpusStore` 只持有客户端句柄，不知道嵌入模型是谁。
3. **per-task ephemeral，per-shared persistent**：两层 corpus 用同一个 `Retriever` Protocol，不同的工厂构造路径。
4. **索引前置、检索内联**：所有 ingest / chunking / embedding 在子进程入口完成；图节点只读检索。
5. **白名单内容过滤**：v2 的写入禁令同样适用 corpus 文档；`Redactor` 在 chunking 之前过滤。
6. **fail closed**：`memory.mode=disabled` 时 RAG 强制关闭，无视 `rag.enabled`；任何 RAG 异常都退化到无 RAG 行为。
7. **picklability 不变**：`RunState` 只增加 `memory_hits` 已有字段中的 corpus hit；不放 store / retriever / embedder 句柄。
8. **观测复用 v2**：所有召回都走 `memory_recall` 事件；新增 `memory_rag_index_built` / `memory_rag_skipped` 两个事件用于运维审计。

---

## 四、目标架构

### 4.1 目录与新增文件

在现有 `memory/` 下展开 `rag/` 子包：

```
src/data_agent_langchain/memory/
  rag/
    __init__.py                  # 由占位改为正式导出
    documents.py                 # 新增：CorpusDocument / CorpusChunk
    loader.py                    # 新增：从 task 目录或 shared 目录扫描 + 解析
    chunker.py                   # 新增：char-window 切片，token-aware 上限
    redactor.py                  # 新增：内容过滤（关键词 / 文件名）
    embedders/
      __init__.py
      base.py                    # 新增：Embedder Protocol
      sentence_transformer.py    # 新增：HarrierEmbedder（或通用 STEmbedder）
      stub.py                    # 新增：DeterministicStubEmbedder（测试用）
    stores/
      __init__.py
      chroma.py                  # 新增：ChromaCorpusStore（ephemeral / persistent）
    retrievers/
      __init__.py
      vector.py                  # 新增：VectorCorpusRetriever
    factory.py                   # 新增：build_embedder / build_corpus_store / build_corpus_retriever
    ingest_cli.py                # 新增：离线 ingest（仅用于 corpus_shared）
```

新增模块约定：

- 所有 dataclass 保持 `frozen=True, slots=True`，picklable。
- 所有 Protocol 用 `@runtime_checkable`，与 v2 一致。
- 不引入对 `langchain` 高层 RAG 抽象的依赖；只用 `langchain-core` 的 message / config 类型，与现有项目风格一致。

### 4.2 前置任务：让 `_dataclass_from_dict` 支持嵌套

v3 的 `MemoryConfig` 需要嵌套 `CorpusRagConfig`，但当前 `_dataclass_from_dict` 不递归。M4.0 必须先：

```python
# config.py
def _dataclass_from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in payload:
            continue
        value = payload[field_info.name]
        if field_info.name in {"root_path", "output_dir", "gateway_caps_path", "path"}:
            value = Path(value)
        elif field_info.name == "model_retry_backoff":
            value = tuple(float(item) for item in value)
        elif is_dataclass(field_info.type) and isinstance(value, dict):
            value = _dataclass_from_dict(field_info.type, value)
        kwargs[field_info.name] = value
    return cls(**kwargs)
```

注意 `field_info.type` 在 `from __future__ import annotations` 下是字符串。需要用 `typing.get_type_hints(cls)` 把它解析回真实类，或者维护一个 `nested_dataclass_fields: dict[str, type]` 注册表。M4.0 验收标准是 `tests/test_phase5_config.py` 全绿 + 新增 `test_phase5_config_nested.py`。

### 4.3 `MemoryConfig` 扩展

```python
@dataclass(frozen=True, slots=True)
class CorpusRagConfig:
    """v3 corpus RAG 配置。"""

    # ----- 总开关 -----
    enabled: bool = False                   # 全局开关；mode=disabled 时强制 false
    task_corpus: bool = True                # L1：当前 task 的文档
    shared_corpus: bool = False             # L2：维护方策划的只读语料
    shared_collections: tuple[str, ...] = ()
    shared_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "corpus_shared"
    )

    # ----- 索引 -----
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    max_chunks_per_doc: int = 200
    max_docs_per_task: int = 100

    # ----- Embedding -----
    embedder_backend: Literal["sentence_transformer", "stub"] = "sentence_transformer"
    embedder_model_id: str = "microsoft/harrier-oss-v1-270m"
    embedder_device: Literal["cpu", "cuda", "auto"] = "cpu"
    embedder_dtype: Literal["float32", "float16", "auto"] = "auto"
    embedder_query_prompt_name: str = "web_search_query"
    embedder_max_seq_len: int = 1024
    embedder_batch_size: int = 8
    embedder_cache_dir: Path | None = None  # None → 走 HF 默认；评测态指向镜像内固定路径

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

挂到 `MemoryConfig`：

```python
@dataclass(frozen=True, slots=True)
class MemoryConfig:
    mode: str = "disabled"
    store_backend: str = "jsonl"
    path: Path = field(...)
    retriever_type: str = "exact"
    retrieval_max_results: int = 5
    rag: CorpusRagConfig = field(default_factory=CorpusRagConfig)
```

#### 三态运行开关与 RAG 的交叉

| `memory.mode` | `rag.enabled` | 实际行为 |
|---|---|---|
| `disabled` | * | RAG 全关；`recall_corpus_*` 直接 no-op |
| `read_only_dataset` | `false` | dataset facts 召回，corpus 全关 |
| `read_only_dataset` | `true` | dataset facts + corpus 召回（task / shared 视子开关） |
| `full` | `false` | dataset facts + tool playbook 写入；corpus 全关 |
| `full` | `true` | 加上 corpus 召回 |

注意：**RAG 没有跨 task 写入路径**。`shared_corpus` 也只通过离线 `ingest_cli` 写；评测时 `ChromaCorpusStore` 持久化模式会以 read-only 标志打开。

### 4.4 文档与 Chunk Dataclass（`memory/rag/documents.py`）

```python
@dataclass(frozen=True, slots=True)
class CorpusDocument:
    doc_id: str                                          # sha1(source_path + size + mtime)[:16]
    source_path: str                                     # context 内相对路径
    doc_kind: Literal["markdown", "text", "doc", "json_schema_note"]
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
- `chunk_id` 全局唯一且可由 `doc_id + ord` 重建，方便 chroma upsert 幂等。

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

1. **doc / query 分离**。Harrier 系列 query 端必须加 instruction prompt（`web_search_query` 等），doc 端不加。Protocol 强制把这两个职责分开，避免调用方混淆。
2. **返回 list[list[float]] 而不是 numpy array**。简化 picklability、跨进程传输。
3. **L2 归一化由 Embedder 实现负责**。`SentenceTransformer(... normalize_embeddings=True)` 默认开启；Harrier 模型内部已含归一化，外部再 normalize 是 no-op 但保险。

### 4.6 `HarrierEmbedder`（`memory/rag/embedders/sentence_transformer.py`）

```python
class HarrierEmbedder:
    """sentence-transformers 后端，默认承载 microsoft/harrier-oss-v1-270m。"""

    def __init__(self, cfg: CorpusRagConfig) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟 import

        device = self._resolve_device(cfg.embedder_device)
        dtype = self._resolve_dtype(cfg.embedder_dtype)

        self._model = SentenceTransformer(
            cfg.embedder_model_id,
            device=device,
            cache_folder=str(cfg.embedder_cache_dir) if cfg.embedder_cache_dir else None,
            model_kwargs={"dtype": dtype} if dtype != "auto" else {},
        )
        self._model.max_seq_length = min(self._model.max_seq_length, cfg.embedder_max_seq_len)
        self._batch_size = cfg.embedder_batch_size
        self._query_prompt_name = cfg.embedder_query_prompt_name
        self._dim = self._model.get_sentence_embedding_dimension()

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

- **延迟 import**：`sentence_transformers` 在模块顶层 import 会触发 torch 加载，对 `disabled` 路径不友好。改成方法内 import。
- **离线模式**：`HF_HOME` / `TRANSFORMERS_OFFLINE=1` 由容器入口设置；`cache_folder` 走 `cfg.embedder_cache_dir`，submission 时显式指向镜像内固定路径（如 `/opt/models/harrier-270m`）。
- **CUDA OOM 容错**：`embed_documents` 在 OOM 时 fall back 到 cpu，并触发 `memory_rag_skipped(reason="cuda_oom")`。

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

1. 扫描 task 输入目录或 shared 目录。
2. 跳过 `is_safe_filename=False` 的文件。
3. 解析 markdown / text / doc / json schema 注释。
4. 截断超过 `max_docs_per_task` 的文件数。
5. 返回 `list[CorpusDocument]`。

不做：

- 不解析二进制（pdf / docx / xlsx）—— 容器内不一定有解析器，且容易触发兼容性问题。M4.5 之后再单独提案。
- 不调用网络 / 子进程（避免破坏 picklability 与 sandbox）。

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
- 不破坏 markdown 段落语义优先（如果当前位置正好在 ` ` 分段点，优先在此处切）。
- 每个 doc 最多产出 `max_chunks_per_doc`，超过截断并 dispatch 一个 `memory_rag_skipped(reason="max_chunks_truncated")`。

### 4.9 `ChromaCorpusStore`（`memory/rag/stores/chroma.py`）

```python
class ChromaCorpusStore:
    """实现 v2 MemoryStore 协议；额外提供向量 upsert / query。"""

    @classmethod
    def ephemeral(cls, namespace: str, embedder: Embedder, *, distance: str = "cosine") -> "ChromaCorpusStore":
        import chromadb
        client = chromadb.EphemeralClient()
        return cls(client, namespace=namespace, embedder=embedder, distance=distance)

    @classmethod
    def persistent_readonly(cls, path: Path, namespace: str, embedder: Embedder, *, distance: str = "cosine") -> "ChromaCorpusStore":
        import chromadb
        client = chromadb.PersistentClient(path=str(path))
        # 评测路径下，store 拒绝 put / delete
        return cls(client, namespace=namespace, embedder=embedder, distance=distance, readonly=True)

    # ----- v2 MemoryStore Protocol -----
    def put(self, record: MemoryRecord) -> None: ...           # readonly 时 raise
    def get(self, namespace: str, record_id: str) -> MemoryRecord | None: ...
    def list(self, namespace: str, *, limit: int = 100) -> list[MemoryRecord]: ...
    def delete(self, namespace: str, record_id: str) -> None: ...

    # ----- corpus 专属：批量 upsert 与向量 query -----
    def upsert_chunks(self, chunks: Sequence[CorpusChunk]) -> None: ...
    def query_by_vector(self, vector: Sequence[float], *, k: int) -> list[RetrievalResult]: ...
```

要点：

- **collection 命名**：`memory.rag.{namespace_safe}`。`namespace_safe` 把 `:` `/` 转 `__`，避免 Chroma collection name 限制。
- **embedding function**：Chroma 默认会自带 OpenAI embedding，必须**显式禁用**，使用 `embedding_function=None` + 手动 `add(embeddings=...)`，避免触发外部网络。
- **元数据**：每个 chunk 在 Chroma 中保存 `doc_id`、`ord`、`source_path`、`doc_kind`、`char_offset`、`char_length`。**不保存** `text` 之外的自由文本。
- **distance**：cosine 是默认。Harrier 已 L2 归一化，`ip` 也等价。

### 4.10 `VectorCorpusRetriever`（`memory/rag/retrievers/vector.py`）

```python
class VectorCorpusRetriever:
    """实现 v2 Retriever 协议。"""

    def __init__(self, store: ChromaCorpusStore, embedder: Embedder, *, k: int) -> None: ...

    def retrieve(self, query: str, *, namespace: str, k: int | None = None) -> list[RetrievalResult]:
        actual_k = k if k is not None else self._k
        if actual_k <= 0:
            return []
        vector = self._embedder.embed_query(query)
        results = self._store.query_by_vector(vector, k=actual_k)
        return [
            RetrievalResult(record=r.record, score=r.score, reason="vector_cosine")
            for r in results
        ]
```

- 召回返回的 `RetrievalResult.reason="vector_cosine"`。
- 召回时**不再对 query 文本做 redact 过滤**，因为 query 不会进入持久化 store；但调用方（`recall_corpus_*`）必须保证 query 仅来自当前 task 内合法字段。

### 4.11 `factory`（`memory/rag/factory.py`）

```python
def build_embedder(cfg: CorpusRagConfig) -> Embedder: ...

def build_task_corpus(cfg: CorpusRagConfig, *, task_id: str, task_input_dir: Path,
                      embedder: Embedder) -> tuple[ChromaCorpusStore, VectorCorpusRetriever] | None:
    """子进程入口构建 per-task corpus；rag 关闭时返回 None。"""

def load_shared_corpus(cfg: CorpusRagConfig, *,
                       embedder: Embedder) -> dict[str, VectorCorpusRetriever]:
    """评测态加载离线构建好的 shared corpus；只读。"""
```

工厂返回的 store / retriever / embedder **通过 contextvar 暴露**给 graph node，不进 `RunState`。沿用 v2 `runtime/context.py` 现有机制。

### 4.12 `RunState` 与 prompt 注入

#### 不修改 `RunState`

v2 已有 `memory_hits: Annotated[list[MemoryHit], operator.add]`。corpus 召回的 hits 同样写到这个字段，仅靠 `MemoryHit.summary` 与 `MemoryHit.namespace` 区分来源。**不新增**字段。

##### `MemoryHit.summary` 渲染规则（corpus 专属）

```text
[<doc_kind>] <source_path>: <text 摘要 ≤ 240 字符>
```

例如：

```text
[markdown] task_344/README.md: transaction_date 字段以 YYYYMMDD 整数存储，应当解析为 UTC 日期...
```

- `text` 摘要由 `chunk.text[:240]` + `...`（截断时）构成。
- 不渲染 `chunk_id` / `doc_id` / `score` / `embedding` 等内部字段。
- 由 `agents/prompts.py` 的新增 `render_corpus_snippets` 函数统一处理。

#### `agents/prompts.py` 新增渲染函数

```python
def render_corpus_snippets(hits: list[MemoryHit], budget_chars: int) -> str:
    """白名单：仅渲染 hit.summary；按 budget 截断；保持稳定排序。"""
```

`build_planning_messages` / `build_plan_solve_execution_messages` / `build_react_messages` 在 dataset facts 段之后追加：

```text
## Reference snippets (from task documentation)
- [markdown] task_344/README.md: ...
- [text] task_344/data_schema.md: ...
```

明确文案：「仅供参考，不替代 read_csv / inspect_sqlite_schema 等真实工具。」

### 4.13 `agents/corpus_recall.py`

与 v2 `agents/memory_recall.py` 并列：

```python
def recall_corpus_snippets(
    cfg: CorpusRagConfig,
    *,
    task_id: str,
    query: str,
    node: str,
    config: RunnableConfig,
) -> list[MemoryHit]: ...
```

实现：

1. 从 contextvar 拿 task / shared retriever。
2. 没拿到（rag 关闭、加载失败）→ 返回 `[]`。
3. `task` 与 `shared` 各召回 `retrieval_k` 条；按 score 合并、按 `prompt_budget_chars` 截断。
4. dispatch `memory_recall(kind="corpus_task" | "corpus_shared", ...)`。
5. 返回 `list[MemoryHit]`。

`planner_node` / `model_node` 在调 `recall_dataset_facts` 之后立即调 `recall_corpus_snippets`，并把两组 hits 合并写回 partial state。

### 4.14 RAG 边界（v2 4.11 占位 → v3 落地）

- `memory/rag/__init__.py` 不再抛 `NotImplementedError`，正式导出 `CorpusDocument` / `CorpusChunk` / `Embedder` / `HarrierEmbedder` / `ChromaCorpusStore` / `VectorCorpusRetriever` / `build_*`。
- corpus 不写入跨 task 自由文本；写入路径**只**在离线 `ingest_cli` 中存在。
- 未来 Phase M5 引入混合检索（BM25 + vector + RRF）只需新增 `retrievers/hybrid.py`，不改图层。

---

## 五、读写数据流

### 5.1 离线 ingest（仅 shared corpus）

```text
ingest_cli.py
  ├─ loader.scan(shared_path)
  ├─ redactor.filter(documents)
  ├─ chunker.chunk(documents)
  ├─ embedder.embed_documents(chunks.text)
  └─ ChromaCorpusStore.persistent_readonly(...)  # 写入 sqlite
       └─ upsert_chunks(chunks)
```

构建产物：

```
assets/corpus_shared/
  index/
    chroma.sqlite3
    ...
  manifest.json                    # 收录的文档清单 + 嵌入模型 id + 维度 + 构建时间
```

打包进镜像后评测态只读加载。

### 5.2 子进程入口（`runner` 启动 task 子进程时）

伪代码：

```text
runner._run_task_in_subprocess(task_id, app_config_dict):
    app_config = AppConfig.from_dict(app_config_dict)
    ...
    if app_config.memory.mode != "disabled" and app_config.memory.rag.enabled:
        embedder = factory.build_embedder(app_config.memory.rag)
        if app_config.memory.rag.task_corpus:
            task_pair = factory.build_task_corpus(
                app_config.memory.rag,
                task_id=task_id,
                task_input_dir=...,
                embedder=embedder,
            )
        if app_config.memory.rag.shared_corpus:
            shared_retrievers = factory.load_shared_corpus(app_config.memory.rag, embedder=embedder)
        runtime.context.set_corpus_handles(embedder, task_pair, shared_retrievers)
    graph.invoke(state)
```

事件：`memory_rag_index_built(task_id, doc_count, chunk_count, elapsed_ms, model_id, dimension)`。

### 5.3 召回路径（`planner_node` / `model_node`）

```text
planner_node:
    state ← MERGE(state, recall_dataset_facts(...))   # v2 已有
    state ← MERGE(state, recall_corpus_snippets(...)) # v3 新增

model_node:
    同上
```

`query` 选取：

- `planner_node` 用 `task.question`。
- `model_node` 用 `last_thought` 或最近一步 tool args 的字符串拼接（≤ 256 字符）。

### 5.4 Prompt 注入

`build_*_messages` 在 dataset facts 段后追加：

```text
## Reference snippets (from task documentation)
- [markdown] task_344/README.md: ...
- [text] task_344/data_schema.md: ...
```

后续紧接现有 user / system messages，不改变 message 顺序约定。

### 5.5 与 gate / dataset_facts 的关系

- corpus 召回**不**置 `preview_done` / `discovery_done`。
- corpus snippets 不替代 dataset_knowledge。它们各自走自己的 namespace、各自渲染。
- 同一段 prompt 内：`dataset facts` 在前、`corpus snippets` 在后；总长度受 `prompt_budget_chars` 控制。

---

## 六、安全边界

### 6.1 数据流禁令

- **跨 task 不持久化任何 `CorpusChunk`**：L1 索引随 task 子进程结束释放（`EphemeralClient`）。
- **shared corpus 评测路径只读**：`ChromaCorpusStore.persistent_readonly` 在评测 mode 下硬关 `put` / `delete`。
- **不存在跨 task corpus 写入路径**：`MemoryWriter` Protocol **不**新增 `write_corpus_*`。仅离线 `ingest_cli` 才能写入 shared corpus。

### 6.2 内容过滤

`Redactor`：

- 命中 `redact_patterns` 的整段直接丢弃。
- 命中 `redact_filenames` 的文件直接 skip。
- 任务输入目录中的 `expected_output.json`、`*label*`、`*solution*`、`ground_truth*` 一律 skip。

### 6.3 picklability

- Embedder / Store / Retriever 通过 contextvar 持有；不进 `RunState`。
- `MemoryHit.summary` 字段保持 `str`，不含任何句柄或 numpy。
- `CorpusRagConfig` 全部基本类型 + `Path`，picklable。

### 6.4 Fail closed

- `memory.mode=disabled` → RAG 强制关闭。
- Embedder 加载失败 → `memory_rag_skipped(reason="embedder_load_failed")`，task 继续。
- 索引构建失败 → 同上。
- Chroma query 异常 → 召回返回 `[]`，task 继续。
- query embed 失败 → 同上。

### 6.5 Resource budget

- `embedder_max_seq_len` 限制 token 数。
- `embedder_batch_size` 限制单次 forward 的 batch。
- `max_chunks_per_doc` / `max_docs_per_task` 限制单 task 索引规模。
- 索引构建超过 `task_corpus_index_timeout_s`（默认 30s）→ skip。

---

## 七、观测与可审计

### 7.1 复用 `memory_recall`

新增字段：

- `kind`：`dataset_knowledge | corpus_task | corpus_shared`
- `query_digest`：sha1(query)[:8]，**不存原文**
- `hit_doc_ids`、`hit_chunk_ids`、`scores`
- `model_id`：召回时使用的 embedder model id（仅 corpus 类）

### 7.2 新增事件

- `memory_rag_index_built`：`{ "scope": "task"|"shared", "namespace": ..., "doc_count": ..., "chunk_count": ..., "elapsed_ms": ..., "model_id": ..., "dimension": ... }`
- `memory_rag_skipped`：`{ "scope": ..., "reason": ..., "elapsed_ms": ... }`

所有事件经 `dispatch_observability_event(...)` 落 trace；MetricsCollector 通过 callbacks 汇总到 metrics.json。

### 7.3 新增 metrics 字段

`metrics.json` 增加：

```json
"memory_rag": {
  "task_index_built": true,
  "task_doc_count": 12,
  "task_chunk_count": 48,
  "shared_collections_loaded": 1,
  "recall_count": {"planner": 1, "model": 6},
  "skipped": []
}
```

仅当 RAG enabled 时出现，避免 baseline metrics 漂移。

---

## 八、容器与离线打包

### 8.1 Submission 镜像构建步骤

1. 在 Dockerfile 中：
   ```dockerfile
   ENV HF_HOME=/opt/models/hf
   ENV TRANSFORMERS_OFFLINE=1
   RUN python -c "from sentence_transformers import SentenceTransformer; \
                  SentenceTransformer('microsoft/harrier-oss-v1-270m', cache_folder='/opt/models/hf')"
   ```
2. 评测态 yaml 显式指定：
   ```yaml
   memory:
     mode: read_only_dataset
     rag:
       enabled: true
       embedder_cache_dir: /opt/models/hf
       embedder_device: cpu
       embedder_dtype: float32
   ```
3. shared corpus 离线构建后，把 `assets/corpus_shared/index/` 一起打包。

### 8.2 体积与依赖估算

| 项 | 增量 |
|---|---|
| `sentence-transformers` + `transformers` + `tokenizers` | ~600 MB |
| Harrier 270m fp32 权重 | ~1.1 GB |
| Harrier 270m fp16 权重 | ~540 MB |
| `chromadb` | ~150 MB |
| shared corpus index | 取决于语料；通常 < 200 MB |

合计：fp16 路线下 + ~1.5 GB；fp32 路线下 + ~2 GB。需要确认竞赛镜像大小限制。

### 8.3 评测态默认值（`submission.build_submission_config`）

- `memory.mode = "read_only_dataset"`（不变）。
- `memory.rag.enabled = false`（M4 上线初期保守关闭，A/B 测试通过后再开）。
- 通过 env `DATA_AGENT_RAG=1` 一键开启。

---

## 九、风险与待决问题

### 9.1 已识别风险

1. **CPU 上 270m decoder-only 编码速度**：未实测，需在 M4.4 用 task_344 这种 task 实测，目标：单 task 索引 ≤ 5s，单次 query embed ≤ 100ms。
2. **chroma 离线模式**：必须确认 `EphemeralClient` 不会触发外部 telemetry / network；设置 `anonymized_telemetry=False`。
3. **token 预算冲突**：corpus snippets 与 dataset facts 共用 prompt 末尾。需要在 M4.5 实测，看是否会挤掉 ReAct 的 working memory。
4. **chromadb Windows 安装**：开发机偶发 SQLite 锁问题。M4.0 必须验证 Windows + ephemeral 路径无 issue。
5. **镜像体积超限**：fp16 + chromadb 增量 ~1.5 GB；如果竞赛镜像有上限（如 10 GB / 20 GB），需提前确认。

### 9.2 待决问题（M4 完成前需回答）

- Q1：评测容器是否允许使用 GPU？如果允许，是否可信 `embedder_device=auto`？默认假设 **CPU only**。
- Q2：shared corpus 是否真的需要？M4.6 是可选项，可以推迟到 M4 完成后再评估。
- Q3：是否需要在 `tool_node` 入口也召回？默认不做，仅 `planner_node` / `model_node` 召回。

---

## 十、与 v2 的差异回顾

| 维度 | v2（已落地） | v3（M4） |
|---|---|---|
| 数据形态 | 结构化 dataclass | 自由文本 chunk |
| 索引 | 无（exact namespace） | 向量（cosine） |
| 检索 | namespace 精确召回 | 向量相似度召回 |
| 写入 | tool_node 成功路径 | **无运行时写入**（仅离线 ingest） |
| 跨 task 持久化 | `dataset_knowledge` JSONL | shared corpus chroma 持久层 |
| 单 task 索引 | 不存在 | ephemeral chroma + Harrier |
| 召回事件 | `memory_recall` | 同；新增 `kind=corpus_*` |
| RunState 影响 | `memory_hits` 字段 | 不变；只增加 hit |
| Prompt 注入 | `render_dataset_facts` | 新增 `render_corpus_snippets` |

v3 不破坏 v2 任何已有契约，是纯增量扩展。
