# Corpus RAG 实施计划 v3.1（M4.0–M4.5，v2 修订）

> 版本：v3.1，2026-05-14
> 设计：`01-design-v2.md`（必读前置）
> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 逐任务推进。步骤用 checkbox（`- [ ]`）跟踪。

**Goal:** 按 `01-design-v2.md` 在 `memory/rag/` 子包内落地基于 **Harrier-OSS-v1-270m + chromadb（ephemeral 模式）** 的 corpus RAG（M4），并把 per-task corpus 接入 task 入口（plan-solve: `planner_node`；ReAct: 新增 `task_entry_node`）。

**Architecture（v2 决策）:** Embedder / ChromaCorpusStore / VectorCorpusRetriever 三层解耦；`MemoryConfig` 新增 `rag: CorpusRagConfig` 嵌套子配置；`RunState` 不变，复用现有 `memory_hits`；`agents/corpus_recall.py` 与 v2 `agents/memory_recall.py` 并列；`agents/prompts.py` 新增 `render_corpus_snippets`；observability 复用 `memory_recall` 事件并新增 `memory_rag_index_built` / `memory_rag_skipped`。

**关键决策：**
- **A2**：仅 task 入口召回 1 次（plan-solve 走 planner_node；ReAct 走新增 task_entry_node）；model_node 不主动召回。
- **B2**：`CorpusStore` 独立 Protocol（不复用 v2 `MemoryStore.put/get/list/delete`）。
- **C2**：`sentence-transformers` / `chromadb` 仅进 `[project.optional-dependencies].rag`，且 rag 子模块**禁止顶层 import 重依赖**。
- **D2**：`shared_corpus` 拆到 `05-shared-corpus-design.md`；本计划仅占位 + fail-closed。

**Tech Stack:** Python 3.10+，dataclasses（`frozen=True, slots=True`），LangGraph 0.4，LangChain core，`sentence-transformers`，`transformers`，`torch`，`chromadb`，pytest，typer CLI。

**前置依赖:** v2 已合入主干（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/memory/rag/__init__.py:1-10` 为占位）。

---

## File Structure

新增文件：

- `src/data_agent_langchain/memory/rag/documents.py`
- `src/data_agent_langchain/memory/rag/loader.py`
- `src/data_agent_langchain/memory/rag/chunker.py`
- `src/data_agent_langchain/memory/rag/redactor.py`
- `src/data_agent_langchain/memory/rag/base.py`（`CorpusStore` Protocol）
- `src/data_agent_langchain/memory/rag/embedders/__init__.py`
- `src/data_agent_langchain/memory/rag/embedders/base.py`
- `src/data_agent_langchain/memory/rag/embedders/sentence_transformer.py`
- `src/data_agent_langchain/memory/rag/embedders/stub.py`
- `src/data_agent_langchain/memory/rag/stores/__init__.py`
- `src/data_agent_langchain/memory/rag/stores/chroma.py`
- `src/data_agent_langchain/memory/rag/retrievers/__init__.py`
- `src/data_agent_langchain/memory/rag/retrievers/vector.py`
- `src/data_agent_langchain/memory/rag/factory.py`
- `src/data_agent_langchain/agents/corpus_recall.py`
- `src/data_agent_langchain/agents/task_entry_node.py`（ReAct 入口召回节点）
- 所有测试统一放在 `tests/test_phase10_rag_*.py`。

**不新增**（D2）：

- ~~`src/data_agent_langchain/memory/rag/ingest_cli.py`~~（独立提案）
- ~~`ChromaCorpusStore.persistent_*`~~（独立提案）

修改文件：

- `src/data_agent_langchain/config.py` —— 让 `_dataclass_from_dict` 支持嵌套 + `tuple[T,...]` + `Optional[Path]`；新增 `CorpusRagConfig` 并嵌入 `MemoryConfig`。
- `src/data_agent_langchain/memory/rag/__init__.py` —— 去掉 `NotImplementedError`，仅导出纯类型（**不**顶层 import HarrierEmbedder / ChromaCorpusStore）。
- `src/data_agent_langchain/runtime/context.py` —— 新增 corpus handles contextvar。
- `src/data_agent_langchain/run/runner.py` —— 子进程入口构建 task corpus + 写 contextvar。
- `src/data_agent_langchain/agents/planner_node.py` —— 在 `recall_dataset_facts` 之后追加 `recall_corpus_snippets`。
- `src/data_agent_langchain/agents/react_graph.py` —— `START → task_entry → execution → finalize`。
- `src/data_agent_langchain/agents/model_node.py` —— `_build_messages_for_state` 内串行渲染 dataset_facts + corpus_snippets。
- `src/data_agent_langchain/agents/prompts.py` —— 新增 `render_corpus_snippets`。
- `src/data_agent_langchain/cli.py` —— 新增 `--memory-rag/--no-memory-rag` 开关。
- `src/data_agent_langchain/submission.py` —— 评测态默认 `rag.enabled=false`，env `DATA_AGENT_RAG=1` 显式打开。
- `src/data_agent_langchain/observability/metrics.py` —— 订阅 `memory_rag_index_built` / `memory_rag_skipped` 聚合到 `metrics.json.memory_rag`。
- `pyproject.toml` —— 新增 `[project.optional-dependencies].rag` 组与 `[tool.pytest.ini_options]` 的 `slow` marker。
- `tests/conftest.py` —— 新增 `--runslow` flag 与 `slow` marker skip 逻辑。

---

## Conventions（所有任务通用）

- 每个 TDD 步骤必须按 **RED → GREEN → COMMIT** 推进；不要把多个测试合并到同一次提交。
- 测试统一用 `tests/test_phase10_rag_*.py` 前缀（v2 用 phase 9；RAG 为 phase 10）。
- 提交信息：`feat(memory-rag): <task>` / `test(memory-rag): <task>` / `refactor(memory-rag): <task>` / `chore(memory-rag): <task>`。
- 跑测试：`pytest tests/test_phase10_rag_*.py -q`；单测：`pytest tests/test_phase10_rag_chunker.py::test_name -v`。
- **不引入对真实 Harrier 权重的强依赖**：除了 M4.2 的一个 opt-in 集成测试外，单测一律用 `DeterministicStubEmbedder`。
- **不允许在 graph node 循环中加载模型或构建索引**。
- 所有 dataclass 新增字段保持 `frozen=True, slots=True` 与 picklable。
- 不允许在写入路径接收 `question` / `answer` / `approach` / `hint` / `summary` 等字段。
- `chromadb` 客户端必须传 `Settings(anonymized_telemetry=False)`，避免外部网络。
- **rag 子模块顶层禁止 import `torch` / `chromadb` / `sentence_transformers`**（M4.6 单独回归测试守护）。
- **所有实施代码的注释必须使用中文**（与现有代码基风格一致，例：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/config.py:38-86` 全部中文 docstring + 行内注释）：
  - 模块 / 类 / 函数 docstring 用中文（含参数说明、返回值说明、副作用说明）。
  - 行内 `#` 注释用中文。
  - 测试用例 docstring 也用中文，描述测试断言的语义。
  - 例外：保留专有名词原文（如 `HF` / `cosine distance` / `ephemeral` / `EphemeralClient` / `sentence-transformers` / `chromadb` / `LangGraph` 等技术术语，不强行翻译）。
  - 提交信息（commit message）仍按 conventional commits 英文 type 前缀 + 中英任一（与项目现状一致；中文 message body 优先）。

---

# Milestone M4.0：嵌套 dataclass 支持 + pytest marker（config 前置）

## Task M4.0.1：让 `_dataclass_from_dict` 支持嵌套与新类型

**Files:**
- Modify: `src/data_agent_langchain/config.py`
- Create: `tests/test_phase10_rag_config_nested.py`

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - `test_nested_dataclass_roundtrip`：构造临时 `@dataclass class Outer: inner: Inner = field(default_factory=Inner)`，断言 `to_dict / from_dict` 互逆。
  - `test_tuple_str_field_roundtrip`：`tuple[str, ...]` 字段从 list 还原回 tuple。
  - `test_optional_path_field_none`：`Path | None` 字段 None 时不变。
  - `test_optional_path_field_value`：`Path | None` 字段 str 时自动转 Path。
  - 现有 `tests/test_phase5_config.py` 全部仍通过。
- [ ] **Step 2（GREEN）**：按 `01-design-v2.md §4.2` 改写 `_dataclass_from_dict`，引入 `_coerce_field` 辅助函数；用 `typing.get_type_hints(cls)` 解析类型；显式处理 `tuple[T, ...]` 与 `Optional[Path]`；嵌套 dataclass 递归。
- [ ] **Step 3（验证）**：
  ```powershell
  pytest tests/test_phase5_config.py tests/test_phase10_rag_config_nested.py -q
  ```
- [ ] **Step 4（COMMIT）**：`refactor(config): support nested dataclass, tuple, Optional in from_dict`。

**Verification:**
- `tests/test_phase5_config.py` 全部回归通过。
- 新增嵌套测试通过。

## Task M4.0.2：pyproject.toml `slow` marker + conftest `--runslow`

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`

**Steps:**

- [ ] **Step 1（RED）**：写 `tests/test_phase10_rag_slow_marker.py`，断言：
  - 默认 `pytest tests/test_phase10_rag_slow_marker.py -q` 不跑标 `@pytest.mark.slow` 的用例。
  - `pytest tests/test_phase10_rag_slow_marker.py --runslow -q` 跑。
- [ ] **Step 2（GREEN）**：
  - `pyproject.toml` 加：
    ```toml
    [tool.pytest.ini_options]
    testpaths = ["tests"]
    markers = ["slow: opt-in slow integration tests; require --runslow"]
    ```
  - `tests/conftest.py` 加：
    ```python
    def pytest_addoption(parser):
        parser.addoption("--runslow", action="store_true", default=False, help="run @pytest.mark.slow tests")

    def pytest_collection_modifyitems(config, items):
        if config.getoption("--runslow"):
            return
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    ```
- [ ] **Step 3（COMMIT）**：`chore(test): add slow marker and --runslow opt-in flag`。

---

# Milestone M4.1：Corpus 类型 + Loader + Chunker + Redactor

## Task M4.1.1：`CorpusDocument` / `CorpusChunk`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/documents.py`
- Create: `tests/test_phase10_rag_documents.py`

**Steps:**

- [ ] **Step 1（RED）**：写 `test_corpus_chunk_is_frozen_and_picklable`，断言：
  - `dataclasses.fields(CorpusChunk)` 包含 `chunk_id` / `doc_id` / `ord` / `text` / `char_offset` / `char_length`。
  - 实例 `pickle.dumps` 成功。
  - 字段 `frozen` —— 赋值抛 `FrozenInstanceError`。
  - 测试同样覆盖 `CorpusDocument`。
- [ ] **Step 2（GREEN）**：实现 `CorpusDocument` / `CorpusChunk`，均 `frozen=True, slots=True`。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add corpus document/chunk dataclasses`。

## Task M4.1.2：`Redactor`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/redactor.py`
- Create: `tests/test_phase10_rag_redactor.py`

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - `is_safe_filename("expected_output.json") is False`
  - `is_safe_filename("ground_truth_v1.csv") is False`
  - `is_safe_filename("README.md") is True`
  - `filter_text` 命中 `\banswer\b` 的整段被丢弃（返回空字符串）。
  - 多 pattern 命中其中之一即丢弃。
- [ ] **Step 2（GREEN）**：实现 `Redactor(cfg)`，基于 `cfg.redact_patterns`（正则）+ `cfg.redact_filenames`（fnmatch glob）。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add Redactor for filename and content filtering`。

## Task M4.1.3：`Loader`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/loader.py`
- Create: `tests/test_phase10_rag_loader.py`
- Test fixtures: 在 `tests/fixtures/rag/task_demo/context/` 下放 `README.md`、`data_schema.md`、`expected_output.json`（应被跳过）、`ground_truth.csv`（应被跳过）。

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - 扫描 fixture `context/` 目录返回 `[CorpusDocument(...)]`，仅含 README.md / data_schema.md。
  - `expected_output.json` 与 `ground_truth.csv` 不出现。
  - 文档数超过 `max_docs_per_task` 时截断并 dispatch `memory_rag_skipped(reason="max_docs_truncated")`。
  - 路径返回 `task context_dir` 内的相对路径（不含绝对 prefix）。
  - **不**接收 `task_dir`（与 `context_dir` 分离的目录）下的文件 —— 调用者只传 `context_dir`。
- [ ] **Step 2（GREEN）**：实现 `Loader.scan(context_dir) -> list[CorpusDocument]`，对每个文件计算 `doc_id = sha1(source_path + size + mtime)[:16]`。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add Loader with redact-aware filename filter`。

## Task M4.1.4：`Chunker`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/chunker.py`
- Create: `tests/test_phase10_rag_chunker.py`

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - 空文本返回 `[]`。
  - 单字符长度 < `chunk_size_chars` 返回单 chunk。
  - 长文本切片数 ≈ `ceil((char_count - overlap) / (size - overlap))`。
  - 超过 `max_chunks_per_doc` 截断并 dispatch `memory_rag_skipped(reason="max_chunks_truncated")`。
  - `chunk_id == f"{doc_id}#{ord:04d}"` 稳定可复现。
  - 相邻窗口重叠正确（窗口 N 末尾 `overlap` 字符等于窗口 N+1 开头）。
- [ ] **Step 2（GREEN）**：实现 `CharWindowChunker(cfg)`。优先在 `\n\n` 段落点处切（如果当前位置 ± 100 字符内有 `\n\n`，移动到该位置）。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add char-window chunker with overlap and caps`。

**Verification (M4.1 全部完成后):**
```powershell
pytest tests/test_phase10_rag_{documents,redactor,loader,chunker}.py -q
```
全绿。

---

# Milestone M4.2：Embedder（Protocol + Stub + Harrier）

## Task M4.2.1：`Embedder` Protocol + `DeterministicStubEmbedder`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/embedders/__init__.py`（只 re-export Embedder Protocol + Stub；**不** import HarrierEmbedder）
- Create: `src/data_agent_langchain/memory/rag/embedders/base.py`
- Create: `src/data_agent_langchain/memory/rag/embedders/stub.py`
- Create: `tests/test_phase10_rag_embedder_stub.py`

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - `DeterministicStubEmbedder(dim=8).dimension == 8`。
  - `DeterministicStubEmbedder(dim=8).model_id == "stub-deterministic-dim8"`。
  - 同一文本两次 `embed_documents` 返回相同向量。
  - 不同文本返回不同向量。
  - `embed_query(text)` 与 `embed_documents([text])[0]` 不一定相同（query 端注入 `"q:"` 前缀）。
  - 返回值是 `list[list[float]]`，不是 numpy。
  - 向量 L2 范数 ≈ 1（归一化）。
- [ ] **Step 2（GREEN）**：实现 `DeterministicStubEmbedder`：基于 `hashlib.sha256(text.encode()).digest()` 截取 `dim` 字节，归一化为单位向量；`embed_query` 在 text 前加 `"q:"` 前缀。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add Embedder protocol and deterministic stub`。

## Task M4.2.2：`HarrierEmbedder`（生产 backend）

**Files:**
- Create: `src/data_agent_langchain/memory/rag/embedders/sentence_transformer.py`
- Create: `tests/test_phase10_rag_embedder_harrier.py`（标注 `@pytest.mark.slow`，opt-in 跑）

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵（全部 `@pytest.mark.slow`）
  - `HarrierEmbedder(cfg).dimension > 0`。
  - `HarrierEmbedder(cfg).model_id == "microsoft/harrier-oss-v1-270m"`（验证 v1 bug 修复：`self._model_id` 必须赋值）。
  - `embed_documents([])` 返回 `[]`。
  - `embed_documents(["hello world"])` 返回长度 1 的 list；向量 L2 归一化（`abs(norm - 1.0) < 1e-3`）。
  - `embed_query("question")` 走 `prompt_name="web_search_query"`，向量也归一化。
  - `embed_documents` 在传入 100 段时不爆 batch（`embedder_batch_size=8` 下应分 13 batch）。
  - **顶层 import 边界**：单独写一个 `test_harrier_module_does_not_import_torch_at_module_load`：
    ```python
    def test_harrier_module_does_not_import_torch_at_module_load():
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import data_agent_langchain.memory.rag.embedders.sentence_transformer; "
             "import sys; "
             "assert 'torch' not in sys.modules, 'torch leaked at module import'"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    ```
    此测试**不**带 `@slow` marker，必跑。
- [ ] **Step 2（GREEN）**：实现 `HarrierEmbedder`：
  - `__init__` 内**方法级延迟 import** `sentence_transformers`、`torch`（仅在 `_resolve_device("auto")` 路径需要）。
  - 必须赋值 `self._model_id = cfg.embedder_model_id`（v1 bug 修复）。
  - `model_kwargs={"dtype": cfg.embedder_dtype}` 始终传，包括 `"auto"`（HF 推荐用法）。
  - 按 `cfg.embedder_device` / `cfg.embedder_dtype` 加载；`max_seq_length` 钳制到 `cfg.embedder_max_seq_len`。
- [ ] **Step 3（验证）**：在本地有 RAG extra 装好的 venv 上手动跑：
  ```powershell
  pytest tests/test_phase10_rag_embedder_harrier.py -q --runslow
  ```
  确认权重能从 `HF_HOME` 缓存离线加载。
- [ ] **Step 4（COMMIT）**：`feat(memory-rag): add HarrierEmbedder sentence-transformers backend`。

**Verification (M4.2 全部完成后):**
- `pytest tests/test_phase10_rag_embedder_stub.py -q` 全绿（默认）。
- `pytest tests/test_phase10_rag_embedder_harrier.py -q --runslow` 在本地手动验证一次。
- `pytest tests/test_phase10_rag_embedder_harrier.py::test_harrier_module_does_not_import_torch_at_module_load -q` 必跑且绿。

---

# Milestone M4.3：CorpusStore Protocol + ChromaCorpusStore（仅 ephemeral）+ VectorCorpusRetriever

## Task M4.3.1：`CorpusStore` Protocol（B2 决策）

**Files:**
- Create: `src/data_agent_langchain/memory/rag/base.py`
- Create: `tests/test_phase10_rag_corpus_store_protocol.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - 断言 `CorpusStore` 是 `@runtime_checkable Protocol`。
  - 断言 `CorpusStore` 仅定义 `namespace` / `dimension` / `upsert_chunks` / `query_by_vector` / `close` —— **不**含 `put` / `get` / `list` / `delete`（B2 决策守护）。
- [ ] **Step 2（GREEN）**：按 `01-design-v2.md §4.9` 实现 `CorpusStore` Protocol。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add CorpusStore protocol decoupled from MemoryStore`。

## Task M4.3.2：`ChromaCorpusStore.ephemeral`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/stores/__init__.py`（不顶层 import chroma）
- Create: `src/data_agent_langchain/memory/rag/stores/chroma.py`
- Create: `tests/test_phase10_rag_chroma_store.py`

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵（全程用 `DeterministicStubEmbedder`）
  - `ChromaCorpusStore.ephemeral(...)` 不写盘、不触发 telemetry（mock 验证 `Settings(anonymized_telemetry=False)` 已传）。
  - `upsert_chunks([CorpusChunk(...)])` 之后 `query_by_vector(vec, k=1)` 命中该 chunk。
  - 多 namespace 隔离：往 `corpus_task:A` 写，从 `corpus_task:B` 查得 `[]`。
  - `query_by_vector` 返回的 `RetrievalResult.record.kind == "corpus"`。
  - `query_by_vector(k=0)` 返回 `[]`，不调用 chroma。
  - **协议守护**：`hasattr(store, 'put')` 应当为 `False`（B2 决策）。
  - **顶层 import 边界**：`test_chroma_module_does_not_import_chromadb_at_module_load`（子进程 sys.modules 验证）。
- [ ] **Step 2（GREEN）**：实现 `ChromaCorpusStore.ephemeral`（方法级延迟 import）。`upsert_chunks` 走 `collection.upsert(ids=, embeddings=, documents=, metadatas=)`；`query_by_vector` 走 `collection.query(query_embeddings=[vec], n_results=k, include=[...])`。collection 名用 `sha1(namespace)[:16]`。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add ChromaCorpusStore ephemeral backend`。

## Task M4.3.3：`VectorCorpusRetriever`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/retrievers/__init__.py`
- Create: `src/data_agent_langchain/memory/rag/retrievers/vector.py`
- Create: `tests/test_phase10_rag_vector_retriever.py`

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - 命中场景：upsert 一组 chunks 后 retrieve 返回 `RetrievalResult(reason="vector_cosine")`，且 `record.payload["source_path"]` / `record.payload["doc_kind"]` 被回填（来自 doc_index）。
  - `k=0` 返回 `[]`，**不调用** embedder。
  - 负数 `k` clamp 到 0。
  - embedder 异常（mock `embed_query` 抛 `RuntimeError`）→ 返回 `[]`（事件由调用方 dispatch）。
  - store 异常 → 同上。
  - retriever 实现 v2 `Retriever` Protocol：`isinstance(retriever, Retriever) is True`。
- [ ] **Step 2（GREEN）**：实现 `VectorCorpusRetriever(store, embedder, doc_index, k)`。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add VectorCorpusRetriever with cosine similarity`。

**Verification (M4.3 全部完成后):**
```powershell
pytest tests/test_phase10_rag_corpus_store_protocol.py tests/test_phase10_rag_chroma_store.py tests/test_phase10_rag_vector_retriever.py -q
```
全绿。Windows 本地额外跑一遍验证 SQLite 锁问题（M4 风险 R3）。

---

# Milestone M4.4：Config 扩展 + Factory + Runner 集成 + pyproject extras

## Task M4.4.1：`CorpusRagConfig` + `MemoryConfig.rag`

**Files:**
- Modify: `src/data_agent_langchain/config.py`
- Create: `tests/test_phase10_rag_config.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - 默认 `CorpusRagConfig()` 字段值符合 `01-design-v2.md §4.3`。
  - `MemoryConfig().rag is not None`，是 `CorpusRagConfig` 实例。
  - `MemoryConfig.path` 默认值仍是 `PROJECT_ROOT / "artifacts" / "memory"`（D7 守护：default_factory 没被截掉）。
  - `AppConfig.to_dict / from_dict` 包含 `memory.rag` 字段且互逆。
  - `tuple[str, ...]` 字段 round-trip 正确（从 list 还原回 tuple）。
  - YAML 解析：从临时 yaml 读 `memory.rag.enabled=true` 走通。
- [ ] **Step 2（GREEN）**：在 `config.py` 新增 `CorpusRagConfig` 与 `MemoryConfig.rag`。注意：必须在 `MemoryConfig` 定义之前 import `Literal`。
- [ ] **Step 3（COMMIT）**：`feat(config): add MemoryConfig.rag CorpusRagConfig nested config`。

## Task M4.4.2：`pyproject.toml` 加 `[project.optional-dependencies].rag`（C2 决策）

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_phase10_rag_packaging.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - `test_rag_extra_declared_but_not_in_base_dependencies`：parse `pyproject.toml`，断言：
    - `project.dependencies` 不含 `sentence-transformers` / `chromadb` / `torch`。
    - `project.optional-dependencies.rag` 含 `sentence-transformers` 与 `chromadb`。
- [ ] **Step 2（GREEN）**：在 `pyproject.toml` 加：
  ```toml
  [project.optional-dependencies]
  rag = [
      "sentence-transformers>=3,<4",
      "chromadb>=0.5,<1",
  ]
  ```
- [ ] **Step 3（COMMIT）**：`chore(pyproject): add optional rag extra for sentence-transformers and chromadb`。

## Task M4.4.3：`factory.build_embedder` / `build_task_corpus`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/factory.py`
- Create: `tests/test_phase10_rag_factory.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - `build_embedder(cfg)`：
    - `cfg.enabled=False` 或 `cfg.embedder_backend` 缺失 → 返回 `None`。
    - `"stub"` 返回 `DeterministicStubEmbedder(dim=...)`。
    - `"sentence_transformer"` 返回 `HarrierEmbedder`（用 monkeypatch 把 `SentenceTransformer` 替换为假对象，避免加载真权重）。
    - `HarrierEmbedder` 构造失败（如 ImportError）→ 返回 `None` + dispatch `memory_rag_skipped(reason="embedder_load_failed")`。
  - `build_task_corpus(cfg, task_id, task_input_dir, embedder, config)`：
    - `cfg.enabled=False or cfg.task_corpus=False` → 返回 `None`。
    - 目录不存在 → 返回 `None` + dispatch `memory_rag_skipped(reason="no_documents")`。
    - fixture 目录 → 返回 `TaskCorpusHandles`，含 store / retriever / embedder；索引中 chunk 数等于 fixture 文档切片总数。
    - 构建成功 → dispatch `memory_rag_index_built(doc_count, chunk_count, model_id, dimension, elapsed_ms)`。
    - 超时（用 monkeypatch 强制 sleep 超过 `task_corpus_index_timeout_s`）→ 返回 `None` + `memory_rag_skipped(reason="index_timeout")`。
  - `shared_corpus=True` 时 dispatch `memory_rag_skipped(reason="shared_corpus_not_implemented")` 并返回 None（不抛）。
- [ ] **Step 2（GREEN）**：实现工厂；遵循「方法内延迟 import 重依赖」原则。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add factory for embedder and task corpus`。

## Task M4.4.4：Contextvar 与 runner 子进程入口

**Files:**
- Modify: `src/data_agent_langchain/runtime/context.py`
- Modify: `src/data_agent_langchain/run/runner.py`
- Create: `tests/test_phase10_rag_runner.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - `runtime/context.py` 新增 `set_current_corpus_handles / get_current_corpus_handles / clear_current_corpus_handles`。
  - 子进程入口：rag enabled 时，`get_current_corpus_handles()` 返回非 None；rag disabled 时返回 None。
  - 索引构建事件：`memory_rag_index_built` 落 trace，含 `doc_count` / `chunk_count` / `model_id` / `dimension`。
  - 失败 fail closed：embedder 加载失败时 `memory_rag_skipped` 落 trace，task 仍按 baseline 运行（图 invoke 不抛）。
  - parity：rag 关闭路径下 task_344 metrics 与 v2 baseline 一致（结构上）。
- [ ] **Step 2（GREEN）**：
  - `runtime/context.py` 增加 `corpus_handles` 的 `ContextVar`（仿 `_APP_CONFIG`）。
  - `runner._execute_task_in_subprocess`（即当前 `run_single_task` 的 contextvar 设置点，`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/run/runner.py:217-236`）在 `set_current_app_config` 之后、`compiled.invoke` 之前调用 `factory.build_embedder` + `factory.build_task_corpus`，把 handles 写 contextvar。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): build per-task corpus in runner subprocess entry`。

## Task M4.4.5：CLI / submission 接入

**Files:**
- Modify: `src/data_agent_langchain/cli.py`
- Modify: `src/data_agent_langchain/submission.py`
- Modify: `tests/test_phase10_rag_config.py`（追加用例）
- Modify: `tests/test_submission_config.py`（追加用例）

**Steps:**

- [ ] **Step 1（RED）**：
  - `dabench-lc run-task --memory-rag` 覆盖 `cfg.memory.rag.enabled=True`。
  - `dabench-lc run-task --no-memory-rag` 覆盖为 `False`。
  - `submission.build_submission_config()` 默认 `rag.enabled=False`。
  - env `DATA_AGENT_RAG=1` 时 `submission.build_submission_config()` 设 `rag.enabled=True`。
  - env `DATA_AGENT_RAG=0` 或 unset 时设 `False`。
- [ ] **Step 2（GREEN）**：
  - `cli.py` 增加 `--memory-rag/--no-memory-rag` typer option（仿 `--memory-mode`，用 `Optional[bool]`，None 表示不覆盖）。
  - `submission.py` 在构造 `MemoryConfig` 时读 `os.environ.get("DATA_AGENT_RAG") == "1"`。
- [ ] **Step 3（COMMIT）**：`feat(cli): add --memory-rag toggle for corpus RAG`。

**Verification (M4.4 全部完成后):**
```powershell
pytest tests/test_phase10_rag_config.py tests/test_phase10_rag_packaging.py tests/test_phase10_rag_factory.py tests/test_phase10_rag_runner.py tests/test_submission_config.py -q
```
全绿。本地手动跑：
```powershell
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --no-memory-rag
```
结果与 v2 baseline 完全一致（structural parity）。

---

# Milestone M4.5：召回 + Prompt 注入 + ReAct 入口节点（A2 决策）

## Task M4.5.1：`recall_corpus_snippets`

**Files:**
- Create: `src/data_agent_langchain/agents/corpus_recall.py`
- Create: `tests/test_phase10_rag_corpus_recall.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - rag 关闭：返回 `[]`，不调 retriever。
  - rag 开启 + 无 task corpus handles（contextvar 是 None）：返回 `[]`。
  - 命中场景：返回 `list[MemoryHit]`，`summary` 形如 `"[markdown] README.md: ..."`；`namespace` 形如 `"corpus_task:<task_id>"`。
  - dispatch `memory_recall(kind="corpus_task", query_digest=..., hit_chunk_ids=..., model_id=..., reason="vector_cosine")`。
  - retriever 异常 → 返回 `[]` 并 dispatch `memory_rag_skipped(reason="retrieve_failed")`。
  - `prompt_budget_chars` 严格遵守：累积 summary 长度不超出预算（如有溢出截断最末几条）。
  - 召回时 query 不被写入任何 store（用 spy 验证）。
  - `query_digest = sha1(query)[:8]`，原文不进 event。
- [ ] **Step 2（GREEN）**：实现 `recall_corpus_snippets(cfg, *, task_id, query, node, config)`，仿 v2 `recall_dataset_facts` 风格。从 `runtime.context.get_current_corpus_handles()` 拿 retriever；`MemoryHit.summary` 渲染为 `[<doc_kind>] <source_path>: <text[:240]>...`。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add recall_corpus_snippets helper`。

## Task M4.5.2：`render_corpus_snippets` + `model_node` 串行注入（D5 决策）

**Files:**
- Modify: `src/data_agent_langchain/agents/prompts.py`
- Modify: `src/data_agent_langchain/agents/model_node.py`
- Create: `tests/test_phase10_rag_prompts.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - `render_corpus_snippets([], budget=1800) == ""`。
  - 非空 hits 渲染输出包含 `## Reference snippets (from task documentation)` 段头。
  - 渲染只用 `hit.summary`；不出现 `record_id` / `namespace` / `score`。
  - **不修改 builder**：`build_planning_messages` / `build_plan_solve_execution_messages` / `build_react_messages` 输出长度与 v2 完全一致（regression）。
  - `model_node._build_messages_for_state(state)` 末尾的 HumanMessage 内容：
    - 仅 dataset_facts：等于 v2 现状。
    - 仅 corpus_snippets：等于 `render_corpus_snippets(...)` 单独输出。
    - 两者都有：dataset_facts 段在前，corpus_snippets 段在后，用 `\n\n` 分隔。
  - `model_node` 通过 `namespace.startswith("corpus_")` 区分两类 hits。
- [ ] **Step 2（GREEN）**：
  - `prompts.py` 新增 `render_corpus_snippets(hits, budget_chars)` —— 段头 + 每条 `- summary`，按 budget 累积截断。
  - `model_node._build_messages_for_state` 在 `hits = state["memory_hits"]` 之后：
    ```python
    dataset_hits = [h for h in hits if not h.namespace.startswith("corpus_")]
    corpus_hits  = [h for h in hits if h.namespace.startswith("corpus_")]
    facts_text    = render_dataset_facts(dataset_hits)
    snippets_text = render_corpus_snippets(
        corpus_hits, budget_chars=app_config.memory.rag.prompt_budget_chars
    )
    extra = "\n\n".join(s for s in (facts_text, snippets_text) if s)
    if extra:
        messages = list(messages) + [HumanMessage(content=extra)]
    ```
- [ ] **Step 3（COMMIT）**：`feat(prompts): render corpus snippets alongside dataset facts in model_node`。

## Task M4.5.3：plan-solve 入口召回（修改 `planner_node`）

**Files:**
- Modify: `src/data_agent_langchain/agents/planner_node.py`
- Create: `tests/test_phase10_rag_planner_recall.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - rag 关闭：planner_node 行为与 v2 完全一致（parity）—— 只产 `memory_hits` 中的 dataset facts。
  - rag 开启 + 命中：`output["memory_hits"]` 同时包含 dataset facts hits 与 corpus hits；事件 trace 含两次 `memory_recall`，`kind` 分别为 `dataset_knowledge` 与 `corpus_task`。
  - 召回顺序：先 `recall_dataset_facts`，后 `recall_corpus_snippets`；两个 hits list 拼接写回 `output["memory_hits"]`。
  - query 来自 `state["question"]`。
- [ ] **Step 2（GREEN）**：在 `planner_node`（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/planner_node.py:54-64`）的 `recall_dataset_facts` 调用之后追加 `recall_corpus_snippets` 调用并合并 hits。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): plan-solve planner also recalls corpus snippets`。

## Task M4.5.4：ReAct 入口召回（新增 `task_entry_node`，A2 决策）

**Files:**
- Create: `src/data_agent_langchain/agents/task_entry_node.py`
- Modify: `src/data_agent_langchain/agents/react_graph.py`
- Create: `tests/test_phase10_rag_react_entry.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - `task_entry_node(state, config)` 调用 `recall_dataset_facts` + `recall_corpus_snippets`，把合并 hits 写入 `output["memory_hits"]`。
  - rag 关闭 + memory mode disabled：`output == {}`（无 hits）。
  - rag 关闭 + memory mode read_only_dataset：`output["memory_hits"]` 只含 dataset hits。
  - rag 开启 + dataset 与 corpus 都命中：`output["memory_hits"]` 同时含两类。
  - 入图后 ReAct 模式在 `model_node` 第一次进入时 `state["memory_hits"]` 已非空（v2 baseline 验证：之前 ReAct 不召回 dataset facts，现在召回了）。
  - parity：rag 关闭、memory disabled 的 ReAct 路径与 v2 完全一致。
- [ ] **Step 2（GREEN）**：
  - 新建 `task_entry_node.py`（按 `01-design-v2.md §4.13` 伪代码）。
  - 修改 `react_graph.build_react_graph`：
    ```python
    g.add_node("task_entry", task_entry_node)
    g.add_edge(START, "task_entry")
    g.add_edge("task_entry", "execution")
    g.add_edge("execution", "finalize")
    g.add_edge("finalize", END)
    ```
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add task_entry_node for ReAct one-shot recall`。

## Task M4.5.5：`model_node` 不召回的护栏测试

**Files:**
- Create: `tests/test_phase10_rag_model_node_no_recall.py`

**Steps:**

- [ ] **Step 1（RED）+ GREEN）**：用 spy 验证：
  - rag 开启时跑一遍 ReAct 60 步，`recall_corpus_snippets` 在整个 task 生命周期内仅被调用 **1 次**（在 task_entry_node），**不**在 model_node 内调用。
  - `state["memory_hits"]` 在 60 步过程中长度不增长（reducer 不再累加）。
- [ ] **Step 2（COMMIT）**：`test(memory-rag): guard model_node does not call recall (A2)`。

**Verification (M4.5 全部完成后):**
```powershell
pytest tests/test_phase10_rag_*.py tests/test_phase4_planner_node.py tests/test_phase3_model.py -q
```
全绿。本地手动 A/B（**Bug 7 修正**：`configs/local.yaml` 默认 `memory.mode=disabled` 时 RAG 守卫会强制关闭，必须同时显式 `--memory-mode read_only_dataset`）：
```powershell
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --no-memory-rag
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode react --memory-mode read_only_dataset --no-memory-rag
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode react --memory-mode read_only_dataset --memory-rag
```
比较 metrics.json 中 `memory_rag` 字段、tool_calls 计数与 token 用量。

---

# Milestone M4.6：观测聚合 + 启动期 import 边界回归

## Task M4.6.1：`MetricsCollector` 聚合 `memory_rag_*` 事件

**Files:**
- Modify: `src/data_agent_langchain/observability/metrics.py`
- Create: `tests/test_phase10_rag_metrics.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - `metrics.json` 在 `rag.enabled=true` 时含 `memory_rag` 段，结构为：
    ```json
    {
      "task_index_built": true,
      "task_doc_count": 12,
      "task_chunk_count": 48,
      "shared_collections_loaded": 0,
      "recall_count": {"task_entry": 1},
      "skipped": []
    }
    ```
  - `rag.enabled=false` 时 `metrics.json` **不**含 `memory_rag` 段（baseline parity）。
  - `memory_rag_skipped` 事件被聚合到 `memory_rag.skipped` 列表（按 reason 去重计数）。
- [ ] **Step 2（GREEN）**：扩展 `MetricsCollector.on_custom_event` 订阅 `memory_rag_index_built` / `memory_rag_skipped`；在 `_build_payload` 中条件性加入 `memory_rag` 段。
- [ ] **Step 3（COMMIT）**：`feat(observability): aggregate memory_rag events into metrics.json`。

## Task M4.6.2：启动期 import 边界回归测试（D11）

**Files:**
- Create: `tests/test_phase10_rag_import_boundary.py`

**Steps:**

- [ ] **Step 1（RED）+ GREEN）**：写 3 个子进程断言测试（每个独立 `subprocess.run`）：
  - `python -c "import data_agent_langchain"` 后 `sys.modules` 不含 `torch` / `chromadb` / `sentence_transformers`。
  - `python -c "import data_agent_langchain.memory.rag"` 后 `sys.modules` 不含上述任一。
  - `python -c "from data_agent_langchain.memory.rag.embedders.sentence_transformer import HarrierEmbedder"` 后 `sys.modules` 不含 `torch`（class 定义不触发 import）。
- [ ] **Step 2（COMMIT）**：`test(memory-rag): guard rag submodules do not import heavy deps at module load`。

**Verification (M4.6 全部完成后):**
```powershell
pytest tests/test_phase10_rag_metrics.py tests/test_phase10_rag_import_boundary.py -q
```
全绿。

---

## Cross-cutting Verification（M4 全部完成后）

最终验证命令：

```powershell
# Phase 10 全套
$files = Get-ChildItem -LiteralPath tests -Filter 'test_phase10_rag_*.py' | ForEach-Object { $_.FullName }
pytest @files -q

# 回归（v2 已有相关）
pytest tests/test_phase1_runstate.py tests/test_phase3_tool_node.py tests/test_phase3_model.py `
       tests/test_phase4_planner_node.py tests/test_phase5_config.py `
       tests/test_phase9_memory_*.py tests/test_submission_config.py -q

# Slow（手动 opt-in）
pytest tests/test_phase10_rag_embedder_harrier.py -q --runslow

# 本地 e2e A/B（4 个组合）。Bug 7：必须带 `--memory-mode read_only_dataset`，
# 否则 local.yaml 默认 mode=disabled 让 RAG 守卫提前 return。
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --no-memory-rag
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode react --memory-mode read_only_dataset --no-memory-rag
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode react --memory-mode read_only_dataset --memory-rag
```

验收标准：

- M4.0–M4.6 全部里程碑 commits 合入 `feature/rag-memory-m4-corpus` 分支。
- 所有 phase 10 测试 + 既有相关回归测试全绿。
- 本地 A/B：`--memory-rag` 路径在 task_344 上 token 用量、tool_call 次数与 metrics 可被 `memory_rag` 字段解释；`--no-memory-rag` 路径与 v2 baseline **结构上完全一致**（structural parity，不要求 token 数完全相同 —— LLM 随机性）。
- 评测态默认 `rag.enabled=false`；env `DATA_AGENT_RAG=1` 显式打开。
- **关键护栏全绿**：
  - `test_phase10_rag_model_node_no_recall.py`（A2 守护）
  - `test_phase10_rag_corpus_store_protocol.py`（B2 守护，禁止 chroma store 实现 v2 MemoryStore.put）
  - `test_phase10_rag_packaging.py`（C2 守护，禁止重依赖进 base dependencies）
  - `test_phase10_rag_import_boundary.py`（D11 守护，禁止 rag 子模块顶层 import 重依赖）

---

## 风险缓解（实施过程中跟踪）

- **R1：CPU 上 270m 编码慢**：M4.4 完成后测速；如果单 task 索引 > 10s，启用 `embedder_dtype="float16"` 或在 evaluator 镜像降为 `bge-small-zh-v1.5`（仅改 `embedder_model_id`，代码不变）。
- **R2：chromadb Windows SQLite 锁**：M4.3 完成后在 Windows 本地跑一遍 `pytest test_phase10_rag_chroma_store.py -q`；ephemeral 模式不写盘理论上无锁问题；如偶发，尝试关闭 `pytest-xdist`。
- **R3：镜像体积超限**：M4.4 完成后用 `docker build --build-arg ENABLE_RAG=1` 实测镜像大小；如果超限，量化或降模型。
- **R4：prompt 预算挤压**：A2 决策已大幅缓解（仅入口召回 1 次，prompt 增量恒定 ≤ 2200 字符）；M4.5 完成后在 task_344 上观察 `tokens.prompt` 是否飙升。
- **R5：rag 子模块顶层 import 重依赖漂移**：M4.6.2 回归测试守护；CI 必须把 `test_phase10_rag_import_boundary.py` 纳入 phase 10 必跑集合。
- **R6：A2 召回精度不足**：仅入口召回 1 次可能错过 ReAct 后期 thought 演进出的查询。M4 完成后用 A/B 实测；若 task_344 上 corpus 召回完全没贡献，再评估是否升级到 A3（动态条件触发，单独提案）。

---

## 与 v1（v3）实施计划的差异回顾

| 维度 | v1 计划 | v2 计划（本版） |
|---|---|---|
| Milestone 数 | M4.0–M4.6（含可选 M4.6 shared corpus） | M4.0–M4.6（**M4.6 改为观测聚合 + import 边界**，shared corpus 拆出独立提案） |
| `_dataclass_from_dict` 测试 | 仅 round-trip | 增加 `tuple[T,...]` / `Optional[Path]` 覆盖 |
| pytest `slow` marker | 标但未明示如何启用 | **M4.0.2** 显式配置 `pyproject.toml` + `conftest.py` |
| ChromaCorpusStore Protocol | 同时实现 v2 MemoryStore + corpus | **独立 CorpusStore Protocol**，护栏测试禁止 `hasattr(store, 'put')` |
| ChromaCorpusStore.persistent | 单独 task `M4.3.2 persistent_readonly` | **删除**（D2） |
| ingest_cli | `M4.6.1` 在主提案内 | **删除**（D2） |
| 召回时机 | planner_node + model_node 都召回 | 仅 task 入口召回 1 次：`planner_node`（plan-solve）+ 新增 `task_entry_node`（ReAct） |
| `model_node` 不召回护栏 | 无 | **M4.5.5** 新增 spy 验证 |
| pyproject extras | 重依赖进 `dependencies` | **M4.4.2** 进 `[project.optional-dependencies].rag`；护栏测试守护 |
| HarrierEmbedder 测试 | `--runslow` 标但无 marker 配置 | **M4.0.2** 已配置 + **M4.2.2** 增加顶层 import 边界单独测试（非 slow） |
| 启动期 import 边界 | 无 | **M4.6.2** 子进程 sys.modules 断言 |
| metrics 聚合 | 设计中有，plan 未列任务 | **M4.6.1** 单独任务 |
| ReAct 入口召回 | 无（plan 完全没注意 ReAct 模式没 planner） | **M4.5.4** 新增 `task_entry_node`（顺带修复 v2 隐藏 gap：ReAct 模式不召回 dataset facts） |

v2 计划与 v2（v2 = 当前主干）完全兼容，是 v1（v3）的事实修正 + 范围收缩 + 护栏强化。
