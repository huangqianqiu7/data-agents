# Corpus RAG 实施计划 v3（M4.0–M4.6）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `01-design.md` 在 `memory/rag/` 子包内落地基于 **Harrier-OSS-v1-270m + chromadb** 的 corpus RAG（M4），并把 per-task corpus 接入 `planner_node` / `model_node`。

**Architecture:** Embedder / ChromaCorpusStore / VectorCorpusRetriever 三层解耦；`MemoryConfig` 新增 `rag: CorpusRagConfig` 嵌套子配置；`RunState` 不变，复用现有 `memory_hits`；`agents/corpus_recall.py` 与 v2 `agents/memory_recall.py` 并列；`agents/prompts.py` 新增 `render_corpus_snippets`；observability 复用 `memory_recall` 事件并新增 `memory_rag_index_built` / `memory_rag_skipped`。

**Tech Stack:** Python 3.11+，dataclasses（`frozen=True, slots=True`），LangGraph 0.4，LangChain core，`sentence-transformers`，`transformers`，`torch`，`chromadb`，pytest，typer CLI。

**前置依赖:** v2 已合入主干，`memory/rag/__init__.py` 当前为占位（`NotImplementedError`）。

---

## File Structure

新增文件：

- `src/data_agent_langchain/memory/rag/documents.py`
- `src/data_agent_langchain/memory/rag/loader.py`
- `src/data_agent_langchain/memory/rag/chunker.py`
- `src/data_agent_langchain/memory/rag/redactor.py`
- `src/data_agent_langchain/memory/rag/embedders/__init__.py`
- `src/data_agent_langchain/memory/rag/embedders/base.py`
- `src/data_agent_langchain/memory/rag/embedders/sentence_transformer.py`
- `src/data_agent_langchain/memory/rag/embedders/stub.py`
- `src/data_agent_langchain/memory/rag/stores/__init__.py`
- `src/data_agent_langchain/memory/rag/stores/chroma.py`
- `src/data_agent_langchain/memory/rag/retrievers/__init__.py`
- `src/data_agent_langchain/memory/rag/retrievers/vector.py`
- `src/data_agent_langchain/memory/rag/factory.py`
- `src/data_agent_langchain/memory/rag/ingest_cli.py`（M4.6 可选）
- `src/data_agent_langchain/agents/corpus_recall.py`
- 所有测试统一放在 `tests/test_phase10_rag_*.py`。

修改文件：

- `src/data_agent_langchain/config.py` —— 让 `_dataclass_from_dict` 支持嵌套；新增 `CorpusRagConfig` 并嵌入 `MemoryConfig`。
- `src/data_agent_langchain/memory/rag/__init__.py` —— 去掉 `NotImplementedError`，导出新模块。
- `src/data_agent_langchain/runtime/context.py` —— 新增 corpus handles contextvar。
- `src/data_agent_langchain/run/runner.py` —— 子进程入口构建 task corpus + 写 contextvar。
- `src/data_agent_langchain/agents/planner_node.py` —— 调 `recall_corpus_snippets`。
- `src/data_agent_langchain/agents/model_node.py` —— 同上（仅 ReAct / execution）。
- `src/data_agent_langchain/agents/prompts.py` —— 新增 `render_corpus_snippets` 并接入三个 builder。
- `src/data_agent_langchain/cli.py` —— 新增 `--memory-rag/--no-memory-rag` 开关。
- `src/data_agent_langchain/submission.py` —— 评测态默认 `rag.enabled=false`，env `DATA_AGENT_RAG=1` 显式打开。
- `pyproject.toml` —— 新增 `sentence-transformers` / `chromadb` 依赖。

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

---

# Milestone M4.0：嵌套 dataclass 支持（config 前置）

## Task M4.0.1：让 `_dataclass_from_dict` 支持嵌套

**Files:**
- Modify: `src/data_agent_langchain/config.py`
- Create: `tests/test_phase10_rag_config_nested.py`

**Steps:**

- [ ] **Step 1（RED）**：写 `test_nested_dataclass_from_dict_roundtrip`，构造一个临时嵌套 dataclass，断言 `to_dict` / `from_dict` 互逆。
- [ ] **Step 2（GREEN）**：在 `_dataclass_from_dict` 中加分支：若 `field_info` 对应类型是 dataclass 且 `value` 是 dict，则递归。注意 `from __future__ import annotations` 下 `field_info.type` 是字符串，需用 `typing.get_type_hints(cls)` 解析。
- [ ] **Step 3（验证）**：跑 `pytest tests/test_phase5_config.py tests/test_phase10_rag_config_nested.py -q`，全绿。
- [ ] **Step 4（COMMIT）**：`refactor(config): support nested dataclass in from_dict`。

**Verification:**
- `tests/test_phase5_config.py` 全部回归通过。
- 新增嵌套测试通过。

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
  - `filter_text` 命中 `\banswer\b` 的整段被丢弃（返回空字符串或原文中删除）。
- [ ] **Step 2（GREEN）**：实现 `Redactor(cfg)`，基于 `cfg.redact_patterns` + `cfg.redact_filenames`（glob）。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add Redactor for filename and content filtering`。

## Task M4.1.3：`Loader`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/loader.py`
- Create: `tests/test_phase10_rag_loader.py`
- Test fixtures: 在 `tests/fixtures/rag/task_demo/` 下放 `README.md`、`data_schema.md`、`expected_output.json`（应被跳过）、`ground_truth.csv`（应被跳过）。

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - 扫描 fixture 目录返回 `[CorpusDocument(...)]`。
  - `expected_output.json` 与 `ground_truth.csv` 不出现。
  - 文档数超过 `max_docs_per_task` 时截断并 dispatch `memory_rag_skipped(reason="max_docs_truncated")`。
  - 不解析二进制文件（`.docx` / `.pdf` 当作 `doc_kind="doc"` 时只读元数据）。
- [ ] **Step 2（GREEN）**：实现 `Loader.scan(path) -> list[CorpusDocument]`，对每个文件计算 `doc_id = sha1(source_path + size + mtime)[:16]`。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add Loader with redact-aware filename filter`。

## Task M4.1.4：`Chunker`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/chunker.py`
- Create: `tests/test_phase10_rag_chunker.py`

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - 空文本返回 `[]`。
  - 单字符长度 < `chunk_size_chars` 返回单 chunk。
  - 长文本切片数 = `ceil((char_count - overlap) / (size - overlap))`。
  - 超过 `max_chunks_per_doc` 截断并 dispatch `memory_rag_skipped(reason="max_chunks_truncated")`。
  - `chunk_id == f"{doc_id}#{ord:04d}"` 稳定可复现。
- [ ] **Step 2（GREEN）**：实现 `CharWindowChunker(cfg)`。可选在 ` ` / 中英文段落点处优先切。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add char-window chunker with overlap and caps`。

**Verification (M4.1 全部完成后):**
- `pytest tests/test_phase10_rag_{documents,redactor,loader,chunker}.py -q` 全绿。

---

# Milestone M4.2：Embedder（Protocol + Harrier + Stub）

## Task M4.2.1：`Embedder` Protocol + `DeterministicStubEmbedder`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/embedders/__init__.py`
- Create: `src/data_agent_langchain/memory/rag/embedders/base.py`
- Create: `src/data_agent_langchain/memory/rag/embedders/stub.py`
- Create: `tests/test_phase10_rag_embedder_stub.py`

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - `DeterministicStubEmbedder(dim=8).dimension == 8`。
  - 同一文本两次 `embed_documents` 返回相同向量。
  - 不同文本返回不同向量。
  - `embed_query(text)` 与 `embed_documents([text])[0]` 不一定相同（query 端可注入 prompt 前缀）。
  - 返回值是 `list[list[float]]`，不是 numpy。
- [ ] **Step 2（GREEN）**：实现 `DeterministicStubEmbedder`：基于 `hashlib.sha256(text.encode()).digest()` 截取 `dim` 字节，归一化为单位向量；`embed_query` 在 text 前加 `"q:"` 前缀。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add Embedder protocol and deterministic stub`。

## Task M4.2.2：`HarrierEmbedder`（生产 backend）

**Files:**
- Create: `src/data_agent_langchain/memory/rag/embedders/sentence_transformer.py`
- Create: `tests/test_phase10_rag_embedder_harrier.py`（标注 `@pytest.mark.slow`，opt-in 跑）

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - `HarrierEmbedder(cfg).dimension > 0`。
  - `embed_documents([])` 返回 `[]`。
  - `embed_documents(["hello world"])` 返回长度 1 的 list；向量 L2 归一化（`abs(norm - 1.0) < 1e-3`）。
  - `embed_query("question")` 走 `prompt_name="web_search_query"`，向量也归一化。
  - `embed_documents` 在传入 100 段时不爆 batch（`embedder_batch_size=8` 下应分 13 batch）。
  - CUDA OOM 时 fall back 到 cpu 并 dispatch `memory_rag_skipped(reason="cuda_oom")`（用 monkeypatch 模拟）。
- [ ] **Step 2（GREEN）**：实现 `HarrierEmbedder`：方法内延迟 import `sentence_transformers`，按 `cfg.embedder_device` / `cfg.embedder_dtype` 加载，`max_seq_length` 钳制到 `cfg.embedder_max_seq_len`。
- [ ] **Step 3（验证）**：在本地 GPU 上手动跑一次 `pytest tests/test_phase10_rag_embedder_harrier.py -q --runslow`，确认权重能从 `HF_HOME` 缓存离线加载。
- [ ] **Step 4（COMMIT）**：`feat(memory-rag): add HarrierEmbedder sentence-transformers backend`。

**Verification (M4.2 全部完成后):**
- `pytest tests/test_phase10_rag_embedder_stub.py -q` 全绿。
- `pytest tests/test_phase10_rag_embedder_harrier.py -q --runslow` 在本地手动验证一次。

---

# Milestone M4.3：ChromaCorpusStore + VectorCorpusRetriever

## Task M4.3.1：`ChromaCorpusStore.ephemeral`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/stores/__init__.py`
- Create: `src/data_agent_langchain/memory/rag/stores/chroma.py`
- Create: `tests/test_phase10_rag_chroma_store.py`

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵（全程用 `DeterministicStubEmbedder`）
  - `ChromaCorpusStore.ephemeral(...)` 不写盘、不触发 telemetry。
  - `upsert_chunks([CorpusChunk(...)])` 之后 `query_by_vector(vec, k=1)` 命中该 chunk。
  - 多 namespace 隔离：往 `corpus_task:A` 写，从 `corpus_task:B` 查得 `[]`。
  - `put / get / list / delete` 走 v2 MemoryStore Protocol 也能用（少量元数据查询）。
  - `chromadb.Settings(anonymized_telemetry=False)` 已开启。
- [ ] **Step 2（GREEN）**：实现 `ChromaCorpusStore.ephemeral`。`upsert_chunks` 走 `collection.add(ids=, embeddings=, documents=, metadatas=)`；`query_by_vector` 走 `collection.query(query_embeddings=[vec], n_results=k)`。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add ChromaCorpusStore ephemeral backend`。

## Task M4.3.2：`ChromaCorpusStore.persistent_readonly`

**Files:**
- Modify: `src/data_agent_langchain/memory/rag/stores/chroma.py`
- Modify: `tests/test_phase10_rag_chroma_store.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - `persistent_readonly(path)` 加载已存在的 chroma sqlite，query 正常。
  - `put` / `upsert_chunks` / `delete` 在 readonly 实例上 raise `MemoryReadOnlyError`。
- [ ] **Step 2（GREEN）**：实现 `readonly=True` 分支；写路径前置 `_check_writable()` 守卫。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add ChromaCorpusStore persistent read-only mode`。

## Task M4.3.3：`VectorCorpusRetriever`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/retrievers/__init__.py`
- Create: `src/data_agent_langchain/memory/rag/retrievers/vector.py`
- Create: `tests/test_phase10_rag_vector_retriever.py`

**Steps:**

- [ ] **Step 1（RED）**：测试矩阵
  - 命中场景：upsert 一组 chunks 后 retrieve 返回 `RetrievalResult(reason="vector_cosine")`。
  - `k=0` 返回 `[]`，**不调用** embedder。
  - 负数 `k` clamp 到 0。
  - embedder 异常（mock `embed_query` 抛 `RuntimeError`）→ 返回 `[]` 并 dispatch `memory_rag_skipped(reason="query_embed_failed")`。
  - store 异常 → 同上 reason="vector_query_failed"。
- [ ] **Step 2（GREEN）**：实现 `VectorCorpusRetriever(store, embedder, k)`，沿用 v2 `Retriever` Protocol。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add VectorCorpusRetriever with cosine similarity`。

**Verification (M4.3 全部完成后):**
- `pytest tests/test_phase10_rag_chroma_store.py tests/test_phase10_rag_vector_retriever.py -q` 全绿。

---

# Milestone M4.4：Config 扩展 + Factory + Runner 集成

## Task M4.4.1：`CorpusRagConfig` + `MemoryConfig.rag`

**Files:**
- Modify: `src/data_agent_langchain/config.py`
- Create: `tests/test_phase10_rag_config.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - 默认 `CorpusRagConfig()` 字段值符合 `01-design.md §4.3`。
  - `AppConfig.to_dict / from_dict` 包含 `memory.rag` 字段且互逆。
  - YAML 解析：从 `configs/local.yaml` 读 `memory.rag.enabled=true` 走通。
  - `mode=disabled` 时 `rag.enabled` 即使为 `true`，构造行为也按关闭处理（由 factory 层判断；这里只测 dataclass 字段保留）。
- [ ] **Step 2（GREEN）**：在 `config.py` 新增 `CorpusRagConfig` 与 `MemoryConfig.rag`。
- [ ] **Step 3（COMMIT）**：`feat(config): add MemoryConfig.rag CorpusRagConfig nested config`。

## Task M4.4.2：`factory.build_embedder` / `build_task_corpus`

**Files:**
- Create: `src/data_agent_langchain/memory/rag/factory.py`
- Create: `tests/test_phase10_rag_factory.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - `build_embedder(cfg)`：`embedder_backend="stub"` 返回 stub；`"sentence_transformer"` 返回 `HarrierEmbedder`（用 monkeypatch 把 `SentenceTransformer` 替换为假对象，避免加载真权重）。
  - `build_task_corpus(cfg, task_id, task_input_dir, embedder)`：rag 关闭返回 `None`；rag 打开返回 `(ChromaCorpusStore, VectorCorpusRetriever)`，索引内含 fixture 中的 chunk 数。
  - `build_task_corpus` 失败（如目录不存在）→ dispatch `memory_rag_skipped` 并返回 `None`。
- [ ] **Step 2（GREEN）**：实现工厂；遵循「方法内延迟 import 重依赖」原则。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add factory for embedder and task corpus`。

## Task M4.4.3：Contextvar 与 runner 子进程入口

**Files:**
- Modify: `src/data_agent_langchain/runtime/context.py`
- Modify: `src/data_agent_langchain/run/runner.py`
- Create: `tests/test_phase10_rag_runner.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - 子进程入口：rag enabled 时，`get_current_corpus_handles()` 返回非空；rag disabled 时返回 `None`。
  - 索引构建事件：`memory_rag_index_built` 落 trace，含 `doc_count` / `chunk_count` / `model_id` / `dimension`。
  - 失败 fail closed：embedder 加载失败时 `memory_rag_skipped` 落 trace，task 仍按 baseline 运行。
  - parity：rag 关闭路径下 task_344 metrics 与 v2 baseline 一致（结构上）。
- [ ] **Step 2（GREEN）**：
  - `runtime/context.py` 增加 `set_corpus_handles / get_current_corpus_handles` 的 contextvar API。
  - `runner._run_task_in_subprocess` 入口在构建图前调用 `factory.build_task_corpus(...)`，把结果写 contextvar。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): build per-task corpus in runner subprocess entry`。

## Task M4.4.4：CLI / submission 接入

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
  - env `DATA_AGENT_RAG=1` 时打开为 `True`。
- [ ] **Step 2（GREEN）**：实现 CLI flag 与 env 读取。
- [ ] **Step 3（COMMIT）**：`feat(cli): add --memory-rag toggle for corpus RAG`。

**Verification (M4.4 全部完成后):**
- `pytest tests/test_phase10_rag_config.py tests/test_phase10_rag_factory.py tests/test_phase10_rag_runner.py tests/test_submission_config.py -q` 全绿。
- 本地手动跑：`dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --no-memory-rag`，结果与 v2 baseline 完全一致。

---

# Milestone M4.5：召回 + Prompt 注入

## Task M4.5.1：`recall_corpus_snippets`

**Files:**
- Create: `src/data_agent_langchain/agents/corpus_recall.py`
- Create: `tests/test_phase10_rag_corpus_recall.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - rag 关闭：返回 `[]`，不调 retriever。
  - rag 开启 + 无 task corpus：返回 `[]`。
  - 命中场景：返回 `list[MemoryHit]`，`summary` 形如 `"[markdown] task_344/README.md: ..."`。
  - dispatch `memory_recall(kind="corpus_task", query_digest=..., hit_chunk_ids=..., model_id=..., reason="vector_cosine")`。
  - retriever 异常 → 返回 `[]` 并 dispatch `memory_rag_skipped(reason="retrieve_failed")`。
  - `prompt_budget_chars` 严格遵守：累积 summary 长度不超出预算。
  - 召回时 query 不被写入任何 store（用 spy 验证）。
- [ ] **Step 2（GREEN）**：实现 `recall_corpus_snippets(cfg, *, task_id, query, node, config)`，沿用 v2 `recall_dataset_facts` 风格。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add recall_corpus_snippets helper`。

## Task M4.5.2：`render_corpus_snippets` + prompts 接入

**Files:**
- Modify: `src/data_agent_langchain/agents/prompts.py`
- Modify: `tests/test_phase3_model.py` 或新增 `tests/test_phase10_rag_prompts.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - `render_corpus_snippets([], budget=1800) == ""`。
  - 非空 hits 渲染输出包含 `## Reference snippets (from task documentation)` 段头。
  - 渲染只用 `hit.summary`；不出现 `record_id` / `namespace` / `score`。
  - `build_planning_messages` / `build_plan_solve_execution_messages` / `build_react_messages` 输出消息列表的最后一条 user message 末尾包含上述段；dataset facts 段在前，corpus snippets 段在后。
  - 总长度截断到 budget。
- [ ] **Step 2（GREEN）**：实现 `render_corpus_snippets`；三个 builder 在已有 `render_dataset_facts` 调用后追加。
- [ ] **Step 3（COMMIT）**：`feat(prompts): render corpus snippets after dataset facts`。

## Task M4.5.3：`planner_node` / `model_node` 召回

**Files:**
- Modify: `src/data_agent_langchain/agents/planner_node.py`
- Modify: `src/data_agent_langchain/agents/model_node.py`
- Modify: `tests/test_phase4_planner_node.py` 或新增 `tests/test_phase10_rag_nodes.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - rag 关闭：节点行为与 v2 完全一致（parity 测试）。
  - rag 开启 + 命中：node 输出 partial state 中 `memory_hits` 同时包含 dataset facts hits 与 corpus hits；事件 trace 含两次 `memory_recall`，`kind` 分别为 `dataset_knowledge` 与 `corpus_task`。
  - `model_node` 用 `last_thought` 作为 query；空 thought fall back 到 `task.question`。
  - 召回顺序：先 `recall_dataset_facts`，后 `recall_corpus_snippets`；两个 hits list 用 `operator.add` 合并写回 state。
- [ ] **Step 2（GREEN）**：在两个节点的现有 `recall_dataset_facts` 调用之后立即追加 `recall_corpus_snippets` 调用。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): inject corpus snippets in planner/model nodes`。

**Verification (M4.5 全部完成后):**
- `pytest tests/test_phase10_rag_*.py tests/test_phase4_planner_node.py tests/test_phase3_model.py -q` 全绿。
- 本地手动 A/B：
  ```powershell
  dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --no-memory-rag
  dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --memory-rag
  ```
  比较 metrics.json 中 `memory_rag` 字段、tool_calls 计数与 token 用量。

---

# Milestone M4.6（可选）：Shared Corpus + Ingest CLI

> 在 M4.0–M4.5 全部完成且生产可用之后再做。如果团队评估 shared corpus 无内容来源，可以跳过此里程碑。

## Task M4.6.1：`ChromaCorpusStore.persistent_writable` + `ingest_cli`

**Files:**
- Modify: `src/data_agent_langchain/memory/rag/stores/chroma.py`（增加 writable persistent 模式，仅离线使用）
- Create: `src/data_agent_langchain/memory/rag/ingest_cli.py`
- Create: `tests/test_phase10_rag_ingest_cli.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - 命令 `python -m data_agent_langchain.memory.rag.ingest_cli build --source assets/corpus_shared/ --output assets/corpus_shared/index/` 成功构建 chroma 索引 + `manifest.json`。
  - `manifest.json` 含 `model_id` / `dimension` / `doc_count` / `chunk_count` / `built_at`。
  - 重复构建幂等（同 `doc_id` upsert）。
  - 评测态 `persistent_readonly` 加载该索引可正常 query。
- [ ] **Step 2（GREEN）**：实现 `ingest_cli.py`（typer），复用现有 loader / chunker / redactor / embedder。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): add offline ingest CLI for shared corpus`。

## Task M4.6.2：Shared corpus 加载与召回接入

**Files:**
- Modify: `src/data_agent_langchain/memory/rag/factory.py`
- Modify: `src/data_agent_langchain/agents/corpus_recall.py`
- Modify: `src/data_agent_langchain/runtime/context.py`
- Modify: `tests/test_phase10_rag_corpus_recall.py`

**Steps:**

- [ ] **Step 1（RED）**：
  - `load_shared_corpus(cfg, embedder)` 返回 `dict[collection_name, VectorCorpusRetriever]`。
  - `recall_corpus_snippets` 同时召回 task 与 shared，事件 `kind` 区分 `corpus_task` / `corpus_shared`。
  - 评测态 store 是 readonly：尝试 `upsert` 抛 `MemoryReadOnlyError`。
- [ ] **Step 2（GREEN）**：实现 `load_shared_corpus` + recall 合并逻辑。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): load shared corpus in subprocess entry and recall together`。

**Verification (M4.6 全部完成后):**
- 离线构建 `assets/corpus_shared/index/` 索引。
- `pytest tests/test_phase10_rag_*.py -q` 全绿。
- 手动验证：本地把比赛 README / 工具用法摘要喂进 ingest_cli，确认 shared collection 在 task_344 上能召回相关片段。

---

## Cross-cutting Verification（M4 全部完成后）

最终验证命令：

```powershell
# 单元 + 集成
$files = Get-ChildItem -LiteralPath tests -Filter 'test_phase10_rag_*.py' | ForEach-Object { $_.FullName }
pytest @files -q

# 回归（v2 已有相关）
pytest tests/test_phase1_runstate.py tests/test_phase3_tool_node.py tests/test_phase3_model.py `
       tests/test_phase4_planner_node.py tests/test_phase5_config.py `
       tests/test_phase9_memory_*.py tests/test_submission_config.py -q

# 本地 e2e A/B
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --no-memory-rag
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --memory-rag
```

验收标准：

- M4.0–M4.5 全部里程碑 commits 合入 `feature/rag-memory-m4-corpus` 分支。
- 所有 phase10 测试 + 既有相关回归测试全绿。
- 本地 A/B：`--memory-rag` 路径在 task_344 上 token 用量、tool_call 次数与 metrics 可被 `memory_rag` 字段解释，且 `--no-memory-rag` 路径与 v2 baseline 完全一致。
- 评测态默认 `rag.enabled=false`；env `DATA_AGENT_RAG=1` 显式打开。

---

## 风险缓解（实施过程中跟踪）

- **R1：CPU 上 270m 编码慢**：M4.4 完成后测速；如果单 task 索引 > 10s，启用 `embedder_dtype="float16"` 或在 evaluator 镜像降为 `bge-small-zh-v1.5`（仅改 `embedder_model_id`，代码不变）。
- **R2：chromadb Windows SQLite 锁**：M4.3 完成后在 Windows 本地跑一遍 `pytest test_phase10_rag_chroma_store.py -q`；如果偶发，尝试关闭 `pytest-xdist`。
- **R3：镜像体积超限**：M4.4 完成后用 `docker build` 实测镜像大小；如果超限，量化或降模型。
- **R4：prompt 预算挤压**：M4.5 完成后在 task_344 上观察 `tokens.prompt` 是否飙升；超出 `agent.max_context_tokens` 时降低 `retrieval_k` 或 `prompt_budget_chars`。
- **R5：上线规模化失败**：评测态默认 `rag.enabled=false`，先在本地与小批量 task 上 A/B；通过后再切提交。
