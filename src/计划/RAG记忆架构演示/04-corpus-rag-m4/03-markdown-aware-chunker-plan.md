# MarkdownAwareChunker 实施计划（M4.7）

> 版本：v1.1，2026-05-15
> 设计动机：当前 `CharWindowChunker` 仅按字符窗口 + `\n\n` 段落对齐切片，对 markdown 类语义文档（如 `knowledge.md`）无法利用 `#`/`##`/`###` 标题层级，召回上下文不完整。改造为"**先按文档结构（markdown header）→ 后按文本结构（recursive character）**"的混合策略。
> **For agentic workers:** REQUIRED SUB-SKILL: 用 `superpowers:executing-plans` 逐 step 推进，每个 step 走完整 RED/GREEN/COMMIT 循环。

---

## Goal

把 `memory/rag/chunker.py` 的 `CharWindowChunker` **直接替换**为 `MarkdownAwareChunker`：

- **第一层（文档结构）**：用 LangChain `MarkdownHeaderTextSplitter` 按 `#` / `##` / `###` 切出语义完整小节。
- **第二层（文本结构）**：每个小节如超过 `chunk_size_chars`，用 `RecursiveCharacterTextSplitter` 递归切（段落 → 句子 → 单词 → 字符）。
- **header 前缀注入**：每个 chunk 的 `text` 前注入 `"> H1 > H2 > H3\n\n"` 形式的 breadcrumb（Q2 选项 A），提升 embedding 召回精度。
- **非 markdown 文档**（`doc_kind in {"text", "doc"}`）：直接走 `RecursiveCharacterTextSplitter`，不做 header 切分。
- **保持现有接口**：`chunk(doc, text) -> list[CorpusChunk]`，`chunk_id` 格式不变（`f"{doc_id}#{ord:04d}"`），`max_chunks_per_doc` 上限与 `memory_rag_skipped(reason="max_chunks_truncated")` 事件继续生效。

---

## 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| **Q1** | 直接替换 `CharWindowChunker` | 代码干净；新策略对所有 doc_kind 都是严格优于的；旧实现在 git 历史可查 |
| **Q2** | header 前缀注入 chunk text（不进 metadata） | Harrier-270M 对上下文敏感；实现简单；`CorpusChunk` 结构不动 |
| **Q3** | `langchain-text-splitters` 加到 `[project.optional-dependencies].rag` | 遵守 D11：RAG 关闭时零依赖增量 |
| **Q4** | 测试优先检查 `rag` extra 声明，再做 import smoke | 可选依赖不应要求默认 `pip install .` 环境必须安装；`pytest.importorskip` 仅用于真实导入 smoke |
| **Q5** | chunk size 预算包含 header prefix | `chunk_size_chars` 语义不漂移；二次 recursive 时用 `chunk_size_chars - len(prefix)` 作为 body 预算 |
| **Q6** | `char_offset` 用单调 best-effort 查找 | `MarkdownHeaderTextSplitter` 不暴露原始 offset；用 `find(..., start_at)` 防重复段落导致 offset 回退 |

---

## Tech Stack 变更

**新增依赖**（仅 RAG extra）：

```toml
rag = [
    "sentence-transformers>=3,<4",
    "chromadb>=1.0,<2",
    "langchain-text-splitters>=0.3,<0.4",   # 新增
]
```

`langchain-text-splitters` 是 langchain 生态的纯 Python 轻量包（<500KB，无 torch/numpy 强依赖），与现有 `langchain-core` 同根，版本兼容性高。

**重要边界**：

- `langchain_text_splitters` 只能在 `MarkdownAwareChunker.__init__` 内延迟 import，不能出现在 `chunker.py` 顶层 import。
- `tests/test_packaging_deps.py` 扫描的是运行时顶层 import；本变更不应把 `langchain-text-splitters` 加到 `[project].dependencies`。
- 依赖声明测试应解析 `[project.optional-dependencies].rag`，不要依赖默认环境已安装 RAG extra。

**保持不变**：

- `CorpusRagConfig.chunk_size_chars` / `chunk_overlap_chars` / `max_chunks_per_doc` 三个旋钮语义不变。
- `CorpusChunk` dataclass 字段不变。
- `factory.build_task_corpus` 的调用点不变（仍 `chunker.chunk(doc, text)`）。

---

## 完成判据（DoD）

1. `pip install -e ".[rag]"` 后，`pytest tests/test_phase11_chunker_deps.py tests/test_phase10_rag_chunker.py -q` 全绿。
2. `pytest tests/test_phase10_rag_*.py -q` 全绿（不破坏 loader / factory / metrics / runner）。
3. `pytest tests/test_packaging_deps.py tests/test_phase10_rag_import_boundary.py tests/test_dockerfile_rag_invariants.py -q` 全绿，证明默认运行时依赖与 D11 import 边界没有漂移。
4. E2E A/B 验证：`dabench-lc run-task task_11 --config configs/local.yaml --memory-mode read_only_dataset --memory-rag` 成功；`metrics.json.memory_rag` 显示 `task_index_built=true`，`task_chunk_count >= 6`。
5. 召回 cosine 分数 ≥ 当前基线（0.75-0.77）；注意 `metrics.json` 只聚合 `recall_count`，分数需要从 `trace.json` / observability event 中的 `memory_recall.scores` 查看。

---

# Task M4.7.1：新增 `langchain-text-splitters` 依赖

**Files:**
- Modify: `pyproject.toml`（追加到 `[project.optional-dependencies].rag`）
- Create: `tests/test_phase11_chunker_deps.py`

**Steps:**

- [ ] **Step 1（RED）**：写测试 `tests/test_phase11_chunker_deps.py`
  - `test_rag_extra_declares_langchain_text_splitters`：用 `tomllib` 解析 `pyproject.toml`，断言 `[project.optional-dependencies].rag` 包含 `langchain-text-splitters`。这条测试不依赖本机已经安装 extra，新增测试后应先失败。
  - `test_langchain_text_splitters_importable_when_rag_extra_installed`：使用 `pytest.importorskip("langchain_text_splitters")` 后断言 `MarkdownHeaderTextSplitter` / `RecursiveCharacterTextSplitter` 可导入。若本机尚未 `pip install -e ".[rag]"`，该 smoke test 允许 skip；安装 RAG extra 后必须通过。
- [ ] **Step 2（GREEN）**：在 `pyproject.toml` 的 `rag` extra 数组里追加 `"langchain-text-splitters>=0.3,<0.4"`。运行 `pip install -e ".[rag]"` 安装。
- [ ] **Step 3（VERIFY）**：`pytest tests/test_phase11_chunker_deps.py -q` 全绿。
- [ ] **Step 4（COMMIT）**：`feat(memory-rag): add langchain-text-splitters to rag extra`。

**Verification:**

```powershell
pip install -e ".[rag]"
pytest tests/test_phase11_chunker_deps.py -q
```

---

# Task M4.7.2：`MarkdownAwareChunker` 类实现

**Files:**
- Modify: `src/data_agent_langchain/memory/rag/chunker.py`（删除 `CharWindowChunker`，新增 `MarkdownAwareChunker`，文件名保持）
- Modify: `tests/test_phase10_rag_chunker.py`（重写，覆盖新策略）

**Steps:**

- [ ] **Step 1（RED）**：在 `tests/test_phase10_rag_chunker.py` 重写测试矩阵：
  - **基本接口**：
    - `test_empty_text_returns_empty_list`：空文本 → `[]`
    - `test_whitespace_only_returns_empty_list`：纯空白 → `[]`
    - `test_chunk_id_format_stable`：`chunk_id == f"{doc_id}#{ord:04d}"` 且重复调用稳定。
    - `test_chunk_offsets_monotonic`：`char_offset` 单调递增（同一文档内）。
    - `test_chunker_returns_list_of_corpus_chunk`：返回 `list[CorpusChunk]`，保持 `store.upsert_chunks` 消费契约。
  - **Markdown 路径（`doc_kind="markdown"`）**：
    - `test_markdown_splits_on_h1_h2_h3`：输入含 3 个 `##` 小节的 markdown → 至少 3 个 chunk，每个 chunk 对应一个 section。
    - `test_markdown_chunk_text_contains_header_prefix`：每个 chunk 的 `text` 以 breadcrumb 开头（如 `"> Knowledge Guide > Core Entities > Patient\n\n..."`）。
    - `test_markdown_large_section_falls_back_to_recursive`：单个 `##` section 超过 `chunk_size_chars=200` 时，自动用 `RecursiveCharacterTextSplitter` 进一步切；切出的子 chunk 共享相同 header prefix。
    - `test_markdown_prefix_counts_toward_chunk_size`：带 breadcrumb 的大 section 切片后，每个 `CorpusChunk.char_length <= chunk_size_chars`（普通 header 长度场景）。
    - `test_markdown_no_headers_falls_back_to_recursive`：markdown 文档没有任何 `#`/`##`/`###` 时，整体走 recursive 路径，且不加 `"> "` 前缀；注意 `MarkdownHeaderTextSplitter` 可能返回 `metadata={}` 的单 section，测试要覆盖这个分支。
    - `test_markdown_repeated_section_text_offsets_stay_monotonic`：多个 section 正文重复时，`char_offset` 不应都回到第一次出现的位置。
  - **非 markdown 路径（`doc_kind="text"` / `"doc"`）**：
    - `test_text_doc_kind_uses_recursive_only`：纯文本输入 → 直接 recursive 切，**不**加 header prefix。
  - **上限与事件**：
    - `test_truncates_when_exceeding_max_chunks`：超过 `max_chunks_per_doc=3` 时截断到 3 个。
    - `test_dispatches_skipped_event_when_truncated`：截断时 dispatch `memory_rag_skipped(reason="max_chunks_truncated", doc_id, limit)`。
    - `test_no_event_when_not_truncated`：未截断时不发事件。
  - **参数校验**：
    - `test_invalid_chunk_size_raises`：`chunk_size_chars <= 0` → `ValueError`。
    - `test_invalid_overlap_raises`：`chunk_overlap_chars < 0` 或 `>= chunk_size_chars` → `ValueError`。
  - 跑测试，确认全 RED。
- [ ] **Step 2（GREEN）**：实现 `MarkdownAwareChunker` 在 `src/data_agent_langchain/memory/rag/chunker.py`：

  ```python
  _RECURSIVE_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", ". ", " ", "")


  class MarkdownAwareChunker:
      """混合分块：markdown 按 header 切，超长 section 二次 recursive 切，
      其他 doc_kind 直接 recursive 切。每个 chunk 注入 header breadcrumb 前缀。
      """

      __slots__ = (
          "_size",
          "_overlap",
          "_max_chunks",
          "_md_splitter",
          "_recursive_splitter_cls",
      )

      def __init__(
          self,
          *,
          chunk_size_chars: int,
          chunk_overlap_chars: int,
          max_chunks_per_doc: int,
      ) -> None:
          # 参数校验（保留原 CharWindowChunker 的校验逻辑）
          if chunk_size_chars <= 0:
              raise ValueError("chunk_size_chars 必须 > 0")
          if not 0 <= chunk_overlap_chars < chunk_size_chars:
              raise ValueError("chunk_overlap_chars 必须 ∈ [0, chunk_size_chars)")
          if max_chunks_per_doc <= 0:
              raise ValueError("max_chunks_per_doc 必须 > 0")

          self._size = chunk_size_chars
          self._overlap = chunk_overlap_chars
          self._max_chunks = max_chunks_per_doc

          # 延迟 import：保持 D11 启动期边界约束
          from langchain_text_splitters import (
              MarkdownHeaderTextSplitter,
              RecursiveCharacterTextSplitter,
          )
          self._md_splitter = MarkdownHeaderTextSplitter(
              headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
              strip_headers=True,  # 我们自己注入 breadcrumb；strip 原始 header 避免重复
          )
          self._recursive_splitter_cls = RecursiveCharacterTextSplitter

      def chunk(self, doc: CorpusDocument, text: str) -> list[CorpusChunk]:
          if not text or not text.strip():
              return []

          if doc.doc_kind == "markdown":
              raw_chunks = self._chunk_markdown(text)
          else:
              raw_chunks = self._chunk_recursive_only(text)

          # 截断 + 事件
          truncated = len(raw_chunks) > self._max_chunks
          if truncated:
              raw_chunks = raw_chunks[: self._max_chunks]

          chunks: list[CorpusChunk] = []
          for ord_idx, (chunk_text, char_offset) in enumerate(raw_chunks):
              chunks.append(CorpusChunk(
                  chunk_id=f"{doc.doc_id}#{ord_idx:04d}",
                  doc_id=doc.doc_id,
                  ord=ord_idx,
                  text=chunk_text,
                  char_offset=char_offset,
                  char_length=len(chunk_text),
              ))

          if truncated:
              dispatch_observability_event(
                  "memory_rag_skipped",
                  {
                      "reason": "max_chunks_truncated",
                      "doc_id": doc.doc_id,
                      "limit": self._max_chunks,
                  },
              )
          return chunks

      def _split_recursive(self, text: str, *, chunk_size: int | None = None) -> list[str]:
          size = chunk_size or self._size
          overlap = min(self._overlap, max(0, size - 1))
          splitter = self._recursive_splitter_cls(
              chunk_size=size,
              chunk_overlap=overlap,
              separators=list(_RECURSIVE_SEPARATORS),
              length_function=len,
          )
          return splitter.split_text(text)

      def _chunk_markdown(self, text: str) -> list[tuple[str, int]]:
          """先按 header 切 section，超长 section 二次 recursive。

          Returns:
              list of (chunk_text_with_prefix, char_offset)
          """
          sections = self._md_splitter.split_text(text)
          if not sections or not any(sec.metadata for sec in sections):
              # 没有 header，或 splitter 返回了 metadata={} 的单 section → 整体 fallback
              return self._chunk_recursive_only(text)

          result: list[tuple[str, int]] = []
          section_search_pos = 0
          for sec in sections:
              prefix = self._build_breadcrumb(sec.metadata)
              body = sec.page_content
              if not body.strip():
                  continue

              section_offset = self._find_offset(text, body, section_search_pos)
              section_search_pos = max(section_search_pos, section_offset + 1)

              body_budget = max(1, self._size - len(prefix)) if prefix else self._size
              if prefix and len(prefix) >= self._size:
                  # 极长 header 属于异常输入；保留完整 breadcrumb，允许该 chunk 超 size。
                  body_budget = 1

              parts = (
                  [body]
                  if len(prefix) + len(body) <= self._size
                  else self._split_recursive(body, chunk_size=body_budget)
              )
              local_search_pos = section_offset
              local_overlap = min(self._overlap, max(0, body_budget - 1))
              for part in parts:
                  part_offset = self._find_offset(text, part, local_search_pos)
                  sub_text = f"{prefix}{part}" if prefix else part
                  result.append((sub_text, part_offset))
                  local_search_pos = max(
                      part_offset + 1,
                      part_offset + max(1, len(part) - local_overlap),
                  )
          return result

      def _chunk_recursive_only(self, text: str) -> list[tuple[str, int]]:
          """非 markdown 文档：纯 recursive，不加 breadcrumb。"""
          chunks = self._split_recursive(text)
          result: list[tuple[str, int]] = []
          search_pos = 0
          for c in chunks:
              offset = self._find_offset(text, c, search_pos)
              result.append((c, offset))
              search_pos = max(offset + 1, offset + max(1, len(c) - self._overlap))
          return result

      @staticmethod
      def _find_offset(text: str, needle: str, start_at: int) -> int:
          """从 start_at 开始单调查找 needle；找不到时返回 start_at 作为诊断兜底。"""
          bounded_start = max(0, min(start_at, len(text)))
          if not needle:
              return bounded_start
          idx = text.find(needle, bounded_start)
          if idx >= 0:
              return idx
          sample = needle[:80].strip()
          if sample:
              idx = text.find(sample, bounded_start)
              if idx >= 0:
                  return idx
          return bounded_start

      @staticmethod
      def _build_breadcrumb(metadata: dict[str, str]) -> str:
          """从 MarkdownHeaderTextSplitter 的 metadata 构造 ``> H1 > H2 > H3\\n\\n``。

          metadata 形如 {"h1": "Knowledge Guide", "h2": "Core Entities", "h3": "Patient"}。
          缺失的层级跳过。无任何 header 时返回空串。
          """
          parts = []
          for key in ("h1", "h2", "h3"):
              value = metadata.get(key)
              if value:
                  parts.append(value.strip())
          if not parts:
              return ""
          return "> " + " > ".join(parts) + "\n\n"
  ```

  - 关键点：
    - 顶层不 import `langchain_text_splitters`（D11 约束），在 `__init__` 内延迟 import。
    - `strip_headers=True` 配合自定义 `_build_breadcrumb` 避免双重 header 前缀。
    - 无 header markdown 要用 `not sections or not any(sec.metadata for sec in sections)` 判断；不能只判断 `not sections`。
    - `prefix` 必须计入 chunk size 预算；普通 header 场景下 `CorpusChunk.char_length <= chunk_size_chars`。
    - `char_offset` 用 `_find_offset(text, part, start_at)` 单调 best-effort 查找；不要用 `text.find(...)` 从头查，否则重复正文会导致 offset 回退。
    - 跑测试，确认全 GREEN。
- [ ] **Step 3（REFACTOR）**：
  - 检查是否有可拆出的辅助函数；
  - docstring 补全；
  - 确认 `__all__ = ["MarkdownAwareChunker"]`（删除旧 `CharWindowChunker` 导出）。
- [ ] **Step 4（COMMIT）**：`feat(memory-rag): replace CharWindowChunker with MarkdownAwareChunker`。

**Verification:**

```powershell
pytest tests/test_phase10_rag_chunker.py -q
```

全绿。

---

# Task M4.7.3：`factory.build_task_corpus` 接入新 chunker

**Files:**
- Modify: `src/data_agent_langchain/memory/rag/factory.py`（把 `CharWindowChunker` 改为 `MarkdownAwareChunker`）
- Modify: `tests/test_phase10_rag_factory.py`

**Steps:**

- [ ] **Step 1（RED）**：在现有 `tests/test_phase10_rag_factory.py` 追加 `test_build_task_corpus_uses_markdown_aware_chunker`：
  - 构造只含 `README.md` 的临时 `context_dir`。
  - `import data_agent_langchain.memory.rag.factory as factory`。
  - 用 `monkeypatch.setattr(factory, "MarkdownAwareChunker", SpyChunker)` 注入 spy；当前 factory 尚无 `MarkdownAwareChunker` 属性，这一步应 RED。
  - `SpyChunker.chunk()` 返回一个带 breadcrumb 的 `CorpusChunk(text="> Title\n\nbody", ...)`，同时记录构造参数，验证来自 `CorpusRagConfig`。
  - monkeypatch `data_agent_langchain.memory.rag.stores.chroma.ChromaCorpusStore.ephemeral` 返回 fake store；fake store 的 `upsert_chunks(chunks)` 把 chunks 存入 `captured_chunks`，从而不依赖真实 chromadb。
  - 调用 `build_task_corpus(...)` 后断言：
    - `SpyChunker` 被构造一次；
    - `captured_chunks[0].text.startswith("> Title\n\n")`；
    - 返回值不是 `None`。
  - 不要从 `TaskCorpusHandles.store` 反查 chunks：`CorpusStore` 协议没有 list/get API，真实 `ChromaCorpusStore` 也不提供按 ID 读取接口。
- [ ] **Step 2（GREEN）**：在 `factory.py` 的 `build_task_corpus`：
  - 把 `from data_agent_langchain.memory.rag.chunker import CharWindowChunker` 改为 `MarkdownAwareChunker`。
  - 把 `chunker = CharWindowChunker(...)` 改为 `chunker = MarkdownAwareChunker(...)`。
  - 同步更新 `factory.py` 顶部 docstring 中的流程说明，把 ``CharWindowChunker.chunk`` 改为 ``MarkdownAwareChunker.chunk``。
  - 跑测试，确认 GREEN。
- [ ] **Step 3（COMMIT）**：`feat(memory-rag): wire MarkdownAwareChunker into build_task_corpus`。

**Verification:**

```powershell
pytest tests/test_phase10_rag_factory.py -q
```

全绿。

---

# Task M4.7.4：D11 启动期 import 边界回归

**Files:**
- Modify（如需）：`tests/test_phase10_rag_import_boundary.py` 或 `tests/test_dockerfile_rag_invariants.py`

**Steps:**

- [ ] **Step 1（RED）**：在 `tests/test_phase10_rag_import_boundary.py` 的 `_HEAVY_MODULES` 或独立断言里加入 `"langchain_text_splitters"`，覆盖以下路径：
  - `import data_agent_langchain`
  - `import data_agent_langchain.memory.rag`
  - `from data_agent_langchain.memory.rag.chunker import MarkdownAwareChunker`
  这些 import 都不应把 `langchain_text_splitters` 放入 `sys.modules`。
- [ ] **Step 2（GREEN）**：在 `chunker.py` 顶层 import 区域确保**没有** `import langchain_text_splitters`；只在 `MarkdownAwareChunker.__init__` 内 import。如有顶层 import 残留，移除。
- [ ] **Step 2.5（VERIFY）**：跑 `tests/test_packaging_deps.py`，确认 `langchain-text-splitters` 没有被误加到 `[project].dependencies`，也没有因顶层 import 被 packaging 扫描拦下。
- [ ] **Step 3（COMMIT，如有改动）**：`fix(memory-rag): keep langchain_text_splitters import lazy`。

**Verification:**

```powershell
pytest tests/test_phase10_rag_import_boundary.py tests/test_packaging_deps.py tests/test_dockerfile_rag_invariants.py -q
```

全绿。

---

# Task M4.7.5：E2E A/B 验证

**Files:**
- 无代码改动；仅运行 + 观察 metrics。

**Steps:**

- [ ] **Step 1**：跑 task_11 RAG 路径，对比改造前后：

  ```powershell
  dabench-lc run-task task_11 --config configs/local.yaml --memory-mode read_only_dataset --memory-rag
  ```

- [ ] **Step 2**：读 `artifacts/runs/<新时间戳>/task_11/metrics.json`，验证：
  - `memory_rag.task_index_built == true`
  - `memory_rag.task_chunk_count >= 6`（按 section 切粒度更细）
  - `memory_rag.recall_count.task_entry >= 1` 或存在其他 corpus recall 节点计数。
  - `wall_clock_s` 与上次（60s）量级相同（不应明显劣化；可能略增因为 chunk 数变多）。
- [ ] **Step 3**：读 `artifacts/runs/<新时间戳>/task_11/trace.json`，确认：
  - 存在 `memory_recall` / corpus 召回事件；
  - 事件 payload 的 `scores` 最高分 ≥ 0.75（基线 cosine；该字段不在 `metrics.json` 聚合结果里）；
  - agent 仍能正确解答 thrombosis 问题（`succeeded=true`）。
- [ ] **Step 4**：把对比结果记到 `artifacts/notes-m4.7-ab.md`（可选）或直接在 PR 描述里。

**Verification（人工）:**

新 metrics.json 与 trace.json 各字段达标。

---

# 全部完成后的最终验证

```powershell
# 单元测试（全 RAG 模块）
pytest tests/test_phase10_rag_*.py tests/test_phase11_*.py -q

# 默认依赖与启动期 import 边界
pytest tests/test_packaging_deps.py tests/test_phase10_rag_import_boundary.py tests/test_dockerfile_rag_invariants.py -q

# E2E 单 task
dabench-lc run-task task_11 --config configs/local.yaml --memory-mode read_only_dataset --memory-rag
```

全绿 + E2E 通过 → 收口；走 `superpowers:finishing-a-development-branch` 决定合并方式。

---

## 风险与回滚

| 风险 | 缓解 |
|---|---|
| `MarkdownHeaderTextSplitter` 对 `knowledge.md` 内特殊格式（如 LaTeX 公式 `\\(...\\)`）解析异常 | M4.7.2 测试矩阵覆盖"无 header / 含 LaTeX / 重复正文"用例；无 header fallback 到 recursive |
| `char_offset` 用 best-effort 子串匹配可能不准 | offset 仅用于诊断；使用单调 `_find_offset(..., start_at)` 避免重复正文导致 offset 回退 |
| header prefix 过长导致单 chunk 超过 `chunk_size_chars` | 普通场景把 prefix 计入 body 预算；极端长 header 保留完整 breadcrumb，测试只约束普通 header 场景 |
| chunk 数变多导致 embedding 时间增加 | `max_chunks_per_doc=200` 上限继续生效；CPU 60s 预算依然充足（实测 task_11 < 10 chunks） |
| 新依赖与 `langchain-core` 版本冲突 | `langchain-text-splitters>=0.3,<0.4` 与 `langchain-core>=0.3.84,<0.4` 同一 minor，官方保证兼容 |

**回滚预案**：git revert M4.7.1–M4.7.4 commit 即可，依赖与代码同时撤销。

---

## 估时

| Task | 估时（含 TDD） |
|---|---|
| M4.7.1 依赖 | 10-15 min |
| M4.7.2 chunker 实现 + 17 个测试 | 75-105 min |
| M4.7.3 factory 接入 | 20 min |
| M4.7.4 边界回归 | 10-15 min |
| M4.7.5 E2E A/B | 10 min |
| **合计** | **~2.5-3 小时** |
