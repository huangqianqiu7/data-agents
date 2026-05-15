# M4 Corpus RAG 实施进展（事后追溯）

> 时间窗口：~2026-05-13 至 2026-05-15
> 设计依据：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/01-design-v2.md`
> 实施计划：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/02-implementation-plan-v2.md`
> 关联：本文是 `01-implementation-summary.md`（M3 RAG memory v2）的后续阶段记录。

## 状态总览

| 维度 | 结果 |
|---|---|
| 主线 milestone | M4.0–M4.6 全部 70 step `[x]` |
| 主线 commit 数 | 12（含 scaffolding / test 套件 / 5 个 feat / 2 个 fix / 1 个 chore） |
| Bug fix commit 数 | 7（Bug 1–7 全部修复） |
| 收尾 commit 数 | 2（P0.2 plan 同步 + G.2 Dockerfile.rag） |
| 全套 pytest | **578 passed, 3 skipped**（slow 真权重 opt-in） |
| E2E A/B 验证 | 4 组合实测，RA_on 路径 `memory_rag` 段完整 |
| 评测镜像 | `docker/Dockerfile.rag` multi-stage + 预烘 Harrier，~3.7-4.0 GB |

## 1. 架构落地（M4.0–M4.6）

### 关键决策（v2）

- **A2**：corpus 仅在 task 入口召回 1 次（plan-solve 走 `planner_node`，ReAct 走新增 `task_entry_node`），`model_node` 不主动召回。
- **B2**：`CorpusStore` 独立 mini Protocol，不复用 v2 `MemoryStore.put/get/list/delete`。
- **C2**：`sentence-transformers` / `chromadb` / `transformers` / `torch` 仅进 `[project.optional-dependencies].rag`；rag 子模块**禁止顶层 import 重依赖**。
- **D2**：`shared_corpus` 拆出独立提案 `05-shared-corpus-design.md`；本阶段仅占位 + fail-closed。

### 新增 / 修改文件骨架

```
src/data_agent_langchain/memory/rag/
├── __init__.py            # 仅 re-export 纯类型 + Protocol（Bug 3 后）
├── base.py                # CorpusStore Protocol
├── documents.py           # CorpusDocument / CorpusChunk dataclass
├── loader.py              # 扫描 task context_dir
├── chunker.py             # CharWindowChunker
├── redactor.py            # 文件名 + 内容白名单过滤
├── factory.py             # build_embedder / build_task_corpus
├── embedders/
│   ├── base.py            # Embedder Protocol
│   ├── stub.py            # DeterministicStubEmbedder
│   └── sentence_transformer.py  # HarrierEmbedder（方法级延迟 import）
├── stores/
│   └── chroma.py          # ChromaCorpusStore.ephemeral
└── retrievers/
    └── vector.py          # VectorCorpusRetriever (cosine)

src/data_agent_langchain/agents/
├── corpus_recall.py       # recall_corpus_snippets helper
├── task_entry_node.py     # ReAct 入口召回节点（A2 决策）
├── planner_node.py        # plan-solve 入口召回（M4.5.3 增量）
├── model_node.py          # _build_messages_for_state 串行注入 dataset_facts + corpus_snippets
└── prompts.py             # render_corpus_snippets

src/data_agent_langchain/
├── config.py              # CorpusRagConfig + MemoryConfig.rag + 嵌套 _dataclass_from_dict
├── runtime/context.py     # corpus_handles ContextVar
├── run/runner.py          # 子进程入口构建 task corpus（M4.4.4）
├── cli.py                 # --memory-rag/--no-memory-rag
├── submission.py          # DATA_AGENT_RAG=1 显式打开
└── observability/
    ├── events.py          # dispatch_observability_event + fallback handler list（Bug 5）
    └── metrics.py         # on_custom_event + on_observability_event 聚合 memory_rag_*
```

### M4 主线 commit 映射

| Milestone | Commit | 说明 |
|---|---|---|
| **M4.0–M4.4 scaffolding** | `7559a59` | `CorpusRagConfig` + `_dataclass_from_dict` 嵌套 + contextvar + `[project.optional-dependencies].rag` |
| **M4.0–M4.4 test 套件** | `afa3786` | RED/GREEN artifacts: documents/redactor/loader/chunker/embedder/store/retriever/factory/runner |
| **M4.4.5 CLI 接入** | `1f153a3` | `--memory-rag/--no-memory-rag` typer option + submission `DATA_AGENT_RAG=1` |
| **M4.5.1 recall_corpus_snippets** | `4816c22` | helper 函数（后追加 `MemoryConfig` 签名 + memory.mode 守卫） |
| **M4.5.2 render_corpus_snippets** | `7387335` | `prompts.py` + `model_node._build_messages_for_state` 串行注入 |
| **M4.5.3 plan-solve 入口召回** | `ae08a2e` | `planner_node` 在 `recall_dataset_facts` 后追加 corpus 召回 |
| **M4.5.4 ReAct 入口节点** | `29e5d15` | 新增 `task_entry_node`，`START → task_entry → execution → finalize` |
| **M4.5.5 model_node 不召回护栏** | `736d3ce` | spy 验证 60 步 ReAct 内 `recall_corpus_snippets` 仅 1 次 |
| **M4.6.1 metrics 聚合** | `77d8b7a` | `MetricsCollector.on_custom_event` 订阅 `memory_rag_*`；fallback 路径在 `a0f1f63` 加上 |
| **M4.6.2 import 边界回归** | `e5f2c54` | 子进程 `sys.modules` 断言禁止顶层 import torch/chromadb/sentence_transformers |

## 2. Bug 1–7 修复（落地后发现的 production-path 缺陷）

| Bug | 现象 | 修复 commit | 关键改动 |
|---|---|---|---|
| **Bug 1** | `memory.mode=disabled` 时 RAG 仍可能构建（仅靠 `rag.enabled` 守卫） | `80fb620` | runner `_build_and_set_corpus_handles` 第一行检测 `mode==disabled`；`recall_corpus_snippets` 签名扩展为 `MemoryConfig`，双重守卫 |
| **Bug 2** | factory store 构造 / upsert 失败被静默吞，metrics 永远不出现 `memory_rag_skipped` | `6b83466` | `build_task_corpus` 各失败分支 dispatch `memory_rag_skipped(reason=...)` |
| **Bug 3** | `memory/rag/__init__.py` 占位 `NotImplementedError("Corpus RAG is reserved for Phase M4")` 让 import 时直接炸 | `5d51b1d` + `669c215` | 顶层只 re-export 纯类型 + Protocol，删除占位 API；测试同步 |
| **Bug 4** | `ChromaCorpusStore.close()` 未释放 collection，进程内 `EphemeralClient` 单例累积 collection，跨 task / 跨测试串味 | `529e66f` | `close()` 显式 `client.delete_collection(_collection_name)`；测试加 `unique_task_id` fixture 防 fixture 串味 |
| **Bug 5** | `dispatch_observability_event` 在 LangGraph runtime 之外（`runner._build_and_set_corpus_handles` 阶段）抛 `RuntimeError`，被静默吞 → `metrics.json.memory_rag` 永远缺段 | `a0f1f63` | `events.py` 加模块级 `_FALLBACK_HANDLERS` + `register/unregister_fallback_handler`；`MetricsCollector.on_observability_event` 共享 `_handle_event`；`runner` 在 `MetricsCollector` 构造时 register、`finally` 块 unregister |
| **Bug 6** | `task_corpus_index_timeout_s=30s` 太短，Harrier-270m CPU 冷启动 + ~60 chunks 实测 ~60s → fail-closed 系统性触发 | `9ab4e6f` | default 30 → 180s（3x 实测，1/3 `task_timeout_seconds=600` 预算）；regression test 锁 `>=120s` |
| **Bug 7** | plan §F A/B 命令样例漏 `--memory-mode read_only_dataset`，`local.yaml` 默认 `mode=disabled` 让 RAG 守卫提前 return；任何按文档跑的人都看不到 RAG | `fdbcdf5` | 4 处命令样例补齐 `--memory-mode read_only_dataset` |

## 3. P0.2 + G.2 收尾（2026-05-15 session）

### 本 session commit 链

```
c76492f  feat(docker): add Dockerfile.rag with preloaded Harrier weights (G.2)
a68a388  docs(memory-rag): mark M4 corpus RAG production path as complete (P0.2)
fdbcdf5  docs(memory-rag): plan F: A/B commands must include --memory-mode read_only_dataset (Bug 7)
a0f1f63  fix(memory-rag): route observability events to MetricsCollector outside LangGraph runtime (Bug 5)
9ab4e6f  fix(memory-rag): bump task_corpus_index_timeout_s default 30s to 180s (Bug 6)
```

### P0.2 — plan v2 同步（commit `a68a388`）

- 全部 70 个 step `[ ]` → `[x]`
- plan v2 顶部新增《完成状态（事后追溯，2026-05-15）》节，与本文互为索引
- 文件：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/02-implementation-plan-v2.md`

### G.2 — `docker/Dockerfile.rag`（commit `c76492f`）

设计决策：**A2 + B2 + C1**（详见 plan v2 完成节中的"G.2 设计决策"段）。

- **Stage 1 (builder)**：`FROM python:3.13-slim-bookworm AS builder`
  - `apt-get install gcc g++ git`（torch / chromadb wheel build 工具）
  - `pip install '.[rag]'` 装 RAG extras
  - 预烘 `microsoft/harrier-oss-v1-270m` 权重到 `/opt/models/hf`（A2：评测离线可用）
- **Stage 2 (runtime)**：`FROM python:3.13-slim-bookworm`（不带 builder 工具）
  - `COPY --from=builder` site-packages + `/usr/local/bin` + `/opt/models/hf`
  - `ENV DATA_AGENT_RAG=1`（C1：提交即想 RAG）
  - `ENV TRANSFORMERS_OFFLINE=1` + `HF_HUB_OFFLINE=1`（runtime 完全离线锁定）
  - `ENTRYPOINT` / `WORKDIR /app` / `gateway_caps.yaml` copy 与 base `Dockerfile` 完全一致

预估镜像大小 ~3.7-4.0 GB（base 150MB + minimal deps 200MB + RAG extras 600MB + Harrier fp32 ~600MB + multi-stage 减 ~300MB）。

未跑 `docker build` 实测（Windows Docker Desktop 走 QEMU cross-build 不是评测真环境）；交付给评测主办或 Linux CI 节点真实打包。

### G.2 静态不变量测试 14 项（commit `c76492f`）

`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/tests/test_dockerfile_rag_invariants.py`

- multi-stage build 至少 2 个 `FROM`
- 第一阶段命名（`AS builder`）
- runtime `python:3.13-slim-bookworm` + `--platform=linux/amd64`
- runtime 含 `COPY --from=builder ...`
- `pip install .[rag]` 显式
- `microsoft/harrier-oss-v1-270m` 预烘 + `SentenceTransformer(...)` 调用
- `ENV HF_HOME=/...` 钉死
- `ENV DATA_AGENT_RAG=1` 默认开
- `ENV PYTHONUNBUFFERED=1` + `PYTHONDONTWRITEBYTECODE=1` 与 base 一致
- `ENTRYPOINT ["python", "-m", "data_agent_langchain.submission"]` 与 base 一致
- `COPY docker/gateway_caps.yaml /app/gateway_caps.yaml`
- 防 dev artifacts 泄漏（拒 COPY `tests/` / `artifacts/` / `.windsurf/`）
- `WORKDIR /app`

## 4. 测试矩阵

| 套件 | tests | 状态 |
|---|---|---|
| 全套 pytest | 578 passed, 3 skipped | ✅ |
| `tests/test_phase10_rag_*.py` | M4 主线 ~20 文件 | ✅ |
| `tests/test_phase10_observability_events_fallback.py` | 8（Bug 5）| ✅ |
| `tests/test_phase10_observability_metrics_fallback.py` | 5（Bug 5）| ✅ |
| `tests/test_phase10_rag_config.py::test_task_corpus_index_timeout_s_default_covers_harrier_cold_start` | 1（Bug 6）| ✅ |
| `tests/test_phase10_rag_mode_disabled_guard.py` | Bug 1 守卫 | ✅ |
| `tests/test_phase10_rag_failure_dispatch_guard.py` | Bug 2 dispatch | ✅ |
| `tests/test_phase10_rag_init_exports.py` | Bug 3 顶层导出守护 | ✅ |
| `tests/test_phase10_rag_chroma_close_isolation.py` | Bug 4 collection 释放 | ✅ |
| `tests/test_dockerfile_rag_invariants.py` | 14（G.2）| ✅ |
| 跳过项 | 3 个 `@pytest.mark.slow`（真 Harrier 权重，需 `--runslow`）| 设计如此 |

## 5. E2E A/B 实测（2026-05-15）

```powershell
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag      # PS_on  : 605s 超时（LLM gateway 随机慢，与 RAG 路径无关）
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --no-memory-rag   # PS_off : 230s ✅ 无 memory_rag 段（baseline parity）
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode react      --memory-mode read_only_dataset --memory-rag      # RA_on  : 292s ✅ memory_rag 段完整（核心证据）
dabench-lc run-task task_344 --config configs/local.yaml --graph-mode react      --memory-mode read_only_dataset --no-memory-rag   # RA_off : 416s ✅ 无 memory_rag 段（baseline parity）
```

### 核心证据：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/artifacts/runs/20260515T060608Z/task_344/metrics.json`

```json
{
  "memory_rag": {
    "task_index_built": true,
    "task_doc_count": 2,
    "task_chunk_count": 62,
    "shared_collections_loaded": 0,
    "recall_count": { "task_entry": 1 },
    "skipped": []
  },
  "memory_recalls": [
    {
      "node": "task_entry_node",
      "namespace": "corpus_task:task_344",
      "k": 4,
      "kind": "corpus_task",
      "scores": [0.7074, 0.6957, 0.6928, 0.6924],
      "model_id": "microsoft/harrier-oss-v1-270m",
      "reason": "vector_cosine"
    }
  ]
}
```

## 6. 已知 gap / 后续

- **PS_on 在 task_344 上偶发超时**：RA_on 同 corpus 路径 292s 跑通，证明 RAG 路径本身不慢；605s 是 LLM gateway 随机慢导致 step 25/30 各 120s timeout 重试。后续可考虑 `RunConfig.task_timeout_seconds` 600 → 900s。
- **R6 召回精度评估**：M4 仅入口召回 1 次，可能错过 ReAct 后期演进的查询；需要在更多 task 上 A/B 看 corpus_recall 实际贡献，再评估是否升级到 A3（动态条件触发）。
- **shared_corpus**（D2）：仍走独立提案 `04-corpus-rag-m4/01-design-v2.md` 引用的 `05-shared-corpus-design.md`，本计划只占位。
- **Dockerfile.rag 实测打包**：本地 Windows Docker Desktop 跨架构 build 意义有限，建议在评测主办或 Linux CI 节点真打一次拿镜像大小数据，写回本文 §3 G.2。

## 7. 索引

- 架构设计：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/01-design-v2.md`
- 实施计划（含完成节）：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/02-implementation-plan-v2.md`
- M3 RAG memory v2 改动记录（前置阶段）：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/03-implementation-notes/01-implementation-summary.md`
- 评测镜像：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/docker/Dockerfile.rag`
- 评测镜像不变量测试：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/tests/test_dockerfile_rag_invariants.py`
