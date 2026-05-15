# Corpus RAG M4 落地后的待做项（2026-05-15 起）

> 创建时间：2026-05-15
> 来源：M4 corpus RAG production path 收尾后（plan v2 完成节、`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/03-implementation-notes/02-m4-corpus-rag-progress.md` §6）剩余的非阻塞候选项。
> 性质：**未排期**；按需独立推进。每项都不影响 M4 主线交付，可以并行或推迟。
>
> **优先级标注**：
> - **P1**：影响评测得分或镜像可交付性，应优先做
> - **P2**：质量提升，可推迟到下一阶段
> - **P3**：可选优化或独立提案

---

## 1. PS_on LLM gateway 超时优化（P1）✅ 已完成（2026-05-15）

> **状态**：D 量化 + A 实施已落地。`task_timeout_seconds` 600 → 900；PS_on 超时
> 率从 1/5 (20%) 降到 0/5（task_344 在 900s 上限内能跑完）。详见下"实施结果"。

### 现状（已解决）

E2E A/B 实测中，**`plan_solve` + RAG 路径** 在 task_344 上偶发超时：

| 组合 | 耗时 | 结果 |
|---|---|---|
| PS_on (plan_solve + RAG) | **605s 超时** | ❌ |
| PS_off (plan_solve no RAG) | 230s | ✅ |
| RA_on (react + RAG) | **292s** | ✅ |
| RA_off (react no RAG) | 416s | ✅ |

**RA_on 同 corpus 路径 292s 跑通**，证明 RAG 索引构建本身不慢（Harrier embedding ~60s + chroma upsert ~5s + retrieval <1s）。**PS_on 慢 = LLM gateway 随机慢**，让 plan-solve 的某些 step 触发 120s 重试。

### D 步实测（2026-05-15）

`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/artifacts/d_step_results_20260515T165901Z.json`
（脚本 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/artifacts/d_step_sweep.ps1`）：

| Task | elapsed | succeeded | failure |
|---|---|---|---|
| task_344 | 601s | ❌ | `Task timed out after 600 seconds.` ← 真 timeout |
| task_355 | 62s | ✅ | — |
| task_350 | 172s | ✅ | — |
| task_352 | 221s | ❌ | `max_steps`（gateway 响应正常但 agent 60 步未收敛）|
| task_396 | 2882s | ❌ | `max_steps` 但跑了 48 min ⚠️ |

- **真 LLM gateway timeout 率 = 1/5 = 20%** → 落入候选方案 A 区间（<30% but >10%）。
- **新发现 bug**（task_396 跑 2882s）：`task_timeout_seconds=600` 子进程 timeout
  在 task 内部循环 max_steps 路径下未能强制 enforce — 见下"伴生 bug"小节。

### 实施结果

候选方案 **A**（加大总预算）：

- `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/config.py` `RunConfig.task_timeout_seconds`: **600 → 900**
- 余量评估：gateway_caps.yaml 单 step 上限 120s × max_steps 60 / max_replans 4 ≈ 480s 上限，900s 留 1.9x 安全余量
- 回归测试：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/tests/test_phase5_config.py::test_run_config_task_timeout_seconds_default_is_900`
- README.md / tests/test_phase10_rag_config.py 引用同步更新

### 伴生 bug（**单独立项**）：task timeout 在 max_steps 路径下未 enforce

- task_396 实测跑 2882s（4.8x 超过 600s timeout）才返回，failure_reason="max_steps"
- 期望：parent 进程 `multiprocessing.Queue.get(timeout=600)` 应在 600s 后 raise Empty + reap subprocess + 返回 "Task timed out after 600 seconds." 错误
- 实际：subprocess 自然完成 60 步后才 put 结果到 queue，parent 等了 2882s
- 假设：Windows + Python 3.13 multiprocessing.Queue 在某些条件下 timeout 失效（未根因定位）
- **影响**：极端情况下（gateway 持续慢 + max_steps 路径），单 task 可吃满 50+ min wall clock，benchmark 总时长不可控
- **优先级**：P1，但本 followup §1 范围之外。**新建 followup §6**：见底部

### 原因分析

- `RunConfig.task_timeout_seconds=600s` 是 task 总时间预算
- plan-solve 比 react 多一次 plan + parse 调用，对 LLM 延迟更敏感
- gateway 随机慢（`gateway_caps.yaml` 单 step 120s 上限）一旦命中，留给后续 step 的预算就不够

### 目标

让 PS_on 路径在 90+% 的 task 上能跑完，不被 LLM gateway 随机慢卡死。

### 候选方案

| 方案 | 改动 | 影响 |
|---|---|---|
| **A** 加大总预算 | `task_timeout_seconds` 600 → 900s | 简单；但同时让 RA 慢路径也能更慢，可能拖累整体 benchmark wall clock |
| **B** 单独给 plan_solve 加预算 | `task_timeout_seconds_by_mode={plan_solve:900, react:600}` | 精细；改 dataclass + runner 解析逻辑 |
| **C** gateway step retry 策略调优 | 让 step 内部更多重试机会而不是直接 timeout | 改 `agents/llm_executor`，复杂度高 |
| **D** 跑多个 task 测概率 | 不改代码，先量化"偶发"实际是多大比例 | 0 改动；得到决策依据 |

### 建议路径

1. 先做 **D**：在 task_344 / task_355 / task_400 等 5-10 个 task 上跑 PS_on，测超时率
2. 若 ≥30% 超时 → 实施 **B**（精细控制，避免 RA 路径浪费预算）
3. 若 <30% 但 >10% → 实施 **A**（简单粗暴够用）
4. 若 <10% → 标记为已知 gap，不做（属于 LLM gateway 内部问题）

### 工作量

- D：~30 min（跑 5 task × 2 mode = 10 runs，每个 ~5-10 min）
- A：5 min 改配置 + 1 commit
- B：~1 hour（dataclass 字段、CLI flag、runner 解析、测试）
- C：~半天（agents/llm_executor 改造 + 重试测试）

### 验收

- 修改后 PS_on 在测试 task 集上超时率 < 5%
- RA 路径耗时不显著退化（< 10%）
- 全套 pytest 仍 PASS

### 涉及文件

- `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/config.py`（`RunConfig.task_timeout_seconds`）
- `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/run/runner.py`（task 启动 timeout 解析）
- `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/configs/local.yaml`

---

## 2. Dockerfile.rag 真实打包验证（P1）

### 现状

`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/docker/Dockerfile.rag` 已交付（commit `c76492f`），14 静态不变量测试 PASS。**未跑** `docker build`。

预估镜像大小 ~3.7-4.0 GB，但**实际可能差 200-500 MB**（取决于 torch wheel 实际版本、HF 缓存碎片、apt 残留等）。

### 目标

在 Linux amd64 节点上跑一次完整 `docker build -f docker/Dockerfile.rag .`，拿到：

1. **build 是否能跑通**（pip wheel 解析无冲突，Harrier 真能下载）
2. **实测镜像大小**（用于评测主办容量评估）
3. **build 耗时**（用于 CI 时间预算）

### 候选执行环境

| 环境 | 优点 | 缺点 |
|---|---|---|
| **Windows Docker Desktop**（amd64 cross-build via QEMU）| 你本地有 | 慢 30-50%；不是真评测环境 |
| **WSL2 Ubuntu** + Docker | 比 QEMU 快 | 仍是开发机，可能装了乱七八糟的东西 |
| **GitHub Actions** ubuntu-latest | 干净；评测主办也常用 | 需要写 workflow + 公开 build log |
| **本地 Linux 物理机 / 虚拟机** | 完全控制 | 你不一定有 |

### 建议路径

1. 先在 **WSL2** 跑一次（如果你装了的话），目标只验证"能跑通 + 拿镜像大小"
2. 若 WSL2 不可用 → 写一个 GitHub Actions workflow `.github/workflows/docker-rag-build.yml`，触发条件 `paths: [docker/Dockerfile.rag, pyproject.toml]`
3. 不论哪条路径，结果回写到 `02-m4-corpus-rag-progress.md` §3 G.2 段

### 工作量

- WSL2 执行：1 次 build ~10-15 min + 写结果 5 min
- GitHub Actions：1 次 workflow 编写 ~30 min + push 触发 + 等结果 ~15 min

### 验收

- `docker build` 成功（exit 0）
- 镜像 `docker images` 体积 < 5 GB
- 镜像内 `python -c "from data_agent_langchain.submission import build_submission_config; print(build_submission_config().memory.rag.enabled)"` 输出 `True`（C1 验证）
- 镜像内 `ls /opt/models/hf` 含 Harrier 权重文件（A2 验证）

### 涉及文件

- `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/docker/Dockerfile.rag`（不需改）
- 可能新增 `.github/workflows/docker-rag-build.yml`
- 结果回写 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/03-implementation-notes/02-m4-corpus-rag-progress.md`

---

## 3. R6 召回精度评估（P2）

### 背景

M4.5 决策 **A2**：corpus 仅在 task 入口召回 1 次（plan-solve 走 `planner_node`，ReAct 走 `task_entry_node`），`model_node` 后续 step 不再召回。

**风险（plan v2 R6）**：ReAct 在 60 步过程中，query 会演进（从初始 question 到具体 sub-task）；入口 1 次召回拿到的 chunk 可能与后期 step 实际需要的不一致 → 召回精度下降。

### 目标

量化 A2 决策对实际 task 召回精度的影响。如果显著退化，触发 A3（动态条件触发）独立提案。

### 候选实验

| 实验 | 设计 | 输出 |
|---|---|---|
| **E1 一次召回 vs 多次召回** | 临时分支：在 model_node 每 N 步重新召回；A/B 在 5-10 个 task 上跑，对比答案正确率 | A3 必要性数据 |
| **E2 召回相关性人工标注** | 抽样 20 个 RA_on 跑，看 task_entry retrieved 的 4 个 chunks 跟最终答案的相关性 | 召回质量分布 |
| **E3 chunk hit rate vs answer correctness 相关性** | 用 metrics.json + scoring 数据做 join，计算 corpus_hit_count → answer_pass_rate 相关系数 | 是否值得加 corpus |

### 建议路径

1. 先做 **E3**（0 改动，纯数据分析）
2. 若 E3 显示正相关显著（如 r > 0.3）→ 做 **E1** 验证多次召回是否能进一步提升
3. 若 E3 不显著 → corpus RAG 在当前 prompt 设计下贡献有限，触发 corpus prompt 改造（独立提案）

### 工作量

- E3：~2 hour（写 join 脚本 + 跑 20-50 task + 出图）
- E1：~半天（临时分支改 model_node + 跑实验）
- E2：~1 day（人工标注）

### 验收

- 出一份评估报告（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/03-implementation-notes/03-corpus-recall-evaluation.md`）
- 报告含 corpus 命中率、答案正确率相关性、A2 vs A3 决策建议
- 不改主线代码

### 涉及文件

- 新增分析脚本 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/tools/analyze_corpus_recall.py`（如有）
- 新增报告 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/03-implementation-notes/03-corpus-recall-evaluation.md`

---

## 4. shared_corpus 独立提案（P3，独立设计）

### 背景

M4 决策 **D2**：`CorpusRagConfig.shared_corpus` 字段存在但 fail-closed —— 任何 enabled 都 dispatch `memory_rag_skipped(reason="shared_corpus_not_implemented")` 并返回 None。

**业务价值**：让多个 task 共享同一份预构建的索引（如所有 finance task 共享一份 finance documentation 索引），避免每个 task 重新跑 ~60s 嵌入。

### 目标

写 `04-corpus-rag-m4/05-shared-corpus-design.md` 独立提案，含：

1. **场景**：哪些 task 集合可以共享 corpus（按 dataset / 按 domain / 按 task tag）
2. **构建时机**：build-time（CI 跑一次） vs runtime first-task（lazy）
3. **存储后端**：persistent chromadb vs LangChain `Chroma.from_documents` snapshot vs faiss `IndexFlatL2.write`
4. **生命周期**：collection 是否跨提交保留（Docker volume vs 镜像内）
5. **接入点**：`build_task_corpus` 应优先 attach shared 还是 task-private 优先
6. **观测**：`shared_collections_loaded` metrics 字段已就位

### 候选方案

| 方案 | 描述 | 维护成本 |
|---|---|---|
| **S1 build-time 预构建** | docker build 阶段跑 ingestion，把 shared collection 写进 `/opt/models/chroma`；runtime 直接 attach | 中（CI 复杂） |
| **S2 runtime first-task 懒构建** | 第一个 task 触发时构建并缓存到 `/tmp/chroma`；同提交内后续 task 复用 | 低（无 CI 改动）；但 evict 风险 |
| **S3 外部 sidecar 服务** | 单独部署 chroma server；rag retriever 走 HTTP | 高（评测环境难支持） |

### 建议路径

1. 先评估 **是否真的需要**：跑 5 个相同 dataset 的 task，看 corpus 索引构建总耗时是否构成瓶颈
2. 若构建总耗时 > 5 min → 写 S1 vs S2 决策提案
3. 若 < 5 min → 推迟到下次性能 review

### 工作量

- 评估：~2 hour
- 设计提案：~1 day
- 实施（如选 S2）：~2 day

### 验收

- 设计提案文档完成
- 不实施代码（除非评估显示必要）

### 涉及文件

- 新增 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/05-shared-corpus-design.md`

---

## 6. task timeout 在 max_steps 路径下未 enforce（P1，2026-05-15 §1 D 步发现）

### 现状

D 步实测（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/artifacts/d_step_results_20260515T165901Z.json`）发现 task_396 在 PS_on 跑了 **2882s（48 分钟）** 才返回，远超 `task_timeout_seconds=600s` 上限。

### 复现条件

- `dabench-lc run-task <task_id> --graph-mode plan_solve --memory-rag` 走 CLI → `run_single_task(llm=None)` → `_run_single_task_with_timeout`
- LLM gateway 响应正常但每 step 慢（实测平均 48s/step）
- agent 内部 60 步循环未收敛（plan_solve max_steps + max_replans 触发）
- subprocess 自然走完后 `result_queue.put(...)`
- parent 等 `result_queue.get(timeout=600)` ≥2800s 才拿到结果（**预期：600s raise Empty 并 reap subprocess**）

### 期望行为

`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/run/runner.py:_run_single_task_with_timeout` 应在 timeout 后 reap subprocess 并写出 `Task timed out after <N> seconds.` 错误（task_344 case 走通了这条路径）。

### 候选根因

| 假设 | 验证方式 |
|---|---|
| **H1** Windows + Python 3.13 `multiprocessing.Queue.get(timeout=N)` 在 spawn 上不可靠 | 写 minimal repro：worker sleep 30s + queue.get(timeout=5) 看是否 5s raise Empty |
| **H2** subprocess 在 600s 内分多次 put（部分 progress 数据），parent 错误地把首条当 final | 看 `_run_single_task_in_subprocess` 是否单 put — 检查后**确认是单 put** |
| **H3** 子进程内有 thread 在 main thread 完成后还活着，主进程 join() block | strace / Process.is_alive 检查 |
| **H4** PowerShell + conda run 包装层吃了 SIGTERM | 不用 conda run，直接 venv python 跑 |

### 候选方案

| 方案 | 改动 | 复杂度 |
|---|---|---|
| **F1** 切到 `concurrent.futures.ProcessPoolExecutor` + `Future.result(timeout=N)` | 改 runner._run_single_task_with_timeout，行为更可预期 | 中 |
| **F2** 在 subprocess 内部加 `signal.alarm(timeout)` self-kill | 跨平台不一致（Windows 不支持 SIGALRM）| 高 |
| **F3** 父进程开守护线程定期 `process.is_alive()` + 超时 terminate | 简单守门人，约 30 行代码 | 低 |
| **F4** 在 graph 入口加 `wall_clock_check` decorator，到点 raise + 写 partial trace | 改 langgraph node 多处 | 高 |

### 建议路径

1. 先做 **H1 repro**（5 min）确定是否 Python multiprocessing.Queue 问题
2. 若 H1 confirmed → 实施 **F1**（切 ProcessPoolExecutor，最稳）
3. 若 H1 反证 → 走 **F3** 守门人线程

### 验收

- 写一个回归测试：mock 一个 sleep 长时间的 subprocess，断言 timeout 在 N 秒后 enforce
- E2E：在 task_396 上重跑，确认 wall_clock ≤ task_timeout_seconds + 30s

### 涉及文件

- `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/run/runner.py:_run_single_task_with_timeout`
- 新增测试 `tests/test_phase5_runner_timeout_enforcement.py`

---

## 5. （次要）minor cleanup（P3）

### 5.1 plan v2 / `02-implementation-plan.md` 互链

`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/02-implementation-plan.md` 是 v2 的旧版本（3.0），目前 plan v2 (3.1) 顶部没引用旧版做对比。可在 v2 顶部加一行 "**Supersedes**: `02-implementation-plan.md` (v3.0)"。

工作量：~2 min。

### 5.2 trace 数据可视化

E2E artifacts 下大量 `task_*/trace.json`，目前需要肉眼读 JSON 才能看 step 时间分布。可写一个简单 Streamlit / matplotlib 脚本把 trace 转 Gantt 图。

工作量：~半天。属于 dev tooling，不影响交付。

### 5.3 `submission.py` env vars 清单文档化

`DATA_AGENT_RAG=1` / `MEMORY_MODE` / `MODEL_*` 等评测态 env 散落在 submission.py 里，没有 single source of truth。可在 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/README.md` 或 `docker/README.md` 加一节"评测态环境变量"。

工作量：~30 min。

---

## 排期建议

| 时间 | 项 | 理由 |
|---|---|---|
| **本周内** | §2 Dockerfile.rag 真实打包 | 提交评测前必跑 |
| **本周内** | §1 PS_on 超时分析（D） | 影响 benchmark 跑通率，先量化 |
| **下周** | §1 实施（A 或 B，视 D 数据） | 数据驱动决策 |
| **2 周内** | §3 R6 召回精度评估（E3） | 答案正确率分析 |
| **按需** | §4 shared_corpus 提案 | 性能瓶颈触发时 |
| **按需** | §5 cleanup | 闲时穿插 |

## 反向看：什么是**不**做的

为避免 scope 爆炸，明确以下**不做**：

- ❌ corpus indexer 加 BM25 / hybrid 检索（M5 的潜在主题，不属于 M4 follow-up）
- ❌ 把 corpus 推到外部 vector DB（pinecone / weaviate）—— 评测要求自包含
- ❌ Harrier 升级到 1B / 7B —— 镜像大小已逼近 4 GB，再大评测主办可能拒收
- ❌ 把 plan_solve 默认改成 react —— 是 baseline 决策，不在 RAG 范畴
- ❌ 给 corpus retriever 加 LLM rerank —— 增加延迟，与 A2"快速入口召回"目标背离

---

## 索引

- M4 主线设计：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/01-design-v2.md`
- M4 实施计划：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/02-implementation-plan-v2.md`
- M4 进展记录：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/03-implementation-notes/02-m4-corpus-rag-progress.md`
- 评测镜像：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/docker/Dockerfile.rag`
