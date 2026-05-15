# R6 Corpus 召回精度评估 — 详细实施计划

> 版本：v1.0，2026-05-15
> 父文档：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/待做/01-corpus-rag-m4-followups.md` §3
> 优先级：**P2**（不阻塞主线交付，影响后续 A2 → A3 决策）
> **For agentic workers:** REQUIRED SUB-SKILL: 使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` 逐任务推进；每个 `- [ ]` 步骤独立可验收。

---

## 1. Goal & 决策上下文

### Goal

量化 **M4.5 A2 决策**（corpus 仅在 task 入口召回 1 次）在实际 task 上的召回精度表现，输出一份评估报告，**数据驱动**地回答：

1. **corpus 命中**是否与**答案正确率**正相关？相关性是否显著？
2. 多次召回（每 N 步重召）是否能进一步提升答案正确率？
3. 是否需要升级到 **A3**（动态条件触发，独立提案）？

### 决策上下文

- **A2（已落地）**：plan-solve 走 `planner_node`，ReAct 走 `task_entry_node`，`model_node` 后续 step 不再召回。
  - 源码：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/task_entry_node.py:14-38`、`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/planner_node.py:41-74`、`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/corpus_recall.py:20-97`
- **A2 风险**（v2 plan R6 原文）：ReAct 60 步内 query 会从初始 question 演进到具体 sub-task；入口 1 次召回的 chunks 与后期 step 实际所需可能不一致 → 召回精度下降。
- **A3 候选**：动态条件触发——当 ReAct 出现连续 N 步 tool 失败 / 出现新关键词时再次召回。**本计划不实施 A3，只评估 A2 是否够用**。

### Exit criteria（本计划完成的判据）

- 产出 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/03-implementation-notes/03-corpus-recall-evaluation.md` 评估报告。
- 报告必须含：
  1. corpus_hit_count → answer_score / 一致性 的相关系数（Pearson / Spearman）。
  2. PS_on vs PS_off、RA_on vs RA_off 在 ≥20 task 上的 mean Score 差异。
  3. A2 vs A3 决策建议（继续 A2 / 升级 A3 / 改 corpus prompt 设计）。
- 不改主线代码；E1 临时分支不合入 main，分析脚本可单独入库 `tools/`。
- 回写父 followup §3 状态行 + 链接评估报告。

---

## 2. 数据来源（artifact contract）

> 所有路径在 Windows 上为反斜杠绝对路径；脚本写法用 `pathlib.Path` 处理跨平台。

### 2.1 per-task metrics（已存在）

`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/artifacts/runs/<run_id>/<task_id>/metrics.json`

样例（task_350，RAG 启用路径）：

```json
{
  "task_id": "task_350",
  "succeeded": true,
  "memory_recalls": [
    {
      "node": "planner_node",
      "namespace": "corpus_task:task_350",
      "k": 4,
      "kind": "corpus_task",
      "hit_ids": ["cb79c265b6eeb91e#0039", "..."],
      "hit_chunk_ids": ["cb79c265b6eeb91e#0039", "..."],
      "hit_doc_ids": ["cb79c265b6eeb91e", "..."],
      "scores": [0.7148, 0.7123, 0.6894, 0.6883],
      "model_id": "microsoft/harrier-oss-v1-270m",
      "reason": "vector_cosine"
    }
  ],
  "memory_rag": {
    "task_index_built": true,
    "task_doc_count": 3,
    "task_chunk_count": 127,
    "shared_collections_loaded": 0,
    "recall_count": {"planner": 1},
    "skipped": []
  },
  "wall_clock_s": 169.17
}
```

**E3 主输入字段**：

| 字段 | 含义 |
|---|---|
| `memory_rag.task_index_built` | 该 task 是否真构建了 corpus 索引（false 即 RAG 未起作用）|
| `memory_rag.task_chunk_count` | 索引 chunk 总数（corpus 规模）|
| `memory_rag.recall_count.{planner,task_entry}` | 入口召回次数（A2 下恒为 0 或 1）|
| `memory_rag.skipped[]` | fail-closed 原因（如 `retrieve_failed`、`factory_import_failed`）|
| `memory_recalls[*].scores` | 每次召回的 cosine score（top-k 列表）|
| `memory_recalls[*].hit_doc_ids` / `hit_chunk_ids` | 命中文档 / chunk id（可用于 E2 抽样标注）|

### 2.2 答案评分（bench_comparison 现成）

`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/bench_comparison.py` 已实现：

- 读 `artifacts/runs/<run_id>/<task_id>/prediction.csv`
- 对 `data/public/output_ans/<task_id>/gold.csv`
- 输出 `Comparison/<run_id>.csv`，列含 `任务` `难度` `一致性` `得分` `召回率` `匹配列数` `预测列数` `多余列数`
- 评分公式（rules.zh.md §6.3）：`Score = max(0, Recall − λ·Extra/Pred)`，λ=0.1

**E3 主输出锚点**：`得分`（连续值，0–1）与 `一致性`（"一致" / "不一致" / 异常类）。

### 2.3 Task 元数据

- 任务难度：`data/public/input/<task_id>/task.json` → `difficulty`
- 任务输入文档：`data/public/input/<task_id>/` 下的 doc 文件（用于 E2 人工标注溯源）

### 2.4 关键 source 文件（只读）

- `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/corpus_recall.py:20-97` — 召回 + dispatch `memory_recall` 事件
- `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/observability/metrics.py:67-148` — 聚合 `memory_rag` 段
- `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/model_node.py`（E1 临时分支需在这里加重召回点）

---

## 3. 阶段划分与决策树

```
E3（数据分析，0 改动）
   ├── 相关系数 r > 0.3 ──── E1（多次召回 A/B，临时分支）
   │                          ├── 多次召回 mean Score 提升 ≥ 3% → 建议升级 A3
   │                          └── 提升 < 3%             → 维持 A2，触发 corpus prompt 改造提案
   └── 相关系数 |r| ≤ 0.3 ─── 触发 corpus prompt 改造提案（不做 E1）
                              └── 可选 E2（抽样标注 20 个，定性看 corpus chunk 与答案相关性）
```

- E3 = 必做（gate 后续阶段）
- E1 = 条件触发
- E2 = 备选（无论 E3 结论如何都可选做）

---

## 4. E3 — 0 改动数据分析（必做，~2h）

### 4.1 输入

- 至少 **1 个全集 run**（≥ 50 个 task）的 `RAG_on` 数据 + 对应的 `RAG_off` 对照 run
- 历史 run 满足条件的候选：
  - `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/artifacts/runs/20260510T130735Z/` （50 task PS）
  - `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/artifacts/runs/example_run_id_19_ps_all/` （150 task PS，需确认是否 RAG_on）
  - 若历史 run 不含 RAG_on 字段（M4 前），需要重跑一次至少 30 task 的 RAG_on benchmark

### 4.2 步骤（checkbox）

- [ ] **E3.1 数据盘点**：跑一个 1-shot 脚本 `tools/list_rag_runs.py`，扫 `artifacts/runs/*/task_*/metrics.json`，按 `memory_rag.task_index_built==true` 筛选出"RAG 实际生效"的 run。输出表：`run_id, task_count, rag_on_count`。
- [ ] **E3.2 确认 baseline run**：若 ≥1 个全集 run 满足条件，直接进 E3.4。否则跑：
  ```powershell
  dabench-lc run-benchmark --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag      # RAG_on
  dabench-lc run-benchmark --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --no-memory-rag   # RAG_off
  ```
  注意：使用 `task_timeout_seconds=900`（§1 已落地），避免 PS_on 在长尾 task 上 timeout 污染样本。
- [ ] **E3.3 跑 `bench_comparison.py`**：对上述 2 个 run 各跑一次，得 `Comparison/<run_id>.csv`。
- [ ] **E3.4 写分析脚本** `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/tools/analyze_corpus_recall.py`：
  - 输入：`--rag-on-run <run_id>` `--rag-off-run <run_id>` `--ans-csv-on <Comparison/xxx.csv>` `--ans-csv-off <Comparison/yyy.csv>` `--output report.json`
  - 计算：
    - **per-task join**：以 `task_id` 为键，左右连接 metrics.json + Comparison CSV
    - **派生字段**：
      - `corpus_hit_count` = 1 if `memory_rag.task_index_built && recall_count.{planner|task_entry}>0` else 0
      - `corpus_chunks_used` = `sum(memory_rag.recall_count.values())` × `retrieval_k`（取 metrics 自报值，不假设）
      - `top1_score` = `memory_recalls[corpus_task][0].scores[0]`（缺失置 NaN）
      - `mean_recall_score` = `mean(scores)`
    - **统计**：
      - `corpus_hit_count` × `得分`：Pearson r、Spearman ρ、p-value
      - `top1_score` × `得分`：Pearson r、Spearman ρ
      - 分组对比：`难度 ∈ {easy, medium, hard}` × `RAG_on/off` 的 mean `得分`
      - paired diff：`task_id` 同时在 on/off 中存在的，看 mean(score_on - score_off)
  - 输出 JSON + 简单 matplotlib 散点图（PNG，落到 `artifacts/r6/`）
- [ ] **E3.5 单测分析脚本**：写 `tests/test_tools_analyze_corpus_recall.py`，至少：
  - 构造 mock `metrics.json` + mock Comparison CSV（fixture 4 个 task）
  - 断言 join 后行数正确、`corpus_hit_count` 派生逻辑正确、相关系数计算无 NaN 传染
- [ ] **E3.6 跑分析**：`python tools/analyze_corpus_recall.py --rag-on-run <X> --rag-off-run <Y> --ans-csv-on Comparison/X.csv --ans-csv-off Comparison/Y.csv --output artifacts/r6/e3_report.json`
- [ ] **E3.7 写中间结论**：先把 E3 数据填到评估报告 §2，**不下 final conclusion**（等 E1 / E2 决策完毕）。

### 4.3 E3 验收

- `tools/analyze_corpus_recall.py` 在 `pytest` 下绿
- `artifacts/r6/e3_report.json` 含全部字段，无 NaN 在主指标列（除非真无数据）
- 评估报告 §2 写明 r、ρ、p-value、样本量

---

## 5. E1 — 多次召回 A/B（条件触发，~半天）

> **触发条件**：E3.4 输出 `corpus_hit_count` × `得分` 的相关系数 `|r| > 0.3` **且** p < 0.05。

### 5.1 临时分支策略

- 分支名：`exp/r6-multi-recall`（**不合入 main**；实验完即可 archive）
- 工作树：使用 `superpowers:using-git-worktrees` 隔离

### 5.2 代码改动

- 在 `model_node`（`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/model_node.py`）的 `_build_messages_for_state` 前增加一段：
  - 读 `state.step_index`；当 `step_index > 0 && step_index % N == 0` 时（N 默认 10），用当前 step 的 `last_observation` 或 `pending_subtask` 作为 query 再调 `recall_corpus_snippets`。
  - 把新召回 chunks merge 进 `memory_hits`（去重按 `record_id`）
- 加可选配置 `CorpusRagConfig.recall_every_n_steps: int = 0`（0 = 维持 A2 行为，与 main 兼容）
- dispatch `memory_recall` 事件时 `node="model_node"`，自动归类到 `recall_count.model`

### 5.3 步骤（checkbox）

- [ ] **E1.1 worktree**：`git worktree add ../worktrees/r6-multi-recall -b exp/r6-multi-recall`
- [ ] **E1.2 改 `CorpusRagConfig`**：加 `recall_every_n_steps` 字段 + dataclass 测试 `tests/test_phase10_rag_config.py`
- [ ] **E1.3 改 `model_node`**：实现多次召回；保持 `recall_every_n_steps=0` 时与 main 行为 bit-identical
- [ ] **E1.4 spy 测试**：扩展 `tests/test_phase10_rag_model_node_recall_disabled.py`，新增 case `recall_every_n_steps=10` 验证 60 步内 corpus_recall 被调 ≥ 5 次
- [ ] **E1.5 跑 A/B**：在 E3 选定的同一批 ≥ 30 task 上跑 3 个 run：
  - A2（main 行为，已有 RAG_on run，可复用）
  - 多次召回 N=10（新 run）
  - 多次召回 N=5（新 run）
- [ ] **E1.6 比较 mean Score**：扩展 `tools/analyze_corpus_recall.py` 接受多 run 输入，输出 mean Score 与 wall_clock 对比表
- [ ] **E1.7 记录决策**：把数据写入评估报告 §3

### 5.4 E1 验收

- 主线 main 在 `recall_every_n_steps=0` 时全套 pytest 仍 PASS
- 多次召回 A/B 数据写入 `artifacts/r6/e1_report.json`
- 评估报告含 mean Score 与 wall_clock 对比

### 5.5 E1 退出条件

- mean Score 提升 ≥ 3% **且** wall_clock 退化 < 15% → 建议主线接入（升级 A3）
- mean Score 提升 < 3% **或** wall_clock 退化 ≥ 15% → 维持 A2，分支废弃

---

## 6. E2 — 人工标注 20 个抽样（备选，~1 day）

> 何时做：E3 显示相关性弱（`|r| < 0.3`）时，用人工标注定性确认 corpus chunks 是否真的"相关但模型没用上"vs"chunks 本身就不相关"。

### 6.1 抽样方法

- 从 E3 join 后的 RAG_on 数据中：
  - **正样本**：corpus 有命中且 `一致性=="一致"` 的 task 抽 10 个
  - **负样本**：corpus 有命中但 `一致性=="不一致"` 的 task 抽 10 个

### 6.2 标注表格

落到 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/artifacts/r6/e2_labels.csv`，列：

| task_id | chunk_rank (0–3) | chunk_doc_id | source_path | snippet | 与最终答案相关度 (0–3) | 备注 |
|---|---|---|---|---|---|---|

- 相关度评分 rubric：
  - 0 = 完全无关
  - 1 = 同领域但与答案无直接关联
  - 2 = 部分相关（含答案所需的 ~30% 信息）
  - 3 = 高度相关（含答案所需的核心信息或直接给出答案）

### 6.3 步骤（checkbox）

- [ ] **E2.1 抽样脚本**：在 `tools/analyze_corpus_recall.py` 加 `--sample-for-labeling N --out-csv labels.csv` 子命令，输出 20 行 chunk-level 数据，`相关度` 列留空
- [ ] **E2.2 人工标注**：用户在 Excel / VSCode 内填 `相关度` 列
- [ ] **E2.3 回收分析**：脚本读回填后的 csv，计算 mean 相关度（正/负样本组分别），写入评估报告 §4
- [ ] **E2.4 结论写入**：若正样本组 mean 相关度 ≥ 2.0 且负样本组 ≤ 1.0 → corpus 本身相关，模型用不好 → 触发 prompt 改造提案

### 6.4 E2 验收

- `artifacts/r6/e2_labels.csv` 20 行填满
- 评估报告 §4 写明正/负样本组 mean 相关度差异

---

## 7. 评估报告结构（最终交付）

`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/03-implementation-notes/03-corpus-recall-evaluation.md`

模板：

```markdown
# Corpus 召回精度评估报告（R6）

> 完成时间：YYYY-MM-DD
> 计划：`@.../src/计划/待做/02-R6-corpus-recall-evaluation-plan.md`
> 父决策：M4.5 A2 — corpus 仅 task 入口召回 1 次

## 1. 摘要（TL;DR）
- 样本量：N task
- corpus_hit_count × 得分 Pearson r = X.XX (p=X.XX)
- A2 vs 多次召回 A/B：mean Score 差 +X.X% / wall_clock 差 +X.X%
- 决策：[维持 A2 / 升级 A3 / 触发 prompt 改造]

## 2. E3 数据分析
- 数据源 run id / 样本量 / 难度分布
- 相关系数表 + 散点图
- 分组对比表

## 3. E1 多次召回 A/B（如执行）
- 实验设计
- mean Score 对比表
- wall_clock 对比表
- 结论

## 4. E2 人工标注（如执行）
- 样本抽法
- 正/负组 mean 相关度
- 典型样例

## 5. 决策建议
- A2 / A3 / prompt 改造的具体建议
- 是否新建 followup 项

## 6. 附录
- 分析脚本路径
- 中间数据 JSON 路径
- 原始 run id 列表
```

---

## 8. 回写父 followup（计划完成后必做）

> **触发时机**：评估报告 §1 摘要写完且决策确定后

需要修改 `@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/待做/01-corpus-rag-m4-followups.md` §3：

- [ ] **W.1 状态行**：标题改 `## 3. R6 召回精度评估（P2）✅ 已完成（YYYY-MM-DD）` 或 `## 3. R6 召回精度评估（P2）⏸ 数据已出，决策维持 A2`
- [ ] **W.2 加"实施结果"小节**：摘要 + 评估报告链接
- [ ] **W.3 排期表**：把 §3 行的"理由"列改 "已完成，详见 03-corpus-recall-evaluation.md"
- [ ] **W.4 若 E1 显示需升级 A3**：在 §3 之后新建 `## 7. A3 动态条件触发召回（P2，2026-XX-XX R6 派生）` 占位
- [ ] **W.5 提交 commit**：`docs(rag): close R6 corpus recall eval, decision <…>`

---

## 9. 工作量估算

| 阶段 | 估时 | 必做 |
|---|---|---|
| E3 数据分析（含脚本 + 测试） | 2-3 h | ✅ |
| E3 跑 benchmark（若缺数据） | +30-90 min | 视情况 |
| E1 多次召回 A/B | 4-6 h | 条件触发 |
| E2 人工标注 | 4-8 h | 备选 |
| 评估报告撰写 | 1-2 h | ✅ |
| 回写 followup | 15 min | ✅ |
| **总计（最小路径，仅 E3）** | **3-4 h** | |
| **总计（E3 + E1）** | **8-10 h** | |
| **总计（E3 + E1 + E2 全做）** | **12-18 h** | |

---

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 历史 run 缺 `memory_rag` 段（M4 前的 run） | E3.2 步重新跑 RAG_on benchmark；用 `task_index_built==true` 严筛 |
| `bench_comparison.py` LLM 评分异常导致 `一致性` 列含"评估异常" | 分析脚本剔除异常行，按 `得分`（算法 score）做主指标，`一致性` 仅做次要校验 |
| PS_on 长尾 task timeout 污染样本 | 依赖 §1 已落地的 `task_timeout_seconds=900`；分析脚本剔除 `failure_reason=="Task timed out"` 的 task |
| E1 临时分支与 main 漂移 | 限制实验周期 ≤ 1 周；rebase 频繁；用 worktree 隔离 |
| Pearson r 受异常值影响大 | 同时报 Spearman ρ；样本 ≥ 30 |
| 难度分布不均（hard task 过少） | 报告按难度分组，hard < 5 时仅作定性说明 |

---

## 11. 索引

- 父 followup：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/待做/01-corpus-rag-m4-followups.md` §3
- A2 决策来源：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/01-design-v2.md`
- A2 实施记录：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/04-corpus-rag-m4/02-implementation-plan-v2.md` §M4.5
- M4 完成记录：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/03-implementation-notes/02-m4-corpus-rag-progress.md` §6
- 召回主代码：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/agents/corpus_recall.py`
- 评分主代码：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/bench_comparison.py`
- metrics 聚合主代码：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/data_agent_langchain/observability/metrics.py`
- 评估报告产出位置（最终）：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/src/计划/RAG记忆架构演示/03-implementation-notes/03-corpus-recall-evaluation.md`
- 分析脚本产出位置（最终）：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/tools/analyze_corpus_recall.py`
- 中间数据落盘目录：`@/c:/Users/18155/learn python/Agent/kddcup2026-data-agents-starter-kit-master/artifacts/r6/`
