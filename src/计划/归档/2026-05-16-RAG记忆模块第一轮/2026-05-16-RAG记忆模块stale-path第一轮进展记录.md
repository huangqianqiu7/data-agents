# RAG/记忆模块 stale path guard 第一轮进展记录

## 1. 记录目的

本文记录 `2026-05-16-RAG记忆模块工具提示词调整计划.md` 第一轮实现、验证结果与后续扩展验证口径。

本轮目标不是一次性解决全部失败任务，而是先验证以下假设：

- stale dataset facts 会诱导 agent 重复访问当前任务不存在的文件或数据库路径。
- 禁用 dataset facts prompt 注入、增加 memory/RAG authority policy、并在 `list_context` 后启用 `known_paths` hard block，可以显著降低 missing context asset 重试。
- corpus RAG recall 应保持可用。
- 原成功任务不应因为路径 hard block 被误伤。

## 2. 已完成实现

当前实现已合并到本地 `main`。

- 最终实现提交：`f422417` (`test(rag): opt in dataset facts prompt injection`)
- 合并后验证：`python -m pytest -q`，结果为 `609 passed, 3 skipped, 49 warnings`

主要变更：

| 方向 | 变更 |
|---|---|
| dataset facts 注入 | 新增 `MemoryConfig.inject_dataset_facts`，默认 `False`，第一轮默认不把 prior-run dataset facts 注入 prompt |
| memory/RAG 权威策略 | 在 dataset facts 与 corpus snippets 渲染中加入 rendering-only authority policy，明确当前 task 工具 observation / `list_context` 高于 memory/RAG |
| 路径硬约束 | `list_context` 成功后记录当前 task 的 `known_paths`；当 `agent.enforce_known_path_only=True` 且 discovery 已完成时，对未知文件/DB 路径 hard block |
| missing asset 分类 | 文件/DB 缺失路径统一转为 validation-style tool error，避免被当作普通 runtime error 继续原样重试 |
| 回归测试 | 更新旧 RAG prompt 测试，使需要 dataset facts 的用例显式 opt in |

## 3. 第一轮实验设置

Baseline：

```text
artifacts/runs/20260515T162819Z
```

Variant：

```text
artifacts/runs/first_round_stale_path_guard_20260516T044638Z
```

实验摘要：

```text
artifacts/runs/first_round_stale_path_guard_20260516T044638Z/first_round_metrics_summary.md
```

实验配置覆盖：

```text
memory.mode=read_only_dataset
memory.rag.enabled=True
memory.inject_dataset_facts=False
agent.enforce_known_path_only=True
```

运行模式：

```text
graph_mode=plan_solve
```

验证任务：

| task | 用途 |
|---|---|
| `task_355` | stale CSV path 典型失败样本，baseline 重复访问 `csv/member.csv` |
| `task_418` | stale DB path 典型失败样本，baseline 重复访问 `db/results.db` |
| `task_259` | 原成功任务，用于验证 hard block 不误伤正常路径 |

## 4. 第一轮实验结果

| task | baseline succeeded | variant succeeded | baseline max missing retry | variant max missing retry | baseline tool calls | variant tool calls | prompt tokens delta | variant corpus recall | variant hard blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `task_355` | False | True | 54 | 0 | 60 | 9 | -1,623,606 | 1 | `{}` |
| `task_418` | False | False | 53 | 0 | 60 | 60 | +396,351 | 1 | `{}` |
| `task_259` | True | True | 0 | 0 | 7 | 9 | +24,910 | 0 | `{}` |

### 4.1 `task_355`

结论：通过。

- Baseline 反复读取当前 task 不存在的 `csv/member.csv`，最大 missing retry 为 54。
- Variant 未再访问 `csv/member.csv`，missing context asset 次数为 0。
- Variant 成功提交答案。
- 工具调用从 60 降到 9。
- Prompt tokens 从 1,689,850 降到 66,244。
- corpus recall 保留，`corpus_task:task_355` 召回 1 次。

### 4.2 `task_418`

结论：stale DB path 修复通过，但任务整体仍失败。

- Baseline 反复 inspect 当前 task 不存在的 `db/results.db`，最大 missing retry 为 53。
- Variant 未再访问 `db/results.db`，missing context asset 次数为 0。
- Variant 未再调用 `inspect_sqlite_schema`。
- corpus recall 保留，`corpus_task:task_418` 召回 1 次。
- Variant 仍然 max steps 失败，失败模式转移为 `execute_python` 循环。

该结果说明第一轮 stale path guard 达到了目标，但 `task_418` 还需要后续单独处理非路径类循环。

### 4.3 `task_259`

结论：通过。

- Baseline 成功，Variant 仍成功。
- Variant 没有 missing path，也没有 known path hard block 误伤。
- Variant 出现一次 SQL schema mismatch：`no such table: comments`，但 agent 恢复并成功提交答案。

## 5. 第一轮结论

第一轮核心验收通过：

- `task_355` 的 stale CSV path loop 被消除，并从失败转为成功。
- `task_418` 的 stale DB path loop 被消除，虽然任务整体仍失败。
- corpus RAG recall 在 `task_355` 和 `task_418` 中仍正常记录。
- 原成功任务 `task_259` 未被 hard block 误伤。

需要注意：

- 当前实现主要处理 missing file / DB path，不处理 SQL schema mismatch 或一般计算循环。
- `dataset:input` memory recall 仍会发生，但默认不注入 prompt；本轮重点是读侧止血与运行时路径约束。
- 如果后续要恢复 dataset facts 的收益，应优先设计 task-scoped namespace，而不是恢复裸 `dataset:input` 自动注入。

## 6. 扩展验证计划

下一步可以进入扩展验证：

```text
task_350
task_408
```

两者必须分开解读。

### 6.1 `task_350` 解读口径

`task_350` 与 `task_355` 类似，主要用于验证 stale CSV path guard 是否泛化。

重点指标：

- `csv/event.csv` 或其他当前 context 不存在 CSV 的 missing retry 是否从几十次下降到不超过 1。
- `read_csv` 总调用是否下降。
- 是否出现 known path hard block 误伤。
- corpus RAG recall 是否正常记录。

如果 `task_350` 仍失败，但 missing path retry 已显著下降，应判定为 stale path guard 泛化有效，但任务可能存在其他推理/数据理解问题。

### 6.2 `task_408` 解读口径

`task_408` 不是第一轮 stale path 的纯验证样本，而是 SQL/schema loop 观察样本。

重点指标：

- 是否仍有 missing context asset retry。
- `no such table` / `no such column` 是否重复出现。
- `execute_context_sql` 是否仍围绕同一错误 schema 假设循环。
- `inspect_sqlite_schema` 后是否仍无视实际 schema。
- corpus RAG recall 是否正常记录。

如果 `task_408` 仍失败，不能直接判定第一轮失败；应归入第二轮 SQL/schema loop guard 或 schema authority policy 问题。

## 7. 扩展验证结果

扩展验证 run：

```text
artifacts/runs/extended_stale_path_guard_20260516T052921Z
```

扩展验证摘要：

```text
artifacts/runs/extended_stale_path_guard_20260516T052921Z/extended_validation_metrics_summary.md
```

扩展验证仍使用与第一轮相同的配置覆盖：

```text
memory.mode=read_only_dataset
memory.rag.enabled=True
memory.inject_dataset_facts=False
agent.enforce_known_path_only=True
```

指标对比：

| task | baseline succeeded | variant succeeded | baseline max missing retry | variant max missing retry | baseline schema mismatch | variant schema mismatch | baseline tool calls | variant tool calls | prompt tokens delta | variant corpus recall | variant hard blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `task_350` | False | True | 50 | 0 | 1 | 0 | 60 | 12 | -1,192,134 | 1 | `{}` |
| `task_408` | False | True | 0 | 0 | 49 | 1 | 60 | 13 | -1,496,681 | 1 | `{}` |

### 7.1 `task_350`：stale CSV path 泛化验证

结论：通过。

- Baseline 反复访问当前 task 不存在的 `csv/event.csv`，最大 missing retry 为 50。
- Variant 没有 missing context asset 调用，最大 missing retry 为 0。
- Variant 成功提交答案。
- `read_csv` 调用从 51 降到 1。
- 总工具调用从 60 降到 12。
- Prompt tokens 从 1,280,670 降到 88,536。
- corpus recall 保留，`corpus_task:task_350` 召回 1 次。

该结果说明 `task_355` 上验证过的 stale path guard 可以泛化到另一个 stale CSV path 样本。

### 7.2 `task_408`：schema/SQL loop 观察样本

结论：本次扩展 run 成功，但仍应与 stale path 结果分开解读。

- Baseline 没有 missing context asset loop，但有 49 次 schema mismatch，其中 `no such table: races` 出现 46 次。
- Variant 成功提交答案。
- Variant schema mismatch 降到 1 次。
- `execute_context_sql` 调用从 54 降到 7。
- 总工具调用从 60 降到 13。
- Prompt tokens 从 1,587,799 降到 91,118。
- corpus recall 保留，`corpus_task:task_408` 召回 1 次。

该结果说明本次配置下 `task_408` 不再复现旧的 schema/SQL loop，但它仍不是第一轮 stale path guard 的纯样本。若未来 full benchmark 中仍出现重复 `no such table` / `no such column`，应归入第二轮 schema mismatch / deterministic failure loop guard 设计。

## 8. 后续建议

短期建议：

1. 将第一轮 stale path guard 视为已通过最小闭环与扩展验证。
2. 继续保留 `task_408` 的 schema/SQL loop 指标，用于后续 full benchmark 观察。
3. 若后续 full benchmark 中仍出现重复 deterministic failures，再单独设计 schema mismatch / repeated deterministic failure loop guard。
4. 在进入 task-scoped dataset memory 前，不恢复裸 `dataset:input` 自动 prompt 注入。

不建议在完成 full benchmark 对照前继续修改 RAG top-k、prompt budget 或动态 recall 策略，以免混淆归因。

## 9. Full benchmark 手动验证计划

### 9.1 执行边界

下一步需要跑第一轮 stale path guard 后的 50-task full benchmark，但该 benchmark 由用户在本机手动启动。

本记录只固定：

- 运行前配置核对项。
- 推荐命令。
- 运行后记录模板。
- 结果解读与后续分支决策。

在 full benchmark 对照完成前，不继续修改 RAG top-k、prompt budget、动态 recall 策略或 dataset facts 注入策略，避免混淆第一轮 guard 的归因。

### 9.2 运行前配置核对

本次 full benchmark 应沿用第一轮和扩展验证相同的行为组合：

```text
graph_mode=plan_solve
memory.mode=read_only_dataset
memory.rag.enabled=True
memory.inject_dataset_facts=False
agent.enforce_known_path_only=True
```

注意：

- CLI 目前可直接覆盖 `memory.mode` 和 `memory.rag.enabled`。
- `memory.inject_dataset_facts` 与 `agent.enforce_known_path_only` 需要由本地 YAML 配置或等价本地实验配置保证。
- 不恢复裸 `dataset:input` dataset facts 自动 prompt 注入。
- corpus RAG 继续保留，用于确认 `corpus_task:{task_id}` recall 在 full benchmark 下仍正常。
- RAG offline env 已由 CLI 在 `memory.mode=read_only_dataset` 且 `--memory-rag` 时默认设置；若本地显式设置了 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`，以本地显式值为准。

### 9.3 推荐手动命令

推荐命令：

```powershell
dabench-lc run-benchmark --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag
```

如果希望减少进度条输出，可使用：

```powershell
dabench-lc run-benchmark --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag --no-progress
```

运行完成后，CLI 会输出：

```text
run: artifacts/runs/<new_run_id>
tasks: 50
succeeded: <N>
```

需要记录 `<new_run_id>`，作为后续和 baseline 对照的 Variant run。

### 9.4 对照基线

Baseline 固定为：

```text
artifacts/runs/20260515T162819Z
```

Variant 待手动 benchmark 完成后填写：

```text
artifacts/runs/<new_run_id>
```

对照重点不是只看最终成功率，而是观察失败结构是否从 stale path loop 迁移到其他更明确的问题类型。

### 9.5 运行后记录模板

```markdown
## Full benchmark 对照结果

- 日期：
- commit：
- Baseline run：`artifacts/runs/20260515T162819Z`
- Variant run：`artifacts/runs/<new_run_id>`
- 配置：
  - graph_mode：
  - memory.mode：
  - memory.rag.enabled：
  - memory.inject_dataset_facts：
  - agent.enforce_known_path_only：
- 总体结果：
  - baseline succeeded：
  - variant succeeded：
  - baseline failed：
  - variant failed：

### 旧失败任务状态

| task | baseline status | variant status | stale path retry | schema mismatch | main new failure mode | notes |
|---|---|---|---:|---:|---|---|
| `task_330` | failed | | | | | |
| `task_344` | failed | | | | | |
| `task_350` | failed | | | | | |
| `task_352` | failed | | | | | |
| `task_355` | failed | | | | | |
| `task_396` | failed | | | | | |
| `task_408` | failed | | | | | |
| `task_418` | failed | | | | | |

### 关键指标

| metric | baseline | variant | interpretation |
|---|---:|---:|---|
| total succeeded | 42 | | |
| total failed | 8 | | |
| max missing retry per path | | | |
| total missing context asset | | | |
| total schema mismatch | | | |
| max schema mismatch per signature | | | |
| total tool calls | | | |
| total prompt tokens | | | |
| corpus recall retained | | | |

### 初步结论

- stale path guard 是否在 full benchmark 泛化：
- 是否出现 known path hard block 误伤：
- `task_350` / `task_355` 是否保持成功：
- `task_418` 是否仍转移为非路径类循环：
- `task_408` 是否仍复现 schema/SQL loop：
- 是否进入第二轮 schema mismatch / deterministic failure loop guard：
- 是否进入 R6 corpus recall evaluation：
```

### 9.6 结果解读口径

#### 9.6.1 第一轮 guard 成功标准

可以将第一轮 stale path guard 视为 full benchmark 通过的条件：

- `task_350` 和 `task_355` 不再重复访问当前 context 不存在的 CSV path。
- `task_418` 不再重复 inspect 当前 context 不存在的 `db/results.db`。
- 原成功任务没有因为 `known_paths` hard block 大量退化。
- corpus RAG recall 仍能在有 task corpus 的任务中记录。
- 总 tool calls 和 prompt tokens 在旧 stale path 失败样本上明显下降。

#### 9.6.2 不应误判为第一轮失败的情况

以下情况不应直接判定 stale path guard 失败：

- `task_418` 不再访问 stale DB path，但仍因 `execute_python` 循环失败。
- `task_408` 仍因 `no such table` / `no such column` 失败。
- `task_330` / `task_344` / `task_352` / `task_396` 仍因数据理解、映射、文档精确匹配或搜索策略失败。

这些应归入后续独立主题，而不是回滚第一轮 stale path guard。

### 9.7 后续分支决策

Full benchmark 完成后按以下顺序决策：

1. 如果 stale path retry 在旧样本中仍为 0 或接近 0，且没有明显 regression，则关闭第一轮 stale path guard 主题。
2. 如果仍高频出现 `no such table` / `no such column` 原样重复，再启动第二轮 schema mismatch / deterministic failure loop guard。
3. 如果主要新失败集中在 `execute_python` 重复执行，则单独设计 execution loop guard 或 answer forcing，不与 SQL/schema guard 混在同一轮。
4. 如果 RAG_on full benchmark artifacts 稳定，再进入 R6 E3 corpus recall evaluation。
5. 如果需要恢复 dataset memory 收益，优先设计 task-scoped dataset memory，不恢复裸 `dataset:input` 自动 prompt 注入。

### 9.8 暂缓事项

在 full benchmark 对照完成并记录前，暂缓：

- 调整 `memory.rag.retrieval_k`。
- 调整 `memory.rag.prompt_budget_chars`。
- 增加动态多次 recall。
- 让 planner 提前使用更多 recall 内容。
- 恢复 dataset facts 默认 prompt 注入。
- 设计 curated shared dataset memory。

## 10. Full benchmark 对照结果

### 10.1 运行信息

- 日期：2026-05-16
- Baseline run：`artifacts/runs/20260515T162819Z`
- Variant run：`artifacts/runs/20260516T061021Z`
- 评分文件：
  - `Comparison/20260515T162819Z.csv`
  - `Comparison/20260516T061021Z.csv`

Variant 沿用第一轮 stale path guard 目标配置：

```text
graph_mode=plan_solve
memory.mode=read_only_dataset
memory.rag.enabled=True
memory.inject_dataset_facts=False
agent.enforce_known_path_only=True
```

### 10.2 总体结果

| metric | baseline | variant | delta |
|---|---:|---:|---:|
| task count | 50 | 50 | 0 |
| agent succeeded | 42 | 42 | 0 |
| agent failed | 8 | 8 | 0 |
| score sum | 31.45 | 33.90 | +2.45 |
| score mean | 0.629 | 0.678 | +0.049 |
| consistency: 一致 | 31 | 33 | +2 |
| consistency: 不一致 | 11 | 9 | -2 |
| consistency: 结果生成失败 | 8 | 8 | 0 |
| prompt tokens | 14,085,670 | 12,895,415 | -1,190,255 |
| total tokens | 14,341,532 | 13,156,306 | -1,185,226 |
| wall clock p50 | 89.03s | 80.58s | -8.45s |
| wall clock p95 | 362.97s | 408.69s | +45.72s |
| wall clock max | 451.33s | 459.27s | +7.94s |

解读：

- Agent 层成功数没有提升，仍为 `42/50`。
- 评分层面提升明显，score sum 从 `31.45` 升到 `33.90`。
- 第一轮 stale path guard 的直接收益主要体现在旧 hard 失败样本 `task_350` / `task_355` 从结果生成失败变为满分。
- 同时出现新的生成失败样本 `task_199` / `task_283`，因此 agent succeeded 总数没有变化。

### 10.3 工具调用与错误模式

| metric | baseline | variant | delta |
|---|---:|---:|---:|
| `list_context` | 210 | 247 | +37 |
| `read_doc` | 75 | 73 | -2 |
| `read_json` | 36 | 34 | -2 |
| `read_csv` | 165 | 42 | -123 |
| `inspect_sqlite_schema` | 99 | 32 | -67 |
| `execute_context_sql` | 159 | 217 | +58 |
| `execute_python` | 181 | 228 | +47 |
| `answer` | 42 | 42 | 0 |
| total missing context asset | 195 | 8 | -187 |
| max missing retry per path | 54 (`task_355`) | 4 (`task_344`) | -50 |
| total schema mismatch | 55 | 51 | -4 |
| max schema mismatch per signature | 46 (`task_408`, `races`) | 42 (`task_408`, `races`) | -4 |

解读：

- Missing path 问题从 `195` 次降到 `8` 次，第一轮 stale path guard 在 full benchmark 中泛化成立。
- `read_csv` 从 `165` 降到 `42`，`inspect_sqlite_schema` 从 `99` 降到 `32`，说明 stale CSV / stale DB path loop 明显减少。
- `execute_context_sql` 和 `execute_python` 上升，说明剩余失败主要迁移到 SQL/schema loop、Python loop、重复 list/search loop，而不是原 stale path loop。
- Schema mismatch 总数变化不大，且仍集中在 `task_408`。

### 10.4 旧失败任务状态

| task | baseline status | variant status | variant main pattern | stale path retry | schema mismatch | corpus recall |
|---|---|---|---|---:|---:|---:|
| `task_330` | failed | failed | `csv/Match.csv` 被当作 SQL DB 反复查询；伴随 `list_context` loop | 3 | 0 | 1 |
| `task_344` | failed | failed | 反复 `list_context`；仍尝试不存在 DB / 文件名 | 4 | 0 | 1 |
| `task_350` | failed | succeeded | stale CSV path 消除，成功提交 | 0 | 0 | 1 |
| `task_352` | failed | failed | `list_context` / `read_doc` loop，无 missing path / schema error | 0 | 0 | 1 |
| `task_355` | failed | succeeded | stale CSV path 消除，成功提交 | 0 | 0 | 1 |
| `task_396` | failed | failed | `list_context` loop，无 missing path / schema error | 0 | 0 | 1 |
| `task_408` | failed | failed | `execute_context_sql` schema loop，`races` / `drivers` 表不存在 | 0 | 45 | 1 |
| `task_418` | failed | failed | stale DB path 消除，但迁移为 `execute_python` 成功调用循环 | 0 | 0 | 1 |

关键样本：

- `task_350`：`read_csv` 降到 1 次，成功提交，score `1.0`。
- `task_355`：`read_csv` 降到 1 次，成功提交，score `1.0`。
- `task_418`：不再访问 `db/results.db`，但 `execute_python` 调用 51 次，仍 max steps。
- `task_408`：`execute_context_sql` 调用 56 次，其中 `no such table: races` 42 次、`no such table: drivers` 3 次，第二轮 schema guard 触发条件成立。

### 10.5 新增失败与分数变化

Variant 失败任务：

```text
task_199, task_283, task_330, task_344, task_352, task_396, task_408, task_418
```

相对 baseline 的分数变化：

| task | baseline score | variant score | delta | status change |
|---|---:|---:|---:|---|
| `task_173` | 1.00 | 0.00 | -1.00 | 一致 → 不一致 |
| `task_283` | 1.00 | 0.00 | -1.00 | 一致 → 结果生成失败 |
| `task_199` | 0.00 | 0.00 | 0.00 | 不一致 → 结果生成失败 |
| `task_249` | 0.45 | 1.00 | +0.55 | 不一致 → 一致 |
| `task_180` | 0.00 | 0.95 | +0.95 | 不一致 → 一致 |
| `task_379` | 0.00 | 0.95 | +0.95 | 不一致 → 不一致 |
| `task_350` | 0.00 | 1.00 | +1.00 | 结果生成失败 → 一致 |
| `task_355` | 0.00 | 1.00 | +1.00 | 结果生成失败 → 一致 |

新增失败解读：

- `task_199`：baseline 虽然提交了答案，但 score 已为 `0.0`；variant 失败主要是 `execute_python` 中 `name 'frm' is not defined` 重复 39 次。
- `task_283`：baseline score `1.0`，variant 失败；trace 显示 `execute_context_sql` 成功调用 57 次但没有提交答案，属于重复成功工具调用 / answer forcing 问题，而不是 stale path。
- `task_173`：仍提交答案但评分从 `1.0` 降到 `0.0`，需要单独看答案差异，暂不归因到 stale path guard。

### 10.6 第一轮结论

第一轮 stale path guard 可以视为通过 full benchmark 验证：

- `task_350` / `task_355` 在 full benchmark 中保持成功。
- `task_418` 的 stale DB path loop 被消除。
- Missing context asset 总数从 `195` 降到 `8`。
- 旧 stale path 样本的 prompt token 和工具调用显著下降。
- Corpus RAG recall 在旧失败任务中仍保留，均有 `corpus_task:{task_id}` recall。

但本轮没有提升 agent succeeded 总数：

- 旧失败中恢复了 `task_350` / `task_355`。
- 新增失败中出现了 `task_199` / `task_283`。
- 总体仍为 `42/50`。

因此，本轮结论应表述为：

```text
stale path loop 问题已被有效压制，并带来评分提升；
剩余主要瓶颈已迁移到 SQL/schema loop、execute_python loop、list_context loop 和 answer forcing。
```

### 10.7 后续决策

下一步建议：

1. 关闭第一轮 stale path guard 主题，不回滚当前实现。
2. 新开第二轮 `schema mismatch / deterministic failure loop guard`，优先覆盖 `task_408`：
   - `no such table` / `no such column` 不允许同表名原样重试几十次。
   - `inspect_sqlite_schema` 成功后，SQL 应受 schema observation 约束。
   - 错误反馈应携带 available tables / columns。
3. 单独设计 `repeated successful tool call / answer forcing`，覆盖 `task_283`：
   - 工具调用成功且结果稳定重复时，应强制 synthesize answer 或 replan。
4. 单独设计 `execute_python deterministic failure loop guard`，覆盖 `task_199` / `task_418`：
   - 相同 Python NameError 或等价代码片段重复失败时，不允许继续原样执行。
   - Python 成功但无新信息时，应转入答案合成或重规划。
5. `task_330` / `task_344` / `task_352` / `task_396` 暂归入 list/search/data-understanding loop，不与 SQL/schema guard 混在同一轮。
6. R6 corpus recall evaluation 可以基于 `20260516T061021Z` 继续做 E3，但不应阻塞第二轮 deterministic loop guard。
