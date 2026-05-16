# RAG 记忆模块 schema-loop 第二轮计划

## 1. 背景

第一轮 stale path guard 已完成最小验证、扩展验证和 50-task full benchmark 对照。

Full benchmark variant：

```text
artifacts/runs/20260516T061021Z
```

Baseline：

```text
artifacts/runs/20260515T162819Z
```

关键结论：

- stale path loop 已被有效压制。
- missing context asset 从 `195` 降到 `8`。
- `task_350` / `task_355` 从失败变为成功，score 均为 `1.0`。
- agent succeeded 仍为 `42/50`，未提升。
- 剩余瓶颈迁移到 SQL/schema loop、execute_python loop、list_context loop 和 answer forcing。

本轮只处理 SQL/schema loop，不扩大到其它 loop 类型。

## 2. 第二轮目标

本轮目标是让 agent 在遇到 SQLite schema mismatch 时，不再对同一个不存在的 table / column 原样重试几十次。

主要问题样本：

```text
task_408
```

`task_408` 在 full benchmark variant 中：

- `execute_context_sql` 调用 `56` 次。
- `no such table: races` 出现 `42` 次。
- `no such table: drivers` 出现 `3` 次。
- `inspect_sqlite_schema` 只调用 `1` 次。
- 失败原因仍为 `Agent did not submit an answer within max_steps.`

本轮验收重点不是立即保证 `task_408` 成功，而是先消除 deterministic SQL schema loop。

审查后补充：

- 本文档是第二轮 P0 的实施计划，不替代后续代码提交说明。
- 本轮必须保持第一轮 stale path guard 已验证成果，不回滚 `known_paths` / `memory.inject_dataset_facts=False` 相关行为。
- 本轮只允许引入可配置、默认关闭的行为改变；实验显式开启后再做 E2E 归因。

## 3. 非目标

本轮明确不处理：

- `execute_python` deterministic loop：`task_199` / `task_418`。
- repeated successful tool call / answer forcing：`task_283`。
- `list_context` loop：`task_352` / `task_396`。
- CSV 被误当 SQLite DB：`task_330`。
- 数据理解和文档精确匹配问题：`task_344` / `task_352`。
- dataset memory namespace 重构。
- RAG top-k、prompt budget、dynamic recall 策略调整。
- task-scoped dataset memory。

这些问题应在后续独立轮次处理，避免本轮归因混乱。

## 4. 现有代码入口

### 4.1 SQL 工具层

相关文件：

```text
src/data_agent_langchain/tools/sqlite.py
src/data_agent_langchain/tools/execute_context_sql.py
src/data_agent_langchain/tools/inspect_sqlite_schema.py
```

当前行为：

- `inspect_sqlite_schema()` 返回 tables + create_sql。
- `execute_read_only_sql()` 直接执行 SQL。
- `ExecuteContextSqlTool._run()` 捕获所有非 path validation 异常，统一返回：

```json
{
  "tool": "execute_context_sql",
  "error": "<sqlite error>"
}
```

并标记：

```text
error_kind="runtime"
```

这保留了错误信息，但没有结构化字段告诉上层：

- 错误是否为 schema mismatch。
- 缺失对象是 table 还是 column。
- 缺失对象名称是什么。
- 当前 DB 中实际有哪些 table。

### 4.2 Agent guard 层

相关文件：

```text
src/data_agent_langchain/agents/tool_node.py
```

第一轮 stale path guard 已在这里形成可复用模式：

- `tool_node()` 在真正调用工具前检查 action 和 action_input。
- `_known_path_failure_result()` 可在 pre-tool 阶段直接返回 `ToolRuntimeResult`。
- pre-tool block 会写入正常 `StepRecord`。
- `last_error_kind` 映射为 `tool_validation` 后，可触发 Plan-and-Solve replan。

第二轮 SQL/schema guard 可以沿用这个结构。

历史记录读取口径：

- 读取 `state["steps"]` 中的历史 `StepRecord`。
- 仅统计 `step.action == "execute_context_sql"` 且 `step.observation["content"]` 为 dict 的记录。
- 仅统计真实工具返回的结构化 schema mismatch：`content["sql_error_kind"] == "schema_mismatch"`。
- 不把 guard 自己返回的 `validation == "sql_schema_loop_guard"` 计入 schema mismatch 次数，避免 self-amplifying block。

### 4.3 路由层

相关文件：

```text
src/data_agent_langchain/agents/advance_node.py
```

当前行为：

- `last_tool_ok is False` 且 `last_error_kind in {tool_error, tool_timeout, tool_validation}` 时触发 `replan_required`。
- 因此 schema loop guard 如果返回 `error_kind="validation"`，会进入现有 replan 机制。
- 不需要为第二轮新增 graph edge。

## 5. 设计方案

采用：

```text
结构化 SQL schema error + pre-tool deterministic schema loop guard
```

不采用 prompt-only，也不只做工具错误增强。

### 5.1 SQL schema error 结构化

当 `execute_context_sql` 捕获 SQLite schema mismatch 时，返回结构化 content。

本轮只识别以下 SQLite error signature：

- `no such table: <identifier>`
- `no such column: <identifier>`

以下错误不归入本轮 schema mismatch：

- read-only violation，例如 `DELETE` / `INSERT` 被拒绝。
- file type mismatch，例如 `file is not a database`。
- SQL syntax error。
- lock / permission / timeout 等运行时错误。

目标形态：

```json
{
  "tool": "execute_context_sql",
  "error": "no such table: races",
  "sql_error_kind": "schema_mismatch",
  "missing_kind": "table",
  "missing_identifier": "races",
  "path": "db/results.db",
  "available_tables": ["results"]
}
```

对于 column mismatch：

```json
{
  "tool": "execute_context_sql",
  "error": "no such column: foo",
  "sql_error_kind": "schema_mismatch",
  "missing_kind": "column",
  "missing_identifier": "foo",
  "path": "db/results.db",
  "available_tables": ["results"]
}
```

保留：

```text
error_kind="runtime"
```

原因：第一次 schema mismatch 仍是工具运行时错误，不应被归为输入 shape validation。

实现边界：

- 结构化字段应作为现有 error payload 的增强，不改变 `ok=False` 与 `error_kind="runtime"`。
- `available_tables` 应尽量从同一个 SQLite 文件实时读取；读取失败时保持原 runtime error，不应二次覆盖原始错误。
- `path` 使用用户传入的相对 context path，不暴露宿主机绝对路径。

### 5.2 pre-tool deterministic schema loop guard

在 `tool_node()` 中新增 pre-tool guard，位置建议在 `_known_path_failure_result()` 之后、真实工具调用之前。

逻辑：

```text
if action == "execute_context_sql":
    从 state["steps"] 中回看历史 execute_context_sql 失败记录
    找出同 path 下的 schema_mismatch 记录
    如果当前 SQL 再次引用已重复失败的 missing table / column
    且同一 missing identifier 达到阈值
    则不调用真实工具，直接返回 validation failure
```

更精确的数据口径：

```text
current_path = normalize(action_input["path"])
current_sql = action_input["sql"]

for each prior step:
    prior_path = normalize(step.action_input["path"])
    content = step.observation["content"]
    if prior_path == current_path
       and content.sql_error_kind == "schema_mismatch"
       and content.missing_kind in {"table", "column"}:
           count by (path, missing_kind, casefold(missing_identifier))
```

触发条件：

- 当前 action 必须是 `execute_context_sql`。
- 当前 `path` 必须与历史 schema mismatch 的 path 相同。
- 历史同一 `(path, missing_kind, missing_identifier)` 的真实 schema mismatch 次数达到阈值。
- 当前 SQL 仍引用该 missing identifier。
- 配置值为 `0` 或 `None` 时不触发。

初始阈值建议：

```text
agent.sql_schema_mismatch_retry_limit = 2  # 实验显式开启值
```

含义：

- 允许模型第一次犯错。
- 允许第二次尝试修正。
- 第三次仍引用同一不存在对象时 hard block。

注意：该阈值表示“真实工具 schema mismatch 的允许次数”，不包含 pre-tool guard 返回的 validation 记录。

### 5.3 Guard 返回内容

目标形态：

```json
{
  "tool": "execute_context_sql",
  "error": "Repeated SQL schema mismatch: table 'races' is not available in db/results.db. Do not retry SQL referencing this identifier. Use inspect_sqlite_schema results and query only available tables.",
  "validation": "sql_schema_loop_guard",
  "path": "db/results.db",
  "missing_kind": "table",
  "missing_identifier": "races",
  "available_tables": ["results"],
  "retry_limit": 2
}
```

返回：

```text
ok=False
error_kind="validation"
```

这样可复用 `advance_node` 的 replan 逻辑。

## 6. SQL 引用识别策略

本轮只做轻量规则，不引入完整 SQL parser。

### 6.1 table 引用

针对 `no such table: races`：

- 将 SQL 归一化为小写。
- 使用 token boundary / 简单正则检查是否包含 table identifier，避免 `race` 误伤 `races_archive`。
- 至少覆盖：
  - `FROM races`
  - `JOIN races`
  - `UPDATE races` 不应出现，因为只读 SQL 已拒绝写操作。
  - `FROM races r`
  - `JOIN races AS r`
  - `FROM "races"`
  - `JOIN [races]`

本项目只允许 read-only SQL，因此核心覆盖 `FROM` / `JOIN` 即可。

不要求覆盖：

- CTE 名称与真实 table 同名的复杂场景。
- 注释 / 字符串字面量中的 identifier 消歧。
- 跨数据库 attach 语法。

### 6.2 column 引用

针对 `no such column: foo`：

- 检查 SQL 中是否仍出现该 column token。
- 只做 token boundary 匹配，避免 `id` 误伤 `driverId`。
- 若错误返回 `table.column` 形态，优先按完整 token 检测；必要时再按最后一段 column 名检测。

### 6.3 大小写

SQLite identifier 大多数场景大小写不敏感，本轮按 case-insensitive 处理。

### 6.4 不解析 alias

本轮不做完整 alias 解析。

如果 `no such column: r.year` 或 `no such column: year` 出现，只按错误中的 identifier 做 token 检测。

## 7. 配置建议

新增 agent config：

```text
agent.sql_schema_mismatch_retry_limit: int | None = 0
```

默认建议：

```text
0
```

实验显式开启建议：

```text
2
```

关闭方式建议：

```text
0 或 None 表示禁用
```

是否默认开启：

- 若追求 parity 保守，可默认关闭，在实验 YAML 中开启。
- 若延续第一轮 guard 的验证路线，可默认关闭，并在 `configs/local.yaml` 实验配置中显式开启。

推荐：默认关闭，实验显式开启。

原因：

- 这是行为改变，先作为第二轮实验 guard 验证。
- 避免影响历史 parity 预期。
- `configs/local.yaml` 属于本地实验配置时，不应把其中的私有路径或密钥作为本轮提交内容；如需可复现实验，应另写临时配置或在运行记录中明确覆盖项。

## 8. TDD 实施拆解

### 8.1 工具层测试

新增或扩展测试文件：

```text
tests/test_phase2_tools_functional.py
```

测试点：

1. missing table 返回结构化字段：
   - `sql_error_kind == "schema_mismatch"`
   - `missing_kind == "table"`
   - `missing_identifier == "missing_table"`
   - `available_tables` 包含真实表。
   - `error_kind == "runtime"` 保持不变。

2. missing column 返回结构化字段：
   - `missing_kind == "column"`
   - `missing_identifier` 正确。

3. 非 schema SQLite 错误仍保持普通 runtime error。

推荐红测改造：

- 扩展现有 `test_execute_context_sql_schema_mismatch_remains_runtime_error`，先让它断言结构化字段并失败。
- 新增 missing column 测试，例如 `SELECT missing_col FROM players`。
- 保留 `test_execute_context_sql_rejects_writes`，证明 read-only violation 不被标记为 `schema_mismatch`。

### 8.2 guard 层测试

新增或扩展测试文件：

```text
tests/test_phase3_tool_node.py
```

测试点：

1. 历史已有两次同 path + `no such table: races`，第三次 SQL 仍 `FROM races`：
   - 不调用真实工具。
   - 返回 `tool_validation`。
   - observation 包含 `sql_schema_loop_guard`。

2. 不同 path 不拦截。

3. 不同 missing table 不拦截。

4. 当前 SQL 不再引用 missing identifier 不拦截。

5. retry limit 未达到不拦截。

6. 配置关闭时不拦截。

实现测试建议：

- 构造历史 `StepRecord` 时直接使用真实 schema mismatch payload shape，避免依赖 LLM。
- monkeypatch `tool_node_module.call_tool_with_timeout`，在应被 guard block 的用例中记录调用次数或直接抛错，证明真实工具未执行。
- 在不应拦截的用例中允许工具正常执行，或 monkeypatch 返回一个 sentinel success，证明 guard 没有误触发。
- 断言返回的 `step.observation["content"]["validation"] == "sql_schema_loop_guard"`。

### 8.3 配置测试

新增或扩展测试文件：

```text
tests/test_phase5_config.py
```

配置测试中确认：

- `AgentConfig` 有 `sql_schema_mismatch_retry_limit` 字段。
- YAML 能覆盖该字段。
- 默认值符合预期。
- `AppConfig.to_dict()` / `AppConfig.from_dict()` round-trip 保留该字段。

不建议把该测试放到 CLI memory mode 文件中，因为这是通用 agent config，不属于 memory CLI override 行为。

## 9. E2E 验证计划

### 9.1 主验证

运行：

```powershell
dabench-lc run-task task_408 --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag
```

前提：本地配置显式开启：

```text
agent.enforce_known_path_only=True
agent.sql_schema_mismatch_retry_limit=2
memory.inject_dataset_facts=False
```

若不希望修改本地 `configs/local.yaml`，可复制一个临时实验 YAML，或在运行命令前确认当前 local config 已包含上述覆盖项。

记录：

- 是否成功。
- `execute_context_sql` 总次数。
- `no such table: races` 次数。
- `no such table: drivers` 次数。
- 是否触发 `sql_schema_loop_guard`。
- 是否进入 replan。
- prompt tokens / total tokens。
- artifact run id。
- 对比基准 run id。

### 9.2 回归验证

至少运行：

```powershell
dabench-lc run-task task_350 --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag

dabench-lc run-task task_355 --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag
```

目标：

- stale path guard 成果不回退。
- 两个任务仍成功。

### 9.3 观察样本

可选运行：

```text
task_199
task_257
task_259
task_261
task_287
task_420
```

这些样本在 full benchmark 中出现过少量 schema mismatch，但不是本轮主样本。

## 10. 验收标准

### 10.1 最小验收

满足以下条件即可认为第二轮 SQL/schema guard 最小闭环通过：

- 单元测试覆盖 schema error 结构化。
- 单元测试覆盖 schema loop pre-tool guard。
- 配置测试覆盖默认关闭、YAML 覆盖和 round-trip。
- `task_408` 中 `no such table: races` 从 `42` 降到不超过 `3`。
- `task_408` 中 `execute_context_sql` 不再接近 `56` 次原样循环。
- `task_350` / `task_355` 不回退。
- 默认关闭时不改变 baseline/parity 行为，至少由配置单测证明默认值为 disabled。

### 10.2 理想验收

如果模型能利用 schema observation 转向正确查询，则进一步目标是：

- `task_408` 成功提交答案。
- `task_408` score 从 `0.0` 提升。

但这不是最小验收条件。

## 11. 风险与回滚

### 11.1 误伤风险

风险：SQL token 检测过粗，误判当前 SQL 仍引用 missing identifier。

缓解：

- 仅在历史重复次数达到阈值后触发。
- 仅对 `execute_context_sql` 生效。
- table 检测优先限制在 `FROM` / `JOIN`。
- 配置默认关闭，实验显式开启。

### 11.2 阈值 off-by-one 风险

风险：`retry_limit=2` 被实现成第二次就拦截，而不是第三次拦截。

缓解：

- 测试必须明确“已有 1 次不拦截、已有 2 次才拦截下一次”。
- 文档口径固定为“允许的真实 schema mismatch 次数”。

### 11.3 Replan 预算耗尽风险

风险：guard 返回 `tool_validation` 后触发 replan，但 replan 预算不足时仍失败。

缓解：

- 验收重点先看 loop 是否被截断。
- 后续可再设计 answer forcing 或 fallback synthesis。

### 11.4 available_tables 获取风险

风险：为了补充 `available_tables`，schema introspection 自身失败，掩盖原始 SQL 错误。

缓解：

- 原始 schema mismatch error 是主错误。
- `available_tables` 是 best-effort 增强；获取失败时可省略或置空，但不能改变原 `error`。

### 11.5 task_408 不成功风险

风险：即使 schema loop 被截断，模型仍无法用 `results` 表和 docs 推理出答案。

缓解：

- 不把 `task_408` 成功作为最小验收。
- 若 loop 已被截断但任务仍失败，下一轮再判断是否需要 SQL result/doc synthesis prompt。

### 11.6 回滚策略

- 将 `agent.sql_schema_mismatch_retry_limit` 设为 `0` 或 `None`。
- 保留结构化 SQL error 字段通常低风险；如影响 prompt 分布，也可只在配置开启时附加 enhanced content。

## 12. 推荐实施顺序

1. 写工具层红测：missing table / missing column 结构化字段。
2. 实现 SQL schema mismatch 解析与 available tables 附加。
3. 写 `tool_node` guard 红测。
4. 实现 `_sql_schema_loop_failure_result()`。
5. 实现 SQL identifier 轻量检测辅助函数。
6. 写非误伤测试。
7. 加配置字段和 `tests/test_phase5_config.py` 配置加载测试。
8. 跑相关单元测试。
9. 跑 `task_408` E2E。
10. 跑 `task_350` / `task_355` 回归。
11. 将结果追加到本文档。

## 13. 当前决策

- 第二轮范围：只做 SQL/schema guard。
- 主样本：`task_408`。
- 推荐实现：结构化 SQL schema error + pre-tool deterministic schema loop guard。
- 默认策略：`AgentConfig.sql_schema_mismatch_retry_limit=0`，实验显式设为 `2`。
- 下一步：按第 12 节推荐顺序进入 TDD 实施，并在第 14 节追加验证结果。

## 14. 实施结果记录模板

实施完成后在本节追加实际结果。

### 14.1 单元测试

```text
pytest tests/test_phase2_tools_functional.py tests/test_phase3_tool_node.py tests/test_phase5_config.py -q
```

结果：

```text
worktree: .worktrees/rag-schema-loop-guard
branch: rag-schema-loop-guard
commits:
- 1d394ee feat(sql): structure schema mismatch errors
- 88ce86b feat(agent): guard repeated SQL schema mismatches
- 57a6636 feat(config): add SQL schema retry limit

python -m pytest tests/test_phase2_tools_functional.py tests/test_phase3_tool_node.py tests/test_phase5_config.py -q
56 passed in 1.73s
```

### 14.2 task_408 主验证

```text
run id: 20260516T073623Z
trace_succeeded: true
metrics_succeeded: true
agent_succeeded: true
score: run-task metrics 未输出 score 字段
execute_context_sql calls: 8
no such table: races count: 1
no such table: drivers count: 0
sql_schema_loop_guard count: 0
inspect_sqlite_schema calls: 1
prompt tokens: 101049
total tokens: 105960
```

### 14.3 回归样本

```text
task_350 run id: 20260516T074142Z
task_350 succeeded/score: succeeded=true, run-task metrics 未输出 score 字段
task_350 tool_calls: list_context=1, read_doc=2, inspect_sqlite_schema=1, read_csv=1, execute_context_sql=3, execute_python=1, answer=1
task_350 tokens: prompt=82191, total=85841

task_355 run id: 20260516T074458Z
task_355 succeeded/score: succeeded=true, run-task metrics 未输出 score 字段
task_355 tool_calls: list_context=2, read_doc=2, read_csv=1, execute_context_sql=1, execute_python=7, answer=1
task_355 tokens: prompt=121183, total=126078
```

### 14.4 结论

```text
达到最小验收。

单元测试覆盖已完成，配置默认关闭由 tests/test_phase5_config.py 覆盖。
task_408 在显式开启 agent.sql_schema_mismatch_retry_limit=2 的临时配置下成功：
execute_context_sql 从 full benchmark variant 的 56 降到 8；
no such table: races 从 42 降到 1；
no such table: drivers 从 3 降到 0。

本次 task_408 未触发 sql_schema_loop_guard，因为结构化 schema mismatch error 后模型已转向 inspect_schema/docs/results 表路径；guard 单元测试覆盖了重复失败时第三次拦截。

task_350 / task_355 回归样本均成功，未观察到第一轮 stale path guard 回退。
```
