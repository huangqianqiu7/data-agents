# 计划归档索引

本文档整理已完成的 RAG / memory 相关计划，避免已关闭主题继续留在 `Doing/` 中。

## 2026-05-16 RAG 记忆模块第一轮

目录：

```text
src/计划/归档/2026-05-16-RAG记忆模块第一轮/
```

文件：

- `2026-05-16-RAG记忆模块工具提示词调整计划.md`
- `2026-05-16-RAG记忆模块stale-path第一轮进展记录.md`

状态：已完成并合并到 `main`。第一轮关闭 stale path guard 主题，验证了 `memory.inject_dataset_facts=False`、memory/RAG authority policy、`known_paths` hard block 与 missing context asset validation 分类。

关键结果：

- Full benchmark variant：`artifacts/runs/20260516T061021Z`
- `task_350` / `task_355` 从失败转为成功。
- missing context asset 从 `195` 降到 `8`。
- stale path guard 主题关闭，不回滚当前实现。

## 2026-05-16 RAG 记忆模块第二轮

目录：

```text
src/计划/归档/2026-05-16-RAG记忆模块第二轮/
```

文件：

- `2026-05-16-RAG记忆模块schema-loop第二轮计划.md`

状态：已完成并合并到 `main`。第二轮关闭 SQL/schema loop guard 最小闭环，默认关闭配置字段为 `AgentConfig.sql_schema_mismatch_retry_limit=0`，实验开启值为 `2`。

关键结果：

- 实现提交：`1d394ee` / `88ce86b` / `57a6636`
- 单元测试：`tests/test_phase2_tools_functional.py tests/test_phase3_tool_node.py tests/test_phase5_config.py` 共 `56 passed`
- 合并后完整测试：`620 passed, 3 skipped, 49 warnings`
- `task_408` run：`20260516T073623Z`
- `task_408` 成功，`execute_context_sql=8`，`no such table: races=1`，`no such table: drivers=0`
- `task_350` / `task_355` 回归样本均成功

## 仍在 Doing 的文档

```text
src/计划/Doing/2026-05-16-RAG记忆模块剩余问题路线图.md
```

该路线图继续作为 P1/P2/P3 剩余问题索引，不代表单个已完成轮次。
