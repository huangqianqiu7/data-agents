# v5 执行笔记（2026-05-08）

按 `正式提交完善计划-v5.md` 完成 §四第 1-6 步本地实现部分。所有改动 in-place，未使用 git / worktree。

## 验证证据

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 完整 pytest | `pytest tests/ -q` | 184 passed in 4.40s |
| RED→GREEN | TDD 22 新测试两阶段确认 | 21 RED → 全 GREEN |
| CLI smoke | `python -m data_agent_langchain.cli --help` | 3 子命令正常 |
| 模块 smoke | `import data_agent_langchain.submission` | 加载 + 常量正确 |
| pyproject 校验 | `tomllib` 解析 | name=`data-agent-langchain`、9 主依赖、仅 `dabench-lc`、hatch packages 显式 |

测试基线：Phase 8 完成时 162 passed → v5 完成时 184 passed（+22 新增）。

## 新增文件（7）

- `tests/test_submission_config.py`（115 行）—— 8 条 AppConfig 构造测试。
- `tests/test_submission_runtime.py`（310 行）—— 10 条 runtime 测试（gateway/output 6 + signal 2 + redaction 2）。
- `tests/test_submission_meta.py`（90 行）—— 4 条元测试（pyproject 2 + dataset schema 2）。
- `src/data_agent_langchain/submission.py`（440 行）—— 提交入口；13 步流程、SIGTERM 主线程协作、per-task try/except、占位 CSV、同目录原子覆盖、脱敏 logging、submission_summary.json schema。
- `Dockerfile`（37 行）—— base image `python:3.13-slim-bookworm`，`pip install --no-cache-dir .`，ENTRYPOINT `python -m data_agent_langchain.submission`。
- `.dockerignore`（58 行）—— 完整排除 `.env` / `configs/` / `data/` / `tests/` / `src/废弃/` / `src/完善计划/` / `artifacts/` / `.git` / 缓存与归档。
- `docker/gateway_caps.yaml`（22 行）—— `tool_calling=true` 字面值，提交态固定入镜像。

## 修改文件（3）

- `pyproject.toml` —— `name` 改 `data-agent-langchain`（v5 implicit gap 修复），主依赖减法 + 版本范围（langchain-core / langchain-openai / langgraph / openai / pandas / pydantic / pyyaml / typer / numpy），baseline 大依赖下放到 `[project.optional-dependencies].baseline`，删除 `dabench` entry point，新增 `[tool.hatch.build.targets.wheel/sdist]` 显式配置。
- `src/data_agent_langchain/benchmark/dataset.py`（`_load_task_record`）—— schema 从「键集严格相等」放宽为「必含 task_id/difficulty/question 三键，多余字段忽略」，对官方未来字段扩展鲁棒。
- `.gitignore`（line 28）—— 注释掉 `# tests/`，让 Cascade 工具能读现有 conftest + 测试，TDD 用。

## v5 计划符合性

P0 全部完成：

- §二.1 Dockerfile + ENTRYPOINT
- §二.2 `docker/gateway_caps.yaml`
- §二.3 `/output` 扁平化 + `result\r\n` 占位 + 同目录原子覆盖 + EXDEV 防护
- §二.4 `MODEL_*` 注入 + 不读 YAML + `graph_mode="plan_solve"`
- §二.5 per-task try/except + SIGTERM 主线程 + `_shutting_down`
- §二.6 `/logs` 脱敏 + summary schema
- §二.7 `.dockerignore` 完整清单
- §二.8 主依赖减法 + 版本固化 + entry point 清理（含 implicit name + hatch 修复）
- §二.9 LangSmith 禁用 + env scrub

P1 完成：

- §三.1 `task.json` schema 放宽
- §三.3 `None` 写 CSV 为空 cell（测试覆盖）
- §三.4 密钥泄漏防回归测试
- §四第 1 步 22 条测试先行
- §四第 0/2/3/4/5/6 步实现 + 本地验证

## 未做项

| 事项 | 状态 | 理由 |
| --- | --- | --- |
| §三.5 `gateway-smoke` CLI 末尾文案修正 | ✅ 已做（2026-05-08） | `cli.py:75-81` 改成 capability-aware 文案：网关支持 tool_calling 时显示 "default action_mode: tool_calling (gateway supports it)"，否则提示 fallback 到 json_action。 |
| 容器 `docker build` / `docker run` smoke | 未做 | 需 Docker daemon；Windows + WSL/Docker Desktop 由队长手动跑。 |
| `docker save` 归档 | 未做 | 同上，依赖容器 smoke。 |
| `uv.lock` 重新生成 | 未做 | v5 §二.8.3 标"可选/推荐"。如要锁版本：dev 环境 `uv lock`。 |

## 2026-05-08 增量（开发态 env-var 兜底）

详见 `2026-05-08-env-var-fallback-design.md`。

- 新增 `config._overlay_agent_env_vars` —— `load_app_config` 读 YAML 后对
  `agent.{model, api_base, api_key}` 做 env 兜底，与 `submission.py` 共享同一份
  `MODEL_NAME` / `MODEL_API_URL` / `MODEL_API_KEY` 协议。YAML 非空赢；既无 YAML
  也无 env 时回退到 dataclass 默认。
- 新增 4 条测试 `tests/test_phase5_config.py:108-198`（184 → 188 passed）。
- README.md `快速上手` §2 重写为 env-var 协议说明，原 3、4 顺延为 4、5。
- 配套：本机 `.env` 用的是 baseline 旧名（`LLM_API_KEY` / `LLM_BASE_URL` /
  `LLM_MODEL_ID`），不是 KDD Cup 新名（`MODEL_*`）。临时方案是 PowerShell session
  内做别名映射 (`$env:MODEL_API_URL = $env:LLM_BASE_URL` 等)；长期建议 `.env` 末尾
  追加 3 行 `MODEL_*` 别名共存。
- 已用 `configs/local.yaml`（无 secret，gitignored）跑通 `gateway-smoke`（4/4
  caps True）+ `run-task task_11`（succeeded=true，wall_clock=15.5s）。

## 2026-05-08 增量 #3（进度条 UX 优化：日志集成 + 紧凑列）

针对增量 #2 实跑 50 题暴露的两个 UX 瑕疵：
1. `agents/model_retry.py` 的 `logger.warning("[model_node] step ... attempt ... failed")` 走默认 stderr handler，与 rich.Progress 的 stderr 原地刷新冲突，造成进度条被打断重画。
2. 终端窄于 ~120 列时 `elapsed / eta` 列被 rich 自动 `…` 截断，UX 不佳。

修复：
- `cli.py` 新增 `_build_progress_columns(compact: bool = True)`，compact 模式去 `TimeElapsedColumn / TimeRemainingColumn`。CLI 默认 compact。
- `cli.py` 新增 `_install_rich_logging(console)` 上下文管理器：进入时给根 logger 装 `RichHandler` 并保存原 handlers；退出时还原。`_run_benchmark_with_progress_bar` 把 `Progress` 包在 `_install_rich_logging` 之内，rich Live 渲染会自动把 `logger.warning` 渲染在 bar 上方、bar 在下方重画，解决冲突。
- 仅影响 dev CLI；容器 `submission.py` 的 `_setup_redacted_logging` 写文件不受影响。
- 新增 5 条测试 `tests/test_phase8_cli_progress.py:287-371`：column composition compact/full/默认 + handler 安装与还原 + 日志路由 ANSI 检测。
- 验证：`pytest tests/` 251 → **256 passed**（5 new）；smoke `--limit 3` 完成 3/3 succeeded。

## 2026-05-08 增量 #4（进度条噪声抑制 + tool_calling prompt 修复）

背景：增量 #3 把 `model_retry` 的 WARNING 整齐渲染在 bar 上方，但 429 / timeout
高频重试（react 50 题单次实跑 ~数十条）仍会刷屏，淹没 `last / ok / fail` 关键字段。
用户明确要求"进度条后面不显示这些 429 重试日志"。

修复 A（CLI 端噪声抑制）：
- `cli.py` 新增 module 常量 `_NOISY_LOGGERS_DURING_PROGRESS: tuple[str, ...] =
  ("data_agent_langchain.agents.model_retry",)`。
- `_install_rich_logging` 进入时保存 noisy logger 原 level → 临时拉到 `ERROR`
  （屏蔽 WARNING / INFO），退出时精确还原；ERROR / CRITICAL 仍透过 RichHandler
  显示（fatal 不被吞）。
- 仅影响 CLI 进度条期间；容器 `submission.py` 路径不触达 `_install_rich_logging`，
  retry warning 仍会被 `_setup_redacted_logging` 写入 `logs_dir`，供赛后复盘。

修复 B（顺带）：`tool_calling` + `plan_solve` 模式下 system_text 仍含 \`\`\`json 残留。
根因：用户上次手改 `prompt_strings.py::PLAN_AND_SOLVE_SYSTEM_PROMPT` rule 5
引入了新写法 ("Do not place the JSON object ... must not start with anything
other than \`\`\`json ...")，而 `prompts.py::_strip_json_action_rules` 的
`blocked` 元组 token 集合不含该新措辞的 substring，导致该行未被剥除。
修法：在 `blocked` 加 `"```json"` 字面量 token —— 任何提到 \`\`\`json 字面量的
规则行都属于 fenced JSON 输出约束，tool_calling 模式无需。单行改动，最小上游修复。

- 新增 3 条 test `tests/test_phase8_cli_progress.py:379-446`：
  - `suppresses_model_retry_warnings` —— WARNING 级输出 console 应不含 `model_node` / `429`
  - `still_shows_model_retry_errors` —— ERROR 级仍透出（防回归）
  - `restores_suppressed_logger_level` —— 退出上下文后 level 还原
- 回归：`pytest tests/` 256 → **262 passed**（3 noisy-logger + 3 顺带修好的旧 flaky）。
- 验证 smoke：`run-benchmark --graph-mode react --limit 5 --progress` 进度条
  运行期间确认无 `[model_node] ... 429 ... retrying` 输出，仅保留 bar + last/ok/fail。

## 2026-05-08 增量 #2（CLI 进度条移植）

详见 baseline 同名实现 (`src/废弃/data_agent_baseline/cli.py:156-257`)。

- `cli.py` 新增 3 个 helper（`_format_compact_rate` / `_format_last_task` /
  `_build_compact_progress_fields`），与 baseline 字段格式对齐。
- `run_benchmark_command` 增 `--progress/--no-progress` 开关（默认 `--progress`）；
  rich 路径走 `_run_benchmark_with_progress_bar`（baseline 风格 16 列），rich
  缺失自动走 `_run_benchmark_with_text_progress`（typer.echo 逐题）。
- 容器路径不受影响：提交容器 `ENTRYPOINT` 是 `python -m
  data_agent_langchain.submission`，不进 CLI，不依赖 rich。
- 新增 9 条测试 `tests/test_phase8_cli_progress.py`：
  - 6 条覆盖纯函数 `_build_compact_progress_fields`（字段集 / running-queue 计算 /
    rate / last_artifact ok|fail）。
  - 3 条 CLI 集成（CliRunner mock `run_benchmark`）：默认开启 progress_callback /
    `--no-progress` 不渲染 ANSI / 计数累计正确。
- 验证：`pytest tests/` 251 passed；手动 smoke `--limit 2` 在 `--progress` 看到
  rich 渲染（输出含 `<truncated 1 lines>` 即 ANSI 序列堆叠），`--no-progress` 输出
  仅最终 `run/tasks/succeeded` 三行。

## 已知小遗留

- `.gitignore` 第 28 行 `# tests/` 是注释状态。如果不再需要 Cascade 读 tests，可手动改回 `tests/` 恢复屏蔽。
- v5 §二.5 SIGTERM 注册依赖 Linux 容器；Windows dev 上 `signal.signal(SIGTERM, ...)` 能 register 但不会被真触发，测试用 `register_signals=False` + 直接拨 `submission._on_sigterm` 验证停止派发逻辑。
- 提交镜像不再装 baseline 大包（datasets / duckdb / huggingface-hub / openpyxl / polars / pyarrow / rich）。本地若要跑 parity / 历史回归测试，需 `pip install ".[baseline]"`。

## 下一步（队长手动）

1. **容器 smoke**：

```text
docker build --platform=linux/amd64 -t <team_id>:v1 .
docker run --rm \
  -v <sample_tasks>:/input:ro \
  -v <local_output>:/output:rw \
  -v <local_logs>:/logs:rw \
  -e MODEL_API_URL=<内部模型服务地址> \
  -e MODEL_API_KEY=<密钥或 EMPTY> \
  -e MODEL_NAME=qwen3.5-35b-a3b \
  <team_id>:v1
```

2. **导出归档**：

```text
docker save <team_id>:v1 | gzip > <team_id>_v1.tar.gz
```

3. **检查清单**：参考 v5 §五完整跑一遍，重点核对镜像大小（< 2 GB 建议、< 10 GB 必须）与 `.dockerignore` 是否漏项。

4. （可选）**清理**：v5 §三.5 P2 文案修正、`.gitignore` 复原 `tests/`、`uv.lock` 重新生成。
