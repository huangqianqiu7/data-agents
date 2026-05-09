# 统一配置（src/计划/统一配置/）

本子目录登记“**本地与容器运行参数统一**”主题的方案文档，自 2026-05-09 起。

## 主题边界

本目录内的方案约束 `data_agent_langchain` 的运行参数来源：

- **唯一权威**：`src/data_agent_langchain/config.py` 的 dataclass 默认值
  （`AgentConfig` / `ToolsConfig` / `RunConfig` / `ObservabilityConfig` /
  `EvaluationConfig`）。
- **本地（`load_app_config`）与容器（`build_submission_config`）仅有的合法差异**：
  1. 三个路径：`dataset.root_path` / `run.output_dir` /
     `observability.gateway_caps_path`（容器固定为 `/input` /
     `/tmp/dabench-runs` / `/app/gateway_caps.yaml`）。
  2. LLM 身份：`agent.api_base` / `agent.api_key`，统一通过 `MODEL_API_URL` /
     `MODEL_API_KEY` 环境变量注入；本地与容器复用 `_overlay_agent_env_vars`
     协议（详见 `../完善计划/2026-05-08-env-var-fallback-design.md`）。

不属于上述差异的字段（`max_steps` / `max_replans` / `max_workers` /
`task_timeout_seconds` / `action_mode` / `model_timeout_s` /
`max_obs_chars` / …）禁止在 `submission.py` 内重写，必须由 dataclass
默认决定。

## 命名规则

仿 `src/计划/完善计划/` 现有风格：

- 长期方案：`<日期>-<主题>-design.md` / `<日期>-<主题>-plan.md`
- 临时备忘：`<日期>-<主题>-notes.md`
- 审查结论：`<原文件名>-审查结论.md`

日期使用 `YYYY-MM-DD`；主题用中文短语或 kebab-case 英文。

## 与现有文档的关系

- **上游 / 姊妹**：`../完善计划/正式提交完善计划-v5.md` §二.4 / §二.8
  （本目录方案修订其中关于 `DEFAULT_MAX_WORKERS` /
  `DEFAULT_TASK_TIMEOUT_SECONDS` / `DEFAULT_MODEL_NAME` 的描述）。
- **上游 / 姊妹**：`../完善计划/2026-05-08-env-var-fallback-design.md`
  （MODEL_* env 协议的延续；本目录在其基础上把“非 LLM 身份字段”也
  收敛到同一权威）。
- **用户级实现计划**（不在仓库内，仅供 Cascade / 维护者参考）：
  `C:\Users\18155\.windsurf\plans\unify-config-defaults-7b3fa6.md`。

## 维护约束

1. 文档内严禁出现真实 `api_key` / `api_base` URL / 内部模型服务地址；
   示例统一用占位符（如 `https://example.com/v1`、`sk-xxx`）。
2. 任何对 `config.py` / `submission.py` 配置行为的代码改动，必须同步
   更新本目录最新一份 `*-design.md`，否则视为弱化文档。
3. 测试改动（`tests/test_submission_config.py` / `test_submission_runtime.py`
   / `test_submission_meta.py` / `test_phase5_config.py`）应在设计文档
   §5（测试影响）中逐条登记“等价替换”，禁止静默删除断言。
4. 容器化资产（`Dockerfile` / `docker/gateway_caps.yaml` / `.dockerignore`）
   不在本主题范围；如需调整，另开主题目录。

## 当前文件清单

- `2026-05-09-统一配置参数-design.md` —— v1 设计文档：把
  `submission.py` 中 `DEFAULT_MAX_WORKERS` / `DEFAULT_TASK_TIMEOUT_SECONDS` /
  `DEFAULT_MODEL_NAME` / `DABENCH_*` env 删除，让本地与容器共用同一份
  dataclass 默认值。
- `2026-05-09-统一配置参数-design-审查结论.md` —— v1 设计的事实复核
  与设计取舍审查（F1-F5 / O1-O7 / D1-D3 / S1）。
- `2026-05-09-统一配置参数-design-v2.md` —— v2 设计文档：吸收
  v1 审查结论。主备方案对调（保持 `AgentConfig.model =
  "gpt-4.1-mini"`，`MODEL_NAME` 在容器路径升级为必填 env），订正 §5
  测试影响表的事实错误，补全字段级等价断言与 TDD 执行顺序。
- `2026-05-09-统一配置参数-design-v2-审查结论.md` —— v2 设计的
  Conditional Approve 审查（M1-M5 微调建议）。
- `2026-05-09-统一配置参数-design-v3.md` —— v3 设计文档：吸收
  v2 审查 M1-M5 的精修。主要决策（D1 / V6 / V7）一字未动；仅修正
  setenv 数字、test 数字、测试名精度、V5/V6 setup 段补
  `delenv("MODEL_API_KEY")`、V7 测试骨架补 `_run_single_task_impl`
  monkeypatch；新增 §9 Step 0 baseline GREEN 校验与 Step 6 commit
  message 范例。
- `2026-05-09-统一配置参数-design-v3-审查结论.md` —— v3 设计的
  Conditional Approve 审查（阻塞项 N1：V7 `_fake_impl` 字段名错；
  N2 / N3 文档精度）。
- `2026-05-09-统一配置参数-design-v4.md` —— **当前最新方案 / 实施
  版**：修复 v3 阻塞项 N1（V7 `_fake_impl` 改用真实
  `TaskRunArtifacts` 字段 `task_output_dir=` / `trace_path=` /
  `failure_reason=None`），消化 N2（README §配置 LLM 凭据章节
  改"追加" D1 副作用而非"移除" `DABENCH_*`）/ N3（统一 V5a / V5b
  记号），新增 §9 Step 1 预期 RED 函数清单与 §4.2 docstring snippet
  范例。主要决策（D1 / V6 / V7）一字未动。
