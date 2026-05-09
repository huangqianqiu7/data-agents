# 开发态 env-var 兜底设计（2026-05-08）

> 本文档是一份独立的设计 spec，配合 `正式提交完善计划-v5.md` §三 / §五的提交态
> env-var 协议（`MODEL_API_URL` / `MODEL_API_KEY` / `MODEL_NAME`），把同一种协议
> 下沉到开发态 `load_app_config` 路径，让 `data_agent_langchain` 的开发与提交两条
> 入口走 **同一份代码同一种 env-var 协议**，对齐 `rules.zh.md` §5.1 / §5.2 / §5.4
> "禁止在配置文件 / Docker 镜像中硬编码 LLM 服务 URL / API Key" 的硬约束。

## 一、动机

### 1.1 当前现状

| 入口 | 模型配置来源 | 是否符合规则 |
| --- | --- | --- |
| `submission.py`（容器 ENTRYPOINT，提交态） | 仅读 `MODEL_API_URL` / `MODEL_API_KEY` / `MODEL_NAME` env vars | ✅ 完全符合 |
| `load_app_config(yaml_path)` + `build_chat_model`（CLI 入口，开发态） | 直接从 YAML `agent.model` / `api_base` / `api_key` 字段读 | ❌ 违反 §5.2 "严禁在配置文件中硬编码" |

### 1.2 目标

让 `load_app_config` 也支持 env-var 兜底，开发态可以**完全不在 YAML 中写 secret**：

```yaml
# configs/local.yaml — 不含任何 secret
dataset:
  root_path: data/public/input
agent:
  action_mode: tool_calling
  progress: true
  # model / api_base / api_key 全部留空，从 env vars 兜底
run:
  output_dir: artifacts/runs
  max_workers: 1
observability:
  langsmith_enabled: false
  gateway_caps_path: artifacts/gateway_caps.yaml
```

环境变量从开发者本地 `.env`（gitignored）注入。

## 二、行为契约

### 2.1 优先级（YAML 非空赢，纯 env overlay）

```text
agent.model    = YAML.agent.model    if non_empty else env.MODEL_NAME    if set else AgentConfig 默认 ("gpt-4.1-mini")
agent.api_base = YAML.agent.api_base if non_empty else env.MODEL_API_URL if set else AgentConfig 默认 ("https://api.openai.com/v1")
agent.api_key  = YAML.agent.api_key  if non_empty else env.MODEL_API_KEY if set else AgentConfig 默认 ("")
```

**"非空"判定**：YAML 字段未出现 OR 值为空字符串 OR 值为 None。

**为什么是 YAML 非空赢，不是 env 赢**：

- 开发者偶尔需要本地 mock gateway 或离线 fake-model 调试 —— YAML 显式写值时应能覆盖
  全局 `MODEL_API_URL`，避免要求开发者 `Remove-Item env:MODEL_API_URL` 再调试
- `rules.zh.md` §5.4 推荐模式只要求"代码必须能从 env 读"，未强制"必须以 env 为唯一源"
- 现有 4-5 条测试在 YAML 里写了 `api_key="yaml-key"` 等显式值，YAML 非空赢可保零回归

### 2.2 既无 YAML 也无 env：用 dataclass 默认值，不抛错

发现现有测试 `test_load_app_config_from_yaml_resolves_relative_paths` /
`test_load_app_config_can_explicitly_keep_json_action`（`tests/test_phase5_config.py`）
的 YAML 都不写 `api_base` / `api_key`，且测试运行时 env vars 不保证已设。

为了保证**零现有测试改动**，本设计采用纯 env overlay 语义：

| YAML 字段 | env 字段 | 最终值 |
| --- | --- | --- |
| 非空 | 任意 | YAML 值 |
| 空 / 缺失 | 设置 | env 值 |
| 空 / 缺失 | 未设 | `AgentConfig` dataclass 默认值（不变） |

`AgentConfig` 当前 dataclass 默认（不动）：
- `model = "gpt-4.1-mini"`
- `api_base = "https://api.openai.com/v1"`
- `api_key = ""`

dev 若三者都不配置，会拿到这些默认值，到 LLM call 时会报 401 / DNS 错误 ——
这是清晰的"配置不全"信号，开发者可以读 trace 后修配置。**这比加自定义 raise
更尊重现有 dataclass 默认契约。**

> 设计权衡说明：原设计草案曾考虑在 `api_base` 既无 YAML 也无 env 时抛
> `ConfigError("Missing MODEL_API_URL")`，类似 `submission.py` 的早期失败语义。
> 但这会破坏 2 条现有测试。考虑到提交态 (`submission.py`) 已有该校验，开发态
> 没必要重复一遍硬约束 —— dev 用错了配置自然会拿到 LLM 报错，不需要 config
> 加载器替他做。这是 YAGNI 的合理应用。

### 2.3 与 `default_app_config()` / `AppConfig.from_dict()` 的边界

- **不动** `default_app_config()`：它返回 `AppConfig()` 默认值，不读 env、不读 YAML，
  现状语义不变（用于纯单测构造默认配置）
- **不动** `AppConfig.from_dict()`：它是子进程序列化用的纯字典还原器，env overlay 在
  YAML 加载点已经完成，子进程接收的是已 resolved 的 payload
- **新增** `_overlay_agent_env_vars(payload: dict) -> dict`：在 `_resolve_config_paths`
  之后调用，唯一一处 env 读取

## 三、涉及文件

| 文件 | 改动 |
| --- | --- |
| `src/data_agent_langchain/config.py` | 新增 `_overlay_agent_env_vars` 函数 + 在 `load_app_config` 调用 + 必填校验抛 `ConfigError` |
| `tests/test_phase5_config.py` | 新增 4 条 env 兜底测试 |
| `src/data_agent_langchain/README.md` | "快速上手"小节加一句 env-var 说明 |
| `src/data_agent_langchain/llm/factory.py` | **不改**（构造 ChatOpenAI 时已直接用 `cfg.agent.api_base/api_key/model`） |
| `src/data_agent_langchain/submission.py` | **不改**（已走 env） |
| `tests/test_submission_config.py` / `tests/test_submission_runtime.py` | **不改**（与 `load_app_config` 路径无关） |

## 四、TDD 测试矩阵（4 条 RED → GREEN）

| 编号 | 测试名 | YAML | env | 期望 |
| --- | --- | --- | --- | --- |
| 1 | `test_load_app_config_overlays_env_when_yaml_field_empty` | `agent.api_key=""` | `MODEL_API_KEY=sk-xx` | `cfg.agent.api_key == "sk-xx"` |
| 2 | `test_load_app_config_yaml_wins_over_env_when_yaml_field_set` | `agent.api_base="http://yaml/v1"` | `MODEL_API_URL=http://env/v1` | `cfg.agent.api_base == "http://yaml/v1"` |
| 3 | `test_load_app_config_overlays_env_when_yaml_field_missing` | YAML 不写 `agent.model` | `MODEL_NAME=qwen-test` | `cfg.agent.model == "qwen-test"` |
| 4 | `test_load_app_config_falls_back_to_dataclass_defaults_when_neither` | YAML 不写任何 `agent.*` 三字段 | 三个 env 全部未设 | `cfg.agent.model == "gpt-4.1-mini"`、`api_base == "https://api.openai.com/v1"`、`api_key == ""` |

每条测试用 pytest `monkeypatch.setenv` / `monkeypatch.delenv` 隔离 env 状态，避免污染
其他测试或宿主 env。测试 4 显式 `delenv` 三个 env 名，证明"既无 YAML 也无 env →
dataclass 默认"的兜底路径成立（且不抛错）。

## 五、不在范围（YAGNI）

1. **不加 lint / pre-commit 检查**阻止 YAML 中出现 `api_key:` —— 若后续发现仍有人硬编码
   再单独提议
2. **不改提交态** `submission.py`：它本来只读 env，不经过 `load_app_config`
3. **不动** `default_app_config()`：仍返回 `AppConfig()` 默认（用于单测的 in-process
   stub，不读 YAML 也不读 env）
4. **不引入 dotenv 依赖**：开发者用 PowerShell `$env:VAR=...` / `.env` 由 IDE / shell
   注入即可；引入 `python-dotenv` 会扩大依赖面且与提交态不一致

## 六、回滚策略

如发现 env 兜底引入意外行为：

1. `git revert` 单条 commit（建议把测试 + 实现 + README 更新 = 1 commit）
2. `_overlay_agent_env_vars` 是新增函数，移除即可，不破坏 `AppConfig` 数据结构
3. 现有 184 条测试覆盖 YAML 加载主路径，不依赖 env overlay

## 七、验收标准

- [ ] 4 条新测试 RED → GREEN
- [ ] `pytest tests/ -q` 全量 188 passed（= 184 + 4），无倒退
- [ ] `dabench-lc gateway-smoke --config configs/local.yaml` 可在 `MODEL_API_*` 已设
      的 PowerShell session 下成功运行（无 `agent.api_*` 字段也能跑通）
- [ ] README 快速上手小节增加 env-var 说明，4 行以内
- [ ] 不抛新的 `ConfigError` 错误码，不影响调用方 try/except 链路

---

## 八、2026-05-09 后续（统一配置参数 v4）

详见同目录上级新文件 `../统一配置/2026-05-09-统一配置参数-design-v4.md`。
关键演进：

1. **删除 `DABENCH_*` env 协议**：`submission.py` 顶部 `DEFAULT_MAX_WORKERS` /
   `DEFAULT_TASK_TIMEOUT_SECONDS` / `DEFAULT_MODEL_NAME` 三常量与 `_int_from_env`
   一并移除；本地与容器使用同一份 `RunConfig` / `AgentConfig` dataclass 默认。
2. **`MODEL_NAME` 升为容器必填**（D1）：与 `MODEL_API_URL` 同级硬约束，
   缺失抛 `SubmissionConfigError("Missing required environment variable:
   MODEL_NAME")`，避免容器静默 fallback 到 `qwen3.5-35b-a3b`。
3. **`AgentConfig.model` 默认保持 `gpt-4.1-mini`**：本设计文档第 45 / 74 / 115 行
   关于 dataclass 默认值的描述**仍正确**，无需修订。
4. **本地路径 env 兜底协议保持不变**：`load_app_config` 的
   `_overlay_agent_env_vars` / YAML > env > dataclass 默认三层优先级**未变**。
   2026-05-09 v4 仅收紧容器路径行为。

测试影响：`tests/test_phase5_config.py` 4 条测试不变；新增
`tests/test_submission_config.py::V5a / V5b / V6` + `test_submission_runtime.py::V7`
+ `test_submission_meta.py::__all__ 防回归` 共 5 条断言守护.
