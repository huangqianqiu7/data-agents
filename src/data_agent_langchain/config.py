"""
LangGraph 后端的 ``AppConfig`` 与 YAML 配置加载工具。

包含 6 个 frozen dataclass：``DatasetConfig`` / ``ToolsConfig`` /
``AgentConfig`` / ``RunConfig`` / ``ObservabilityConfig`` /
``EvaluationConfig``，以及汇总它们的 :class:`AppConfig`。所有 dataclass
都是 ``frozen=True, slots=True``，保证 picklable + immutable，与 legacy
backend 约定一致。

公开函数：

  - :func:`default_app_config` —— 返回完全用默认值填的 ``AppConfig``。
  - :func:`load_app_config` —— 从 YAML 文件加载 + 路径解析 + 评测约束校验。
  - :func:`validate_eval_config` —— 当 ``evaluation.reproducible=true``
    时强制 ``agent.seed`` 设置且禁止 LangSmith（v4 E5 / §11.4.5）。

子进程序列化由 ``AppConfig.to_dict`` / ``AppConfig.from_dict`` 完成
（v4 E13），避免直接 pickle 嵌套 dataclass。
"""
from __future__ import annotations

import os
from dataclasses import fields, is_dataclass, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from data_agent_langchain.exceptions import ConfigError, ReproducibilityViolationError

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 子配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """数据集端配置项。

    ``root_path`` 与 legacy ``DatasetConfig.root_path`` 对齐；图节点只会
    通过 ``RunState["dataset_root"]`` 看到一个字符串副本，不会直接接触
    这个 ``Path``。
    """
    root_path: Path = field(default_factory=lambda: Path("."))


@dataclass(frozen=True, slots=True)
class ToolsConfig:
    """工具运行时配置项。"""
    # ``execute_python`` 沙箱单次最长执行时间（秒）；保持与
    # ``tools/python_exec.EXECUTE_PYTHON_TIMEOUT_SECONDS`` 一致（30s），
    # 让 description 字符串里写的数与实际超时永不漂移。
    python_timeout_s: float = 30.0
    # ``execute_context_sql`` 默认行数上限。
    sql_row_limit: int = 200


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """LLM / 主循环相关参数。

    默认值与 legacy ``data_agent_refactored.config.PlanAndSolveAgentConfig``
    保持一致，让 parity 测试无需为任何上限做特殊处理：

      - ``max_steps=20``（legacy plan-solve；ReAct 跑到 16 步、Plan-Solve 到 20）
      - ``max_replans=4``
      - ``max_gate_retries=2``（gate L2 触发；L3 在此之上再 +2）
      - ``max_model_retries=3``，backoff ``(2, 5, 10)`` 秒
      - 模型超时 ``120s`` / 工具超时 ``180s``
      - ``max_obs_chars=3000`` / ``max_context_tokens=24000``

    LLM 身份字段（``model`` / ``api_base`` / ``api_key``）的取值协议
    （详见 ``src/计划/统一配置/2026-05-09-统一配置参数-design-v4.md`` §4.2）：

      - **本地路径**（``load_app_config``）：YAML 显式非空值 > env vars
        （``MODEL_NAME`` / ``MODEL_API_URL`` / ``MODEL_API_KEY``）> dataclass
        默认。三层均缺失时落到本 dataclass 默认（``model = "gpt-4.1-mini"``、
        ``api_base = "https://api.openai.com/v1"``、``api_key = ""``），可纯
        离线开发。
      - **容器路径**（``submission.build_submission_config``）：
        ``MODEL_API_URL`` / ``MODEL_NAME`` 为必填 env，缺失抛
        ``SubmissionConfigError``；``MODEL_API_KEY`` 缺失回退
        ``EMPTY_API_KEY``。容器路径不读 YAML，不消费 dataclass 默认作为
        身份值，避免静默漂移。
    """
    # ----- 身份配置 -----
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.0
    backend: str = "langgraph"

    # ----- 主循环控制 -----
    max_steps: int = 60
    max_replans: int = 4
    max_gate_retries: int = 4
    require_data_preview_before_compute: bool = True
    progress: bool = False

    # ----- 动作模式 -----
    # Phase 0.5 网关 smoke 已对评测 gateway 通过，所以 Phase 6 把默认值切到
    # ``tool_calling``。显式 YAML 仍可设回 ``json_action`` 走 legacy fallback /
    # 离线 fake-model 测试。
    action_mode: str = "tool_calling"

    # ----- 重试 / 超时 -----
    model_timeout_s: float = 120.0
    tool_timeout_s: float = 180.0
    max_model_retries: int = 6
    model_retry_backoff: tuple[float, ...] = (2.0, 5.0, 10.0)

    # ----- 上下文预算 -----
    max_obs_chars: int = 3000
    max_context_tokens: int = 24000

    # ----- 发现 / 已知路径（parity 默认 OFF，v2 C13） -----
    enforce_discovery_gate: bool = False
    enforce_known_path_only: bool = False

    # ----- 决定性（reproducible 模式必填） -----
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class AppConfig:
    """汇总所有子配置的顶级配置对象。

    每个子配置都用 ``default_factory``，新增字段时向后兼容；YAML 缺少的
    字段会自动取默认值。
    """
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    run: RunConfig = field(default_factory=lambda: RunConfig())
    observability: ObservabilityConfig = field(default_factory=lambda: ObservabilityConfig())
    evaluation: EvaluationConfig = field(default_factory=lambda: EvaluationConfig())

    def to_dict(self) -> dict[str, Any]:
        """转为可 pickle 的纯字典（``Path`` 转 str、``tuple`` 转 list）。"""
        return _to_plain_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        """从纯字典还原 ``AppConfig``（与 :meth:`to_dict` 互逆）。"""
        return cls(
            dataset=_dataclass_from_dict(DatasetConfig, payload.get("dataset", {})),
            agent=_dataclass_from_dict(AgentConfig, payload.get("agent", {})),
            tools=_dataclass_from_dict(ToolsConfig, payload.get("tools", {})),
            run=_dataclass_from_dict(RunConfig, payload.get("run", {})),
            observability=_dataclass_from_dict(
                ObservabilityConfig, payload.get("observability", {})
            ),
            evaluation=_dataclass_from_dict(EvaluationConfig, payload.get("evaluation", {})),
        )


# ---------------------------------------------------------------------------
# 测试与节点入口都用得到的便捷构造器
# ---------------------------------------------------------------------------

def default_app_config() -> AppConfig:
    """返回一个全部字段都取默认值的 ``AppConfig``。"""
    return AppConfig()


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Runner / 子进程相关参数。"""
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "runs")
    run_id: str | None = None
    max_workers: int = 5
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """LangSmith / Gateway caps 等可观测性参数。"""
    langsmith_enabled: bool = False
    gateway_caps_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "gateway_caps.yaml"
    )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """评测约束（v4 E5 / §11.4.5）。"""
    reproducible: bool = False


def load_app_config(config_path: Path) -> AppConfig:
    """从 YAML 加载并验证 ``AppConfig``。失败抛 :class:`ConfigError`。"""
    try:
        payload: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ConfigError(f"Failed to load config from {config_path}: {exc}") from exc
    payload = _resolve_config_paths(payload)
    payload = _overlay_agent_env_vars(payload)
    cfg = AppConfig.from_dict(payload)
    validate_eval_config(cfg)
    return cfg


# 环境变量与 ``AgentConfig`` 字段的映射；与 ``submission.py`` 中 ``MODEL_*``
# 协议保持完全一致（``rules.zh.md`` §3.5 / §5.2 / §5.4）。
_AGENT_ENV_VAR_MAP: dict[str, str] = {
    "model": "MODEL_NAME",
    "api_base": "MODEL_API_URL",
    "api_key": "MODEL_API_KEY",
}


def _overlay_agent_env_vars(payload: dict[str, Any]) -> dict[str, Any]:
    """对 ``payload['agent']`` 中三个 LLM 字段做 env-var 兜底（YAML 非空赢）。

    设计契约（详见 ``src/完善计划/2026-05-08-env-var-fallback-design.md``）：
      1. YAML 显式非空值 → 保持 YAML 值
      2. YAML 字段缺失 / 空字符串 / None → 用对应 env var 兜底
      3. 既无 YAML 也无 env → 不写入 payload，由 ``AgentConfig`` 提供
         dataclass 默认值

    与 ``submission.build_submission_config`` 共用同一份 env 协议，让开发态与
    提交态走同一种 ``MODEL_*`` 环境变量约定，对齐 ``rules.zh.md`` §5.1
    "同一份代码两种环境"原则。
    """
    result = dict(payload)
    agent = dict(result.get("agent", {}))
    for field_name, env_name in _AGENT_ENV_VAR_MAP.items():
        yaml_value = agent.get(field_name)
        if yaml_value:  # 非空字符串 / 非 None / 非缺失 → YAML 赢
            continue
        env_value = os.environ.get(env_name)
        if env_value:
            agent[field_name] = env_value
        # 既无 YAML 也无 env：保持字段缺失，让 AppConfig.from_dict 走 dataclass 默认
    result["agent"] = agent
    return result


def validate_eval_config(config: AppConfig) -> None:
    """``evaluation.reproducible=true`` 时强制 seed 已设且禁用 LangSmith。"""
    if not config.evaluation.reproducible:
        return
    if config.agent.seed is None:
        raise ReproducibilityViolationError(
            "evaluation.reproducible=true requires agent.seed to be set."
        )
    if config.observability.langsmith_enabled:
        raise ReproducibilityViolationError(
            "evaluation.reproducible=true cannot be combined with LangSmith tracing."
        )


# ---------------------------------------------------------------------------
# 内部辅助：YAML 解析时的路径相对化 / dataclass 序列化
# ---------------------------------------------------------------------------

def _resolve_config_paths(payload: dict[str, Any]) -> dict[str, Any]:
    """把 YAML 里相对路径解析为相对项目根的绝对路径。"""
    result = dict(payload)
    dataset = dict(result.get("dataset", {}))
    run = dict(result.get("run", {}))
    observability = dict(result.get("observability", {}))
    if "root_path" in dataset:
        dataset["root_path"] = _path_value(dataset["root_path"])
    if "output_dir" in run:
        run["output_dir"] = _path_value(run["output_dir"])
    if "gateway_caps_path" in observability:
        observability["gateway_caps_path"] = _path_value(observability["gateway_caps_path"])
    result["dataset"] = dataset
    result["run"] = run
    result["observability"] = observability
    return result


def _path_value(raw_value: str | Path) -> Path:
    """绝对路径直接返回；相对路径相对于 :data:`PROJECT_ROOT` 解析。"""
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _to_plain_dict(value: Any) -> Any:
    """递归把 dataclass / Path / tuple 转为可 JSON / pickle 的纯结构。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_plain_dict(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if is_dataclass(value):
        return {field.name: _to_plain_dict(getattr(value, field.name)) for field in fields(value)}
    return value


def _dataclass_from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    """与 :func:`_to_plain_dict` 互逆：把纯字典塞进 dataclass。"""
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in payload:
            continue
        value = payload[field_info.name]
        if field_info.name in {"root_path", "output_dir", "gateway_caps_path"}:
            value = Path(value)
        elif field_info.name == "model_retry_backoff":
            value = tuple(float(item) for item in value)
        kwargs[field_info.name] = value
    return cls(**kwargs)


__all__ = [
    "AgentConfig",
    "AppConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ObservabilityConfig",
    "RunConfig",
    "ToolsConfig",
    "default_app_config",
    "load_app_config",
    "validate_eval_config",
]