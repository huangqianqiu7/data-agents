"""
LangGraph 后端的 ``AppConfig`` 与 YAML 配置加载工具。

包含 7 个 frozen dataclass：``DatasetConfig`` / ``ToolsConfig`` /
``AgentConfig`` / ``RunConfig`` / ``ObservabilityConfig`` /
``EvaluationConfig`` / ``MemoryConfig``，以及汇总它们的 :class:`AppConfig`。所有 dataclass
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
import types
import typing
from dataclasses import fields, is_dataclass, dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

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
class CorpusRagConfig:
    """v3 corpus RAG 配置（``01-design-v2.md §4.3``）。

    挂在 :class:`MemoryConfig.rag` 下；与 ``memory.mode`` 三态运行开关交叉决定
    实际行为：

      - ``memory.mode=disabled`` → 无论 ``rag.enabled`` 如何都强制关闭 RAG。
      - ``memory.mode=read_only_dataset`` 或 ``full`` 时 ``rag.enabled=true``
        才走 corpus 召回路径。

    所有字段都用 dataclass 默认值；``CorpusRagConfig()`` 等价于"禁用 corpus
    RAG，但保留所有合理默认参数等待开关打开"。
    """

    # ----- 总开关 -----
    # 全局开关；``memory.mode=disabled`` 时被强制视为 False。
    enabled: bool = False
    # L1：当前 task 的文档（M4 唯一实现）。
    task_corpus: bool = True
    # L2：维护方策划的只读语料（M4 占位，``05-shared-corpus-design.md`` 落地）。
    shared_corpus: bool = False
    shared_collections: tuple[str, ...] = ()
    # M4 不使用；保留供 shared corpus 独立提案。
    shared_path: Path | None = None

    # ----- 索引 -----
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    max_chunks_per_doc: int = 200
    max_docs_per_task: int = 100
    # Bug 6 修复：原 default=30.0 对 Harrier-270m CPU 冷启动 + ~60 chunks 编码
    # 不够（实测 60s），导致 production 路径 fail-closed 系统性触发，metrics 永远
    # 不出现 ``memory_rag`` 段。提高到 180.0（3x 实测）以 cover 200-300 chunks 常规
    # 负载，并保持 1/3 ``RunConfig.task_timeout_seconds=600`` 的预算比例。
    task_corpus_index_timeout_s: float = 180.0

    # ----- Embedding -----
    embedder_backend: Literal["sentence_transformer", "stub"] = "sentence_transformer"
    embedder_model_id: str = "microsoft/harrier-oss-v1-270m"
    embedder_device: Literal["cpu", "cuda", "auto"] = "cpu"
    embedder_dtype: Literal["float32", "float16", "auto"] = "auto"
    embedder_query_prompt_name: str = "web_search_query"
    embedder_max_seq_len: int = 1024
    embedder_batch_size: int = 8
    # ``None`` 走 HF 默认 ``HF_HOME``；评测镜像可指向 ``/opt/models/hf``。
    embedder_cache_dir: Path | None = None

    # ----- 向量库 -----
    vector_backend: Literal["chroma"] = "chroma"
    vector_distance: Literal["cosine", "ip", "l2"] = "cosine"

    # ----- 检索 -----
    retrieval_k: int = 4
    prompt_budget_chars: int = 1800

    # ----- 内容过滤（``Redactor`` 使用）-----
    # 命中任一 pattern 的整段文档将被丢弃；patterns 自带 ``(?i)`` 保证大小写不敏感。
    redact_patterns: tuple[str, ...] = (
        r"(?i)\banswer\b",
        r"(?i)\bhint\b",
        r"(?i)\bapproach\b",
        r"(?i)\bsolution\b",
    )
    # 命中任一 glob 的文件名将被跳过。
    redact_filenames: tuple[str, ...] = (
        "expected_output.json",
        "ground_truth*",
        "*label*",
        "*solution*",
    )


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Cross-task memory configuration (v2 design section 4.2)."""
    mode: str = "disabled"                          # disabled | read_only_dataset | full
    store_backend: str = "jsonl"                    # jsonl | sqlite (future)
    path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "artifacts" / "memory"
    )
    retriever_type: str = "exact"
    retrieval_max_results: int = 5
    # v3 corpus RAG 嵌套子配置（``01-design-v2.md §4.3``）；M4.4.1 引入。
    rag: CorpusRagConfig = field(default_factory=CorpusRagConfig)


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
    memory: MemoryConfig = field(default_factory=MemoryConfig)

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
            memory=_dataclass_from_dict(MemoryConfig, payload.get("memory", {})),
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
    memory = dict(result.get("memory", {}))
    if "root_path" in dataset:
        dataset["root_path"] = _path_value(dataset["root_path"])
    if "output_dir" in run:
        run["output_dir"] = _path_value(run["output_dir"])
    if "gateway_caps_path" in observability:
        observability["gateway_caps_path"] = _path_value(observability["gateway_caps_path"])
    if "path" in memory:
        memory["path"] = _path_value(memory["path"])
    result["dataset"] = dataset
    result["run"] = run
    result["observability"] = observability
    result["memory"] = memory
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


def _is_path_hint(hint: Any) -> bool:
    """判断类型 hint 是否为 ``Path`` 或 ``Path | None`` / ``Optional[Path]``。

    支持两种 Union 形式：

      - ``typing.Union[Path, None]`` / ``Optional[Path]`` —— ``get_origin`` 是
        ``typing.Union``。
      - ``Path | None`` （PEP 604，Python 3.10+）—— ``get_origin`` 是
        ``types.UnionType``。
    """
    if hint is Path:
        return True
    origin = get_origin(hint)
    union_origins: tuple[Any, ...] = (typing.Union,)
    union_type = getattr(types, "UnionType", None)
    if union_type is not None:
        union_origins = union_origins + (union_type,)
    if origin in union_origins:
        return any(arg is Path for arg in get_args(hint))
    return False


def _dataclass_from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    """与 :func:`_to_plain_dict` 互逆：把纯字典塞进 dataclass。

    支持的字段形状（按优先级匹配）：

      1. ``Path`` / ``Path | None`` / ``Optional[Path]`` 类型：value 为 ``None`` 时
         保持 ``None``，否则 ``Path(value)``。识别基于 ``typing.get_type_hints``
         解析后的真实类型，无需字段名硬编码。
      2. ``model_retry_backoff``：list → ``tuple[float, ...]``（向后兼容）。
      3. 嵌套 dataclass：``value`` 是 dict 时递归构造（v3 ``MemoryConfig.rag``）。
      4. 通用 ``tuple[T, ...]``：list → tuple（v3 ``CorpusRagConfig.shared_collections``
         / ``redact_patterns`` / ``redact_filenames``）。元素类型不强转。
      5. 其他类型（含 ``Literal[...]`` / 基本类型）：原样传入，dataclass 不做强
         校验，保持现状。

    ``from __future__ import annotations`` 下 ``field_info.type`` 是字符串，
    必须用 ``typing.get_type_hints(cls)`` 解析为真实类型，才能识别嵌套
    dataclass / ``tuple[T, ...]`` / ``Path | None`` 等结构化注解。
    """
    type_hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in payload:
            continue
        value = payload[field_info.name]
        hint = type_hints.get(field_info.name)
        kwargs[field_info.name] = _coerce_field(field_info.name, hint, value)
    return cls(**kwargs)


def _coerce_field(name: str, hint: Any, value: Any) -> Any:
    """根据字段名 + 类型 hint 把字典中的 ``value`` 强转回 dataclass 期望类型。"""
    # 1) Path / Path | None：``Path(None)`` 会爆，None 直通。
    if _is_path_hint(hint):
        return None if value is None else Path(value)

    # 2) 显式 tuple[float, ...] 字段（向后兼容；元素需要强转 float）。
    if name == "model_retry_backoff":
        return tuple(float(item) for item in value)

    # 3) 嵌套 dataclass：value 是 dict 时递归。
    if hint is not None and is_dataclass(hint) and isinstance(value, dict):
        return _dataclass_from_dict(hint, value)

    # 4) 通用 ``tuple[T, ...]``：list → tuple。
    if hint is not None and get_origin(hint) is tuple and isinstance(value, (list, tuple)):
        return tuple(value)

    # 5) 其他类型：原样返回。
    return value


__all__ = [
    "AgentConfig",
    "AppConfig",
    "CorpusRagConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "MemoryConfig",
    "ObservabilityConfig",
    "PROJECT_ROOT",
    "RunConfig",
    "ToolsConfig",
    "default_app_config",
    "load_app_config",
    "validate_eval_config",
]
