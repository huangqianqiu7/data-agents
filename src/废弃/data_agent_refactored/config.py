"""
Global configuration — dataclass hierarchy loaded from a YAML file.

全局配置模块 —— 基于 dataclass 的配置层级结构，支持从 YAML 文件加载。

Hierarchy::

    AppConfig                          ← 应用顶层配置，聚合以下三个子配置
    ├── DatasetConfig                  ← 数据集根路径配置
    ├── AgentConfig                    ← 模型 / API / 推理参数配置
    └── RunConfig                      ← 运行输出目录 / 并发 / 超时配置

    ReActAgentConfig                   ← ReAct 范式 Agent 的可调参数
    PlanAndSolveAgentConfig            ← Plan-and-Solve 范式 Agent 的可调参数

设计原则：
  1. 所有 dataclass 均为 frozen（不可变）+ slots（内存优化）
  2. 每个字段都提供合理的默认值，保证零配置即可运行
  3. YAML 加载函数 `load_app_config` 负责类型转换与路径解析
"""
from __future__ import annotations  # 允许在类型注解中使用 str | None 等新语法

from dataclasses import dataclass, field  # dataclass 装饰器与字段工厂
from pathlib import Path                  # 跨平台路径处理
from typing import Any                    # 通用类型提示

import yaml                               # YAML 解析库

from data_agent_refactored.exceptions import ConfigError  # 自定义配置异常

# 项目根目录：从当前文件 (config.py) 向上回溯两级
# 即 src/data_agent_refactored/config.py → src/data_agent_refactored → src → 项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    """返回数据集的默认根路径：<PROJECT_ROOT>/data/public/input"""
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    """返回运行产物的默认输出目录：<PROJECT_ROOT>/artifacts/runs"""
    return PROJECT_ROOT / "artifacts" / "runs"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """数据集配置：指定数据集文件存放的根目录。"""
    root_path: Path = field(default_factory=_default_dataset_root)  # 数据集根路径


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Agent 通用配置：控制模型选择、API 连接以及推理行为。"""
    model: str = "gpt-4.1-mini"                    # 使用的 LLM 模型名称
    api_base: str = "https://api.openai.com/v1"    # OpenAI 兼容 API 的基础 URL
    api_key: str = ""                              # API 密钥（留空时从环境变量读取）
    max_steps: int = 16                            # Agent 最大推理步数
    temperature: float = 0.0                       # 采样温度（0.0 = 贪心解码，确定性最高）
    progress: bool = False                         # 是否在终端打印每一步的进度与错误信息


@dataclass(frozen=True, slots=True)
class RunConfig:
    """运行配置：控制输出路径、并发度和超时。"""
    output_dir: Path = field(default_factory=_default_run_output_dir)  # 运行产物输出目录
    run_id: str | None = None       # 可选的运行 ID，为 None 时自动生成
    max_workers: int = 4             # 并行执行任务的最大工作线程数
    task_timeout_seconds: int = 600  # 单个任务的超时时间（秒），默认 10 分钟


@dataclass(frozen=True, slots=True)
class AppConfig:
    """应用顶层配置：聚合数据集、Agent、运行三个子配置。

    通过 `load_app_config()` 从 YAML 文件加载，也可直接构造使用默认值。
    """
    dataset: DatasetConfig = field(default_factory=DatasetConfig)  # 数据集子配置
    agent: AgentConfig = field(default_factory=AgentConfig)        # Agent 子配置
    run: RunConfig = field(default_factory=RunConfig)              # 运行子配置


# ---------------------------------------------------------------------------
# Agent 专属配置 dataclass（按 Agent 范式分别定义）
# 这些配置类通过 agents/__init__.py re-export，
# 可用 from data_agent_refactored.agents import XxxConfig 导入。
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    """ReAct Agent 的可调参数。

    ReAct（Reasoning + Acting）范式在每一步中先推理再行动，
    通过 Thought → Action → Observation 循环完成任务。
    """
    max_steps: int = 16                                     # 最大推理-行动循环步数
    progress: bool = False                                  # 是否显示 tqdm 进度条
    require_data_preview_before_compute: bool = True         # 是否强制在计算前先预览数据
    model_timeout_s: float = 120.0                          # 单次模型调用超时（秒）
    tool_timeout_s: float = 180.0                           # 单次工具执行超时（秒）
    max_model_retries: int = 3                              # 模型调用失败后的最大重试次数
    model_retry_backoff: tuple[float, ...] = (2.0, 5.0, 10.0)  # 重试退避间隔序列（秒）


@dataclass(frozen=True, slots=True)
class PlanAndSolveAgentConfig:
    """Plan-and-Solve Agent 的可调参数。

    Plan-and-Solve 范式分两阶段工作：
      Phase 1 — Planning：LLM 生成编号步骤计划
      Phase 2 — Execution：逐步执行计划，必要时触发 Replan
    """
    max_steps: int = 20                                     # 执行阶段最大步数（所有计划累计）
    max_replans: int = 2                                    # 允许的最大重新规划次数
    progress: bool = False                                  # 是否显示 tqdm 进度条
    require_data_preview_before_compute: bool = True         # 是否强制在计算前先预览数据
    max_obs_chars: int = 3000                               # 工具观测结果的最大字符数（超出截断）
    max_context_tokens: int = 24000                         # 历史上下文的最大 token 数（防止超出窗口）
    model_timeout_s: float = 120.0                          # 单次模型调用超时（秒）
    tool_timeout_s: float = 180.0                           # 单次工具执行超时（秒）
    max_gate_retries: int = 2                               # 门控（Gate）检查失败后的最大重试次数
    max_model_retries: int = 3                              # 模型调用失败后的最大重试次数
    model_retry_backoff: tuple[float, ...] = (2.0, 5.0, 10.0)  # 重试退避间隔序列（秒）


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _path_value(raw_value: str | None, default_value: Path) -> Path:
    """将 YAML 中的原始字符串路径解析为 Path 对象。

    解析规则：
      1. 如果 raw_value 为空/None → 返回 default_value
      2. 如果 raw_value 是绝对路径 → 直接返回
      3. 否则视为相对于 PROJECT_ROOT 的相对路径 → 拼接并 resolve
    """
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


# ---------------------------------------------------------------------------
# YAML 配置加载器
# ---------------------------------------------------------------------------

def load_app_config(config_path: Path) -> AppConfig:
    """从 YAML 文件加载 :class:`AppConfig`。

    加载流程：
      1. 读取并解析 YAML 文件
      2. 分别提取 dataset / agent / run 三个顶层字段
      3. 对每个字段进行类型转换，缺失时使用默认值
      4. 组装并返回 AppConfig 实例

    Args:
        config_path: YAML 配置文件的路径。

    Returns:
        解析后的 AppConfig 实例。

    Raises:
        ConfigError: 文件不存在、读取失败或 YAML 语法错误时抛出。
    """
    # ---- 第 1 步：读取 YAML 文件 ----
    try:
        payload: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:
        raise ConfigError(f"Failed to load config from {config_path}: {exc}") from exc

    # ---- 第 2 步：创建各子配置的默认实例（用于取默认值） ----
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    # ---- 第 3 步：从 YAML payload 中提取各子字典 ----
    dataset_payload: dict[str, Any] = payload.get("dataset", {})  # YAML 中 dataset: 部分
    agent_payload: dict[str, Any] = payload.get("agent", {})      # YAML 中 agent: 部分
    run_payload: dict[str, Any] = payload.get("run", {})          # YAML 中 run: 部分

    # ---- 第 4 步：构建 DatasetConfig ----
    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )

    # ---- 第 5 步：构建 AgentConfig（逐字段类型转换） ----
    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        progress=bool(agent_payload.get("progress", agent_defaults.progress)),
    )

    # ---- 第 6 步：构建 RunConfig ----
    # run_id 需要特殊处理：去除空白后，空字符串视为 None
    raw_run_id = run_payload.get("run_id")
    run_id: str | None = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None  # 空字符串 → None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(
            run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)
        ),
    )

    # ---- 第 7 步：组装并返回顶层 AppConfig ----
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
