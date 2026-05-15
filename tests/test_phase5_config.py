from __future__ import annotations

from pathlib import Path

import pytest

from data_agent_langchain.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    EvaluationConfig,
    ObservabilityConfig,
    PROJECT_ROOT,
    RunConfig,
    ToolsConfig,
    load_app_config,
    validate_eval_config,
)
from data_agent_langchain.exceptions import ReproducibilityViolationError


def test_app_config_round_trip_preserves_paths_and_tuples(tmp_path: Path):
    cfg = AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "input"),
        agent=AgentConfig(
            model="demo-model",
            action_mode="tool_calling",
            model_retry_backoff=(0.5, 1.5),
            seed=123,
        ),
        tools=ToolsConfig(python_timeout_s=11.0, sql_row_limit=77),
        run=RunConfig(output_dir=tmp_path / "runs", run_id="r1", max_workers=2, task_timeout_seconds=9),
        observability=ObservabilityConfig(langsmith_enabled=False, gateway_caps_path=tmp_path / "caps.yaml"),
        evaluation=EvaluationConfig(reproducible=True),
    )
    payload = cfg.to_dict()
    assert payload["dataset"]["root_path"] == str(tmp_path / "input")
    assert payload["agent"]["model_retry_backoff"] == [0.5, 1.5]
    assert AppConfig.from_dict(payload) == cfg


def test_load_app_config_from_yaml_resolves_relative_paths(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  root_path: data/public/input
agent:
  model: test-model
  max_steps: 5
  backend: langgraph
run:
  output_dir: artifacts/out
  run_id: demo
  max_workers: 1
observability:
  langsmith_enabled: false
  gateway_caps_path: artifacts/gateway_caps.yaml
evaluation:
  reproducible: false
""".strip(),
        encoding="utf-8",
    )
    cfg = load_app_config(config_path)
    assert cfg.dataset.root_path.is_absolute()
    assert cfg.dataset.root_path.name == "input"
    assert cfg.agent.model == "test-model"
    assert cfg.agent.backend == "langgraph"
    assert cfg.run.run_id == "demo"
    assert cfg.observability.gateway_caps_path.name == "gateway_caps.yaml"


def test_load_app_config_from_yaml_resolves_memory_path(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
memory:
  path: artifacts/memory
""".strip(),
        encoding="utf-8",
    )

    cfg = load_app_config(config_path)

    assert cfg.memory.path == PROJECT_ROOT / "artifacts" / "memory"
    assert cfg.memory.path.is_absolute()


def test_validate_eval_config_rejects_reproducible_without_seed():
    cfg = AppConfig(evaluation=EvaluationConfig(reproducible=True), agent=AgentConfig(seed=None))
    with pytest.raises(ReproducibilityViolationError):
        validate_eval_config(cfg)


def test_validate_eval_config_rejects_langsmith_in_reproducible_mode():
    cfg = AppConfig(
        evaluation=EvaluationConfig(reproducible=True),
        agent=AgentConfig(seed=1),
        observability=ObservabilityConfig(langsmith_enabled=True),
    )
    with pytest.raises(ReproducibilityViolationError):
        validate_eval_config(cfg)


def test_agent_config_defaults_to_tool_calling_after_gateway_smoke():
    from data_agent_langchain.config import AgentConfig, default_app_config

    assert AgentConfig().action_mode == "tool_calling"
    assert default_app_config().agent.action_mode == "tool_calling"


def test_run_config_task_timeout_seconds_default_is_900():
    """2026-05-15 P1 §1 followup 回归：``task_timeout_seconds`` 从 600 提升到 900。

    背景：M4 corpus RAG E2E A/B 实测中，``plan_solve + RAG`` (PS_on) 在 task_344
    上偶发命中 600s 上限（gateway_caps.yaml 单 step 120s × 5 step 即吃满预算）。
    5-task 量化数据：1/5 ≈ 20% 触发率，落入 followups §1 候选方案 A 区间
    （<30% but >10% → 实施 A）。

    bump 到 900s 给 PS_on 慢路径留 1.5x 余量，同时与 gateway_caps.yaml 单 step
    120s × max_steps 60 / max_replans 4 ≈ 480s 上限保持安全余量。
    """
    assert RunConfig().task_timeout_seconds == 900


def test_load_app_config_can_explicitly_keep_json_action(tmp_path: Path):
    from data_agent_langchain.config import load_app_config

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "agent:\n  action_mode: json_action\n",
        encoding="utf-8",
    )

    cfg = load_app_config(cfg_path)

    assert cfg.agent.action_mode == "json_action"


# ---------------------------------------------------------------------------
# Env-var fallback for agent.{model, api_base, api_key} —— 2026-05-08 design
# 详见 src/完善计划/2026-05-08-env-var-fallback-design.md。
#
# 设计契约：
#   1. YAML 显式非空值 → 使用 YAML 值（YAML 非空赢）
#   2. YAML 字段缺失 / 空字符串 → 用 env vars 兜底
#         agent.model    ← MODEL_NAME
#         agent.api_base ← MODEL_API_URL
#         agent.api_key  ← MODEL_API_KEY
#   3. 既无 YAML 也无 env → 用 AgentConfig dataclass 默认值
# ---------------------------------------------------------------------------


def test_load_app_config_overlays_env_when_yaml_field_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """YAML 里 agent.api_key 为空字符串时，env MODEL_API_KEY 应被采用。"""
    monkeypatch.setenv("MODEL_API_KEY", "sk-from-env")
    monkeypatch.delenv("MODEL_API_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        'agent:\n  api_key: ""\n',
        encoding="utf-8",
    )

    cfg = load_app_config(cfg_path)

    assert cfg.agent.api_key == "sk-from-env"


def test_load_app_config_yaml_wins_over_env_when_yaml_field_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """YAML 显式写了 api_base，则即使 env MODEL_API_URL 也设了，也用 YAML 值。"""
    monkeypatch.setenv("MODEL_API_URL", "http://env-gateway/v1")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "agent:\n  api_base: http://yaml-gateway/v1\n",
        encoding="utf-8",
    )

    cfg = load_app_config(cfg_path)

    assert cfg.agent.api_base == "http://yaml-gateway/v1"


def test_load_app_config_overlays_env_when_yaml_field_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """YAML 完全不写 agent.model 时，env MODEL_NAME 应被采用。"""
    monkeypatch.setenv("MODEL_NAME", "qwen-from-env-test")
    monkeypatch.delenv("MODEL_API_URL", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "agent:\n  action_mode: tool_calling\n",
        encoding="utf-8",
    )

    cfg = load_app_config(cfg_path)

    assert cfg.agent.model == "qwen-from-env-test"


def test_load_app_config_falls_back_to_dataclass_defaults_when_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """YAML 与 env 都未提供时，回退到 AgentConfig dataclass 默认值（不抛错）。"""
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_API_URL", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "agent:\n  action_mode: tool_calling\n",
        encoding="utf-8",
    )

    cfg = load_app_config(cfg_path)

    assert cfg.agent.model == "gpt-4.1-mini"
    assert cfg.agent.api_base == "https://api.openai.com/v1"
    assert cfg.agent.api_key == ""
