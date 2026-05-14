"""v5 §四第 1 步「配置构造」测试 + 2026-05-09 统一配置参数收敛断言。

提交入口 ``data_agent_langchain.submission`` 必须从 ``MODEL_*`` 环境变量
直接构造 ``AppConfig``，绝不走 ``load_app_config`` 读 YAML（避免敏感配置
泄漏）。本文件断言：

1. ``MODEL_*`` 三件套 env 到 ``AppConfig`` 字段的映射；
2. 容器路径硬编码（``/input`` / ``/tmp/dabench-runs`` /
   ``/app/gateway_caps.yaml``）与 ``langsmith_enabled=False`` /
   ``graph_mode="plan_solve"`` 强约束；
3. ``2026-05-09 统一配置参数 v4`` 设计：
   ``build_submission_config`` 不再含 ``DABENCH_*`` env 与 ``DEFAULT_*``
   常量，``cfg.run`` / ``cfg.agent`` 非身份字段全部等于对应 dataclass 默认；
   ``MODEL_NAME`` 缺失抛 ``SubmissionConfigError``（V5a / V5b / V6）。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest


def test_missing_model_api_url_raises_redacted_config_error(monkeypatch):
    """v5 §二.4：缺 MODEL_API_URL 立即失败。"""
    from data_agent_langchain import submission

    monkeypatch.delenv("MODEL_API_URL", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    with pytest.raises(submission.SubmissionConfigError) as exc_info:
        submission.build_submission_config()

    message = str(exc_info.value)
    # 错误信息只能脱敏：必须提示 env 名，不准透露任何 URL / key 字面值
    assert "MODEL_API_URL" in message
    assert "://" not in message
    assert "EMPTY" not in message  # 不要把回退值也写进错误


def test_missing_model_api_key_falls_back_to_empty(monkeypatch):
    """v5 §二.4：MODEL_API_KEY 缺失回退 ``EMPTY``。

    2026-05-09 v4 D1：``MODEL_NAME`` 已升为容器必填；本测试仅守护
    ``MODEL_API_KEY`` 的 fallback 语义，需先 setenv ``MODEL_NAME``。
    """
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    cfg = submission.build_submission_config()

    assert cfg.agent.api_key == "EMPTY"


def test_missing_model_name_raises_redacted_config_error(monkeypatch):
    """2026-05-09 v4 §4.1 D1：``MODEL_NAME`` 缺失抛 ``SubmissionConfigError``。

    与 ``test_missing_model_api_url_raises_redacted_config_error`` 同模式：
    错误信息只点名 env 名，不含字面值或回退提示。
    """
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    with pytest.raises(submission.SubmissionConfigError) as exc_info:
        submission.build_submission_config()

    message = str(exc_info.value)
    # 错误信息只能脱敏：必须提示 env 名，不准透露字面值或 fallback hint
    assert "MODEL_NAME" in message
    assert "://" not in message
    assert "EMPTY" not in message
    # 不要在错误里提示具体模型名（即便是历史 fallback）
    assert "qwen" not in message.lower()
    assert "gpt" not in message.lower()


def test_dataset_root_path_defaults_to_slash_input(monkeypatch):
    """v5 §二.4：dataset.root_path 固定 ``/input``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")

    cfg = submission.build_submission_config()

    assert cfg.dataset.root_path == Path("/input")


def test_run_output_dir_defaults_to_tmp_dabench_runs(monkeypatch):
    """v5 §二.4：run.output_dir 固定 ``/tmp/dabench-runs``（内部目录，非官方 /output）。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")

    cfg = submission.build_submission_config()

    assert cfg.run.output_dir == Path("/tmp/dabench-runs")


def test_gateway_caps_path_defaults_to_app_gateway_caps_yaml(monkeypatch):
    """v5 §二.2：observability.gateway_caps_path 固定 ``/app/gateway_caps.yaml``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")

    cfg = submission.build_submission_config()

    assert cfg.observability.gateway_caps_path == Path("/app/gateway_caps.yaml")


def test_langsmith_enabled_is_false(monkeypatch):
    """v5 §二.9：observability.langsmith_enabled 在提交态强制 False。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    # 即便环境里被注入了 LangSmith 变量，提交态也必须忽略
    monkeypatch.setenv("LANGSMITH_API_KEY", "should-not-be-used")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    cfg = submission.build_submission_config()

    assert cfg.observability.langsmith_enabled is False


def test_graph_mode_is_plan_solve(monkeypatch):
    """v5 §二.4：graph_mode 在提交入口固定为 ``plan_solve``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")

    assert submission.SUBMISSION_GRAPH_MODE == "plan_solve"


# ---------------------------------------------------------------------------
# 2026-05-09 v4 §6：统一配置参数收敛断言（V5a / V5b / V6）
# ---------------------------------------------------------------------------

def test_default_max_workers_matches_runconfig_default(monkeypatch):
    """v4 V5a：容器路径 ``cfg.run.max_workers`` == ``RunConfig().max_workers``。

    ``submission.py`` 删除 ``DEFAULT_MAX_WORKERS`` 常量与
    ``_int_from_env("DABENCH_MAX_WORKERS", ...)`` 后，``max_workers``
    必须由 ``RunConfig`` dataclass 默认接管。
    """
    from data_agent_langchain import submission
    from data_agent_langchain.config import RunConfig

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("DABENCH_MAX_WORKERS", raising=False)

    cfg = submission.build_submission_config()

    assert cfg.run.max_workers == RunConfig().max_workers


def test_default_task_timeout_matches_runconfig_default(monkeypatch):
    """v4 V5b：``cfg.run.task_timeout_seconds`` == ``RunConfig().task_timeout_seconds``。

    ``submission.py`` 删除 ``DEFAULT_TASK_TIMEOUT_SECONDS`` 常量与
    ``DABENCH_TASK_TIMEOUT_SECONDS`` env 读取后由 ``RunConfig`` 默认接管。
    """
    from data_agent_langchain import submission
    from data_agent_langchain.config import RunConfig

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("DABENCH_TASK_TIMEOUT_SECONDS", raising=False)

    cfg = submission.build_submission_config()

    assert cfg.run.task_timeout_seconds == RunConfig().task_timeout_seconds


def test_submission_appconfig_equals_default_except_paths_and_identity(monkeypatch):
    """v4 V6（D2）：``build_submission_config()`` 与 ``default_app_config()`` 在
    "非身份 / 非容器特化路径"字段上 dataclass 级别相等。

    把容器路径与 LLM 身份字段 (model / api_base / api_key) 归一化到
    baseline 后，两个 ``AppConfig`` 必须完全相等。未来若有人偷偷在
    ``build_submission_config`` 引入新字段差异，本测试立即 RED。
    """
    from data_agent_langchain import submission
    from data_agent_langchain.config import default_app_config

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    cfg = submission.build_submission_config()
    baseline = default_app_config()

    normalized = replace(
        cfg,
        # 容器特化路径：dataset.root_path / run.output_dir /
        # observability.gateway_caps_path 都允许与 baseline 不同
        dataset=baseline.dataset,
        agent=replace(
            cfg.agent,
            model=baseline.agent.model,
            api_base=baseline.agent.api_base,
            api_key=baseline.agent.api_key,
        ),
        run=replace(cfg.run, output_dir=baseline.run.output_dir),
        observability=replace(
            cfg.observability,
            gateway_caps_path=baseline.observability.gateway_caps_path,
        ),
        memory=replace(cfg.memory, mode=baseline.memory.mode),
    )

    assert normalized == baseline
