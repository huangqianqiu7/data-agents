"""v5 §四第 1 步「配置构造」8 条测试。

提交入口 ``data_agent_langchain.submission`` 必须从 ``MODEL_*`` /
``DABENCH_*`` 环境变量直接构造 ``AppConfig``，绝不走 ``load_app_config``
读 YAML（避免敏感配置泄漏）。本文件断言这套环境变量到 AppConfig
字段的映射，以及容器路径硬编码（``/input`` / ``/tmp/dabench-runs`` /
``/app/gateway_caps.yaml``）与 ``langsmith_enabled=False``、
``graph_mode="plan_solve"`` 的强约束。
"""
from __future__ import annotations

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
    """v5 §二.4：MODEL_API_KEY 缺失回退 ``EMPTY``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    cfg = submission.build_submission_config()

    assert cfg.agent.api_key == "EMPTY"


def test_missing_model_name_falls_back_to_qwen(monkeypatch):
    """v5 §二.4：MODEL_NAME 缺失回退 ``qwen3.5-35b-a3b``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.delenv("MODEL_NAME", raising=False)

    cfg = submission.build_submission_config()

    assert cfg.agent.model == "qwen3.5-35b-a3b"


def test_dataset_root_path_defaults_to_slash_input(monkeypatch):
    """v5 §二.4：dataset.root_path 固定 ``/input``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")

    cfg = submission.build_submission_config()

    assert cfg.dataset.root_path == Path("/input")


def test_run_output_dir_defaults_to_tmp_dabench_runs(monkeypatch):
    """v5 §二.4：run.output_dir 固定 ``/tmp/dabench-runs``（内部目录，非官方 /output）。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")

    cfg = submission.build_submission_config()

    assert cfg.run.output_dir == Path("/tmp/dabench-runs")


def test_gateway_caps_path_defaults_to_app_gateway_caps_yaml(monkeypatch):
    """v5 §二.2：observability.gateway_caps_path 固定 ``/app/gateway_caps.yaml``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")

    cfg = submission.build_submission_config()

    assert cfg.observability.gateway_caps_path == Path("/app/gateway_caps.yaml")


def test_langsmith_enabled_is_false(monkeypatch):
    """v5 §二.9：observability.langsmith_enabled 在提交态强制 False。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    # 即便环境里被注入了 LangSmith 变量，提交态也必须忽略
    monkeypatch.setenv("LANGSMITH_API_KEY", "should-not-be-used")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    cfg = submission.build_submission_config()

    assert cfg.observability.langsmith_enabled is False


def test_graph_mode_is_plan_solve(monkeypatch):
    """v5 §二.4：graph_mode 在提交入口固定为 ``plan_solve``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")

    assert submission.SUBMISSION_GRAPH_MODE == "plan_solve"
