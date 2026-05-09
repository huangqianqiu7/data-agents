from __future__ import annotations

import yaml
from dataclasses import replace
from pathlib import Path

from data_agent_langchain.config import AgentConfig, AppConfig, ObservabilityConfig
from data_agent_langchain.observability.gateway_caps import GatewayCaps


class _FakeMessage:
    def __init__(self, *, tool_calls=None):
        self.tool_calls = tool_calls or []
        self.content = "ok"


class _SuccessfulProbeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.bound_kwargs = []

    def bind_tools(self, tools, **kwargs):
        self.bound_kwargs.append(kwargs)
        return self

    def invoke(self, messages):
        return _FakeMessage(tool_calls=[{"name": "probe_tool", "args": {}}])


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        agent=AgentConfig(
            model="probe-model",
            api_base="https://gateway.example/v1",
            api_key="yaml-key",
        ),
        observability=ObservabilityConfig(gateway_caps_path=tmp_path / "gateway_caps.yaml"),
    )


def test_gateway_smoke_writes_successful_caps(tmp_path: Path):
    from data_agent_langchain.observability.gateway_smoke import run_gateway_smoke

    output_path = tmp_path / "gateway_caps.yaml"
    caps = run_gateway_smoke(_config(tmp_path), output_path=output_path, chat_model_factory=_SuccessfulProbeModel)

    assert caps == GatewayCaps(
        tool_calling=True,
        parallel_tool_calls=True,
        seed_param=True,
        strict_mode=True,
    )
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "gateway_caps": {
            "tool_calling": True,
            "parallel_tool_calls": True,
            "seed_param": True,
            "strict_mode": True,
        }
    }


class _FailingToolCallingModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def bind_tools(self, tools, **kwargs):
        raise RuntimeError("tool calling unsupported")

    def invoke(self, messages):
        return _FakeMessage()


class _RecordingProbeModel(_SuccessfulProbeModel):
    seen_kwargs: list[dict] = []

    def __init__(self, **kwargs):
        type(self).seen_kwargs.append(kwargs)
        super().__init__(**kwargs)


def test_gateway_smoke_records_false_when_tool_calling_fails(tmp_path: Path):
    from data_agent_langchain.observability.gateway_smoke import run_gateway_smoke

    caps = run_gateway_smoke(_config(tmp_path), chat_model_factory=_FailingToolCallingModel)

    assert caps.tool_calling is False
    assert caps.parallel_tool_calls is None
    assert caps.seed_param is True
    assert caps.strict_mode is False


def test_gateway_smoke_falls_back_to_openai_api_key(tmp_path: Path, monkeypatch):
    from data_agent_langchain.observability.gateway_smoke import run_gateway_smoke

    _RecordingProbeModel.seen_kwargs = []
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    base = _config(tmp_path)
    cfg = replace(base, agent=replace(base.agent, api_key=""))

    run_gateway_smoke(cfg, chat_model_factory=_RecordingProbeModel)

    assert _RecordingProbeModel.seen_kwargs
    assert _RecordingProbeModel.seen_kwargs[0]["api_key"] == "env-key"


def test_cli_exposes_gateway_smoke_command():
    from typer.testing import CliRunner

    from data_agent_langchain.cli import app

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "gateway-smoke" in result.output


def test_cli_gateway_smoke_writes_to_output_path(tmp_path: Path, monkeypatch):
    from typer.testing import CliRunner

    import data_agent_langchain.cli as cli_module
    from data_agent_langchain.config import AppConfig
    from data_agent_langchain.observability.gateway_caps import GatewayCaps

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "agent:\n  model: probe\n  api_base: https://gateway.example/v1\n"
        f"observability:\n  gateway_caps_path: {tmp_path / 'default_caps.yaml'}\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "custom_caps.yaml"
    seen = {}

    def fake_smoke(config: AppConfig, *, output_path=None, chat_model_factory=None):
        seen["output_path"] = output_path
        return GatewayCaps(
            tool_calling=True,
            parallel_tool_calls=False,
            seed_param=False,
            strict_mode=False,
        )

    monkeypatch.setattr(cli_module, "run_gateway_smoke", fake_smoke)

    result = CliRunner().invoke(
        cli_module.app,
        ["gateway-smoke", "--config", str(config_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert seen["output_path"] == output_path
    assert "tool_calling: True" in result.output
