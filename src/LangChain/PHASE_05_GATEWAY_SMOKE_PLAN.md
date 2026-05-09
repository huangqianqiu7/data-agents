# Phase 0.5 Gateway Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dabench-lc gateway-smoke` and enforce gateway caps for explicit `tool_calling` runs without changing the default `json_action` mode.

**Architecture:** A new focused `observability/gateway_smoke.py` module performs capability probes and writes `gateway_caps.yaml`. The Typer CLI exposes the command under `dabench-lc` only. The runner loads `GatewayCaps` only when `config.agent.action_mode == "tool_calling"`, binds tools through the existing LLM factory path, and leaves `json_action` unchanged.

**Tech Stack:** Python 3.13, Typer, PyYAML, LangChain `ChatOpenAI` / Runnable fakes in tests, existing `AppConfig`, `GatewayCaps`, runner, and `FakeListChatModel` / `RunnableLambda` for offline tests.

---

## File map

- Create: `src/data_agent_langchain/observability/gateway_smoke.py`
  - Responsibility: credential resolution, capability probing, YAML writing.
- Modify: `src/data_agent_langchain/cli.py`
  - Responsibility: add `dabench-lc gateway-smoke` command only.
- Modify: `src/data_agent_langchain/run/runner.py`
  - Responsibility: enforce caps for explicit `tool_calling` runs and inject a caps-bound LLM.
- Modify: `src/data_agent_langchain/observability/gateway_caps.py`
  - Responsibility: update stale command hint from `dabench gateway-smoke` to `dabench-lc gateway-smoke` if needed.
- Test: `tests/test_phase05_gateway_smoke.py`
  - Responsibility: offline tests for smoke probes, output YAML, API key fallback, and CLI registration.
- Test: `tests/test_phase5_runner_cli.py`
  - Responsibility: runner startup check and tool-calling binding tests.
- Modify: `src/LangChain/MIGRATION_STATUS.md`
  - Responsibility: mark Phase 0.5 gateway smoke implementation status and record verification.

Use the direct conda env Python for commands:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest <tests> -q
```

---

### Task 1: Gateway smoke core module

**Files:**
- Create: `src/data_agent_langchain/observability/gateway_smoke.py`
- Test: `tests/test_phase05_gateway_smoke.py`

- [ ] **Step 1: Write failing tests for successful smoke and YAML output**

Append/create `tests/test_phase05_gateway_smoke.py` with:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase05_gateway_smoke.py::test_gateway_smoke_writes_successful_caps -q
```

Expected: FAIL with `ModuleNotFoundError` or missing `run_gateway_smoke`.

- [ ] **Step 3: Implement minimal gateway smoke core**

Create `src/data_agent_langchain/observability/gateway_smoke.py`:

```python
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import yaml
from langchain_core.tools import tool

from data_agent_langchain.config import AppConfig
from data_agent_langchain.observability.gateway_caps import GatewayCaps

ChatModelFactory = Callable[..., Any]


@tool
def probe_tool() -> str:
    return "ok"


def run_gateway_smoke(
    config: AppConfig,
    *,
    output_path: Path | None = None,
    chat_model_factory: ChatModelFactory | None = None,
) -> GatewayCaps:
    factory = chat_model_factory or _default_chat_model_factory
    resolved_config = _config_with_resolved_api_key(config)
    target_path = output_path or resolved_config.observability.gateway_caps_path

    tool_calling = _probe_tool_calling(factory, resolved_config)
    parallel_tool_calls = (
        _probe_parallel_tool_calls(factory, resolved_config) if tool_calling else None
    )
    seed_param = _probe_seed_param(factory, resolved_config)
    strict_mode = _probe_strict_mode(factory, resolved_config) if tool_calling else False

    caps = GatewayCaps(
        tool_calling=tool_calling,
        parallel_tool_calls=parallel_tool_calls,
        seed_param=seed_param,
        strict_mode=strict_mode,
    )
    write_gateway_caps(caps, target_path)
    return caps


def write_gateway_caps(caps: GatewayCaps, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gateway_caps": {
            "tool_calling": caps.tool_calling,
            "parallel_tool_calls": caps.parallel_tool_calls,
            "seed_param": caps.seed_param,
            "strict_mode": caps.strict_mode,
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _config_with_resolved_api_key(config: AppConfig) -> AppConfig:
    if config.agent.api_key:
        return config
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if not env_key:
        return config
    return replace(config, agent=replace(config.agent, api_key=env_key))


def _default_chat_model_factory(**kwargs: Any) -> Any:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(**kwargs)


def _model_kwargs(config: AppConfig, *, seed: int | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": config.agent.model,
        "base_url": config.agent.api_base,
        "api_key": config.agent.api_key,
        "temperature": config.agent.temperature,
        "request_timeout": float(config.agent.model_timeout_s),
        "max_retries": 0,
    }
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs


def _probe_tool_calling(factory: ChatModelFactory, config: AppConfig) -> bool:
    try:
        llm = factory(**_model_kwargs(config))
        response = llm.bind_tools([probe_tool]).invoke("Call the probe tool.")
        return bool(getattr(response, "tool_calls", None))
    except Exception:
        return False


def _probe_parallel_tool_calls(factory: ChatModelFactory, config: AppConfig) -> bool:
    try:
        llm = factory(**_model_kwargs(config))
        response = llm.bind_tools([probe_tool], parallel_tool_calls=False).invoke(
            "Call the probe tool."
        )
        return bool(getattr(response, "tool_calls", None))
    except Exception:
        return False


def _probe_seed_param(factory: ChatModelFactory, config: AppConfig) -> bool:
    try:
        factory(**_model_kwargs(config, seed=42)).invoke("Reply ok.")
        return True
    except Exception:
        return False


def _probe_strict_mode(factory: ChatModelFactory, config: AppConfig) -> bool:
    try:
        llm = factory(**_model_kwargs(config))
        response = llm.bind_tools([probe_tool], strict=True).invoke("Call the probe tool.")
        return bool(getattr(response, "tool_calls", None))
    except Exception:
        return False


__all__ = ["run_gateway_smoke", "write_gateway_caps"]
```

- [ ] **Step 4: Run test to verify it passes**

Run the same targeted test. Expected: PASS.

- [ ] **Step 5: Add failing tests for failures and API key fallback**

Append to `tests/test_phase05_gateway_smoke.py`:

```python
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
    cfg = replace(_config(tmp_path), agent=replace(_config(tmp_path).agent, api_key=""))

    run_gateway_smoke(cfg, chat_model_factory=_RecordingProbeModel)

    assert _RecordingProbeModel.seen_kwargs
    assert _RecordingProbeModel.seen_kwargs[0]["api_key"] == "env-key"
```

- [ ] **Step 6: Run new tests to verify they fail or pass for the expected reason**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase05_gateway_smoke.py -q
```

Expected after Step 3 implementation: all Task 1 tests should PASS. If the failure test reveals strict probe behavior issues, adjust only the relevant probe function.

---

### Task 2: CLI command `dabench-lc gateway-smoke`

**Files:**
- Modify: `src/data_agent_langchain/cli.py`
- Test: `tests/test_phase05_gateway_smoke.py`

- [ ] **Step 1: Write failing CLI registration test**

Append:

```python
def test_cli_exposes_gateway_smoke_command():
    from typer.testing import CliRunner

    from data_agent_langchain.cli import app

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "gateway-smoke" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase05_gateway_smoke.py::test_cli_exposes_gateway_smoke_command -q
```

Expected: FAIL because `gateway-smoke` is not in help.

- [ ] **Step 3: Implement CLI command**

Modify `src/data_agent_langchain/cli.py`:

```python
from data_agent_langchain.observability.gateway_smoke import run_gateway_smoke
```

Add below `run_benchmark_command`:

```python
@app.command("gateway-smoke")
def gateway_smoke_command(
    config: Annotated[Path, typer.Option("--config", "-c", help="YAML config path.")],
    output: Annotated[Path | None, typer.Option("--output", help="gateway_caps.yaml output path.")] = None,
) -> None:
    cfg = load_app_config(config)
    caps = run_gateway_smoke(cfg, output_path=output)
    written_path = output or cfg.observability.gateway_caps_path
    typer.echo(f"gateway_caps: {written_path}")
    typer.echo(f"tool_calling: {caps.tool_calling}")
    typer.echo(f"parallel_tool_calls: {caps.parallel_tool_calls}")
    typer.echo(f"seed_param: {caps.seed_param}")
    typer.echo(f"strict_mode: {caps.strict_mode}")
    typer.echo("default action_mode remains json_action until Phase 6 default-switch approval")
```

- [ ] **Step 4: Run CLI registration test**

Expected: PASS.

- [ ] **Step 5: Add command behavior test with monkeypatched smoke**

Append:

```python
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
```

- [ ] **Step 6: Run CLI tests**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase05_gateway_smoke.py -q
```

Expected: PASS.

---

### Task 3: Runner startup check and tool binding for explicit `tool_calling`

**Files:**
- Modify: `src/data_agent_langchain/run/runner.py`
- Modify: `src/data_agent_langchain/observability/gateway_caps.py`
- Test: `tests/test_phase5_runner_cli.py`

- [ ] **Step 1: Write failing runner tests**

Append to `tests/test_phase5_runner_cli.py`:

```python
def test_run_single_task_core_json_action_does_not_require_gateway_caps(tmp_path: Path):
    _make_task(tmp_path, "task_1")
    cfg = _config(tmp_path, tmp_path / "runs")
    fake_llm = FakeListChatModel(responses=[_LIST_CTX, _READ_CSV, _ANSWER])

    result = run_single_task(
        task_id="task_1",
        config=cfg,
        run_output_dir=tmp_path / "runs" / "run1",
        llm=fake_llm,
        graph_mode="react",
    )

    assert result.succeeded is True


def test_run_single_task_core_tool_calling_requires_gateway_caps(tmp_path: Path):
    import pytest
    from dataclasses import replace

    from data_agent_langchain.exceptions import GatewayCapsMissingError

    _make_task(tmp_path, "task_1")
    cfg = _config(tmp_path, tmp_path / "runs")
    cfg = replace(cfg, agent=replace(cfg.agent, action_mode="tool_calling"))

    with pytest.raises(GatewayCapsMissingError):
        run_single_task(
            task_id="task_1",
            config=cfg,
            run_output_dir=tmp_path / "runs" / "run1",
            llm=FakeListChatModel(responses=[]),
            graph_mode="react",
        )
```

- [ ] **Step 2: Run tests to verify the second fails**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase5_runner_cli.py::test_run_single_task_core_tool_calling_requires_gateway_caps -q
```

Expected: FAIL because runner currently does not load caps before invoke.

- [ ] **Step 3: Implement startup check helper**

Modify `src/data_agent_langchain/run/runner.py` imports:

```python
from data_agent_langchain.llm.factory import bind_tools_for_gateway
from data_agent_langchain.observability.gateway_caps import GatewayCaps
from data_agent_langchain.tools.factory import create_all_tools
from data_agent_langchain.runtime.rehydrate import build_runtime
from data_agent_langchain.exceptions import ConfigError
```

Add helper near `_build_compiled_graph`:

```python
def _llm_for_action_mode(task: PublicTask, config: AppConfig, llm: Any | None) -> Any | None:
    if config.agent.action_mode != "tool_calling":
        return llm
    caps = GatewayCaps.from_yaml(config.observability.gateway_caps_path)
    if not caps.tool_calling:
        raise ConfigError(
            f"agent.action_mode='tool_calling' requires gateway tool_calling support; "
            f"caps file reports tool_calling=false: {config.observability.gateway_caps_path}"
        )
    resolved_llm = llm
    if resolved_llm is None:
        from data_agent_langchain.llm.factory import build_chat_model
        resolved_llm = build_chat_model(config)
    runtime = build_runtime(task, config)
    tools = create_all_tools(task, runtime)
    return bind_tools_for_gateway(resolved_llm, tools, caps)
```

Modify `_run_single_task_core`:

```python
    task = DABenchPublicDataset(config.dataset.root_path).get_task(task_id)
    resolved_llm = _llm_for_action_mode(task, config, llm)
    ...
    if resolved_llm is not None:
        runnable_config["configurable"] = {"llm": resolved_llm}
```

- [ ] **Step 4: Run runner caps tests**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase5_runner_cli.py::test_run_single_task_core_json_action_does_not_require_gateway_caps tests/test_phase5_runner_cli.py::test_run_single_task_core_tool_calling_requires_gateway_caps -q
```

Expected: PASS.

- [ ] **Step 5: Write binding test when caps exist**

Append:

```python
def test_run_single_task_core_tool_calling_binds_tools_with_caps(tmp_path: Path, monkeypatch):
    import yaml
    from dataclasses import replace

    import data_agent_langchain.run.runner as runner_module
    from data_agent_common.benchmark.schema import AnswerTable

    _make_task(tmp_path, "task_1")
    caps_path = tmp_path / "gateway_caps.yaml"
    caps_path.write_text(
        yaml.safe_dump({
            "gateway_caps": {
                "tool_calling": True,
                "parallel_tool_calls": False,
                "seed_param": False,
                "strict_mode": False,
            }
        }),
        encoding="utf-8",
    )
    cfg = _config(tmp_path, tmp_path / "runs")
    cfg = replace(
        cfg,
        agent=replace(cfg.agent, action_mode="tool_calling"),
        observability=replace(cfg.observability, gateway_caps_path=caps_path),
    )
    seen = {}

    class FakeCompiledGraph:
        def invoke(self, state, config):
            seen["llm"] = config["configurable"]["llm"]
            return {**state, "answer": AnswerTable(columns=["id"], rows=[[1]]), "failure_reason": None}

    class FakeToolCallingLLM:
        def bind_tools(self, tools, **kwargs):
            seen["tool_names"] = [tool.name for tool in tools]
            seen["bind_kwargs"] = kwargs
            return "bound-llm"

    monkeypatch.setattr(runner_module, "_build_compiled_graph", lambda mode: FakeCompiledGraph())

    runner_module._run_single_task_core(
        task_id="task_1",
        config=cfg,
        task_output_dir=tmp_path / "out" / "task_1",
        llm=FakeToolCallingLLM(),
        graph_mode="react",
    )

    assert seen["llm"] == "bound-llm"
    assert "answer" in seen["tool_names"]
    assert seen["bind_kwargs"] == {"parallel_tool_calls": False}
```

- [ ] **Step 6: Run binding test**

Expected: PASS.

- [ ] **Step 7: Update gateway caps error hint**

Modify `src/data_agent_langchain/observability/gateway_caps.py` line 38 from:

```python
"Run `dabench gateway-smoke --config <yaml>` (Phase 0.5) before "
```

to:

```python
"Run `dabench-lc gateway-smoke --config <yaml>` (Phase 0.5) before "
```

Run a targeted test that asserts this message if one exists; otherwise rely on the runner missing-caps test.

---

### Task 4: Documentation and migration status

**Files:**
- Modify: `src/LangChain/MIGRATION_STATUS.md`
- Modify: `src/LangChain/PHASE_05_GATEWAY_SMOKE_DESIGN.md` only if implementation deviates from design.

- [ ] **Step 1: Update migration status after code passes**

In `src/LangChain/MIGRATION_STATUS.md`, update:

- Phase 0.5 row from skipped to implemented smoke command if code is complete but real API not run.
- Next steps to say real gateway execution still requires credentials.
- Test matrix to include `tests/test_phase05_gateway_smoke.py` count.
- Verification line after full suite passes.

Use wording that distinguishes:

```markdown
| **0.5** gateway smoke test | §15 / Phase 0.5 | ✅ **命令已实现，真实网关待运行** | `dabench-lc gateway-smoke` writes `gateway_caps.yaml`; requires real API key to execute against gateway |
```

- [ ] **Step 2: Run docs grep sanity check**

Run:

```powershell
Select-String -Path "src\LangChain\MIGRATION_STATUS.md" -Pattern "待最终复跑|119 passed|dabench gateway-smoke|metrics.jsonl"
```

Expected: no stale matches except intentional historical references.

---

### Task 5: Final verification

**Files:**
- No new code changes unless tests fail.

- [ ] **Step 1: Run Phase 0.5 + Phase 5 targeted tests**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase05_gateway_smoke.py tests/test_phase5_runner_cli.py tests/test_phase5_config.py tests/test_phase5_llm_factory.py -q
```

Expected: all pass.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/ -q
```

Expected: all pass. Record the exact count and runtime in `MIGRATION_STATUS.md`.

- [ ] **Step 3: Review changed files**

Run:

```powershell
Get-ChildItem -Path src\data_agent_langchain\observability,src\data_agent_langchain\run,src\LangChain -File | Select-Object FullName
```

Expected: new `gateway_smoke.py`, design doc, and plan doc are present.

- [ ] **Step 4: Completion note**

Report:

- Implemented files.
- Test command and exact output.
- Whether real gateway smoke was executed. If not, state it remains pending because credentials were not provided.

---

## Self-review

- Spec coverage: CLI command, YAML output, credential fallback, gateway probes, runner check, non-goals, and tests are covered by Tasks 1-5.
- Placeholder scan: no unfinished placeholder markers are present.
- Type consistency: plan uses existing `AppConfig`, `GatewayCaps`, `bind_tools_for_gateway`, `run_single_task`, and `Typer` patterns.
- Scope check: default `tool_calling` switch and legacy cleanup are explicitly excluded from this plan, matching the approved Phase 0.5 prerequisite scope.
