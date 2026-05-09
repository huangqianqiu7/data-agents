# Phase 6 Tool Calling Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the LangGraph backend default action mode from `json_action` to `tool_calling` while keeping explicit `json_action` fallback support.

**Architecture:** The default is changed only at the `AgentConfig` layer. Existing runner safety checks continue to require `gateway_caps.yaml` for `tool_calling`, and existing tests that use JSON fake-model responses remain explicit `json_action` tests. Documentation records that real gateway smoke passed and that legacy cleanup remains a separate follow-up.

**Tech Stack:** Python 3.13, dataclasses, PyYAML config loading, LangGraph runner, pytest, existing `GatewayCaps` and `bind_tools_for_gateway`.

---

## File map

- Modify: `src/data_agent_langchain/config.py`
  - Change `AgentConfig.action_mode` default to `tool_calling` and update comments.
- Modify: `tests/test_phase5_config.py`
  - Add default action-mode assertions and explicit YAML fallback assertion.
- Modify: `tests/test_phase5_runner_cli.py`
  - Add default runner caps requirement test; keep offline JSON tests explicit.
- Modify: `src/LangChain/MIGRATION_STATUS.md`
  - Mark real gateway smoke passed and Phase 6 default switch complete after verification.
- Read-only constraint: do not read or modify `configs/react_baseline.example.yaml`.

Use direct conda env Python:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest <tests> -q
```

---

### Task 1: Config default action mode

**Files:**
- Modify: `tests/test_phase5_config.py`
- Modify: `src/data_agent_langchain/config.py`

- [ ] **Step 1: Write failing default tests**

Append to `tests/test_phase5_config.py`:

```python

def test_agent_config_defaults_to_tool_calling_after_gateway_smoke():
    from data_agent_langchain.config import AgentConfig, default_app_config

    assert AgentConfig().action_mode == "tool_calling"
    assert default_app_config().agent.action_mode == "tool_calling"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase5_config.py::test_agent_config_defaults_to_tool_calling_after_gateway_smoke -q
```

Expected: FAIL because current default is `json_action`.

- [ ] **Step 3: Change default implementation**

Modify `src/data_agent_langchain/config.py`:

```python
    # ----- Action mode -----
    # Phase 0.5 gateway smoke passed for the configured evaluation gateway,
    # so Phase 6 defaults LangGraph runs to tool-calling. Explicit YAML can
    # still set ``json_action`` for legacy fallback / offline fake-model tests.
    action_mode: str = "tool_calling"
```

- [ ] **Step 4: Run default test to verify it passes**

Run the same targeted test. Expected: PASS.

- [ ] **Step 5: Add explicit YAML fallback test**

Append to `tests/test_phase5_config.py`:

```python

def test_load_app_config_can_explicitly_keep_json_action(tmp_path: Path):
    from data_agent_langchain.config import load_app_config

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "agent:\n  action_mode: json_action\n",
        encoding="utf-8",
    )

    cfg = load_app_config(cfg_path)

    assert cfg.agent.action_mode == "json_action"
```

- [ ] **Step 6: Run config tests**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase5_config.py -q
```

Expected: PASS.

---

### Task 2: Runner behavior under new default

**Files:**
- Modify: `tests/test_phase5_runner_cli.py`

- [ ] **Step 1: Write failing runner default test**

Append to `tests/test_phase5_runner_cli.py`:

```python

def test_run_single_task_default_tool_calling_requires_gateway_caps(tmp_path: Path):
    import pytest

    from data_agent_langchain.exceptions import GatewayCapsMissingError

    _make_task(tmp_path, "task_1")
    cfg = AppConfig(
        dataset=DatasetConfig(root_path=tmp_path),
        agent=AgentConfig(max_steps=8, max_model_retries=1, model_retry_backoff=(0.0,)),
        tools=ToolsConfig(),
        run=RunConfig(output_dir=tmp_path / "runs", run_id="run1", max_workers=1, task_timeout_seconds=0),
    )

    with pytest.raises(GatewayCapsMissingError):
        run_single_task(
            task_id="task_1",
            config=cfg,
            run_output_dir=tmp_path / "runs" / "run1",
            llm=FakeListChatModel(responses=[]),
            graph_mode="react",
        )
```

- [ ] **Step 2: Run test to verify behavior**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase5_runner_cli.py::test_run_single_task_default_tool_calling_requires_gateway_caps -q
```

Expected after Task 1 default switch: PASS. If it fails, inspect whether test helper `_config()` is forcing `json_action`; this test intentionally constructs `AppConfig` directly without forcing `action_mode`.

- [ ] **Step 3: Verify existing explicit JSON fallback runner test still passes**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase5_runner_cli.py::test_run_single_task_core_json_action_does_not_require_gateway_caps -q
```

Expected: PASS, because `_config()` explicitly sets `action_mode="json_action"`.

- [ ] **Step 4: Run full runner/CLI phase tests**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase5_runner_cli.py -q
```

Expected: PASS.

---

### Task 3: Documentation update

**Files:**
- Modify: `src/LangChain/MIGRATION_STATUS.md`

- [ ] **Step 1: Update status after code tests pass**

Update `MIGRATION_STATUS.md`:

- Header scope includes Phase 6 default switch.
- Top summary says real gateway smoke passed and default `tool_calling` switch is complete.
- Phase table row 6 becomes completed for default switch, while legacy cleanup remains follow-up.
- Phase 0.5 row says real gateway smoke passed with all four capabilities true.
- Current available features mention default `tool_calling` and explicit `json_action` fallback.
- Next steps say legacy cleanup and golden parity remain pending.
- Verification line should be updated only after full suite is run.

Use concrete wording:

```markdown
| **0.5** gateway smoke test | §15 / Phase 0.5 | ✅ **真实网关通过** | `tool_calling=true`, `parallel_tool_calls=true`, `seed_param=true`, `strict_mode=true` |
| **6** 清理 + 默认 tool_calling 切换 | §15 / Phase 6 | ✅ **默认切换完成；legacy 清理待后续** | `AgentConfig.action_mode` 默认 `tool_calling`；`json_action` 显式 fallback 保留 |
```

- [ ] **Step 2: Run stale text grep**

Run:

```powershell
Select-String -Path "src\LangChain\MIGRATION_STATUS.md" -Pattern "默认切换仍未开始|真实网关尚未运行|Phase 6 默认切换仍未开始|待最终复跑|143 passed|151 passed"
```

Expected: only current final verification line may mention `151 passed` until final run updates it.

---

### Task 4: Final verification

**Files:**
- No code changes unless tests fail.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/test_phase5_config.py tests/test_phase5_runner_cli.py tests/test_phase05_gateway_smoke.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full suite**

Run:

```powershell
& "C:\Users\18155\anaconda3\envs\dataagent\python.exe" -m pytest tests/ -q
```

Expected: PASS. Record exact count and runtime.

- [ ] **Step 3: Update final verification count**

Update `MIGRATION_STATUS.md` verification lines with the exact full-suite result.

- [ ] **Step 4: Completion report**

Report:

- `AgentConfig.action_mode` default is now `tool_calling`.
- `json_action` fallback remains explicit and tested.
- Full test command and exact output.
- No secret config was read or modified by Cascade.

---

## Self-review

- Spec coverage: default config switch, explicit fallback, runner safety behavior, documentation, and verification are covered.
- Placeholder scan: no unfinished placeholder markers are present.
- Scope check: legacy cleanup and golden parity are explicitly excluded.
- Type consistency: plan uses existing `AgentConfig`, `AppConfig`, `GatewayCapsMissingError`, `run_single_task`, and test helper patterns.
