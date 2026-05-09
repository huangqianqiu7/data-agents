# Phase 0.5 Gateway Smoke Design

## Scope

Implement the Phase 0.5 gateway smoke test as a prerequisite for Phase 6. This work does not switch the default `AgentConfig.action_mode` to `tool_calling`. It adds a LangGraph-only smoke command and a runner startup check for explicit `tool_calling` runs.

## Approved approach

Use the conservative Phase 0.5 path:

- Add `dabench-lc gateway-smoke` only.
- Do not modify the legacy `dabench` CLI.
- Generate `artifacts/gateway_caps.yaml` from real gateway probes.
- Keep the default action mode as `json_action`.
- Require gateway caps only when a run explicitly uses `tool_calling`.

## Configuration and credentials

The command loads the existing `AppConfig` from `--config`.

Model connection values:

- `agent.model` from YAML.
- `agent.api_base` from YAML.
- `agent.api_key` from YAML when present.
- If `agent.api_key` is empty, fall back to `OPENAI_API_KEY`.

The command writes to `config.observability.gateway_caps_path` by default, with an optional `--output` override.

## Gateway probes

The smoke command probes these capabilities and writes a single YAML file:

```yaml
gateway_caps:
  tool_calling: true
  parallel_tool_calls: false
  seed_param: false
  strict_mode: false
```

Probe behavior:

- `tool_calling`: true only if a minimal bound-tool request succeeds and returns a tool call.
- `parallel_tool_calls`: true or false if the gateway accepts an explicit `parallel_tool_calls` request; null only if tool calling is unsupported or the probe is skipped.
- `seed_param`: true if a minimal request with `seed=42` succeeds.
- `strict_mode`: true if a minimal strict-tool schema request succeeds.

Probe failures must not crash the whole command unless the config itself is invalid. Individual probe failures are recorded as `false` or `null`.

## Runner startup check

When `config.agent.action_mode == "tool_calling"`, the LangGraph runner must load `GatewayCaps.from_yaml(config.observability.gateway_caps_path)` before invoking the graph.

Startup behavior:

- Missing caps file: raise `GatewayCapsMissingError`.
- `tool_calling: false`: raise `GatewayCapsMissingError` or `ConfigError` with a clear message.
- `tool_calling: true`: bind tools through `bind_tools_for_gateway` using the loaded caps.
- `json_action` runs do not require the caps file.

This preserves the existing offline/test path while preventing accidental `tool_calling` runs without a verified gateway.

## CLI design

Add a new Typer command:

```powershell
dabench-lc gateway-smoke --config path/to/config.yaml [--output artifacts/gateway_caps.yaml]
```

Output should include:

- The written caps path.
- The detected boolean/null capability values.
- A short reminder that default `json_action` remains unchanged until Phase 6 default-switch approval.

## Tests

Use TDD with mocked chat models and monkeypatching. No test should require network access.

Required tests:

- `gateway_smoke` writes a valid `gateway_caps.yaml` from successful fake probes.
- Failed tool-calling probe records `tool_calling: false` and skips parallel probe as `null`.
- API key falls back from YAML to `OPENAI_API_KEY`.
- `dabench-lc gateway-smoke` is registered in the Typer CLI.
- Runner in `json_action` mode does not require `gateway_caps.yaml`.
- Runner in `tool_calling` mode raises when caps are missing.
- Runner in `tool_calling` mode uses `bind_tools_for_gateway` when caps are present.

## Non-goals

- Do not change `AgentConfig.action_mode` default.
- Do not modify legacy `data_agent_refactored` behavior or `dabench` CLI.
- Do not implement Phase 6 cleanup or wrapper removal in this step.
- Do not require real API credentials in automated tests.

## Open follow-up

After this design is implemented and a real gateway smoke is run, Phase 6 can decide whether to switch the default action mode to `tool_calling`.
