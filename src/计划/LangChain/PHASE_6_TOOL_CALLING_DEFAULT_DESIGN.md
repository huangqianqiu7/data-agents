# Phase 6 Tool Calling Default Design

## Scope

Switch the LangGraph backend default action mode from `json_action` to `tool_calling` now that real Phase 0.5 gateway smoke has passed.

This change applies only to `data_agent_langchain`. It does not modify the legacy `data_agent_refactored` backend or the legacy `dabench` CLI.

## Preconditions

The user manually ran real gateway smoke from the project root with the key-bearing config file. Cascade did not read or modify that config file.

Observed gateway capabilities:

```text
tool_calling: True
parallel_tool_calls: True
seed_param: True
strict_mode: True
```

Therefore the Phase 6 default action-mode switch is allowed.

## Design

### Configuration

Change `AgentConfig.action_mode` default from `json_action` to `tool_calling`.

Explicit YAML and programmatic configuration still override the default, so users can continue to set:

```yaml
agent:
  action_mode: json_action
```

when they need the legacy JSON action path.

### Runner behavior

Keep the existing runner safety checks:

- `tool_calling` requires `GatewayCaps.from_yaml(config.observability.gateway_caps_path)`.
- Missing caps file raises `GatewayCapsMissingError`.
- Caps with `tool_calling: false` raise `ConfigError`.
- Caps with `tool_calling: true` bind tools through `bind_tools_for_gateway`.
- `json_action` does not require `gateway_caps.yaml`.

No runner architecture change is needed.

### Tests

Update tests so default config assertions expect `tool_calling`.

Keep offline graph and runner tests deterministic by explicitly setting `action_mode="json_action"` where they use JSON string fake-model responses.

Add tests that prove:

- `AgentConfig().action_mode == "tool_calling"`.
- `default_app_config().agent.action_mode == "tool_calling"`.
- YAML can still explicitly set `json_action`.
- The runner default now requires gateway caps when not overridden.
- Explicit `json_action` remains the offline fallback path.

### Documentation

Update `MIGRATION_STATUS.md` to record:

- Real gateway smoke passed.
- Phase 6 default `tool_calling` switch is complete.
- Legacy cleanup and golden parity remain separate follow-up work.

## Non-goals

- Do not remove `json_action` support.
- Do not clean up legacy wrappers in this step.
- Do not modify legacy `data_agent_refactored` behavior.
- Do not modify legacy `dabench` CLI.
- Do not read or modify `configs/react_baseline.example.yaml`.
- Do not run commands that load `configs/react_baseline.example.yaml` unless the user explicitly performs them or grants separate permission.
