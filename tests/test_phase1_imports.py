"""
Phase 1 import sanity tests.

Verifies that:
  - The new ``data_agent_common`` package exposes every symbol both
    backends rely on.
  - The legacy ``data_agent_refactored`` import paths still resolve
    after being converted to thin re-exports (no behaviour change for
    existing call sites).
  - The new ``data_agent_langchain`` skeleton can be imported (it does
    not yet depend on langchain / langgraph in Phase 1).

These tests are the first layer of "verification before completion" for
LANGCHAIN_MIGRATION_PLAN.md Phase 1 (§15).
"""
from __future__ import annotations


def test_data_agent_common_top_level_imports():
    """All canonical symbols live under ``data_agent_common`` and import."""
    from data_agent_common.benchmark.schema import (
        AnswerTable,
        PublicTask,
        TaskAssets,
        TaskRecord,
    )
    from data_agent_common.benchmark.dataset import DABenchPublicDataset
    from data_agent_common.exceptions import (
        AgentError,
        ConfigError,
        ContextAssetNotFoundError,
        ContextPathEscapeError,
        DataAgentError,
        DatasetError,
        InvalidRunIdError,
        ModelCallError,
        ModelResponseParseError,
        ReadOnlySQLViolationError,
        RunnerError,
        TaskNotFoundError,
        ToolError,
        ToolTimeoutError,
        ToolValidationError,
        UnknownToolError,
    )
    from data_agent_common.tools import (
        EXECUTE_PYTHON_TIMEOUT_SECONDS,
        execute_python_code,
        execute_read_only_sql,
        inspect_sqlite_schema,
        list_context_tree,
        read_csv_preview,
        read_doc_preview,
        read_json_preview,
        resolve_context_path,
    )
    from data_agent_common.agents.runtime import (
        AgentRunResult,
        AgentRuntimeState,
        ModelMessage,
        ModelStep,
        StepRecord,
    )
    from data_agent_common.agents.json_parser import parse_model_step, parse_plan
    from data_agent_common.agents.sanitize import (
        ERROR_SANITIZED_RESPONSE,
        SANITIZE_ACTIONS,
    )
    from data_agent_common.constants import FALLBACK_STEP_PROMPT

    # Smoke: types are not None / are actually classes.
    assert isinstance(EXECUTE_PYTHON_TIMEOUT_SECONDS, int)
    assert isinstance(FALLBACK_STEP_PROMPT, str)
    assert isinstance(SANITIZE_ACTIONS, frozenset)
    assert isinstance(ERROR_SANITIZED_RESPONSE, str)
    # Touch the imported names so linters do not flag them as unused.
    _ = (
        AnswerTable, PublicTask, TaskAssets, TaskRecord, DABenchPublicDataset,
        AgentError, ConfigError, ContextAssetNotFoundError,
        ContextPathEscapeError, DataAgentError, DatasetError,
        InvalidRunIdError, ModelCallError, ModelResponseParseError,
        ReadOnlySQLViolationError, RunnerError, TaskNotFoundError,
        ToolError, ToolTimeoutError, ToolValidationError, UnknownToolError,
        execute_python_code, execute_read_only_sql, inspect_sqlite_schema,
        list_context_tree, read_csv_preview, read_doc_preview,
        read_json_preview, resolve_context_path,
        AgentRunResult, AgentRuntimeState, ModelMessage, ModelStep,
        StepRecord, parse_model_step, parse_plan,
    )


def test_legacy_refactored_paths_still_resolve_via_reexport():
    """Existing call sites that import from ``data_agent_refactored.*`` keep working."""
    from data_agent_refactored.exceptions import DataAgentError, ToolValidationError
    from data_agent_refactored.benchmark.schema import AnswerTable, PublicTask
    from data_agent_refactored.benchmark.dataset import DABenchPublicDataset
    from data_agent_refactored.tools.filesystem import resolve_context_path
    from data_agent_refactored.tools.python_exec import (
        EXECUTE_PYTHON_TIMEOUT_SECONDS,
        execute_python_code,
    )
    from data_agent_refactored.tools.sqlite import execute_read_only_sql
    from data_agent_refactored.agents.runtime import (
        AgentRunResult,
        StepRecord,
    )
    from data_agent_refactored.agents.json_parser import parse_model_step, parse_plan
    from data_agent_refactored.agents.model import (
        ModelMessage,
        ModelStep,
        OpenAIModelAdapter,
    )
    from data_agent_refactored.agents.context_manager import (
        ERROR_SANITIZED_RESPONSE,
        build_history_messages,
    )
    from data_agent_refactored.agents.plan_solve_agent import PlanAndSolveAgent
    from data_agent_refactored.agents.react_agent import ReActAgent

    # Smoke: classes are class objects.
    assert OpenAIModelAdapter.__name__ == "OpenAIModelAdapter"
    assert PlanAndSolveAgent.__name__ == "PlanAndSolveAgent"
    assert ReActAgent.__name__ == "ReActAgent"
    _ = (
        DataAgentError, ToolValidationError, AnswerTable, PublicTask,
        DABenchPublicDataset, resolve_context_path,
        EXECUTE_PYTHON_TIMEOUT_SECONDS, execute_python_code,
        execute_read_only_sql, AgentRunResult, StepRecord,
        parse_model_step, parse_plan, ModelMessage, ModelStep,
        ERROR_SANITIZED_RESPONSE, build_history_messages,
    )


def test_legacy_and_common_classes_are_identical_objects():
    """Re-exports must point at the same class objects, not copies.

    If they were duplicated, ``isinstance`` checks across the two import
    paths would silently fail and parity tests would mis-classify objects.
    """
    import data_agent_common.benchmark.schema as common_schema
    import data_agent_refactored.benchmark.schema as legacy_schema

    assert common_schema.AnswerTable is legacy_schema.AnswerTable
    assert common_schema.PublicTask is legacy_schema.PublicTask

    import data_agent_common.agents.runtime as common_runtime
    import data_agent_refactored.agents.runtime as legacy_runtime

    assert common_runtime.StepRecord is legacy_runtime.StepRecord
    assert common_runtime.AgentRunResult is legacy_runtime.AgentRunResult

    import data_agent_common.exceptions as common_exc
    import data_agent_refactored.exceptions as legacy_exc

    assert common_exc.DataAgentError is legacy_exc.DataAgentError
    assert common_exc.ToolValidationError is legacy_exc.ToolValidationError


def test_data_agent_langchain_skeleton_imports():
    """The new package skeleton imports without requiring langchain / langgraph."""
    import data_agent_langchain
    from data_agent_langchain.runtime.state import (
        ActionMode,
        AgentMode,
        GateDecision,
        LastErrorKind,
        RunState,
        SubgraphExit,
    )
    from data_agent_langchain.runtime.context import (
        get_current_app_config,
        set_current_app_config,
    )
    from data_agent_langchain.agents.runtime import (
        AgentRunResult,
        ModelMessage,
        ModelStep,
        StepRecord,
    )
    from data_agent_langchain.observability.gateway_caps import GatewayCaps
    from data_agent_langchain.observability.tracer import build_callbacks
    from data_agent_langchain.observability.metrics import MetricsCollector
    from data_agent_langchain.exceptions import (
        GatewayCapsMissingError,
        ReproducibilityViolationError,
    )

    assert data_agent_langchain.__version__ == "0.1.0"
    _ = (
        ActionMode, AgentMode, GateDecision, LastErrorKind, RunState,
        SubgraphExit, get_current_app_config, set_current_app_config,
        AgentRunResult, ModelMessage, ModelStep, StepRecord, GatewayCaps,
        build_callbacks, MetricsCollector, GatewayCapsMissingError,
        ReproducibilityViolationError,
    )
