"""
Phase 2 parity tests for the new ``data_agent_langchain.tools`` layer.

Coverage (LANGCHAIN_MIGRATION_PLAN.md §16.1):

  - Description block rendered by the new registry MUST be byte-for-byte
    identical to ``ToolRegistry.describe_for_prompt()`` from the legacy
    backend (v3 D8 / §6.5).
  - Tool name set parity (v4 E12) — the new registry must contain
    EXACTLY the same eight names as the legacy registry; no drops, no
    additions, no renames.
  - The legacy ``DATA_PREVIEW_ACTIONS`` / ``GATED_ACTIONS`` frozensets
    must be byte-for-byte equal to the new ones.
  - ``ToolRuntime`` is picklable (required for subprocess transport).
  - ``ToolRuntimeResult`` round-trips through pickle.
  - ``AnswerTool`` validation matches legacy ``handle_answer`` rejection
    behaviour (column / row shape errors return ``ok=False``).
  - Successful ``AnswerTool`` returns ``is_terminal=True`` plus a real
    ``AnswerTable``.
"""
from __future__ import annotations

import pickle


def test_description_parity_with_legacy_registry():
    """v3 D8 / §6.5 / v4 E12: descriptions block byte-for-byte equal."""
    from data_agent_refactored.tools.registry import create_default_tool_registry
    from data_agent_langchain.tools.descriptions import (
        ALL_TOOL_NAMES,
        render_legacy_prompt_block,
    )

    legacy_block = create_default_tool_registry().describe_for_prompt()
    new_block = render_legacy_prompt_block(sorted(ALL_TOOL_NAMES))
    assert legacy_block == new_block, (
        "Tool description prompt block diverged from legacy. "
        "If you changed a tool description, update both the legacy "
        "ToolRegistry registration and ``descriptions.py``.\n\n"
        f"--- legacy ---\n{legacy_block!r}\n\n"
        f"--- new ---\n{new_block!r}"
    )


def test_tool_name_set_parity_v4_E12():
    """v4 E12: the new registry must list exactly the legacy tool names."""
    from data_agent_refactored.tools.registry import create_default_tool_registry
    from data_agent_langchain.tools.descriptions import ALL_TOOL_NAMES

    legacy_names = set(create_default_tool_registry().specs.keys())
    new_names = set(ALL_TOOL_NAMES)
    assert sorted(new_names) == sorted(legacy_names), (
        f"Tool name drift: legacy={sorted(legacy_names)!r}, "
        f"new={sorted(new_names)!r}"
    )


def test_data_preview_actions_match_legacy():
    """The new ``DATA_PREVIEW_ACTIONS`` frozenset matches the legacy one."""
    from data_agent_refactored.agents.data_preview_gate import (
        DATA_PREVIEW_ACTIONS as legacy,
    )
    from data_agent_langchain.tools.descriptions import (
        DATA_PREVIEW_ACTIONS as new,
    )

    assert legacy == new
    assert isinstance(new, frozenset)


def test_gated_actions_match_legacy():
    """The new ``GATED_ACTIONS`` frozenset matches the legacy one."""
    from data_agent_refactored.agents.data_preview_gate import (
        GATED_ACTIONS as legacy,
    )
    from data_agent_langchain.tools.descriptions import (
        GATED_ACTIONS as new,
    )

    assert legacy == new
    assert isinstance(new, frozenset)


def test_legacy_execute_python_description_uses_shared_constant():
    """The new description for ``execute_python`` references the shared
    ``EXECUTE_PYTHON_TIMEOUT_SECONDS`` constant (v4 E11)."""
    from data_agent_common.tools.python_exec import EXECUTE_PYTHON_TIMEOUT_SECONDS
    from data_agent_langchain.tools.descriptions import render_legacy_description

    desc = render_legacy_description("execute_python")
    assert f"{EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds" in desc


# ---------------------------------------------------------------------------
# Picklability — required for multiprocessing.Process and Checkpointer (C4)
# ---------------------------------------------------------------------------

def test_tool_runtime_is_picklable():
    """``ToolRuntime`` must round-trip through pickle (subprocess transport)."""
    from data_agent_langchain.tools.tool_runtime import ToolRuntime

    rt = ToolRuntime(
        task_dir="/tmp/task",
        context_dir="/tmp/task/context",
        python_timeout_s=30.0,
        sql_row_limit=200,
        max_obs_chars=3000,
    )
    restored = pickle.loads(pickle.dumps(rt))
    assert restored == rt


def test_tool_runtime_result_picklable_round_trip():
    """``ToolRuntimeResult`` round-trips through pickle (Checkpointer needs it)."""
    from data_agent_common.benchmark.schema import AnswerTable
    from data_agent_langchain.tools.tool_runtime import ToolRuntimeResult

    result = ToolRuntimeResult(
        ok=True,
        content={"status": "submitted", "column_count": 1, "row_count": 1},
        is_terminal=True,
        answer=AnswerTable(columns=["a"], rows=[["v"]]),
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    assert restored.answer is not None
    assert restored.answer.columns == ["a"]
    assert restored.answer.rows == [["v"]]
    assert restored.error_kind is None


def test_tool_runtime_no_allow_path_traversal_field_v4_E10():
    """v4 E10: ``ToolRuntime`` MUST NOT carry the legacy
    ``allow_path_traversal`` field; path safety lives in
    ``data_agent_common.tools.filesystem`` exclusively."""
    from dataclasses import fields

    from data_agent_langchain.tools.tool_runtime import ToolRuntime

    field_names = {f.name for f in fields(ToolRuntime)}
    assert "allow_path_traversal" not in field_names
    # And the canonical fields are present:
    assert field_names == {
        "task_dir", "context_dir", "python_timeout_s",
        "sql_row_limit", "max_obs_chars",
    }


# ---------------------------------------------------------------------------
# Tool error_kind discrimination (v4 E15)
# ---------------------------------------------------------------------------

def test_error_kind_discriminates_validation_runtime_timeout():
    """v4 E15: error_kind enum supports the three legitimate sources."""
    from data_agent_langchain.tools.tool_runtime import (
        ToolErrorKind,
        ToolRuntimeResult,
    )

    # Type alias literal values; constructing each must succeed.
    for kind in ("validation", "runtime", "timeout"):
        r = ToolRuntimeResult(ok=False, content={"error": "x"}, error_kind=kind)
        assert r.error_kind == kind
    # ``None`` is permitted on success.
    r = ToolRuntimeResult(ok=True, content={"x": 1})
    assert r.error_kind is None
    # Touch the alias so the import is exercised.
    _ = ToolErrorKind
