"""Self-contained guard for ``data_agent_langchain``.

Asserts that nothing under ``src/data_agent_langchain/`` imports from
``data_agent_common`` / ``data_agent_refactored`` / ``data_agent_baseline``.

Rationale: the langchain backend must stand on its own so you can ``rm -rf``
any of the sibling packages and still have a working ``dabench-lc``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROHIBITED_PREFIXES = (
    "data_agent_common",
    "data_agent_refactored",
    "data_agent_baseline",
)

LANGCHAIN_SRC = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "data_agent_langchain"
)


def _collect_violations() -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for py_path in LANGCHAIN_SRC.rglob("*.py"):
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        except SyntaxError as exc:  # pragma: no cover — should not happen
            pytest.fail(f"Syntax error parsing {py_path}: {exc}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in PROHIBITED_PREFIXES
                ):
                    violations.append((py_path, node.lineno, f"from {module} import ..."))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == prefix or alias.name.startswith(prefix + ".")
                        for prefix in PROHIBITED_PREFIXES
                    ):
                        violations.append((py_path, node.lineno, f"import {alias.name}"))
    return violations


def test_langchain_package_has_no_external_sibling_imports() -> None:
    violations = _collect_violations()
    if violations:
        pretty = "\n".join(
            f"  {path.relative_to(LANGCHAIN_SRC.parent.parent)}:{lineno}: {stmt}"
            for path, lineno, stmt in violations
        )
        pytest.fail(
            "data_agent_langchain must not import from data_agent_common / "
            "data_agent_refactored / data_agent_baseline. Offenders:\n" + pretty
        )
