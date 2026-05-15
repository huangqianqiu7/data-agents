from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("tomllib requires Python 3.11+", allow_module_level=True)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _rag_extra_deps() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    specs = data["project"]["optional-dependencies"]["rag"]
    deps: set[str] = set()
    for spec in specs:
        name = re.split(r"[<>=!~;\s\[]", spec, maxsplit=1)[0].strip().lower()
        deps.add(name.replace("_", "-"))
    return deps


def test_rag_extra_declares_langchain_text_splitters() -> None:
    assert "langchain-text-splitters" in _rag_extra_deps()


def test_langchain_text_splitters_importable_when_rag_extra_installed() -> None:
    pytest.importorskip("langchain_text_splitters")

    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    assert MarkdownHeaderTextSplitter is not None
    assert RecursiveCharacterTextSplitter is not None
