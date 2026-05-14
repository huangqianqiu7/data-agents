"""RAG 启动期 import 边界回归测试（M4.6.2）。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_HEAVY_MODULES = ("torch", "chromadb", "sentence_transformers")


def _run_import_check(code: str) -> subprocess.CompletedProcess[str]:
    """在干净子进程中执行 import 边界断言。"""
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_import_package_does_not_load_heavy_rag_dependencies() -> None:
    """导入顶层包不应加载 RAG 重依赖。"""
    code = f"""
import data_agent_langchain
import sys
leaked = [name for name in {_HEAVY_MODULES!r} if name in sys.modules]
assert not leaked, leaked
"""
    result = _run_import_check(code)
    assert result.returncode == 0, result.stderr


def test_import_rag_package_does_not_load_heavy_dependencies() -> None:
    """导入 ``memory.rag`` 包不应加载 torch/chromadb/sentence-transformers。"""
    code = f"""
import data_agent_langchain.memory.rag
import sys
leaked = [name for name in {_HEAVY_MODULES!r} if name in sys.modules]
assert not leaked, leaked
"""
    result = _run_import_check(code)
    assert result.returncode == 0, result.stderr


def test_import_harrier_class_does_not_load_torch() -> None:
    """导入 HarrierEmbedder 类定义不应触发 torch 加载。"""
    code = """
from data_agent_langchain.memory.rag.embedders.sentence_transformer import HarrierEmbedder
import sys
assert "torch" not in sys.modules, "torch leaked at HarrierEmbedder import"
"""
    result = _run_import_check(code)
    assert result.returncode == 0, result.stderr

