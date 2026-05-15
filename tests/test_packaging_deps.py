"""
打包依赖回归测试 —— 防止"源码 import 了某个第三方包但 pyproject 没声明"。

v1 提交失败根因：``data_agent_langchain.agents.json_parser`` 在模块顶层
``import json_repair``，但 ``pyproject.toml`` 的 ``[project].dependencies``
漏掉了 ``json-repair``。官方容器构建只跑 ``pip install .``，因此镜像启动
时 ``ENTRYPOINT python -m data_agent_langchain.submission`` 立刻
``ModuleNotFoundError: No module named 'json_repair'``，整个 v1 评测得 0 分。

本测试在 unit-test 层把这一类"声明缺失"暴露出来：

- :func:`test_json_repair_is_declared_runtime_dependency`
    针对 v1 失败原因的最小直接回归。

- :func:`test_runtime_top_level_imports_are_declared`
    扫描 ``src/data_agent_langchain`` 下所有 ``.py`` 的**模块级**第三方
    import，要求每个名字要么是 stdlib、要么是本地包、要么已在
    ``pyproject.toml`` 中声明。模块级 import = 容器一启动就会被 Python
    执行的 import，所以一旦有遗漏就会重演 v1 事故。

    ``cli.py`` 例外：它只用于本地 ``dabench-lc`` 入口，里面对 ``rich`` /
    ``python-dotenv`` 都做了 ``try/except ImportError`` 容错；容器
    ENTRYPOINT 走 ``submission.py``，不会触发 ``cli.py``。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - dev/container 都 >= 3.13
    import pytest

    pytest.skip(
        "tomllib requires Python 3.11+; pyproject parsing skipped.",
        allow_module_level=True,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
RUNTIME_SRC_DIR = PROJECT_ROOT / "src" / "data_agent_langchain"

# cli.py 是本地开发入口；容器走 submission.py，所以 cli.py 的可选依赖
# (rich / python-dotenv) 不强制声明。这一份白名单越小越好——只要不是
# dev-only 入口，新文件就要受全量扫描覆盖。
DEV_ONLY_FILES: frozenset[Path] = frozenset(
    {
        RUNTIME_SRC_DIR / "cli.py",
    }
)

# `import X` 名字 → PyPI distribution 名（PEP 503 normalized）。
# pyproject 依赖里用 distribution 名，但源码里用 import 名，需要这个映射
# 才能对得上。任何未在此表里的名字按 ``name.replace("_", "-")`` fallback。
IMPORT_TO_DISTRIBUTION: dict[str, str] = {
    "yaml": "pyyaml",
    "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai",
    "json_repair": "json-repair",
    "dotenv": "python-dotenv",
}


def _declared_runtime_deps() -> set[str]:
    """读取 ``pyproject.toml`` 的 ``[project].dependencies``，返回归一化包名集合。"""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    out: set[str] = set()
    for spec in data["project"]["dependencies"]:
        # 截到第一个版本/标记/extra 字符之前，剩下就是包名
        name = re.split(r"[<>=!~;\s\[]", spec, maxsplit=1)[0].strip().lower()
        out.add(name.replace("_", "-"))
    return out


def _module_level_imports(path: Path) -> set[str]:
    """返回 *path* 的**模块级**顶层 import 名（不含函数/类体内 lazy import）。

    模块级 import = ``ast.parse(...).body`` 中直接出现的 ``Import`` /
    ``ImportFrom``。容器一启动 Python 就会执行它们；漏装即崩溃。
    函数体内的 ``import`` 不计入：那些通常是带 ``try/except ImportError``
    的可选 dev 依赖（例如 cli.py 里的 rich / dotenv）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def test_json_repair_is_declared_runtime_dependency() -> None:
    """v1 失败的直接回归：``json-repair`` 必须出现在 pyproject 运行时依赖里。

    没有这条声明，官方 ``pip install .`` 不会安装 ``json_repair``，
    容器一启动就会在 ``json_parser.py:16`` 处崩溃。
    """
    deps = _declared_runtime_deps()
    assert "json-repair" in deps, (
        "json-repair must appear in pyproject [project].dependencies. "
        "data_agent_langchain.agents.json_parser imports json_repair at "
        "module top level; without this declaration the official container "
        "build (which runs `pip install .`) produces an image that fails "
        "with ModuleNotFoundError on ENTRYPOINT."
    )


def test_project_readme_points_to_root_readme() -> None:
    """The packaged README should be the maintained root README."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    readme = data["project"].get("readme")
    expected = "README.md"
    assert readme == expected, (
        f"pyproject [project].readme should point to {expected!r}; got {readme!r}."
    )
    assert (PROJECT_ROOT / expected).is_file(), (
        f"pyproject [project].readme target does not exist: {expected}"
    )


def test_runtime_top_level_imports_are_declared() -> None:
    """全量扫描：``src/data_agent_langchain`` 顶层 import 都已声明或属于 stdlib/本地包。

    把同类"漏依赖"在 CI 阶段拦下来，而不是等容器评测打回。
    """
    declared = _declared_runtime_deps()
    stdlib = set(sys.stdlib_module_names)
    local_pkg = "data_agent_langchain"

    undeclared: dict[str, set[str]] = {}
    for py in sorted(RUNTIME_SRC_DIR.rglob("*.py")):
        if py in DEV_ONLY_FILES:
            continue
        for imp in _module_level_imports(py):
            if imp in stdlib or imp == local_pkg:
                continue
            dist = IMPORT_TO_DISTRIBUTION.get(imp, imp.replace("_", "-"))
            if dist not in declared:
                rel = py.relative_to(PROJECT_ROOT).as_posix()
                undeclared.setdefault(rel, set()).add(f"{imp} -> {dist}")
    assert not undeclared, (
        "These runtime source files have top-level imports that are NOT "
        "declared in pyproject [project].dependencies:\n"
        + "\n".join(
            f"  {rel}: {sorted(missing)}" for rel, missing in sorted(undeclared.items())
        )
        + "\nAdd the missing distribution(s) to pyproject.toml or move the "
          "import inside a function with try/except ImportError fallback."
    )
