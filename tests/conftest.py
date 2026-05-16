"""Pytest configuration: make ``src/`` importable without installing the package.

Worktree 隔离：本仓库在 git worktree 子目录跑测试时，``pip install -e`` 已
把主 worktree 的 ``src/`` 加到 ``sys.path`` 较高位，可能在 conftest 之前就
让 ``import data_agent_langchain`` 解析到主 worktree 副本。本 conftest 通过
两步守护当前 worktree 自带 ``src/`` 优先：

1. **强制 insert** 当前 worktree 的 ``src/`` 与 ``src/废弃/`` 到 ``sys.path[0]``
   （不再走 ``if _SRC not in sys.path`` 早退路径，避免误命中主 worktree
   src 的同名条目）。
2. 清掉 ``sys.modules`` 里所有 ``data_agent_*`` 缓存，让后续 ``import`` 走
   新的 ``sys.path`` 顺序。

v3 corpus RAG（M4.0.2）：另注册 ``--runslow`` flag 与 ``slow`` marker 的
collection 钩子。默认 ``@pytest.mark.slow`` 的用例被 skip；传 ``--runslow``
时才执行。适用于 HarrierEmbedder 等需要真权重 / 网络的集成测试。
"""
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

# Phase 8 之后，data_agent_common / data_agent_refactored / data_agent_baseline
# 三个 sibling 包被归档到 src/废弃/。它们不再是 langchain 的运行时依赖
# （由 tests/test_langchain_self_contained.py 强制保证），但 parity / 历史
# 回归测试仍会从这三个包读取 byte-for-byte 参考值，所以这里把归档目录也
# 加入 sys.path。
_DEPRECATED = _SRC / "废弃"
if _DEPRECATED.is_dir():
    sys.path.insert(0, str(_DEPRECATED))

_LEGACY_ARCHIVE_TEST_FILES = {
    "test_phase1_imports.py",
    "test_phase1_parity_constants.py",
    "test_phase25_working_memory.py",
    "test_phase2_descriptions_parity.py",
    "test_phase3_advance.py",
    "test_phase3_finalize.py",
    "test_phase3_gate.py",
    "test_phase4_planner_node.py",
    "test_phase5_runner_cli.py",
}


def pytest_ignore_collect(collection_path, config):
    if _DEPRECATED.is_dir():
        return None
    if Path(str(collection_path)).name in _LEGACY_ARCHIVE_TEST_FILES:
        return True
    return None

_PACKAGE_PREFIXES = ("data_agent_langchain",
                     "data_agent_common",
                     "data_agent_refactored",
                     "data_agent_baseline")
for _name in [_n for _n in list(sys.modules)
              if _n in _PACKAGE_PREFIXES
              or any(_n.startswith(p + ".") for p in _PACKAGE_PREFIXES)]:
    del sys.modules[_name]


# ---------------------------------------------------------------------------
# v3 corpus RAG（M4.0.2）：``slow`` marker + ``--runslow`` flag。
# - 默认 ``@pytest.mark.slow`` 用例被 skip。
# - 传 ``--runslow`` 时正常执行。
# marker 注册见 ``pyproject.toml`` 的 ``[tool.pytest.ini_options].markers``。
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """注册 ``--runslow`` flag。"""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="运行被 ``@pytest.mark.slow`` 标记的慢测试（默认 skip）。",
    )


def pytest_collection_modifyitems(config, items):
    """没有 ``--runslow`` 时，给所有 ``@slow`` 用例加 skip marker。"""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="需要 --runslow flag 才运行")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)