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
"""
import sys
from pathlib import Path

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

_PACKAGE_PREFIXES = ("data_agent_langchain",
                     "data_agent_common",
                     "data_agent_refactored",
                     "data_agent_baseline")
for _name in [_n for _n in list(sys.modules)
              if _n in _PACKAGE_PREFIXES
              or any(_n.startswith(p + ".") for p in _PACKAGE_PREFIXES)]:
    del sys.modules[_name]