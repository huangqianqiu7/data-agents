"""Pytest configuration: make ``src/`` importable without installing the package."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Phase 8 之后，data_agent_common / data_agent_refactored / data_agent_baseline
# 三个 sibling 包被归档到 src/废弃/。它们不再是 langchain 的运行时依赖
# （由 tests/test_langchain_self_contained.py 强制保证），但 parity / 历史
# 回归测试仍会从这三个包读取 byte-for-byte 参考值，所以这里把归档目录也
# 加入 sys.path。
_DEPRECATED = _SRC / "废弃"
if _DEPRECATED.is_dir() and str(_DEPRECATED) not in sys.path:
    sys.path.insert(0, str(_DEPRECATED))