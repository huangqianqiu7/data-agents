"""``slow`` marker 与 ``--runslow`` flag 配置自检（M4.0.2）。

直接通过 ``request.config`` 访问当前 pytest session 的配置，验证：

  - ``--runslow`` flag 已被 ``conftest.pytest_addoption`` 注册。
  - ``slow`` marker 已在 ``pyproject.toml`` 的 ``[tool.pytest.ini_options]``
    中声明（避免 ``PytestUnknownMarkWarning``）。

注：这里**不**用 ``pytester.runpytest`` —— 后者无论 in-process 还是 subprocess，
都会以 tmp_path 为 rootdir 启独立 session，主项目 ``conftest.py`` / ``pyproject.toml``
都不会被加载，因此无法验证我们的配置。直接读当前 session 状态更稳。

``pytest_collection_modifyitems`` 钩子的真实生效会在 M4.2.2
``test_phase10_rag_embedder_harrier.py`` 中天然验证：那里的 HarrierEmbedder
集成测试都标了 ``@pytest.mark.slow``，默认 phase 10 测试集运行时它们应被 skip。
"""
from __future__ import annotations

import pytest


def test_runslow_option_is_registered(request: pytest.FixtureRequest) -> None:
    """``conftest.pytest_addoption`` 应已注册 ``--runslow`` flag。"""
    # 未注册时 ``getoption`` 会抛 ``ValueError``；注册后返回 bool。
    val = request.config.getoption("--runslow")
    assert isinstance(val, bool), f"--runslow 应为 bool，实际为 {type(val).__name__}"


def test_slow_marker_is_registered_in_pyproject(request: pytest.FixtureRequest) -> None:
    """``pyproject.toml`` 应已声明 ``slow`` marker，避免未知 mark 警告。"""
    markers = request.config.getini("markers")
    assert any(m.startswith("slow:") for m in markers), (
        f"未在 pyproject.toml [tool.pytest.ini_options].markers 找到 slow，"
        f"当前 markers: {markers}"
    )
