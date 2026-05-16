from pathlib import Path
import importlib.util
import sys


def test_legacy_archive_tests_are_ignored_when_deprecated_archive_is_missing(monkeypatch, tmp_path):
    package_prefixes = (
        "data_agent_langchain",
        "data_agent_common",
        "data_agent_refactored",
        "data_agent_baseline",
    )
    module_snapshot = {
        name: module
        for name, module in sys.modules.items()
        if name in package_prefixes or any(name.startswith(prefix + ".") for prefix in package_prefixes)
    }
    path_snapshot = list(sys.path)
    spec = importlib.util.spec_from_file_location("project_conftest", Path("tests/conftest.py"))
    conftest = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(conftest)

        monkeypatch.setattr(conftest, "_DEPRECATED", tmp_path / "missing")

        assert conftest.pytest_ignore_collect(Path("tests/test_phase1_imports.py"), None) is True
        assert conftest.pytest_ignore_collect(Path("tests/test_langchain_self_contained.py"), None) is None
    finally:
        for name in [
            key
            for key in list(sys.modules)
            if key in package_prefixes or any(key.startswith(prefix + ".") for prefix in package_prefixes)
        ]:
            del sys.modules[name]
        sys.modules.update(module_snapshot)
        sys.path[:] = path_snapshot
