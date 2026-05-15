# RAG Offline Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local RAG CLI runs default to HuggingFace offline/local-cache mode and lock the RAG container runtime offline invariants with tests and docs.

**Architecture:** Add a focused CLI helper that sets HuggingFace offline environment variables only when the final CLI config enables corpus RAG with an active memory mode. Keep `docker/Dockerfile.rag` as the build-time preload/runtime-offline path and add static tests to prevent regressions. Update README with the local offline default and container packaging behavior.

**Tech Stack:** Python 3.13, Typer CLI, pytest, Dockerfile static invariant tests, HuggingFace Hub environment variables.

---

## File Structure

- `src/data_agent_langchain/cli.py` — Add `_apply_hf_offline_defaults_for_rag` and call it from `run-task` and `run-benchmark` after memory overrides.
- `tests/test_phase9_cli_memory_mode.py` — Add CLI unit tests for offline env defaults and user override preservation.
- `tests/test_dockerfile_rag_invariants.py` — Add static tests for RAG runtime offline envs and `/opt/models/hf` copy.
- `README.md` — Document local default offline behavior and RAG container build/runtime cache behavior.

---

### Task 1: CLI RAG offline defaults

**Files:**
- Modify: `tests/test_phase9_cli_memory_mode.py`
- Modify: `src/data_agent_langchain/cli.py`

- [ ] **Step 1: Write failing tests**

Append these tests to `tests/test_phase9_cli_memory_mode.py`:

```python
def test_run_task_memory_rag_sets_hf_offline_defaults(tmp_path: Path, monkeypatch):
    import data_agent_langchain.cli as cli_module

    captured = {}
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(cli_module, "load_app_config", lambda path: default_app_config())
    monkeypatch.setattr(cli_module, "create_run_output_dir", lambda output_dir, run_id=None: ("run", tmp_path))

    def fake_run_single_task(*, task_id, config, run_output_dir, graph_mode, show_progress):
        captured["hf_hub_offline"] = __import__("os").environ.get("HF_HUB_OFFLINE")
        captured["transformers_offline"] = __import__("os").environ.get("TRANSFORMERS_OFFLINE")
        return SimpleNamespace(trace_path=tmp_path / "trace.json")

    monkeypatch.setattr(cli_module, "run_single_task", fake_run_single_task)

    result = CliRunner().invoke(
        app,
        [
            "run-task",
            "task_1",
            "--config",
            str(tmp_path / "config.yaml"),
            "--memory-mode",
            "read_only_dataset",
            "--memory-rag",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "hf_hub_offline": "1",
        "transformers_offline": "1",
    }


def test_run_benchmark_memory_rag_preserves_explicit_hf_offline_env(tmp_path: Path, monkeypatch):
    import data_agent_langchain.cli as cli_module

    captured = {}
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setattr(cli_module, "load_app_config", lambda path: default_app_config())

    def fake_run_benchmark(*, config, limit, graph_mode):
        captured["hf_hub_offline"] = __import__("os").environ.get("HF_HUB_OFFLINE")
        captured["transformers_offline"] = __import__("os").environ.get("TRANSFORMERS_OFFLINE")
        return tmp_path, []

    monkeypatch.setattr(cli_module, "run_benchmark", fake_run_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "run-benchmark",
            "--config",
            str(tmp_path / "config.yaml"),
            "--memory-mode",
            "read_only_dataset",
            "--memory-rag",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "hf_hub_offline": "0",
        "transformers_offline": "0",
    }


def test_run_benchmark_disabled_memory_does_not_set_hf_offline_env(tmp_path: Path, monkeypatch):
    import data_agent_langchain.cli as cli_module

    captured = {}
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(cli_module, "load_app_config", lambda path: default_app_config())

    def fake_run_benchmark(*, config, limit, graph_mode):
        captured["hf_hub_offline"] = __import__("os").environ.get("HF_HUB_OFFLINE")
        captured["transformers_offline"] = __import__("os").environ.get("TRANSFORMERS_OFFLINE")
        return tmp_path, []

    monkeypatch.setattr(cli_module, "run_benchmark", fake_run_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "run-benchmark",
            "--config",
            str(tmp_path / "config.yaml"),
            "--memory-mode",
            "disabled",
            "--memory-rag",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "hf_hub_offline": None,
        "transformers_offline": None,
    }
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
pytest tests/test_phase9_cli_memory_mode.py::test_run_task_memory_rag_sets_hf_offline_defaults tests/test_phase9_cli_memory_mode.py::test_run_benchmark_memory_rag_preserves_explicit_hf_offline_env tests/test_phase9_cli_memory_mode.py::test_run_benchmark_disabled_memory_does_not_set_hf_offline_env -q
```

Expected: first test fails because `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are not set by the CLI yet.

- [ ] **Step 3: Implement minimal CLI helper**

Modify `src/data_agent_langchain/cli.py`:

```python
import logging
import os
from contextlib import contextmanager
```

Add this helper after `_apply_memory_overrides`:

```python
def _apply_hf_offline_defaults_for_rag(cfg: AppConfig) -> None:
    """Default local corpus RAG runs to HuggingFace offline/local-cache mode."""
    if cfg.memory.mode not in {"read_only_dataset", "full"}:
        return
    if not cfg.memory.rag.enabled:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
```

Call it in `run_task_command` immediately after `_apply_memory_overrides`:

```python
    cfg = load_app_config(config)
    cfg = _apply_memory_overrides(cfg, memory_mode=memory_mode, memory_rag=memory_rag)
    _apply_hf_offline_defaults_for_rag(cfg)
```

Call it in `run_benchmark_command` immediately after `_apply_memory_overrides`:

```python
    cfg = load_app_config(config)

    cfg = _apply_memory_overrides(cfg, memory_mode=memory_mode, memory_rag=memory_rag)
    _apply_hf_offline_defaults_for_rag(cfg)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```powershell
pytest tests/test_phase9_cli_memory_mode.py -q
```

Expected: all tests in `test_phase9_cli_memory_mode.py` pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/data_agent_langchain/cli.py tests/test_phase9_cli_memory_mode.py
git commit -m "feat(cli): default rag runs to hf offline cache"
```

---

### Task 2: Dockerfile RAG offline invariant tests

**Files:**
- Modify: `tests/test_dockerfile_rag_invariants.py`

- [ ] **Step 1: Add static invariant tests**

Append these tests near the existing RAG dependency and Harrier weight tests in `tests/test_dockerfile_rag_invariants.py`:

```python
def test_dockerfile_rag_runtime_sets_huggingface_offline_envs(dockerfile_text: str) -> None:
    """runtime 阶段必须强制 HuggingFace / transformers 离线，避免评测节点联网。"""
    assert re.search(r"\bHF_HUB_OFFLINE\s*=\s*1\b", dockerfile_text), (
        "Dockerfile.rag runtime must set HF_HUB_OFFLINE=1 so huggingface_hub "
        "does not issue HEAD requests during evaluation"
    )
    assert re.search(r"\bTRANSFORMERS_OFFLINE\s*=\s*1\b", dockerfile_text), (
        "Dockerfile.rag runtime must set TRANSFORMERS_OFFLINE=1 so transformers "
        "loads only from the pre-baked cache during evaluation"
    )


def test_dockerfile_rag_copies_prebaked_hf_cache_to_runtime(dockerfile_text: str) -> None:
    """runtime 阶段必须 copy builder 中预烘的 HF cache。"""
    assert re.search(
        r"COPY\s+--from=builder\s+/opt/models/hf\s+/opt/models/hf",
        dockerfile_text,
    ), (
        "Dockerfile.rag must copy /opt/models/hf from builder into runtime so "
        "HF_HOME=/opt/models/hf has the preloaded Harrier cache"
    )
```

- [ ] **Step 2: Run Dockerfile invariant tests**

Run:

```powershell
pytest tests/test_dockerfile_rag_invariants.py -q
```

Expected: pass. These are regression tests for existing container behavior.

- [ ] **Step 3: Commit**

Run:

```powershell
git add tests/test_dockerfile_rag_invariants.py
git commit -m "test(docker): guard rag offline runtime cache"
```

---

### Task 3: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update local RAG command docs**

In the `开启 M4 per-task corpus RAG` subsection, keep the existing command and add this paragraph after the RAG mode note:

```markdown
本地 CLI 在最终配置启用 corpus RAG 且 `memory.mode` 为 `read_only_dataset` 或
`full` 时，会默认设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，
让 `sentence-transformers` 直接使用本机 HuggingFace 缓存，避免并发 benchmark
worker 反复请求 `huggingface.co`。如果首次运行本机尚无
`microsoft/harrier-oss-v1-270m` 缓存，可先临时设置 `HF_ENDPOINT=https://hf-mirror.com`
或取消离线变量后跑一次单任务预热缓存。
```

- [ ] **Step 2: Update full benchmark RAG command docs**

Ensure the benchmark section includes both commands:

```markdown
开启 M4 per-task corpus RAG 跑全量任务（默认带 `rich` 进度条）：

```bash
dabench-lc run-benchmark --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag
```

关闭进度条版本：

```bash
dabench-lc run-benchmark --config configs/local.yaml --graph-mode plan_solve --memory-mode read_only_dataset --memory-rag --no-progress
```
```

- [ ] **Step 3: Add RAG container packaging note**

Add this paragraph near the RAG Dockerfile or quick-start RAG discussion:

```markdown
RAG 提交镜像使用 `docker/Dockerfile.rag`：build 阶段会联网下载
`microsoft/harrier-oss-v1-270m` 到 `/opt/models/hf` 并打包进镜像；runtime 阶段
默认设置 `HF_HOME=/opt/models/hf`、`HF_HUB_OFFLINE=1` 与
`TRANSFORMERS_OFFLINE=1`，因此评测节点不需要访问 HuggingFace。
```

- [ ] **Step 4: Commit**

Run:

```powershell
git add README.md
git commit -m "docs: document rag offline cache defaults"
```

---

### Task 4: Final verification

**Files:**
- Verify only.

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
pytest tests/test_phase9_cli_memory_mode.py tests/test_dockerfile_rag_invariants.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Inspect final diff**

Run:

```powershell
git status --short
git log --oneline -4
git diff HEAD~3..HEAD -- src/data_agent_langchain/cli.py tests/test_phase9_cli_memory_mode.py tests/test_dockerfile_rag_invariants.py README.md
```

Expected: only intended CLI, tests, README, and plan/spec commits are present on the worktree branch.

---

## Self-Review

- Spec coverage: local CLI offline defaults are covered by Task 1; Docker runtime offline invariants are covered by Task 2; README documentation is covered by Task 3; verification is covered by Task 4.
- Placeholder scan: no task contains unresolved placeholders.
- Type consistency: helper accepts `AppConfig`, reads existing `cfg.memory.mode` and `cfg.memory.rag.enabled`, and only mutates `os.environ` before runner workers spawn.
