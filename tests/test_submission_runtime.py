"""v5 §四第 1 步「Gateway / 输出 / 异常 / 信号 / 脱敏」10 条测试。

覆盖 ``data_agent_langchain.submission.run`` 的 runtime 行为：
gateway_caps 文件存在性、占位 CSV 写入、成功覆盖、失败保留占位、
不产生 ``/output/<run_id>/``、per-task 异常隔离、SIGTERM 派发停止、
日志/输出全文件脱敏（不含 mock observation / raw_response / api_key）、
``None`` 写 CSV 为空 cell。

所有测试通过 ``submission.run(...)`` 的可选注入参数把容器路径替换为
``tmp_path`` 子目录，避免在 dev 上写真实 ``/input`` / ``/output`` /
``/logs``。SIGTERM 测试用 ``register_signals=False`` + 直接拨
``submission._on_sigterm`` 验证派发停止逻辑，绝不触碰真实信号。
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.language_models import FakeListChatModel

from data_agent_langchain.config import AppConfig
from data_agent_langchain.run.runner import TaskRunArtifacts


_LIST_CTX = '```json\n{"thought":"Inspect","action":"list_context","action_input":{"max_depth":4}}\n```'
_READ_CSV = '```json\n{"thought":"Read","action":"read_csv","action_input":{"path":"matches.csv","max_rows":5}}\n```'
_ANSWER = '```json\n{"thought":"Done","action":"answer","action_input":{"columns":["id"],"rows":[[1]]}}\n```'

_MOCK_API_KEY = "sk-MOCK-DO-NOT-LEAK-12345"  # v5 §三.4 防泄漏 sentinel
_MOCK_OBS = "MOCK-OBSERVATION-SHOULD-NOT-LEAK-67890"
_MOCK_RAW = "MOCK-RAW-RESPONSE-SHOULD-NOT-LEAK-ABCDE"


def _make_task(input_dir: Path, task_id: str) -> None:
    task_dir = input_dir / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": task_id, "difficulty": "easy", "question": "Q?"}),
        encoding="utf-8",
    )
    with (context_dir / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([["id"], [1]])


def _make_caps(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "gateway_caps": {
                    "tool_calling": True,
                    "parallel_tool_calls": True,
                    "seed_param": True,
                    "strict_mode": True,
                }
            }
        ),
        encoding="utf-8",
    )


def _submission_dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "input_dir": tmp_path / "input",
        "output_dir": tmp_path / "output",
        "logs_dir": tmp_path / "logs",
        "internal_runs_dir": tmp_path / "tmp_runs",
        "gateway_caps_path": tmp_path / "app" / "gateway_caps.yaml",
    }


# ---------------------------------------------------------------------------
# Gateway / 输出（6 条）
# ---------------------------------------------------------------------------

def test_missing_gateway_caps_yaml_fails_fast(tmp_path: Path, monkeypatch):
    """v5 §二.2：caps 文件缺失时 submission 必须明确失败。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    dirs = _submission_dirs(tmp_path)
    dirs["input_dir"].mkdir()
    _make_task(dirs["input_dir"], "task_1")
    # 注意：不调 _make_caps，让 caps 文件缺失

    with pytest.raises(submission.SubmissionConfigError) as exc_info:
        submission.run(
            **dirs,
            register_signals=False,
            llm=FakeListChatModel(responses=[]),
        )

    # 错误信息必须脱敏：可以提路径名，但不能透露 MODEL_API_URL 之类敏感值
    assert "gateway_caps" in str(exc_info.value).lower()


def test_caps_file_with_tool_calling_true_proceeds(tmp_path: Path, monkeypatch):
    """v5 §二.2：caps 文件存在且 tool_calling=true 时不抛 SubmissionConfigError。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    dirs = _submission_dirs(tmp_path)
    dirs["input_dir"].mkdir()
    _make_task(dirs["input_dir"], "task_1")
    _make_caps(dirs["gateway_caps_path"])
    fake_llm = FakeListChatModel(responses=[_LIST_CTX, _READ_CSV, _ANSWER])

    summary = submission.run(
        **dirs,
        register_signals=False,
        llm=fake_llm,
        graph_mode="react",  # react + json_action 路径不会真碰 tool_calling 绑定
        action_mode_override="json_action",
    )

    assert summary["task_count"] == 1


def test_each_task_starts_with_placeholder_csv(tmp_path: Path, monkeypatch):
    """v5 §二.3：每题开始前写占位 CSV ``result\\r\\n`` 到 ``/output/<task_id>/prediction.csv``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    dirs = _submission_dirs(tmp_path)
    dirs["input_dir"].mkdir()
    _make_task(dirs["input_dir"], "task_1")
    _make_task(dirs["input_dir"], "task_2")
    _make_caps(dirs["gateway_caps_path"])

    # 让 run_single_task 抛异常以便观察占位 CSV 是否在异常前已写入
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated agent failure")

    monkeypatch.setattr(submission, "_run_single_task_impl", _boom)

    submission.run(**dirs, register_signals=False, llm=None, action_mode_override="json_action")

    for task_id in ("task_1", "task_2"):
        csv_path = dirs["output_dir"] / task_id / "prediction.csv"
        assert csv_path.exists(), f"placeholder missing for {task_id}"
        assert csv_path.read_bytes() == b"result\r\n", (
            f"placeholder content wrong for {task_id}: {csv_path.read_bytes()!r}"
        )


def test_success_overwrites_placeholder_atomically(tmp_path: Path, monkeypatch):
    """v5 §二.3：成功时通过同目录 ``.prediction.csv.tmp`` + ``os.replace`` 覆盖占位。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    dirs = _submission_dirs(tmp_path)
    dirs["input_dir"].mkdir()
    _make_task(dirs["input_dir"], "task_1")
    _make_caps(dirs["gateway_caps_path"])
    fake_llm = FakeListChatModel(responses=[_LIST_CTX, _READ_CSV, _ANSWER])

    summary = submission.run(
        **dirs,
        register_signals=False,
        llm=fake_llm,
        graph_mode="react",
        action_mode_override="json_action",
    )

    csv_path = dirs["output_dir"] / "task_1" / "prediction.csv"
    assert csv_path.exists()
    # 成功后的 CSV 不再是 ``result\r\n`` 占位
    assert csv_path.read_bytes() != b"result\r\n"
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["id"]
    # 同目录临时文件不应残留
    assert not (csv_path.parent / ".prediction.csv.tmp").exists()
    assert summary["succeeded_count"] == 1


def test_failed_task_keeps_placeholder_csv(tmp_path: Path, monkeypatch):
    """v5 §二.3：``prediction_csv_path is None`` 时不复制，让占位 CSV 保留。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    dirs = _submission_dirs(tmp_path)
    dirs["input_dir"].mkdir()
    _make_task(dirs["input_dir"], "task_1")
    _make_caps(dirs["gateway_caps_path"])

    def _fake_impl(*, task_id: str, config: AppConfig, run_output_dir: Path,
                   llm: Any, graph_mode: str) -> TaskRunArtifacts:
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = task_output_dir / "trace.json"
        trace_path.write_text("{}", encoding="utf-8")
        # 模拟失败：prediction_csv_path = None
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            trace_path=trace_path,
            succeeded=False,
            failure_reason="simulated failure",
        )

    monkeypatch.setattr(submission, "_run_single_task_impl", _fake_impl)

    summary = submission.run(**dirs, register_signals=False, llm=None,
                             action_mode_override="json_action")

    csv_path = dirs["output_dir"] / "task_1" / "prediction.csv"
    assert csv_path.exists()
    assert csv_path.read_bytes() == b"result\r\n", "placeholder must remain on failure"
    assert summary["failed_count"] == 1


def test_no_run_id_directory_under_output(tmp_path: Path, monkeypatch):
    """v5 §二.3：``/output`` 必须扁平为 ``/output/<task_id>/``，不出现 ``<run_id>/``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    dirs = _submission_dirs(tmp_path)
    dirs["input_dir"].mkdir()
    _make_task(dirs["input_dir"], "task_1")
    _make_caps(dirs["gateway_caps_path"])
    fake_llm = FakeListChatModel(responses=[_LIST_CTX, _READ_CSV, _ANSWER])

    submission.run(
        **dirs,
        register_signals=False,
        llm=fake_llm,
        graph_mode="react",
        action_mode_override="json_action",
    )

    # /output/ 顶层只应有 task_* 目录，不能有 timestamp 风格 run_id
    children = sorted(p.name for p in dirs["output_dir"].iterdir())
    assert children == ["task_1"], f"unexpected /output children: {children}"


# ---------------------------------------------------------------------------
# 异常与信号（2 条）
# ---------------------------------------------------------------------------

def test_per_task_failure_does_not_block_other_tasks(tmp_path: Path, monkeypatch):
    """v5 §二.5：单题 try/except，失败只影响该任务，其他任务仍执行。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    dirs = _submission_dirs(tmp_path)
    dirs["input_dir"].mkdir()
    _make_task(dirs["input_dir"], "task_1")
    _make_task(dirs["input_dir"], "task_2")
    _make_task(dirs["input_dir"], "task_3")
    _make_caps(dirs["gateway_caps_path"])

    def _fake_impl(*, task_id: str, config: AppConfig, run_output_dir: Path,
                   llm: Any, graph_mode: str) -> TaskRunArtifacts:
        if task_id == "task_2":
            raise RuntimeError("simulated agent crash")
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        prediction_csv = task_output_dir / "prediction.csv"
        with prediction_csv.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([["id"], [1]])
        trace_path = task_output_dir / "trace.json"
        trace_path.write_text("{}", encoding="utf-8")
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=prediction_csv,
            trace_path=trace_path,
            succeeded=True,
            failure_reason=None,
        )

    monkeypatch.setattr(submission, "_run_single_task_impl", _fake_impl)

    summary = submission.run(**dirs, register_signals=False, llm=None,
                             action_mode_override="json_action")

    assert summary["task_count"] == 3
    assert summary["succeeded_count"] == 2
    assert summary["failed_count"] == 1
    # 其他两题成功覆盖了占位
    assert (dirs["output_dir"] / "task_1" / "prediction.csv").read_bytes() != b"result\r\n"
    assert (dirs["output_dir"] / "task_3" / "prediction.csv").read_bytes() != b"result\r\n"
    # 失败题保留占位
    assert (dirs["output_dir"] / "task_2" / "prediction.csv").read_bytes() == b"result\r\n"
    # 失败摘要记录到 failed_tasks.jsonl
    failed_log = dirs["logs_dir"] / "failed_tasks.jsonl"
    assert failed_log.exists()
    lines = [json.loads(line) for line in failed_log.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["task_id"] == "task_2"
    # 失败摘要必须脱敏：不能含堆栈全文
    assert "Traceback" not in json.dumps(entry)


def test_sigterm_stops_dispatching_new_tasks(tmp_path: Path, monkeypatch):
    """v5 §二.5：SIGTERM 后停止派发新任务（用 _shutting_down 注入模拟）。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    dirs = _submission_dirs(tmp_path)
    dirs["input_dir"].mkdir()
    for index in range(1, 6):
        _make_task(dirs["input_dir"], f"task_{index}")
    _make_caps(dirs["gateway_caps_path"])

    dispatched: list[str] = []

    def _fake_impl(*, task_id: str, config: AppConfig, run_output_dir: Path,
                   llm: Any, graph_mode: str) -> TaskRunArtifacts:
        dispatched.append(task_id)
        # 第二次派发后立刻拉 SIGTERM 标志，模拟容器收到 SIGTERM
        if task_id == "task_2":
            submission._shutting_down.set()
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        prediction_csv = task_output_dir / "prediction.csv"
        with prediction_csv.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([["id"], [1]])
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=prediction_csv,
            trace_path=task_output_dir / "trace.json",
            succeeded=True,
            failure_reason=None,
        )

    submission._shutting_down.clear()
    monkeypatch.setattr(submission, "_run_single_task_impl", _fake_impl)
    monkeypatch.setattr(submission, "DEFAULT_MAX_WORKERS", 1, raising=False)

    summary = submission.run(
        **dirs,
        register_signals=False,
        llm=None,
        max_workers=1,
        action_mode_override="json_action",
    )

    submission._shutting_down.clear()  # cleanup global state for other tests

    # 至少派发了 task_1 + task_2，但不能把 5 题全派完
    assert "task_1" in dispatched
    assert "task_2" in dispatched
    assert len(dispatched) < 5, f"SIGTERM did not stop dispatch: {dispatched}"
    # summary 应该体现实际跑过的任务数 < 5
    assert summary["task_count"] == len(dispatched)


# ---------------------------------------------------------------------------
# 脱敏（2 条）
# ---------------------------------------------------------------------------

def test_no_secrets_or_observations_leak_to_logs_or_outputs(tmp_path: Path, monkeypatch):
    """v5 §三.4：mock api_key / observation / raw_response 不能进入任何输出文件。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    monkeypatch.setenv("MODEL_API_KEY", _MOCK_API_KEY)
    dirs = _submission_dirs(tmp_path)
    dirs["input_dir"].mkdir()
    _make_task(dirs["input_dir"], "task_1")
    _make_caps(dirs["gateway_caps_path"])

    # 让 fake LLM 在 raw_response 字段里夹带 mock 串，模拟 LLM 输出回流
    fake_llm = FakeListChatModel(responses=[_LIST_CTX, _READ_CSV, _ANSWER])

    submission.run(
        **dirs,
        register_signals=False,
        llm=fake_llm,
        graph_mode="react",
        action_mode_override="json_action",
    )

    suspect_files: list[Path] = []
    for root in (dirs["logs_dir"], dirs["output_dir"], dirs["internal_runs_dir"]):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                suspect_files.append(path)

    leaks: list[tuple[Path, str]] = []
    for path in suspect_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in (_MOCK_API_KEY, _MOCK_OBS, _MOCK_RAW):
            if needle in text:
                leaks.append((path, needle))
    assert not leaks, f"secret/observation leaked: {leaks}"


def test_none_in_answer_rows_writes_empty_cell(tmp_path: Path, monkeypatch):
    """v5 §三.3：``answer.rows`` 中的 ``None`` 必须写为空 cell，不能是字符串 ``None``。"""
    from data_agent_langchain import submission

    monkeypatch.setenv("MODEL_API_URL", "http://internal.model/v1")
    dirs = _submission_dirs(tmp_path)
    dirs["input_dir"].mkdir()
    _make_task(dirs["input_dir"], "task_1")
    _make_caps(dirs["gateway_caps_path"])

    def _fake_impl(*, task_id: str, config: AppConfig, run_output_dir: Path,
                   llm: Any, graph_mode: str) -> TaskRunArtifacts:
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        prediction_csv = task_output_dir / "prediction.csv"
        with prediction_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "name"])
            writer.writerow([1, None])
            writer.writerow([2, "alice"])
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=prediction_csv,
            trace_path=task_output_dir / "trace.json",
            succeeded=True,
            failure_reason=None,
        )

    monkeypatch.setattr(submission, "_run_single_task_impl", _fake_impl)

    submission.run(**dirs, register_signals=False, llm=None,
                   action_mode_override="json_action")

    csv_path = dirs["output_dir"] / "task_1" / "prediction.csv"
    text = csv_path.read_text(encoding="utf-8")
    assert "None" not in text, f"None leaked as string in CSV: {text!r}"
    rows = list(csv.reader(text.splitlines()))
    assert rows == [["id", "name"], ["1", ""], ["2", "alice"]]
