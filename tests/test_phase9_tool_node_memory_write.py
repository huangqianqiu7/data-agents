from dataclasses import replace
from pathlib import Path

from data_agent_langchain.config import MemoryConfig, default_app_config
from data_agent_langchain.memory.factory import build_store, build_writer
from data_agent_langchain.runtime.state import RunState
from data_agent_langchain.tools.tool_runtime import ToolRuntimeResult


def _state_for_read_csv() -> RunState:
    return {
        "task_id": "t1",
        "question": "q",
        "action": "read_csv",
        "action_input": {"file_path": "transactions.csv", "max_rows": 5},
        "step_index": 1,
        "skip_tool": False,
        "last_error_kind": None,
    }


def test_tool_node_writes_dataset_knowledge_on_success(tmp_path: Path, monkeypatch):
    from data_agent_langchain.agents import tool_node as tn

    def fake_tool_node_body(state, config=None):
        cfg = default_app_config()
        cfg_with_mem = type(cfg)(
            **{
                **{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()},
                "memory": MemoryConfig(mode="full", path=tmp_path),
            }
        )
        store = build_store(cfg_with_mem.memory)
        writer = build_writer(cfg_with_mem.memory, store=store)
        tn._maybe_write_dataset_knowledge(
            writer=writer,
            dataset="ds",
            action="read_csv",
            action_input=state.get("action_input") or {},
            content={
                "columns": ["date", "amount"],
                "dtypes": {"date": "string", "amount": "float"},
                "row_count_estimate": 100,
            },
        )
        return store

    store = fake_tool_node_body(_state_for_read_csv())
    recs = store.list("dataset:ds")
    assert len(recs) == 1
    assert recs[0].payload["file_path"] == "transactions.csv"
    assert recs[0].payload["schema"] == {"date": "string", "amount": "float"}


def test_real_tool_node_writes_memory_on_success(tmp_path: Path, monkeypatch):
    from data_agent_langchain.agents import tool_node as tn

    dataset_root = tmp_path / "dataset_ds"
    task_dir = dataset_root / "task_test"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        '{"task_id": "task_test", "difficulty": "easy", "question": "Q?"}',
        encoding="utf-8",
    )
    (context_dir / "transactions.csv").write_text(
        "date,amount\n2026-01-01,10\n", encoding="utf-8"
    )

    cfg = replace(
        default_app_config(),
        memory=MemoryConfig(mode="full", path=tmp_path / "memory"),
    )
    monkeypatch.setattr(tn, "_safe_get_app_config", lambda: cfg)
    monkeypatch.setattr(
        tn,
        "call_tool_with_timeout",
        lambda tool, action_input, timeout_s: ToolRuntimeResult(
            ok=True,
            content={
                "path": "transactions.csv",
                "columns": ["date", "amount"],
                "dtypes": {"date": "string", "amount": "float"},
                "row_count_estimate": 1,
            },
        ),
    )

    update = tn.tool_node(
        {
            "task_id": "task_test",
            "question": "Q?",
            "dataset_root": str(dataset_root),
            "context_dir": str(context_dir),
            "task_dir": str(task_dir),
            "action": "read_csv",
            "action_input": {"path": "transactions.csv", "max_rows": 5},
            "step_index": 1,
            "skip_tool": False,
            "last_error_kind": None,
        }
    )

    assert update["last_tool_ok"] is True
    assert update["preview_done"] is True
    store = build_store(cfg.memory)
    recs = store.list("dataset:dataset_ds")
    assert len(recs) == 1
    assert recs[0].payload["file_path"] == "transactions.csv"
    assert recs[0].payload["schema"] == {"date": "string", "amount": "float"}


def test_maybe_write_skips_non_dataset_actions(tmp_path: Path):
    from data_agent_langchain.agents import tool_node as tn

    cfg = MemoryConfig(mode="full", path=tmp_path)
    store = build_store(cfg)
    writer = build_writer(cfg, store=store)
    tn._maybe_write_dataset_knowledge(
        writer=writer,
        dataset="ds",
        action="execute_python",
        action_input={},
        content={"columns": []},
    )
    assert store.list("dataset:ds") == []


def test_maybe_write_skips_when_schema_missing(tmp_path: Path):
    from data_agent_langchain.agents import tool_node as tn

    cfg = MemoryConfig(mode="full", path=tmp_path)
    store = build_store(cfg)
    writer = build_writer(cfg, store=store)
    tn._maybe_write_dataset_knowledge(
        writer=writer,
        dataset="ds",
        action="read_csv",
        action_input={"path": "transactions.csv"},
        content={"columns": ["date", "amount"], "row_count": 1},
    )
    assert store.list("dataset:ds") == []
