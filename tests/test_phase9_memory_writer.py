from pathlib import Path

from data_agent_langchain.memory.records import (
    DatasetKnowledgeRecord,
    ToolPlaybookRecord,
)
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore
from data_agent_langchain.memory.writers.store_backed import StoreBackedMemoryWriter


def _writer(tmp_path: Path, mode: str) -> tuple[StoreBackedMemoryWriter, JsonlMemoryStore]:
    store = JsonlMemoryStore(root=tmp_path)
    return StoreBackedMemoryWriter(store, mode=mode), store


def _dk() -> DatasetKnowledgeRecord:
    return DatasetKnowledgeRecord(
        file_path="a.csv",
        file_kind="csv",
        schema={"a": "string"},
        row_count_estimate=10,
    )


def _tp() -> ToolPlaybookRecord:
    return ToolPlaybookRecord(
        tool_name="read_csv",
        input_template={"max_rows": 20},
        preconditions=["preview_done"],
    )


def test_full_mode_writes_both(tmp_path: Path):
    w, store = _writer(tmp_path, mode="full")
    w.write_dataset_knowledge("ds", _dk())
    w.write_tool_playbook("ds", "read_csv", _tp())
    assert len(store.list("dataset:ds")) == 1
    assert len(store.list("dataset:ds/tool:read_csv")) == 1


def test_read_only_dataset_blocks_tool_playbook(tmp_path: Path):
    w, store = _writer(tmp_path, mode="read_only_dataset")
    w.write_dataset_knowledge("ds", _dk())
    w.write_tool_playbook("ds", "read_csv", _tp())
    assert len(store.list("dataset:ds")) == 1
    assert store.list("dataset:ds/tool:read_csv") == []


def test_disabled_writes_nothing(tmp_path: Path):
    w, store = _writer(tmp_path, mode="disabled")
    w.write_dataset_knowledge("ds", _dk())
    w.write_tool_playbook("ds", "read_csv", _tp())
    assert store.list("dataset:ds") == []
    assert store.list("dataset:ds/tool:read_csv") == []
