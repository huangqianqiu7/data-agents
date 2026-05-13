from pathlib import Path

from data_agent_langchain.memory.base import MemoryRecord
from data_agent_langchain.memory.stores.jsonl import JsonlMemoryStore, _safe_filename


def test_put_then_list(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    rec = MemoryRecord(
        id="dk:ds:a.csv",
        namespace="dataset:ds",
        kind="dataset_knowledge",
        payload={"file_path": "a.csv"},
    )
    store.put(rec)
    results = store.list("dataset:ds", limit=5)
    assert len(results) == 1
    assert results[0].id == "dk:ds:a.csv"
    assert results[0].payload["file_path"] == "a.csv"


def test_list_returns_recent_first(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    for i in range(3):
        store.put(
            MemoryRecord(
                id=f"x{i}",
                namespace="dataset:ds",
                kind="dataset_knowledge",
                payload={"i": i},
            )
        )
    results = store.list("dataset:ds", limit=2)
    assert [r.id for r in results] == ["x2", "x1"]


def test_namespace_isolated(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(
        MemoryRecord(
            id="a",
            namespace="dataset:ds1",
            kind="dataset_knowledge",
            payload={},
        )
    )
    assert store.list("dataset:ds2") == []


def test_collision_looking_namespaces_are_isolated(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(
        MemoryRecord(
            id="colon",
            namespace="a:b",
            kind="dataset_knowledge",
            payload={"namespace": "colon"},
        )
    )
    store.put(
        MemoryRecord(
            id="slash",
            namespace="a/b",
            kind="dataset_knowledge",
            payload={"namespace": "slash"},
        )
    )

    assert [r.id for r in store.list("a:b")] == ["colon"]
    assert [r.id for r in store.list("a/b")] == ["slash"]


def test_delete_in_collision_looking_namespace_stays_isolated(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(
        MemoryRecord(
            id="same",
            namespace="a:b",
            kind="dataset_knowledge",
            payload={"namespace": "colon"},
        )
    )
    store.put(
        MemoryRecord(
            id="same",
            namespace="a/b",
            kind="dataset_knowledge",
            payload={"namespace": "slash"},
        )
    )

    store.delete("a/b", "same")

    colon_record = store.get("a:b", "same")
    assert colon_record is not None
    assert colon_record.payload["namespace"] == "colon"
    assert store.get("a/b", "same") is None


def test_case_only_filename_collision_namespaces_are_isolated(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(
        MemoryRecord(
            id="lower",
            namespace="aaa",
            kind="dataset_knowledge",
            payload={"namespace": "aaa"},
        )
    )
    store.put(
        MemoryRecord(
            id="mixed",
            namespace="aaG",
            kind="dataset_knowledge",
            payload={"namespace": "aaG"},
        )
    )

    assert [r.id for r in store.list("aaa")] == ["lower"]
    assert [r.id for r in store.list("aaG")] == ["mixed"]


def test_safe_filename_avoids_case_insensitive_collisions():
    assert _safe_filename("aaa").lower() != _safe_filename("aaG").lower()


def test_empty_namespace_returns_empty(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    assert store.list("dataset:none") == []


def test_get_returns_latest_or_none(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(
        MemoryRecord(
            id="a",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={"v": 1},
        )
    )
    store.put(
        MemoryRecord(
            id="a",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={"v": 2},
        )
    )
    rec = store.get("dataset:ds", "a")
    assert rec is not None
    assert rec.payload["v"] == 2
    assert store.get("dataset:ds", "missing") is None


def test_delete_writes_tombstone(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(
        MemoryRecord(
            id="a",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={},
        )
    )
    store.delete("dataset:ds", "a")
    assert store.get("dataset:ds", "a") is None
    assert store.list("dataset:ds") == []


def test_put_after_delete_reactivates_record(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    store.put(
        MemoryRecord(
            id="a",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={"v": 1},
        )
    )
    store.delete("dataset:ds", "a")
    store.put(
        MemoryRecord(
            id="a",
            namespace="dataset:ds",
            kind="dataset_knowledge",
            payload={"v": 2},
        )
    )

    rec = store.get("dataset:ds", "a")
    assert rec is not None
    assert rec.payload["v"] == 2
    results = store.list("dataset:ds")
    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].payload["v"] == 2


def test_safe_filename_replaces_windows_separators():
    filename = _safe_filename("dataset\\..\\outside")

    assert "\\" not in filename
    assert "/" not in filename
    assert filename.endswith(".jsonl")


def test_windows_namespace_separator_cannot_escape_root(tmp_path: Path):
    store = JsonlMemoryStore(root=tmp_path)
    escaped = tmp_path.parent / "jsonl_store_escape_probe.jsonl"
    assert not escaped.exists()

    store.put(
        MemoryRecord(
            id="a",
            namespace="..\\jsonl_store_escape_probe",
            kind="dataset_knowledge",
            payload={},
        )
    )

    assert not escaped.exists()
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].is_file()
    assert "\\" not in files[0].name
    assert "/" not in files[0].name
