from datetime import datetime

import pytest

from data_agent_langchain.memory.base import MemoryRecord, RetrievalResult


def test_memory_record_is_frozen_and_picklable():
    import pickle

    rec = MemoryRecord(
        id="dk:ds:transactions.csv",
        namespace="dataset:ds",
        kind="dataset_knowledge",
        payload={"file_path": "transactions.csv"},
    )
    # frozen
    with pytest.raises(Exception):
        rec.id = "x"  # type: ignore[misc]
    # picklable
    assert pickle.loads(pickle.dumps(rec)) == rec
    assert isinstance(rec.created_at, datetime)


def test_retrieval_result_carries_reason():
    rec = MemoryRecord(
        id="x", namespace="dataset:ds", kind="dataset_knowledge", payload={}
    )
    res = RetrievalResult(record=rec, score=1.0, reason="exact_namespace")
    assert res.reason == "exact_namespace"
