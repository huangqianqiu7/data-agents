import pickle

from data_agent_langchain.memory.types import MemoryHit


def test_memory_hit_is_picklable_and_frozen():
    hit = MemoryHit(
        record_id="dk:ds:x.csv",
        namespace="dataset:ds",
        score=1.0,
        summary="File: x.csv  Columns: ['a']",
    )
    assert pickle.loads(pickle.dumps(hit)) == hit
