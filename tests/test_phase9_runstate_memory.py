import operator
import pickle
from typing import get_args, get_type_hints

from data_agent_langchain.memory.types import MemoryHit
from data_agent_langchain.runtime.state import RunState


def test_runstate_memory_hits_reducer_and_picklable():
    hints = get_type_hints(RunState, include_extras=True)
    annotation = hints["memory_hits"]
    assert get_args(annotation)[1] is operator.add

    state = {
        "memory_hits": [
            MemoryHit(
                record_id="dk:ds:a.csv",
                namespace="dataset:ds",
                score=1.0,
                summary="File: a.csv",
            )
        ]
    }
    assert pickle.loads(pickle.dumps(state)) == state
