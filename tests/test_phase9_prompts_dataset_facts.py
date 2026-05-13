from data_agent_langchain.agents.prompts import render_dataset_facts
from data_agent_langchain.memory.types import MemoryHit


def test_render_emits_only_whitelisted_summary_text():
    hits = [
        MemoryHit(
            record_id="dk:ds:a.csv",
            namespace="dataset:ds",
            score=1.0,
            summary="File: a.csv  Kind: csv  Columns: ['a', 'b']  Rows~: 10",
        )
    ]
    text = render_dataset_facts(hits)
    assert "a.csv" in text
    # only depends on summary; should not leak repr/internal fields
    assert "MemoryHit" not in text


def test_render_empty_hits_returns_empty_string():
    assert render_dataset_facts([]) == ""
