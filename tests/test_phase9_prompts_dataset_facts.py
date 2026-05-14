from data_agent_langchain.agents.prompts import render_dataset_facts
from data_agent_langchain.memory.types import MemoryHit


def test_render_emits_only_whitelisted_summary_text():
    hits = [
        MemoryHit(
            record_id="LEAK_RECORD_ID",
            namespace="LEAK_NAMESPACE",
            score=999.123,
            summary="SUMMARY_ALLOWED: File: a.csv  Kind: csv  Columns: ['a', 'b']",
        )
    ]
    text = render_dataset_facts(hits)
    assert "SUMMARY_ALLOWED" in text
    assert "a.csv" in text
    assert "LEAK_RECORD_ID" not in text
    assert "LEAK_NAMESPACE" not in text
    assert "999.123" not in text
    assert "MemoryHit" not in text


def test_render_empty_hits_returns_empty_string():
    assert render_dataset_facts([]) == ""
