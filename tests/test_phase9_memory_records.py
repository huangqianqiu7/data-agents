import pytest

from data_agent_langchain.memory.records import (
    DatasetKnowledgeRecord,
    ToolPlaybookRecord,
)


def test_dataset_knowledge_record_fields():
    rec = DatasetKnowledgeRecord(
        file_path="transactions.csv",
        file_kind="csv",
        schema={"date": "string", "amount": "float"},
        row_count_estimate=12450,
    )
    assert rec.file_path == "transactions.csv"
    assert rec.encoding is None
    assert rec.sample_columns == []


def test_dataset_knowledge_record_rejects_freetext_fields():
    """Field-level block for question / answer / approach / hint / summary."""
    with pytest.raises(TypeError):
        DatasetKnowledgeRecord(  # type: ignore[call-arg]
            file_path="x.csv",
            file_kind="csv",
            schema={},
            row_count_estimate=None,
            question="leaked",
        )


def test_tool_playbook_record_fields():
    rec = ToolPlaybookRecord(
        tool_name="read_csv",
        input_template={"max_rows": 20},
        preconditions=["preview_done"],
    )
    assert rec.typical_failures == []


def test_tool_playbook_record_rejects_freetext_fields():
    with pytest.raises(TypeError):
        ToolPlaybookRecord(  # type: ignore[call-arg]
            tool_name="read_csv",
            input_template={},
            preconditions=[],
            example_answer="leaked",
        )
