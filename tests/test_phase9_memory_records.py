import pytest

from data_agent_langchain.memory.records import (
    DatasetKnowledgeRecord,
    ToolPlaybookRecord,
)


BANNED_FREE_TEXT_FIELDS = (
    "question",
    "answer",
    "approach",
    "hint",
    "summary",
    "example_answer",
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


@pytest.mark.parametrize("field_name", BANNED_FREE_TEXT_FIELDS)
def test_dataset_knowledge_record_rejects_freetext_fields(field_name: str):
    """Field-level block for question / answer / approach / hint / summary."""
    with pytest.raises(TypeError):
        DatasetKnowledgeRecord(  # type: ignore[call-arg]
            file_path="x.csv",
            file_kind="csv",
            schema={},
            row_count_estimate=None,
            **{field_name: "leaked"},
        )


def test_dataset_knowledge_record_default_lists_are_independent():
    first = DatasetKnowledgeRecord(
        file_path="first.csv",
        file_kind="csv",
        schema={},
        row_count_estimate=None,
    )
    second = DatasetKnowledgeRecord(
        file_path="second.csv",
        file_kind="csv",
        schema={},
        row_count_estimate=None,
    )

    first.sample_columns.append("date")

    assert second.sample_columns == []


def test_tool_playbook_record_fields():
    rec = ToolPlaybookRecord(
        tool_name="read_csv",
        input_template={"max_rows": 20},
        preconditions=["preview_done"],
    )
    assert rec.typical_failures == []


@pytest.mark.parametrize("field_name", BANNED_FREE_TEXT_FIELDS)
def test_tool_playbook_record_rejects_freetext_fields(field_name: str):
    with pytest.raises(TypeError):
        ToolPlaybookRecord(  # type: ignore[call-arg]
            tool_name="read_csv",
            input_template={},
            preconditions=[],
            **{field_name: "leaked"},
        )


def test_tool_playbook_record_default_lists_are_independent():
    first = ToolPlaybookRecord(
        tool_name="read_csv",
        input_template={},
        preconditions=[],
    )
    second = ToolPlaybookRecord(
        tool_name="load_json",
        input_template={},
        preconditions=[],
    )

    first.typical_failures.append("missing_file")

    assert second.typical_failures == []
