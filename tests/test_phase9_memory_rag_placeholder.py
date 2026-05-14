import pytest

import data_agent_langchain.memory.rag as rag


def test_corpus_rag_placeholder_raises():
    with pytest.raises(NotImplementedError, match="Phase M4"):
        rag.build_corpus_retriever()
