from data_agent_langchain.submission import build_submission_config


def test_submission_default_memory_is_read_only_dataset(monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", "http://example.com")
    monkeypatch.setenv("MODEL_NAME", "gpt-test")

    cfg = build_submission_config()

    assert cfg.memory.mode == "read_only_dataset"
