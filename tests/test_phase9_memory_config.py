from data_agent_langchain.config import AppConfig, MemoryConfig, default_app_config


def test_memory_config_defaults_safe_for_eval():
    cfg = MemoryConfig()
    assert cfg.mode == "disabled"
    assert cfg.store_backend == "jsonl"
    assert cfg.retriever_type == "exact"
    assert cfg.retrieval_max_results == 5


def test_app_config_round_trip_with_memory():
    cfg = default_app_config()
    assert isinstance(cfg.memory, MemoryConfig)
    payload = cfg.to_dict()
    assert payload["memory"]["mode"] == "disabled"
    restored = AppConfig.from_dict(payload)
    assert restored.memory == cfg.memory


def test_memory_config_picklable():
    import pickle

    cfg = MemoryConfig(mode="full")
    assert pickle.loads(pickle.dumps(cfg)) == cfg
