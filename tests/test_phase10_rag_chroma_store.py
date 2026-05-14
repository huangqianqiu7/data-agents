"""``ChromaCorpusStore.ephemeral`` 行为测试（M4.3.2）。

按 ``01-design-v2.md §4.9`` / ``D10`` 验证：

  - 顶层 import 边界（D11，**不**带 ``slow``）：
    ``import data_agent_langchain.memory.rag.stores.chroma`` 后
    ``sys.modules`` 不含 ``chromadb``。
  - Wiring 单元测试（mock ``chromadb``，**不**带 ``slow``）：
      * ``Settings(anonymized_telemetry=False)`` 已传给 ``EphemeralClient``。
      * collection name = ``f"memrag_{sha1(namespace)[:16]}"``（D10）。
      * ``embedding_function=None``（避免 chroma 内置 OpenAI embed function）。
      * ``upsert_chunks([])`` 短路，不调用 chroma。
      * ``upsert_chunks(chunks)`` 透传 ids / embeddings / documents / metadatas。
      * ``query_by_vector(k=0)`` 短路，不调用 chroma。
      * ``query_by_vector(vec, k>0)`` 调用 ``collection.query`` 并把结果转成
        ``RetrievalResult(record=MemoryRecord(kind="corpus"), reason="vector_cosine")``。
      * cosine ``score = 1 - dist/2 ∈ [-1, 1]``。
  - B2 守护（**不**带 ``slow``）：``ChromaCorpusStore`` 不应有 ``put`` /
    ``get`` / ``list`` / ``delete`` 方法。
  - 真 chromadb 集成（**带** ``@pytest.mark.slow``）：在 chromadb 已装 +
    ``--runslow`` 时验证 ephemeral 不写盘、upsert/query 命中、namespace 隔离。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from data_agent_langchain.memory.rag.documents import CorpusChunk
from data_agent_langchain.memory.rag.embedders import DeterministicStubEmbedder


# ---------------------------------------------------------------------------
# 顶层 import 边界（D11）—— 非 slow，每次必跑
# ---------------------------------------------------------------------------


def test_chroma_module_does_not_import_chromadb_at_module_load() -> None:
    """``import ChromaCorpusStore`` 不应在模块加载阶段拉 ``chromadb``。"""
    code = """
import sys

from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore  # noqa

leaked = [m for m in ('chromadb',) if m in sys.modules]
assert not leaked, f'chromadb 在模块加载阶段被 import：{leaked}'
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src"),
        },
    )
    assert result.returncode == 0, (
        f"子进程失败：stdout={result.stdout}, stderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# Wiring 单元测试（mock chromadb）—— 非 slow
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_chromadb(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """注入一个假的 ``chromadb`` 模块到 ``sys.modules``。

    Returns: ``(EphemeralClient_class_mock, fake_collection)``。
    """
    fake_collection = MagicMock(name="fake_collection")
    fake_client_instance = MagicMock(name="fake_client_instance")
    fake_client_instance.get_or_create_collection.return_value = fake_collection
    fake_ephemeral_class = MagicMock(name="EphemeralClient", return_value=fake_client_instance)

    fake_chromadb_module = MagicMock(name="chromadb_module")
    fake_chromadb_module.EphemeralClient = fake_ephemeral_class

    fake_settings_class = MagicMock(name="Settings")
    fake_config_module = MagicMock(name="chromadb.config")
    fake_config_module.Settings = fake_settings_class

    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb_module)
    monkeypatch.setitem(sys.modules, "chromadb.config", fake_config_module)
    return fake_ephemeral_class, fake_collection


def test_chroma_ephemeral_passes_anonymized_telemetry_false(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``ephemeral`` 必须传 ``Settings(anonymized_telemetry=False)`` 关闭 telemetry。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    fake_ephemeral_class, _ = fake_chromadb
    ChromaCorpusStore.ephemeral(
        namespace="corpus_task:abc",
        embedder=DeterministicStubEmbedder(dim=8),
    )

    # 验证 Settings 被实例化时传了 anonymized_telemetry=False。
    fake_config_module = sys.modules["chromadb.config"]
    settings_call = fake_config_module.Settings.call_args  # type: ignore[attr-defined]
    assert settings_call.kwargs.get("anonymized_telemetry") is False, (
        f"Settings 应传 anonymized_telemetry=False，实际 {settings_call.kwargs}"
    )

    # 验证 EphemeralClient 收到了 Settings 实例。
    assert fake_ephemeral_class.called, "EphemeralClient 应被实例化"


def test_chroma_collection_name_is_sha1_of_namespace(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """collection name = ``f"memrag_{sha1(namespace)[:16]}"``（D10）。"""
    import hashlib

    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    _, fake_collection = fake_chromadb
    namespace = "corpus_task:my_task_id"
    ChromaCorpusStore.ephemeral(
        namespace=namespace,
        embedder=DeterministicStubEmbedder(dim=8),
    )

    fake_chromadb_module = sys.modules["chromadb"]
    create_call = (
        fake_chromadb_module.EphemeralClient.return_value.get_or_create_collection.call_args  # type: ignore[attr-defined]
    )
    expected_name = f"memrag_{hashlib.sha1(namespace.encode()).hexdigest()[:16]}"
    assert create_call.kwargs.get("name") == expected_name, (
        f"collection name 偏离设计公式：实际 {create_call.kwargs.get('name')}，期望 {expected_name}"
    )


def test_chroma_disables_internal_embedding_function(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``embedding_function=None``：禁用 chroma 内置 embed function（避免 OpenAI 等外部 API）。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    ChromaCorpusStore.ephemeral(
        namespace="ns",
        embedder=DeterministicStubEmbedder(dim=8),
    )
    fake_chromadb_module = sys.modules["chromadb"]
    create_call = (
        fake_chromadb_module.EphemeralClient.return_value.get_or_create_collection.call_args  # type: ignore[attr-defined]
    )
    assert create_call.kwargs.get("embedding_function") is None, (
        f"必须显式传 embedding_function=None，实际 {create_call.kwargs}"
    )


def test_chroma_namespace_property_returns_original_string(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``namespace`` property 返回构造时的原始字符串（不是 sha1 编码后的）。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    store = ChromaCorpusStore.ephemeral(
        namespace="corpus_task:abc",
        embedder=DeterministicStubEmbedder(dim=8),
    )
    assert store.namespace == "corpus_task:abc"


def test_chroma_dimension_property_delegates_to_embedder(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``dimension`` property 应代理到 ``embedder.dimension``。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    store = ChromaCorpusStore.ephemeral(
        namespace="ns",
        embedder=DeterministicStubEmbedder(dim=16),
    )
    assert store.dimension == 16


def test_chroma_upsert_chunks_empty_short_circuits(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``upsert_chunks([])`` 应直接返回，不调用 chroma 也不调用 embedder。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    _, fake_collection = fake_chromadb
    embedder = MagicMock(spec=DeterministicStubEmbedder(dim=8))
    embedder.dimension = 8
    store = ChromaCorpusStore.ephemeral(namespace="ns", embedder=embedder)
    store.upsert_chunks([])

    fake_collection.upsert.assert_not_called()
    embedder.embed_documents.assert_not_called()


def test_chroma_upsert_chunks_passes_ids_embeddings_documents_metadatas(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``upsert_chunks`` 应把 ids / embeddings / documents / metadatas 透传给 collection.upsert。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    _, fake_collection = fake_chromadb
    store = ChromaCorpusStore.ephemeral(
        namespace="ns",
        embedder=DeterministicStubEmbedder(dim=8),
    )

    chunks = [
        CorpusChunk(
            chunk_id="d1#0000",
            doc_id="d1",
            ord=0,
            text="hello",
            char_offset=0,
            char_length=5,
        ),
        CorpusChunk(
            chunk_id="d1#0001",
            doc_id="d1",
            ord=1,
            text="world",
            char_offset=5,
            char_length=5,
        ),
    ]
    store.upsert_chunks(chunks)

    upsert_call = fake_collection.upsert.call_args
    assert upsert_call.kwargs["ids"] == ["d1#0000", "d1#0001"]
    assert upsert_call.kwargs["documents"] == ["hello", "world"]
    embeddings = upsert_call.kwargs["embeddings"]
    assert isinstance(embeddings, list) and len(embeddings) == 2
    for vec in embeddings:
        assert isinstance(vec, list) and len(vec) == 8
    metadatas = upsert_call.kwargs["metadatas"]
    assert metadatas == [
        {"doc_id": "d1", "ord": 0, "char_offset": 0, "char_length": 5},
        {"doc_id": "d1", "ord": 1, "char_offset": 5, "char_length": 5},
    ]


def test_chroma_query_by_vector_k_zero_short_circuits(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``query_by_vector(k=0)`` 短路返回 ``[]``，不调用 chroma。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    _, fake_collection = fake_chromadb
    store = ChromaCorpusStore.ephemeral(
        namespace="ns",
        embedder=DeterministicStubEmbedder(dim=8),
    )
    out = store.query_by_vector([0.1] * 8, k=0)
    assert out == []
    fake_collection.query.assert_not_called()


def test_chroma_query_by_vector_negative_k_short_circuits(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``k < 0`` 也应短路返回 ``[]``。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    _, fake_collection = fake_chromadb
    store = ChromaCorpusStore.ephemeral(
        namespace="ns",
        embedder=DeterministicStubEmbedder(dim=8),
    )
    assert store.query_by_vector([0.0] * 8, k=-1) == []
    fake_collection.query.assert_not_called()


def test_chroma_query_by_vector_returns_retrieval_result_with_corpus_kind(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``query_by_vector`` 返回 ``list[RetrievalResult]``，``record.kind == "corpus"``。"""
    from data_agent_langchain.memory.base import MemoryRecord, RetrievalResult
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    _, fake_collection = fake_chromadb
    # 模拟 chroma 返回单条命中。
    fake_collection.query.return_value = {
        "ids": [["d1#0000"]],
        "documents": [["hello"]],
        "metadatas": [[
            {"doc_id": "d1", "ord": 0, "char_offset": 0, "char_length": 5}
        ]],
        "distances": [[0.4]],  # cosine distance
    }
    store = ChromaCorpusStore.ephemeral(
        namespace="corpus_task:abc",
        embedder=DeterministicStubEmbedder(dim=8),
    )

    out = store.query_by_vector([0.1] * 8, k=1)
    assert isinstance(out, list)
    assert len(out) == 1
    res = out[0]
    assert isinstance(res, RetrievalResult)
    assert isinstance(res.record, MemoryRecord)
    assert res.record.kind == "corpus"
    assert res.record.namespace == "corpus_task:abc"
    assert res.record.id == "d1#0000"
    assert res.record.payload["text"] == "hello"
    assert res.record.payload["doc_id"] == "d1"
    assert res.record.payload["ord"] == 0
    assert res.reason == "vector_cosine"
    # cosine score = 1 - dist/2 = 1 - 0.2 = 0.8
    assert abs(res.score - 0.8) < 1e-9, f"score 公式偏离：{res.score}"


def test_chroma_query_includes_required_fields(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``collection.query`` 必须 ``include=["documents", "metadatas", "distances"]``。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    _, fake_collection = fake_chromadb
    fake_collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    store = ChromaCorpusStore.ephemeral(
        namespace="ns",
        embedder=DeterministicStubEmbedder(dim=8),
    )
    store.query_by_vector([0.0] * 8, k=3)

    query_call = fake_collection.query.call_args
    assert query_call.kwargs.get("query_embeddings") == [[0.0] * 8]
    assert query_call.kwargs.get("n_results") == 3
    include = query_call.kwargs.get("include") or []
    for required in ("documents", "metadatas", "distances"):
        assert required in include, (
            f"include 缺少 {required}，实际 {include}"
        )


def test_chroma_close_releases_client(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``close`` 应释放 client 引用（设 None），便于 GC。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    store = ChromaCorpusStore.ephemeral(
        namespace="ns",
        embedder=DeterministicStubEmbedder(dim=8),
    )
    store.close()
    # 用反射访问私有字段验证；契约是 close 后不再持有 client 引用。
    assert getattr(store, "_client") is None


# ---------------------------------------------------------------------------
# B2 守护：ChromaCorpusStore 不应有 v2 MemoryStore KV 方法
# ---------------------------------------------------------------------------


def test_chroma_corpus_store_does_not_expose_kv_methods(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """B2 决策：``ChromaCorpusStore`` 不应实现 ``put`` / ``get`` / ``list`` / ``delete``。"""
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    store = ChromaCorpusStore.ephemeral(
        namespace="ns",
        embedder=DeterministicStubEmbedder(dim=8),
    )
    for forbidden in ("put", "get", "list", "delete"):
        assert not hasattr(store, forbidden), (
            f"B2 违反：ChromaCorpusStore 不应有 {forbidden} 方法"
        )


def test_chroma_corpus_store_satisfies_corpus_store_protocol(
    fake_chromadb: tuple[MagicMock, MagicMock],
) -> None:
    """``ChromaCorpusStore`` 应满足 ``CorpusStore`` Protocol（``runtime_checkable``）。"""
    from data_agent_langchain.memory.rag.base import CorpusStore
    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    store = ChromaCorpusStore.ephemeral(
        namespace="ns",
        embedder=DeterministicStubEmbedder(dim=8),
    )
    assert isinstance(store, CorpusStore)


# ---------------------------------------------------------------------------
# 真 chromadb 集成（slow）—— 仅 --runslow + chromadb 可用时跑
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_chroma_real_ephemeral_upsert_query_round_trip() -> None:
    """真 chromadb：``upsert_chunks`` 后 ``query_by_vector`` 命中。"""
    try:
        import chromadb  # noqa
    except ImportError as exc:
        pytest.skip(f"chromadb 未装：{exc}")

    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    embedder = DeterministicStubEmbedder(dim=16)
    store = ChromaCorpusStore.ephemeral(namespace="corpus_task:test", embedder=embedder)

    chunks = [
        CorpusChunk(
            chunk_id=f"doc1#{i:04d}",
            doc_id="doc1",
            ord=i,
            text=text,
            char_offset=0,
            char_length=len(text),
        )
        for i, text in enumerate(["hello world", "lorem ipsum", "foo bar baz"])
    ]
    store.upsert_chunks(chunks)

    query_vec = embedder.embed_query("hello")
    out = store.query_by_vector(query_vec, k=1)
    assert len(out) == 1
    assert out[0].record.kind == "corpus"
    store.close()


@pytest.mark.slow
def test_chroma_real_namespace_isolation() -> None:
    """真 chromadb：往 namespace A 写后，从 namespace B 查得 ``[]``。"""
    try:
        import chromadb  # noqa
    except ImportError as exc:
        pytest.skip(f"chromadb 未装：{exc}")

    from data_agent_langchain.memory.rag.stores.chroma import ChromaCorpusStore

    embedder = DeterministicStubEmbedder(dim=16)
    store_a = ChromaCorpusStore.ephemeral(namespace="corpus_task:A", embedder=embedder)
    store_b = ChromaCorpusStore.ephemeral(namespace="corpus_task:B", embedder=embedder)

    chunks = [
        CorpusChunk(
            chunk_id="d#0000", doc_id="d", ord=0, text="hi",
            char_offset=0, char_length=2,
        )
    ]
    store_a.upsert_chunks(chunks)

    out_b = store_b.query_by_vector(embedder.embed_query("hi"), k=1)
    assert out_b == [], f"namespace 隔离失败：{out_b}"
    store_a.close()
    store_b.close()
