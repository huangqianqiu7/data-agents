"""Bug 4 守卫测试：``ChromaCorpusStore.close()`` 必须真正释放 collection。

根因（已坐实）：``chromadb.EphemeralClient`` 在同一 Python 进程内是**单例缓存**
设计（chromadb 0.4+），``ChromaCorpusStore.close()`` 当前仅 ``self._client = None``
丢引用，并不显式 ``delete_collection``。这会导致：

  1. **测试隔离 bug**：所有测试都用 ``task_id="t1"`` → collection name 固定
     ``memrag_<sha1("corpus_task:t1")[:16]>``，跨测试累积 chunk，使
     ``test_build_task_corpus_index_can_retrieve_indexed_chunks`` 在按文件顺序
     跑时 retriever 返回上次写入的 chunk。
  2. **production 内存累积**：评测态如果同进程多次复用 task_id（即便概率小）
     也会累积；shared / persistent backend 落地后这个隐患会放大。

修复方向（minimal upstream fix）：在 ``close()`` 内显式调
``client.delete_collection(name=self._collection_name)`` 并对已删除场景幂等。

本测试集合验证：

  - **Isolation**：close 后用同 namespace 再 build_task_corpus 不应继承上次 chunk。
  - **Idempotent**：``close()`` 重复调用 + delete 不存在的 collection 应静默不抛。
  - **No leak via factory**：``build_task_corpus`` 失败路径（chroma_store_load_failed
    / index_timeout）后再次 build 同 namespace 不应继承之前 chunk。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from data_agent_langchain.config import CorpusRagConfig
from data_agent_langchain.memory.rag.embedders.stub import DeterministicStubEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ctx(root: Path, label: str, payload: str) -> Path:
    """构造一个含单一 markdown 文档的 context_dir。``payload`` 用于让两次 build
    产生不同的 doc_id（doc_id = sha1(source_path + size + mtime)[:16]）。"""
    ctx = root / f"ctx_{label}"
    ctx.mkdir()
    (ctx / "README.md").write_text(payload, encoding="utf-8")
    return ctx


# ---------------------------------------------------------------------------
# Bug 4.1：close 后 collection 释放，下次同 namespace build 不继承之前 chunk
# ---------------------------------------------------------------------------


def test_chroma_store_close_releases_collection_for_next_build(
    tmp_path: Path,
) -> None:
    """连续两次 ``build_task_corpus(task_id="t1")`` 用不同文档，第一次 close 后
    第二次 retriever 召回的 doc_id 应全部出现在**第二次自己的** doc_index 中。

    回归 Bug 4：``EphemeralClient`` 进程内单例 + close 不删 collection 会让第二次
    继承第一次的 chunk → retriever 返回上次 doc_id → doc_index 查不到 →
    payload 不含 source_path / doc_kind。
    """
    pytest.importorskip("chromadb")
    from data_agent_langchain.memory.rag.factory import build_task_corpus

    cfg = CorpusRagConfig(enabled=True, retrieval_k=4)
    embedder = DeterministicStubEmbedder(dim=16)

    # 第一次 build：context A（"first content"）。
    ctx_a = _make_ctx(tmp_path, "a", "first content about apples\n" * 5)
    handles_a = build_task_corpus(
        cfg, task_id="t1", task_input_dir=ctx_a, embedder=embedder
    )
    assert handles_a is not None
    doc_ids_a = set(handles_a.retriever._doc_index.keys())
    handles_a.store.close()

    # 第二次 build：同 task_id="t1" 但 context B（不同内容 → 不同 doc_id）。
    ctx_b = _make_ctx(tmp_path, "b", "second content about bananas\n" * 5)
    handles_b = build_task_corpus(
        cfg, task_id="t1", task_input_dir=ctx_b, embedder=embedder
    )
    assert handles_b is not None
    doc_ids_b = set(handles_b.retriever._doc_index.keys())

    # 关键断言：两组 doc_id 不应有交集。
    assert not (doc_ids_a & doc_ids_b), (
        f"两次 build 用不同文档应产生不同 doc_id；存在共享：{doc_ids_a & doc_ids_b}"
    )

    # 关键断言：第二次召回的所有结果的 doc_id 必须在第二次 doc_index 中。
    results = handles_b.retriever.retrieve("content", namespace="corpus_task:t1")
    assert results, "第二次 retriever 应至少召回 1 条"
    leaked_doc_ids = {
        r.record.payload["doc_id"] for r in results
    } - doc_ids_b
    assert not leaked_doc_ids, (
        f"close 后第二次 retriever 仍拿到上次 build 的 doc_id：{leaked_doc_ids}；"
        f"说明 chroma collection 未被释放。"
    )

    # source_path 必须被回填（doc_index 命中）。
    for r in results:
        assert "source_path" in r.record.payload, (
            f"retriever 应回填 source_path；实际 payload keys：{sorted(r.record.payload)}"
        )

    handles_b.store.close()


# ---------------------------------------------------------------------------
# Bug 4.2：close 幂等
# ---------------------------------------------------------------------------


def test_chroma_store_close_is_idempotent(tmp_path: Path) -> None:
    """``close()`` 重复调用不应抛异常；第二次 close 应为 no-op。"""
    pytest.importorskip("chromadb")
    from data_agent_langchain.memory.rag.factory import build_task_corpus

    cfg = CorpusRagConfig(enabled=True)
    embedder = DeterministicStubEmbedder(dim=16)
    ctx = _make_ctx(tmp_path, "idem", "some content\n")
    handles = build_task_corpus(
        cfg, task_id="t_idem", task_input_dir=ctx, embedder=embedder
    )
    assert handles is not None
    store = handles.store

    # 两次连续 close 不应抛。
    store.close()
    store.close()


# ---------------------------------------------------------------------------
# Bug 4.3：close 对底层 delete_collection 抛错也要静默（fail-closed）
# ---------------------------------------------------------------------------


def test_chroma_store_close_swallows_delete_collection_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``client.delete_collection`` 抛任何异常时 ``close()`` 应静默 fail-closed，
    与 store 其他方法的 fail-closed 风格一致。"""
    pytest.importorskip("chromadb")
    from data_agent_langchain.memory.rag.factory import build_task_corpus

    cfg = CorpusRagConfig(enabled=True)
    embedder = DeterministicStubEmbedder(dim=16)
    ctx = _make_ctx(tmp_path, "fail", "content\n")
    handles = build_task_corpus(
        cfg, task_id="t_fail", task_input_dir=ctx, embedder=embedder
    )
    assert handles is not None
    store = handles.store

    # 让 delete_collection 既记录调用又抛错。close 应：
    #   (a) 尝试调用 client.delete_collection（必须被 call 过，否则修复无效）。
    #   (b) 静默吞掉抛出的异常，不向上传播。
    calls: list[tuple[tuple, dict]] = []

    def _spy_raise(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        calls.append((args, kwargs))
        raise RuntimeError("simulated delete failure")

    monkeypatch.setattr(store._client, "delete_collection", _spy_raise)
    store.close()  # 不应抛
    assert calls, (
        "close() 应尝试 client.delete_collection；当前实现仅丢引用，"
        "导致 collection 在进程内累积。"
    )
