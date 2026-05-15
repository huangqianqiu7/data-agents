"""``Loader`` 扫描 task ``context_dir`` 的行为测试（M4.1.3）。

按 ``01-design-v2.md §4.7`` / ``§6.2`` / ``D8`` 验证：

  - 仅扫描调用方提供的 ``context_dir``，不递归到 ``task_dir`` 等兄弟目录。
  - ``Redactor.is_safe_filename=False`` 的文件（如 ``expected_output.json``、
    ``ground_truth.csv``）被跳过。
  - 不识别的扩展名（如 ``.bin``、``.pyc``）被跳过。
  - ``source_path`` 是相对于 ``context_dir`` 的 POSIX 风格相对路径。
  - ``doc_id`` 由 ``sha1(source_path + size + mtime)[:16]`` 生成且对相同输入稳定。
  - ``doc_kind`` 按扩展名分类：``.md`` → markdown；``.txt`` → text。
    ``.json`` 已移除（大 JSON 数据文件导致 CPU 索引超时）。
  - 文档数超过 ``max_docs_per_task`` 时截断，并 dispatch
    ``memory_rag_skipped(reason="max_docs_truncated")``。

设计偏离：``Loader.scan`` 接受 ``context_dir`` + ``redactor`` + ``max_docs_per_task``
作为独立参数，避免依赖 ``CorpusRagConfig``；``factory.build_task_corpus`` 在
M4.4.3 简单包装。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from data_agent_langchain.memory.rag.documents import CorpusDocument
from data_agent_langchain.memory.rag.loader import Loader
from data_agent_langchain.memory.rag.redactor import Redactor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def redactor() -> Redactor:
    """默认 redactor —— 拦截 expected_output.json / ground_truth* / *label*。"""
    return Redactor(
        redact_patterns=(),  # 内容过滤交给 chunking 阶段，不在 loader scan 内做
        redact_filenames=(
            "expected_output.json",
            "ground_truth*",
            "*label*",
        ),
    )


@pytest.fixture()
def context_dir(tmp_path: Path) -> Path:
    """构造一个典型的 task ``context_dir``：

    .. code-block::

        context/
          README.md            <-- 应收入
          data_schema.md       <-- 应收入
          notes.txt            <-- 应收入
          schema.json          <-- 不识别扩展名（.json 已移除），跳过
          expected_output.json <-- 不识别扩展名，跳过
          ground_truth_v1.csv  <-- redactor 拦截
          train_labels.parquet <-- redactor 拦截
          binary.bin           <-- 不识别扩展名，跳过
    """
    ctx = tmp_path / "context"
    ctx.mkdir()
    (ctx / "README.md").write_text("# Title\nIntro.\n", encoding="utf-8")
    (ctx / "data_schema.md").write_text("# Schema\nA: int\nB: str\n", encoding="utf-8")
    (ctx / "notes.txt").write_text("free text notes\n", encoding="utf-8")
    (ctx / "schema.json").write_text('{"a": "int", "b": "str"}\n', encoding="utf-8")
    (ctx / "expected_output.json").write_text("{}", encoding="utf-8")
    (ctx / "ground_truth_v1.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (ctx / "train_labels.parquet").write_bytes(b"\x00\x01\x02")
    (ctx / "binary.bin").write_bytes(b"\xff\xfe")
    return ctx


# ---------------------------------------------------------------------------
# 基本扫描行为
# ---------------------------------------------------------------------------


def test_loader_scan_returns_only_safe_known_kinds(
    context_dir: Path, redactor: Redactor
) -> None:
    """扫描仅返回安全文件名 + 已知扩展名的文档。"""
    loader = Loader(redactor=redactor, max_docs_per_task=100)
    docs = loader.scan(context_dir)

    source_paths = sorted(d.source_path for d in docs)
    assert source_paths == [
        "README.md",
        "data_schema.md",
        "notes.txt",
    ], f"扫描结果偏离白名单：{source_paths}"


def test_loader_scan_classifies_doc_kind_by_extension(
    context_dir: Path, redactor: Redactor
) -> None:
    """``doc_kind`` 按扩展名映射。"""
    loader = Loader(redactor=redactor, max_docs_per_task=100)
    docs = {d.source_path: d for d in loader.scan(context_dir)}

    assert docs["README.md"].doc_kind == "markdown"
    assert docs["data_schema.md"].doc_kind == "markdown"
    assert docs["notes.txt"].doc_kind == "text"
    assert "schema.json" not in docs, ".json 已从白名单移除"


def test_loader_scan_skips_redactor_blocked_filenames(
    context_dir: Path, redactor: Redactor
) -> None:
    """``expected_output.json`` 等被 redactor 拦截，不出现在结果中。"""
    loader = Loader(redactor=redactor, max_docs_per_task=100)
    source_paths = {d.source_path for d in loader.scan(context_dir)}

    assert "expected_output.json" not in source_paths
    assert "ground_truth_v1.csv" not in source_paths
    assert "train_labels.parquet" not in source_paths


def test_loader_scan_skips_unknown_extensions(
    context_dir: Path, redactor: Redactor
) -> None:
    """``.bin`` 等未知扩展名被跳过。"""
    loader = Loader(redactor=redactor, max_docs_per_task=100)
    source_paths = {d.source_path for d in loader.scan(context_dir)}
    assert "binary.bin" not in source_paths


def test_loader_scan_returns_relative_posix_paths(
    tmp_path: Path, redactor: Redactor
) -> None:
    """``source_path`` 是相对于 ``context_dir`` 的 POSIX 风格相对路径，
    支持子目录嵌套。"""
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "README.md").write_text("a", encoding="utf-8")
    sub = ctx / "guides"
    sub.mkdir()
    (sub / "intro.md").write_text("b", encoding="utf-8")

    loader = Loader(redactor=redactor, max_docs_per_task=100)
    docs = loader.scan(ctx)

    paths = sorted(d.source_path for d in docs)
    assert paths == ["README.md", "guides/intro.md"], paths
    # 不含绝对路径前缀 / 平台分隔符 ``\``。
    for p in paths:
        assert not Path(p).is_absolute()
        assert "\\" not in p


# ---------------------------------------------------------------------------
# 元数据（doc_id / bytes_size / char_count / collection）
# ---------------------------------------------------------------------------


def test_loader_doc_id_is_sha1_of_path_size_mtime(
    tmp_path: Path, redactor: Redactor
) -> None:
    """``doc_id`` 计算应等价于 ``sha1(source_path + size + mtime)[:16]``。"""
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    f = ctx / "README.md"
    f.write_text("hello", encoding="utf-8")

    loader = Loader(redactor=redactor, max_docs_per_task=100)
    docs = loader.scan(ctx)
    assert len(docs) == 1
    doc = docs[0]

    size = f.stat().st_size
    mtime = f.stat().st_mtime
    expected = hashlib.sha1(
        f"{doc.source_path}|{size}|{mtime}".encode("utf-8")
    ).hexdigest()[:16]
    assert doc.doc_id == expected, f"doc_id 不匹配设计公式：实际 {doc.doc_id}，期望 {expected}"


def test_loader_doc_id_is_stable_across_calls(
    tmp_path: Path, redactor: Redactor
) -> None:
    """同一文件多次扫描应返回相同 ``doc_id``（用于跨 task 缓存与去重）。"""
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "README.md").write_text("hello", encoding="utf-8")

    loader = Loader(redactor=redactor, max_docs_per_task=100)
    docs1 = loader.scan(ctx)
    docs2 = loader.scan(ctx)
    assert docs1[0].doc_id == docs2[0].doc_id


def test_loader_records_bytes_size_and_char_count(
    tmp_path: Path, redactor: Redactor
) -> None:
    """``bytes_size`` 取自 ``stat().st_size``；``char_count`` 来自 UTF-8 解码后字符数。"""
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    text = "中文 + ASCII abc"
    (ctx / "README.md").write_text(text, encoding="utf-8")

    loader = Loader(redactor=redactor, max_docs_per_task=100)
    docs = loader.scan(ctx)
    assert len(docs) == 1

    raw_bytes = (ctx / "README.md").read_bytes()
    assert docs[0].bytes_size == len(raw_bytes)
    assert docs[0].char_count == len(text)
    # 中文 1 字符 = 3 UTF-8 字节，所以 bytes_size > char_count。
    assert docs[0].bytes_size > docs[0].char_count


def test_loader_collection_field_is_task_corpus(
    context_dir: Path, redactor: Redactor
) -> None:
    """M4 阶段 Loader 一律返回 ``collection="task_corpus"``；shared 由独立提案。"""
    loader = Loader(redactor=redactor, max_docs_per_task=100)
    docs = loader.scan(context_dir)
    for d in docs:
        assert d.collection == "task_corpus", (
            f"M4 不应出现非 task_corpus collection：{d.collection}"
        )


# ---------------------------------------------------------------------------
# 截断与事件
# ---------------------------------------------------------------------------


def test_loader_truncates_when_exceeding_max_docs(
    tmp_path: Path, redactor: Redactor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """文档数超过 ``max_docs_per_task`` 时截断到上限。"""
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    for i in range(10):
        (ctx / f"doc_{i:02d}.md").write_text(f"content {i}", encoding="utf-8")

    loader = Loader(redactor=redactor, max_docs_per_task=5)
    docs = loader.scan(ctx)
    assert len(docs) == 5, f"max_docs_per_task=5 但收到 {len(docs)} 个文档"


def test_loader_dispatches_skipped_event_when_truncated(
    tmp_path: Path, redactor: Redactor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """截断时应 dispatch ``memory_rag_skipped(reason="max_docs_truncated")``。"""
    captured: list[tuple[str, dict[str, Any]]] = []

    def _fake_dispatch(name: str, data: dict[str, Any], config: Any | None = None) -> None:
        captured.append((name, data))

    # 替换 loader 内部使用的 dispatch（它走 observability.events）。
    monkeypatch.setattr(
        "data_agent_langchain.memory.rag.loader.dispatch_observability_event",
        _fake_dispatch,
    )

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    for i in range(10):
        (ctx / f"doc_{i:02d}.md").write_text("x", encoding="utf-8")

    loader = Loader(redactor=redactor, max_docs_per_task=3)
    loader.scan(ctx)

    skipped = [d for n, d in captured if n == "memory_rag_skipped"]
    assert skipped, f"未发现 memory_rag_skipped 事件，captured={captured}"
    assert any(d.get("reason") == "max_docs_truncated" for d in skipped), skipped
    # 事件 payload 应附带规模信息便于诊断。
    truncated = next(d for d in skipped if d.get("reason") == "max_docs_truncated")
    assert truncated.get("found") == 10
    assert truncated.get("limit") == 3


def test_loader_no_event_when_not_truncated(
    context_dir: Path, redactor: Redactor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未截断时不应 dispatch 任何 ``memory_rag_skipped`` 事件。"""
    captured: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "data_agent_langchain.memory.rag.loader.dispatch_observability_event",
        lambda name, data, config=None: captured.append((name, data)),
    )

    loader = Loader(redactor=redactor, max_docs_per_task=100)
    loader.scan(context_dir)
    assert not captured, f"未截断却 dispatch 了事件：{captured}"


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------


def test_loader_returns_empty_for_nonexistent_dir(
    tmp_path: Path, redactor: Redactor
) -> None:
    """``context_dir`` 不存在时返回空列表，不抛错。"""
    loader = Loader(redactor=redactor, max_docs_per_task=100)
    result = loader.scan(tmp_path / "missing")
    assert result == []


def test_loader_returns_empty_for_empty_dir(
    tmp_path: Path, redactor: Redactor
) -> None:
    """空目录返回空列表。"""
    ctx = tmp_path / "empty"
    ctx.mkdir()
    loader = Loader(redactor=redactor, max_docs_per_task=100)
    assert loader.scan(ctx) == []


# ---------------------------------------------------------------------------
# 边界（D8）：Loader 不接受 task_dir
# ---------------------------------------------------------------------------


def test_loader_does_not_walk_into_task_dir_sibling(
    tmp_path: Path, redactor: Redactor
) -> None:
    """传入 ``context_dir`` 时，**不**应递归到上级 task_dir 下的兄弟目录。

    防御性测试：调用方有职责传 ``task.context_dir`` 而非 ``task.task_dir``，
    Loader 自身只忠实扫描传入路径，不上行。
    """
    task = tmp_path / "task_xxx"
    task.mkdir()
    # task 根目录里放 expected_output.json（典型 ground truth 文件）。
    (task / "expected_output.json").write_text("{}", encoding="utf-8")
    ctx = task / "context"
    ctx.mkdir()
    (ctx / "README.md").write_text("hi", encoding="utf-8")

    loader = Loader(redactor=redactor, max_docs_per_task=100)
    docs = loader.scan(ctx)
    paths = {d.source_path for d in docs}
    assert paths == {"README.md"}, f"loader 越界扫到 task_dir：{paths}"


# ---------------------------------------------------------------------------
# 文本读取（factory 第 3 步使用）
# ---------------------------------------------------------------------------


def test_loader_read_document_text_round_trip(
    tmp_path: Path, redactor: Redactor
) -> None:
    """``read_document_text(doc, context_dir)`` 应原样返回文件内容。"""
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    text = "# Title\n\nbody paragraph\n"
    (ctx / "README.md").write_text(text, encoding="utf-8")

    loader = Loader(redactor=redactor, max_docs_per_task=100)
    docs = loader.scan(ctx)
    assert loader.read_document_text(docs[0], ctx) == text
