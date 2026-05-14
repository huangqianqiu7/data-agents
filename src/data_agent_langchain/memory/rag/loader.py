"""Corpus RAG 文档扫描器（M4.1.3）。

按 ``01-design-v2.md §4.7`` / ``D8`` 实现：

  - :meth:`Loader.scan` 扫描调用方指定的 ``context_dir``（务必传入
    ``task.context_dir`` 而非 ``task.task_dir``，避免误读 ``expected_output.json``）。
  - 仅收入文件名安全（``Redactor.is_safe_filename=True``）且扩展名已知
    （``.md`` / ``.txt`` / ``.json``）的文件。
  - 超过 ``max_docs_per_task`` 的部分被截断；同时 dispatch 一个
    ``memory_rag_skipped(reason="max_docs_truncated")`` 事件，便于 metrics 聚合。
  - 文档元数据：``doc_id = sha1(source_path|size|mtime)[:16]``，``source_path``
    使用相对于 ``context_dir`` 的 POSIX 风格相对路径。

设计偏离：
  - 构造器接受独立的 ``redactor`` / ``max_docs_per_task`` 参数（非
    ``CorpusRagConfig`` 整体），与 ``Redactor`` / ``Chunker`` 一致地保持解耦。
  - 当前 M4 阶段 ``.doc`` / ``.docx`` 二进制文档被跳过（不识别），未来如需支持
    再单独立项；目前 task ``context_dir`` 实际只有 markdown / text / json schema。

Loader 同时提供 :meth:`read_document_text` 让 ``factory.build_task_corpus``
在 ``Redactor.filter_text`` 与 ``Chunker.chunk`` 之前读取正文。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from data_agent_langchain.memory.rag.documents import CorpusDocument, DocKind
from data_agent_langchain.memory.rag.redactor import Redactor
from data_agent_langchain.observability.events import dispatch_observability_event


# 扩展名 → ``DocKind`` 的固定映射；不在白名单内的扩展名直接跳过。
_EXTENSION_TO_KIND: dict[str, DocKind] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".json": "json_schema_note",
}


class Loader:
    """task ``context_dir`` 文档扫描器。

    Args:
        redactor: 文件名安全过滤器；``is_safe_filename=False`` 的文件被跳过。
        max_docs_per_task: 单个 task 最多收入的文档数；超出时截断。
    """

    __slots__ = ("_redactor", "_max_docs")

    def __init__(self, *, redactor: Redactor, max_docs_per_task: int) -> None:
        self._redactor = redactor
        self._max_docs = max_docs_per_task

    def scan(self, context_dir: Path) -> list[CorpusDocument]:
        """扫描 ``context_dir`` 并返回符合条件的文档列表。

        - 不存在或非目录 → 返回 ``[]``。
        - 递归扫描子目录；返回的 ``source_path`` 是相对于 ``context_dir``
          的 POSIX 风格相对路径。
        - 文件按相对路径排序，保证多次调用返回顺序一致。
        - 超过 ``max_docs_per_task`` 时截断到上限并 dispatch 事件。
        """
        if not context_dir.is_dir():
            return []

        # 一次性收集所有候选文件相对路径（POSIX 风格），便于排序与截断决策。
        candidates: list[Path] = []
        for path in self._iter_files(context_dir):
            if path.suffix.lower() not in _EXTENSION_TO_KIND:
                continue
            if not self._redactor.is_safe_filename(path.name):
                continue
            candidates.append(path)

        # 排序：按 POSIX 相对路径，确保结果稳定（不同 OS 文件系统枚举顺序可能不同）。
        candidates.sort(key=lambda p: p.relative_to(context_dir).as_posix())

        truncated = False
        if len(candidates) > self._max_docs:
            dispatch_observability_event(
                "memory_rag_skipped",
                {
                    "reason": "max_docs_truncated",
                    "found": len(candidates),
                    "limit": self._max_docs,
                },
            )
            candidates = candidates[: self._max_docs]
            truncated = True
        # 即使未截断也保留变量便于阅读；no-op。
        del truncated

        docs: list[CorpusDocument] = []
        for path in candidates:
            doc = self._build_document(path, context_dir)
            if doc is not None:
                docs.append(doc)
        return docs

    def read_document_text(self, doc: CorpusDocument, context_dir: Path) -> str:
        """读取 ``doc.source_path`` 对应的文件正文（UTF-8）。

        - 错误处理：解码失败时 ``errors="replace"`` 用 ``\ufffd`` 替代非法字节。
        - 不再做 redact / chunk —— 那是调用方（``factory.build_task_corpus``）
          的职责。
        """
        path = context_dir / doc.source_path
        return path.read_text(encoding="utf-8", errors="replace")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_files(root: Path) -> Iterable[Path]:
        """递归枚举 ``root`` 下所有普通文件（不跟符号链接到外部目录）。"""
        for child in root.rglob("*"):
            # ``rglob`` 会枚举目录条目；只保留普通文件。
            if child.is_file():
                yield child

    def _build_document(
        self, path: Path, context_dir: Path
    ) -> CorpusDocument | None:
        """从 fs 路径构造 :class:`CorpusDocument`；解码失败时返回 None。"""
        rel = path.relative_to(context_dir).as_posix()
        kind = _EXTENSION_TO_KIND[path.suffix.lower()]
        try:
            stat = path.stat()
        except OSError:
            return None
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        doc_id = hashlib.sha1(
            f"{rel}|{stat.st_size}|{stat.st_mtime}".encode("utf-8")
        ).hexdigest()[:16]
        return CorpusDocument(
            doc_id=doc_id,
            source_path=rel,
            doc_kind=kind,
            bytes_size=int(stat.st_size),
            char_count=len(text),
            collection="task_corpus",
        )


__all__ = ["Loader"]
