"""Corpus RAG 的文档与 chunk dataclass（M4.1.1）。

按 ``01-design-v2.md §4.4`` 定义两个 ``frozen=True, slots=True`` 的不可变值
对象：

  - :class:`CorpusDocument` —— 表示从 task ``context_dir`` 扫描得到的一个原文档
    的元数据（不含正文）。``doc_id`` 由调用方按 ``sha1(source_path + size + mtime)[:16]``
    计算保证去重稳定。
  - :class:`CorpusChunk` —— 表示文档切片后的最小 embedding 单位。``chunk_id``
    必须形如 ``f"{doc_id}#{ord:04d}"`` 以便在召回结果中反查 ``CorpusDocument``。

设计禁令（§6.1 数据流禁令）：两个 dataclass **故意**没有 ``question`` /
``answer`` / ``hint`` / ``approach`` / ``predicted_label`` / ``summary`` 字段，
确保 corpus 索引层面无法承载评测态语义信息。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ``Literal`` 字面量类型联合：限定 ``CorpusDocument.doc_kind`` 取值，
# 与 ``Loader`` 的 MIME / 扩展名识别逻辑一一对应。
DocKind = Literal["markdown", "text", "doc"]


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """表示一个 task ``context_dir`` 内待索引文档的元数据（不含正文）。

    Attributes:
        doc_id: 16 字符的 sha1 摘要，由 ``sha1(source_path + size + mtime)[:16]``
            生成；用于在 chroma collection 内做去重。
        source_path: 相对于 task ``context_dir`` 的相对路径（POSIX 风格），
            用于召回结果中给 LLM 提供来源指示。
        doc_kind: 文档类型分类，限定为 ``markdown`` / ``text`` / ``doc``
            三类；其他类型在 ``Loader`` 阶段被过滤掉。``.json`` 已从白名单
            移除（大 JSON 数据文件导致 CPU 索引超时，且 agent 已有
            ``read_json`` / ``execute_python`` 工具处理结构化数据）。
        bytes_size: 原始文件字节数；用于诊断与 ``Loader`` 的最大尺寸截断。
        char_count: 解码后字符数；用于 ``Chunker`` 估算切片个数。
        collection: 归属 collection 名，``"task_corpus"`` 或 ``"shared:<name>"``。
            v3.1 M4 仅实现 ``task_corpus`` 路径；shared 由独立提案落地。
    """

    doc_id: str
    source_path: str
    doc_kind: DocKind
    bytes_size: int
    char_count: int
    collection: str


@dataclass(frozen=True, slots=True)
class CorpusChunk:
    """表示一个 ``CorpusDocument`` 切片后的最小 embedding 单位。

    Attributes:
        chunk_id: 全局唯一切片标识，必须形如 ``f"{doc_id}#{ord:04d}"`` 以便
            在召回结果中反查所属文档；调用方（``Chunker``）负责生成。
        doc_id: 所属 ``CorpusDocument.doc_id``。
        ord: 切片在文档中的序号，从 0 开始递增。
        text: 切片正文；必须已通过 ``Redactor`` 过滤。
        char_offset: 切片在原文中的起始字符偏移。
        char_length: 切片字符长度（``len(text)``）。
    """

    chunk_id: str
    doc_id: str
    ord: int
    text: str
    char_offset: int
    char_length: int


__all__ = ["CorpusDocument", "CorpusChunk", "DocKind"]
