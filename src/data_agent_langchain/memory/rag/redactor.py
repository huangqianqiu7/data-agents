"""Corpus RAG 内容过滤器（M4.1.2）。

按 ``01-design-v2.md §4.7`` / ``§6.2`` 实现：

  - :meth:`Redactor.is_safe_filename` —— 用 ``fnmatch`` glob（大小写不敏感）
    判断文件名是否落在 ``redact_filenames`` 黑名单内。命中 → 不安全。
  - :meth:`Redactor.filter_text` —— 任一 ``redact_patterns`` 正则在文本中
    搜到子串，即将整段文本丢弃（返回空字符串）。**不做** mask 替换，因为
    mask 会破坏文本结构、引入伪造内容；整段丢弃是更安全的策略。

设计偏离（与 ``§4.7`` 接口签名）：构造器从 ``__init__(cfg)`` 改为接受
``redact_patterns`` / ``redact_filenames`` 两个 tuple 参数；这样 Redactor
不依赖 ``CorpusRagConfig`` 完整定义，便于在 milestone M4.1.2 阶段独立测试，
也符合单一职责原则。``factory.build_task_corpus`` 在 M4.4.3 简单包装：
``Redactor(redact_patterns=cfg.redact_patterns, redact_filenames=cfg.redact_filenames)``。
"""
from __future__ import annotations

import fnmatch
import re
from typing import Sequence


class Redactor:
    """文件名 + 内容双层过滤器。

    Args:
        redact_patterns: 正则字符串元组；任一 pattern 命中文本即整段丢弃。
            约定 patterns 自带必要的 ``(?i)`` 标志（设计文档 §4.3 默认 patterns
            都带 ``(?i)``）；本类**不**自动追加 ``re.IGNORECASE``。
        redact_filenames: ``fnmatch`` glob 字符串元组；命中即文件不安全。
            匹配走 ``fnmatch.fnmatchcase`` + 小写化（达到大小写不敏感效果）。
    """

    __slots__ = ("_patterns", "_filename_globs")

    def __init__(
        self,
        *,
        redact_patterns: Sequence[str],
        redact_filenames: Sequence[str],
    ) -> None:
        # 一次性 compile：避免 ``filter_text`` 在每次调用时重新编译正则。
        self._patterns: tuple[re.Pattern[str], ...] = tuple(
            re.compile(p) for p in redact_patterns
        )
        # 不预编译 fnmatch；fnmatch 没有 compile 步骤，glob 数量也很少。
        self._filename_globs: tuple[str, ...] = tuple(redact_filenames)

    def is_safe_filename(self, name: str) -> bool:
        """判断文件名是否安全。

        命中任一 ``redact_filenames`` glob → 不安全（返回 False）。
        匹配大小写不敏感：把 ``name`` 与 glob 都转小写后再 ``fnmatchcase``，
        以兼容 Windows 文件系统的大小写折叠语义并避免依赖 ``fnmatch.fnmatch``
        的平台依赖行为（POSIX 大小写敏感、Windows 不敏感）。
        """
        lower_name = name.lower()
        for glob in self._filename_globs:
            if fnmatch.fnmatchcase(lower_name, glob.lower()):
                return False
        return True

    def filter_text(self, text: str) -> str:
        """文本内容过滤。

        任一 ``redact_patterns`` 在文本中 ``re.search`` 命中 → 返回空字符串
        （整段丢弃）。空输入直接返回空串，不抛错。
        """
        if not text:
            return ""
        for pattern in self._patterns:
            if pattern.search(text):
                return ""
        return text


__all__ = ["Redactor"]
