"""``Redactor`` 文件名与内容过滤行为测试（M4.1.2）。

按 ``01-design-v2.md §4.7`` 与 ``§6.2`` 验证：

  - ``is_safe_filename`` 对配置的 glob 黑名单返回 False（不安全），其他返回 True。
  - ``filter_text`` 命中任一 ``redact_patterns`` 中的正则 → 返回空字符串（整段丢弃）。
  - 不命中任何 pattern → 原样返回。
  - 多 pattern 命中其中之一即丢弃（OR 语义）。

设计偏离：将 ``Redactor.__init__`` 从设计文档 §4.7 写的
``__init__(self, cfg: CorpusRagConfig)`` 改为接受独立的两个 tuple 参数
``redact_patterns`` / ``redact_filenames``，原因：

  1. SRP —— Redactor 不依赖完整 ``CorpusRagConfig``，只需两个字段。
  2. 解除与 ``CorpusRagConfig``（M4.4.1 才引入）的循环依赖，使 M4.1.2 阶段可独立测试。
  3. ``factory.build_task_corpus`` 在 M4.4.3 阶段会简单包装：
     ``Redactor(redact_patterns=cfg.redact_patterns, redact_filenames=cfg.redact_filenames)``。

行为契约与设计完全一致。
"""
from __future__ import annotations

import pytest

from data_agent_langchain.memory.rag.redactor import Redactor


# 沿用 ``01-design-v2.md §4.3`` ``CorpusRagConfig`` 的默认值，确保单测覆盖真实配置。
_DEFAULT_PATTERNS = (
    r"(?i)\banswer\b",
    r"(?i)\bhint\b",
    r"(?i)\bapproach\b",
    r"(?i)\bsolution\b",
)
_DEFAULT_FILENAMES = (
    "expected_output.json",
    "ground_truth*",
    "*label*",
)


@pytest.fixture()
def redactor() -> Redactor:
    """默认配置下的 Redactor 实例。"""
    return Redactor(
        redact_patterns=_DEFAULT_PATTERNS,
        redact_filenames=_DEFAULT_FILENAMES,
    )


# ---------------------------------------------------------------------------
# is_safe_filename
# ---------------------------------------------------------------------------


def test_is_safe_filename_blocks_expected_output(redactor: Redactor) -> None:
    """``expected_output.json`` 精确匹配应被判定为不安全。"""
    assert redactor.is_safe_filename("expected_output.json") is False


def test_is_safe_filename_blocks_ground_truth_glob(redactor: Redactor) -> None:
    """``ground_truth*`` glob 应命中 ``ground_truth_v1.csv``、``ground_truth.json`` 等。"""
    assert redactor.is_safe_filename("ground_truth_v1.csv") is False
    assert redactor.is_safe_filename("ground_truth.json") is False


def test_is_safe_filename_blocks_label_glob(redactor: Redactor) -> None:
    """``*label*`` glob 应命中含 ``label`` 子串的文件名。"""
    assert redactor.is_safe_filename("train_labels.parquet") is False
    assert redactor.is_safe_filename("labels.csv") is False


def test_is_safe_filename_allows_readme(redactor: Redactor) -> None:
    """``README.md`` 不在任何黑名单内，应被判定为安全。"""
    assert redactor.is_safe_filename("README.md") is True


def test_is_safe_filename_allows_data_schema(redactor: Redactor) -> None:
    """``data_schema.md`` 是 task 文档典型文件，应被判定为安全。"""
    assert redactor.is_safe_filename("data_schema.md") is True


def test_is_safe_filename_is_case_insensitive(redactor: Redactor) -> None:
    """文件名匹配应大小写不敏感（Windows 文件系统兼容性）。

    ``Expected_Output.JSON`` / ``GROUND_TRUTH.csv`` 等大小写变体均应被判定为不安全。
    """
    assert redactor.is_safe_filename("Expected_Output.JSON") is False
    assert redactor.is_safe_filename("GROUND_TRUTH.csv") is False
    assert redactor.is_safe_filename("Train_Labels.csv") is False


# ---------------------------------------------------------------------------
# filter_text
# ---------------------------------------------------------------------------


def test_filter_text_drops_segment_containing_answer(redactor: Redactor) -> None:
    """文本含 ``answer`` 应整段返回空字符串（不做 mask）。"""
    text = "The answer to question is 42."
    assert redactor.filter_text(text) == ""


def test_filter_text_drops_segment_containing_hint(redactor: Redactor) -> None:
    """文本含 ``hint`` 应整段丢弃。"""
    text = "Some hint about the solution path."
    assert redactor.filter_text(text) == ""


def test_filter_text_is_case_insensitive(redactor: Redactor) -> None:
    """patterns 已带 ``(?i)`` 标志，大写 ``Answer`` 也应被丢弃。"""
    assert redactor.filter_text("Answer is 42") == ""
    assert redactor.filter_text("HINT: try X first") == ""


def test_filter_text_keeps_clean_text(redactor: Redactor) -> None:
    """不命中任何 pattern 的文本应原样返回。"""
    clean = "This is a normal sentence describing the dataset schema."
    assert redactor.filter_text(clean) == clean


def test_filter_text_multiple_patterns_or_semantics(redactor: Redactor) -> None:
    """多 pattern 命中其中之一即丢弃（OR 语义，不是 AND）。"""
    # 只含 approach（命中 \bapproach\b）。
    assert redactor.filter_text("Our approach uses bagging.") == ""
    # 只含 solution（命中 \bsolution\b）。
    assert redactor.filter_text("See solution.py for details.") == ""


def test_filter_text_word_boundary_avoids_false_positive(redactor: Redactor) -> None:
    """``\\banswer\\b`` 应只命中独立单词，不命中 ``answers``、``answered`` 等？

    设计选择：``re.search`` + ``(?i)\\banswer\\b`` 会命中 ``answers``（boundary
    在 ``answer`` 后），属于已知 over-block；本测试记录当前行为以便后续如要收窄
    再单独立项。

    具体而言：``\\banswer\\b`` 在 ``answers`` 中匹配前 6 个字符不算独立词，正则
    ``\\b`` 在 ``r`` 与 ``s`` 之间不是边界，所以 ``answers`` **不**会命中 ——
    这是符合预期的精确行为。
    """
    # ``answered`` / ``answers`` 不应被丢弃（``\b`` 在 r/s, r/e 之间非边界）。
    text = "The dataset has 100 answers labeled."
    # 但 "labeled" 命中 ``*label*`` glob ... 那是文件名层面的，文本层面不该被影响。
    # 这条文本不含独立 ``answer`` / ``hint`` / ``approach`` / ``solution``，应保留。
    assert redactor.filter_text(text) == text


def test_filter_text_empty_input(redactor: Redactor) -> None:
    """空字符串应返回空字符串，不抛错。"""
    assert redactor.filter_text("") == ""


# ---------------------------------------------------------------------------
# 不变量
# ---------------------------------------------------------------------------


def test_redactor_compiles_patterns_once() -> None:
    """构造时应一次性 compile 正则；filter_text 不重复 compile（性能不变量）。

    通过反射访问 ``_patterns`` 属性确认其为 ``tuple[re.Pattern, ...]``。
    """
    import re

    r = Redactor(
        redact_patterns=_DEFAULT_PATTERNS,
        redact_filenames=_DEFAULT_FILENAMES,
    )
    patterns = getattr(r, "_patterns")
    assert isinstance(patterns, tuple), "_patterns 应为 tuple（不可变集合）"
    assert len(patterns) == len(_DEFAULT_PATTERNS)
    assert all(isinstance(p, re.Pattern) for p in patterns), (
        "_patterns 每个元素应已 re.compile，避免 filter_text 内重复编译"
    )
