"""bench_comparison.py 中 §6 评分机制实现的单元测试（TDD）。

按计划文件 .windsurf/plans/bench-comparison-scoring-f656c9.md §6 全部 case 写。
跑：``pytest tests/test_bench_comparison_scoring.py -v``。

实现尚未完成时本文件应全部 RED；实现完成后应全部 GREEN。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 把项目根加进 sys.path，让 bench_comparison（项目根的脚本）可被 import。
# 现有 tests/conftest.py 只把 src/ 加进 sys.path，没覆盖项目根。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import bench_comparison as bc  # noqa: E402


# ---------------------------------------------------------------------------
# §6.1 normalize_cell：§6.5 标准化规则
# ---------------------------------------------------------------------------


class TestNormalizeNull:
    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "  ",
            "null",
            "NULL",
            "Null",
            "none",
            "NONE",
            "nan",
            "NaN",
            "nat",
            "NAT",
            "<na>",
            "<NA>",
        ],
    )
    def test_null_literals_to_empty(self, raw: str) -> None:
        assert bc.normalize_cell(raw) == ""


class TestNormalizeNumeric:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("4200000", "4200000.00"),
            ("0.005", "0.01"),  # ROUND_HALF_UP 关键样例
            ("0.004", "0.00"),
            ("-3.14159", "-3.14"),
            ("0", "0.00"),
            ("100.5", "100.50"),
            ("1e3", "1000.00"),
            ("-0.005", "-0.01"),  # 负数 ROUND_HALF_UP 远离零
        ],
    )
    def test_decimal_round_half_up(self, raw: str, expected: str) -> None:
        assert bc.normalize_cell(raw) == expected


class TestNormalizeDate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2024-03-01", "2024-03-01"),
            ("2024-3-1", "2024-03-01"),
            ("2024/3/1", "2024-03-01"),
            ("2024.3.1", "2024-03-01"),
        ],
    )
    def test_iso_normalization(self, raw: str, expected: str) -> None:
        assert bc.normalize_cell(raw) == expected


class TestNormalizeDatetime:
    def test_datetime_with_tz_to_utc_z(self) -> None:
        # +08:00 → UTC 减 8 小时
        assert bc.normalize_cell("2024-03-01T08:00:00+08:00") == "2024-03-01T00:00:00Z"

    def test_datetime_with_z_suffix(self) -> None:
        # 已经是 UTC 'Z' 结尾，应保持
        assert bc.normalize_cell("2024-03-01T00:00:00Z") == "2024-03-01T00:00:00Z"

    def test_datetime_without_tz_keeps_iso(self) -> None:
        # 不带时区，转 ISO 格式（空格替换为 T）
        assert bc.normalize_cell("2024-03-01 08:00:00") == "2024-03-01T08:00:00"

    def test_datetime_iso_without_tz(self) -> None:
        assert bc.normalize_cell("2024-03-01T08:00:00") == "2024-03-01T08:00:00"


class TestNormalizeString:
    def test_strip_whitespace_and_crlf(self) -> None:
        assert bc.normalize_cell("  hello\r\n") == "hello"

    def test_strip_inner_crlf(self) -> None:
        # \r\n 在中间也会被去掉（保守解读 §6.5："去除 \r\n"）
        assert bc.normalize_cell("hello\r\nworld") == "helloworld"

    def test_case_sensitive(self) -> None:
        assert bc.normalize_cell("East Asia") == "East Asia"
        assert bc.normalize_cell("east asia") == "east asia"
        assert bc.normalize_cell("East Asia") != bc.normalize_cell("east asia")

    def test_full_name_kept(self) -> None:
        assert bc.normalize_cell("John Smith") == "John Smith"

    def test_random_words_kept_as_string(self) -> None:
        # 不是日期、不是数字、不是 null → 字符串原样保留（trim 后）
        assert bc.normalize_cell("Tuesday") == "Tuesday"
        assert bc.normalize_cell("yes") == "yes"


# ---------------------------------------------------------------------------
# §6.2 column_signature
# ---------------------------------------------------------------------------


class TestColumnSignature:
    def test_row_order_irrelevant(self) -> None:
        s1 = pd.Series(["a", "b", "c"])
        s2 = pd.Series(["c", "a", "b"])
        assert bc.column_signature(s1) == bc.column_signature(s2)

    def test_duplicate_values_preserved_in_signature(self) -> None:
        s1 = pd.Series(["a", "a", "b"])
        s2 = pd.Series(["a", "b", "b"])
        assert bc.column_signature(s1) != bc.column_signature(s2)

    def test_signature_normalizes_cells(self) -> None:
        # "1" 与 "1.00" 标准化为同一个；"null" 与 "" 也是
        s1 = pd.Series(["1", "2.000", "null"])
        s2 = pd.Series(["", "1.00", "2.00"])
        assert bc.column_signature(s1) == bc.column_signature(s2)

    def test_signature_is_tuple(self) -> None:
        sig = bc.column_signature(pd.Series(["a", "b"]))
        assert isinstance(sig, tuple)


# ---------------------------------------------------------------------------
# §6.3 compute_task_score
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    """简单 CSV 写入辅助，使用 \\n 分隔，UTF-8。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(",".join(r) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")


class TestComputeTaskScore:
    def test_perfect_match(self, tmp_path: Path) -> None:
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(pred, [["A", "B"], ["1", "x"], ["2", "y"]])
        _write_csv(gold, [["A", "B"], ["1", "x"], ["2", "y"]])
        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)
        assert info["score"] == 1.0
        assert info["recall"] == 1.0
        assert info["matched_cols"] == 2
        assert info["gold_cols"] == 2
        assert info["pred_cols"] == 2
        assert info["extra_cols"] == 0

    def test_one_extra_column_penalty(self, tmp_path: Path) -> None:
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(pred, [["A", "B", "C"], ["1", "x", "n"], ["2", "y", "t"]])
        _write_csv(gold, [["A", "B"], ["1", "x"], ["2", "y"]])
        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)
        assert info["recall"] == 1.0
        assert info["pred_cols"] == 3
        assert info["matched_cols"] == 2
        assert info["extra_cols"] == 1
        # score = 1.0 - 0.1 * (1/3) ≈ 0.9667
        assert info["score"] == pytest.approx(1.0 - 0.1 / 3, abs=1e-3)

    def test_missing_one_column_recall_drops(self, tmp_path: Path) -> None:
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(pred, [["A"], ["1"], ["2"]])
        _write_csv(gold, [["A", "B"], ["1", "x"], ["2", "y"]])
        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)
        assert info["matched_cols"] == 1
        assert info["recall"] == 0.5
        # extra=0 (pred_cols=1, matched=1) → 无惩罚
        assert info["score"] == 0.5

    def test_column_names_ignored(self, tmp_path: Path) -> None:
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(
            pred,
            [["wrong_name", "another_wrong"], ["1", "x"], ["2", "y"]],
        )
        _write_csv(gold, [["A", "B"], ["1", "x"], ["2", "y"]])
        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)
        assert info["score"] == 1.0

    def test_row_order_ignored(self, tmp_path: Path) -> None:
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(pred, [["A", "B"], ["2", "y"], ["1", "x"]])
        _write_csv(gold, [["A", "B"], ["1", "x"], ["2", "y"]])
        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)
        assert info["score"] == 1.0

    def test_prediction_missing(self, tmp_path: Path) -> None:
        pred = tmp_path / "pred.csv"  # 不创建
        gold = tmp_path / "gold.csv"
        _write_csv(gold, [["A"], ["1"]])
        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)
        assert info["score"] == 0.0
        assert info.get("error")
        assert "不存在" in info["error"]

    def test_placeholder_csv_yields_zero(self, tmp_path: Path) -> None:
        # 提交占位 b"result\r\n"：1 列名、0 行
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        pred.write_bytes(b"result\r\n")
        _write_csv(gold, [["A", "B"], ["1", "x"], ["2", "y"]])
        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)
        assert info["score"] == 0.0
        assert info["matched_cols"] == 0
        assert info["pred_cols"] == 1
        assert info["gold_cols"] == 2

    def test_lambda_changes_score(self, tmp_path: Path) -> None:
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(pred, [["A", "B", "C"], ["1", "x", "n"], ["2", "y", "t"]])
        _write_csv(gold, [["A", "B"], ["1", "x"], ["2", "y"]])
        info_05 = bc.compute_task_score(pred, gold, lambda_penalty=0.5)
        # score = 1.0 - 0.5 * (1/3) ≈ 0.8333
        assert info_05["score"] == pytest.approx(1.0 - 0.5 / 3, abs=1e-3)

    def test_score_clamped_at_zero(self, tmp_path: Path) -> None:
        # gold 1 列、pred 50 列且都不匹配 → score = 0 - 0.1*1 = -0.1 → clamp 0
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        cols = [f"c{i}" for i in range(50)]
        rows = [[str(i + j * 1000) for j in range(50)] for i in range(3)]
        _write_csv(pred, [cols] + rows)
        _write_csv(gold, [["only_col"], ["unique_value"]])
        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)
        assert info["score"] == 0.0  # 不能为负

    def test_numeric_normalization_aligns_pred_gold(self, tmp_path: Path) -> None:
        # pred 写整数, gold 写 2 位小数 → 标准化后应一致
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(pred, [["A"], ["100"], ["200"]])
        _write_csv(gold, [["A"], ["100.00"], ["200.00"]])
        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)
        assert info["score"] == 1.0

    def test_null_normalization_aligns_pred_gold(self, tmp_path: Path) -> None:
        # pred 用 "null" / "NaN", gold 用 "" → 标准化后一致
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(pred, [["A"], ["null"], ["NaN"], ["1"]])
        _write_csv(gold, [["A"], [""], [""], ["1.00"]])
        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)
        assert info["score"] == 1.0

    def test_name_split_gold_matches_full_name_prediction(self, tmp_path: Path) -> None:
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(
            pred,
            [["full_name"], ["Trent Smith"], ["Tyler Hewitt"], ["Annabella Warren"]],
        )
        _write_csv(
            gold,
            [
                ["first_name", "last_name"],
                ["Trent", "Smith"],
                ["Tyler", "Hewitt"],
                ["Annabella", "Warren"],
            ],
        )

        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)

        assert info["score"] == 1.0
        assert info["recall"] == 1.0
        assert info["matched_cols"] == 1
        assert info["gold_cols"] == 1
        assert info["pred_cols"] == 1
        assert info["extra_cols"] == 0

    def test_name_split_gold_matches_full_name_plus_metric_prediction(
        self, tmp_path: Path
    ) -> None:
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(pred, [["full_name", "total_cost"], ["Sacha Harrison", "866.25"]])
        _write_csv(
            gold,
            [["first_name", "last_name", "SUM(T2.cost)"], ["Sacha", "Harrison", "866.25"]],
        )

        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)

        assert info["score"] == 1.0
        assert info["recall"] == 1.0
        assert info["matched_cols"] == 2
        assert info["gold_cols"] == 2
        assert info["pred_cols"] == 2
        assert info["extra_cols"] == 0

    def test_member_name_prediction_matches_first_last_gold(self, tmp_path: Path) -> None:
        pred = tmp_path / "pred.csv"
        gold = tmp_path / "gold.csv"
        _write_csv(pred, [["member_name", "cost"], ["Elijah Allen", "28.15"]])
        _write_csv(gold, [["first_name", "last_name", "cost"], ["Elijah", "Allen", "28.15"]])

        info = bc.compute_task_score(pred, gold, lambda_penalty=0.1)

        assert info["score"] == 1.0
        assert info["recall"] == 1.0
        assert info["matched_cols"] == 2
        assert info["gold_cols"] == 2
        assert info["pred_cols"] == 2
        assert info["extra_cols"] == 0


# ---------------------------------------------------------------------------
# §6.4 check_full_coverage
# ---------------------------------------------------------------------------


class TestCheckFullCoverage:
    def test_all_present(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        run_dir = tmp_path / "run"
        for tid in ("task_1", "task_2", "task_3"):
            (input_dir / tid).mkdir(parents=True)
            (run_dir / tid).mkdir(parents=True)
            (run_dir / tid / "prediction.csv").touch()
        is_complete, missing = bc.check_full_coverage(input_dir, run_dir)
        assert is_complete is True
        assert missing == []

    def test_partial_missing(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        run_dir = tmp_path / "run"
        for tid in ("task_1", "task_2", "task_3"):
            (input_dir / tid).mkdir(parents=True)
        # 只 task_1 有 prediction.csv
        (run_dir / "task_1").mkdir(parents=True)
        (run_dir / "task_1" / "prediction.csv").touch()
        is_complete, missing = bc.check_full_coverage(input_dir, run_dir)
        assert is_complete is False
        assert sorted(missing) == ["task_2", "task_3"]

    def test_empty_input_treated_as_complete(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        run_dir = tmp_path / "run"
        input_dir.mkdir()
        run_dir.mkdir()
        is_complete, missing = bc.check_full_coverage(input_dir, run_dir)
        assert is_complete is True
        assert missing == []

    def test_ignores_non_task_dirs(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        run_dir = tmp_path / "run"
        (input_dir / "task_1").mkdir(parents=True)
        # 干扰目录与文件
        (input_dir / ".ipynb_checkpoints").mkdir(parents=True)
        (input_dir / "README.md").touch()
        (run_dir / "task_1").mkdir(parents=True)
        (run_dir / "task_1" / "prediction.csv").touch()
        is_complete, missing = bc.check_full_coverage(input_dir, run_dir)
        assert is_complete is True
        assert missing == []

    def test_run_dir_missing_task_subfolder(self, tmp_path: Path) -> None:
        # 即使 task 目录都不存在于 run_dir（更彻底的失败），也应当报缺失
        input_dir = tmp_path / "input"
        run_dir = tmp_path / "run"
        (input_dir / "task_1").mkdir(parents=True)
        (input_dir / "task_2").mkdir(parents=True)
        run_dir.mkdir()
        is_complete, missing = bc.check_full_coverage(input_dir, run_dir)
        assert is_complete is False
        assert sorted(missing) == ["task_1", "task_2"]
