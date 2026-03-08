"""Tests for src/validation/run_protocol_validation.py."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from validation.run_protocol_validation import run_protocol_validation


STANDARD_COLUMNS = [
    "source_file",
    "page_index",
    "chart_type",
    "group",
    "outcome",
    "value",
    "unit",
    "error_bar_type",
]


def _row(
    source_file: str,
    page_index: str,
    chart_type: str,
    group: str,
    outcome: str,
    value: float,
    unit: str = "u",
    error_bar_type: str = "SD",
) -> Dict[str, str]:
    return {
        "source_file": source_file,
        "page_index": page_index,
        "chart_type": chart_type,
        "group": group,
        "outcome": outcome,
        "value": str(value),
        "unit": unit,
        "error_bar_type": error_bar_type,
    }


def _write_csv(path: Path, rows: List[Dict[str, str]], columns: Optional[Sequence[str]] = None) -> None:
    fieldnames = list(columns) if columns is not None else STANDARD_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_validation(
    tmp_path: Path,
    *,
    pred_rows: List[Dict[str, str]],
    gold_rows: List[Dict[str, str]],
    pred_columns: Optional[Sequence[str]] = None,
    gold_columns: Optional[Sequence[str]] = None,
    **kwargs,
):
    pred_csv = tmp_path / "pred.csv"
    gold_csv = tmp_path / "gold.csv"
    out_json = tmp_path / "report.json"

    _write_csv(pred_csv, pred_rows, pred_columns)
    _write_csv(gold_csv, gold_rows, gold_columns)

    code = run_protocol_validation(
        pred_csv=str(pred_csv),
        gold_csv=str(gold_csv),
        out_json=str(out_json),
        **kwargs,
    )
    report = json.loads(out_json.read_text(encoding="utf-8"))
    return code, report, out_json


def _passing_gold_pred_rows():
    gold_rows = [
        _row("doc_a.pdf", "0", "bar", "A", "Weight", 1.0, unit="g", error_bar_type="SD"),
        _row("doc_a.pdf", "0", "line", "B", "Volume", 2.0, unit="ml", error_bar_type="SEM"),
    ]
    pred_rows = [
        _row("doc_a.pdf", "0", "bar", "A", "Weight", 1.0, unit="g", error_bar_type="SD"),
        _row("doc_a.pdf", "0", "line", "B", "Volume", 2.0, unit="ml", error_bar_type="SEM"),
    ]
    return gold_rows, pred_rows


def test_pass_all_gates(tmp_path: Path):
    gold_rows, pred_rows = _passing_gold_pred_rows()
    code, report, _ = _run_validation(tmp_path, pred_rows=pred_rows, gold_rows=gold_rows)

    assert code == 0
    assert report["all_thresholds_met"] is True
    assert report["metrics"]["success_rate"] == pytest.approx(1.0)
    assert report["metrics"]["accuracy"] == pytest.approx(1.0)
    assert report["metrics"]["ccc"] == pytest.approx(1.0)
    assert report["metrics"]["cohens_kappa"] == pytest.approx(1.0)


def test_fail_low_ccc(tmp_path: Path):
    gold_rows, pred_rows = _passing_gold_pred_rows()
    pred_rows[0]["value"] = "2.0"
    pred_rows[1]["value"] = "1.0"

    code, report, _ = _run_validation(tmp_path, pred_rows=pred_rows, gold_rows=gold_rows)

    assert code == 2
    assert report["gates"]["ccc"]["pass"] is False
    assert report["metrics"]["ccc"] is not None
    assert report["metrics"]["ccc"] < 0.90


def test_fail_low_kappa(tmp_path: Path):
    gold_rows, pred_rows = _passing_gold_pred_rows()
    pred_rows[0]["unit"] = "ml"
    pred_rows[0]["error_bar_type"] = "SEM"
    pred_rows[1]["unit"] = "g"
    pred_rows[1]["error_bar_type"] = "SD"

    code, report, _ = _run_validation(tmp_path, pred_rows=pred_rows, gold_rows=gold_rows)

    assert code == 2
    assert report["gates"]["cohens_kappa"]["pass"] is False
    assert report["metrics"]["cohens_kappa"] is not None
    assert report["metrics"]["cohens_kappa"] < 0.81


def test_fail_low_accuracy(tmp_path: Path):
    gold_rows = [
        _row("doc_a.pdf", "0", "bar", "A", "Weight", 1.0, unit="g", error_bar_type="SD"),
        _row("doc_a.pdf", "0", "line", "B", "Volume", 2.0, unit="ml", error_bar_type="SEM"),
        _row("doc_a.pdf", "0", "scatter", "C", "Count", 3.0, unit="n", error_bar_type="SE"),
    ]
    pred_rows = [dict(r) for r in gold_rows]
    pred_rows[2]["unit"] = "wrong_unit"

    code, report, _ = _run_validation(tmp_path, pred_rows=pred_rows, gold_rows=gold_rows)

    assert code == 2
    assert report["gates"]["accuracy"]["pass"] is False
    assert report["metrics"]["accuracy"] == pytest.approx(2.0 / 3.0)


def test_unmatched_rows_reduce_success_rate(tmp_path: Path):
    gold_rows, pred_rows = _passing_gold_pred_rows()
    pred_rows.append(_row("doc_extra.pdf", "0", "bar", "X", "Extra", 9.9, unit="g", error_bar_type="SD"))

    code, report, _ = _run_validation(tmp_path, pred_rows=pred_rows, gold_rows=gold_rows)

    assert code == 2
    assert report["alignment"]["matched"] == 2
    assert report["alignment"]["unmatched_pred"] == 1
    assert report["alignment"]["unmatched_gold"] == 0
    assert report["metrics"]["success_rate"] == pytest.approx(2.0 / 3.0)


def test_schema_error_missing_columns(tmp_path: Path):
    gold_rows, pred_rows = _passing_gold_pred_rows()
    columns_without_value = [c for c in STANDARD_COLUMNS if c != "value"]

    code, report, _ = _run_validation(
        tmp_path,
        pred_rows=pred_rows,
        gold_rows=gold_rows,
        pred_columns=columns_without_value,
    )

    assert code == 1
    assert "error" in report
    assert report["all_thresholds_met"] is False


def test_empty_csvs_schema_valid(tmp_path: Path):
    code, report, _ = _run_validation(tmp_path, pred_rows=[], gold_rows=[])

    assert code == 2
    assert "error" not in report
    assert report["alignment"]["total"] == 0
    assert report["metrics"]["success_rate"] == pytest.approx(1.0)
    assert report["metrics"]["accuracy"] == pytest.approx(1.0)
    assert report["metrics"]["ccc"] is None
    assert report["metrics"]["cohens_kappa"] is None


def test_timing_comparison_included(tmp_path: Path):
    gold_rows, pred_rows = _passing_gold_pred_rows()
    code, report, _ = _run_validation(
        tmp_path,
        pred_rows=pred_rows,
        gold_rows=gold_rows,
        pred_runtime_seconds=100.0,
        manual_runtime_seconds=200.0,
        comparator_runtime_seconds=150.0,
    )

    assert code == 0
    assert "runtime" in report
    assert report["runtime"]["ratio_vs_manual"] == pytest.approx(0.5)
    assert report["runtime"]["savings_pct_vs_manual"] == pytest.approx(50.0)
    assert report["runtime"]["ratio_vs_comparator"] == pytest.approx(100.0 / 150.0)


def test_timing_not_included_when_absent(tmp_path: Path):
    gold_rows, pred_rows = _passing_gold_pred_rows()
    code, report, _ = _run_validation(tmp_path, pred_rows=pred_rows, gold_rows=gold_rows)

    assert code == 0
    assert "runtime" not in report


def test_report_always_written(tmp_path: Path):
    gold_rows, pred_rows = _passing_gold_pred_rows()
    pred_rows[0]["value"] = "2.0"
    pred_rows[1]["value"] = "1.0"

    code, report, out_json = _run_validation(tmp_path, pred_rows=pred_rows, gold_rows=gold_rows)

    assert code == 2
    assert out_json.exists()
    assert report["all_thresholds_met"] is False


def test_custom_thresholds(tmp_path: Path):
    gold_rows, pred_rows = _passing_gold_pred_rows()
    pred_rows.append(_row("doc_extra.pdf", "0", "bar", "X", "Extra", 9.9, unit="g", error_bar_type="SD"))

    code, report, _ = _run_validation(
        tmp_path,
        pred_rows=pred_rows,
        gold_rows=gold_rows,
        min_success_rate=0.50,
        min_accuracy=0.50,
        min_ccc=0.90,
        min_kappa=-1.0,
    )

    assert code == 0
    assert report["gates"]["success_rate"]["threshold"] == pytest.approx(0.50)
    assert report["gates"]["accuracy"]["threshold"] == pytest.approx(0.50)
    assert report["gates"]["ccc"]["threshold"] == pytest.approx(0.90)
    assert report["gates"]["cohens_kappa"]["threshold"] == pytest.approx(-1.0)


def test_alignment_diagnostics(tmp_path: Path):
    gold_rows, pred_rows = _passing_gold_pred_rows()
    gold_rows.append(_row("doc_gold_only.pdf", "0", "bar", "G", "GoldOnly", 7.0))
    pred_rows.append(_row("doc_pred_only.pdf", "0", "bar", "P", "PredOnly", 8.0))

    code, report, _ = _run_validation(tmp_path, pred_rows=pred_rows, gold_rows=gold_rows)

    assert code == 2
    assert report["alignment"]["matched"] == 2
    assert report["alignment"]["unmatched_gold"] == 1
    assert report["alignment"]["unmatched_pred"] == 1
    assert report["alignment"]["total"] == 4


# ---------------------------------------------------------------------------
# require_ccc / require_kappa flag tests
# ---------------------------------------------------------------------------


def test_skip_ccc_gate_when_not_required(tmp_path: Path):
    """When CCC is None and require_ccc=False, the gate should pass (skipped)."""
    gold_rows = [_row("doc.pdf", "0", "bar", "A", "W", 1.0)]
    pred_rows = [_row("doc.pdf", "0", "bar", "A", "W", 1.0)]

    code, report, _ = _run_validation(
        tmp_path,
        pred_rows=pred_rows,
        gold_rows=gold_rows,
        require_ccc=False,
        require_kappa=False,
        min_success_rate=1.0,
        min_accuracy=1.0,
    )

    assert report["metrics"]["ccc"] is None
    assert report["gates"]["ccc"]["pass"] is True
    assert report["gates"]["ccc"]["skipped"] is True


def test_skip_kappa_gate_when_not_required(tmp_path: Path):
    """When Kappa is None and require_kappa=False, the gate should pass (skipped)."""
    gold_rows = [_row("doc.pdf", "0", "bar", "A", "W", 1.0)]
    pred_rows = [_row("doc.pdf", "0", "bar", "A", "W", 1.0)]

    code, report, _ = _run_validation(
        tmp_path,
        pred_rows=pred_rows,
        gold_rows=gold_rows,
        require_ccc=False,
        require_kappa=False,
        min_success_rate=1.0,
        min_accuracy=1.0,
    )

    assert report["gates"]["cohens_kappa"]["pass"] is True
    assert report["gates"]["cohens_kappa"]["skipped"] is True


def test_empty_csvs_pass_when_not_required(tmp_path: Path):
    """Empty CSVs should pass all gates when CCC/Kappa are not required."""
    code, report, _ = _run_validation(
        tmp_path, pred_rows=[], gold_rows=[],
        require_ccc=False, require_kappa=False,
    )

    assert code == 0
    assert report["all_thresholds_met"] is True
    assert report["metrics"]["ccc"] is None
    assert report["metrics"]["cohens_kappa"] is None
    assert report["gates"]["ccc"]["skipped"] is True
    assert report["gates"]["cohens_kappa"]["skipped"] is True


def test_require_flags_default_to_true(tmp_path: Path):
    """When require flags are not specified (default=True), None metrics fail gates."""
    gold_rows = [_row("doc.pdf", "0", "bar", "A", "W", 1.0)]
    pred_rows = [_row("doc.pdf", "0", "bar", "A", "W", 1.0)]

    code, report, _ = _run_validation(
        tmp_path, pred_rows=pred_rows, gold_rows=gold_rows,
    )

    assert code == 2
    assert report["gates"]["ccc"]["pass"] is False
    assert report["gates"]["ccc"].get("skipped", False) is False


# ---------------------------------------------------------------------------
# Per-chart-type breakdown tests
# ---------------------------------------------------------------------------


def test_per_chart_type_in_report(tmp_path: Path):
    """Report should include per_chart_type breakdown when there are matched rows."""
    gold_rows, pred_rows = _passing_gold_pred_rows()
    code, report, _ = _run_validation(tmp_path, pred_rows=pred_rows, gold_rows=gold_rows)

    assert "per_chart_type" in report
    assert "bar" in report["per_chart_type"]
    assert "line" in report["per_chart_type"]
    assert report["per_chart_type"]["bar"]["count"] == 1
    assert report["per_chart_type"]["bar"]["accuracy"] == pytest.approx(1.0)
    assert report["per_chart_type"]["line"]["count"] == 1
    assert report["per_chart_type"]["line"]["accuracy"] == pytest.approx(1.0)


def test_per_chart_type_with_mismatch(tmp_path: Path):
    """Per-type accuracy should reflect per-type signature mismatches."""
    gold_rows = [
        _row("doc.pdf", "0", "bar", "A", "W", 1.0, unit="g", error_bar_type="SD"),
        _row("doc.pdf", "0", "bar", "B", "W", 2.0, unit="g", error_bar_type="SD"),
        _row("doc.pdf", "1", "line", "C", "V", 3.0, unit="ml", error_bar_type="SEM"),
    ]
    pred_rows = [dict(r) for r in gold_rows]
    pred_rows[0]["unit"] = "wrong"  # bar mismatch

    code, report, _ = _run_validation(
        tmp_path, pred_rows=pred_rows, gold_rows=gold_rows,
        require_ccc=False, require_kappa=False,
    )

    assert "per_chart_type" in report
    assert report["per_chart_type"]["bar"]["accuracy"] == pytest.approx(0.5)
    assert report["per_chart_type"]["line"]["accuracy"] == pytest.approx(1.0)


def test_per_chart_type_absent_for_empty(tmp_path: Path):
    """per_chart_type should not appear when there are no matched rows."""
    code, report, _ = _run_validation(
        tmp_path, pred_rows=[], gold_rows=[],
        require_ccc=False, require_kappa=False,
    )

    assert "per_chart_type" not in report


# ---------------------------------------------------------------------------
# Fixture-based end-to-end tests
# ---------------------------------------------------------------------------


_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "protocol"


def test_fixture_perfect_pass(tmp_path: Path):
    """pred_perfect.csv should pass all gates against gold.csv."""
    out_json = tmp_path / "report.json"
    code = run_protocol_validation(
        pred_csv=str(_FIXTURE_DIR / "pred_perfect.csv"),
        gold_csv=str(_FIXTURE_DIR / "gold.csv"),
        out_json=str(out_json),
    )
    report = json.loads(out_json.read_text(encoding="utf-8"))

    assert code == 0
    assert report["all_thresholds_met"] is True
    assert report["metrics"]["ccc"] == pytest.approx(1.0)
    assert report["metrics"]["accuracy"] == pytest.approx(1.0)
    assert "per_chart_type" in report
    assert len(report["per_chart_type"]) == 4  # bar, line, scatter, box


def test_fixture_failing_fails(tmp_path: Path):
    """pred_failing.csv should fail CCC gate against gold.csv."""
    out_json = tmp_path / "report.json"
    code = run_protocol_validation(
        pred_csv=str(_FIXTURE_DIR / "pred_failing.csv"),
        gold_csv=str(_FIXTURE_DIR / "gold.csv"),
        out_json=str(out_json),
    )
    report = json.loads(out_json.read_text(encoding="utf-8"))

    assert code == 2
    assert report["all_thresholds_met"] is False
    assert report["gates"]["ccc"]["pass"] is False
