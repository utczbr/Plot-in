"""
Empirical test and stress harness for scripts/diagnose_dataset_pipeline.py.

Verifies:
1. CLI Argument Handling (--help, --limit, --verbose, --output-dir)
2. Execution Stability, Runtime & Clean Termination (exit code 0)
3. Schema Conformance & RFC 8259 Integrity of reports/
4. Edge Case Resilience (empty inputs, zero division, outliers, malformed ground truth)
5. Adversarial Flaw Identification (subset runs KeyError, float('inf') serialization)
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "diagnose_dataset_pipeline.py"
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"

# Dynamically import scripts/diagnose_dataset_pipeline.py
spec = importlib.util.spec_from_file_location("diagnose_dataset_pipeline", str(SCRIPT_PATH))
diag = importlib.util.module_from_spec(spec)
sys.modules["diagnose_dataset_pipeline"] = diag
spec.loader.exec_module(diag)


# ===========================================================================
# 1. CLI Argument Handling & Execution Stability
# ===========================================================================

def test_cli_help_flag():
    """Verify that --help exits cleanly with code 0."""
    res = subprocess.run(
        [str(PYTHON_BIN), str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0
    assert "Dataset Pipeline Diagnostic & Ablation Harness" in res.stdout


def test_cli_limit_1_execution(tmp_path: Path):
    """Verify that --limit 1 executes cleanly and writes to output dir."""
    res = subprocess.run(
        [
            str(PYTHON_BIN),
            str(SCRIPT_PATH),
            "--limit", "1",
            "--verbose",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0
    assert "Found 1 ground-truth samples" in res.stderr or "Found 1 ground-truth samples" in res.stdout
    assert (tmp_path / "ablation_results.json").exists()
    assert (tmp_path / "propagation_matrix.json").exists()

    with (tmp_path / "ablation_results.json").open() as f:
        data = json.load(f)
    assert len(data["per_chart_ablation"]) == 1


def test_cli_limit_10_execution(tmp_path: Path):
    """Verify that --limit 10 processes exactly 10 samples."""
    res = subprocess.run(
        [
            str(PYTHON_BIN),
            str(SCRIPT_PATH),
            "--limit", "10",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0
    with (tmp_path / "ablation_results.json").open() as f:
        data = json.load(f)
    assert len(data["per_chart_ablation"]) == 10
    assert data["summary"]["total_charts"] == 10


def test_full_100_execution_and_runtime(tmp_path: Path):
    """Verify that full execution on all 100 samples completes cleanly in < 30 seconds (machine-independent budget)."""
    import time
    start = time.perf_counter()
    res = subprocess.run(
        [
            str(PYTHON_BIN),
            str(SCRIPT_PATH),
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.perf_counter() - start
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert elapsed < 30.0, f"Full execution too slow: {elapsed:.2f}s"
    assert (tmp_path / "ablation_results.json").exists()
    assert (tmp_path / "propagation_matrix.json").exists()


# ===========================================================================
# 2. Schema Conformance & RFC 8259 Compliance
# ===========================================================================

def test_ablation_results_schema_conformance():
    """Verify schema conformance of reports/ablation_results.json."""
    report_path = PROJECT_ROOT / "reports" / "ablation_results.json"
    assert report_path.exists(), "reports/ablation_results.json does not exist"

    raw_text = report_path.read_text(encoding="utf-8")
    # Verify strict RFC 8259 compliance (no NaN, Infinity, -Infinity)
    assert not re.findall(r"\b(NaN|Infinity|-Infinity)\b", raw_text), "Found non-standard NaN/Infinity in JSON"

    data = json.loads(raw_text)
    assert "summary" in data
    assert "per_chart_ablation" in data

    summary = data["summary"]
    assert summary["total_charts"] == 100
    assert summary["runs_evaluated"] == ["Run_A", "Run_B", "Run_C", "Run_D", "Run_E"]

    for r in ["Run_A", "Run_B", "Run_C", "Run_D", "Run_E"]:
        assert r in summary["run_aggregates"]
        agg = summary["run_aggregates"][r]
        assert isinstance(agg["avg_mae"], (int, float)) and not np.isnan(agg["avg_mae"])
        assert isinstance(agg["avg_mape"], (int, float)) and not np.isnan(agg["avg_mape"])
        assert isinstance(agg["avg_ccc"], (int, float)) and not np.isnan(agg["avg_ccc"])

    expected_types = {"area", "bar", "box", "histogram", "line", "scatter"}
    assert set(summary["by_chart_type"].keys()) == expected_types

    assert "stage_metrics" in summary
    for stage_key in ["stage1_detection_and_layout", "stage2_ocr_and_numeric", "stage3_spatial_classification_and_routing", "stage4_calibration_and_baseline"]:
        assert stage_key in summary["stage_metrics"]

    assert len(data["per_chart_ablation"]) == 100
    for rec in data["per_chart_ablation"]:
        assert "sample_id" in rec
        assert "chart_type" in rec
        assert "metrics" in rec
        for r in ["Run_A", "Run_B", "Run_C", "Run_D", "Run_E"]:
            m = rec["metrics"][r]
            assert isinstance(m["mae"], (int, float))
            assert isinstance(m["mape"], (int, float))
            assert isinstance(m["ccc"], (int, float))
            assert isinstance(m["count_match"], bool)


def test_propagation_matrix_schema_conformance():
    """Verify schema conformance of reports/propagation_matrix.json."""
    report_path = PROJECT_ROOT / "reports" / "propagation_matrix.json"
    assert report_path.exists(), "reports/propagation_matrix.json does not exist"

    raw_text = report_path.read_text(encoding="utf-8")
    assert not re.findall(r"\b(NaN|Infinity|-Infinity)\b", raw_text), "Found non-standard NaN/Infinity in JSON"

    data = json.loads(raw_text)
    assert "summary" in data
    assert "chart_attributions" in data

    summary = data["summary"]
    assert summary["total_charts"] == 100
    counts = summary["failure_counts_by_root_cause"]
    assert sum(counts.values()) == 100

    percentages = summary["failure_percentages"]
    assert abs(sum(percentages.values()) - 100.0) < 1e-4

    amp = summary["stage_amplification_factors"]
    assert set(amp.keys()) == {
        "Stage1_to_Stage2", "Stage2_to_Stage3", "Stage3_to_Stage4", "Stage4_to_Stage5"
    }

    assert len(data["chart_attributions"]) == 100
    for rec in data["chart_attributions"]:
        assert "sample_id" in rec
        assert "chart_type" in rec
        assert "primary_failure_cause" in rec
        assert "earliest_failing_stage" in rec
        assert "explanation" in rec
        assert "stage_metrics" in rec


# ===========================================================================
# 3. Mathematical Edge Cases & Resilience
# ===========================================================================

def test_compute_value_metrics_division_by_zero():
    """Verify that zeros in ground truth do not raise ZeroDivisionError in MAPE."""
    res = diag.compute_value_metrics([0.0, 0.0], [0.0, 0.0])
    assert res["mae"] == 0.0
    assert res["mape"] == 0.0
    assert res["ccc"] == 1.0
    assert res["count_match"] is True


def test_compute_value_metrics_zero_gt_nonzero_pred():
    """Verify MAPE behavior when GT is 0 and prediction is non-zero."""
    res = diag.compute_value_metrics([0.0], [1.0])
    assert res["mae"] == 1.0
    # Denominator uses 1e-10 epsilon, so relative error is 1.0 / 1e-10 * 100
    assert res["mape"] > 1e11
    assert res["count_match"] is True


def test_compute_value_metrics_constant_arrays():
    """Verify Lin's CCC fallback when variance is 0."""
    res_match = diag.compute_value_metrics([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert res_match["ccc"] == 1.0

    res_mismatch = diag.compute_value_metrics([5.0, 5.0, 5.0], [6.0, 6.0, 6.0])
    assert res_mismatch["ccc"] == 0.0


def test_compute_value_metrics_mismatched_lengths():
    """Verify truncation to min_len and count_match flag."""
    res = diag.compute_value_metrics([1.0, 2.0, 3.0], [1.0, 2.0])
    assert res["count_match"] is False
    assert res["num_gt"] == 3
    assert res["num_pred"] == 2
    assert res["mae"] == 0.0


def test_extract_gt_values_robustness():
    """Verify extract_gt_values against edge cases and malformed annotations."""
    assert diag.extract_gt_values({}) == ("unknown", [])
    assert diag.extract_gt_values({"chart_type": "invalid"}) == ("invalid", [])

    # Bar chart with non-numeric string and valid values
    bar_annot = {
        "chart_type": "bar",
        "annotations": [
            {"class_name": "bar", "attrs": {"value": "invalid_text"}},
            {"class_name": "bar", "attrs": {"top": 42.0}},
            {"class_name": "bar", "attrs": {}},
        ],
    }
    ctype, vals = diag.extract_gt_values(bar_annot)
    assert ctype == "bar"
    assert vals == [42.0]

    # Boundary test: string "nan" is parsed by float() without ValueError
    bar_nan_leak = {
        "chart_type": "bar",
        "annotations": [
            {"class_name": "bar", "attrs": {"value": "nan"}},
        ],
    }
    _, nan_vals = diag.extract_gt_values(bar_nan_leak)
    assert len(nan_vals) == 1 and np.isnan(nan_vals[0])


def test_extract_oracle_calibration_model_edge_cases():
    """Verify extract_oracle_calibration_model edge cases."""
    # Under-determined
    assert diag.extract_oracle_calibration_model({}) is None

    # Degenerate vertical/horizontal alignment
    degenerate_px = {
        "annotations": [
            {"class_name": "axis_labels", "text": "10", "xyxy": [10, 10, 20, 20], "attrs": {"label_role": "scale", "axis": "y"}},
            {"class_name": "axis_labels", "text": "20", "xyxy": [10, 10, 20, 20], "attrs": {"label_role": "scale", "axis": "y"}},
        ]
    }
    assert diag.extract_oracle_calibration_model(degenerate_px, axis="y") is None


def test_attribute_chart_failure_cascade():
    """Verify strict priority ordering of earliest failing stage attribution."""
    # Extractor Bug (Stage 5) takes precedence if Run A fails
    c, s, _ = diag.attribute_chart_failure({
        "Run_A": {"mape": 10.0},
        "Run_D": {"mape": 10.0},
        "Run_C": {"mape": 10.0},
        "Run_B": {"mape": 10.0},
        "Run_E": {"mape": 10.0},
    })
    assert c == diag.CAUSE_EXTRACTOR_BUG and s == 5

    # Calibration Offset (Stage 4)
    c, s, _ = diag.attribute_chart_failure({
        "Run_A": {"mape": 2.0},
        "Run_D": {"mape": 8.0},
    })
    assert c == diag.CAUSE_CALIBRATION_OFFSET and s == 4

    # Role Misclassification (Stage 3)
    c, s, _ = diag.attribute_chart_failure({
        "Run_A": {"mape": 2.0},
        "Run_D": {"mape": 2.0},
        "Run_C": {"mape": 8.0},
    })
    assert c == diag.CAUSE_ROLE_MISCLASS and s == 3

    # OCR Misread (Stage 2)
    c, s, _ = diag.attribute_chart_failure({
        "Run_A": {"mape": 2.0},
        "Run_D": {"mape": 2.0},
        "Run_C": {"mape": 2.0},
        "Run_B": {"mape": 8.0},
    })
    assert c == diag.CAUSE_OCR_MISREAD and s == 2

    # Detection Miss (Stage 1)
    c, s, _ = diag.attribute_chart_failure({
        "Run_A": {"mape": 2.0},
        "Run_D": {"mape": 2.0},
        "Run_C": {"mape": 2.0},
        "Run_B": {"mape": 2.0},
        "Run_E": {"mape": 8.0},
    })
    assert c == diag.CAUSE_DETECTION_MISS and s == 1

    # Success (Stage 0)
    c, s, _ = diag.attribute_chart_failure({
        "Run_A": {"mape": 2.0},
        "Run_D": {"mape": 2.0},
        "Run_C": {"mape": 2.0},
        "Run_B": {"mape": 2.0},
        "Run_E": {"mape": 4.0},
    })
    assert c == diag.CAUSE_NONE and s == 0


# ===========================================================================
# 4. Remediation & Methodological Regression Guards
# ===========================================================================

def test_cli_subset_runs_succeeds(tmp_path: Path):
    """Passing a subset of runs via --runs (e.g., Run_A,Run_E) exits cleanly without KeyError."""
    res = subprocess.run(
        [
            str(PYTHON_BIN),
            str(SCRIPT_PATH),
            "--runs", "Run_A,Run_E",
            "--limit", "2",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0, f"Subset runs failed: {res.stderr}"
    assert (tmp_path / "ablation_results.json").exists()


def test_empty_predictions_strict_rfc8259_json():
    """When gt_vals or pred_vals is empty, compute_value_metrics emits strict RFC 8259 finite JSON."""
    res = diag.compute_value_metrics([], [1.0, 2.0])
    assert np.isfinite(res["mae"])
    raw_json = json.dumps(res)
    assert "Infinity" not in raw_json and "NaN" not in raw_json

    def reject_constant(c):
        raise ValueError(f"Strict RFC 8259 JSON violation: non-standard constant {c}")

    parsed = json.loads(raw_json, parse_constant=reject_constant)
    assert parsed["count_match"] is False


def test_histogram_gt_uses_bar_annotations_not_bin_edges():
    """Histogram GT extraction must prefer bar annotation values over X-axis binning.edges."""
    gt_hist = {
        "chart_type": "histogram",
        "annotations": [
            {"class_name": "bar", "attrs": {"value": 12.0}},
            {"class_name": "bar", "attrs": {"value": 18.0}},
        ],
        "subplots": [{"histogram": {"binning": {"edges": [-5.0, 0.0, 5.0]}}}],
    }
    ctype, vals = diag.extract_gt_values(gt_hist)
    assert ctype == "histogram"
    assert vals == [12.0, 18.0]


def test_series_keypoints_uses_pixel_keypoints_not_data_values():
    """build_detection_dict must reconstruct line/area points from pixel keypoints, not attrs.data."""
    annots = [
        {
            "class_name": "series_keypoints",
            "keypoints": [[150.0, 420.0, 2], [250.0, 310.0, 2]],
            "attrs": {"data": [[0.0, 1.5], [1.0, 4.2]]},
        }
    ]
    dets = diag.build_detection_dict(annots, "line", jitter=False)
    assert "data_point" in dets
    pts = dets["data_point"]
    assert len(pts) == 2
    assert pts[0]["xyxy"] == [148.0, 418.0, 152.0, 422.0]
    assert pts[1]["xyxy"] == [248.0, 308.0, 252.0, 312.0]
