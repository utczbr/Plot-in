"""Unit tests for src/core/review_summary.py (T-014.1, T-014.1b)."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.review_summary import (
    build_summary,
    clear_failure_marker,
    read_summary,
    write_failure_marker,
    write_summary,
)
from pipelines.chart_pipeline import ChartAnalysisPipeline


class TestBuildSummary:
    """Test build_summary schema compliance and calculation accuracy."""

    def test_build_summary_full_result(self):
        result = {
            "image_file": "sample_chart.png",
            "chart_type": "bar",
            "review_status": "auto",
            "protocol_rows": [
                {"label": "A", "value": 10.0, "confidence": 0.90, "review_status": "auto"},
                {"label": "B", "value": 20.0, "confidence": 0.80, "review_status": "auto"},
                {
                    "label": "C",
                    "value": 35.0,
                    "confidence": 0.70,
                    "review_status": "corrected",
                    "_original": {"value": 30.0},
                },
            ],
            "metadata": {
                "calibration_quality": "high",
                "warnings": ["Minor warning 1"],
            },
        }

        summary = build_summary(result)

        assert summary["schema_version"] == "1.0"
        assert summary["image_file"] == "sample_chart.png"
        assert summary["chart_type"] == "bar"
        assert summary["row_count"] == 3
        assert summary["corrected_rows"] == 1
        assert pytest.approx(summary["min_confidence"]) == 0.70
        assert pytest.approx(summary["mean_confidence"]) == 0.80
        assert summary["calibration_quality"] == "high"
        assert summary["warning_count"] == 1
        # Has 1 corrected row and 1 warning, so derived review_status is needs_review
        assert summary["review_status"] == "needs_review"

        # Validate saved_at is a valid ISO-8601 timestamp
        dt = datetime.fromisoformat(summary["saved_at"])
        assert dt is not None

    def test_build_summary_clean_reviewed(self):
        result = {
            "image_file": "reviewed_chart.png",
            "chart_type": "line",
            "review_status": "reviewed",
            "protocol_rows": [
                {"label": "P1", "value": 5.0, "confidence": 0.95, "review_status": "reviewed"},
                {"label": "P2", "value": 10.0, "confidence": 0.92, "review_status": "reviewed"},
            ],
            "metadata": {"calibration_quality": "high"},
        }
        summary = build_summary(result)
        assert summary["row_count"] == 2
        assert summary["corrected_rows"] == 0
        assert summary["review_status"] == "reviewed"

    def test_build_summary_fallback_elements_and_worst_r2(self):
        # Result without protocol_rows or metadata calibration_quality
        result = {
            "image_file": "scatter_chart.png",
            "chart_type": "scatter",
            "elements": [
                {"x": 1.0, "y": 2.0, "confidence": 0.88},
                {"x": 2.0, "y": 4.0, "confidence": 0.92},
            ],
            "calibration": {
                "x": {"r2": 0.98},
                "y": {"r2": 0.99},
            },
        }
        summary = build_summary(result)
        assert summary["row_count"] == 2
        assert summary["corrected_rows"] == 0
        assert pytest.approx(summary["min_confidence"]) == 0.88
        assert pytest.approx(summary["mean_confidence"]) == 0.90
        assert summary["calibration_quality"] == "high"
        assert summary["review_status"] == "auto"

    def test_build_summary_minimal_empty_result(self):
        summary = build_summary({})
        assert summary["schema_version"] == "1.0"
        assert summary["image_file"] == ""
        assert summary["chart_type"] == "unknown"
        assert summary["row_count"] == 0
        assert summary["min_confidence"] is None
        assert summary["mean_confidence"] is None
        assert summary["calibration_quality"] is None
        assert summary["warning_count"] == 0
        assert summary["corrected_rows"] == 0


class TestWriteAndReadSummary:
    """Test atomic summary persistence and indentation per DP-01."""

    def test_write_summary_with_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "chart1_summary.json"
            data = {
                "schema_version": "1.0",
                "image_file": "chart1.png",
                "chart_type": "bar",
                "row_count": 5,
            }
            written_path = write_summary(out_path, data)
            assert written_path == out_path
            assert out_path.exists()

            # Verify indentation (indent=2 produces newlines and 2 spaces)
            raw = out_path.read_text(encoding="utf-8")
            assert '  "schema_version": "1.0"' in raw

            read_back = read_summary(out_path)
            assert read_back == data

    def test_write_summary_with_output_dir_and_stem(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            data = {"schema_version": "1.0", "row_count": 2}
            written_path = write_summary(out_dir, "test_stem", data)

            expected_path = out_dir / "test_stem_summary.json"
            assert written_path == expected_path
            assert expected_path.exists()
            assert read_summary(expected_path) == data

    def test_read_summary_nonexistent_returns_none(self):
        assert read_summary("/nonexistent/path/summary.json") is None


class TestFailureMarker:
    """Test failure marker creation and deletion (T-014.1b / T-010.1 dependency)."""

    def test_write_and_clear_failure_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            stem = "problem_chart"
            err_msg = "Axis calibration failed: insufficient inliers"

            # Write marker
            marker_path = write_failure_marker(out_dir, stem, err_msg)
            assert marker_path == out_dir / f"{stem}_failed.json"
            assert marker_path.exists()

            with open(marker_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            assert content["schema_version"] == "1.0"
            assert content["image_stem"] == stem
            assert content["error_message"] == err_msg
            assert content["reason"] == err_msg
            assert "timestamp" in content

            # Clear marker
            cleared = clear_failure_marker(out_dir, stem)
            assert cleared is True
            assert not marker_path.exists()

            # Clearing already removed marker returns False without error
            assert clear_failure_marker(out_dir, stem) is False


class TestPipelineSaveHookIntegration:
    """Verify that ChartAnalysisPipeline._save_results generates summary and clears failure markers."""

    def test_save_results_creates_summary_and_clears_failure_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            stem = "sample_hook_test"

            # Pre-create a failure marker to verify it gets cleared
            marker_path = write_failure_marker(out_dir, stem, "Prior transient failure")
            assert marker_path.exists()

            pipeline = ChartAnalysisPipeline.__new__(ChartAnalysisPipeline)
            pipeline.logger = MagicMock()

            result = {
                "image_file": f"{stem}.png",
                "chart_type": "bar",
                "review_status": "auto",
                "protocol_rows": [
                    {"label": "Cat A", "value": 10.0, "confidence": 0.85, "review_status": "auto"}
                ],
                "metadata": {"calibration_quality": "high"},
            }

            img = MagicMock()
            pipeline._save_results(result, img, out_dir, annotated=False, output_stem=stem)

            # 1. Analysis JSON was saved
            assert (out_dir / f"{stem}_analysis.json").exists()

            # 2. Summary JSON was saved
            summary_path = out_dir / f"{stem}_summary.json"
            assert summary_path.exists()

            summary = read_summary(summary_path)
            assert summary["schema_version"] == "1.0"
            assert summary["chart_type"] == "bar"
            assert summary["row_count"] == 1
            assert summary["review_status"] == "auto"

            # 3. Prior failure marker was cleared
            assert not marker_path.exists()
