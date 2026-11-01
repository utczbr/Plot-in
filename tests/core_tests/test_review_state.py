"""Unit tests for src/core/review_state.py (T-014.0, T-014.1)."""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

from core.review_state import (
    ALL_IMAGE_STATES,
    ImageReviewState,
    STATE_AUTO,
    STATE_FAILED,
    STATE_NEEDS_REVIEW,
    STATE_PENDING,
    STATE_REVIEWED,
    STATE_UNKNOWN,
    derive_image_state,
    recompute_row_status,
)


class TestReviewStateConstants:
    """Validate ImageReviewState enumeration and string constants."""

    def test_enum_members_and_values(self):
        assert ImageReviewState.PENDING == "pending"
        assert ImageReviewState.AUTO == "auto"
        assert ImageReviewState.NEEDS_REVIEW == "needs_review"
        assert ImageReviewState.REVIEWED == "reviewed"
        assert ImageReviewState.FAILED == "failed"
        assert ImageReviewState.UNKNOWN == "unknown"

    def test_string_constants_match_enum(self):
        assert STATE_PENDING == ImageReviewState.PENDING.value
        assert STATE_AUTO == ImageReviewState.AUTO.value
        assert STATE_NEEDS_REVIEW == ImageReviewState.NEEDS_REVIEW.value
        assert STATE_REVIEWED == ImageReviewState.REVIEWED.value
        assert STATE_FAILED == ImageReviewState.FAILED.value
        assert STATE_UNKNOWN == ImageReviewState.UNKNOWN.value

    def test_all_image_states_set(self):
        assert len(ALL_IMAGE_STATES) == 6
        assert ImageReviewState.AUTO in ALL_IMAGE_STATES


class TestDeriveImageState:
    """Comprehensive unit tests for derive_image_state precedence and rules."""

    def test_failed_flag_has_highest_precedence(self):
        assert derive_image_state(None, failed=True) == STATE_FAILED
        res = {"review_status": "reviewed", "chart_type": "bar"}
        assert derive_image_state(res, failed=True) == STATE_FAILED

    def test_failed_status_in_result_or_summary(self):
        assert derive_image_state({"failed": True}) == STATE_FAILED
        assert derive_image_state({"review_status": "failed"}) == STATE_FAILED
        assert derive_image_state({"error": "Pipeline exception occurred"}) == STATE_FAILED

    def test_pending_when_result_missing_or_empty(self):
        assert derive_image_state(None) == STATE_PENDING
        assert derive_image_state({}) == STATE_PENDING

    def test_unknown_status_preserved(self):
        assert derive_image_state({"review_status": "unknown"}) == STATE_UNKNOWN

    def test_reviewed_with_zero_corrections(self):
        res = {
            "review_status": "reviewed",
            "protocol_rows": [
                {"review_status": "reviewed", "confidence": 0.95},
                {"review_status": "reviewed", "confidence": 0.92},
            ],
            "corrected_rows": 0,
        }
        assert derive_image_state(res) == STATE_REVIEWED

    def test_reviewed_invalidated_by_corrected_rows(self):
        res = {
            "review_status": "reviewed",
            "protocol_rows": [
                {"review_status": "reviewed", "confidence": 0.95},
                {"review_status": "corrected", "confidence": 0.92, "_original": {"y": 10}},
            ],
            "corrected_rows": 1,
        }
        assert derive_image_state(res) == STATE_NEEDS_REVIEW

    def test_reviewed_invalidated_by_unreviewed_original_row(self):
        res = {
            "review_status": "reviewed",
            "protocol_rows": [
                {"review_status": "reviewed", "confidence": 0.95},
                {"_original": {"y": 10}, "confidence": 0.92},
            ],
        }
        assert derive_image_state(res) == STATE_NEEDS_REVIEW

    def test_confidence_threshold_derivation(self):
        # All confidences >= 0.60
        res_ok = {
            "chart_type": "bar",
            "protocol_rows": [
                {"confidence": 0.85},
                {"confidence": 0.62},
            ],
        }
        assert derive_image_state(res_ok, threshold=0.60) == STATE_AUTO

        # One row at 0.55 below threshold 0.60
        res_low = {
            "chart_type": "bar",
            "protocol_rows": [
                {"confidence": 0.85},
                {"confidence": 0.55},
            ],
        }
        assert derive_image_state(res_low, threshold=0.60) == STATE_NEEDS_REVIEW
        # Same result with lower threshold 0.50 classifies as auto
        assert derive_image_state(res_low, threshold=0.50) == STATE_AUTO

    def test_confidence_from_summary_dict(self):
        summary_low = {"min_confidence": 0.58, "chart_type": "line"}
        assert derive_image_state(summary_low, threshold=0.60) == STATE_NEEDS_REVIEW

        summary_ok = {"min_confidence": 0.72, "chart_type": "line"}
        assert derive_image_state(summary_ok, threshold=0.60) == STATE_AUTO

    def test_calibration_quality_low_approximate_failed(self):
        for q in ("low", "approximate", "failed", "uncalibrated"):
            res = {"calibration_quality": q, "chart_type": "bar"}
            assert derive_image_state(res) == STATE_NEEDS_REVIEW

        res_high = {"calibration_quality": "high", "chart_type": "bar"}
        assert derive_image_state(res_high) == STATE_AUTO

    def test_fallback_worst_r2_under_threshold(self):
        # Simulates scatter_00002 audit finding (r2=0.765, calibration_quality missing)
        res = {
            "chart_type": "scatter",
            "calibration": {
                "x": {"r2": 0.765, "is_inverted": False},
                "y": {"r2": 0.991, "is_inverted": False},
            },
        }
        assert derive_image_state(res) == STATE_NEEDS_REVIEW

    def test_fallback_worst_r2_good(self):
        res = {
            "chart_type": "scatter",
            "calibration": {
                "x": {"r2": 0.985, "is_inverted": False},
                "y": {"r2": 0.992, "is_inverted": False},
            },
        }
        assert derive_image_state(res) == STATE_AUTO

    def test_manual_2point_calibration_skipped_in_worst_r2(self):
        # 2-point manual fits have r2=1.0 and n_points=2
        res = {
            "chart_type": "line",
            "calibration": {
                "x": {"r2": 1.0, "provenance": "manual", "n_points": 2},
                "y": {"r2": 0.98, "is_inverted": False},
            },
        }
        assert derive_image_state(res) == STATE_AUTO

    def test_unexpected_axis_inversion_triggers_needs_review(self):
        # X axis inverted is unexpected
        res_x_inverted = {
            "chart_type": "bar",
            "calibration": {
                "x": {"r2": 0.99, "is_inverted": True},
                "y": {"r2": 0.99, "is_inverted": False},
            },
        }
        assert derive_image_state(res_x_inverted) == STATE_NEEDS_REVIEW

        # Normal Cartesian chart: X not inverted
        res_normal = {
            "chart_type": "bar",
            "calibration": {
                "x": {"r2": 0.99, "is_inverted": False},
                "y": {"r2": 0.99, "is_inverted": False},
            },
        }
        assert derive_image_state(res_normal) == STATE_AUTO

    def test_box_quartile_monotonicity(self):
        # Valid monotonic box
        res_box_ok = {
            "chart_type": "box",
            "elements": [
                {
                    "whisker_low": 10.0,
                    "q1": 20.0,
                    "median": 30.0,
                    "q3": 40.0,
                    "whisker_high": 50.0,
                }
            ],
        }
        assert derive_image_state(res_box_ok) == STATE_AUTO

        # Non-monotonic box: q1 > median
        res_box_bad = {
            "chart_type": "box",
            "elements": [
                {
                    "whisker_low": 10.0,
                    "q1": 35.0,
                    "median": 30.0,
                    "q3": 40.0,
                    "whisker_high": 50.0,
                }
            ],
        }
        assert derive_image_state(res_box_bad) == STATE_NEEDS_REVIEW

        # Non-monotonic box: median > q3
        res_box_bad2 = {
            "chart_type": "box",
            "protocol_rows": [
                {
                    "min": 10.0,
                    "q1": 20.0,
                    "median": 45.0,
                    "q3": 40.0,
                    "max": 50.0,
                }
            ],
        }
        assert derive_image_state(res_box_bad2) == STATE_NEEDS_REVIEW

    def test_pie_percent_sum_error(self):
        # Sum = 100% -> auto
        res_pie_100 = {
            "chart_type": "pie",
            "elements": [{"percent": 40.0}, {"percent": 60.0}],
        }
        assert derive_image_state(res_pie_100) == STATE_AUTO

        # Sum = 104% (error <= 5%) -> auto
        res_pie_104 = {
            "chart_type": "pie",
            "elements": [{"percent": 44.0}, {"percent": 60.0}],
        }
        assert derive_image_state(res_pie_104) == STATE_AUTO

        # Sum = 108% (error > 5%) -> needs_review
        res_pie_108 = {
            "chart_type": "pie",
            "elements": [{"percent": 48.0}, {"percent": 60.0}],
        }
        assert derive_image_state(res_pie_108) == STATE_NEEDS_REVIEW

        # Sum = 92% (error > 5%) -> needs_review
        res_pie_92 = {
            "chart_type": "pie",
            "elements": [{"percent": 40.0}, {"percent": 52.0}],
        }
        assert derive_image_state(res_pie_92) == STATE_NEEDS_REVIEW

    def test_pipeline_warnings_trigger_needs_review(self):
        res_warn = {
            "chart_type": "bar",
            "warnings": ["Poor calibration fit"],
        }
        assert derive_image_state(res_warn) == STATE_NEEDS_REVIEW

        res_warn_count = {
            "chart_type": "bar",
            "warning_count": 2,
        }
        assert derive_image_state(res_warn_count) == STATE_NEEDS_REVIEW

    def test_unreviewed_corrections_trigger_needs_review(self):
        res = {
            "chart_type": "bar",
            "corrected_rows": 1,
        }
        assert derive_image_state(res) == STATE_NEEDS_REVIEW


class TestRecomputeRowStatus:
    """Unit tests for recompute_row_status semantics (ADR-014)."""

    def test_row_without_original_remains_untouched(self):
        row = {"label": "Alpha", "value": 10.0, "review_status": "auto"}
        assert recompute_row_status(row) == "auto"
        assert row["review_status"] == "auto"

    def test_float_tolerance_within_1e7(self):
        # 10.00000001 vs 10.0 has difference 1e-8, well within 1e-7 rel_tol
        row = {
            "value": 10.00000001,
            "review_status": "corrected",
            "_original": {"value": 10.0},
            "_original_status": "auto",
        }
        status = recompute_row_status(row)
        assert status == "auto"
        assert row["review_status"] == "auto"
        assert "_original" not in row
        assert "_original_status" not in row

    def test_float_change_outside_tolerance(self):
        row = {
            "value": 10.01,
            "review_status": "auto",
            "_original": {"value": 10.0},
            "_original_status": "auto",
        }
        status = recompute_row_status(row)
        assert status == "corrected"
        assert row["review_status"] == "corrected"
        assert row["_original"]["value"] == 10.0

    def test_nan_equivalence(self):
        row = {
            "value": float("nan"),
            "review_status": "corrected",
            "_original": {"value": float("nan")},
            "_original_status": "auto",
        }
        status = recompute_row_status(row)
        assert status == "auto"
        assert "_original" not in row

    def test_none_and_empty_string_equivalence(self):
        # None -> ""
        row1 = {
            "group": "",
            "review_status": "corrected",
            "_original": {"group": None},
            "_original_status": "auto",
        }
        assert recompute_row_status(row1) == "auto"
        assert "_original" not in row1

        # "" -> None
        row2 = {
            "group": None,
            "review_status": "corrected",
            "_original": {"group": ""},
            "_original_status": "auto",
        }
        assert recompute_row_status(row2) == "auto"
        assert "_original" not in row2

    def test_undo_restores_original_status(self):
        # Edit then undo
        row = {
            "label": "Cat A",
            "value": 12.0,
            "_original": {"label": "Cat A", "value": 10.0},
            "_original_status": "auto",
        }
        # First verify it is corrected
        assert recompute_row_status(row) == "corrected"

        # Undo the value edit back to 10.0
        row["value"] = 10.0
        assert recompute_row_status(row) == "auto"
        assert row["review_status"] == "auto"
        assert "_original" not in row

    def test_preserve_reviewed_original_status(self):
        # Row was originally reviewed, edited, then reverted
        row = {
            "value": 15.0,
            "review_status": "corrected",
            "_original": {"value": 15.0},
            "_original_status": "reviewed",
        }
        assert recompute_row_status(row) == "reviewed"
        assert row["review_status"] == "reviewed"
        assert "_original" not in row


class TestADR014TableAcceptance:
    """Explicitly verify the table-driven acceptance criteria from ADR-014."""

    def test_acceptance_table_cases(self):
        # 1. No result => pending
        assert derive_image_state(None) == "pending"

        # 2. Failed marker exists => failed
        assert derive_image_state(None, failed=True) == "failed"

        # 3. Result with all rows conf >= 0.6 and good calibration => auto
        res_good = {
            "chart_type": "bar",
            "calibration_quality": "high",
            "protocol_rows": [{"confidence": 0.8}, {"confidence": 0.9}],
        }
        assert derive_image_state(res_good, threshold=0.60) == "auto"

        # 4. One row 0.55 at threshold 0.6 => needs_review
        res_threshold = {
            "chart_type": "bar",
            "calibration_quality": "high",
            "protocol_rows": [{"confidence": 0.8}, {"confidence": 0.55}],
        }
        assert derive_image_state(res_threshold, threshold=0.60) == "needs_review"

        # 5. review_status='reviewed' with all rows uncorrected => reviewed
        res_rev_clean = {
            "chart_type": "bar",
            "review_status": "reviewed",
            "corrected_rows": 0,
            "protocol_rows": [{"confidence": 0.8, "review_status": "reviewed"}],
        }
        assert derive_image_state(res_rev_clean) == "reviewed"

        # 6. review_status='reviewed' with 1 corrected row => needs_review
        res_rev_corr = {
            "chart_type": "bar",
            "review_status": "reviewed",
            "corrected_rows": 1,
            "protocol_rows": [
                {"confidence": 0.8, "review_status": "corrected", "_original": {"value": 10.0}}
            ],
        }
        assert derive_image_state(res_rev_corr) == "needs_review"

        # 7. Row: float 10.00000001 edited then restored to 10.0 => restored to _original_status (e.g. auto)
        row = {
            "value": 10.0,
            "review_status": "corrected",
            "_original": {"value": 10.00000001},
            "_original_status": "auto",
        }
        assert recompute_row_status(row) == "auto"
        assert "_original" not in row
