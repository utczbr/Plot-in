"""Image review state model and row-level status derivation.

Defines the normative review states (ADR-014) and pure helper functions for
deriving image review state and recomputing row-level edit statuses.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, Optional, Union


class ImageReviewState(str, Enum):
    """Normative per-image review states defined in ADR-014."""
    PENDING = "pending"
    AUTO = "auto"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"
    FAILED = "failed"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


# String constants for direct comparison and convenience
STATE_PENDING = ImageReviewState.PENDING.value
STATE_AUTO = ImageReviewState.AUTO.value
STATE_NEEDS_REVIEW = ImageReviewState.NEEDS_REVIEW.value
STATE_REVIEWED = ImageReviewState.REVIEWED.value
STATE_FAILED = ImageReviewState.FAILED.value
STATE_UNKNOWN = ImageReviewState.UNKNOWN.value

ALL_IMAGE_STATES = frozenset({
    ImageReviewState.PENDING,
    ImageReviewState.AUTO,
    ImageReviewState.NEEDS_REVIEW,
    ImageReviewState.REVIEWED,
    ImageReviewState.FAILED,
    ImageReviewState.UNKNOWN,
})


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two cell values with float tolerance, NaN matching, and None/empty string equivalence."""
    a_empty = (a is None or a == "")
    b_empty = (b is None or b == "")
    if a_empty and b_empty:
        return True
    if a_empty or b_empty:
        return False

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))

    a_float = None
    b_float = None
    if isinstance(a, (int, float)):
        a_float = float(a)
    elif isinstance(a, str):
        try:
            a_float = float(a)
        except ValueError:
            pass

    if isinstance(b, (int, float)):
        b_float = float(b)
    elif isinstance(b, str):
        try:
            b_float = float(b)
        except ValueError:
            pass

    if a_float is not None and b_float is not None:
        if math.isnan(a_float) and math.isnan(b_float):
            return True
        if math.isnan(a_float) or math.isnan(b_float):
            return False
        return math.isclose(a_float, b_float, rel_tol=1e-7, abs_tol=1e-9)

    return a == b


def recompute_row_status(row: Dict[str, Any]) -> str:
    """
    Recompute row review_status by comparing editable fields with row.get('_original').

    - Uses math.isclose(rel_tol=1e-7, abs_tol=1e-9) for floats.
    - Treats NaN as equivalent to NaN.
    - Treats None and "" as equivalent.
    - If identical to _original: restores _original_status (default 'auto') and removes _original.
    - If changed: sets review_status to 'corrected' and returns 'corrected'.
    """
    original = row.get("_original")
    if original is None:
        return row.get("review_status", "auto")

    if not isinstance(original, dict):
        if _values_equal(row.get("value"), original):
            restored = row.pop("_original_status", "auto")
            row.pop("_original", None)
            row["review_status"] = restored
            return restored
        else:
            row["review_status"] = "corrected"
            return "corrected"

    is_changed = False
    for key, orig_val in original.items():
        curr_val = row.get(key)
        if not _values_equal(curr_val, orig_val):
            is_changed = True
            break

    if not is_changed:
        restored = row.pop("_original_status", "auto")
        row.pop("_original", None)
        row["review_status"] = restored
        return restored
    else:
        row["review_status"] = "corrected"
        return "corrected"


def derive_image_state(
    result_or_summary: Optional[Dict[str, Any]],
    threshold: float = 0.60,
    failed: bool = False,
) -> str:
    """
    Derive the image review state from an analysis result dictionary or summary dictionary.

    Precedence order per ADR-014:
        failed > unknown > (reviewed iff no rows are corrected) > needs_review > auto;
        pending applies when no result exists.

    Criteria for needs_review:
    - Min confidence < threshold
    - calibration_quality in {'low', 'approximate', 'failed', 'uncalibrated'}
    - fallback worst_r2 < 0.95
    - unexpected axis inversion (e.g. inverted X axis)
    - non-monotonic box quartiles
    - pie sum error > 5% (sum outside [95%, 105%])
    - pipeline warnings > 0
    - unreviewed row corrections (corrected_rows > 0)
    """
    if failed:
        return ImageReviewState.FAILED.value

    if result_or_summary is None:
        return ImageReviewState.PENDING.value

    if not isinstance(result_or_summary, dict) or not result_or_summary:
        return ImageReviewState.PENDING.value

    if (
        result_or_summary.get("failed") is True
        or result_or_summary.get("review_status") == "failed"
        or bool(result_or_summary.get("error"))
    ):
        return ImageReviewState.FAILED.value

    if result_or_summary.get("review_status") == "unknown":
        return ImageReviewState.UNKNOWN.value

    # Extract corrected rows count
    corrected_rows = result_or_summary.get("corrected_rows")
    protocol_rows = result_or_summary.get("protocol_rows") or []
    if corrected_rows is None:
        corrected_rows = sum(
            1 for r in protocol_rows
            if isinstance(r, dict) and (r.get("review_status") == "corrected" or r.get("_original") is not None)
        )

    review_status = result_or_summary.get("review_status")
    if review_status == "reviewed" and corrected_rows == 0:
        return ImageReviewState.REVIEWED.value

    # --- Needs Review Criteria ---
    # 1. Unreviewed corrections or reviewed invalidation by correction
    if corrected_rows > 0 or review_status == "corrected":
        return ImageReviewState.NEEDS_REVIEW.value
    if review_status == "reviewed":  # corrected_rows > 0 already handled, but safeguard
        return ImageReviewState.NEEDS_REVIEW.value

    # 2. Pipeline warnings
    warning_count = result_or_summary.get("warning_count")
    if warning_count is None:
        warnings = result_or_summary.get("warnings")
        if warnings is None:
            warnings = result_or_summary.get("metadata", {}).get("warnings")
        if isinstance(warnings, list):
            warning_count = len(warnings)
        elif warnings:
            warning_count = 1
        else:
            warning_count = 0
    if warning_count > 0:
        return ImageReviewState.NEEDS_REVIEW.value

    # 3. Confidence threshold
    min_conf = result_or_summary.get("min_confidence")
    if min_conf is None:
        confs = []
        for r in protocol_rows:
            if isinstance(r, dict) and r.get("confidence") is not None:
                try:
                    confs.append(float(r["confidence"]))
                except (TypeError, ValueError):
                    pass
        for e in result_or_summary.get("elements") or []:
            if isinstance(e, dict) and e.get("confidence") is not None:
                try:
                    confs.append(float(e["confidence"]))
                except (TypeError, ValueError):
                    pass
        if confs:
            min_conf = min(confs)
    if min_conf is not None and min_conf < threshold:
        return ImageReviewState.NEEDS_REVIEW.value

    # 4. Calibration quality & fallback worst_r2
    cal_quality = result_or_summary.get("calibration_quality")
    if not cal_quality:
        cal_quality = result_or_summary.get("metadata", {}).get("calibration_quality")

    if cal_quality in {"low", "approximate", "failed", "uncalibrated"}:
        return ImageReviewState.NEEDS_REVIEW.value

    worst_r2 = result_or_summary.get("worst_r2")
    if worst_r2 is None:
        worst_r2 = result_or_summary.get("metadata", {}).get("worst_r2")

    calibrations = result_or_summary.get("calibration") or {}
    if worst_r2 is None and isinstance(calibrations, dict):
        r2_list = []
        for cal in calibrations.values():
            if isinstance(cal, dict) and cal.get("r2") is not None:
                # Skip 2-point manual fits where n_points == 2 (ADR-014 T-014.0)
                if cal.get("provenance") == "manual" and cal.get("n_points") == 2:
                    continue
                try:
                    r2_list.append(float(cal["r2"]))
                except (TypeError, ValueError):
                    pass
        if r2_list:
            worst_r2 = min(r2_list)

    if worst_r2 is not None:
        if worst_r2 < 0.95:
            return ImageReviewState.NEEDS_REVIEW.value
        if not cal_quality:
            try:
                from calibration.conformal import derive_calibration_quality
                derived_q = derive_calibration_quality(worst_r2)
                if derived_q in {"low", "approximate", "failed", "uncalibrated"}:
                    return ImageReviewState.NEEDS_REVIEW.value
            except ImportError:
                pass

    # 5. Unexpected axis inversion (e.g. inverted X axis)
    if isinstance(calibrations, dict):
        x_cal = calibrations.get("x")
        if isinstance(x_cal, dict) and x_cal.get("is_inverted") is True:
            return ImageReviewState.NEEDS_REVIEW.value
        for cal in calibrations.values():
            if isinstance(cal, dict) and cal.get("axis") == "x" and cal.get("is_inverted") is True:
                return ImageReviewState.NEEDS_REVIEW.value

    # 6. Non-monotonic box quartiles
    chart_type = result_or_summary.get("chart_type")
    if chart_type == "box":
        box_items = list(protocol_rows) + list(result_or_summary.get("elements") or [])
        for item in box_items:
            if not isinstance(item, dict):
                continue
            for key_seq in [
                ("whisker_low", "q1", "median", "q3", "whisker_high"),
                ("min", "q1", "median", "q3", "max"),
            ]:
                vals = []
                for k in key_seq:
                    if item.get(k) is not None:
                        try:
                            vals.append(float(item[k]))
                        except (TypeError, ValueError):
                            pass
                if len(vals) >= 2:
                    if any(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
                        return ImageReviewState.NEEDS_REVIEW.value

    # 7. Pie percent sum error > 5%
    if chart_type == "pie":
        pie_items = (
            list(protocol_rows)
            or list(result_or_summary.get("elements") or [])
            or list(result_or_summary.get("slices") or [])
        )
        percents = []
        for item in pie_items:
            if isinstance(item, dict):
                p = item.get("percent")
                if p is None and "percent" not in item:
                    p = item.get("percentage")
                if p is not None:
                    try:
                        percents.append(float(p))
                    except (TypeError, ValueError):
                        pass
        if percents:
            total_pct = sum(percents)
            if total_pct <= 1.5:  # Fraction form (1.0 = 100%)
                total_pct *= 100.0
            if abs(total_pct - 100.0) > 5.0:
                return ImageReviewState.NEEDS_REVIEW.value

    return ImageReviewState.AUTO.value
