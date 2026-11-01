"""Export Pipeline v2 - Tidy CSV Schema v2 with Invariant 35-Column Union Header.

Pure Python module: ZERO Qt imports.
Resolves T-011.2 per ADR-011, DP-01, DP-04, and DP-07.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from calibration.conformal import derive_calibration_quality
from core.protocol_row_builder import build_protocol_rows
from core.review_state import derive_image_state

# ---------------------------------------------------------------------------
# Normative Invariant Union Header (exactly 35 columns in strict order)
# ---------------------------------------------------------------------------
EXPORT_V2_COLUMNS: List[str] = [
    'source_file',
    'page_index',
    'image_width_px',
    'image_height_px',
    'chart_type',
    'series_id',
    'group',
    'outcome',
    'value',
    'value_x',
    'unit',
    'unit_x',
    'scale_type_y',
    'scale_type_x',
    'calibration_quality_y',
    'calibration_quality_x',
    'r2_y',
    'r2_x',
    'baseline_value',
    'uncertainty_minus',
    'uncertainty_plus',
    'confidence_interval_95_low',
    'confidence_interval_95_high',
    'error_bar_type',
    'error_bar_value',
    'detection_confidence',
    'ocr_confidence',
    'overall_confidence',
    'needs_review',
    'review_status',
    'reviewed_by',
    'reviewed_at',
    'correction_count',
    'is_divergent',
    'notes',
]


def _format_number(val: Any) -> str:
    """Format numeric values with full lossless float precision (repr) or empty string."""
    if val is None or val == "":
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return str(val)
    try:
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return ""
        if isinstance(val, str) and val.strip().isdigit():
            return val.strip()
        # In Python 3.8+, repr(float) guarantees shortest IEEE-754 roundtrip
        return repr(fval)
    except (TypeError, ValueError):
        return str(val)


def _resolve_element(
    row: Dict[str, Any],
    elements: List[Dict[str, Any]],
    row_idx: int,
) -> Optional[Dict[str, Any]]:
    """Resolve pipeline element corresponding to a protocol row."""
    elem_idx = row.get("element_index")
    if elem_idx is not None and isinstance(elem_idx, int) and 0 <= elem_idx < len(elements):
        return elements[elem_idx]
    if elem_idx is not None and elements:
        for e in elements:
            if isinstance(e, dict) and e.get("index") == elem_idx:
                return e
    if 0 <= row_idx < len(elements):
        return elements[row_idx]
    return None


def _build_row_dict_35(
    row: Dict[str, Any],
    result: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    row_idx: int = 0,
) -> Dict[str, str]:
    """Extract and format a single protocol row into the invariant 35-column mapping."""
    elements = result.get("elements") or []
    elem = _resolve_element(row, elements, row_idx) if isinstance(elements, list) else None
    calibration = result.get("calibration") or {}
    metadata = result.get("metadata") or {}
    provenance = result.get("_provenance") or {}

    chart_type = str(row.get("chart_type") or result.get("chart_type") or "").lower()

    # Source file & page index
    source_file = (
        row.get("source_file")
        or result.get("image_file")
        or provenance.get("source_document")
        or ""
    )
    page_index = row.get("page_index")
    if page_index is None:
        page_index = provenance.get("page_index")
    page_index_str = str(page_index) if page_index is not None and page_index != "" else ""

    # Image dimensions
    w_px = (
        row.get("image_width_px")
        or result.get("image_width_px")
        or result.get("image_width")
        or metadata.get("image_width")
    )
    h_px = (
        row.get("image_height_px")
        or result.get("image_height_px")
        or result.get("image_height")
        or metadata.get("image_height")
    )
    if w_px is None or h_px is None:
        img_size = result.get("image_size")
        if isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
            if w_px is None:
                w_px = img_size[0]
            if h_px is None:
                h_px = img_size[1]

    # Core data fields
    series_id = str(row.get("series_id") or "")
    group = str(row.get("group") or "")
    outcome = str(row.get("outcome") or "")
    value = row.get("value")
    value_x = row.get("value_x") if row.get("value_x") is not None else row.get("x")
    if value_x is None and elem:
        value_x = elem.get("x") or elem.get("x_value")

    unit = str(row.get("unit") or "")
    unit_x = str(
        row.get("unit_x")
        or (context.get("unit_x") if context else "")
        or ""
    )

    # Axis routing & calibration resolution
    axis = row.get("axis") or (elem.get("axis") if elem else None) or "primary"
    if axis == "secondary" and isinstance(calibration.get("secondary"), dict):
        cal_y = calibration["secondary"]
    else:
        cal_y = calibration.get("y") or calibration.get("primary")

    cal_x = calibration.get("x")

    # R2
    r2_y_raw = row.get("r2_y")
    if r2_y_raw is None:
        r2_y_raw = row.get("calibration_r2")
    if r2_y_raw is None and isinstance(cal_y, dict):
        r2_y_raw = cal_y.get("r2")

    r2_x_raw = row.get("r2_x")
    if r2_x_raw is None and isinstance(cal_x, dict):
        r2_x_raw = cal_x.get("r2")

    # Scale types
    scale_type_y = row.get("scale_type_y") or row.get("axis_scale_y")
    if not scale_type_y:
        if isinstance(cal_y, dict):
            scale_type_y = cal_y.get("scale_type")
        if not scale_type_y and chart_type in ("bar", "line", "scatter", "box", "histogram") and (cal_y or calibration):
            scale_type_y = "unknown"
        elif not scale_type_y:
            scale_type_y = ""

    scale_type_x = row.get("scale_type_x") or row.get("axis_scale_x")
    if not scale_type_x:
        if isinstance(cal_x, dict):
            scale_type_x = cal_x.get("scale_type")
        if not scale_type_x and chart_type in ("scatter", "line") and cal_x:
            scale_type_x = "unknown"
        elif not scale_type_x:
            scale_type_x = ""

    # Calibration quality
    cal_qual_y = row.get("calibration_quality_y") or row.get("calibration_quality")
    if not cal_qual_y:
        meta_qual = metadata.get("calibration_quality")
        if meta_qual:
            cal_qual_y = str(meta_qual)
        elif r2_y_raw is not None:
            try:
                cal_qual_y = derive_calibration_quality(float(r2_y_raw))
            except Exception:
                cal_qual_y = "uncalibrated"
        elif chart_type in ("bar", "line", "scatter", "box"):
            cal_qual_y = "uncalibrated"
        else:
            cal_qual_y = ""

    cal_qual_x = row.get("calibration_quality_x")
    if not cal_qual_x and r2_x_raw is not None:
        try:
            cal_qual_x = derive_calibration_quality(float(r2_x_raw))
        except Exception:
            cal_qual_x = ""
    elif not cal_qual_x:
        cal_qual_x = ""

    # Baseline value
    baseline_value = row.get("baseline_value")

    # Uncertainties and intervals
    unc_minus = row.get("uncertainty_minus")
    if unc_minus is None and elem:
        unc_minus = elem.get("uncertainty_minus")
    if unc_minus is None and elem:
        unc_minus = elem.get("uncertainty")
    if unc_minus is None:
        unc_minus = row.get("uncertainty")

    unc_plus = row.get("uncertainty_plus")
    if unc_plus is None and elem:
        unc_plus = elem.get("uncertainty_plus")
    if unc_plus is None and elem:
        unc_plus = elem.get("uncertainty")
    if unc_plus is None:
        unc_plus = row.get("uncertainty")

    ci_low = row.get("confidence_interval_95_low")
    if ci_low is None:
        ci_low = row.get("value_lo")
    if ci_low is None and elem:
        ci_list = elem.get("confidence_interval_95")
        if isinstance(ci_list, (list, tuple)) and len(ci_list) >= 1:
            ci_low = ci_list[0]

    ci_high = row.get("confidence_interval_95_high")
    if ci_high is None:
        ci_high = row.get("value_hi")
    if ci_high is None and elem:
        ci_list = elem.get("confidence_interval_95")
        if isinstance(ci_list, (list, tuple)) and len(ci_list) >= 2:
            ci_high = ci_list[1]

    # Error bar
    error_bar_type = str(row.get("error_bar_type") or "")
    error_bar_val = row.get("error_bar_value")
    if error_bar_val is None and elem and isinstance(elem.get("error_bar"), dict):
        error_bar_val = elem["error_bar"].get("margin")

    # Confidences
    det_conf = row.get("detection_confidence")
    if det_conf is None and elem:
        det_conf = elem.get("conf") if elem.get("conf") is not None else elem.get("detection_confidence")
    if det_conf is None:
        det_conf = row.get("confidence")

    ocr_conf = row.get("ocr_confidence")
    if ocr_conf is None and elem:
        ocr_conf = elem.get("ocr_confidence")

    overall_conf = row.get("overall_confidence")
    if overall_conf is None:
        overall_conf = row.get("confidence")
    if overall_conf is None and elem:
        overall_conf = elem.get("confidence")

    # Needs review & review status
    needs_review_raw = row.get("needs_review")
    if needs_review_raw is not None:
        needs_review_str = str(bool(needs_review_raw))
    else:
        is_nr = False
        if row.get("review_status") == "needs_review":
            is_nr = True
        elif result.get("review_status") == "needs_review":
            is_nr = True
        else:
            try:
                is_nr = (derive_image_state(result) == "needs_review")
            except Exception:
                is_nr = False
        needs_review_str = str(is_nr)

    review_status_str = str(row.get("review_status") or result.get("review_status") or "auto")
    reviewed_by_str = str(row.get("reviewed_by") or result.get("reviewed_by") or metadata.get("reviewed_by") or "")
    reviewed_at_str = str(row.get("reviewed_at") or result.get("reviewed_at") or metadata.get("reviewed_at") or "")

    # Correction count
    corr = row.get("correction_count")
    if corr is not None:
        try:
            correction_count_str = str(int(corr))
        except (TypeError, ValueError):
            correction_count_str = str(corr)
    elif row.get("review_status") == "corrected" or row.get("_original") is not None:
        correction_count_str = "1"
    else:
        correction_count_str = "0"

    # Divergence
    is_div_raw = row.get("is_divergent")
    if is_div_raw is not None:
        is_divergent_str = str(bool(is_div_raw))
    elif result.get("is_divergent") is not None:
        is_divergent_str = str(bool(result["is_divergent"]))
    else:
        is_divergent_str = "False"

    notes_str = str(row.get("notes") or "")

    return {
        'source_file': str(source_file),
        'page_index': page_index_str,
        'image_width_px': _format_number(w_px),
        'image_height_px': _format_number(h_px),
        'chart_type': chart_type,
        'series_id': series_id,
        'group': group,
        'outcome': outcome,
        'value': _format_number(value),
        'value_x': _format_number(value_x),
        'unit': unit,
        'unit_x': unit_x,
        'scale_type_y': str(scale_type_y),
        'scale_type_x': str(scale_type_x),
        'calibration_quality_y': str(cal_qual_y),
        'calibration_quality_x': str(cal_qual_x),
        'r2_y': _format_number(r2_y_raw),
        'r2_x': _format_number(r2_x_raw),
        'baseline_value': _format_number(baseline_value),
        'uncertainty_minus': _format_number(unc_minus),
        'uncertainty_plus': _format_number(unc_plus),
        'confidence_interval_95_low': _format_number(ci_low),
        'confidence_interval_95_high': _format_number(ci_high),
        'error_bar_type': error_bar_type,
        'error_bar_value': _format_number(error_bar_val),
        'detection_confidence': _format_number(det_conf),
        'ocr_confidence': _format_number(ocr_conf),
        'overall_confidence': _format_number(overall_conf),
        'needs_review': needs_review_str,
        'review_status': review_status_str,
        'reviewed_by': reviewed_by_str,
        'reviewed_at': reviewed_at_str,
        'correction_count': correction_count_str,
        'is_divergent': is_divergent_str,
        'notes': notes_str,
    }


def export_tidy_csv(
    result: Union[Dict[str, Any], List[Dict[str, Any]]],
    out_csv_path: Union[str, Path],
    context: Optional[Dict[str, Any]] = None,
    *,
    scope: str = "all",
    filters: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True,
) -> Path:
    """Export pipeline results to Tidy CSV Schema v2 with invariant 35-column union header.

    Args:
        result: Analysis result dictionary, or list of analysis result dicts.
        out_csv_path: Destination path for the exported CSV file.
        context: Optional context dictionary for outcome/unit/error_bar_type mapping.
        scope: Export scope ('all' or 'visible').
        filters: Optional active filter mappings when scope='visible'.
        include_metadata: If True, writes a <stem>.meta.json sidecar next to the CSV.

    Returns:
        Path to the successfully written CSV.

    Raises:
        ValueError: On post-write validation failure (header mismatch or row count mismatch).
    """
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize inputs into a list of (result_dict, rows_list)
    result_list: List[Dict[str, Any]] = (
        result if isinstance(result, list) else [result]
    )

    formatted_rows: List[Dict[str, str]] = []
    image_metadata_list: List[Dict[str, Any]] = []

    for res in result_list:
        if not isinstance(res, dict):
            continue

        # Dynamic row regeneration per ADR-011
        protocol_rows = res.get("protocol_rows")
        if not protocol_rows:
            protocol_rows = build_protocol_rows(res, context)

        # Apply filtering if scope is 'visible' and filters are specified
        candidate_rows = protocol_rows
        if scope == "visible" and filters:
            outcome_f = filters.get("outcome")
            group_f = filters.get("group")
            status_f = filters.get("review_status") or filters.get("status")

            if outcome_f and outcome_f != "All":
                candidate_rows = [r for r in candidate_rows if r.get("outcome") == outcome_f]
            if group_f and group_f != "All":
                candidate_rows = [r for r in candidate_rows if r.get("group") == group_f]
            if status_f and status_f != "All":
                candidate_rows = [r for r in candidate_rows if r.get("review_status") == status_f]

        for idx, row in enumerate(candidate_rows):
            formatted_row = _build_row_dict_35(row, res, context, row_idx=idx)
            formatted_rows.append(formatted_row)

        # Assemble image metadata entry for sidecar
        src_file = res.get("image_file") or (res.get("_provenance") or {}).get("source_document") or ""
        meta_entry = {
            "source_file": str(src_file),
            "chart_type": str(res.get("chart_type") or ""),
            "calibration": res.get("calibration") or {},
            "calibration_quality": (res.get("metadata") or {}).get("calibration_quality"),
            "warnings": res.get("warnings") or (res.get("metadata") or {}).get("warnings") or [],
            "run_settings": (res.get("metadata") or {}).get("run_settings"),
        }
        image_metadata_list.append(meta_entry)

    # Write CSV with RFC 4180 quoting and \n endings
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_V2_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in formatted_rows:
            writer.writerow(r)

    # Post-write verification: strict check on header and row count
    with open(out_path, "r", encoding="utf-8") as vf:
        reader = csv.reader(vf)
        try:
            written_header = next(reader)
        except StopIteration:
            written_header = []
        written_rows = list(reader)

    if written_header != EXPORT_V2_COLUMNS:
        raise ValueError(
            f"Post-write verification failed: header mismatch.\n"
            f"Expected: {EXPORT_V2_COLUMNS}\n"
            f"Got:      {written_header}"
        )

    if len(written_rows) != len(formatted_rows):
        raise ValueError(
            f"Post-write verification failed: row count mismatch. "
            f"Expected {len(formatted_rows)} rows, but read back {len(written_rows)} rows from {out_path}."
        )

    # Write <stem>.meta.json sidecar if requested (pretty-printed per DP-01)
    if include_metadata:
        meta_path = out_path.with_name(f"{out_path.stem}.meta.json")
        meta_content = {
            "schema_version": "2.0",
            "plotin_version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "scope": scope,
            "filters": filters or {},
            "row_count": len(formatted_rows),
            "column_count": len(EXPORT_V2_COLUMNS),
            "columns": EXPORT_V2_COLUMNS,
            "images": image_metadata_list,
            "app": {
                "plotin_version": "0.1.0",
            },
        }
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta_content, mf, indent=2, ensure_ascii=False)

    return out_path
