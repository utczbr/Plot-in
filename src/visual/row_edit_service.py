# -*- coding: utf-8 -*-
"""
Pure Write-Through RowEditService (ADR-015 T-015.2, ADR-014 T-014.2).

Synchronizes edits across `protocol_rows`, `elements`, and `bars` while
enforcing domain rules:
- Box plot soft monotonicity warning (`_monotonicity_warning`, DP-09)
- Outlier string coercion (`List[float]`)
- Single-row Pie `percent` <-> `value` sync without rescaling siblings
- Scatter/Line/Area `value_x` -> `group` mirror
- Exact byte-for-byte state restoration on `revert(record)`
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

from core.numeric_format import parse_user_float
from core.protocol_row_builder import ensure_row_uids, get_protocol_columns
from core.review_state import recompute_row_status

_UNSET = object()

BOX_MONOTONIC_FIELDS = ("whisker_low", "q1", "median", "q3", "whisker_high")


@dataclass
class EditRecord:
    """Immutable snapshot of a single logical cell edit for exact undo/redo."""

    result: Dict[str, Any]
    row_uid: str
    field: str
    old_value: Any
    new_value: Any
    element_index: Optional[int] = None
    had_original: bool = False
    previous_original: Optional[Dict[str, Any]] = None
    previous_original_status: Any = _UNSET
    previous_row_snapshot: Dict[str, Any] = field(default_factory=dict)
    previous_element_snapshot: Optional[Dict[str, Any]] = None
    previous_bar_snapshot: Optional[Dict[str, Any]] = None
    previous_result_review_status: Any = _UNSET
    noop: bool = False

    @property
    def previous_row_Snapshot(self) -> Dict[str, Any]:
        return self.previous_row_snapshot

    @previous_row_Snapshot.setter
    def previous_row_Snapshot(self, val: Dict[str, Any]) -> None:
        self.previous_row_snapshot = val


def evaluate_box_monotonicity(row_or_el: Dict[str, Any]) -> bool:
    """Return True if all present box plot summary statistics are monotonic:
    whisker_low <= q1 <= median <= q3 <= whisker_high.
    """
    vals: List[float] = []
    for k in BOX_MONOTONIC_FIELDS:
        v = row_or_el.get(k)
        if v is None or v == "":
            continue
        try:
            fval = float(v)
            if math.isfinite(fval):
                vals.append(fval)
        except (TypeError, ValueError):
            continue
    return all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def parse_outliers_value(raw: Any) -> Tuple[Optional[List[float]], Optional[str]]:
    """Parse an outlier list or comma/semicolon-separated string into List[float]."""
    if raw is None or raw == "":
        return [], None
    if isinstance(raw, list):
        out: List[float] = []
        for item in raw:
            res = parse_user_float(str(item), allow_empty=False)
            if res.error or res.value is None or not math.isfinite(res.value):
                return None, res.error or f"Invalid outlier token: {item!r}"
            out.append(float(res.value))
        return out, None

    text = str(raw).strip()
    if not text:
        return [], None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        return [], None

    tokens = [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]
    out = []
    for tok in tokens:
        res = parse_user_float(tok, allow_empty=False)
        if res.error or res.value is None or not math.isfinite(res.value):
            return None, res.error or f"Invalid outlier token: {tok!r}"
        out.append(float(res.value))
    return out, None


def _recompute_status(row: Dict[str, Any], chart_type: str = "") -> str:
    """Delegate to core.review_state.recompute_row_status."""
    return recompute_row_status(row)


class RowEditService:
    """Stateless write-through service for protocol row and element edits."""

    @staticmethod
    def find_row(result: Optional[Dict[str, Any]], row_uid: str) -> Optional[Dict[str, Any]]:
        if not isinstance(result, dict):
            return None
        rows = result.get("protocol_rows", [])
        if not isinstance(rows, list):
            return None
        ensure_row_uids(rows)
        for r in rows:
            if isinstance(r, dict) and r.get("_uid") == row_uid:
                return r
        return None

    @classmethod
    def apply(
        cls,
        result: Dict[str, Any],
        row_uid: str,
        field: str,
        value: Any,
    ) -> EditRecord:
        row = cls.find_row(result, row_uid)
        if row is None:
            raise KeyError(f"Protocol row with _uid={row_uid!r} not found")

        chart_type = str(row.get("chart_type") or result.get("chart_type") or "").lower()
        el_idx = row.get("element_index")
        rows = result.get("protocol_rows", [])
        if el_idx is None and isinstance(rows, list):
            try:
                el_idx = rows.index(row)
                row["element_index"] = el_idx
            except ValueError:
                el_idx = None
        elements = result.get("elements") if isinstance(result.get("elements"), list) else None
        bars = result.get("bars") if isinstance(result.get("bars"), list) else None

        target_el: Optional[Dict[str, Any]] = None
        target_bar: Optional[Dict[str, Any]] = None
        if isinstance(el_idx, int) and el_idx >= 0:
            if elements is not None and el_idx < len(elements) and isinstance(elements[el_idx], dict):
                target_el = elements[el_idx]
            if (
                bars is not None
                and el_idx < len(bars)
                and isinstance(bars[el_idx], dict)
                and bars[el_idx] is not target_el
            ):
                target_bar = bars[el_idx]

        # Coerce outliers if editing outliers field
        coerced_value = value
        if field in ("outliers", "outliers_str"):
            parsed_outliers, err = parse_outliers_value(value)
            if err is not None or parsed_outliers is None:
                raise ValueError(err or f"Invalid outliers value: {value!r}")
            coerced_value = parsed_outliers

        old_val = row.get(field)

        # Capture full snapshots for exact byte-for-byte revert
        had_orig = isinstance(row.get("_original"), dict)
        prev_orig = copy.deepcopy(row.get("_original")) if had_orig else None
        prev_orig_status = row.get("_original_status", _UNSET)
        prev_row_snap = copy.deepcopy(row)
        prev_el_snap = copy.deepcopy(target_el) if target_el is not None else None
        prev_bar_snap = copy.deepcopy(target_bar) if target_bar is not None else None
        prev_res_status = result.get("review_status", _UNSET)

        record = EditRecord(
            result=result,
            row_uid=row_uid,
            field=field,
            old_value=copy.deepcopy(old_val),
            new_value=copy.deepcopy(coerced_value),
            element_index=el_idx if isinstance(el_idx, int) else None,
            had_original=had_orig,
            previous_original=prev_orig,
            previous_original_status=prev_orig_status,
            previous_row_snapshot=prev_row_snap,
            previous_element_snapshot=prev_el_snap,
            previous_bar_snapshot=prev_bar_snap,
            previous_result_review_status=prev_res_status,
        )

        # Snapshot _original on first non-status edit
        if field != "review_status" and not isinstance(row.get("_original"), dict):
            active_cols = get_protocol_columns(chart_type)
            snap_keys = {c[0] for c in active_cols} | set(row.keys())
            if field in ("outliers", "outliers_str"):
                snap_keys |= {"outliers", "outliers_str"}
            snap_keys -= {"review_status", "_uid", "_original", "_original_status", "_monotonicity_warning"}
            row["_original"] = {k: copy.deepcopy(row.get(k)) for k in snap_keys}
            if "_original_status" not in row:
                row["_original_status"] = row.get("review_status", "auto")

        # 1. Update protocol row
        row[field] = coerced_value
        if field in ("outliers", "outliers_str"):
            row["outliers"] = coerced_value
            row["outliers_str"] = ", ".join(str(v) for v in coerced_value)

        # Mirror value_x -> group for scatter / line / area
        if field == "value_x":
            row["group"] = "" if coerced_value is None else str(coerced_value)

        # Single-row Pie sync (never rescale sibling rows)
        if chart_type == "pie":
            if field == "percent":
                if isinstance(coerced_value, (int, float)) and math.isfinite(float(coerced_value)):
                    row["value"] = float(coerced_value) / 100.0
                elif coerced_value is None:
                    row["value"] = None
            elif field == "value":
                if isinstance(coerced_value, (int, float)) and math.isfinite(float(coerced_value)):
                    row["percent"] = float(coerced_value) * 100.0
                elif coerced_value is None:
                    row["percent"] = None

        # 2. Write through to elements[el_idx] and bars[el_idx]
        targets = [t for t in (target_el, target_bar) if t is not None]
        for tgt in targets:
            cls._write_field_to_element(tgt, row, chart_type, field, coerced_value)

        # 3. Box plot soft monotonicity warning (DP-09)
        if chart_type == "box" or field in BOX_MONOTONIC_FIELDS:
            is_mono = evaluate_box_monotonicity(row)
            if not is_mono:
                row["_monotonicity_warning"] = True
                for tgt in targets:
                    tgt["_monotonicity_warning"] = True
            else:
                row.pop("_monotonicity_warning", None)
                for tgt in targets:
                    tgt.pop("_monotonicity_warning", None)

        # 4. Row status & image-level review_status unlock (T-014.2)
        if field != "review_status":
            new_status = _recompute_status(row, chart_type)
            if new_status == "corrected" and result.get("review_status") == "reviewed":
                result["review_status"] = None
        else:
            # Explicit status edit
            if coerced_value == "corrected" and result.get("review_status") == "reviewed":
                result["review_status"] = None

        return record

    @classmethod
    def _write_field_to_element(
        cls,
        el: Dict[str, Any],
        row: Dict[str, Any],
        chart_type: str,
        field: str,
        value: Any,
    ) -> None:
        if field == "group":
            if chart_type == "heatmap":
                # Heatmap rows have no group; their protocol "group" is always empty. Writing that
                # back blanked every cell's col_label on Save (the Results table edits Row/Col
                # labels directly), so an empty value must not overwrite the label.
                if str(value or "").strip():
                    el["col_label"] = value
            elif chart_type == "pie":
                el["label"] = value
                if "text_label" in el:
                    el["text_label"] = value
            else:
                if isinstance(el.get("tick_label"), dict):
                    el["tick_label"]["text"] = value
                elif "bar_label" in el or chart_type in ("bar", "histogram"):
                    el["bar_label"] = value
                    if "label" in el:
                        el["label"] = value
                else:
                    el["label"] = value
        elif field == "outcome":
            if chart_type == "heatmap":
                if str(value or "").strip():   # same reason as "group" above
                    el["row_label"] = value
            else:
                el["series_label"] = value
                if "legend_label" in el:
                    el["legend_label"] = value
        elif field == "series_id":
            if chart_type == "heatmap":
                el["row_label"] = value
            else:
                el["series_id"] = value
        elif field == "value":
            el["value_is_pixels"] = False  # an edited value is a real data value, not a pixel length
            if chart_type == "scatter":
                el["y"] = value
                if "value" in el:
                    el["value"] = value
            elif chart_type == "pie":
                el["value"] = value
                el["percent"] = row.get("percent")
                if "percentage" in el:
                    el["percentage"] = row.get("percent")
            elif chart_type == "box":
                el["median"] = value
                if "value" in el:
                    el["value"] = value
            else:
                el["value"] = value
                if "estimated_value" in el:
                    el["estimated_value"] = value
                if chart_type in ("line", "area") and "y" in el:
                    el["y"] = value
        elif field == "percent":
            el["percent"] = value
            if "percentage" in el:
                el["percentage"] = value
            if chart_type == "pie":
                el["value"] = row.get("value")
        elif field == "value_x":
            el["x"] = value
            el["value_x"] = value
        elif field == "value_y":
            el["y"] = value
            el["value"] = value
        elif field == "error_bar_value":
            eb = el.get("error_bar")
            if isinstance(eb, dict):
                eb["value"] = value
                eb["margin"] = value
            else:
                el["error_bar"] = {"value": value, "margin": value}
            el["error_bar_value"] = value
        elif field == "error_bar_type":
            el["error_bar_type"] = value
            if isinstance(el.get("error_bar"), dict):
                el["error_bar"]["type"] = value
        elif field in BOX_MONOTONIC_FIELDS:
            el[field] = value
            if field == "median":
                row["value"] = value
                if "value" in el:
                    el["value"] = value
        elif field in ("outliers", "outliers_str"):
            el["outliers"] = list(value) if isinstance(value, list) else []
        elif field in ("row_label", "col_label", "unit", "notes"):
            el[field] = value
            if field == "row_label" and chart_type == "heatmap":
                row["outcome"] = value
            elif field == "col_label" and chart_type == "heatmap":
                row["group"] = value
        else:
            el[field] = value

    @classmethod
    def revert(cls, record: EditRecord) -> None:
        if record.noop:
            return
        result = record.result
        row = cls.find_row(result, record.row_uid)
        if row is not None and record.previous_row_Snapshot:
            row.clear()
            row.update(copy.deepcopy(record.previous_row_Snapshot))

        el_idx = record.element_index
        if isinstance(el_idx, int) and el_idx >= 0:
            elements = result.get("elements")
            if (
                isinstance(elements, list)
                and el_idx < len(elements)
                and isinstance(elements[el_idx], dict)
                and record.previous_element_snapshot is not None
            ):
                elements[el_idx].clear()
                elements[el_idx].update(copy.deepcopy(record.previous_element_snapshot))

            bars = result.get("bars")
            if (
                isinstance(bars, list)
                and el_idx < len(bars)
                and isinstance(bars[el_idx], dict)
                and record.previous_bar_snapshot is not None
                and (not isinstance(elements, list) or bars[el_idx] is not elements[el_idx])
            ):
                bars[el_idx].clear()
                bars[el_idx].update(copy.deepcopy(record.previous_bar_snapshot))

        if record.previous_result_review_status is _UNSET:
            result.pop("review_status", None)
        else:
            result["review_status"] = record.previous_result_review_status