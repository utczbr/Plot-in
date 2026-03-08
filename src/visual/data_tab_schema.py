"""Schema-driven data-tab helpers for chart-type-aware GUI rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DataTabColumnSpec:
    """Column definition used by the Data tab table renderer."""
    id: str
    title: str
    editable: bool = False
    value_type: str = "text"


@dataclass(frozen=True)
class DataTabRow:
    """A single logical row in the Data tab model."""
    element_index: int
    source: str
    values: Dict[str, Any]
    editable_fields: List[str]
    overlay_bbox: Optional[List[float]]
    overlay_class: Optional[str]
    kind: str = "element"


@dataclass(frozen=True)
class DataTabSchema:
    """Chart-specific schema consumed by the Data tab renderer."""
    schema_id: str
    columns: List[DataTabColumnSpec]
    rows: List[DataTabRow]
    summary: Dict[str, Any]
    pagination: Dict[str, Any]
    empty_message: str


@dataclass(frozen=True)
class DataTabEditBinding:
    """Binding between a table cell and an element field."""
    element_index: int
    source: str
    field: str
    parser: str = "str"


_CARTESIAN_TYPES = {"bar", "histogram", "line", "scatter", "box", "area"}


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to finite float when possible."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _value_or_estimated(element: Dict[str, Any]) -> Optional[float]:
    for key in ("value", "estimated_value"):
        parsed = _safe_float(element.get(key))
        if parsed is not None:
            return parsed
    return None


def _bbox_from_element(element: Dict[str, Any]) -> Optional[List[float]]:
    """Resolve overlay bbox from common keys."""
    bbox = element.get("xyxy")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return bbox[:4]
    bbox = element.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return bbox[:4]
    return None


def _coerce_outliers(value: Any) -> List[float]:
    """Parse outlier values from list/tuple/string."""
    if isinstance(value, list):
        parsed = [_safe_float(item) for item in value]
        return [item for item in parsed if item is not None]
    if isinstance(value, tuple):
        parsed = [_safe_float(item) for item in value]
        return [item for item in parsed if item is not None]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        parsed = [_safe_float(part) for part in parts if part]
        return [item for item in parsed if item is not None]
    parsed = _safe_float(value)
    return [parsed] if parsed is not None else []


def normalize_pie_values(elements: List[Dict[str, Any]]) -> None:
    """Normalize pie slice values to sum to 1.0 when possible."""
    pie_indices: List[int] = []
    values: List[float] = []
    for idx, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        if element.get("type") != "pie_slice":
            continue
        value = _safe_float(element.get("value"))
        if value is None:
            continue
        pie_indices.append(idx)
        values.append(value)

    if not values:
        return

    total = sum(values)
    if total <= 0:
        return

    for idx, original in zip(pie_indices, values):
        elements[idx]["value"] = float(original / total)


def autocorrect_box_statistics(element: Dict[str, Any]) -> None:
    """Ensure box statistics remain in valid monotonic order."""
    keys = ("whisker_low", "q1", "median", "q3", "whisker_high")
    values = [_safe_float(element.get(key)) for key in keys]
    if any(value is None for value in values):
        return
    sorted_values = sorted(values)  # type: ignore[arg-type]
    for key, value in zip(keys, sorted_values):
        element[key] = float(value)


def _columns_for_chart(chart_type: str) -> List[DataTabColumnSpec]:
    if chart_type in {"bar", "histogram"}:
        return [
            DataTabColumnSpec("label", "Label", editable=True),
            DataTabColumnSpec("value", "Value", editable=True, value_type="float"),
            DataTabColumnSpec("confidence", "Conf.", value_type="float"),
            DataTabColumnSpec("pixel_size", "Pixel Size", editable=True, value_type="float"),
            DataTabColumnSpec("error_bar", "Error Bar", editable=True, value_type="float"),
        ]
    if chart_type == "pie":
        return [
            DataTabColumnSpec("label", "Label", editable=True),
            DataTabColumnSpec("value", "Value", editable=True, value_type="float"),
            DataTabColumnSpec("percent", "%", editable=True, value_type="float"),
            DataTabColumnSpec("angle", "Angle", editable=True, value_type="float"),
            DataTabColumnSpec("confidence", "Conf.", value_type="float"),
        ]
    if chart_type == "line":
        return [
            DataTabColumnSpec("index", "#", value_type="int"),
            DataTabColumnSpec("value", "Value", editable=True, value_type="float"),
            DataTabColumnSpec("position", "Position", editable=True, value_type="float"),
            DataTabColumnSpec("confidence", "Conf.", value_type="float"),
        ]
    if chart_type == "scatter":
        return [
            DataTabColumnSpec("index", "#", value_type="int"),
            DataTabColumnSpec("x", "X", editable=True, value_type="float"),
            DataTabColumnSpec("y", "Y", editable=True, value_type="float"),
            DataTabColumnSpec("center_x", "Center X", value_type="float"),
            DataTabColumnSpec("center_y", "Center Y", value_type="float"),
            DataTabColumnSpec("confidence", "Conf.", value_type="float"),
        ]
    if chart_type == "box":
        return [
            DataTabColumnSpec("label", "Label", editable=True),
            DataTabColumnSpec("whisker_low", "Min", editable=True, value_type="float"),
            DataTabColumnSpec("q1", "Q1", editable=True, value_type="float"),
            DataTabColumnSpec("median", "Median", editable=True, value_type="float"),
            DataTabColumnSpec("q3", "Q3", editable=True, value_type="float"),
            DataTabColumnSpec("whisker_high", "Max", editable=True, value_type="float"),
            DataTabColumnSpec("outliers", "Outliers", editable=True),
        ]
    if chart_type == "heatmap":
        return [
            DataTabColumnSpec("row", "Row", value_type="int"),
            DataTabColumnSpec("col", "Col", value_type="int"),
            DataTabColumnSpec("row_label", "Row Label", editable=True),
            DataTabColumnSpec("col_label", "Col Label", editable=True),
            DataTabColumnSpec("value", "Value", editable=True, value_type="float"),
            DataTabColumnSpec("confidence", "Conf.", value_type="float"),
        ]
    if chart_type == "area":
        return [
            DataTabColumnSpec("kind", "Kind"),
            DataTabColumnSpec("index", "#", value_type="int"),
            DataTabColumnSpec("position", "Position", editable=True, value_type="float"),
            DataTabColumnSpec("value", "Value", editable=True, value_type="float"),
            DataTabColumnSpec("confidence", "Conf.", value_type="float"),
            DataTabColumnSpec("auc", "AUC", value_type="float"),
            DataTabColumnSpec("num_points", "Points", value_type="int"),
        ]
    return [
        DataTabColumnSpec("index", "#", value_type="int"),
        DataTabColumnSpec("value", "Value", editable=True, value_type="float"),
        DataTabColumnSpec("confidence", "Conf.", value_type="float"),
    ]


def _format_outliers(element: Dict[str, Any]) -> str:
    outliers = element.get("outliers")
    parsed = _coerce_outliers(outliers)
    if not parsed:
        return ""
    return ", ".join(f"{value:.4g}" for value in parsed)


def _editable_fields_for_chart(chart_type: str, row_kind: str = "element") -> List[str]:
    if row_kind != "element":
        return []
    mapping = {
        "bar": ["label", "value", "pixel_size", "error_bar"],
        "histogram": ["label", "value", "pixel_size", "error_bar"],
        "pie": ["label", "value", "percent", "angle"],
        "line": ["value", "position"],
        "scatter": ["x", "y"],
        "box": ["label", "whisker_low", "q1", "median", "q3", "whisker_high", "outliers"],
        "heatmap": ["row_label", "col_label", "value"],
        "area": ["position", "value"],
    }
    return mapping.get(chart_type, ["value"])


def _build_rows(result: Dict[str, Any], chart_type: str) -> List[DataTabRow]:
    rows: List[DataTabRow] = []

    if chart_type in {"bar", "histogram"}:
        source = "bars"
        bars = result.get("bars")
        if not isinstance(bars, list):
            bars = []
        for idx, bar in enumerate(bars):
            if not isinstance(bar, dict):
                continue
            tick_label = bar.get("tick_label")
            tick_text = ""
            if isinstance(tick_label, dict):
                tick_text = str(tick_label.get("text", "")).strip()
            label = str(bar.get("bar_label", "")).strip() or tick_text or str(bar.get("text_label", "")).strip()
            error_bar = bar.get("error_bar")
            error_margin = None
            if isinstance(error_bar, dict):
                error_margin = _safe_float(error_bar.get("margin"))
            row_values = {
                "label": label,
                "value": _value_or_estimated(bar),
                "confidence": _safe_float(bar.get("confidence", bar.get("conf"))),
                "pixel_size": _safe_float(bar.get("pixel_height", bar.get("pixel_dimension"))),
                "error_bar": error_margin,
            }
            rows.append(
                DataTabRow(
                    element_index=idx,
                    source=source,
                    values=row_values,
                    editable_fields=_editable_fields_for_chart(chart_type),
                    overlay_bbox=_bbox_from_element(bar),
                    overlay_class="bar",
                )
            )
        return rows

    elements = result.get("elements")
    if not isinstance(elements, list):
        elements = []

    for idx, element in enumerate(elements):
        if not isinstance(element, dict):
            continue

        row_kind = "summary" if str(element.get("type", "")).endswith("_summary") else "element"
        editable_fields = _editable_fields_for_chart(chart_type, row_kind=row_kind)
        overlay_class = None
        if chart_type == "pie":
            overlay_class = "slice"
        elif chart_type == "line":
            overlay_class = "line"
        elif chart_type == "scatter":
            overlay_class = "data_point"
        elif chart_type == "box":
            overlay_class = "box"
        elif chart_type == "heatmap":
            overlay_class = "cell"
        elif chart_type == "area":
            overlay_class = "data_point"

        if chart_type == "pie":
            value = _safe_float(element.get("value")) or 0.0
            row_values = {
                "label": str(element.get("label", "")).strip(),
                "value": value,
                "percent": value * 100.0,
                "angle": _safe_float(element.get("angle")),
                "confidence": _safe_float(element.get("confidence")),
            }
        elif chart_type == "line":
            row_values = {
                "index": idx,
                "value": _value_or_estimated(element),
                "position": _safe_float(element.get("position")),
                "confidence": _safe_float(element.get("confidence", element.get("conf"))),
            }
        elif chart_type == "scatter":
            center = element.get("center")
            center_x = None
            center_y = None
            if isinstance(center, list) and len(center) >= 2:
                center_x = _safe_float(center[0])
                center_y = _safe_float(center[1])
            row_values = {
                "index": idx,
                "x": _safe_float(element.get("x")),
                "y": _safe_float(element.get("y")),
                "center_x": center_x,
                "center_y": center_y,
                "confidence": _safe_float(element.get("confidence", element.get("conf"))),
            }
        elif chart_type == "box":
            tick_label = element.get("tick_label")
            label_text = ""
            if isinstance(tick_label, dict):
                label_text = str(tick_label.get("text", "")).strip()
            row_values = {
                "label": label_text,
                "whisker_low": _safe_float(element.get("whisker_low")),
                "q1": _safe_float(element.get("q1")),
                "median": _safe_float(element.get("median")),
                "q3": _safe_float(element.get("q3")),
                "whisker_high": _safe_float(element.get("whisker_high")),
                "outliers": _format_outliers(element),
            }
        elif chart_type == "heatmap":
            row_values = {
                "row": element.get("row"),
                "col": element.get("col"),
                "row_label": str(element.get("row_label", "")).strip(),
                "col_label": str(element.get("col_label", "")).strip(),
                "value": _safe_float(element.get("value")),
                "confidence": _safe_float(element.get("confidence")),
            }
        elif chart_type == "area":
            row_values = {
                "kind": "summary" if row_kind == "summary" else "point",
                "index": idx,
                "position": _safe_float(element.get("position")),
                "value": _value_or_estimated(element),
                "confidence": _safe_float(element.get("confidence", element.get("conf"))),
                "auc": _safe_float(element.get("auc")),
                "num_points": element.get("num_points"),
            }
        else:
            row_values = {
                "index": idx,
                "value": _value_or_estimated(element),
                "confidence": _safe_float(element.get("confidence", element.get("conf"))),
            }

        rows.append(
            DataTabRow(
                element_index=idx,
                source="elements",
                values=row_values,
                editable_fields=editable_fields,
                overlay_bbox=_bbox_from_element(element),
                overlay_class=overlay_class,
                kind=row_kind,
            )
        )

    return rows


def _empty_message(chart_type: str) -> str:
    readable = chart_type.replace("_", " ").strip().title() or "Chart"
    return f"No editable {readable} data found."


def build_data_tab_model(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a schema-driven model for the Data tab from a normalized result payload.
    """
    chart_type = str(result.get("chart_type", "bar")).strip().lower() or "bar"
    columns = _columns_for_chart(chart_type)
    rows = _build_rows(result, chart_type)

    page_size = 100
    total_rows = len(rows)
    total_pages = max(1, math.ceil(total_rows / page_size)) if total_rows else 1
    pagination_enabled = chart_type == "heatmap" and total_rows > page_size

    schema = DataTabSchema(
        schema_id=chart_type,
        columns=columns,
        rows=rows,
        summary={
            "chart_type": chart_type,
            "row_count": total_rows,
            "editable_count": sum(1 for row in rows if row.editable_fields),
            "is_cartesian": chart_type in _CARTESIAN_TYPES,
        },
        pagination={
            "enabled": pagination_enabled,
            "page_size": page_size,
            "page": 0,
            "total_rows": total_rows,
            "total_pages": total_pages,
        },
        empty_message=_empty_message(chart_type),
    )

    return {
        "schema_id": schema.schema_id,
        "columns": [asdict(column) for column in schema.columns],
        "rows": [asdict(row) for row in schema.rows],
        "summary": schema.summary,
        "pagination": schema.pagination,
        "empty_message": schema.empty_message,
    }


def _apply_label(target: Dict[str, Any], label: str) -> None:
    target["label"] = label
    if "bar_label" in target or "tick_label" in target:
        target["bar_label"] = label
    tick_label = target.get("tick_label")
    if isinstance(tick_label, dict):
        tick_label["text"] = label


def _apply_bar_field(target: Dict[str, Any], field: str, parsed: Any) -> None:
    if field == "label":
        _apply_label(target, str(parsed).strip())
        return
    if field == "value":
        target["estimated_value"] = _safe_float(parsed)
        return
    if field == "pixel_size":
        numeric = _safe_float(parsed)
        target["pixel_height"] = numeric
        target["pixel_dimension"] = numeric
        return
    if field == "error_bar":
        numeric = _safe_float(parsed)
        if numeric is None:
            target["error_bar"] = None
        else:
            error_bar = target.get("error_bar")
            if not isinstance(error_bar, dict):
                error_bar = {"bbox": None}
            error_bar["margin"] = numeric
            target["error_bar"] = error_bar


def _apply_pie_field(target: Dict[str, Any], field: str, parsed: Any) -> None:
    if field == "label":
        _apply_label(target, str(parsed).strip())
        return
    if field == "value":
        numeric = _safe_float(parsed)
        if numeric is not None:
            target["value"] = numeric
        return
    if field == "percent":
        numeric = _safe_float(parsed)
        if numeric is not None:
            target["value"] = numeric / 100.0
        return
    if field == "angle":
        numeric = _safe_float(parsed)
        if numeric is not None:
            target["angle"] = numeric


def _apply_line_area_field(target: Dict[str, Any], field: str, parsed: Any) -> None:
    if field == "value":
        numeric = _safe_float(parsed)
        if "value" in target:
            target["value"] = numeric
        else:
            target["estimated_value"] = numeric
        return
    if field == "position":
        numeric = _safe_float(parsed)
        if numeric is not None:
            target["position"] = numeric


def _apply_scatter_field(target: Dict[str, Any], field: str, parsed: Any) -> None:
    numeric = _safe_float(parsed)
    if numeric is None:
        return
    if field == "x":
        target["x"] = numeric
    elif field == "y":
        target["y"] = numeric


def _apply_box_field(target: Dict[str, Any], field: str, parsed: Any) -> None:
    if field == "label":
        tick_label = target.get("tick_label")
        if not isinstance(tick_label, dict):
            tick_label = {}
        tick_label["text"] = str(parsed).strip()
        target["tick_label"] = tick_label
        return

    if field == "outliers":
        target["outliers"] = _coerce_outliers(parsed)
        return

    numeric = _safe_float(parsed)
    if numeric is not None:
        target[field] = numeric


def _apply_heatmap_field(target: Dict[str, Any], field: str, parsed: Any) -> None:
    if field in {"row_label", "col_label"}:
        target[field] = str(parsed).strip()
        return
    if field == "value":
        numeric = _safe_float(parsed)
        if numeric is not None:
            target["value"] = numeric


def _apply_edit(chart_type: str, target: Dict[str, Any], field: str, parsed: Any) -> None:
    if chart_type in {"bar", "histogram"}:
        _apply_bar_field(target, field, parsed)
    elif chart_type == "pie":
        _apply_pie_field(target, field, parsed)
    elif chart_type == "line":
        _apply_line_area_field(target, field, parsed)
    elif chart_type == "scatter":
        _apply_scatter_field(target, field, parsed)
    elif chart_type == "box":
        _apply_box_field(target, field, parsed)
    elif chart_type == "heatmap":
        _apply_heatmap_field(target, field, parsed)
    elif chart_type == "area":
        _apply_line_area_field(target, field, parsed)
    elif field == "value":
        numeric = _safe_float(parsed)
        if numeric is not None:
            target["value"] = numeric


def _parse_edit_value(value: Any, parser: str) -> Any:
    if parser == "float":
        return _safe_float(value)
    if parser == "int":
        numeric = _safe_float(value)
        return int(numeric) if numeric is not None else None
    if parser == "outliers":
        return _coerce_outliers(value)
    return str(value) if value is not None else ""


def apply_data_tab_edits(result: Dict[str, Any], edits: List[Dict[str, Any]]) -> None:
    """
    Apply Data tab edits back into the underlying normalized result payload.
    """
    chart_type = str(result.get("chart_type", "bar")).strip().lower()
    if not edits:
        result["data_tab_model"] = build_data_tab_model(result)
        return

    elements = result.get("elements")
    if not isinstance(elements, list):
        elements = []
    bars = result.get("bars")
    if not isinstance(bars, list):
        bars = []

    touched_box_indices: set[int] = set()
    touched_pie = False

    for edit in edits:
        if not isinstance(edit, dict):
            continue
        source = str(edit.get("source", "elements"))
        try:
            element_index = int(edit.get("element_index"))
        except (TypeError, ValueError):
            continue
        field = str(edit.get("field", "")).strip()
        if not field:
            continue
        parser = str(edit.get("parser", "str")).strip() or "str"
        parsed = _parse_edit_value(edit.get("value"), parser)

        collection = bars if source == "bars" else elements
        if element_index < 0 or element_index >= len(collection):
            continue
        target = collection[element_index]
        if not isinstance(target, dict):
            continue

        _apply_edit(chart_type, target, field, parsed)

        # Keep bars/elements in sync for bar/hist charts when both collections exist.
        if chart_type in {"bar", "histogram"} and source == "bars":
            if element_index < len(elements) and isinstance(elements[element_index], dict):
                _apply_edit(chart_type, elements[element_index], field, parsed)

        if chart_type == "box":
            touched_box_indices.add(element_index)
        if chart_type == "pie" and field in {"value", "percent"}:
            touched_pie = True

    if chart_type == "box":
        for index in touched_box_indices:
            if index < len(elements) and isinstance(elements[index], dict):
                autocorrect_box_statistics(elements[index])
    if chart_type == "pie" and touched_pie:
        normalize_pie_values(elements)

    if elements:
        result["elements"] = elements
    if bars:
        result["bars"] = bars
    result["data_tab_model"] = build_data_tab_model(result)
