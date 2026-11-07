"""
OCR Tab record categorizer and helper module.

Axis labels of Cartesian charts are split by *role*, not by text shape:

* ``tick_label``    - the independent axis (categories, or the numeric X axis of line /
                      scatter charts; the Y axis of horizontal charts).
* ``scale_label``   - the first dependent (value) axis.
* ``scale_label_2`` - the second dependent axis of a dual-axis (3-axis) chart.

The tick-vs-scale decision is made by the extraction pipeline (``label_type`` tags,
``label_classification`` lists, per-label ``axis`` tags). This module only renders that
decision; a text heuristic is used solely for labels the pipeline never classified
(legacy caches, manual detections).
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from core.label_overrides import refresh_cleaned_value  # noqa: F401  (re-exported for callers/tests)
from utils.validation_utils import is_numeric

Record = Tuple[Dict[str, Any], str]
SectionRecords = Dict[str, List[Record]]

# Cartesian chart types: their "axis labels" are split into tick / scale labels.
CARTESIAN_CHART_TYPES = {"bar", "histogram", "line", "scatter", "box", "area"}

# Highlight palette entries (see MainWindow.colors) for each split section.
_SECTION_SOURCE_CLASS = {
    "tick_label": "tick_label",
    "scale_label": "scale_label",
    "scale_label_2": "scale_label",
}


def _label_key(item: Dict[str, Any]) -> Any:
    """Identity key for a label: bbox when available (stable under text edits), else text."""
    bbox = item.get("xyxy") or item.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return ("bbox", tuple(round(float(v), 1) for v in bbox[:4]))
    return ("text", str(item.get("text", "")).strip())


_CURRENCY = "$€£¥"


def _looks_numeric(text: Any) -> bool:
    """Strict numeric test that also accepts a currency prefix and a trailing percent sign."""
    s = str(text or "").strip().lstrip(_CURRENCY).rstrip("%").strip()
    return is_numeric(s)


def _keys_of(items: Any) -> Set[Any]:
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return set()
    return {_label_key(i) for i in items if isinstance(i, dict)}


def _independent_axis(result: Dict[str, Any]) -> str:
    """'x' for vertical charts (and scatter/line/area); 'y' for horizontal bar-like charts."""
    chart_type = str(result.get("chart_type", "")).lower()
    if chart_type in {"scatter", "line", "area"}:
        return "x"
    return "y" if _result_orientation(result).startswith("h") else "x"


def _result_orientation(result: Dict[str, Any]) -> str:
    """'horizontal' / 'vertical'. Falls back from the result-level key (missing in older saved
    results) to the extractor's ``orientation_info`` and then to the elements' own tags,
    instead of silently assuming vertical."""
    value = result.get("orientation")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    metadata = result.get("metadata")
    info = metadata.get("orientation_info") if isinstance(metadata, dict) else None
    if isinstance(info, dict) and isinstance(info.get("orientation"), str) and info["orientation"].strip():
        return info["orientation"].strip().lower()
    tags = []
    for key in ("bars", "elements"):
        for el in result.get(key) or []:
            tag = el.get("orientation") if isinstance(el, dict) else None
            if isinstance(tag, str) and tag.strip():
                tags.append(tag.strip().lower())
    if tags and sum(t.startswith("h") for t in tags) * 2 > len(tags):
        return "horizontal"
    return "vertical"


def _split_axis_labels(
    axis_records: List[Record],
    result: Dict[str, Any],
) -> Tuple[SectionRecords, Dict[str, Any]]:
    """Split merged axis-label records into tick / first scale / second scale groups.

    Decision order for each label:
      1. A label on the *independent* axis (per-label ``axis`` tag, or the classifier's
         x/y scale pools) is a tick label even when its text is numeric (years, doses, a
         numeric X axis). This keeps a dual-axis chart with numeric X from merging X ticks
         into the 1st value axis.
      2. Explicit ``label_type`` tag from the pipeline.
      3. Membership in the ``label_classification`` tick / scale lists.
      4. Last resort, only for labels the pipeline never classified: strictly numeric text
         is a scale label, anything else a tick label.
    A scale label listed by the dual-axis decision as secondary goes to ``scale_label_2``.

    Returns ``(split, info)``; ``info`` has the ``layout`` ("two_axis" or "three_axis")
    and the section ``titles`` to show.
    """
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    label_cls = metadata.get("label_classification")
    label_cls = label_cls if isinstance(label_cls, dict) else {}
    tick_keys: Set[Any] = set()
    scale_keys: Set[Any] = set()
    for k in ("tick_labels", "tick_label"):
        tick_keys |= _keys_of(label_cls.get(k))
    for k in ("scale_labels", "scale_label", "x_scale_labels", "y_scale_labels"):
        scale_keys |= _keys_of(label_cls.get(k))
    axis_pool = {
        "x": _keys_of(label_cls.get("x_scale_labels")),
        "y": _keys_of(label_cls.get("y_scale_labels")),
    }

    dual = metadata.get("dual_axis")
    dual = dual if isinstance(dual, dict) else {}
    secondary_keys = _keys_of(dual.get("secondary_labels")) if dual.get("has_dual_axis") else set()
    if dual.get("has_dual_axis"):
        # Labels the dual-axis detector assigned to a value axis are scale labels by definition.
        scale_keys |= secondary_keys | _keys_of(dual.get("primary_labels"))

    independent = _independent_axis(result)
    split: SectionRecords = {"tick_label": [], "scale_label": [], "scale_label_2": []}
    numeric_independent = False
    seen: Set[Any] = set()
    for item, _source in axis_records:
        key = _label_key(item)
        # The same label can arrive from detections and from label_classification.
        if key in seen:
            continue
        seen.add(key)

        axis = item.get("axis")
        if axis not in ("x", "y"):
            axis = next((a for a, pool in axis_pool.items() if key in pool), None)

        label_type = item.get("label_type")
        if axis == independent:
            section = "tick_label"
            # Remember whether this axis is numeric so the title can say "(X Axis)".
            if label_type == "scale_label" or key in scale_keys or _looks_numeric(item.get("text")):
                numeric_independent = True
        elif label_type in ("tick_label", "scale_label"):
            section = label_type
        elif key in scale_keys:
            section = "scale_label"
        elif key in tick_keys:
            section = "tick_label"
        else:
            section = "scale_label" if _looks_numeric(item.get("text")) else "tick_label"

        if section == "scale_label" and key in secondary_keys:
            section = "scale_label_2"
        split[section].append((item, _SECTION_SOURCE_CLASS[section]))

    three_axis = bool(split["scale_label_2"])
    titles = {
        "tick_label": f"Tick Labels ({independent.upper()} Axis)" if numeric_independent else "Tick Labels",
        "scale_label": "Scale Labels (1st Axis)" if three_axis else "Scale Labels",
        "scale_label_2": "Scale Labels (2nd Axis)",
    }
    return split, {"layout": "three_axis" if three_axis else "two_axis", "titles": titles}


def _split_heatmap_labels(axis_records: List[Record], elements: Any) -> SectionRecords:
    """Split heatmap axis labels into row labels (beside the grid) and column labels
    (above/below it).

    Uses grid geometry first: a label is a row label when it lies further outside the grid
    horizontally than vertically, a column label in the opposite case. Comparing distances
    (rather than requiring the centre to be strictly inside the grid's span) keeps rotated
    or long column labels, whose centres drift past the grid edge, in the right group.
    Falls back to the cells' row/column label text; labels that cannot be placed stay in
    ``axis_labels``.
    """
    cells = [e for e in (elements if isinstance(elements, list) else []) if isinstance(e, dict)]
    boxes = []
    row_texts: Set[str] = set()
    col_texts: Set[str] = set()
    for cell in cells:
        bbox = cell.get("xyxy") or cell.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            boxes.append([float(v) for v in bbox[:4]])
        if str(cell.get("row_label", "")).strip():
            row_texts.add(str(cell["row_label"]).strip())
        if str(cell.get("col_label", "")).strip():
            col_texts.add(str(cell["col_label"]).strip())
    grid = None
    if boxes:
        grid = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    out: SectionRecords = {"row_label": [], "col_label": [], "axis_labels": []}
    for item, source in axis_records:
        kind: Optional[str] = None
        bbox = item.get("xyxy") or item.get("bbox")
        if grid and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            dx = max(grid[0] - cx, 0.0, cx - grid[2])
            dy = max(grid[1] - cy, 0.0, cy - grid[3])
            if dx > dy:
                kind = "row_label"
            elif dy > dx:
                kind = "col_label"
        if kind is None:
            text = str(item.get("text", "")).strip()
            in_row, in_col = text in row_texts, text in col_texts
            if in_row != in_col:
                kind = "row_label" if in_row else "col_label"
        out[kind or "axis_labels"].append((item, source))
    return out


def categorize_ocr_records(result: Dict[str, Any]) -> SectionRecords:
    """Categorize OCR detections into section records.

    Returns a dict mapping every section key to a list of (detection_item, source_class)
    tuples. Use :func:`categorize_ocr_sections` when the section titles are needed too.
    """
    return categorize_ocr_sections(result)[0]


def categorize_ocr_sections(result: Dict[str, Any]) -> Tuple[SectionRecords, Dict[str, Any]]:
    """Categorize OCR detections into section records plus layout information.

    Returns ``(section_records, layout_info)``. ``section_records`` maps section keys to
    lists of (detection_item, source_class). ``layout_info["layout"]`` is "two_axis",
    "three_axis" (Cartesian with a second value axis), "heatmap_axes" or "merged", and
    ``layout_info["titles"]`` maps section keys to the title to display for this chart.
    """
    detections = result.get('detections', {})
    metadata = result.get('metadata', {})
    chart_type = str(result.get('chart_type', '')).lower()

    assigned_bar_labels = result.get('_assigned_bar_labels', {})
    bar_label_texts = set(assigned_bar_labels.get('texts', []))
    bar_label_bboxes = set(tuple(bbox) for bbox in assigned_bar_labels.get('bboxes', []))

    section_records: SectionRecords = {
        "chart_title": [],
        "axis_title": [],
        "axis_labels": [],
        "tick_label": [],
        "scale_label": [],
        "scale_label_2": [],
        "row_label": [],
        "col_label": [],
        "legend": [],
        "data_label": [],
        "color_bar_label": [],
        "color_bar_title": [],
        "other": [],
        "layout_text": [],
    }

    def _extend_section(section_key: str, source_class: str, items: Any):
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, dict):
                section_records[section_key].append((item, source_class))

    # Direct detector classes
    direct_mapping = {
        "chart_title": "chart_title",
        "axis_title": "axis_title",
        "axis_labels": "axis_labels",
        "legend": "legend",
        "data_label": "data_label",
        "color_bar_label": "color_bar_label",
        "color_bar_title": "color_bar_title",
        "other": "other",
    }
    for detection_key, section_key in direct_mapping.items():
        _extend_section(section_key, detection_key, detections.get(detection_key, []))

    # Also route any legacy scale_label/tick_label into axis_labels
    for legacy_key in ("scale_label", "tick_label", "scale_labels", "tick_labels"):
        if legacy_key in detections:
            _extend_section("axis_labels", "axis_labels", detections.get(legacy_key, []))

    # Collect from label_classification metadata if available
    label_classification = metadata.get("label_classification", {})
    if isinstance(label_classification, dict):
        for k in ("scale_labels", "scale_label", "x_scale_labels", "y_scale_labels",
                  "tick_labels", "tick_label", "axis_labels"):
            _extend_section("axis_labels", "axis_labels", label_classification.get(k, []))
        _extend_section(
            "axis_title",
            "axis_title",
            label_classification.get("axis_titles", label_classification.get("axis_title", [])),
        )

    # Fallback: use extracted element labels for bar-like and box charts
    if not section_records["axis_labels"] and chart_type in {"bar", "histogram", "box"}:
        element_key = "bars" if chart_type != "box" else "elements"
        for element in result.get(element_key, []):
            if isinstance(element, dict):
                label_text = element.get("label") or element.get("series") or element.get("x_label")
                if label_text:
                    section_records["axis_labels"].append(({"text": str(label_text), "confidence": 1.0, "label_type": "tick_label"}, "extracted_element"))

    # Layout text from doclayout
    layout_items = metadata.get("layout_elements", [])
    _extend_section("layout_text", "doclayout", layout_items)

    # Cartesian charts: split the merged axis labels by role (see module docstring).
    # Heatmaps: row / column labels. Everything else keeps the merged "axis_labels" section.
    layout_info: Dict[str, Any] = {"layout": "merged", "titles": {}}
    if chart_type in CARTESIAN_CHART_TYPES:
        split, layout_info = _split_axis_labels(section_records["axis_labels"], result)
        section_records.update(split)
        section_records["axis_labels"] = []
    elif chart_type == "heatmap":
        split = _split_heatmap_labels(section_records["axis_labels"], result.get("elements"))
        if split["row_label"] or split["col_label"]:
            section_records.update(split)
            layout_info = {
                "layout": "heatmap_axes",
                "titles": {
                    "row_label": "Row Labels",
                    "col_label": "Column Labels",
                    "axis_labels": "Other Axis Labels",
                },
            }

    return section_records, layout_info
