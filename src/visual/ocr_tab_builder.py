"""
OCR Tab record categorizer and helper module.
"""

from typing import Any, Dict, List, Tuple


def categorize_ocr_records(result: Dict[str, Any]) -> Dict[str, List[Tuple[Dict[str, Any], str]]]:
    """
    Categorize OCR detections into section records based on chart type and metadata.

    Returns a dict mapping section keys to lists of (detection_item, source_class) tuples.
    """
    detections = result.get('detections', {})
    metadata = result.get('metadata', {})
    chart_type = str(result.get('chart_type', '')).lower()

    assigned_bar_labels = result.get('_assigned_bar_labels', {})
    bar_label_texts = set(assigned_bar_labels.get('texts', []))
    bar_label_bboxes = set(tuple(bbox) for bbox in assigned_bar_labels.get('bboxes', []))

    section_records: Dict[str, List[Tuple[Dict[str, Any], str]]] = {
        "chart_title": [],
        "axis_title": [],
        "axis_labels": [],
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
        for k in ("scale_labels", "scale_label", "tick_labels", "tick_label", "axis_labels"):
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
                    section_records["axis_labels"].append(({"text": str(label_text), "confidence": 1.0}, "extracted_element"))

    # Layout text from doclayout
    layout_items = metadata.get("layout_elements", [])
    _extend_section("layout_text", "doclayout", layout_items)

    # Backward compatibility aliases
    section_records["scale_label"] = section_records["axis_labels"]
    section_records["tick_label"] = section_records["axis_labels"]

    return section_records
