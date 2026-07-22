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
        "scale_label": [],
        "tick_label": [],
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
        "legend": "legend",
        "data_label": "data_label",
        "color_bar_label": "color_bar_label",
        "color_bar_title": "color_bar_title",
        "other": "other",
    }
    for detection_key, section_key in direct_mapping.items():
        _extend_section(section_key, detection_key, detections.get(detection_key, []))

    # Prefer classifier outputs for scale/tick labels when available
    label_classification = metadata.get("label_classification", {})
    if isinstance(label_classification, dict):
        _extend_section(
            "scale_label",
            "axis_labels",
            label_classification.get("scale_labels", label_classification.get("scale_label", [])),
        )
        _extend_section(
            "tick_label",
            "axis_labels",
            label_classification.get("tick_labels", label_classification.get("tick_label", [])),
        )
        _extend_section(
            "axis_title",
            "axis_title",
            label_classification.get("axis_titles", label_classification.get("axis_title", [])),
        )

    has_classified_scale_tick = bool(section_records["scale_label"] or section_records["tick_label"])

    # Fallback: split raw axis_labels into numeric (scale) and non-numeric (tick)
    raw_axis_labels = detections.get("axis_labels", [])
    if isinstance(raw_axis_labels, list) and not has_classified_scale_tick:
        for item in raw_axis_labels:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            cleaned_value = item.get("cleaned_value")

            looks_numeric = cleaned_value is not None
            if not looks_numeric and text:
                candidate = (
                    text.replace(",", ".")
                    .replace("%", "")
                    .replace("$", "")
                    .replace("€", "")
                    .strip()
                )
                try:
                    float(candidate)
                    looks_numeric = True
                except ValueError:
                    looks_numeric = False

            target_section = "scale_label" if looks_numeric else "tick_label"
            _extend_section(target_section, "axis_labels", [item])

    # Fallback: use extracted element tick labels for bar-like and box charts
    if not section_records["tick_label"] and chart_type in {"bar", "histogram", "box"}:
        element_key = "bars" if chart_type != "box" else "elements"
        for element in result.get(element_key, []):
            if isinstance(element, dict):
                label_text = element.get("label") or element.get("series") or element.get("x_label")
                if label_text:
                    section_records["tick_label"].append(({"text": str(label_text), "confidence": 1.0}, "extracted_element"))

    # Layout text from doclayout
    layout_items = metadata.get("layout_elements", [])
    _extend_section("layout_text", "doclayout", layout_items)

    return section_records
