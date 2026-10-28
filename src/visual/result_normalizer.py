"""Result payload normalization and confidence partitioning for GUI widgets."""

from typing import Any, Dict, List, Optional, Tuple


def _extract_calibration_r2(calibration_entry: Any) -> Optional[float]:
    if not calibration_entry:
        return None
    val = (
        calibration_entry.get("r2", calibration_entry.get("r_squared"))
        if isinstance(calibration_entry, dict)
        else getattr(calibration_entry, "r2", getattr(calibration_entry, "r_squared", None))
    )
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _resolve_baseline_coord(baselines: Any, orientation: str) -> Optional[float]:
    if not isinstance(baselines, list) or not baselines:
        return None
    axis_id = "y" if str(orientation).lower() == "vertical" else "x"
    entry = next((b for b in baselines if isinstance(b, dict) and b.get("axis_id") == axis_id), None)
    if entry is None and isinstance(baselines[0], dict):
        entry = baselines[0]
    try:
        return float(entry.get("value")) if entry and "value" in entry else None
    except (TypeError, ValueError):
        return None


def _normalize_result_payload_for_gui(
    result: Dict[str, Any],
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Normalize runtime pipeline result payloads into the legacy keys expected by GUI widgets.
    Keeps backward compatibility for existing legacy payload shapes.
    """
    normalized = dict(result)

    detections = normalized.get("detections")
    if not isinstance(detections, dict):
        detections = {}
    normalized["detections"] = detections

    orientation = str(normalized.get("orientation", "vertical")).lower()
    chart_type = str(normalized.get("chart_type", "")).lower()

    bars = normalized.get("bars")
    if not isinstance(bars, list):
        if chart_type in {"bar", "histogram"}:
            elements = normalized.get("elements")
            if isinstance(elements, list):
                bars = [element for element in elements if isinstance(element, dict)]
            else:
                bars = []
        else:
            bars = []

    normalized_bars = []
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        item = dict(bar)
        tick_label = item.get("tick_label")
        if not item.get("bar_label") and isinstance(tick_label, dict):
            tick_text = tick_label.get("text")
            if isinstance(tick_text, str) and tick_text.strip():
                item["bar_label"] = tick_text.strip()
            tick_bbox = tick_label.get("bbox")
            if "bar_label_bbox" not in item and isinstance(tick_bbox, list):
                item["bar_label_bbox"] = tick_bbox
        if item.get("pixel_height") is None:
            pixel_dimension = item.get("pixel_dimension")
            if pixel_dimension is not None:
                item["pixel_height"] = pixel_dimension
        normalized_bars.append(item)

    normalized["bars"] = normalized_bars

    scale_info = normalized.get("scale_info")
    if not isinstance(scale_info, dict):
        scale_info = {}
    normalized["scale_info"] = scale_info

    calibration = normalized.get("calibration")
    if isinstance(calibration, dict):
        primary = calibration.get("primary")
    else:
        primary = None
    r2 = _extract_calibration_r2(primary)
    if r2 is None:
        r2 = _extract_calibration_r2(scale_info)
    if r2 is not None:
        scale_info["r_squared"] = r2

    baseline_coord = normalized.get("baseline_coord")
    if baseline_coord is None:
        baseline_coord = _resolve_baseline_coord(normalized.get("baselines"), orientation)
    if baseline_coord is not None:
        try:
            baseline_coord = float(baseline_coord)
            normalized["baseline_coord"] = baseline_coord
            # Legacy UI expects this legacy field for visibility toggles.
            scale_info.setdefault("baseline_y_coord", baseline_coord)
        except (TypeError, ValueError):
            pass

    if "_assigned_bar_labels" not in normalized:
        metadata = normalized.get("metadata")
        if isinstance(metadata, dict):
            assigned = metadata.get("assigned_bar_labels", metadata.get("_assigned_bar_labels"))
            if isinstance(assigned, dict):
                normalized["_assigned_bar_labels"] = assigned

    image_dimensions = normalized.get("image_dimensions")
    if not isinstance(image_dimensions, dict):
        image_dimensions = {}
    if image_size is not None:
        width, height = image_size
        image_dimensions.setdefault("width", int(width))
        image_dimensions.setdefault("height", int(height))
    normalized["image_dimensions"] = image_dimensions

    from visual.data_tab_schema import build_data_tab_model
    data_model = normalized.get("data_tab_model")
    if not isinstance(data_model, dict):
        data_model = build_data_tab_model(normalized)
        normalized["data_tab_model"] = data_model

    return normalized


def partition_assets_by_confidence(
    assets: List[Any],
    benchmark_cache: Dict[str, str],
    conf_cache: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> Tuple[List[int], List[int]]:
    """Partition assets into (normal_indices, low_conf_indices) for exclusive sidebar display."""
    from services.confidence_extractor import extract_confidence_from_analysis_json

    def _is_low(asset: Any) -> bool:
        path = str(getattr(asset, 'image_path', asset))
        json_path = benchmark_cache.get(path)
        if json_path:
            if conf_cache is not None:
                if json_path not in conf_cache:
                    conf_cache[json_path] = extract_confidence_from_analysis_json(json_path)
                info = conf_cache[json_path]
            else:
                info = extract_confidence_from_analysis_json(json_path)
            if info is not None:
                return bool(info.get("is_low_confidence", False))
        info = getattr(asset, 'confidence_info', None)
        return bool(info.get("is_low_confidence", False)) if isinstance(info, dict) else False

    low = [i for i, a in enumerate(assets) if _is_low(a)]
    low_set = set(low)
    normal = [i for i in range(len(assets)) if i not in low_set]
    return normal, low
