"""
Benchmark manifest adapter utilities for isolated A/B evaluation.

This module adapts ChartQA/PlotQA-style manifest records into the current
directory-based evaluator contract:
  - GT files: <sample_id>_gt.json
  - Prediction files: <sample_id>_analysis.json
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence


_MANIFEST_LIST_KEYS = (
    "records",
    "samples",
    "items",
    "data",
    "examples",
    "questions",
)


@dataclass(frozen=True)
class ResolvedPair:
    sample_id: str
    gt_file: Path
    baseline_pred_file: Path
    candidate_pred_file: Path
    gt_unified_file: Optional[Path] = None
    gt_format: str = "gt_json"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializedSubset:
    gt_dir: Path
    baseline_pred_dir: Path
    candidate_pred_dir: Path
    resolved_count: int
    skipped_count: int
    skipped_reasons: Dict[str, int] = field(default_factory=dict)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load manifest rows from JSON array/object or JSONL."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row {line_no} must be an object.")
                rows.append(row)
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return _extract_manifest_rows(payload)

    raise ValueError(f"Unsupported manifest format: {path}. Expected .json or .jsonl")


def _extract_manifest_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in _MANIFEST_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        if rows is None:
            raise ValueError(
                f"Manifest JSON object must include one list key in {_MANIFEST_LIST_KEYS}."
            )
    else:
        raise ValueError("Manifest JSON must be an array or an object containing a list.")

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row at index {idx} must be an object.")

    return rows


def normalize_manifest_record(record: Dict[str, Any], benchmark_format: str) -> Dict[str, Any]:
    """
    Normalize one manifest row to a compact adapter schema.

    benchmark_format: auto|chartqa|plotqa
    """
    if not isinstance(record, dict):
        raise ValueError("Manifest record must be an object.")

    fmt = (benchmark_format or "auto").strip().lower()
    if fmt not in {"auto", "chartqa", "plotqa"}:
        raise ValueError(f"Unsupported benchmark_format: {benchmark_format}")

    resolved_format = _detect_format(record) if fmt == "auto" else fmt
    sample_id = resolve_sample_id(record)

    question = record.get("query", record.get("question"))
    answer = record.get("label", record.get("answer"))
    qa_type = record.get("type", record.get("qa_type"))

    metadata: Dict[str, Any] = {}
    if question is not None:
        metadata["question"] = question
    if answer is not None:
        metadata["answer"] = answer
    if qa_type is not None:
        metadata["qa_type"] = qa_type

    return {
        "sample_id": sample_id,
        "benchmark_format": resolved_format,
        "metadata": metadata,
    }


def _detect_format(record: Dict[str, Any]) -> str:
    if "image_index" in record:
        return "plotqa"
    if "imgname" in record:
        return "chartqa"
    if "query" in record or "label" in record:
        return "chartqa"
    if "question" in record and "answer" in record:
        return "plotqa"
    return "chartqa"


def resolve_sample_id(record: Dict[str, Any]) -> str:
    """Resolve sample id from preferred key order."""
    key_order = ("sample_id", "imgname", "image", "image_path", "image_index")

    for key in key_order:
        if key not in record:
            continue
        raw_value = record.get(key)
        if raw_value is None:
            continue
        if key == "sample_id":
            token = str(raw_value).strip()
        elif key == "image_index":
            token = _normalize_index_value(raw_value)
        else:
            token = _normalize_path_like_id(raw_value)
        if token:
            return token

    raise ValueError(
        "Could not resolve sample_id from record. Expected one of "
        "sample_id/imgname/image/image_path/image_index."
    )


def _normalize_index_value(value: Any) -> str:
    if isinstance(value, bool):
        return ""

    if isinstance(value, int):
        token = str(value)
    elif isinstance(value, float):
        token = str(int(value)) if value.is_integer() else str(value)
    else:
        token = str(value).strip()

    if not token:
        return ""

    return token


def _normalize_path_like_id(value: Any) -> str:
    token = str(value).strip()
    if not token:
        return ""

    token_path = Path(token)
    token_name = token_path.name
    if token_name:
        stem = Path(token_name).stem
        if stem:
            return stem.strip()

    return token.strip()


def convert_unified_to_gt_payload(unified: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    """
    Convert generator unified labels into evaluator-compatible *_gt.json payload.
    """
    image_metadata = unified.get("image_metadata", {})
    chart_analysis = unified.get("chart_analysis", {})
    chart_generation_metadata = unified.get("chart_generation_metadata", {})

    image_path = image_metadata.get("image_path") or f"{sample_id}.png"
    payload: Dict[str, Any] = {
        "image_path": image_path,
        "charts": [],
        "annotations": [],
    }

    resolution = image_metadata.get("resolution", {})
    width = _coerce_int(resolution.get("width"))
    height = _coerce_int(resolution.get("height"))
    if width is not None and height is not None:
        payload["image_size"] = {"width": width, "height": height}

    raw_annotations = unified.get("raw_annotations", [])
    if isinstance(raw_annotations, list):
        payload["annotations"] = _convert_annotations(raw_annotations)

    chart_type = str(chart_analysis.get("chart_type", "unknown")).strip().lower() or "unknown"
    axis_calibration = _build_axis_calibration(chart_generation_metadata.get("scale_axis_info", {}))

    chart: Dict[str, Any] = {
        "chart_type": chart_type,
        "axis_calibration": axis_calibration,
    }

    if chart_type == "bar":
        chart["bar_values"] = _extract_bar_values(chart_generation_metadata)
    elif chart_type == "histogram":
        chart["histogram_bins"] = _extract_histogram_bins(chart_generation_metadata)
    elif chart_type in {"line", "scatter", "area"}:
        chart["data_points"] = _extract_data_points(chart_generation_metadata)
    elif chart_type == "box":
        chart["boxplot_statistics"] = _extract_boxplot_statistics(chart_generation_metadata)

    payload["charts"] = [chart]
    return payload


def _convert_annotations(raw_annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for annotation in raw_annotations:
        if not isinstance(annotation, dict):
            continue
        class_id_raw = annotation.get("class_id")
        class_id = _coerce_int(class_id_raw)

        entry: Dict[str, Any] = {
            "bbox": annotation.get("bbox", []),
            "text": annotation.get("text"),
            "confidence": 1.0,
        }
        entry["class_id"] = class_id if class_id is not None else class_id_raw
        if "class_name" in annotation:
            entry["class_name"] = annotation.get("class_name")

        converted.append(entry)
    return converted


def _build_axis_calibration(scale_axis_info: Any) -> Dict[str, Any]:
    scale_axis_info = scale_axis_info if isinstance(scale_axis_info, dict) else {}

    return {
        "x_axis": {
            "min": _coerce_float(scale_axis_info.get("x_min")),
            "max": _coerce_float(scale_axis_info.get("x_max")),
            "scale": scale_axis_info.get("x_scale", "linear"),
        },
        "y_axis": {
            "min": _coerce_float(scale_axis_info.get("y_min")),
            "max": _coerce_float(scale_axis_info.get("y_max")),
            "scale": scale_axis_info.get("y_scale", "linear"),
        },
        "primary_scale_axis": scale_axis_info.get("primary_scale_axis", "y"),
        "secondary_scale_axis": scale_axis_info.get("secondary_scale_axis"),
    }


def _extract_bar_values(chart_generation_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    bar_info = chart_generation_metadata.get("bar_info", [])
    if not isinstance(bar_info, list):
        return []

    values: List[Dict[str, Any]] = []
    for bar in bar_info:
        if not isinstance(bar, dict):
            continue
        value = _coerce_float(bar.get("top"))
        if value is None:
            value = _coerce_float(bar.get("height"))
        if value is None:
            value = 0.0

        item: Dict[str, Any] = {"value": float(value)}
        center = _coerce_float(bar.get("center"))
        bottom = _coerce_float(bar.get("bottom"))
        height = _coerce_float(bar.get("height"))
        width = _coerce_float(bar.get("width"))
        if center is not None:
            item["x_position"] = center
        if bottom is not None:
            item["y_position"] = bottom
        if height is not None:
            item["height"] = height
        if width is not None:
            item["width"] = width
        values.append(item)
    return values


def _extract_histogram_bins(chart_generation_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    bar_info = chart_generation_metadata.get("bar_info", [])
    if not isinstance(bar_info, list):
        return []

    bins: List[Dict[str, Any]] = []
    for idx, bar in enumerate(bar_info):
        if not isinstance(bar, dict):
            continue
        frequency = _coerce_float(bar.get("height"))
        if frequency is None:
            frequency = _coerce_float(bar.get("top"))
        if frequency is None:
            frequency = 0.0

        center = _coerce_float(bar.get("center"))
        width = _coerce_float(bar.get("width"))
        left_edge = center - (width / 2.0) if center is not None and width is not None else None
        right_edge = center + (width / 2.0) if center is not None and width is not None else None

        bin_payload: Dict[str, Any] = {
            "bin_index": idx,
            "frequency": float(frequency),
        }
        if left_edge is not None:
            bin_payload["left_edge"] = left_edge
        if right_edge is not None:
            bin_payload["right_edge"] = right_edge
        bins.append(bin_payload)
    return bins


def _extract_data_points(chart_generation_metadata: Dict[str, Any]) -> List[Dict[str, float]]:
    keypoint_info = chart_generation_metadata.get("keypoint_info", [])
    if not isinstance(keypoint_info, list):
        return []

    points: List[Dict[str, float]] = []
    for series in keypoint_info:
        if isinstance(series, dict):
            series_points = series.get("points", [])
            if isinstance(series_points, list):
                iterable = series_points
            else:
                iterable = [series]
        elif isinstance(series, list):
            iterable = series
        else:
            iterable = [series]

        for point in iterable:
            if not isinstance(point, dict):
                continue
            x_value = _coerce_float(point.get("x"))
            y_value = _coerce_float(point.get("y"))
            if x_value is None or y_value is None:
                continue
            points.append({"x": x_value, "y": y_value})
    return points


def _extract_boxplot_statistics(chart_generation_metadata: Dict[str, Any]) -> List[Dict[str, float]]:
    boxplot_metadata = chart_generation_metadata.get("boxplot_metadata", {})
    if not isinstance(boxplot_metadata, dict):
        return []

    stats: List[Dict[str, float]] = []
    candidates = boxplot_metadata.get("statistics")
    if not isinstance(candidates, list):
        candidates = boxplot_metadata.get("medians", [])

    if not isinstance(candidates, list):
        return []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        q1 = _first_float(candidate, ("q1", "quartile_1", "lower_quartile"))
        median = _first_float(candidate, ("median", "median_value", "q2"))
        q3 = _first_float(candidate, ("q3", "quartile_3", "upper_quartile"))
        if q1 is None or median is None or q3 is None:
            continue
        stats.append({"q1": q1, "median": median, "q3": q3})
    return stats


def _first_float(container: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in container:
            value = _coerce_float(container.get(key))
            if value is not None:
                return value
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_pair_paths(
    record: Dict[str, Any],
    gt_root: Path,
    baseline_root: Path,
    candidate_root: Path,
    gt_format: str = "auto",
) -> ResolvedPair:
    """Resolve one normalized manifest record to GT/baseline/candidate file paths."""
    sample_id = record.get("sample_id")
    if not sample_id:
        sample_id = resolve_sample_id(record)

    metadata = dict(record.get("metadata", {}))

    gt_root = Path(gt_root)
    baseline_root = Path(baseline_root)
    candidate_root = Path(candidate_root)
    requested_gt_format = (gt_format or "auto").strip().lower()
    if requested_gt_format not in {"auto", "gt_json", "unified_json"}:
        raise ValueError("gt_format must be one of: auto, gt_json, unified_json")

    gt_file = gt_root / f"{sample_id}_gt.json"
    gt_unified_file = gt_root / f"{sample_id}_unified.json"
    resolved_gt_format = requested_gt_format
    if requested_gt_format == "auto":
        if gt_file.exists():
            resolved_gt_format = "gt_json"
        elif gt_unified_file.exists():
            resolved_gt_format = "unified_json"
        else:
            resolved_gt_format = "gt_json"

    return ResolvedPair(
        sample_id=sample_id,
        gt_file=gt_file,
        baseline_pred_file=baseline_root / f"{sample_id}_analysis.json",
        candidate_pred_file=candidate_root / f"{sample_id}_analysis.json",
        gt_unified_file=gt_unified_file if resolved_gt_format == "unified_json" else None,
        gt_format=resolved_gt_format,
        metadata=metadata,
    )


def materialize_normalized_subset(
    pairs: Sequence[ResolvedPair],
    output_root: Path,
    missing_policy: str,
) -> MaterializedSubset:
    """
    Materialize resolved pairs into the evaluator-compatible folder layout.

    Output layout:
      <output_root>/manifest_adapter/gt/<N>_gt.json
      <output_root>/manifest_adapter/baseline/<N>_analysis.json
      <output_root>/manifest_adapter/candidate/<N>_analysis.json
    """
    policy = (missing_policy or "error").strip().lower()
    if policy not in {"error", "skip"}:
        raise ValueError("missing_policy must be 'error' or 'skip'.")

    base_dir = Path(output_root) / "manifest_adapter"
    gt_dir = base_dir / "gt"
    baseline_dir = base_dir / "baseline"
    candidate_dir = base_dir / "candidate"

    for directory in (gt_dir, baseline_dir, candidate_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for stale_file in directory.glob("*.json"):
            stale_file.unlink()

    skipped_reasons: Counter[str] = Counter()
    skipped_count = 0
    resolved_count = 0

    for pair in pairs:
        missing = []
        if pair.gt_format == "unified_json":
            if pair.gt_unified_file is None or not pair.gt_unified_file.exists():
                missing.append("missing_gt_unified")
        elif not pair.gt_file.exists():
            missing.append("missing_gt")
        if not pair.baseline_pred_file.exists():
            missing.append("missing_baseline_pred")
        if not pair.candidate_pred_file.exists():
            missing.append("missing_candidate_pred")

        if missing:
            if policy == "error":
                detail = ", ".join(missing)
                raise FileNotFoundError(
                    f"Missing required files for sample '{pair.sample_id}': {detail}"
                )
            skipped_count += 1
            skipped_reasons.update(missing)
            continue

        sample_name = f"{resolved_count:06d}"
        gt_output_file = gt_dir / f"{sample_name}_gt.json"
        if pair.gt_format == "unified_json":
            assert pair.gt_unified_file is not None
            with pair.gt_unified_file.open("r", encoding="utf-8") as fh:
                unified_payload = json.load(fh)
            converted_payload = convert_unified_to_gt_payload(unified_payload, pair.sample_id)
            with gt_output_file.open("w", encoding="utf-8") as fh:
                json.dump(converted_payload, fh, ensure_ascii=False, indent=2)
        else:
            shutil.copy2(pair.gt_file, gt_output_file)
        shutil.copy2(pair.baseline_pred_file, baseline_dir / f"{sample_name}_analysis.json")
        shutil.copy2(pair.candidate_pred_file, candidate_dir / f"{sample_name}_analysis.json")
        resolved_count += 1

    return MaterializedSubset(
        gt_dir=gt_dir,
        baseline_pred_dir=baseline_dir,
        candidate_pred_dir=candidate_dir,
        resolved_count=resolved_count,
        skipped_count=skipped_count,
        skipped_reasons=dict(skipped_reasons),
    )
