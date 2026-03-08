"""Geometry helpers for baseline detection."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from services.orientation_service import Orientation


def validate_xyxy(xyxy: Any) -> bool:
    """Validate bounding box has 4 finite numeric elements."""
    if not isinstance(xyxy, (list, tuple, np.ndarray)):
        return False
    if len(xyxy) < 4:
        return False
    try:
        vals = [float(x) for x in xyxy[:4]]
        return all(np.isfinite(v) for v in vals)
    except (TypeError, ValueError):
        return False


def aggregate_stack_near_ends(
    elements: List[Dict],
    orientation: Orientation,
    img_h: int,
    band_frac: float = 0.02,
    inverted_axis: bool = False,
) -> np.ndarray:
    """Collapse stacked segments into one representative near-end per band."""
    if not elements:
        return np.zeros((0,), dtype=np.float32)

    valid_elements = [el for el in elements if "xyxy" in el and validate_xyxy(el["xyxy"])]
    if not valid_elements:
        return np.zeros((0,), dtype=np.float32)

    arr = np.array([el["xyxy"][:4] for el in valid_elements], dtype=np.float32)

    if orientation == Orientation.HORIZONTAL:
        y_centers = (arr[:, 1] + arr[:, 3]) / 2.0
        band_h = max(1.0, band_frac * float(img_h))
        bands = np.floor(y_centers / band_h).astype(np.int32)

        near = np.minimum(arr[:, 0], arr[:, 2])
        far = np.maximum(arr[:, 0], arr[:, 2])
        pick = far if inverted_axis else near
    else:
        x_centers = (arr[:, 0] + arr[:, 2]) / 2.0
        band_h = max(1.0, band_frac * float(img_h))
        bands = np.floor(x_centers / band_h).astype(np.int32)

        near = np.maximum(arr[:, 1], arr[:, 3])
        far = np.minimum(arr[:, 1], arr[:, 3])
        pick = far if inverted_axis else near

    agg: Dict[int, List[float]] = {}
    for b, v in zip(bands, pick):
        agg.setdefault(int(b), []).append(float(v))

    if orientation == Orientation.VERTICAL and not inverted_axis:
        reps = [float(np.nanmax(vals)) for vals in agg.values()]
    elif orientation == Orientation.HORIZONTAL and not inverted_axis:
        reps = [float(np.nanmin(vals)) for vals in agg.values()]
    elif orientation == Orientation.VERTICAL and inverted_axis:
        reps = [float(np.nanmin(vals)) for vals in agg.values()]
    else:
        reps = [float(np.nanmax(vals)) for vals in agg.values()]

    return np.array(reps, dtype=np.float32)


def extract_near_far_ends(elements: List[Dict], is_vertical: bool) -> Dict[str, Any]:
    """Extract baseline-side and data-side coordinates with validation."""
    valid_elements = [el for el in elements if "xyxy" in el and validate_xyxy(el["xyxy"])]

    if not valid_elements:
        return {
            "near_ends": np.array([], dtype=np.float32),
            "far_ends": np.array([], dtype=np.float32),
            "near_min": 0.0,
            "near_max": 0.0,
            "far_min": 0.0,
            "far_max": 0.0,
            "near_avg": 0.0,
            "far_avg": 0.0,
        }

    xyxy_list = np.array([el["xyxy"][:4] for el in valid_elements], dtype=np.float32)

    if is_vertical:
        y1 = xyxy_list[:, 1]
        y2 = xyxy_list[:, 3]
        y_pair = np.stack([y1, y2], axis=1)
        near = np.max(y_pair, axis=1)
        far = np.min(y_pair, axis=1)
    else:
        x1 = xyxy_list[:, 0]
        x2 = xyxy_list[:, 2]
        x_pair = np.stack([x1, x2], axis=1)
        near = np.min(x_pair, axis=1)
        far = np.max(x_pair, axis=1)

    return {
        "near_ends": near.astype(np.float32),
        "far_ends": far.astype(np.float32),
        "near_min": float(np.min(near)),
        "near_max": float(np.max(near)),
        "far_min": float(np.min(far)),
        "far_max": float(np.max(far)),
        "near_avg": float(np.nanmean(near)),
        "far_avg": float(np.nanmean(far)),
    }


def axis_label_centers(axis_labels: List[Dict]) -> np.ndarray:
    """Compute label centers from [x1, y1, x2, y2] boxes."""
    if not axis_labels:
        return np.zeros((0, 2), dtype=np.float32)

    valid_labels = [lbl for lbl in axis_labels if "xyxy" in lbl and validate_xyxy(lbl["xyxy"])]
    if not valid_labels:
        return np.zeros((0, 2), dtype=np.float32)

    centers = np.array(
        [
            (
                (lbl["xyxy"][0] + lbl["xyxy"][2]) / 2.0,
                (lbl["xyxy"][1] + lbl["xyxy"][3]) / 2.0,
            )
            for lbl in valid_labels
        ],
        dtype=np.float32,
    )
    return centers


def element_centers_perp(elements: List[Dict], orientation: Orientation) -> np.ndarray:
    """Extract perpendicular-axis centers for elements."""
    if not elements:
        return np.zeros((0,), dtype=np.float32)

    valid_elements = [el for el in elements if "xyxy" in el and validate_xyxy(el["xyxy"])]
    if not valid_elements:
        return np.zeros((0,), dtype=np.float32)

    arr = np.array([el["xyxy"][:4] for el in valid_elements], dtype=np.float32)

    if orientation == Orientation.VERTICAL:
        centers = (arr[:, 0] + arr[:, 2]) / 2.0
    else:
        centers = (arr[:, 1] + arr[:, 3]) / 2.0

    return centers.astype(np.float32)


__all__ = [
    "validate_xyxy",
    "aggregate_stack_near_ends",
    "extract_near_far_ends",
    "axis_label_centers",
    "element_centers_perp",
]
