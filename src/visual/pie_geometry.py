"""Helpers for rendering pie pose detections as wedge/ray overlays."""

from math import isfinite
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[int, int]


def _to_scaled_point(raw_point: Any, scale_x: float, scale_y: float) -> Optional[Point]:
    """Convert a keypoint-like value to an integer screen point."""
    if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
        return None

    try:
        x = float(raw_point[0])
        y = float(raw_point[1])
    except (TypeError, ValueError):
        return None

    if not (isfinite(x) and isfinite(y)):
        return None

    return int(round(x * scale_x)), int(round(y * scale_y))


def extract_slice_overlay_points(
    det: Dict[str, Any],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    expected_keypoints: int = 5,
) -> Optional[Tuple[Point, List[Point]]]:
    """
    Extract pie overlay geometry from pose keypoints.

    Contract:
    - keypoints[0] = center
    - keypoints[1] = arc_start
    - keypoints[2] = arc_inter_1
    - keypoints[3] = arc_inter_2
    - keypoints[4] = arc_end
    """
    keypoints = det.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != expected_keypoints:
        return None

    points: List[Point] = []
    for raw_point in keypoints:
        point = _to_scaled_point(raw_point, scale_x, scale_y)
        if point is None:
            return None
        points.append(point)

    center_pt = points[0]
    arc_points = points[1:]
    return center_pt, arc_points
