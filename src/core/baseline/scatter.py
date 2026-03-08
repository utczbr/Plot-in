"""Scatter-specific baseline estimation helpers."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .geometry import validate_xyxy


def scatter_axis_baseline(img: np.ndarray, axis_labels: Optional[List[Dict]], axis: str) -> Optional[float]:
    """Estimate scatter baseline from edge-aligned labels with adaptive thresholds."""
    if not axis_labels:
        return None

    h, w = img.shape[:2]
    valid_labels = [lbl for lbl in axis_labels if "xyxy" in lbl and validate_xyxy(lbl["xyxy"])]

    if not valid_labels:
        return None

    if axis == "x":
        bottom_labels = [
            lbl for lbl in valid_labels
            if ((lbl["xyxy"][1] + lbl["xyxy"][3]) / 2.0) >= h * 0.75
        ]
        if bottom_labels:
            ys = np.array([(lbl["xyxy"][1] + lbl["xyxy"][3]) / 2.0 for lbl in bottom_labels], dtype=np.float32)
            return float(np.nanmedian(ys))

        top_labels = [
            lbl for lbl in valid_labels
            if ((lbl["xyxy"][1] + lbl["xyxy"][3]) / 2.0) <= h * 0.25
        ]
        if top_labels:
            ys = np.array([(lbl["xyxy"][1] + lbl["xyxy"][3]) / 2.0 for lbl in top_labels], dtype=np.float32)
            return float(np.nanmedian(ys))
    else:
        left_labels = [
            lbl for lbl in valid_labels
            if ((lbl["xyxy"][0] + lbl["xyxy"][2]) / 2.0) <= w * 0.25
        ]
        if left_labels:
            xs = np.array([(lbl["xyxy"][0] + lbl["xyxy"][2]) / 2.0 for lbl in left_labels], dtype=np.float32)
            return float(np.nanmedian(xs))

        right_labels = [
            lbl for lbl in valid_labels
            if ((lbl["xyxy"][0] + lbl["xyxy"][2]) / 2.0) >= w * 0.75
        ]
        if right_labels:
            xs = np.array([(lbl["xyxy"][0] + lbl["xyxy"][2]) / 2.0 for lbl in right_labels], dtype=np.float32)
            return float(np.nanmedian(xs))

    return None


__all__ = ["scatter_axis_baseline"]
