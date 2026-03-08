"""Statistical helpers for baseline detection."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def robust_location(values: np.ndarray, method: str = "median") -> float:
    """Compute robust location estimate with NaN handling."""
    if len(values) == 0:
        return float("nan")

    if method == "mean":
        return float(np.nanmean(values))

    return float(np.nanmedian(values))


def baseline_from_cluster(values: np.ndarray, mask: np.ndarray, method: str = "median") -> float:
    """Compute baseline for a cluster subset."""
    if values.size == 0 or not np.any(mask):
        return robust_location(values, method=method)

    return robust_location(values[mask], method=method)


def edge_proximity_scores(centers_1d: np.ndarray, low_edge: float, high_edge: float) -> Tuple[float, float]:
    """Compute evidence for labels concentrated near low/high edges."""
    if len(centers_1d) == 0:
        return 0.0, 0.0

    rng = max(float(high_edge - low_edge), 1e-6)
    norm = (centers_1d - low_edge) / rng
    norm = np.clip(norm, 0.0, 1.0)

    low_score = float(np.mean(np.exp(-norm / 0.08)))
    high_score = float(np.mean(np.exp(-(1.0 - norm) / 0.08)))

    return low_score, high_score


def half_label_balance(centers_1d: np.ndarray, median_pos: float) -> float:
    """Compute balance of labels across halves for dual-axis decision."""
    if len(centers_1d) == 0:
        return 0.0

    left = int(np.sum(centers_1d < median_pos))
    right = int(np.sum(centers_1d >= median_pos))
    total = left + right

    if total == 0:
        return 0.0

    return 1.0 - abs(left - right) / float(total)


__all__ = [
    "robust_location",
    "baseline_from_cluster",
    "edge_proximity_scores",
    "half_label_balance",
]
