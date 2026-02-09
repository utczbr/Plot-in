"""Compatibility facade for baseline detection.

Canonical implementation now lives under ``core.baseline``.
This module keeps the historical import surface stable.
"""

from __future__ import annotations

from core.enums import ChartType
from core.baseline import (
    BaselineLine,
    BaselineResult,
    DetectorConfig,
    ModularBaselineDetector,
    detect_baselines,
)
from core.baseline.geometry import (
    aggregate_stack_near_ends as _aggregate_stack_near_ends,
    axis_label_centers as _axis_label_centers,
    element_centers_perp as _element_centers_perp,
    extract_near_far_ends as _extract_near_far_ends,
    validate_xyxy as _validate_xyxy,
)
from core.baseline.stats import (
    baseline_from_cluster as _baseline_from_cluster,
    robust_location as _robust_location,
)
from core.baseline.scatter import scatter_axis_baseline as _scatter_axis_baseline
from core.baseline.zero_crossing import (
    baseline_fallback_interpolation as _baseline_fallback_interpolation,
    baseline_from_scale_zero as _baseline_from_scale_zero,
)
from core.baseline.clustering import (
    Clusterer,
    DBSCANClusterer,
    HDBSCANClusterer,
    KMeansGumbelClusterer,
    gumbel_softmax,
    cluster_bars_by_axis,
)

__all__ = [
    "ChartType",
    "BaselineLine",
    "BaselineResult",
    "DetectorConfig",
    "ModularBaselineDetector",
    "detect_baselines",
    "Clusterer",
    "DBSCANClusterer",
    "HDBSCANClusterer",
    "KMeansGumbelClusterer",
    "gumbel_softmax",
    "cluster_bars_by_axis",
    "_validate_xyxy",
    "_aggregate_stack_near_ends",
    "_extract_near_far_ends",
    "_axis_label_centers",
    "_element_centers_perp",
    "_robust_location",
    "_baseline_from_cluster",
    "_scatter_axis_baseline",
    "_baseline_from_scale_zero",
    "_baseline_fallback_interpolation",
]
