"""Types and config for baseline detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.orientation_service import Orientation


@dataclass
class BaselineLine:
    axis_id: str
    orientation: Orientation
    value: float
    confidence: float = 0.5
    members: List[int] = field(default_factory=list)


@dataclass
class BaselineResult:
    baselines: List[BaselineLine]
    method: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorConfig:
    cluster_backend: str = "dbscan"
    dbscan_eps_px: float = 8.0
    dbscan_min_samples: int = 2
    hdbscan_min_cluster_size: int = 3
    hdbscan_min_samples: Optional[int] = None
    k_range: Tuple[int, int] = (1, 3)
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    gumbel_temperature: float = 0.7
    min_abs_sep_ratio: float = 0.30
    min_balance: float = 0.35
    vertical_valid_slack_px: float = 10.0
    horizontal_valid_slack_px: float = 10.0
    location_method: str = "median"
    calibration_mode: str = "prosac"
    stack_band_frac: float = 0.02
