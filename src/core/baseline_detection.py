# baseline_detection_production.py
# Production-ready modular baseline detection with all bugs fixed, enhanced multi-cluster support, robust calibration, and comprehensive #diagnostics.

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Protocol, runtime_checkable

import numpy as np

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
except Exception:
    DBSCAN = None  # type: ignore
    KMeans = None  # type: ignore
    silhouette_score = None  # type: ignore
    calinski_harabasz_score = None  # type: ignore

try:
    import hdbscan as hdbscan_mod
except Exception:
    hdbscan_mod = None

try:
    from .calibration.calibration_precise import prosac_weighted_optimized
except Exception:
    try:
        from calibration_precise import prosac_weighted_optimized
    except Exception:
        prosac_weighted_optimized = None

try:
    from .calibration import CalibrationFactory
    from .calibration.calibration_base import CalibrationResult, BaseCalibration
except ImportError:
    try:
        from calibration import CalibrationFactory
        from calibration.calibration_base import CalibrationResult, BaseCalibration
    except ImportError:
        CalibrationFactory = None
        CalibrationResult = None
        BaseCalibration = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    LINE = "line"
    AREA = "area"
    BOX = "box"


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


from utils.clustering import (
    Clusterer,
    DBSCANClusterer,
    HDBSCANClusterer,
    KMeansGumbelClusterer,
    gumbel_softmax,
    cluster_bars_by_axis
)


def _validate_xyxy(xyxy: Any) -> bool:
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


def _aggregate_stack_near_ends(
    elements: List[Dict],
    orientation: Orientation,
    img_h: int,
    band_frac: float = 0.02,
    inverted_axis: bool = False,
) -> np.ndarray:
    """
    Collapse stacked segments into one representative near-end per band.
    
    This is the KEY FIX for stacked bar charts - it removes internal segment joints
    from contaminating the baseline calculation.
    
    - For HORIZONTAL: group by Y center into bands; take min X (left) per band if not inverted,
      else take max X (right) per band.
    - For VERTICAL: group by X center into bands; take max Y (bottom) per band if not inverted,
      else take min Y (top) per band.
    
    Args:
        elements: List of detected chart elements
        orientation: HORIZONTAL or VERTICAL
        img_h: Image height for band size calculation
        band_frac: Fraction of image height to use as band width
        inverted_axis: Whether the axis is inverted (right-to-left or top-to-bottom)
    
    Returns:
        Array of representative near-end coordinates, one per stack band
    """
    if not elements:
        return np.zeros((0,), dtype=np.float32)

    valid_elements = [el for el in elements if "xyxy" in el and _validate_xyxy(el["xyxy"])]
    if not valid_elements:
        return np.zeros((0,), dtype=np.float32)

    arr = np.array([el["xyxy"][:4] for el in valid_elements], dtype=np.float32)
    
    if orientation == Orientation.HORIZONTAL:
        # Group by Y center (perpendicular to bar direction)
        y_centers = (arr[:, 1] + arr[:, 3]) / 2.0
        band_h = max(1.0, band_frac * float(img_h))
        bands = np.floor(y_centers / band_h).astype(np.int32)
        
        # Near end = left for normal, right for inverted
        near = np.minimum(arr[:, 0], arr[:, 2])
        far = np.maximum(arr[:, 0], arr[:, 2])
        pick = far if inverted_axis else near
    else:
        # Group by X center (perpendicular to bar direction)
        x_centers = (arr[:, 0] + arr[:, 2]) / 2.0
        band_h = max(1.0, band_frac * float(img_h))
        bands = np.floor(x_centers / band_h).astype(np.int32)
        
        # Near end = bottom for normal, top for inverted
        near = np.maximum(arr[:, 1], arr[:, 3])
        far = np.minimum(arr[:, 1], arr[:, 3])
        pick = far if inverted_axis else near

    # Aggregate by band - for baseline, take the extreme (furthest from data) per band
    agg = {}
    for b, v in zip(bands, pick):
        if b not in agg:
            agg[b] = [v]
        else:
            agg[b].append(v)

    # CRITICAL FIX: For baseline (near), take EXTREME per band, not median
    # For vertical normal: baseline at bottom (max Y), so take max
    # For horizontal normal: baseline at left (min X), so take min
    # For inverted: inverse logic
    if orientation == Orientation.VERTICAL and not inverted_axis:
        # Vertical normal: baseline at bottom (max Y)
        reps = [float(np.nanmax(vals)) for vals in agg.values()]
    elif orientation == Orientation.HORIZONTAL and not inverted_axis:
        # Horizontal normal: baseline at left (min X)
        reps = [float(np.nanmin(vals)) for vals in agg.values()]
    elif orientation == Orientation.VERTICAL and inverted_axis:
        # Vertical inverted: baseline at top (min Y)
        reps = [float(np.nanmin(vals)) for vals in agg.values()]
    else:  # Horizontal inverted
        # Horizontal inverted: baseline at right (max X)
        reps = [float(np.nanmax(vals)) for vals in agg.values()]
    
    return np.array(reps, dtype=np.float32)


def _extract_near_far_ends(elements: List[Dict], is_vertical: bool) -> Dict[str, Any]:
    """Extract baseline-side and data-side coordinates with comprehensive validation."""
    valid_elements = [el for el in elements if "xyxy" in el and _validate_xyxy(el["xyxy"])]
    
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


def _axis_label_centers(axis_labels: List[Dict]) -> np.ndarray:
    """Compute label centers correctly from [x1, y1, x2, y2] bounding boxes."""
    if not axis_labels:
        return np.zeros((0, 2), dtype=np.float32)
    
    valid_labels = [lbl for lbl in axis_labels if "xyxy" in lbl and _validate_xyxy(lbl["xyxy"])]
    if not valid_labels:
        return np.zeros((0, 2), dtype=np.float32)
    
    centers = np.array(
        [
            (
                (lbl["xyxy"][0] + lbl["xyxy"][2]) / 2.0,
                (lbl["xyxy"][1] + lbl["xyxy"][3]) / 2.0
            )
            for lbl in valid_labels
        ],
        dtype=np.float32,
    )
    return centers


def _element_centers_perp(elements: List[Dict], orientation: Orientation) -> np.ndarray:
    """Extract perpendicular axis centers for dual-axis assignment."""
    if not elements:
        return np.zeros((0,), dtype=np.float32)
    
    valid_elements = [el for el in elements if "xyxy" in el and _validate_xyxy(el["xyxy"])]
    if not valid_elements:
        return np.zeros((0,), dtype=np.float32)
    
    arr = np.array([el["xyxy"][:4] for el in valid_elements], dtype=np.float32)
    
    if orientation == Orientation.VERTICAL:
        centers = (arr[:, 0] + arr[:, 2]) / 2.0
    else:
        centers = (arr[:, 1] + arr[:, 3]) / 2.0
    
    return centers.astype(np.float32)


def _edge_proximity_scores(centers_1d: np.ndarray, low_edge: float, high_edge: float) -> Tuple[float, float]:
    """Compute evidence for labels concentrated near low/high edges."""
    if len(centers_1d) == 0:
        return 0.0, 0.0
    
    rng = max(float(high_edge - low_edge), 1e-6)
    norm = (centers_1d - low_edge) / rng
    norm = np.clip(norm, 0.0, 1.0)
    
    low_score = float(np.mean(np.exp(-norm / 0.08)))
    high_score = float(np.mean(np.exp(-(1.0 - norm) / 0.08)))
    
    return low_score, high_score


def _half_label_balance(centers_1d: np.ndarray, median_pos: float) -> float:
    """Compute balance of labels across halves for dual-axis decision."""
    if len(centers_1d) == 0:
        return 0.0
    
    left = int(np.sum(centers_1d < median_pos))
    right = int(np.sum(centers_1d >= median_pos))
    tot = left + right
    
    if tot == 0:
        return 0.0
    
    return 1.0 - abs(left - right) / float(tot)


def _decide_dual_axis(
    orientation: Orientation,
    axis_label_centers: np.ndarray,
    cluster_labels: np.ndarray,
    img_w: int,
    img_h: int,
    min_abs_sep_ratio: float = 0.30,
    min_balance: float = 0.35,
    backend_hint: str = "",
) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
    """Decide if dual axes exist using cluster structure and heuristics."""
    labs = cluster_labels[cluster_labels >= 0]
    unique = np.unique(labs)
    n_clusters = len(unique)
    
    if n_clusters < 2:
        return False, None, {"n_clusters": n_clusters, "reason": "single_cluster"}
    
    coord = axis_label_centers[:, 0] if orientation == Orientation.VERTICAL else axis_label_centers[:, 1]
    low_edge, high_edge = (0.0, float(img_w)) if orientation == Orientation.VERTICAL else (0.0, float(img_h))
    
    centroids = np.array([coord[cluster_labels == cid].mean() for cid in unique], dtype=np.float32)
    centroids_sorted = np.sort(centroids)
    
    sep = float(abs(centroids_sorted[-1] - centroids_sorted[0]))
    abs_thresh = (img_w if orientation == Orientation.VERTICAL else img_h) * float(min_abs_sep_ratio)
    
    low_score, high_score = _edge_proximity_scores(centroids, low_edge, high_edge)
    balance = _half_label_balance(coord, (low_edge + high_edge) / 2.0)
    
    kmeans_bias = 0.05 if "kmeans" in backend_hint.lower() else 0.0
    
    dual_basic = (sep >= abs_thresh) and (low_score > 0.5 and high_score > 0.5) and (balance > (min_balance + kmeans_bias))
    dual_cluster_hint = (n_clusters >= 3) and (balance > (0.25 + kmeans_bias))
    
    dual = bool(dual_basic or dual_cluster_hint)
    ordered_ids = unique[np.argsort(centroids)] if dual else None
    
    return dual, ordered_ids, {
        "n_clusters": int(n_clusters),
        "sep": float(sep),
        "abs_thresh": float(abs_thresh),
        "low_score": float(low_score),
        "high_score": float(high_score),
        "balance": float(balance),
        "dual_basic": bool(dual_basic),
        "dual_cluster_hint": bool(dual_cluster_hint),
    }


def _cluster_axis_labels_for_dual(
    axis_labels: List[Dict],
    orientation: Orientation,
    clusterer: Clusterer,
    img_w: int,
    img_h: int,
    min_abs_sep_ratio: float,
    min_balance: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Cluster axis labels to identify potential dual-axis structure."""
    centers = _axis_label_centers(axis_labels)
    
    if len(centers) == 0:
        return np.full((0,), -1, dtype=int), None, {"reason": "no_labels"}
    
    X = centers[:, 0:1] if orientation == Orientation.VERTICAL else centers[:, 1:2]
    labels = clusterer.fit_predict(X)
    
    dual, ordered_ids, daux = _decide_dual_axis(
        orientation,
        centers,
        labels,
        img_w,
        img_h,
        min_abs_sep_ratio=min_abs_sep_ratio,
        min_balance=min_balance,
        backend_hint=clusterer.name(),
    )
    
    info = {
        "dual": dual,
        "labels": labels,
        "ordered_ids": ordered_ids,
        "clusterer": clusterer.name(),
        "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "decision": daux,
    }
    
    return labels, ordered_ids, info


def _robust_location(values: np.ndarray, method: str = "median") -> float:
    """Compute robust location estimate with NaN handling."""
    if len(values) == 0:
        return float("nan")
    
    if method == "mean":
        return float(np.nanmean(values))
    
    return float(np.nanmedian(values))


def _baseline_from_cluster(values: np.ndarray, mask: np.ndarray, method: str = "median") -> float:
    """Compute baseline for a cluster subset."""
    if values.size == 0 or not np.any(mask):
        return _robust_location(values, method=method)
    
    return _robust_location(values[mask], method=method)


def _axis_id_map_for_dual(orientation: Orientation) -> Tuple[str, str]:
    """Map ordered cluster IDs to axis identifiers."""
    if orientation == Orientation.VERTICAL:
        return "y1", "y2"
    else:
        return "x1", "x2"


def _axis_id_single(orientation: Orientation) -> str:
    """Get axis identifier for single-axis case."""
    return "y" if orientation == Orientation.VERTICAL else "x"


def _group_clusters_into_axes(
    cluster_labels: np.ndarray,
    ordered_ids: np.ndarray,
    centers_1d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group multiple clusters into left/right (or top/bottom) axis groups.
    Uses median centroid position to split clusters.
    """
    if len(ordered_ids) < 2:
        return ordered_ids, np.array([], dtype=ordered_ids.dtype)
    
    if len(ordered_ids) == 2:
        return np.array([ordered_ids[0]]), np.array([ordered_ids[1]])
    
    centroids = np.array([centers_1d[cluster_labels == cid].mean() for cid in ordered_ids], dtype=np.float32)
    median_cent = np.median(centroids)
    
    left_group = ordered_ids[centroids < median_cent]
    right_group = ordered_ids[centroids >= median_cent]
    
    if len(left_group) == 0:
        left_group = np.array([ordered_ids[0]])
        right_group = ordered_ids[1:]
    elif len(right_group) == 0:
        right_group = np.array([ordered_ids[-1]])
        left_group = ordered_ids[:-1]
    
    return left_group, right_group


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
    stack_band_frac: float = 0.02  # NEW: band size for stack aggregation


from .enums import ChartType

class ModularBaselineDetector:
    """
    Modular, multi-algorithm baseline detection system with enhanced diagnostics.
    
    This class integrates multiple strategies for baseline detection:
    - Statistical clustering (DBSCAN, HDBSCAN, KMeans)
    - Scale-based calibration (zero-crossing from axis labels)
    - Heuristics (e.g., image edges, label positions)
    
    It follows a configurable policy to select the best method for a given chart.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize the detector with a configuration.
        
        Args:
            config: DetectorConfig object or None for defaults
        """
        self.cfg = config or DetectorConfig()
        logger.info(f"ModularBaselineDetector initialized with config: {self.cfg}")

    def _chart_type_constraints(self, chart_type: ChartType) -> Dict[str, Any]:
        """Get constraints for chart type."""
        if chart_type == ChartType.SCATTER:
            return {"force_dual": True, "allow_dual": True}
        if chart_type in (ChartType.LINE, ChartType.AREA, ChartType.HISTOGRAM):
            return {"allow_dual": False}
        if chart_type in (ChartType.BAR, ChartType.BOX):
            return {"allow_dual": True}
        return {"allow_dual": False}

    def detect(
        self,
        img: np.ndarray,
        chart_elements: List[Dict],
        axis_labels: Optional[List[Dict]] = None,
        orientation: Orientation = Orientation.VERTICAL,
        chart_type: ChartType = ChartType.BAR,
        image_size: Optional[Tuple[int, int]] = None,
        primary_calibration_zero: Optional[float] = None,  # NEW
        primary_calibration_result: Optional[CalibrationResult] = None,  # NEW
        secondary_axis_labels: Optional[List[Dict]] = None,  # NEW: for dual-axis
        dual_axis_info: Optional[Dict] = None,  # NEW: dual-axis metadata
    ) -> BaselineResult:
        try:
            if image_size is None:
                h, w = img.shape[:2]
            else:
                w, h = image_size
            
            is_vertical = orientation == Orientation.VERTICAL
            policy = self._chart_type_constraints(chart_type)

            # Handle dual-axis charts if dual-axis information is provided
            if (dual_axis_info and dual_axis_info.get('has_dual_axis', False) and 
                secondary_axis_labels is not None and len(secondary_axis_labels) >= 2):
                logger.info("Dual-axis chart detected: computing two baselines...")
                
                # Cluster bars by position if dual-axis info doesn't have threshold
                from utils.clustering import cluster_bars_by_axis
                bar_clusters = cluster_bars_by_axis(chart_elements, w, dual_axis_info)
                
                # Compute primary baseline (left bars)
                primary_baseline = self._detect_single_axis(
                    img,
                    bar_clusters['primary_bars'],
                    axis_labels,
                    orientation,
                    chart_type,
                    image_size
                )
                
                # Compute secondary baseline (right bars)
                secondary_baseline = self._detect_single_axis(
                    img,
                    bar_clusters['secondary_bars'],
                    secondary_axis_labels,
                    orientation,
                    chart_type,
                    image_size
                )
                
                if orientation == Orientation.VERTICAL:
                    baseline_lines = [
                        BaselineLine(axis_id='y1', value=primary_baseline, confidence=0.8, orientation=orientation),
                        BaselineLine(axis_id='y2', value=secondary_baseline, confidence=0.8, orientation=orientation)
                    ]
                else:
                    baseline_lines = [
                        BaselineLine(axis_id='x1', value=primary_baseline, confidence=0.8, orientation=orientation),
                        BaselineLine(axis_id='x2', value=secondary_baseline, confidence=0.8, orientation=orientation)
                    ]
                
                return BaselineResult(
                    baselines=baseline_lines,
                    method="dual_axis_clustering",
                    diagnostics={
                        'dual_axis': True,
                        'threshold_x': bar_clusters['threshold_x'],
                        'primary_elements_count': len(bar_clusters['primary_bars']),
                        'secondary_elements_count': len(bar_clusters['secondary_bars'])
                    }
                )

            # FIX: For horizontal charts, use primary zero directly from analysis.py
            #      This eliminates the duplicate recalibration that caused the 173.53-unit error
            if (orientation == Orientation.HORIZONTAL and 
                primary_calibration_zero is not None and 
                primary_calibration_result is not None):
                baseline_value = primary_calibration_zero
                logger.info(
                    f"✓ Using primary calibration zero for horizontal baseline: {baseline_value:.2f}px "
                    f"(skipping local recalibration)"
                )
                
                # Detect inversion from PRIMARY slope, not local refit
                inverted_axis = False
                if primary_calibration_result.coeffs:
                    m_primary = primary_calibration_result.coeffs[0]
                    inverted_axis = m_primary < 0.0
                    if inverted_axis:
                        logger.warning(f"⚠️ INVERTED X-axis: primary slope={m_primary:.6f} < 0")
                
                return BaselineResult(
                    baselines=[
                        BaselineLine(
                            axis_id='x',
                            orientation=orientation,
                            value=baseline_value,
                            confidence=0.9  # High confidence from primary calibration
                        )
                    ],
                    method="primary_calibration_zero",
                    diagnostics={
                        "primary_zero_used": True,
                        "inverted_axis": inverted_axis,
                        "source": "analysis.py primary calibration"
                    }
                )

            if chart_type == ChartType.SCATTER:
                y_b = self._scatter_axis_baseline(img, axis_labels, axis="y")
                x_b = self._scatter_axis_baseline(img, axis_labels, axis="x")
                bl: List[BaselineLine] = []
                if y_b is not None:
                    bl.append(BaselineLine(axis_id="y", orientation=orientation, value=float(y_b), confidence=0.7))
                if x_b is not None:
                    bl.append(BaselineLine(axis_id="x", orientation=orientation, value=float(x_b), confidence=0.7))
                return BaselineResult(baselines=bl, method="scatter_dual", diagnostics={"chart_type": chart_type.value})

            if not chart_elements:
                logger.warning("No chart elements provided")
                return BaselineResult(baselines=[], method="no_elements", diagnostics={"reason": "no_chart_elements"})

            # FIX #2: Detect axis inversion from calibration
            inverted_axis = False
            if primary_calibration_result is not None and hasattr(primary_calibration_result, 'coeffs'):
                # Use primary calibration result to detect inversion if available
                if primary_calibration_result.coeffs:
                    slope = primary_calibration_result.coeffs[0]
                    if orientation == Orientation.HORIZONTAL:
                        inverted_axis = slope < 0.0
                    else:
                        inverted_axis = slope > 0.0
                    if inverted_axis:
                        logger.info(f"Detected INVERTED axis from primary calibration: slope={slope:.4f}")
            elif self.cfg.calibration_mode != "none" and axis_labels and BaseCalibration is not None:
                try:
                    # Quick local linear fit to detect slope sign
                    coords, vals, _ = BaseCalibration._extract_points(
                        axis_labels, 
                        'x' if orientation == Orientation.HORIZONTAL else 'y', 
                        prefer_cleaned=True
                    )
                    if len(coords) >= 2:
                        A = np.vstack([coords, np.ones_like(coords)]).T
                        sol = np.linalg.lstsq(A, vals, rcond=None)[0]
                        slope = float(sol[0])
                        
                        # For HORIZONTAL: positive slope = left-to-right (normal)
                        #                 negative slope = right-to-left (inverted)
                        # For VERTICAL: negative slope = bottom-to-top (normal, Y increases down)
                        #               positive slope = top-to-bottom (inverted)
                        if orientation == Orientation.HORIZONTAL:
                            inverted_axis = slope < 0.0
                        else:
                            inverted_axis = slope > 0.0
                        
                        if inverted_axis:
                            logger.info(f"Detected INVERTED axis: slope={slope:.4f} for {orientation.value}")
                except Exception as e:
                    logger.debug(f"Could not detect axis inversion: {e}")
                    inverted_axis = False

            # FIX: Check primary_calibration_result BEFORE recalibrating
            if primary_calibration_result is not None and hasattr(primary_calibration_result, 'func'):
                # Primary calibration was successful - use it
                zero_baseline = primary_calibration_zero
                logger.info(
                    f"✅ Using PRIMARY calibration zero: {zero_baseline:.1f}px "
                    f"(R²={primary_calibration_result.r2:.4f})"
                )
            else:
                # Fallback: recalibrate locally
                zero_baseline = self._baseline_from_scale_zero(
                    axis_labels or [], 
                    is_vertical, 
                    use_as_fallback_only=True  # This triggers the warning when called as fallback
                ) if self.cfg.calibration_mode != "none" else None

            # Scatter charts use dual baseline logic
            if chart_type == ChartType.SCATTER:
                y_b = self._scatter_axis_baseline(img, axis_labels, axis="y")
                x_b = self._scatter_axis_baseline(img, axis_labels, axis="x")
                bl: List[BaselineLine] = []
                if y_b is not None:
                    bl.append(BaselineLine(axis_id="y", orientation=orientation, value=float(y_b), confidence=0.7))
                if x_b is not None:
                    bl.append(BaselineLine(axis_id="x", orientation=orientation, value=float(x_b), confidence=0.7))
                return BaselineResult(baselines=bl, method="scatter_dual", diagnostics={"chart_type": chart_type.value})

            # FIX #3: Stack-aware baseline estimation
            agg_near = _aggregate_stack_near_ends(
                chart_elements, 
                orientation, 
                img_h=h, 
                band_frac=self.cfg.stack_band_frac,
                inverted_axis=inverted_axis
            )
            
            if agg_near.size >= 2:
                # FIX #4: Use percentile-based estimation for stacks
                if orientation == Orientation.HORIZONTAL:
                    # Left-origin: choose low percentile; right-origin: choose high percentile
                    pct = 5.0 if not inverted_axis else 95.0
                    single_est = float(np.nanpercentile(agg_near, pct))
                else:
                    # Bottom-origin: choose high percentile; top-origin (inverted): choose low percentile
                    pct = 95.0 if not inverted_axis else 5.0
                    single_est = float(np.nanpercentile(agg_near, pct))
                
                logger.info(
                    f"Stack-aware baseline: {single_est:.1f}px (pct={pct:.0f}, "
                    f"inverted={inverted_axis}, n_bands={len(agg_near)})"
                )
            else:
                # Fallback for < 2 elements
                logger.warning("Insufficient elements for stack aggregation, using simple median")
                if is_vertical:
                    coords_raw = [max(el['xyxy'][1], el['xyxy'][3]) for el in chart_elements if _validate_xyxy(el.get('xyxy'))]
                else:
                    coords_raw = [min(el['xyxy'][0], el['xyxy'][2]) for el in chart_elements if _validate_xyxy(el.get('xyxy'))]
                    # FIX #4: Add fallback for single-bar cases in horizontal charts
                    if len(chart_elements) == 1 and axis_labels and not is_vertical:
                        # Single bar: baseline is likely at left edge of image or first label position
                        label_x_coords = [
                            (lbl['xyxy'][0] + lbl['xyxy'][2]) / 2.0 
                            for lbl in axis_labels 
                            if 'xyxy' in lbl and _validate_xyxy(lbl.get('xyxy'))
                        ]
                        if label_x_coords:
                            baseline_hint = min(label_x_coords) - 20  # 20px margin
                            logger.info(f"Single horizontal bar: using label-based baseline hint {baseline_hint:.1f}px")
                            coords_raw.append(baseline_hint)
                single_est = float(np.nanmedian(coords_raw)) if coords_raw else 0.0

            # CRITICAL FIX: For horizontal charts, PRIORITY is calibration zero (X-axis baseline)
            baseline_value = single_est
            if zero_baseline is not None and not np.isnan(zero_baseline):
                # For horizontal charts, ALWAYS use calibrated zero as the primary baseline, regardless of tolerance
                if orientation == Orientation.HORIZONTAL:
                    logger.info(
                        f"Using calibrated X-axis baseline for horizontal chart: {zero_baseline:.1f}px "
                        f"(zero-crossing from X-axis scale labels, overriding statistical estimate {single_est:.1f}px)"
                    )
                    baseline_value = float(zero_baseline)
                else:
                    # For vertical charts, use tolerance-based snapping as before
                    snap_tol = 10.0  # Increased tolerance for stacks
                    if abs(baseline_value - float(zero_baseline)) <= snap_tol:
                        logger.info(f"Snapping baseline {baseline_value:.1f} → {zero_baseline:.1f} (calibration zero)")
                        baseline_value = float(zero_baseline)
                    else:
                        # For vertical charts, if the difference is large, still consider using the calibration zero
                        # because statistical estimation can be unreliable in some cases
                        logger.warning(
                            f"Baseline {baseline_value:.1f} differs from calibration zero {zero_baseline:.1f} "
                            f"by {abs(baseline_value - zero_baseline):.1f}px (> {snap_tol}px tolerance)"
                        )
                        # For vertical charts with large discrepancy, warn user but still allow to use statistical for now
                        logger.warning(
                            f"Vertical chart has significant baseline discrepancy. "
                            f"Consider checking axis labels or calibration quality."
                        )
                        # Still warn user but allow the statistical estimate to be used for vertical charts
                        # when calibration doesn't match well

            return BaselineResult(
                baselines=[
                    BaselineLine(
                        axis_id=_axis_id_single(orientation), 
                        orientation=orientation, 
                        value=baseline_value, 
                        confidence=0.8
                    )
                ],
                method=f"single_stackaware_{self.cfg.cluster_backend}",
                diagnostics={
                    "inverted_axis": inverted_axis,
                    "n_bands": int(len(agg_near)),
                    "percentile_used": pct if agg_near.size >= 2 else None,
                    "calibration_zero": float(zero_baseline) if zero_baseline is not None else None,
                }
            )
        
        except Exception as e:
            logger.error(f"Baseline detection failed: {e}", exc_info=True)
            return BaselineResult(baselines=[], method="error", diagnostics={"error": str(e)})

    def _scatter_axis_baseline(self, img: np.ndarray, axis_labels: Optional[List[Dict]], axis: str) -> Optional[float]:
        """Estimate scatter baseline from edge-aligned labels with adaptive thresholds."""
        if not axis_labels:
            return None
        
        h, w = img.shape[:2]
        valid_labels = [lbl for lbl in axis_labels if "xyxy" in lbl and _validate_xyxy(lbl["xyxy"])]
        
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

    def _baseline_from_scale_zero(self, axis_labels: List[Dict], is_vertical: bool, use_as_fallback_only: bool = False) -> Optional[float]:
        """
        Infer baseline from scale labels using robust calibration with full factory integration.
        
        This method now uses the CalibrationFactory to leverage PROSAC/RANSAC for
        outlier-robust zero-crossing estimation, addressing the issue where minor OCR
        errors in "correct" values cause significant baseline shifts.
        
        Args:
            use_as_fallback_only: If True, log as fallback (primary calibration unavailable).
        """
        # Log intent
        if use_as_fallback_only:
            logger.warning(
                "Using _baseline_from_scale_zero as FALLBACK "
                "(primary calibration not available)"
            )
        
        if len(axis_labels) == 0:
            return None
        
        # Filter for usable labels with cleanedvalue
        def is_numeric(val):
            try:
                float(val)
                return True
            except (TypeError, ValueError):
                return False
        
        # For horizontal charts, filter low-confidence OCR labels
        usable = [
            lbl for lbl in axis_labels
            if "cleanedvalue" in lbl 
            and lbl["cleanedvalue"] is not None 
            and _validate_xyxy(lbl.get("xyxy"))
            and is_numeric(lbl["cleanedvalue"])
        ]
        
        # FIX #3: Improve OCR robustness for horizontal axes
        if not is_vertical and len(usable) > 3:  # Only for horizontal (X-axis) with sufficient labels
            # Filter out low-confidence OCR for X-axis labels
            high_conf_labels = [
                lbl for lbl in usable 
                if lbl.get('ocr_confidence', 0.5) > 0.7 and lbl.get('cleanedvalue') is not None
            ]
            
            if len(high_conf_labels) >= 3:  # Ensure we still have sufficient labels
                logger.info(f"Using {len(high_conf_labels)}/{len(usable)} high-confidence X-axis labels for calibration")
                usable = high_conf_labels
        
        if not usable:
            return None
        
        # Quick check for explicit zero labels
        zeros = [lbl for lbl in usable if abs(float(lbl["cleanedvalue"])) < 1e-6]
        if zeros:
            if is_vertical:
                coords = [(z["xyxy"][1] + z["xyxy"][3]) / 2.0 for z in zeros]
            else:
                coords = [(z["xyxy"][0] + z["xyxy"][2]) / 2.0 for z in zeros]
            return float(np.nanmean(coords))
        
        if len(usable) < 2:
            return None
        
        # Use CalibrationFactory based on config mode for robust fitting
        if CalibrationFactory is None:
            logger.warning("CalibrationFactory not available, falling back to basic interpolation")
            return self._baseline_fallback_interpolation(usable, is_vertical)
        
        try:
            axis_type = 'y' if is_vertical else 'x'
            
            # Map calibration_mode to engine selection
            if self.cfg.calibration_mode == 'fast':
                engine = CalibrationFactory.create('fast', use_weights=True)
            elif self.cfg.calibration_mode == 'ransac':
                engine = CalibrationFactory.create('ransac', max_trials=300, residual_threshold=3.0)
            elif self.cfg.calibration_mode in ('prosac', 'precise'):
                engine = CalibrationFactory.create('prosac', max_trials=500, lo_iters=2)
            elif self.cfg.calibration_mode == 'optimized':
                engine = CalibrationFactory.create('prosac', max_trials=400)
            else:
                logger.warning(f"Unknown calibration mode '{self.cfg.calibration_mode}', using prosac")
                engine = CalibrationFactory.create('prosac')
            
            # Perform robust calibration
            result: Optional[CalibrationResult] = engine.calibrate(usable, axis_type)
            
            if result is None or result.coeffs is None:
                logger.debug("Calibration returned None, falling back to interpolation")
                return self._baseline_fallback_interpolation(usable, is_vertical)
            
            m, b = result.coeffs
            
            # Validate slope magnitude
            if abs(m) < 1e-6:
                logger.debug("Calibration slope near zero, cannot solve for baseline")
                return self._baseline_fallback_interpolation(usable, is_vertical)
            
            # ADDED: Slope sign validation for axis orientation
            expected_negative = is_vertical  # Vertical axes typically have negative slope (y increases downward in pixels)
            if (m < 0) != expected_negative:
                logger.warning(
                    f"Unexpected slope sign: m={m:.4f} for {'vertical' if is_vertical else 'horizontal'} axis. "
                    f"Expected {'negative' if expected_negative else 'positive'}. Check axis orientation or data."
                )
            
            # Solve for zero crossing: 0 = m * coord + b => coord = -b / m
            zero_coord = -b / m
            
            # Log diagnostic information
            logger.info(
                f"Calibration-derived baseline: {zero_coord:.1f}px "
                f"(R²={result.r2:.3f}, slope={m:.4f}, intercept={b:.2f}, "
                f"inliers={result.inliers.sum() if result.inliers is not None else len(usable)}/{len(usable)})"
            )
            
            return float(zero_coord)
        
        except Exception as e:
            logger.warning(f"Calibration failed: {e}, falling back to interpolation")
            return self._baseline_fallback_interpolation(usable, is_vertical)

    def _baseline_fallback_interpolation(self, usable: List[Dict], is_vertical: bool) -> Optional[float]:
        """
        Fallback interpolation/extrapolation without undefined variables.
        
        Uses only coords and values from usable labels, no external weights or calibration.
        """
        try:
            if is_vertical:
                coords = np.array([(l["xyxy"][1] + l["xyxy"][3]) / 2.0 for l in usable], dtype=np.float32)
            else:
                coords = np.array([(l["xyxy"][0] + l["xyxy"][2]) / 2.0 for l in usable], dtype=np.float32)
            
            values = np.array([float(l["cleanedvalue"]) for l in usable], dtype=np.float32)
            
            # Sort by values for sign-flip detection
            order = np.argsort(values)
            v = values[order]
            c = coords[order]
            
            # Interpolate across sign flip (between negative and positive)
            signs = np.sign(v)
            diff_signs = np.diff(signs)
            flips = np.where(diff_signs != 0)[0]
            
            if len(flips) > 0:
                i = int(flips[0])
                v1, v2, c1, c2 = float(v[i]), float(v[i+1]), float(c[i]), float(c[i+1])
                if abs(v2 - v1) > 1e-6:
                    frac = abs(v1) / abs(v2 - v1)
                    zero = float(c1 + frac * (c2 - c1))
                    logger.info(f"Fallback interpolation: baseline at {zero:.1f}px (between {v1} and {v2})")
                    return zero
            
            # Extrapolate for all-positive values (use first two points)
            if v[0] > 0 and len(v) >= 2:
                denom = c[1] - c[0]
                if abs(denom) > 1e-6:
                    slope = (v[1] - v[0]) / denom
                    if abs(slope) > 1e-6:
                        zero = float(c[0] - v[0] / slope)
                        logger.info(f"Fallback extrapolation (all positive): baseline at {zero:.1f}px")
                        return zero
            
            # Extrapolate for all-negative values (use last two points)
            if v[-1] < 0 and len(v) >= 2:
                denom = c[-1] - c[-2]
                if abs(denom) > 1e-6:
                    slope = (v[-1] - v[-2]) / denom
                    if abs(slope) > 1e-6:
                        zero = float(c[-1] - v[-1] / slope)
                        logger.info(f"Fallback extrapolation (all negative): baseline at {zero:.1f}px")
                        return zero
            
            logger.warning("Fallback interpolation: no valid method found for baseline")
            return None
            
        except Exception as e:
            logger.error(f"Fallback interpolation failed: {e}", exc_info=True)
            return None


def detect_baselines(
    img: np.ndarray,
    chart_elements: List[Dict],
    axis_labels: Optional[List[Dict]] = None,
    orientation: Orientation = Orientation.VERTICAL,
    chart_type: ChartType = ChartType.BAR,
    image_size: Optional[Tuple[int, int]] = None,
    config: Optional[DetectorConfig] = None,
) -> BaselineResult:
    """
    Convenience function for baseline detection.
    
    Args:
        img: Input image
        chart_elements: Detected chart elements with 'xyxy' boxes
        axis_labels: Optional axis labels with 'xyxy' boxes and 'cleanedvalue'
        orientation: VERTICAL or HORIZONTAL
        chart_type: Chart type enum
        image_size: Optional (width, height) override
        config: Optional detector configuration
        
    Returns:
        BaselineResult with baselines and diagnostics
    """
    detector = ModularBaselineDetector(config=config)
    return detector.detect(
        img=img,
        chart_elements=chart_elements,
        axis_labels=axis_labels,
        orientation=orientation,
        chart_type=chart_type,
        image_size=image_size,
    )
