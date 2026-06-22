"""
HybridGridAnchor — Phase 1 grid phase fusion.

Fuses the FFT-derived periods (T_x, T_y) from GoertzelLatticeDetector with
YOLO cell centroids to determine the exact phase (origin) of the grid,
using circular statistics to avoid the periodic-boundary fallacy.

Patches applied (heatmap_pipeline_corrected.md §3.4):
  Patch 1 — Circular Mean Fallacy fix: uses atan2(mean_sin, mean_cos)
             instead of arithmetic mean of (c_k mod T), which wraps
             incorrectly at period boundaries.
  Addition 2 — Resultant Vector Confidence Guard: if R̄ < circular_coherence_min
               the centroids have no coherent phase → return None → DBSCAN fallback.
"""
import logging
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class HybridGridAnchor:
    """
    Aligns a mathematical lattice (from Goertzel periods) to empirical YOLO
    cell centroids using circular-statistics phase estimation.

    Parameters
    ----------
    confidence_threshold : float
        Minimum YOLO detection confidence to include a cell as a phase anchor.
    snap_tolerance_ratio : float
        Maximum displacement (as fraction of period) to snap a mathematical
        grid line to the nearest empirical centroid.
    circular_coherence_min : float
        Minimum resultant-vector length R̄ required to accept the phase
        estimate.  Below this the centroids are too noisy → fallback.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        snap_tolerance_ratio: float = 0.25,
        circular_coherence_min: float = 0.2,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.snap_tolerance_ratio = snap_tolerance_ratio
        self.circular_coherence_min = circular_coherence_min

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def align_grid_to_detections(
        self,
        yolo_cells: List[Dict[str, Any]],
        T_x: float,
        T_y: float,
        image_shape: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return (col_centers, row_centers) as sorted pixel-coordinate arrays.

        Returns (None, None) if:
          - fewer than 3 high-confidence cells are available, or
          - circular coherence R̄ < circular_coherence_min for either axis.

        Parameters
        ----------
        yolo_cells : list of detection dicts with keys 'xyxy' or 'x1/y1/x2/y2'
                     and 'conf'.
        T_x, T_y   : grid periods in pixels (from GoertzelLatticeDetector).
        image_shape : (height, width) of the full image (not the ROI).
        """
        h, w = image_shape

        # Filter to high-confidence cells only
        high_conf = [
            c for c in yolo_cells if c.get("conf", 0) >= self.confidence_threshold
        ]
        if len(high_conf) < 3:
            logger.debug(
                "HybridGridAnchor: only %d high-conf cells (need ≥3) → DBSCAN fallback.",
                len(high_conf),
            )
            return None, None

        # Compute centroids — support both 'xyxy' list and x1/y1/x2/y2 keys
        cx_arr, cy_arr = self._extract_centroids(high_conf)

        # Circular phase estimation (Patch 1 + Addition 2)
        phase_x = self._circular_phase_estimate(cx_arr, T_x)
        phase_y = self._circular_phase_estimate(cy_arr, T_y)

        if phase_x is None or phase_y is None:
            logger.debug(
                "HybridGridAnchor: incoherent phase (R̄ below threshold) → DBSCAN fallback."
            )
            return None, None

        # Generate mathematical grid clipped to full-image bounds
        math_cols = np.arange(phase_x - T_x, w + T_x, T_x)
        math_cols = math_cols[(math_cols >= 0) & (math_cols <= w)]

        math_rows = np.arange(phase_y - T_y, h + T_y, T_y)
        math_rows = math_rows[(math_rows >= 0) & (math_rows <= h)]

        if len(math_cols) == 0 or len(math_rows) == 0:
            logger.debug("HybridGridAnchor: empty mathematical grid → DBSCAN fallback.")
            return None, None

        # Snap mathematical lines toward nearest empirical centroids
        col_centers = self._snap_to_empirical(
            math_cols, cx_arr, T_x * self.snap_tolerance_ratio
        )
        row_centers = self._snap_to_empirical(
            math_rows, cy_arr, T_y * self.snap_tolerance_ratio
        )

        return col_centers, row_centers

    # ──────────────────────────────────────────────────────────────────────────
    # Core helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _circular_phase_estimate(
        self, centroids: np.ndarray, period: float
    ) -> Optional[float]:
        """
        Compute the phase offset of 1-D centroid positions w.r.t. period T
        using circular (directional) statistics.

        Patch 1: maps positions to angles on the unit circle before averaging,
        so centroids near 0 and T-1 (which are adjacent) are handled correctly.

        Addition 2: if R̄ < circular_coherence_min the centroids are uniformly
        distributed in phase space (noisy YOLO) → return None to force fallback.

        Returns
        -------
        phase : float in [0, T) representing the grid origin offset, or None.
        """
        if period <= 0:
            return None

        # Map positions modulo T → angle on unit circle
        angles = 2.0 * np.pi * (centroids % period) / period

        mean_sin = np.mean(np.sin(angles))
        mean_cos = np.mean(np.cos(angles))

        # Resultant vector length (measures circular concentration)
        R = float(np.hypot(mean_sin, mean_cos))

        if R < self.circular_coherence_min:
            logger.debug(
                "Circular coherence R̄=%.3f < %.3f for period=%.1f → incoherent.",
                R, self.circular_coherence_min, period,
            )
            return None

        mean_angle = float(np.arctan2(mean_sin, mean_cos))
        if mean_angle < 0:
            mean_angle += 2.0 * np.pi

        return float(mean_angle * period / (2.0 * np.pi))

    @staticmethod
    def _snap_to_empirical(
        theoretical_pts: np.ndarray,
        empirical_pts: np.ndarray,
        tolerance: float,
    ) -> np.ndarray:
        """
        For each theoretical grid line, snap to the nearest empirical centroid
        if it falls within tolerance.  Lines with no nearby centroid remain
        at the mathematical position.
        """
        snapped = theoretical_pts.copy().astype(np.float64)
        if len(empirical_pts) == 0:
            return snapped

        for i, tp in enumerate(theoretical_pts):
            dists = np.abs(empirical_pts - tp)
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] <= tolerance:
                snapped[i] = float(empirical_pts[nearest_idx])

        return snapped

    @staticmethod
    def _extract_centroids(
        cells: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract (cx, cy) from detection dicts.
        Supports both 'xyxy' list format and explicit x1/y1/x2/y2 keys.
        """
        cx_list, cy_list = [], []
        for c in cells:
            if "xyxy" in c:
                x1, y1, x2, y2 = c["xyxy"]
            else:
                x1, y1 = c.get("x1", 0), c.get("y1", 0)
                x2, y2 = c.get("x2", 0), c.get("y2", 0)
            cx_list.append((x1 + x2) / 2.0)
            cy_list.append((y1 + y2) / 2.0)
        return np.array(cx_list, dtype=np.float64), np.array(cy_list, dtype=np.float64)
