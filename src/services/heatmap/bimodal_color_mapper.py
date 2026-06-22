"""
BimodalColorMapper — Phase 2 discrete/continuous color bar router.

Drop-in replacement for ColorMappingService when use_bimodal_router=True.
Exposes the same public API: calibrate_from_known_values(), map_color_to_value().

Patches applied (heatmap_pipeline_corrected.md §4.3):
  Patch 3  — Dynamic MeanShift bandwidth via estimate_bandwidth (quantile=0.1)
             with hard floor of 8.0 in uint8 LAB space.
  Addition 1 — Spatial order preservation: bins sorted by first-appearance
               index in the sequentially sampled pixel array, not by L*.
  Fix 6    — Colormap directionality: linspace(start_val, end_val) preserves
             physical sampling direction instead of linspace(min, max).
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from services.color_mapping_service import ColorMappingService
from services.heatmap.color_inverter import LUTColorInverter

logger = logging.getLogger(__name__)


class BimodalColorMapper:
    """
    Routes discrete colorbars to MeanShift + CIEDE2000 nearest-bin matching
    and continuous colorbars to the LUTColorInverter trilinear path.

    Parameters
    ----------
    base_mapper : ColorMappingService
        Existing mapper used for continuous bar inference (LUT-backed when
        use_ciede2000=True, legacy BGR curve otherwise).
    inverter : LUTColorInverter
        CIEDE2000 inverter; provides calculate_ciede2000() for classification
        and discrete-bin matching.
    sparsity_thresh : float
        Sparsity index ρ = P95(∇B) / (median(∇B) + ε) above which the bar
        is classified as discrete.
    eps : float
        Numerical stability floor in ρ denominator.
    """

    # Minimum MeanShift bandwidth in OpenCV uint8 LAB space.
    # 2.5 (designed for CIE L* ∈ [0,100]) causes severe over-segmentation
    # at uint8 scale. 8.0 ≈ 3.1 CIE L* units — a perceptually meaningful floor.
    _UINT8_LAB_MIN_BANDWIDTH: float = 8.0

    def __init__(
        self,
        base_mapper: ColorMappingService,
        inverter: LUTColorInverter,
        sparsity_thresh: float = 15.0,
        eps: float = 1e-4,
    ) -> None:
        self.base_mapper = base_mapper
        self.inverter = inverter
        self.sparsity_thresh = sparsity_thresh
        self.eps = eps

        self.is_discrete: bool = False
        self.discrete_bins: List[Tuple[np.ndarray, float]] = []
        self._last_confidence: float = 0.0
        self._last_value_source: str = "none"

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate_from_known_values(
        self,
        color_samples: List[Tuple[np.ndarray, float]],
        color_mode: str = "legacy",
    ) -> None:
        """
        Classify the colorbar and calibrate the appropriate path.

        Parameters
        ----------
        color_samples : list of (patch_bgr, scalar_value) tuples in
                        physical sampling order (top→bottom or left→right).
        color_mode    : forwarded to base_mapper for legacy/lut path selection.
        """
        # Always calibrate the base mapper for continuous fallback
        self.base_mapper.calibrate_from_known_values(color_samples, color_mode)

        # Also calibrate LUT path — always rebuild when re-calibrating
        # to avoid stale LUT from a previous image in batch mode.
        if hasattr(self.base_mapper, "calibration_curve"):
            self.inverter._is_calibrated = False  # force rebuild
            try:
                self.inverter.precompute_lut(
                    calibration_curve=self.base_mapper.calibration_curve,
                    min_val=self.base_mapper.min_value,
                    max_val=self.base_mapper.max_value,
                )
            except Exception as exc:
                logger.warning("BimodalColorMapper: LUT precomputation failed: %s", exc)

        # Extract pixel arrays for classification
        pixel_patches = [s[0] for s in color_samples if s[0].size > 0]
        if not pixel_patches:
            return

        # Convert patches to mean BGR → LAB for CIEDE2000 gradient analysis
        bar_means = np.array([
            cv2.cvtColor(p.astype(np.uint8), cv2.COLOR_BGR2LAB).mean(axis=(0, 1))
            if p.ndim == 3 else np.zeros(3)
            for p in pixel_patches
        ], dtype=np.uint8)

        self.is_discrete = self._classify_color_bar(bar_means)

        if self.is_discrete:
            # Fix 6: use first/last sample values (preserves sampling direction)
            start_val = float(color_samples[0][1])
            end_val   = float(color_samples[-1][1])
            self._calibrate_discrete_bins(bar_means, start_val, end_val)

    def _classify_color_bar(self, bar_pixels: np.ndarray) -> bool:
        """
        Classify bar as discrete vs continuous via CIEDE2000 gradient sparsity.

        ρ = P95(∇B) / (median(∇B) + ε)
        Discrete if ρ > sparsity_thresh or Var(∇B) > 50.
        """
        if len(bar_pixels) < 2:
            return False

        lab_f32 = bar_pixels.astype(np.float32)
        gradients = np.array([
            self.inverter.calculate_ciede2000(
                lab_f32[i].astype(np.float64),
                lab_f32[i + 1].astype(np.float64),
            )
            for i in range(len(lab_f32) - 1)
        ])

        if gradients.size == 0:
            return False

        p95 = float(np.percentile(gradients, 95))
        med = float(np.median(gradients))
        rho = p95 / (med + self.eps)

        return bool(rho > self.sparsity_thresh or float(np.var(gradients)) > 50.0)

    def _calibrate_discrete_bins(
        self,
        bar_pixels: np.ndarray,
        start_val: float,
        end_val: float,
    ) -> None:
        """
        Cluster colorbar samples with MeanShift and assign scalar values.

        Patch 3:   bandwidth = max(estimate_bandwidth(quantile=0.1), 8.0)
        Addition 1: bins sorted by first-appearance index in sampling order.
        Fix 6:     scalar values run from start_val → end_val (not min→max).
        """
        try:
            from sklearn.cluster import MeanShift, estimate_bandwidth
        except ImportError:
            logger.warning("BimodalColorMapper: sklearn not available — treating as continuous.")
            self.is_discrete = False
            return

        lab_f32 = bar_pixels.astype(np.float32)

        # Patch 3: dynamic bandwidth with uint8 stability floor
        try:
            estimated_bw = float(estimate_bandwidth(
                lab_f32,
                quantile=0.1,
                n_samples=min(500, len(lab_f32)),
            ))
        except Exception:
            estimated_bw = 0.0

        bandwidth = max(estimated_bw, self._UINT8_LAB_MIN_BANDWIDTH)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
        ms.fit(lab_f32)

        centroids      = ms.cluster_centers_   # (M, 3) float64
        cluster_labels = ms.labels_            # (N,)
        n_bins         = len(centroids)

        if n_bins <= 1:
            logger.warning("BimodalColorMapper: MeanShift produced ≤1 bin — treating as continuous.")
            self.is_discrete = False
            self.discrete_bins = []
            return

        # Addition 1: sort bins by first-appearance index in pixel sequence
        first_appearance = [int(np.argmax(cluster_labels == i)) for i in range(n_bins)]
        sort_order       = np.argsort(first_appearance)
        centroids_sorted = centroids[sort_order]

        # Fix 6: linspace from start_val → end_val preserves physical sampling direction
        scalar_values = np.linspace(start_val, end_val, n_bins)

        self.discrete_bins = [
            (centroids_sorted[i].astype(np.float64), float(scalar_values[i]))
            for i in range(n_bins)
        ]
        logger.info(
            "BimodalColorMapper: discrete bar with %d bins, values [%.3f, %.3f]",
            n_bins, start_val, end_val,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def map_color_to_value(self, cell_image: np.ndarray) -> float:
        """
        Map cell color to scalar value.

        Discrete path  → nearest-bin CIEDE2000 matching.
        Continuous path → LUTColorInverter.map_color_to_value() (O(1) trilinear).
        """
        if cell_image.size == 0:
            self._last_confidence   = 0.0
            self._last_value_source = "empty"
            return float(self.base_mapper.min_value)

        if self.is_discrete and self.discrete_bins:
            # Convert mean BGR → LAB for CIEDE2000 distance
            mean_bgr = cell_image.mean(axis=(0, 1)).astype(np.uint8)
            mean_lab = cv2.cvtColor(
                mean_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
            )[0, 0].astype(np.float64)

            best_val, best_dist = self.discrete_bins[0][1], float("inf")
            for centroid, scalar in self.discrete_bins:
                dist = self.inverter.calculate_ciede2000(mean_lab, centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_val  = scalar

            self._last_value_source = "discrete_ciede2000"
            self._last_confidence   = max(0.0, 1.0 - best_dist / 50.0)
            return best_val

        # Continuous path — use LUT inverter if calibrated, else base mapper
        if self.inverter._is_calibrated:
            val = self.inverter.map_color_to_value(cell_image)
            self._last_value_source = "lut_trilinear"
            self._last_confidence   = 0.95
            return val

        result = self.base_mapper.map_color_to_value(cell_image)
        self._last_value_source = getattr(self.base_mapper, "last_value_source", "base_mapper")
        self._last_confidence   = getattr(self.base_mapper, "last_confidence", 0.5)
        return result

    # ── Duck-type compatibility with ColorMappingService ──────────────────────

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    @property
    def last_value_source(self) -> str:
        return self._last_value_source

    @property
    def min_value(self) -> float:
        return self.base_mapper.min_value

    @property
    def max_value(self) -> float:
        return self.base_mapper.max_value

    @min_value.setter
    def min_value(self, v: float) -> None:
        self.base_mapper.min_value = v

    @max_value.setter
    def max_value(self, v: float) -> None:
        self.base_mapper.max_value = v
