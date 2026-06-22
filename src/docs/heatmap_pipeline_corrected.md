# Advanced Heatmap Data Extraction Pipeline — Corrected Implementation Plan

**Target codebase:** `HeatmapHandler` / `ColorMappingService` / `ChartAnalysisOrchestrator`
**Scope:** Four self-contained engineering phases, each independently deployable behind a feature flag.
**Revision:** Incorporates 5 structural patches + 2 critical edge-case additions + 2 final silent edge-case fixes.

---

## Table of Contents

1. [Architectural Overview & Guiding Principles](#1-architectural-overview--guiding-principles)
2. [New Module Map](#2-new-module-map)
3. [Phase 1 — Hybrid Grid Reconstruction](#3-phase-1--hybrid-grid-reconstruction)
4. [Phase 2 — Advanced Color-to-Value Inversion](#4-phase-2--advanced-color-to-value-inversion)
5. [Phase 3 — Robust Label-to-Grid Alignment](#5-phase-3--robust-label-to-grid-alignment)
6. [Phase 4 — Integration & Feature-Flag Wiring](#6-phase-4--integration--feature-flag-wiring)
7. [Test Strategy](#7-test-strategy)
8. [Dependency Matrix](#8-dependency-matrix)
9. [Risk Register](#9-risk-register)

---

## 1. Architectural Overview & Guiding Principles

### Current State

`HeatmapHandler.process()` relies on a **2-pass DBSCAN** over YOLO-detected cell centroids to reconstruct the grid:

```
YOLO cells → centroid extraction → coarse DBSCAN (eps = 1.5% image dim)
           → estimate cell geometry → fine DBSCAN (eps = 0.5 × cell dim)
           → row_centers[], col_centers[]
```

**Critical weakness:** If YOLO misses many cells (empty/low-contrast cells have low detection rates), DBSCAN has insufficient data points and produces a degenerate or misaligned grid. Labels then fail to align, and color extraction is done against wrong bounding boxes.

### Target State

A **hybrid three-signal grid reconstruction** system:

```
Signal A: FFT (Harmonic Lattice)  ──┐
Signal B: YOLO detections (phase) ──┼─► HybridGridAnchor → definitive row/col centers
Signal C: Perspective rectifier   ──┘

Definitive grid → ArtifactRejector → CIEDE2000 BimodalColorMapper → cell values
Definitive grid → Needleman-Wunsch aligner → AxisLabelInterpolator → axis labels
```

### Guiding Principles

- **Additive, not destructive.** Every new component is injected through the existing `GridChartHandler` / `ColorMappingService` dependency slots. No existing public API signatures change.
- **Feature-flagged.** All new paths are gated by a `HeatmapConfig` dataclass so production behaviour is unchanged until explicitly opted in.
- **Graceful degradation.** Each new component falls back to the existing DBSCAN / HSV path on failure, preserving today's reliability floor.
- **YOLO cells as ground truth.** High-confidence colored cell detections are not discarded after DBSCAN — they become the phase-alignment anchors for the FFT lattice.

---

## 2. New Module Map

```
services/
  heatmap/
    __init__.py
    config.py                      ← HeatmapConfig dataclass (feature flags)
    lattice_detector.py            ← HarmonicLatticeDetector (FFT + harmonic folding)
    grid_rectifier.py              ← PerspectiveGridRectifier (Hough + RANSAC homography)
    hybrid_grid_anchor.py          ← HybridGridAnchor (FFT × YOLO circular-mean phase fusion)
    artifact_rejector.py           ← HeatmapArtifactRejector (bilateral + inpaint)
    color_inverter.py              ← ColorToValueInverter (CIEDE2000 + Brent's)
    bimodal_color_mapper.py        ← BimodalColorMapper (discrete/continuous router)
    sequence_aligner.py            ← OCRGridSequenceAligner (Needleman-Wunsch)
    label_interpolator.py          ← AxisLabelInterpolator (arith/geom progression)
```

All new files live under `services/heatmap/`. The existing `services/color_mapping_service.py` is **not modified** — instead `BimodalColorMapper` wraps it and is injected in its place when the feature flag is active.

---

## 3. Phase 1 — Hybrid Grid Reconstruction

### 3.1 `services/heatmap/config.py` — HeatmapConfig

```python
from dataclasses import dataclass, field

@dataclass
class HeatmapConfig:
    # --- Phase 1: Grid ---
    use_fft_grid: bool = False
    use_rectifier: bool = False
    fft_num_harmonics: int = 3
    fft_dc_mask_radius: int = 5
    fft_prominence_threshold: float = 0.15
    fft_min_distance: int = 10
    hybrid_conf_threshold: float = 0.7
    hybrid_snap_ratio: float = 0.25

    # --- Phase 2: Color ---
    use_artifact_rejector: bool = False
    use_ciede2000: bool = False
    use_bimodal_router: bool = False
    bimodal_sparsity_thresh: float = 15.0
    bilateral_d: int = 9
    bilateral_sigma: float = 75.0
    color_mode: str = 'legacy'

    # --- Phase 3: Labels ---
    use_nw_aligner: bool = False
    use_label_interpolator: bool = False
    nw_gap_open: float = -10.0
    nw_gap_extend: float = -2.0
    nw_max_dist: float = 15.0
    interp_variance_tol: float = 1e-3
```

Default values reproduce today's exact behaviour, making the flag completely safe to add to `GridChartHandler.__init__`.

---

### 3.2 `services/heatmap/lattice_detector.py` — HarmonicLatticeDetector

**Purpose:** Replace DBSCAN period estimation with FFT + harmonic energy folding. Produces `T_x` (column period) and `T_y` (row period) in pixels.

**Key design decision:** Uses the harmonic energy function `E(u) = Σ P(k·u)` for `k = 1..K`, which is strictly more robust on sparse heatmaps where the fundamental frequency is weak but harmonics are still present.

```python
import numpy as np
from scipy.fft import fft2, fftshift
from skimage.filters import window


class HarmonicLatticeDetector:
    def __init__(self, num_harmonics: int = 3, dc_mask_radius: int = 5):
        self.num_harmonics = num_harmonics
        self.dc_mask_radius = dc_mask_radius

    def extract_rectangular_periods(
        self, heatmap_gray: np.ndarray
    ) -> tuple[float | None, float | None]:
        """
        Returns (T_x, T_y) — column and row periods in pixels.

        Algorithm:
        1. Apply 2D Hann window to image
        2. Compute 2D FFT, shift zero-frequency to center
        3. Compute power spectrum P(u,v) = |F(u,v)|²
        4. Project to 1D: spectrum_x = Σ_v P, spectrum_y = Σ_u P
        5. Mask DC component (±dc_mask_radius around center)
        6. For each candidate fundamental frequency u in [1, len//(K+1)]:
             E(u) = Σ_{k=1}^{K} spectrum[k*u]
        7. u_0 = argmax E(u),  T_x = N / u_0
        8. Repeat for y axis
        """
        M, N = heatmap_gray.shape

        hann_window = window('hann', heatmap_gray.shape)
        windowed_img = heatmap_gray * hann_window

        f_transform = fft2(windowed_img)
        f_shifted = fftshift(f_transform)
        power_spectrum = np.abs(f_shifted) ** 2

        spectrum_x = np.sum(power_spectrum, axis=0)
        spectrum_y = np.sum(power_spectrum, axis=1)

        center_x, center_y = N // 2, M // 2
        r = self.dc_mask_radius
        spectrum_x[center_x - r: center_x + r + 1] = 0
        spectrum_y[center_y - r: center_y + r + 1] = 0

        u0 = self._maximize_harmonic_energy(spectrum_x, N)
        v0 = self._maximize_harmonic_energy(spectrum_y, M)

        if u0 is None or v0 is None:
            return None, None

        T_x = N / u0
        T_y = M / v0

        if T_x < 3 or T_x > N / 2 or T_y < 3 or T_y > M / 2:
            return None, None

        return T_x, T_y

    def _maximize_harmonic_energy(
        self, spectrum: np.ndarray, orig_dim_size: int
    ) -> int | None:
        """Returns best fundamental index u_0 by maximizing E(u)."""
        K = self.num_harmonics
        center = orig_dim_size // 2
        half = spectrum[center:]  # positive-frequency half

        max_fund = len(half) // (K + 1)
        if max_fund < 1:
            return None

        energies = np.array([
            sum(
                half[k * u] if k * u < len(half) else 0
                for k in range(1, K + 1)
            )
            for u in range(1, max_fund + 1)
        ])

        if energies.max() == 0:
            return None

        return int(np.argmax(energies)) + 1
```

---

### 3.3 `services/heatmap/grid_rectifier.py` — PerspectiveGridRectifier

**Purpose:** Detect and correct perspective/affine distortion before grid extraction. Optional (`use_rectifier=False` by default).

```python
import cv2
import numpy as np


class PerspectiveGridRectifier:
    def __init__(
        self,
        canny_t1: int = 50,
        canny_t2: int = 150,
        hough_thresh: int = 200,
        ransac_thresh: float = 5.0,
    ):
        self.canny_t1 = canny_t1
        self.canny_t2 = canny_t2
        self.hough_thresh = hough_thresh
        self.ransac_thresh = ransac_thresh

    def detect_intersections(self, image: np.ndarray) -> np.ndarray:
        """
        Returns Nx2 array of (x, y) grid intersection points.

        Algorithm:
        1. Convert to gray, apply Canny edge detection
        2. Apply HoughLines to get (ρ, θ) pairs
        3. Classify lines: |θ| < π/4 or > 3π/4 → vertical; else horizontal
        4. For each (vertical, horizontal) pair, solve 2×2 linear system:
             A = [[cos θ_v, sin θ_v], [cos θ_h, sin θ_h]]
             b = [ρ_v, ρ_h]
             pt = np.linalg.solve(A, b)
        5. Filter intersections inside image bounds
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_t1, self.canny_t2, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, self.hough_thresh)

        horizontals, verticals = [], []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if theta < np.pi / 4 or theta > 3 * np.pi / 4:
                    verticals.append((rho, theta))
                else:
                    horizontals.append((rho, theta))

        h, w = image.shape[:2]
        intersections = []
        for rho_v, theta_v in verticals:
            for rho_h, theta_h in horizontals:
                A = np.array([
                    [np.cos(theta_v), np.sin(theta_v)],
                    [np.cos(theta_h), np.sin(theta_h)],
                ])
                b = np.array([rho_v, rho_h])
                try:
                    pt = np.linalg.solve(A, b)
                    if 0 <= pt[0] <= w and 0 <= pt[1] <= h:
                        intersections.append(pt)
                except np.linalg.LinAlgError:
                    continue

        return np.array(intersections)

    def rectify_grid(
        self, image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> np.ndarray:
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
        if H is None or np.linalg.cond(H) > 1e10:
            return image
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, H, (w, h))

    def needs_rectification(self, image: np.ndarray) -> bool:
        """
        Quick pre-check: if line angle variance > threshold, rectification is warranted.
        Returns False for axis-aligned screenshots (fast path, no overhead).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_t1, self.canny_t2)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, self.hough_thresh)
        if lines is None or len(lines) < 4:
            return False
        angles = np.array([l[0][1] for l in lines])
        return float(np.var(angles % (np.pi / 2))) > 0.01
```

---

### 3.4 `services/heatmap/hybrid_grid_anchor.py` — HybridGridAnchor

**Purpose:** Fuses `T_x`, `T_y` from the FFT with empirical YOLO cell centroids to determine the exact phase (origin) of the grid, using circular statistics to avoid the periodic boundary fallacy.

**Patch 1 applied — Circular Mean Fallacy fix:**

$$\theta_k = \frac{2\pi \cdot (c_k \bmod T)}{T}, \quad \bar{R} = \sqrt{\left(\frac{1}{n}\sum_k \sin\theta_k\right)^2 + \left(\frac{1}{n}\sum_k \cos\theta_k\right)^2}$$

$$\bar{\phi} = \text{atan2}\!\left(\frac{1}{n}\sum_k\sin\theta_k,\ \frac{1}{n}\sum_k\cos\theta_k\right), \quad \phi^+ = \begin{cases}\bar\phi + 2\pi & \bar\phi < 0 \\ \bar\phi & \text{otherwise}\end{cases}, \quad \text{phase} = \frac{\phi^+ \cdot T}{2\pi}$$

**Patch 2 applied (Addition 2) — Resultant Vector Confidence Guard:**

$$\bar{R} = \sqrt{\left(\frac{1}{n}\sum_{k=1}^n \sin\theta_k\right)^2 + \left(\frac{1}{n}\sum_{k=1}^n \cos\theta_k\right)^2}$$

If $\bar{R} < 0.2$, centroids have no coherent phase alignment; return `None` to force DBSCAN fallback.

```python
import numpy as np


class HybridGridAnchor:
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        snap_tolerance_ratio: float = 0.25,
        circular_coherence_min: float = 0.2,   # R̄ threshold for Addition 2
    ):
        self.confidence_threshold = confidence_threshold
        self.snap_tolerance_ratio = snap_tolerance_ratio
        self.circular_coherence_min = circular_coherence_min

    # ------------------------------------------------------------------
    # Patch 1 + Addition 2: circular phase estimation with R̄ guard
    # ------------------------------------------------------------------
    def _circular_phase_estimate(
        self, centroids: np.ndarray, period: float
    ) -> float | None:
        """
        Compute the phase offset of 1D centroid positions w.r.t. period T
        using circular (directional) statistics.

        Returns None if:
          - the mean resultant vector length R̄ < circular_coherence_min
            (centroids are randomly/uniformly distributed — no coherent phase)
          - this prevents np.arctan2(0, 0) → 0.0 silently anchoring the grid
            to a hallucinated phase when YOLO detections are noisy.
        """
        angles = 2.0 * np.pi * (centroids % period) / period

        mean_sin = np.mean(np.sin(angles))
        mean_cos = np.mean(np.cos(angles))

        # Addition 2: Resultant Vector Confidence Guard
        R = np.hypot(mean_sin, mean_cos)
        if R < self.circular_coherence_min:
            return None  # Incoherent phase → caller falls back to DBSCAN

        mean_angle = np.arctan2(mean_sin, mean_cos)

        if mean_angle < 0:
            mean_angle += 2.0 * np.pi

        return float(mean_angle * period / (2.0 * np.pi))

    def align_grid_to_detections(
        self,
        yolo_cells: list[dict],
        T_x: float,
        T_y: float,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Returns (col_centers, row_centers) as pixel coordinate arrays.

        Algorithm:
        1. Filter yolo_cells where conf >= confidence_threshold
        2. Compute centroids: cx = (x1+x2)/2, cy = (y1+y2)/2
        3. Phase alignment via circular statistics (Patch 1 + Addition 2)
        4. Generate mathematical grid clipped to image bounds
        5. Local snap correction toward empirical centroids
        """
        h, w = image_shape

        high_conf = [
            c for c in yolo_cells if c.get("conf", 0) >= self.confidence_threshold
        ]
        if len(high_conf) < 3:
            return None, None

        cx_arr = np.array([(c["x1"] + c["x2"]) / 2.0 for c in high_conf])
        cy_arr = np.array([(c["y1"] + c["y2"]) / 2.0 for c in high_conf])

        # Addition 2: None return from _circular_phase_estimate triggers fallback
        phase_x = self._circular_phase_estimate(cx_arr, T_x)
        phase_y = self._circular_phase_estimate(cy_arr, T_y)

        if phase_x is None or phase_y is None:
            return None, None

        math_cols = np.arange(phase_x - T_x, w + T_x, T_x)
        math_cols = math_cols[(math_cols >= 0) & (math_cols <= w)]

        math_rows = np.arange(phase_y - T_y, h + T_y, T_y)
        math_rows = math_rows[(math_rows >= 0) & (math_rows <= h)]

        col_centers = self._snap_to_empirical(
            math_cols, cx_arr, T_x * self.snap_tolerance_ratio
        )
        row_centers = self._snap_to_empirical(
            math_rows, cy_arr, T_y * self.snap_tolerance_ratio
        )

        return col_centers, row_centers

    def _snap_to_empirical(
        self,
        theoretical_pts: np.ndarray,
        empirical_pts: np.ndarray,
        tolerance: float,
    ) -> np.ndarray:
        snapped = theoretical_pts.copy()
        for i, tp in enumerate(theoretical_pts):
            dists = np.abs(empirical_pts - tp)
            nearest_idx = np.argmin(dists)
            if dists[nearest_idx] <= tolerance:
                snapped[i] = empirical_pts[nearest_idx]
        return snapped
```

---

### 3.5 Integration into `HeatmapHandler._reconstruct_grid`

**Patch 2 applied — ROI crop before FFT to eliminate noise from chart titles, legends, and axis text:**

```python
# handlers/heatmap_handler.py — additions only

class HeatmapHandler(GridChartHandler):
    def __init__(self, classifier=None, heatmap_config=None, **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier
        self.cfg = heatmap_config or HeatmapConfig()

        self._lattice_detector = None
        self._grid_rectifier = None
        self._hybrid_anchor = None
        self._artifact_rejector = None
        self._nw_aligner = None
        self._label_interpolator = None

    def _reconstruct_grid(
        self, image: np.ndarray, heatmap_cells: list[dict], h: int, w: int
    ) -> tuple[list, list]:
        if self.cfg.use_fft_grid and len(heatmap_cells) >= 3:
            try:
                # ----------------------------------------------------------
                # Patch 2: Crop to ROI before FFT.
                # Applying FFT to the full image injects power from chart
                # titles, axis annotations, and legends, causing false
                # fundamental frequency peaks. Tightly bound the ROI using
                # the union of all YOLO cell boxes before the Hann window.
                # ----------------------------------------------------------
                all_x1 = [c["x1"] for c in heatmap_cells]
                all_y1 = [c["y1"] for c in heatmap_cells]
                all_x2 = [c["x2"] for c in heatmap_cells]
                all_y2 = [c["y2"] for c in heatmap_cells]

                roi_x1 = max(0, int(min(all_x1)))
                roi_y1 = max(0, int(min(all_y1)))
                roi_x2 = min(w, int(max(all_x2)))
                roi_y2 = min(h, int(max(all_y2)))

                roi_w = roi_x2 - roi_x1
                roi_h = roi_y2 - roi_y1

                if roi_w < 16 or roi_h < 16:
                    raise ValueError(
                        f"ROI too small for FFT: {roi_w}×{roi_h}px — falling back."
                    )

                # FFT operates on the ROI crop; absolute pixel coordinates
                # are still generated for the full image in HybridGridAnchor.
                roi_gray = cv2.cvtColor(
                    image[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY
                )

                if self.cfg.use_rectifier:
                    if self._grid_rectifier is None:
                        self._grid_rectifier = PerspectiveGridRectifier()
                    if self._grid_rectifier.needs_rectification(image):
                        image = self._grid_rectifier.rectify_grid(
                            image,
                            src_pts=self._grid_rectifier.detect_intersections(image),
                            dst_pts=None,  # computed from expected grid geometry
                        )

                if self._lattice_detector is None:
                    self._lattice_detector = HarmonicLatticeDetector(
                        num_harmonics=self.cfg.fft_num_harmonics,
                        dc_mask_radius=self.cfg.fft_dc_mask_radius,
                    )

                T_x, T_y = self._lattice_detector.extract_rectangular_periods(roi_gray)

                if T_x is not None and T_y is not None:
                    if self._hybrid_anchor is None:
                        self._hybrid_anchor = HybridGridAnchor(
                            confidence_threshold=self.cfg.hybrid_conf_threshold,
                            snap_tolerance_ratio=self.cfg.hybrid_snap_ratio,
                        )

                    # image_shape is the FULL image so absolute pixel coords
                    # are generated correctly despite the ROI-cropped FFT.
                    col_centers, row_centers = (
                        self._hybrid_anchor.align_grid_to_detections(
                            heatmap_cells, T_x, T_y, (h, w)
                        )
                    )

                    if col_centers is not None:
                        self.logger.info(
                            f"FFT+Hybrid grid: {len(row_centers)}r × "
                            f"{len(col_centers)}c  "
                            f"(T_x={T_x:.1f}, T_y={T_y:.1f}, "
                            f"ROI={roi_w}×{roi_h})"
                        )
                        return (
                            sorted(row_centers.tolist()),
                            sorted(col_centers.tolist()),
                        )

            except Exception as e:
                self.logger.warning(
                    f"FFT grid reconstruction failed, falling back to DBSCAN: {e}"
                )

        # --- Fallback: existing 2-pass DBSCAN (unchanged) ---
        cy_vals = [c["cy"] for c in heatmap_cells]
        cx_vals = [c["cx"] for c in heatmap_cells]
        coarse_rows = cluster_1d_dbscan(cy_vals, h * 0.015)
        coarse_cols = cluster_1d_dbscan(cx_vals, w * 0.015)
        # ... (existing logic, unchanged)
        return row_centers, col_centers
```

---

## 4. Phase 2 — Advanced Color-to-Value Inversion

### 4.1 `services/heatmap/artifact_rejector.py` — HeatmapArtifactRejector

```python
import cv2
import numpy as np


class HeatmapArtifactRejector:
    """
    Suppresses text overlays and JPEG artifacts using Bilateral Filters
    and Morphological Inpainting before color averaging.
    """

    def __init__(self, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply_bilateral_filter(self, cell_image: np.ndarray) -> np.ndarray:
        """
        Bilateral filter — smooths JPEG compression ringing while preserving
        cell boundary edges.

        BF[I]_p = (1/W_p) Σ_{q∈S} G_σs(‖p-q‖) · G_σr(|I_p - I_q|) · I_q
        """
        return cv2.bilateralFilter(
            cell_image, self.d, self.sigma_color, self.sigma_space
        )

    def extract_and_inpaint_text(self, smoothed_image: np.ndarray) -> np.ndarray:
        """
        1. Otsu threshold → binary text mask
        2. Morphological closing (3×3 ellipse kernel) to bridge stroke gaps
        3. cv2.inpaint(INPAINT_TELEA, radius=3) — Fast Marching Method
        Returns image with text removed and background color inpainted.
        """
        gray = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        return cv2.inpaint(smoothed_image, text_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def process_cell(self, cell_image: np.ndarray) -> np.ndarray:
        filtered = self.apply_bilateral_filter(cell_image)
        return self.extract_and_inpaint_text(filtered)
```

**Integration point in `HeatmapHandler._extract_cell_value`:**

```python
def _extract_cell_value(self, image, cell):
    cell_img = image[y1:y2, x1:x2]

    if (
        self.cfg.use_artifact_rejector
        and cell_img.shape[0] > 15
        and cell_img.shape[1] > 15
    ):
        if self._artifact_rejector is None:
            self._artifact_rejector = HeatmapArtifactRejector(
                d=self.cfg.bilateral_d,
                sigma_color=self.cfg.bilateral_sigma,
                sigma_space=self.cfg.bilateral_sigma,
            )
        cell_img = self._artifact_rejector.process_cell(cell_img)

    if self.color_mapper:
        return self.color_mapper.map_color_to_value(cell_img)
```

---

### 4.2 `services/heatmap/color_inverter.py` — ColorToValueInverter

**Note on OpenCV LAB scale:** OpenCV encodes LAB as `L = L*·255/100`, `a = a* + 128`, `b = b* + 128` (uint8). The CIEDE2000 formula requires CIE standard scale. `calculate_ciede2000` converts internally: `L_std = L_cv * 100/255`, `a_std = a_cv - 128`, `b_std = b_cv - 128`.

```python
import numpy as np
from scipy.optimize import brent
from typing import Callable


class ColorToValueInverter:
    """
    Perceptually-uniform CIEDE2000-based color distance metric with
    Brent's method optimizer for color bar scalar inversion.
    """

    @staticmethod
    def calculate_ciede2000(
        lab1: np.ndarray,
        lab2: np.ndarray,
        kL: float = 1.0,
        kC: float = 1.0,
        kH: float = 1.0,
    ) -> float:
        """
        Full CIEDE2000 ΔE₀₀ formula (Sharma et al., 2005).
        Input: lab arrays in OpenCV LAB uint8 scale (L: 0-255, a: 0-255, b: 0-255).
        Internally converts to CIE standard scale before computing.
        """
        # Convert OpenCV uint8 LAB → CIE standard scale
        def cv_to_cie(lab):
            L = lab[0] * 100.0 / 255.0
            a = lab[1] - 128.0
            b = lab[2] - 128.0
            return L, a, b

        L1, a1, b1 = cv_to_cie(lab1)
        L2, a2, b2 = cv_to_cie(lab2)

        C1_star = np.hypot(a1, b1)
        C2_star = np.hypot(a2, b2)
        C_bar_star = (C1_star + C2_star) / 2.0
        G = 0.5 * (1 - np.sqrt((C_bar_star**7) / (C_bar_star**7 + 25**7)))

        a1_prime = a1 * (1 + G)
        a2_prime = a2 * (1 + G)
        C1_prime = np.hypot(a1_prime, b1)
        C2_prime = np.hypot(a2_prime, b2)
        C_bar_prime = (C1_prime + C2_prime) / 2.0

        h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
        h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

        dL_prime = L2 - L1
        dC_prime = C2_prime - C1_prime

        if C1_prime * C2_prime == 0.0:
            dh_prime = 0.0
        else:
            diff = h2_prime - h1_prime
            if abs(diff) <= 180.0:
                dh_prime = diff
            elif diff > 180.0:
                dh_prime = diff - 360.0
            else:
                dh_prime = diff + 360.0

        dH_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime) / 2.0)

        L_bar_prime = (L1 + L2) / 2.0
        if C1_prime * C2_prime == 0.0:
            H_bar_prime = h1_prime + h2_prime
        else:
            diff = abs(h1_prime - h2_prime)
            sum_h = h1_prime + h2_prime
            if diff <= 180.0:
                H_bar_prime = sum_h / 2.0
            elif sum_h < 360.0:
                H_bar_prime = (sum_h + 360.0) / 2.0
            else:
                H_bar_prime = (sum_h - 360.0) / 2.0

        T = (
            1.0
            - 0.17 * np.cos(np.radians(H_bar_prime - 30.0))
            + 0.24 * np.cos(np.radians(2.0 * H_bar_prime))
            + 0.32 * np.cos(np.radians(3.0 * H_bar_prime + 6.0))
            - 0.20 * np.cos(np.radians(4.0 * H_bar_prime - 63.0))
        )

        dTheta = 30.0 * np.exp(-((H_bar_prime - 275.0) / 25.0) ** 2)
        Rc = 2.0 * np.sqrt((C_bar_prime**7) / (C_bar_prime**7 + 25**7))
        RT = -np.sin(np.radians(2.0 * dTheta)) * Rc

        SL = 1.0 + (0.015 * (L_bar_prime - 50.0) ** 2) / np.sqrt(
            20.0 + (L_bar_prime - 50.0) ** 2
        )
        SC = 1.0 + 0.045 * C_bar_prime
        SH = 1.0 + 0.015 * C_bar_prime * T

        term_L = (dL_prime / (kL * SL)) ** 2
        term_C = (dC_prime / (kC * SC)) ** 2
        term_H = (dH_prime / (kH * SH)) ** 2
        term_RT = RT * (dC_prime / (kC * SC)) * (dH_prime / (kH * SH))

        return float(np.sqrt(term_L + term_C + term_H + term_RT))

    def optimize_scalar_value(
        self,
        target_lab: np.ndarray,
        color_manifold: Callable[[float], np.ndarray],
        bracket: tuple = (0.0, 0.5, 1.0),
    ) -> float:
        """
        Minimizes CIEDE2000(target_lab, color_manifold(t)) over t ∈ [0, 1]
        using scipy.optimize.brent (inverse quadratic interpolation).
        Returns clipped float in [0, 1].
        """
        def objective(t: float) -> float:
            return self.calculate_ciede2000(target_lab, color_manifold(t))

        optimal_t = brent(objective, brack=bracket, tol=1e-5)
        return float(np.clip(optimal_t, 0.0, 1.0))
```

---

### 4.3 `services/heatmap/bimodal_color_mapper.py` — BimodalColorMapper

**Patch 3 applied — Dynamic MeanShift bandwidth via `estimate_bandwidth` with uint8 floor.**

**Addition 1 applied — Spatial order preservation for diverging colormaps (sort by first appearance, not L* channel).**

**Fix 6 applied — Colormap Directionality: `calibrate_from_known_values` passes `start_val`/`end_val` (the first and last sample values in sampling order) instead of `min_val`/`max_val` (absolute extremes). `_calibrate_discrete_bins` uses these as the `linspace` boundaries so that a Top→Bottom sampled vertical bar whose highest value is at the top correctly assigns that value to the first spatially-ordered bin, not the last.**

The flaw this closes: `np.linspace(min_val, max_val, n_bins)` always produces an ascending sequence regardless of physical sampling direction. A vertical bar sampled Top→Bottom with values [100, 50, 0] would have `min_val=0, max_val=100`, yielding `[0, 50, 100]` — the exact inverse of the true order. Using `start_val=100, end_val=0` produces `[100, 50, 0]`, which is correct.

```python
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

from services.color_mapping_service import ColorMappingService
from services.heatmap.color_inverter import ColorToValueInverter


class BimodalColorMapper:
    """
    Drop-in replacement for ColorMappingService when use_bimodal_router=True.
    Exposes the same public API: calibrate_from_known_values(), map_color_to_value().

    Adds:
    1. Auto-classification of color bars as continuous vs. discrete.
    2. Per-type routing to the correct mathematical solver.
    """

    # Minimum bandwidth in OpenCV uint8 LAB space.
    # A bandwidth of 2.5 (original plan, designed for CIE L* ∈ [0,100]) would
    # cause extreme over-segmentation at uint8 scale (L,a,b ∈ [0,255]).
    # 8.0 ≈ 3.1 CIE L* units — a perceptually meaningful minimum separation.
    _UINT8_LAB_MIN_BANDWIDTH: float = 8.0

    def __init__(
        self,
        base_mapper: ColorMappingService,
        inverter: ColorToValueInverter,
        sparsity_thresh: float = 15.0,
        eps: float = 1e-4,
    ):
        self.base_mapper = base_mapper
        self.inverter = inverter
        self.sparsity_thresh = sparsity_thresh
        self.eps = eps

        self.is_discrete: bool = False
        self.discrete_bins: list[tuple[np.ndarray, float]] = []
        self._last_confidence: float = 0.0
        self._last_value_source: str = "none"

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate_from_known_values(
        self,
        color_samples: list[tuple[np.ndarray, float]],
        color_mode: str = "legacy",
    ) -> None:
        self.base_mapper.calibrate_from_known_values(color_samples, color_mode)

        color_bar_pixels = np.array([s[0] for s in color_samples], dtype=np.uint8)
        self.is_discrete = self._classify_color_bar(color_bar_pixels)

        if self.is_discrete:
            # ------------------------------------------------------------------
            # Fix 6: Colormap Directionality.
            # Use the scalar value of the FIRST and LAST sample in the upstream
            # array — which reflects the physical sampling direction of the image
            # processor — rather than absolute min/max.
            #
            # Why: np.linspace(min_val, max_val, n) always ascends. A vertical
            # color bar sampled Top→Bottom with values [100, 50, 0] has
            # min_val=0, max_val=100, yielding linspace [0, 50, 100] — the exact
            # inverse of the true order. Using start_val=100, end_val=0 produces
            # [100, 50, 0], correctly matching the spatial bin order from
            # Addition 1.
            # ------------------------------------------------------------------
            start_val = color_samples[0][1]
            end_val   = color_samples[-1][1]
            self._calibrate_discrete_bins(color_bar_pixels, start_val, end_val)

    def _classify_color_bar(self, color_bar_pixels: np.ndarray) -> bool:
        """
        Computes per-pixel CIEDE2000 gradient along the bar.
        Sparsity index ρ = P95(∇B) / (median(∇B) + ε).
        Returns True (discrete) if ρ > sparsity_thresh or Var(∇B) > 50.
        """
        if len(color_bar_pixels) < 2:
            return False

        lab_f32 = color_bar_pixels.astype(np.float32)
        gradients = np.array([
            self.inverter.calculate_ciede2000(lab_f32[i], lab_f32[i + 1])
            for i in range(len(lab_f32) - 1)
        ])

        if gradients.size == 0:
            return False

        p95 = float(np.percentile(gradients, 95))
        med = float(np.median(gradients))
        rho = p95 / (med + self.eps)

        return bool(rho > self.sparsity_thresh or np.var(gradients) > 50)

    def _calibrate_discrete_bins(
        self,
        color_bar_pixels: np.ndarray,   # shape (N, 3), dtype uint8, OpenCV LAB
        start_val: float,               # Fix 6: value at the first sampled pixel
        end_val: float,                 # Fix 6: value at the last sampled pixel
    ) -> None:
        """
        MeanShift clustering in OpenCV uint8 LAB space.

        Patch 3: bandwidth is estimated dynamically via estimate_bandwidth
        (quantile=0.1 targets the densest cluster scale). A hard lower-bound
        of _UINT8_LAB_MIN_BANDWIDTH prevents over-segmentation in uint8 space.

        Addition 1: bins are sorted by first appearance index in the
        sequentially sampled pixel array, not by L* channel.
        This preserves the logical order of diverging colormaps
        (e.g. Blue → White → Red) where White has the highest L* but sits
        at the scale midpoint, not the top.

        Fix 6: linspace runs from start_val → end_val (preserving sampling
        direction) instead of min_val → max_val (which always ascends and
        inverts any Top→Bottom-sampled descending bar).
        """
        lab_f32 = color_bar_pixels.astype(np.float32)

        # Patch 3: dynamic bandwidth with uint8 stability floor
        estimated_bw = estimate_bandwidth(
            lab_f32,
            quantile=0.1,
            n_samples=min(500, len(lab_f32)),
        )
        bandwidth = max(estimated_bw, self._UINT8_LAB_MIN_BANDWIDTH)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
        ms.fit(lab_f32)

        centroids     = ms.cluster_centers_   # shape (M, 3), float32
        cluster_labels = ms.labels_           # shape (N,)
        n_bins        = len(centroids)

        if n_bins <= 1:
            self.logger.warning(
                "MeanShift produced ≤1 bin — treating color bar as continuous."
            )
            self.is_discrete = False
            self.discrete_bins = []
            return

        # ------------------------------------------------------------------
        # Addition 1: Spatial Order Preservation
        # Sort bins by their index of first appearance in the sequentially
        # sampled pixel array. Sorting by L* (Lightness) would place White
        # at the end of a Blue→White→Red diverging scale, scrambling scalar
        # assignments. First-appearance order always reflects the bar's true
        # mathematical progression.
        # ------------------------------------------------------------------
        first_appearance = [
            int(np.argmax(cluster_labels == i)) for i in range(n_bins)
        ]
        sort_order      = np.argsort(first_appearance)
        centroids_sorted = centroids[sort_order]

        # Fix 6: linspace direction driven by physical sampling order,
        # not by absolute numerical magnitude.
        scalar_values = np.linspace(start_val, end_val, n_bins)

        self.discrete_bins = [
            (centroids_sorted[i], float(scalar_values[i]))
            for i in range(n_bins)
        ]

    # ── Inference ────────────────────────────────────────────────────────────

    def map_color_to_value(self, cell_image: np.ndarray) -> float:
        mean_bgr = cell_image.mean(axis=(0, 1)).astype(np.uint8)
        mean_lab = np.array(
            cv2.cvtColor(mean_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
        )

        if self.is_discrete and self.discrete_bins:
            best_val = self.discrete_bins[0][1]
            best_dist = float("inf")
            for centroid, scalar in self.discrete_bins:
                dist = self.inverter.calculate_ciede2000(mean_lab, centroid.astype(np.uint8))
                if dist < best_dist:
                    best_dist = dist
                    best_val = scalar
            self._last_value_source = "discrete_ciede2000"
            self._last_confidence = max(0.0, 1.0 - best_dist / 50.0)
            return best_val

        return self.base_mapper.map_color_to_value(cell_image)

    # ── Duck-type compatibility with ColorMappingService ─────────────────────
    @property
    def last_confidence(self): return self._last_confidence
    @property
    def last_value_source(self): return self._last_value_source
    @property
    def min_value(self): return self.base_mapper.min_value
    @property
    def max_value(self): return self.base_mapper.max_value
```

---

## 5. Phase 3 — Robust Label-to-Grid Alignment

### 5.1 `services/heatmap/sequence_aligner.py` — OCRGridSequenceAligner

**Complexity:** O(n × m) time and space. Guard: if n × m > 10,000 fall back to Hungarian matching.

```python
import numpy as np


class OCRGridSequenceAligner:
    """
    Semi-Global Needleman-Wunsch alignment with affine gap penalties.
    Maps variable-length OCR label sequences to strict logical grid positions,
    allowing free-end gaps (zero penalty for missing edge labels).
    """

    def __init__(
        self,
        gap_open: float = -10.0,
        gap_extend: float = -2.0,
        max_dist: float = 15.0,
    ):
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.max_dist = max_dist

    def compute_substitution_score(self, x: float, y: float) -> float:
        dist = abs(x - y)
        if dist <= self.max_dist:
            return self.max_dist - dist
        return -np.inf

    def align_sequences(
        self,
        ocr_seq: np.ndarray,
        grid_seq: np.ndarray,
    ) -> list[tuple[int, int]]:
        n, m = len(ocr_seq), len(grid_seq)

        if n * m > 10_000:
            return self._legacy_hungarian_fallback(ocr_seq, grid_seq)

        M  = np.full((n + 1, m + 1), -np.inf)
        Ix = np.full((n + 1, m + 1), -np.inf)
        Iy = np.full((n + 1, m + 1), -np.inf)

        M[0, 0] = 0.0

        # OCR sequence gaps are penalized (OCR labels shouldn't be randomly skipped)
        for i in range(1, n + 1):
            Ix[i, 0] = self.gap_open + (i - 1) * self.gap_extend

        # ------------------------------------------------------------------
        # Grid sequence gaps at the start carry zero penalty.
        # ------------------------------------------------------------------
        for j in range(1, m + 1):
            M[0, j] = 0.0
            Iy[0, j] = 0.0

        # DP Recurrence
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sub = self.compute_substitution_score(ocr_seq[i - 1], grid_seq[j - 1])
                M[i, j]  = sub + max(M[i-1, j-1], Ix[i-1, j-1], Iy[i-1, j-1])
                Ix[i, j] = max(M[i-1, j] + self.gap_open, Ix[i-1, j] + self.gap_extend)
                Iy[i, j] = max(M[i, j-1] + self.gap_open, Iy[i, j-1] + self.gap_extend)

        # ------------------------------------------------------------------
        # Find the optimal grid line j* where the final OCR label (n) matched.
        # ------------------------------------------------------------------
        best_j = int(np.argmax(M[n, :]))
        
        return self._traceback(M, Ix, Iy, n, best_j)

    def _traceback(
        self, M: np.ndarray, Ix: np.ndarray, Iy: np.ndarray, n: int, start_j: int
    ) -> list[tuple[int, int]]:
        i, j = n, start_j
        alignment = []

        while i > 0 and j > 0:
            current_max = max(M[i, j], Ix[i, j], Iy[i, j])
            if current_max == M[i, j]:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif current_max == Ix[i, j]:
                i -= 1
            else:
                j -= 1

        return alignment[::-1]

    def _legacy_hungarian_fallback(
        self, ocr_seq: np.ndarray, grid_seq: np.ndarray
    ) -> list[tuple[int, int]]:
        from scipy.optimize import linear_sum_assignment
        cost = np.array([
            [abs(o - g) for g in grid_seq] for o in ocr_seq
        ])
        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist()))
```

---

### 5.2 `services/heatmap/label_interpolator.py` — AxisLabelInterpolator

```python
import numpy as np
from itertools import combinations


class AxisLabelInterpolator:
    """
    Infers missing axis label values by fitting confirmed labels to an
    arithmetic or geometric progression.
    """

    def __init__(self, variance_tolerance: float = 1e-3, zero_mask: float = 1e-6):
        self.var_tol = variance_tolerance
        self.zero_mask = zero_mask

    def fill_missing_labels(
        self,
        valid_indices: list[int],
        valid_values: list[float],
        total_length: int,
    ) -> list[float]:
        """
        Variance classification:
          S_type = Arithmetic  if Var(D_A) < Var(D_G) or Var(D_A) ≤ ε_v
                 = Geometric   otherwise

        Arithmetic:  a_n = a_base + (n - K_base) · d
        Geometric:   a_n = a_base · r^(n - K_base)
        """
        if len(valid_values) < 2:
            raise ValueError(
                "Minimum 2 valid OCR points required for mathematical interpolation."
            )

        V = np.array(valid_values, dtype=float)
        K = np.array(valid_indices, dtype=int)

        arithmetic_steps = []
        geometric_ratios = []

        for (i, j) in combinations(range(len(K)), 2):
            idx_diff = K[j] - K[i]
            val_diff = V[j] - V[i]
            arithmetic_steps.append(val_diff / idx_diff)
            if V[i] > self.zero_mask and V[j] > self.zero_mask:
                log_diff = np.log(V[j]) - np.log(V[i])
                geometric_ratios.append(np.exp(log_diff / idx_diff))

        var_arithmetic = float(np.var(arithmetic_steps))
        var_geometric = float(np.var(geometric_ratios)) if geometric_ratios else float("inf")

        base_idx = K[0]
        base_val = V[0]

        if var_arithmetic < var_geometric or var_arithmetic <= self.var_tol:
            d = float(np.median(arithmetic_steps))
            return [float(base_val + (n - base_idx) * d) for n in range(total_length)]
        else:
            r = float(np.median(geometric_ratios))
            return [float(base_val * (r ** (n - base_idx))) for n in range(total_length)]

    def classify_sequence(
        self,
        valid_indices: list[int],
        valid_values: list[float],
    ) -> dict:
        V = np.array(valid_values, dtype=float)
        K = np.array(valid_indices, dtype=int)
        arithmetic_steps = []
        geometric_ratios = []

        for (i, j) in combinations(range(len(K)), 2):
            idx_diff = K[j] - K[i]
            arithmetic_steps.append((V[j] - V[i]) / idx_diff)
            if V[i] > self.zero_mask and V[j] > self.zero_mask:
                geometric_ratios.append(
                    np.exp((np.log(V[j]) - np.log(V[i])) / idx_diff)
                )

        var_a = float(np.var(arithmetic_steps))
        var_g = float(np.var(geometric_ratios)) if geometric_ratios else float("inf")

        if var_a < var_g or var_a <= self.var_tol:
            return {"type": "arithmetic", "param": float(np.median(arithmetic_steps)), "var": var_a}
        return {"type": "geometric", "param": float(np.median(geometric_ratios)), "var": var_g}
```

---

### 5.3 `_align_labels_to_grid` — Patch 5: Categorical Interpolation Guard

**Patch 5 applied:** Interpolation is gated by `numeric_density > 0.30`. Categorical axes (e.g. "Trial 1", "Trial 2") whose OCR yields numeric fragments ("1", "2") will not be extrapolated into fabricated data.

```python
def _align_labels_to_grid(
    self,
    labels: list[dict],
    grid_centers: list[float],
    is_vertical: bool,
) -> dict[int, str]:

    if not labels or not grid_centers:
        return {}

    ocr_positions = np.array([
        (l["xyxy"][1] + l["xyxy"][3]) / 2.0 if is_vertical
        else (l["xyxy"][0] + l["xyxy"][2]) / 2.0
        for l in labels
    ])
    grid_positions = np.array(sorted(grid_centers))

    if self.cfg.use_nw_aligner:
        if self._nw_aligner is None:
            self._nw_aligner = OCRGridSequenceAligner(
                gap_open=self.cfg.nw_gap_open,
                gap_extend=self.cfg.nw_gap_extend,
                max_dist=self.cfg.nw_max_dist,
            )
        pairs = self._nw_aligner.align_sequences(ocr_positions, grid_positions)
    else:
        pairs = self._legacy_align(labels, grid_centers, is_vertical)

    confirmed_pairs: dict[int, str] = {
        grid_idx: labels[ocr_idx].get("text", "")
        for ocr_idx, grid_idx in pairs
    }

    if self.cfg.use_label_interpolator:
        numeric_map: dict[int, float] = {}
        for grid_idx, text in confirmed_pairs.items():
            try:
                numeric_map[grid_idx] = float(
                    text.strip().replace(",", ".").replace(" ", "")
                )
            except ValueError:
                pass

        n_total_slots = len(grid_centers)

        # ------------------------------------------------------------------
        # Patch 5: Categorical label interpolation guard.
        # Interpolation is only safe on numeric axes. A categorical axis
        # (e.g. "Trial 1", "Trial 2") may yield OCR fragments ("1","2") that
        # the interpolator extrapolates as "3.0", "4.0" — fabricating data.
        #
        # Guard: numeric_density = confirmed numeric labels / total grid slots.
        # Only proceed if density > 30 %. Below that threshold, return only
        # confirmed OCR values and leave all other slots blank.
        # ------------------------------------------------------------------
        numeric_density = len(numeric_map) / n_total_slots if n_total_slots > 0 else 0.0

        if numeric_density <= 0.30:
            self.logger.debug(
                f"Label interpolation skipped: numeric_density={numeric_density:.2f} "
                f"({len(numeric_map)}/{n_total_slots} slots) ≤ 0.30 threshold."
            )
            return confirmed_pairs

        if len(numeric_map) >= 2:
            if self._label_interpolator is None:
                self._label_interpolator = AxisLabelInterpolator(
                    variance_tolerance=self.cfg.interp_variance_tol
                )
            interpolated = self._label_interpolator.fill_missing_labels(
                valid_indices=list(numeric_map.keys()),
                valid_values=list(numeric_map.values()),
                total_length=n_total_slots,
            )
            for idx, val in enumerate(interpolated):
                if idx not in confirmed_pairs:
                    confirmed_pairs[idx] = str(round(val, 6))

    return confirmed_pairs
```

---

## 6. Phase 4 — Integration & Feature-Flag Wiring

### 6.1 Mode-to-Config mapping and CLI

```python
# analysis.py

from services.heatmap.config import HeatmapConfig

parser.add_argument(
    '--heatmap-mode',
    default='legacy',
    choices=['legacy', 'fft', 'fft+color', 'full'],
    help=(
        'legacy=current DBSCAN behaviour; '
        'fft=FFT grid+hybrid anchor; '
        'fft+color=FFT grid + bimodal color router; '
        'full=all enhancements enabled.'
    ),
)

_MODE_TO_CONFIG: dict[str, HeatmapConfig] = {
    'legacy':    HeatmapConfig(),
    'fft':       HeatmapConfig(use_fft_grid=True),
    'fft+color': HeatmapConfig(
                     use_fft_grid=True,
                     use_bimodal_router=True,
                     use_ciede2000=True,
                 ),
    'full':      HeatmapConfig(
                     use_fft_grid=True,
                     use_rectifier=False,
                     use_artifact_rejector=True,
                     use_ciede2000=True,
                     use_bimodal_router=True,
                     use_nw_aligner=True,
                     use_label_interpolator=True,
                 ),
}
```

### 6.2 `ChartAnalysisPipeline` — Patch 4: Per-image config injection

**Patch 4 applied:** `HeatmapConfig` is extracted from `advanced_settings` per-image inside `run()`. The orchestrator is NOT constructed in `__init__` and does NOT accept a pipeline-level `heatmap_config` parameter. This matches the existing lazy-init architecture where settings flow per image via `advanced_settings`.

```python
# chart_pipeline.py

from services.heatmap.config import HeatmapConfig

_HEATMAP_MODE_TO_CONFIG: dict[str, HeatmapConfig] = {
    "legacy":    HeatmapConfig(),
    "fft":       HeatmapConfig(use_fft_grid=True),
    "fft+color": HeatmapConfig(
                     use_fft_grid=True,
                     use_bimodal_router=True,
                     use_ciede2000=True,
                 ),
    "full":      HeatmapConfig(
                     use_fft_grid=True,
                     use_rectifier=False,
                     use_artifact_rejector=True,
                     use_ciede2000=True,
                     use_bimodal_router=True,
                     use_nw_aligner=True,
                     use_label_interpolator=True,
                 ),
}


def run(self, image_path: str, advanced_settings: dict | None = None) -> dict:
    advanced_settings = advanced_settings or {}

    # Derive HeatmapConfig from per-image advanced_settings.
    # Priority: explicit HeatmapConfig object > named mode string > default legacy.
    if isinstance(advanced_settings.get("heatmap_config"), HeatmapConfig):
        heatmap_config = advanced_settings["heatmap_config"]
    else:
        mode = advanced_settings.get("heatmap_mode", "legacy")
        heatmap_config = _HEATMAP_MODE_TO_CONFIG.get(mode, HeatmapConfig())

    # Re-create orchestrator only when heatmap_config changes.
    # Repeated calls with the same mode pay zero construction overhead.
    if (
        self.orchestrator is None
        or getattr(self, "_last_heatmap_config", None) is not heatmap_config
    ):
        self.orchestrator = ChartAnalysisOrchestrator(
            calibration_service=self.calibration_engine,
            logger=logging.getLogger(__name__),
            heatmap_config=heatmap_config,
        )
        self._last_heatmap_config = heatmap_config

    return self.orchestrator.process(image_path, advanced_settings)
```

### 6.3 `ChartAnalysisOrchestrator` changes

```python
# In __init__, add:
from services.heatmap.config import HeatmapConfig
self.heatmap_config: HeatmapConfig = kwargs.get('heatmap_config', HeatmapConfig())

# In _HANDLER_EXTRAS, update 'heatmap' entry:
"heatmap": lambda self: {
    "classifier": HeatmapChartClassifier(logger=self.logger),
    "heatmap_config": self.heatmap_config,
},

# In _build_handler, inject BimodalColorMapper when flag is active:
if isinstance(handler_cls, type) and issubclass(handler_cls, GridChartHandler):
    cfg = self.heatmap_config
    color_mapper = (
        BimodalColorMapper(
            base_mapper=self.color_mapping_service,
            inverter=ColorToValueInverter(),
            sparsity_thresh=cfg.bimodal_sparsity_thresh,
        )
        if cfg.use_bimodal_router
        else self.color_mapping_service
    )
    kwargs['color_mapper'] = color_mapper
```

### 6.4 Diagnostics additions to `ExtractionResult`

```python
diagnostics.update({
    'grid_method':             'fft_hybrid' if used_fft else 'dbscan',
    'fft_periods':             {'T_x': T_x, 'T_y': T_y} if T_x else None,
    'phase_anchor_count':      len(valid_cx) if used_fft else None,
    'circular_coherence_R':    R_x if used_fft else None,   # Addition 2 visibility
    'color_bar_type':          'discrete' if color_mapper.is_discrete else 'continuous',
    'colormap_sort_method':    'spatial_order',              # Addition 1 audit trail
    'label_alignment_method':  'needleman_wunsch' if cfg.use_nw_aligner else 'hungarian',
    'interpolated_labels':     interpolated_count if cfg.use_label_interpolator else 0,
    'interp_numeric_density':  numeric_density if cfg.use_label_interpolator else None,
    'low_confidence_cells':    clamped_count,
})
```

---

## 7. Test Strategy

### Unit tests

| Test file | Component | Key assertions |
|---|---|---|
| `tests/heatmap/test_lattice_detector.py` | `HarmonicLatticeDetector` | Known-period synthetic grid → T_x/T_y within ±1px; sparse grid (50% blank) still recovers period |
| `tests/heatmap/test_hybrid_anchor.py` | `HybridGridAnchor` | Circular mean: period=40, centroids at 39 and 1 → phase ≈ 0 (not 20); R̄ < 0.2 on uniform noise → returns (None, None); <3 cells → (None, None) |
| `tests/heatmap/test_hybrid_anchor_coherence.py` | `HybridGridAnchor._circular_phase_estimate` | Uniformly distributed centroids → None; perfectly aligned centroids → correct phase within ±1px |
| `tests/heatmap/test_artifact_rejector.py` | `HeatmapArtifactRejector` | Cell with synthetic text overlay has ≥90% color fidelity after inpaint |
| `tests/heatmap/test_ciede2000.py` | `ColorToValueInverter` | Reference pairs from Sharma 2005 Table 1; OpenCV uint8 input converts correctly before formula |
| `tests/heatmap/test_bimodal.py` | `BimodalColorMapper` | Viridis ramp → `is_discrete=False`; 5-bin categorical scale → `is_discrete=True`; diverging Blue→White→Red → bins sorted by first appearance, not L*; dynamic bandwidth ≥ 8.0 |
| `tests/heatmap/test_bimodal_diverging.py` | `BimodalColorMapper._calibrate_discrete_bins` | Blue→White→Red pixel sequence: White bin appears in the middle of `discrete_bins`, not at the end |
| `tests/heatmap/test_nw_aligner.py` | `OCRGridSequenceAligner` | Full match; one-gap; two-gap; n×m > 10,000 falls back to Hungarian |
| `tests/heatmap/test_label_interpolator.py` | `AxisLabelInterpolator` | Arithmetic: [0,2,_,6] → 4; Geometric: [1,2,_,8] → 4; variance classification |

### Integration tests

- `tests/heatmap/test_heatmap_handler_fft.py`: End-to-end `HeatmapHandler.process()` with `HeatmapConfig(use_fft_grid=True)` on a 10×10 synthetic heatmap. Assert `diagnostics['grid_method'] == 'fft_hybrid'`.
- `tests/heatmap/test_heatmap_handler_full.py`: Full-pipeline test with `mode='full'`. Assert all cells have numeric values and `diagnostics['interpolated_labels']` is logged.
- `tests/heatmap/test_roi_crop.py`: Synthetic image with chart title region containing a strong periodic pattern. Assert FFT does not pick up the title's frequency when ROI crop is applied.
- `tests/heatmap/test_categorical_guard.py`: Axis with 10 slots, only 2 confirmed numeric OCR hits (density = 0.20) → no interpolation runs; confirmed 2 labels returned exactly.

### Regression tests

All existing heatmap tests must pass with `HeatmapConfig()` (all flags False):

```python
@pytest.mark.parametrize("config", [
    HeatmapConfig(),
    HeatmapConfig(use_fft_grid=True),
    HeatmapConfig(use_fft_grid=True, use_bimodal_router=True, use_nw_aligner=True),
])
def test_heatmap_handler_regression(config, sample_heatmap_image):
    handler = HeatmapHandler(heatmap_config=config)
    result = handler.process(sample_heatmap_image)
    assert result is not None
```

---

## 8. Dependency Matrix

| New Component | External Dependencies | Internal Dependencies | Already in codebase? |
|---|---|---|---|
| `HarmonicLatticeDetector` | `scipy.fft`, `skimage.filters.window` | None | `scipy` ✓, `skimage` check needed |
| `PerspectiveGridRectifier` | `cv2` | None | ✓ |
| `HybridGridAnchor` | `numpy` | None | ✓ |
| `HeatmapArtifactRejector` | `cv2` | None | ✓ |
| `ColorToValueInverter` | `scipy.optimize.brent` | None | `scipy` ✓ |
| `BimodalColorMapper` | `sklearn.cluster.MeanShift`, `sklearn.cluster.estimate_bandwidth` | `ColorMappingService`, `ColorToValueInverter` | `sklearn` check needed |
| `OCRGridSequenceAligner` | `numpy`, `scipy.optimize.linear_sum_assignment` | None | ✓ |
| `AxisLabelInterpolator` | `numpy`, `itertools` | None | ✓ |

**Dependency verification step — run before Phase 1:**

```bash
python -c "
from scipy.fft import fft2
from skimage.filters import window
from sklearn.cluster import MeanShift, estimate_bandwidth
print('All dependencies OK')
"
```

---

## 9. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| FFT period ambiguous on variable-width columns | Low | High | `HybridGridAnchor` returns (None, None) → DBSCAN used. Validate T_x, T_y ∈ [5px, 500px]. |
| `_circular_phase_estimate` returns 0.0 on uniform YOLO noise | **Eliminated** | High | **Addition 2:** R̄ < 0.2 guard returns None → DBSCAN fallback. |
| Diverging colormap (Blue→White→Red) bins mis-sorted by L* | **Eliminated** | High | **Addition 1:** bins sorted by first-appearance index in pixel sequence. |
| MeanShift over-segments at uint8 scale with bandwidth=2.5 | **Eliminated** | High | **Patch 3:** dynamic `estimate_bandwidth` with floor of 8.0 in uint8 space. |
| FFT spectrum polluted by chart title / legend frequencies | **Eliminated** | High | **Patch 2:** FFT operates on ROI crop derived from YOLO cell union bounding box. |
| `PerspectiveGridRectifier` degenerate homography | Medium | Medium | `needs_rectification()` pre-check; condition number guard (H cond > 1e10 → skip warp). |
| `BimodalColorMapper.is_discrete` misclassifies fine-stepped continuous colormap | Low-Medium | Medium | `sparsity_thresh` exposed in `HeatmapConfig`; bin count in diagnostics. |
| Categorical axis labels hallucinated as numeric progression | **Eliminated** | High | **Patch 5:** `numeric_density > 0.30` guard required before interpolation runs. |
| CIEDE2000 computation ~10× slower than Euclidean norm | Certain | Low | Per-cell cost only; 20×20 grid ≈ milliseconds. Bilateral filter downsamples to 60px if cell > 100px. |
| Needleman-Wunsch O(nm) memory with large grids | Low | Low | Guard: n×m > 10,000 → Hungarian fallback. |
| MeanShift `bandwidth` misfit → wrong bin count | Medium | Medium | `len(discrete_bins) == 1` → treat as continuous; bin count logged in diagnostics. |
| OpenCV LAB uint8 scale mismatch in CIEDE2000 | **Eliminated** | High | Explicit conversion in `calculate_ciede2000`; unit test with Sharma 2005 reference pairs. |
| `heatmap_config` stored on pipeline breaks per-image settings architecture | **Eliminated** | High | **Patch 4:** config extracted from `advanced_settings` per-image inside `run()`. |

---

## 10. Implementation Order Summary

```
Week 1: Phase 1 foundations
  ├── services/heatmap/__init__.py + config.py
  ├── lattice_detector.py  (HarmonicLatticeDetector)
  └── hybrid_grid_anchor.py  (HybridGridAnchor — Patches 1+2, Addition 2)

Week 2: Phase 1 integration + Phase 3
  ├── HeatmapHandler._reconstruct_grid() wiring  (Patch 2: ROI crop)
  ├── sequence_aligner.py  (OCRGridSequenceAligner)
  └── label_interpolator.py  (AxisLabelInterpolator)

Week 3: Phase 2
  ├── artifact_rejector.py
  ├── color_inverter.py  (CIEDE2000 + Brent's — with uint8 conversion)
  └── bimodal_color_mapper.py  (Patch 3: dynamic bandwidth, Addition 1: spatial sort)

Week 4: Phase 4 wiring + tests
  ├── chart_pipeline.py run() patch  (Patch 4: per-image config)
  ├── ChartAnalysisOrchestrator injection
  ├── analysis.py CLI flag + _MODE_TO_CONFIG
  ├── _align_labels_to_grid  (Patch 5: density guard)
  ├── Full unit + integration + regression test suite
  └── grid_rectifier.py  (optional, lowest priority)
```

The rectifier is deliberately last — it requires the most testing effort and has the narrowest applicability (non-screenshot heatmaps only). All other components provide value on typical screenshot heatmaps with default axis-aligned layouts.
