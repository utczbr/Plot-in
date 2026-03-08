"""
Color mapping service for heatmaps.

Maps pixel colors to numeric values based on a defined color scale.

§4.2: Supports CIELAB B-spline colormap inversion (feature flag: heatmap_color_mode).
"""
import numpy as np
import cv2
import logging
from typing import Tuple, List, Optional


class ColorMappingService:
    """
    Service to map colors to numeric values for heatmaps.
    
    Supports various color mapping strategies including:
    - Predefined color scales (viridis, plasma, etc.)
    - Custom color ranges
    - HSV-based mapping
    """
    
    def __init__(self, color_scale: Optional[str] = None, 
                 min_value: float = 0.0, 
                 max_value: float = 1.0):
        """
        Initialize the color mapping service.
        
        Args:
            color_scale: Predefined color scale ('viridis', 'plasma', 'hot', etc.) or None
            min_value: Minimum value in the data range
            max_value: Maximum value in the data range
        """
        self.color_scale = color_scale
        self.min_value = min_value
        self.max_value = max_value
        self.value_range = max_value - min_value
        
    def calibrate_from_known_values(self, color_samples: List[Tuple[np.ndarray, float]],
                                    color_mode: str = 'legacy') -> None:
        """
        Calibrate using 3D RGB trajectory mapping and optionally CIELAB B-splines.

        §4.2: When color_mode='lab_spline', additionally fits cubic B-splines per
        CIELAB channel for perceptually-uniform colormap inversion.
        """
        if not color_samples or len(color_samples) < 2:
            return

        # 1. Extract mean BGR and CIELAB vectors for each sample
        points = []
        for color_sample, val in color_samples:
            if color_sample.size > 0:
                avg_bgr = np.mean(color_sample, axis=(0, 1))
                points.append({
                    'val': val,
                    'vec': avg_bgr.astype(float)  # [B, G, R]
                })

        if len(points) < 2:
            return

        # 2. Sort by value to define the trajectory order
        points.sort(key=lambda x: x['val'])

        # 3. Store calibration curve (legacy BGR path)
        self.calibration_curve = points
        self.is_calibrated = True

        self.min_value = points[0]['val']
        self.max_value = points[-1]['val']
        self.value_range = self.max_value - self.min_value

        # 4. §4.2: Fit CIELAB B-splines if requested
        self._lab_splines = None
        self._lab_samples = None

        if color_mode == 'lab_spline':
            self._fit_lab_splines(color_samples)

    def map_color_to_value(self, cell_image: np.ndarray) -> float:
        """
        Map color to value using orthogonal projection onto the calibrated 3D curve.

        Fallback hierarchy:
        1. Calibrated 3D RGB curve projection (most accurate)
        2. LAB lightness mapping (good for grayscale/intensity scales)
        3. HSV hue mapping (good for colorscale heatmaps like red→blue)
        4. HSV brightness (final fallback)

        §4.2.4: Stores last_confidence and last_value_source after each call.
        """
        if cell_image.size == 0:
            self.last_confidence = 0.0
            self.last_value_source = 'empty'
            return 0.0

        # §4.2: Tier 0 — CIELAB B-spline inversion (most perceptually accurate)
        if hasattr(self, '_lab_splines') and self._lab_splines is not None:
            try:
                value, confidence = self._invert_lab_spline(cell_image)
                # §4.2.6: Clamp to calibrated range
                value = max(self.min_value, min(self.max_value, value))
                self.last_confidence = confidence
                self.last_value_source = 'lab_spline'
                return value
            except Exception:
                pass  # Fall through to legacy tiers

        # Extract query vector
        query_vec = np.mean(cell_image, axis=(0, 1)).astype(float)  # [B, G, R]

        # Tier 1: Calibrated curve projection
        if hasattr(self, 'is_calibrated') and self.is_calibrated:
            try:
                value, confidence = self._project_onto_curve_with_confidence(query_vec)
                # §4.2.6: Clamp to calibrated range
                value = max(self.min_value, min(self.max_value, value))
                self.last_confidence = confidence
                self.last_value_source = 'rgb_curve'
                return value
            except Exception:
                pass  # Fall through to fallbacks

        # Tier 2: LAB lightness mapping (good for grayscale/intensity colorscales)
        try:
            lab = cv2.cvtColor(cell_image, cv2.COLOR_BGR2LAB)
            avg_l = np.mean(lab[:, :, 0])  # L channel (lightness)
            normalized = avg_l / 255.0
            self.last_confidence = 0.6
            self.last_value_source = 'lab_lightness'
            return self.min_value + normalized * self.value_range
        except Exception:
            pass

        # Tier 3: HSV hue mapping (good for rainbow/colorscale heatmaps)
        try:
            hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
            avg_h = np.mean(hsv[:, :, 0])  # H channel (hue, 0-179 in OpenCV)
            avg_s = np.mean(hsv[:, :, 1])  # S channel (saturation)

            # Only use hue if saturation is high enough (colored, not grayscale)
            if avg_s > 30:  # Threshold for "has color"
                if avg_h <= 120:
                    hue_normalized = 1.0 - (avg_h / 120.0)
                else:
                    hue_normalized = 1.0 - ((180 - avg_h) / 120.0)
                    hue_normalized = max(0.0, min(1.0, hue_normalized))

                self.last_confidence = 0.4
                self.last_value_source = 'hsv_hue'
                return self.min_value + hue_normalized * self.value_range
        except Exception:
            pass

        # Tier 4: HSV brightness (final fallback)
        try:
            hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
            avg_v = np.mean(hsv[:, :, 2])
            normalized = avg_v / 255.0
            self.last_confidence = 0.2
            self.last_value_source = 'hsv_brightness'
            return self.min_value + normalized * self.value_range
        except Exception:
            self.last_confidence = 0.0
            self.last_value_source = 'absolute_fallback'
            return self.min_value  # Absolute fallback

    def _project_onto_curve_with_confidence(self, query: np.ndarray) -> Tuple[float, float]:
        """
        Find closest point on piecewise linear curve, return (value, confidence).

        §4.2.4: Confidence based on distance from query to curve:
            conf = exp(-d_min² / (2 * σ²))
        where σ ≈ 15 in BGR space (≈5 CIELAB units scaled to BGR magnitude).
        """
        curve = self.calibration_curve
        best_dist_sq = float('inf')
        best_val = curve[0]['val']

        for i in range(len(curve) - 1):
            p1 = curve[i]
            p2 = curve[i + 1]

            v = p2['vec'] - p1['vec']
            w = query - p1['vec']

            v_len_sq = np.dot(v, v)
            if v_len_sq < 1e-6:
                t = 0.0
            else:
                t = np.dot(w, v) / v_len_sq
                t = max(0.0, min(1.0, t))

            closest = p1['vec'] + t * v
            d = closest - query
            dist_sq = np.dot(d, d)

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_val = p1['val'] + t * (p2['val'] - p1['val'])

        # §4.2.4: Distance-based confidence
        # σ_bgr ≈ 15 (roughly equivalent to σ_lab ≈ 5 in perceptual terms)
        sigma_bgr = 15.0
        confidence = float(np.exp(-best_dist_sq / (2.0 * sigma_bgr ** 2)))

        return float(best_val), confidence

    def _project_onto_curve(self, query: np.ndarray) -> float:
        """
        Find closest point on the piecewise linear curve and return its interpolated value.
        Legacy wrapper for backward compatibility.
        """
        value, _ = self._project_onto_curve_with_confidence(query)
        return value

    # ── §4.2: CIELAB B-Spline Colormap Inversion ────────────────────────

    def _fit_lab_splines(self, color_samples: List[Tuple[np.ndarray, float]]) -> None:
        """
        §4.2.2: Fit cubic B-splines per CIELAB channel to the colorbar samples.

        Staff Refinement — Schoenberg-Whitney: Dynamically set M based on unique samples.
        Falls back to linear interpolation if too few unique samples.
        """
        try:
            from scipy.interpolate import make_lsq_spline
        except ImportError:
            logging.warning("scipy.interpolate not available; skipping lab_spline calibration")
            return

        # Collect (s, L, a, b) tuples where s ∈ [0, 1] is normalized position
        entries = []
        for color_sample, val in color_samples:
            if color_sample.size > 0:
                avg_bgr = np.mean(color_sample, axis=(0, 1)).astype(np.uint8)
                # Convert single pixel BGR → LAB
                pixel = avg_bgr.reshape(1, 1, 3)
                lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB).reshape(3).astype(float)
                entries.append((val, lab))

        if len(entries) < 4:
            return

        entries.sort(key=lambda e: e[0])
        vals = np.array([e[0] for e in entries])
        labs = np.array([e[1] for e in entries])  # (N, 3)

        # Normalize values to s ∈ [0, 1]
        v_min, v_max = vals[0], vals[-1]
        if v_max - v_min < 1e-10:
            return
        s = (vals - v_min) / (v_max - v_min)

        # §4.2.2 Staff Refinement: Deduplicate and determine M
        # Count unique LAB vectors (ΔE < 1.0)
        unique_mask = np.ones(len(labs), dtype=bool)
        for i in range(1, len(labs)):
            if np.linalg.norm(labs[i] - labs[i - 1]) < 1.0:
                unique_mask[i] = False
        n_unique = int(np.sum(unique_mask))

        if n_unique < 4:
            # Fall back to simple linear interpolation stored as _lab_linear
            self._lab_linear = (s, labs, v_min, v_max)
            self._lab_splines = None
            logging.info(f"Lab spline: only {n_unique} unique samples; using linear interpolation fallback")
            return

        M = max(4, min(12, n_unique // 3))

        # Build clamped knot vector
        # Interior knots: M - 4 uniformly spaced in (0, 1) for cubic (k=3) B-spline
        n_interior = M - 4  # make_lsq_spline needs t = [0,0,0,0, interior..., 1,1,1,1]
        if n_interior <= 0:
            interior = np.array([])
        else:
            interior = np.linspace(0, 1, n_interior + 2)[1:-1]
        knots = np.concatenate([
            np.zeros(4),
            interior,
            np.ones(4)
        ])

        # Fit one spline per channel
        splines = []
        try:
            for ch in range(3):
                spl = make_lsq_spline(s, labs[:, ch], knots, k=3)
                splines.append(spl)
        except Exception as e:
            logging.warning(f"Lab spline fitting failed: {e}")
            # Store linear fallback
            self._lab_linear = (s, labs, v_min, v_max)
            return

        self._lab_splines = splines
        self._lab_samples = (s, labs)
        self._lab_val_range = (v_min, v_max)
        logging.info(f"Lab spline calibration: M={M}, N={len(entries)}, unique={n_unique}")

    def _invert_lab_spline(self, cell_image: np.ndarray) -> Tuple[float, float]:
        """
        §4.2.3: Invert CIELAB B-spline to map cell color → scalar value via Brent's method.

        Returns (value, confidence) where confidence = exp(-d²/(2σ²)).
        """
        from scipy.optimize import minimize_scalar

        # Convert cell average to CIELAB
        avg_bgr = np.mean(cell_image, axis=(0, 1)).astype(np.uint8)
        pixel = avg_bgr.reshape(1, 1, 3)
        lab_obs = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB).reshape(3).astype(float)

        # Check for linear fallback
        if self._lab_splines is None and hasattr(self, '_lab_linear') and self._lab_linear is not None:
            return self._invert_lab_linear(lab_obs)

        splines = self._lab_splines
        v_min, v_max = self._lab_val_range

        # Define distance function D(s) = ||f(s) - y_obs||₂
        def distance_sq(s_val):
            f_lab = np.array([spl(s_val) for spl in splines])
            return float(np.sum((f_lab - lab_obs) ** 2))

        # §4.2.3: Initialize by finding nearest colorbar sample
        s_samples, lab_samples = self._lab_samples
        dists = np.linalg.norm(lab_samples - lab_obs, axis=1)
        k_star = np.argmin(dists)
        s_init = s_samples[k_star]

        # Bracket around the nearest sample
        h = 1.0 / max(len(s_samples) - 1, 1)
        a = max(0.0, s_init - h)
        b = min(1.0, s_init + h)

        # Brent's method bounded search
        result = minimize_scalar(distance_sq, bounds=(a, b), method='bounded',
                                 options={'xatol': 1e-3})
        s_star = result.x

        # Also try the full [0, 1] range in case the bracket missed
        result_full = minimize_scalar(distance_sq, bounds=(0.0, 1.0), method='bounded',
                                      options={'xatol': 1e-3})
        if distance_sq(result_full.x) < distance_sq(s_star):
            s_star = result_full.x

        # §4.2.6: Clamp to [0, 1]
        s_star = max(0.0, min(1.0, s_star))

        # Map s* → value
        value = v_min + s_star * (v_max - v_min)

        # §4.2.4: Distance-based confidence
        d_min = np.sqrt(distance_sq(s_star))
        sigma_lab = 5.0
        confidence = float(np.exp(-d_min ** 2 / (2.0 * sigma_lab ** 2)))

        return value, confidence

    def _invert_lab_linear(self, lab_obs: np.ndarray) -> Tuple[float, float]:
        """Linear interpolation fallback when B-spline fitting failed."""
        s_arr, lab_arr, v_min, v_max = self._lab_linear
        dists = np.linalg.norm(lab_arr - lab_obs, axis=1)
        k = np.argmin(dists)
        d_min = dists[k]

        # Linear interpolation between nearest neighbors
        s_star = s_arr[k]
        value = v_min + s_star * (v_max - v_min)

        sigma_lab = 5.0
        confidence = float(np.exp(-d_min ** 2 / (2.0 * sigma_lab ** 2)))
        return value, confidence