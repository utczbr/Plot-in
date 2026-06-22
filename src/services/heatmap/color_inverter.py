"""
LUTColorInverter — Phase 2: 3D LUT + branchless trilinear interpolation.

Replaces per-pixel Brent's method with two-phase design:
  Phase A (calibration): CIEDE2000 precomputes LUT once — O(N³ × M)
  Phase B (runtime):     Vectorised trilinear interp — O(1) per pixel

OpenCV LAB uint8 scale note: L=L*·255/100, a=a*+128, b=b*+128.
calculate_ciede2000() converts internally before computing ΔE₀₀.
"""
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LUTColorInverter:
    """3D LUT color-to-value inverter with CIEDE2000 precomputation."""

    def __init__(self, lut_resolution: int = 33) -> None:
        self.lut_resolution = lut_resolution
        self.lut: Optional[np.ndarray] = None
        self.min_value: float = 0.0
        self.max_value: float = 1.0
        self._is_calibrated: bool = False

    # ── Phase A: Offline LUT precomputation ──────────────────────────────────

    def precompute_lut(
        self,
        calibration_curve: List[Dict],
        min_val: float,
        max_val: float,
    ) -> None:
        """
        Build (N, N, N) float32 LUT: for every RGB lattice node find the
        nearest calibration sample via CIEDE2000, then store its scalar value.
        Runs once per color-bar calibration.
        """
        N = self.lut_resolution
        self.min_value = min_val
        self.max_value = max_val

        if not calibration_curve:
            logger.warning("LUTColorInverter: empty calibration curve — LUT not built.")
            return

        # Pre-convert all calibration samples to OpenCV uint8 LAB
        curve_lab, curve_val = [], []
        for s in calibration_curve:
            val = float(s["val"])
            if "lab" in s:
                lab = np.asarray(s["lab"], dtype=np.float64)
            else:
                bgr = np.asarray(s["vec"], dtype=np.uint8).reshape(1, 1, 3)
                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
            curve_lab.append(lab)
            curve_val.append(val)

        curve_lab_arr = np.array(curve_lab)   # (M, 3)
        curve_val_arr = np.array(curve_val)   # (M,)

        grid = np.linspace(0, 255, N, dtype=np.float32)
        lut  = np.zeros((N, N, N), dtype=np.float32)

        logger.info("LUTColorInverter: precomputing %d³=%d nodes …", N, N ** 3)

        for ri in range(N):
            for gi in range(N):
                # Process whole B-column in one batch LAB conversion
                b_col = grid.reshape(N, 1)
                bgr_col = np.concatenate([
                    b_col,
                    np.full((N, 1), grid[gi], dtype=np.float32),
                    np.full((N, 1), grid[ri], dtype=np.float32),
                ], axis=1).astype(np.uint8).reshape(N, 1, 3)

                lab_col = cv2.cvtColor(bgr_col, cv2.COLOR_BGR2LAB).reshape(N, 3).astype(np.float64)

                for bi in range(N):
                    dists = np.array([
                        self.calculate_ciede2000(lab_col[bi], curve_lab_arr[m])
                        for m in range(len(curve_val_arr))
                    ])
                    lut[ri, gi, bi] = float(curve_val_arr[np.argmin(dists)])

        self.lut = lut
        self._is_calibrated = True
        logger.info("LUTColorInverter: LUT precomputation complete.")

    # ── Phase B: Runtime O(1) inference ──────────────────────────────────────

    def map_color_to_value(self, cell_image: np.ndarray) -> float:
        """Map mean BGR of a cell crop to a scalar via trilinear LUT interp."""
        if cell_image.size == 0:
            return float(self.min_value)

        if not self._is_calibrated or self.lut is None:
            # Uncalibrated fallback: Hue if saturated, else LAB lightness
            try:
                hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
                avg_s = np.mean(hsv[:, :, 1])
                if avg_s > 30:
                    avg_h = np.mean(hsv[:, :, 0])
                    if avg_h <= 120:
                        norm = 1.0 - (avg_h / 120.0)
                    else:
                        norm = 1.0 - ((180 - avg_h) / 120.0)
                    norm = max(0.0, min(1.0, norm))
                    return float(self.min_value + norm * (self.max_value - self.min_value))
                
                # Grayscale fallback
                lab = cv2.cvtColor(cell_image, cv2.COLOR_BGR2LAB)
                L = float(np.mean(lab[:, :, 0])) / 255.0
                return float(self.min_value + L * (self.max_value - self.min_value))
            except Exception:
                return float(self.min_value)

        mean_bgr = np.mean(cell_image, axis=(0, 1)).astype(np.float32).reshape(1, 1, 3)
        val = float(self._apply_3d_lut_vectorized(mean_bgr, self.lut)[0, 0])
        return float(np.clip(val, self.min_value, self.max_value))

    def map_image_batch(self, image_bgr: np.ndarray) -> np.ndarray:
        """Map every pixel in a (H, W, 3) BGR image → (H, W) float32 values."""
        if not self._is_calibrated or self.lut is None:
            raise RuntimeError("LUT not precomputed — call precompute_lut() first.")
        return self._apply_3d_lut_vectorized(image_bgr.astype(np.float32), self.lut)

    # ── Branchless trilinear interpolation ───────────────────────────────────

    @staticmethod
    def _apply_3d_lut_vectorized(image_bgr: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """
        Branchless vectorised trilinear interpolation over the full image tensor.

        Parameters: image_bgr (..., 3) float32 [0,255]; lut (N, N, N) float32.
        Returns:    (...) float32 scalar values.

        No Python branches in the hot path — np.clip() is the only guard
        and executes element-wise in C, keeping SIMD pipelines full.
        """
        max_idx = lut.shape[0] - 1
        coords  = image_bgr.astype(np.float32) * (max_idx / 255.0)

        idx0 = coords.astype(np.int32)
        idx1 = np.clip(idx0 + 1, 0, max_idx)   # branchless upper-bound clamp
        w1   = (coords - idx0).astype(np.float32)
        w0   = 1.0 - w1

        # Channel split: image is BGR so axis-2 = [B, G, R]
        b0, g0, r0 = idx0[..., 0], idx0[..., 1], idx0[..., 2]
        b1, g1, r1 = idx1[..., 0], idx1[..., 1], idx1[..., 2]

        # Gather 8 bounding LUT vertices (lut indexed [r, g, b])
        V000 = lut[r0, g0, b0];  V001 = lut[r0, g0, b1]
        V010 = lut[r0, g1, b0];  V011 = lut[r0, g1, b1]
        V100 = lut[r1, g0, b0];  V101 = lut[r1, g0, b1]
        V110 = lut[r1, g1, b0];  V111 = lut[r1, g1, b1]

        wb0, wb1 = w0[..., 0], w1[..., 0]
        wg0, wg1 = w0[..., 1], w1[..., 1]
        wr0, wr1 = w0[..., 2], w1[..., 2]

        # Trilinear blend: B → G → R
        c00 = V000 * wb0 + V001 * wb1
        c01 = V010 * wb0 + V011 * wb1
        c10 = V100 * wb0 + V101 * wb1
        c11 = V110 * wb0 + V111 * wb1
        c0  = c00  * wg0 + c01  * wg1
        c1  = c10  * wg0 + c11  * wg1
        return c0 * wr0 + c1 * wr1

    # ── CIEDE2000 ΔE₀₀ (Sharma et al. 2005) ─────────────────────────────────

    @staticmethod
    def calculate_ciede2000(
        lab1: np.ndarray, lab2: np.ndarray,
        kL: float = 1.0, kC: float = 1.0, kH: float = 1.0,
    ) -> float:
        """Full CIEDE2000. Input: OpenCV uint8 LAB (L:0-255, a:0-255, b:0-255)."""
        def _cv(lab):
            return float(lab[0]) * 100.0 / 255.0, float(lab[1]) - 128.0, float(lab[2]) - 128.0

        L1, a1, b1 = _cv(lab1)
        L2, a2, b2 = _cv(lab2)

        C1s = np.hypot(a1, b1);  C2s = np.hypot(a2, b2)
        Cb  = (C1s + C2s) / 2.0
        G   = 0.5 * (1.0 - np.sqrt((Cb ** 7) / (Cb ** 7 + 25.0 ** 7)))

        a1p = a1 * (1.0 + G);  a2p = a2 * (1.0 + G)
        C1p = np.hypot(a1p, b1);  C2p = np.hypot(a2p, b2)
        Cbp = (C1p + C2p) / 2.0

        h1p = float(np.degrees(np.arctan2(b1, a1p)) % 360.0)
        h2p = float(np.degrees(np.arctan2(b2, a2p)) % 360.0)

        dLp = L2 - L1
        dCp = C2p - C1p

        if C1p * C2p == 0.0:
            dhp = 0.0
        else:
            d = h2p - h1p
            dhp = d - 360.0 if d > 180.0 else (d + 360.0 if d < -180.0 else d)

        dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

        Lbp = (L1 + L2) / 2.0
        if C1p * C2p == 0.0:
            Hbp = h1p + h2p
        else:
            s = h1p + h2p;  d = abs(h1p - h2p)
            Hbp = s / 2.0 if d <= 180.0 else ((s + 360.0) / 2.0 if s < 360.0 else (s - 360.0) / 2.0)

        T = (1.0
             - 0.17 * np.cos(np.radians(Hbp - 30.0))
             + 0.24 * np.cos(np.radians(2.0 * Hbp))
             + 0.32 * np.cos(np.radians(3.0 * Hbp + 6.0))
             - 0.20 * np.cos(np.radians(4.0 * Hbp - 63.0)))

        dTheta = 30.0 * np.exp(-((Hbp - 275.0) / 25.0) ** 2)
        Rc  = 2.0 * np.sqrt((Cbp ** 7) / (Cbp ** 7 + 25.0 ** 7))
        RT  = -np.sin(np.radians(2.0 * dTheta)) * Rc

        SL = 1.0 + (0.015 * (Lbp - 50.0) ** 2) / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
        SC = 1.0 + 0.045 * Cbp
        SH = 1.0 + 0.015 * Cbp * T

        tL = (dLp / (kL * SL)) ** 2
        tC = (dCp / (kC * SC)) ** 2
        tH = (dHp / (kH * SH)) ** 2
        tR = RT * (dCp / (kC * SC)) * (dHp / (kH * SH))

        return float(np.sqrt(max(0.0, tL + tC + tH + tR)))
