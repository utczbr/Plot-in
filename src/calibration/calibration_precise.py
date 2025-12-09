"""
PROSAC calibration with all critical bugs fixed: proper lstsq unpacking, convergence checks, local optimization, and enhanced diagnostics.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calibration_base import BaseCalibration, CalibrationResult

logger = logging.getLogger(__name__)


class PROSACCalibration(BaseCalibration):
    """
    PROSAC-based robust calibration with MSAC scoring and local optimization.
    
    Features:
    - PROSAC sampling: Prioritizes high-confidence labels
    - MSAC scoring: Soft penalty for outliers beyond threshold
    - Local optimization: LO-RANSAC style refinement
    - Convergence detection: Early stopping based on score history
    - Adaptive threshold: Weighted MAD for robust scale estimation
    """
    
    __slots__ = (
        'max_trials', 'residual_threshold', 'min_inliers', 'random_state',
        'lo_iters', 'prosac_growth', 'early_termination_ratio', 'convergence_threshold'
    )
    
    def __init__(
        self,
        max_trials: int = 800,
        residual_threshold: float = 3.0,
        min_inliers: int = 3,
        random_state: Optional[int] = None,
        lo_iters: int = 2,
        prosac_growth: int = 25,
        early_termination_ratio: float = 0.9,
        convergence_threshold: float = 1e-6,
    ):
        """
        Initialize PROSAC calibration.
        
        Args:
            max_trials: Maximum iterations
            residual_threshold: Base inlier threshold (pixels)
            min_inliers: Minimum inliers for valid model
            random_state: RNG seed
            lo_iters: Local optimization iterations per trial
            prosac_growth: Grow pool every N iterations
            early_termination_ratio: Stop when this fraction are inliers
            convergence_threshold: Stop when score improvement < this
        """
        if max_trials <= 0:
            raise ValueError(f"max_trials must be positive, got {max_trials}")
        if residual_threshold <= 0:
            raise ValueError(f"residual_threshold must be positive, got {residual_threshold}")
        if min_inliers < 2:
            raise ValueError(f"min_inliers must be at least 2, got {min_inliers}")
        if lo_iters < 0:
            raise ValueError(f"lo_iters must be non-negative, got {lo_iters}")
        if prosac_growth <= 0:
            raise ValueError(f"prosac_growth must be positive, got {prosac_growth}")
        if not (0.0 < early_termination_ratio <= 1.0):
            raise ValueError(f"early_termination_ratio must be in (0, 1], got {early_termination_ratio}")
        
        self.max_trials = int(max_trials)
        self.residual_threshold = float(residual_threshold)
        self.min_inliers = int(min_inliers)
        self.random_state = random_state
        self.lo_iters = int(lo_iters)
        self.prosac_growth = int(prosac_growth)
        self.early_termination_ratio = float(early_termination_ratio)
        self.convergence_threshold = float(convergence_threshold)
    
    def calibrate(self, scale_labels: List[Dict], axis_type: str) -> Optional[CalibrationResult]:
        """Perform PROSAC calibration with adaptive threshold and detailed debugging."""
        import time
        t_start = time.perf_counter()
        
        logger.info(f"📊 PROSAC CALIBRATION ATTEMPT (axis={axis_type}):")
        logger.info(f"   ├─ Input labels: {len(scale_labels)}")
        logger.info(f"   ├─ Min inliers required: {self.min_inliers}")
        logger.info(f"   └─ Max trials: {self.max_trials}")

        # Use adaptive threshold extraction
        t_extract_start = time.perf_counter()
        try:
            coords, values, weights = self.extract_points_with_adaptive_threshold(
                scale_labels, axis_type,
                initial_confidence_threshold=0.8,
                min_points_required=self.min_inliers
            )
        except Exception as e:
            logger.error(f"Error extracting points in PROSACCalibration: {e}")
            return None
        t_extract_end = time.perf_counter()
        logger.info(f"⏱️ Point extraction took: {(t_extract_end - t_extract_start)*1000:.1f}ms")
        
        # Filter NaN/Inf
        valid_mask = np.isfinite(coords) & np.isfinite(values) & np.isfinite(weights)
        if not np.all(valid_mask):
            logger.warning(f"Filtered {(~valid_mask).sum()} invalid samples")
            coords = coords[valid_mask]
            values = values[valid_mask]
            weights = weights[valid_mask]

        n = coords.size

        # Debug: Log extracted points details
        if n > 0:
            logger.debug(f"   ├─ Confidence distribution:")
            logger.debug(f"   │  ├─ Min: {np.min(weights):.3f}")
            logger.debug(f"   │  ├─ Max: {np.max(weights):.3f}")
            logger.debug(f"   │  ├─ Mean: {np.mean(weights):.3f}")
            logger.debug(f"   │  └─ Std: {np.std(weights):.3f}")

            logger.debug(f"   ├─ Coordinate range: [{np.min(coords):.1f}, {np.max(coords):.1f}]px")
            logger.debug(f"   └─ Value range: [{np.min(values):.2f}, {np.max(values):.2f}]")

        if n < 2:
            logger.warning(f"Not enough points for PROSACCalibration: {n} < 2")
            return None
        
        # FIXED: Adaptive minimum inliers for small datasets
        adaptive_min_inliers = max(2, min(self.min_inliers, n // 2))
        
        if n < adaptive_min_inliers:
            logger.warning(
                f"Not enough points for PROSACCalibration: {n} < {adaptive_min_inliers} "
                f"(original min_inliers={self.min_inliers})"
            )
            return None
        
        # Sort by confidence descending (PROSAC)
        order = np.argsort(-weights)
        x_all = coords[order]
        y_all = values[order]
        w_all = np.clip(weights[order], 1e-3, 1.0)
        
        # Adaptive threshold using BaseCalibration unified method
        t_thresh_start = time.perf_counter()
        thr = self._compute_mad_threshold(x_all, y_all, self.residual_threshold, w_all)
        t_thresh_end = time.perf_counter()
        logger.info(f"⏱️ Adaptive threshold took: {(t_thresh_end - t_thresh_start)*1000:.1f}ms")
        
        # FIXED: More lenient early termination for small datasets
        if n <= 4:
            adaptive_ratio = 0.5  # Allow 50% inliers for tiny datasets
        elif n <= 10:
            adaptive_ratio = max(0.6, self.early_termination_ratio)
        else:
            adaptive_ratio = max(0.7, 1.0 - 2.0 / max(n, 2), self.early_termination_ratio)
        
        logger.info(
            f"PROSAC parameters: n={n}, adaptive_min_inliers={adaptive_min_inliers}, "
            f"adaptive_ratio={adaptive_ratio:.2f}, threshold={thr:.2f}px"
        )
        
        best_score = -np.inf
        best_m, best_b = 0.0, 0.0
        best_inliers: Optional[np.ndarray] = None
        
        pool_k = max(2, min(10, n))
        rng = np.random.default_rng(self.random_state)
        
        score_history: List[float] = []
        
        # Timing accumulators for main loop
        t_loop_start = time.perf_counter()
        t_lo_total = 0.0
        t_msac_total = 0.0
        lo_count = 0
        
        for it in range(self.max_trials):
            # FIXED: Convergence check (only if we have a valid model)
            if best_inliers is not None and len(score_history) >= 10:
                recent = score_history[-5:]
                if max(recent) - min(recent) < self.convergence_threshold:
                    logger.info(f"Convergence achieved at iteration {it}, stopping early")
                    break
            
            # Progress logging every 100 iterations
            if it > 0 and it % 100 == 0:
                t_now = time.perf_counter()
                elapsed = t_now - t_loop_start
                logger.info(f"⏱️ PROSAC progress: iter {it}/{self.max_trials}, elapsed={elapsed:.2f}s, best_inliers={best_inliers.sum() if best_inliers is not None else 0}")
            
            # Grow PROSAC pool
            if it > 0 and it % self.prosac_growth == 0:
                pool_k = min(n, pool_k + 1)
            
            if pool_k < 2:
                continue
            
            # Sample 2 points from high-confidence pool
            idx = rng.choice(pool_k, size=2, replace=False)
            x_s = x_all[idx]
            y_s = y_all[idx]
            
            denom = x_s[1] - x_s[0]
            if abs(denom) < 1e-12:
                continue
            
            m = (y_s[1] - y_s[0]) / denom
            b = y_s[0] - m * x_s[0]
            
            # MSAC scoring on all points
            t_msac_start = time.perf_counter()
            score, inliers = self._msac_score(x_all, y_all, m, b, thr, w_all)
            t_msac_total += time.perf_counter() - t_msac_start
            n_inliers = inliers.sum()
            
            # Early termination
            if n_inliers >= n * adaptive_ratio:
                logger.info(f"Early termination at iteration {it}: {n_inliers}/{n} inliers exceed {adaptive_ratio:.2f} ratio")
                
                if n_inliers >= self.min_inliers:
                    try:
                        # Use weighted refitting for final model via unified helper
                        # Mask weights to inliers
                        x_in = x_all[inliers]
                        y_in = y_all[inliers]
                        w_in = w_all[inliers]
                        
                        m_refit, b_refit = self._refit_linear(x_in, y_in, w_in)
                        
                        # LOG SCALE CHECK
                        # If simple linear R2 is poor (< 0.95), try Log fit: y = m*log(x) + b or log(y) = mx+b
                        # Typical scientific plots use Log axis meaning log(Value) ~ Pixel.
                        # Since 'y' here is Value (from OCR) and 'x' is Pixel, we expect log(y) = m*x + b
                        
                        best_m, best_b = m_refit, b_refit
                        best_model_type = 'linear'
                        
                        r2_linear = self._r2(x_in, y_in, m_refit, b_refit)
                        
                        # Only try log if we have enough positive values
                        if r2_linear < 0.99 and np.all(y_in > 0):
                            try:
                                log_y_in = np.log(y_in)
                                m_log, b_log = self._refit_linear(x_in, log_y_in, w_in)
                                
                                # Calculate R2 for log model (in linear space for fair comparison)
                                y_pred_log = np.exp(m_log * x_in + b_log)
                                ss_res = np.sum((y_in - y_pred_log) ** 2)
                                ss_tot = np.sum((y_in - np.mean(y_in)) ** 2)
                                r2_log = 1.0 - (ss_res / ss_tot)
                                
                                if r2_log > r2_linear and r2_log > 0.9:
                                    logger.info(f"Logarithmic fit detected! R2_log={r2_log:.4f} > R2_lin={r2_linear:.4f}")
                                    best_m, best_b = m_log, b_log
                                    best_model_type = 'log'
                                    r2 = r2_log
                                    
                                    # We need to signal this model type. 
                                    # BaseCalibration currently stores (m,b).
                                    # We will store a special flag or modify make_func.
                            except Exception:
                                pass

                        if best_model_type == 'linear':
                             r2 = r2_linear

                        best_inliers = inliers
                        best_score = score
                        
                        # Package best model found
                        self._last_best_model_type = best_model_type # Store for final result construction
                        break
                    except Exception as e:
                        logger.debug(f"Early termination refit failed: {e}")
                        continue
            
            if n_inliers < self.min_inliers:
                continue
            
            # Local optimization
            t_lo_start = time.perf_counter()
            m_lo, b_lo, inliers_lo, score_lo = self._local_optimize(
                x_all, y_all, w_all, inliers, thr, iters=self.lo_iters
            )
            t_lo_total += time.perf_counter() - t_lo_start
            lo_count += 1
            
            if score_lo > best_score:
                best_score = score_lo
                best_m, best_b = m_lo, b_lo
                best_inliers = inliers_lo
            
            score_history.append(best_score)
        
        t_loop_end = time.perf_counter()
        logger.info(f"⏱️ Main loop timing: total={t_loop_end - t_loop_start:.2f}s, MSAC={t_msac_total*1000:.1f}ms, LO={t_lo_total*1000:.1f}ms ({lo_count} calls)")
        
        if best_inliers is None:
            logger.error(
                f"❌ PROSAC CALIBRATION FAILED (axis={axis_type}):\n"
                f"   ├─ Input points: {n}\n"
                f"   ├─ Adaptive min inliers: {adaptive_min_inliers}\n"
                f"   ├─ Iterations completed: {len(score_history)}/{self.max_trials}\n"
                f"   ├─ Best score: {best_score}\n"
                f"   ├─ Threshold: {thr:.2f}px\n"
                f"   └─ Adaptive ratio: {adaptive_ratio:.2f}\n"
                f"\n"
                f"   DIAGNOSIS:\n"
            )
            
            # Additional diagnostics
            if n < 3:
                logger.error("   → Too few labels detected (need ≥3 for robust calibration)")
            
            # Check if data is monotonic
            if n >= 2:
                coord_order = np.argsort(x_all)
                values_sorted = y_all[coord_order]
                is_monotonic = (np.all(np.diff(values_sorted) > 0) or 
                              np.all(np.diff(values_sorted) < 0))
                
                if not is_monotonic:
                    logger.error("   → Labels are NOT monotonic (values don't increase/decrease with position)")
            
            # Check spacing uniformity
            if n >= 3:
                coord_spacings = np.diff(np.sort(x_all))
                spacing_cv = np.std(coord_spacings) / np.mean(coord_spacings)
                
                if spacing_cv > 0.5:
                    logger.error(
                        f"   → High spacing variance (CV={spacing_cv:.2f}): "
                        "labels may be irregularly distributed"
                    )
            
            logger.error(
                f"\n"
                f"   FALLBACK RECOMMENDATION:\n"
                f"   → Try calibration_type='fast' (requires only 2 points)\n"
                f"   → Or calibration_type='adaptive' (RANSAC with relaxed thresholds)"
            )
            
            return None
        
        # Compute final R² on inliers
        x_in = x_all[best_inliers]
        y_in = y_all[best_inliers]
        r2 = self._r2(x_in, y_in, best_m, best_b)
        
        # Calculate zero-crossing point (where value = 0)
        zero_crossing = -best_b / best_m if abs(best_m) > 1e-6 else None
        
        t_end = time.perf_counter()
        logger.info(
            f"📊 CALIBRATION COMPLETE (axis={axis_type}):\n"
            f"  ├─ R² = {r2:.4f}\n"
            f"  ├─ Slope (m) = {best_m:.6f}\n"
            f"  ├─ Intercept (b) = {best_b:.4f}\n"
            f"  ├─ Zero-crossing (x₀=-b/m) = {zero_crossing:.2f}px\n"
            f"  ├─ Inliers = {best_inliers.sum()}/{n}\n"
            f"  └─ ⏱️ Total time: {(t_end - t_start)*1000:.1f}ms"
        )
        
        # Validate slope direction
        if axis_type == 'x':  # Horizontal charts
            if best_m < 0:
                logger.warning(
                    f"⚠️  INVERTED X-AXIS DETECTED: slope={best_m:.6f} < 0\n"
                    f"    → Values decrease left-to-right (right-origin chart)"
                )
            else:
                logger.info(f"✓ Normal X-axis: slope={best_m:.6f} > 0 (left-origin)")
        elif axis_type == 'y':  # Vertical charts
            if best_m > 0:
                logger.warning(
                    f"⚠️  INVERTED Y-AXIS DETECTED: slope={best_m:.6f} > 0\n"
                    f"    → Values decrease bottom-to-top (inverted vertical)"
                )
            else:
                logger.info(f"✓ Normal Y-axis: slope={best_m:.6f} < 0 (bottom-origin)")
        
        # Map inliers back to original order
        orig_inliers = self._unorder_mask(best_inliers, order, n)
        
        # Determine coordinate system based on axis type and slope
        is_inverted = (axis_type.lower() == 'y' and best_m > 0)  # Y-axis with positive slope means inverted
        coordinate_system = 'image'  # Default assumption: image coordinates where Y increases down
        
        model_type = getattr(self, '_last_best_model_type', 'linear')
        
        result = CalibrationResult(
            func=self._make_func_log(best_m, best_b) if model_type == 'log' else self._make_func(best_m, best_b),
            r2=float(r2),
            coeffs=(float(best_m), float(best_b)),
            inliers=orig_inliers,
            is_inverted=is_inverted,
            coordinate_system=coordinate_system,
        )
        if model_type == 'log':
             result.metadata = {'scale_type': 'log'}
             
        logger.info(f"PROSACCalibration.calibrate returning: {result}")
        return result
    
    @staticmethod
    def _make_func_log(m: float, b: float):
        """Create a callable for log model: y = exp(m*x + b)."""
        return lambda x: np.exp(m * x + b)
    
    @staticmethod
    def _unorder_mask(mask_sorted: np.ndarray, order: np.ndarray, n: int) -> np.ndarray:
        """Map boolean mask from sorted order back to original indices."""
        orig_mask = np.zeros(n, dtype=bool)
        orig_mask[order] = mask_sorted
        return orig_mask
    
    @staticmethod
    def _msac_score(
        x: np.ndarray, y: np.ndarray, m: float, b: float, thr: float, w: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        MSAC (M-estimator SAC) scoring: min(r², thr²) with confidence weighting.
        Returns negative cost as score (higher is better) plus inlier mask.
        """
        r = np.abs(y - (m * x + b))
        inliers = r <= thr
        
        # MSAC cost: clamp residuals at threshold
        cost = np.minimum(r * r, thr * thr)
        
        # Weight by confidence (high confidence = more penalty if outlier)
        weighted_cost = cost * np.maximum(1e-6, w)
        
        # Negative total cost + small bonus for inlier count
        score = -float(np.sum(weighted_cost)) + 0.5 * float(inliers.sum())
        
        return score, inliers
    
    def _local_optimize(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        inliers_0: np.ndarray,
        thr: float,
        iters: int = 2,
    ) -> Tuple[float, float, np.ndarray, float]:
        """
        Local optimization (LO-RANSAC style): refine model by iteratively
        refitting on inliers with tightening threshold.
        """
        inliers = inliers_0.copy()
        
        try:
            m, b = self._refit_linear(x[inliers], y[inliers], w[inliers])
        except ValueError:
            return 0.0, 0.0, inliers_0, -np.inf
        
        best_score, best_mask = self._msac_score(x, y, m, b, thr, w)
        
        for _ in range(max(1, iters)):
            # Tighten threshold slightly for refinement
            tight_thr = max(1e-6, 0.9 * thr)
            _, mask_tight = self._msac_score(x, y, m, b, tight_thr, w)
            
            if mask_tight.sum() < 2:
                continue
            
            try:
                m, b = self._refit_linear(x[mask_tight], y[mask_tight], w[mask_tight])
            except ValueError:
                continue
            
            # Re-score with original threshold
            score, mask = self._msac_score(x, y, m, b, thr, w)
            
            if score > best_score:
                best_score, best_mask = score, mask
        
        return m, b, best_mask, best_score