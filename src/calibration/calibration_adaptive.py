"""
RANSAC calibration with all critical bugs fixed: proper lstsq unpacking, adaptive early termination, NaN filtering, and enhanced logging.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calibration_base import BaseCalibration, CalibrationResult

logger = logging.getLogger(__name__)


class RANSACCalibration(BaseCalibration):
    """
    Standard RANSAC for robust line fitting: value = m * coord + b.
    
    Features:
    - Random minimal sampling (2 points)
    - Inlier classification by residual threshold
    - Adaptive threshold using MAD
    - Early termination with adaptive ratio
    - Refit on inliers for final model
    """
    
    __slots__ = ('max_trials', 'residual_threshold', 'min_inliers', 'random_state', 'early_termination_ratio')
    
    def __init__(
        self,
        max_trials: int = 500,
        residual_threshold: float = 3.0,
        min_inliers: int = 3,
        random_state: Optional[int] = None,
        early_termination_ratio: float = 0.9,
    ):
        """
        Initialize RANSAC calibration.
        
        Args:
            max_trials: Maximum RANSAC iterations
            residual_threshold: Base inlier threshold (pixels)
            min_inliers: Minimum inliers required for valid model
            random_state: RNG seed for reproducibility
            early_termination_ratio: Stop when this fraction of points are inliers (adaptive)
        """
        if max_trials <= 0:
            raise ValueError(f"max_trials must be positive, got {max_trials}")
        if residual_threshold <= 0:
            raise ValueError(f"residual_threshold must be positive, got {residual_threshold}")
        if min_inliers < 2:
            raise ValueError(f"min_inliers must be at least 2, got {min_inliers}")
        if not (0.0 < early_termination_ratio <= 1.0):
            raise ValueError(f"early_termination_ratio must be in (0, 1], got {early_termination_ratio}")
        
        self.max_trials = int(max_trials)
        self.residual_threshold = float(residual_threshold)
        self.min_inliers = int(min_inliers)
        self.random_state = random_state
        self.early_termination_ratio = float(early_termination_ratio)
    
    def calibrate(self, scale_labels: List[Dict], axis_type: str) -> Optional[CalibrationResult]:
        """Perform RANSAC calibration with adaptive threshold and detailed debugging."""
        logger.info(f"📊 RANSAC CALIBRATION ATTEMPT (axis={axis_type}):")
        logger.info(f"   ├─ Input labels: {len(scale_labels)}")
        logger.info(f"   └─ Min inliers required: {self.min_inliers}")

        # Use adaptive threshold extraction
        try:
            coords, values, weights = self.extract_points_with_adaptive_threshold(
                scale_labels, axis_type,
                initial_confidence_threshold=0.8,
                min_points_required=self.min_inliers
            )
        except Exception as e:
            logger.error(f"Error extracting points in RANSACCalibration: {e}")
            return None
        
        # Filter out NaN/Inf values
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
            logger.warning(f"Not enough points for RANSACCalibration: {n} < 2")
            return None

        if n < self.min_inliers:
            logger.warning(f"Not enough points for RANSACCalibration: {n} < {self.min_inliers}")
            return None
        
        # Adaptive threshold using BaseCalibration unified method
        thr = self._compute_mad_threshold(coords, values, self.residual_threshold)
        
        # Adaptive early termination ratio for small n
        adaptive_ratio = max(0.7, 1.0 - 1.0 / n, self.early_termination_ratio)
        
        best_inliers: Optional[np.ndarray] = None
        best_m, best_b, best_r2 = 0.0, 0.0, -np.inf
        
        rng = np.random.default_rng(self.random_state)
        
        for trial in range(self.max_trials):
            # Sample 2 distinct indices
            idx = rng.choice(n, size=2, replace=False)
            x_s = coords[idx]
            y_s = values[idx]
            
            # Fit line through 2 points
            denom = x_s[1] - x_s[0]
            if abs(denom) < 1e-12:
                continue
            
            m = (y_s[1] - y_s[0]) / denom
            b = y_s[0] - m * x_s[0]
            
            # Compute residuals and inliers
            residuals = np.abs(values - (m * coords + b))
            inliers = residuals <= thr
            n_inliers = inliers.sum()
            
            # Early termination if enough inliers
            if n_inliers >= n * adaptive_ratio:
                logger.info(f"Early termination at trial {trial}: {n_inliers}/{n} inliers exceed {adaptive_ratio:.2f} ratio")
                
                if n_inliers >= self.min_inliers:
                    x_in = coords[inliers]
                    y_in = values[inliers]
                    
                    try:
                        # Refit using unified helper
                        m_refit, b_refit = self._refit_linear(x_in, y_in)
                        r2 = self._r2(x_in, y_in, m_refit, b_refit)
                        
                        best_m, best_b = m_refit, b_refit
                        best_r2 = r2
                        best_inliers = inliers
                        break
                    except Exception as e:
                        logger.debug(f"Early termination refit failed: {e}")
                        pass
            
            # Skip if too few inliers
            if n_inliers < self.min_inliers:
                continue
            
            # Refit on all inliers using unified helper
            x_in = coords[inliers]
            y_in = values[inliers]
            
            try:
                m_refit, b_refit = self._refit_linear(x_in, y_in)
            except Exception as e:
                logger.debug(f"Refit failed at trial {trial}: {e}")
                continue
            
            r2 = self._r2(x_in, y_in, m_refit, b_refit)
            
            if r2 > best_r2:
                best_r2 = r2
                best_m, best_b = m_refit, b_refit
                best_inliers = inliers
        
        if best_inliers is None:
            logger.warning("RANSACCalibration failed to find a valid model")
            return None
        
        logger.info(
            f"RANSACCalibration completed: R²={best_r2:.4f}, slope={best_m:.4f}, "
            f"intercept={best_b:.4f}, inliers={best_inliers.sum()}/{n}"
        )
        
        # Determine coordinate system based on axis type and slope
        is_inverted = (axis_type.lower() == 'y' and best_m > 0)  # Y-axis with positive slope means inverted
        coordinate_system = 'image'  # Default assumption: image coordinates where Y increases down
        
        return CalibrationResult(
            func=self._make_func(best_m, best_b),
            r2=float(best_r2),
            coeffs=(best_m, best_b),
            inliers=best_inliers,
            is_inverted=is_inverted,
            coordinate_system=coordinate_system,
        )