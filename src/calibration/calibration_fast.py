"""
Fast calibration with optional weighted least squares for consistency with robust methods.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
from .calibration_base import BaseCalibration, CalibrationResult

logger = logging.getLogger(__name__)


class FastCalibration(BaseCalibration):
    """
    Fast least-squares calibration with optional weighting.
    
    Now supports confidence-weighted fits for consistency with RANSAC/PROSAC.
    """
    
    __slots__ = ('use_weights',)
    
    def __init__(self, use_weights: bool = False):
        """
        Initialize fast calibration.
        
        Args:
            use_weights: If True, use confidence weights in least squares
        """
        self.use_weights = use_weights
    
    def calibrate(self, scale_labels: List[Dict], axis_type: str) -> Optional[CalibrationResult]:
        """Perform weighted or unweighted linear regression."""
        try:
            coords, values, weights = self._extract_points(scale_labels, axis_type, prefer_cleaned=True)
        except Exception as e:
            logger.error(f"Error extracting points in FastCalibration: {e}")
            return None
        
        # Filter NaN/Inf
        valid_mask = np.isfinite(coords) & np.isfinite(values)
        if not np.all(valid_mask):
            logger.warning(f"Filtered {(~valid_mask).sum()} invalid points")
            coords = coords[valid_mask]
            values = values[valid_mask]
            weights = weights[valid_mask]
        
        n = coords.size
        if n < 2:
            logger.warning(f"Not enough points for FastCalibration: {n} < 2")
            return None
        

        try:
            if self.use_weights:
                # Weighted least squares via unified helper
                m, b = self._refit_linear(coords, values, weights)
            else:
                # Standard unweighted least squares via unified helper
                m, b = self._refit_linear(coords, values)
            
            r2 = self._r2(coords, values, m, b)
            
            logger.info(
                f"FastCalibration ({'weighted' if self.use_weights else 'unweighted'}) completed: "
                f"R²={r2:.4f}, slope={m:.4f}, intercept={b:.4f}"
            )
            
            # Determine coordinate system based on axis type
            is_inverted = (axis_type.lower() == 'y' and m > 0)  # Y-axis with positive slope means inverted
            coordinate_system = 'image'  # Default assumption: image coordinates where Y increases down
            
            return CalibrationResult(
                func=self._make_func(m, b),
                r2=float(r2),
                coeffs=(m, b),
                inliers=None,
                is_inverted=is_inverted,
                coordinate_system=coordinate_system,
            )
        
        except Exception as e:
            logger.error(f"FastCalibration fitting failed: {e}")
            return None