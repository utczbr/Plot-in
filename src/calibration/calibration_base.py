"""
Base calibration interfaces and shared typing with enhanced robustness.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for calibration function
CalibrationFunc = Callable[[np.ndarray | float | List[float]], np.ndarray | float]


@dataclass(frozen=True)
class CalibrationResult:
    """
    Container for calibrated mapping and diagnostics.
    
    Attributes:
        func: Callable mapping pixel coordinates to values
        r2: Coefficient of determination on inliers
        coeffs: (slope, intercept) tuple
        inliers: Boolean mask of inliers (None for pure linear fit)
        is_inverted: True if Y-axis inversion handled internally (for vertical charts)
        coordinate_system: 'image' if pixel coordinates follow image convention (Y increases down), 'data' if follows data convention
    """
    func: CalibrationFunc
    r2: float
    coeffs: Tuple[float, float]
    inliers: Optional[np.ndarray] = None
    is_inverted: bool = False
    coordinate_system: str = 'image'  # 'image' or 'data'


class BaseCalibration(ABC):
    """
    Abstract base class for calibration engines.
    
    Each implementation must be independent and provide its own
    calibration pipeline from axis labels to linear model.
    """
    
    @abstractmethod
    def calibrate(self, scale_labels: List[Dict], axis_type: str) -> Optional[CalibrationResult]:
        """
        Calibrate a linear mapping: value = m * pixel + b.

        Args:
            scale_labels: List of dicts with keys:
                - 'xyxy': [x1, y1, x2, y2] bounding box
                - 'text' or 'cleanedvalue': label text/value
                - 'ocr_confidence' (optional): confidence in [0,1]
            axis_type: 'x' or 'y' to select pixel coordinate

        Returns:
            CalibrationResult or None if calibration fails
        """
        raise NotImplementedError

    @staticmethod
    def extract_points_with_adaptive_threshold(
        scale_labels: List[Dict],
        axis_type: str,
        initial_confidence_threshold: float = 0.8,
        min_points_required: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract coords, values, weights from labels with ADAPTIVE confidence threshold.
        This is a public method that can be used by concrete implementations.

        Strategy:
        1. Start with initial_confidence_threshold (0.8)
        2. If insufficient points, reduce threshold by 0.1 steps
        3. Stop when min_points_required is reached or threshold hits 0.0

        Args:
            scale_labels: List of label dicts with 'xyxy', 'text', 'ocr_confidence' keys
            axis_type: 'x' or 'y' to select coordinate extraction
            initial_confidence_threshold: Starting confidence threshold (0.8)
            min_points_required: Minimum points needed for calibration (2)

        Returns:
            Tuple of (coords, values, weights):
            - coords: np.ndarray of pixel coordinates
            - values: np.ndarray of numeric label values
            - weights: np.ndarray of confidence scores
        """
        return BaseCalibration._extract_points_with_adaptive_threshold(
            scale_labels, axis_type, True, initial_confidence_threshold, min_points_required
        )
    
    @staticmethod
    def _extract_points_with_adaptive_threshold(
        scale_labels: List[Dict],
        axis_type: str,
        prefer_cleaned: bool = True,
        initial_confidence_threshold: float = 0.8,
        min_points_required: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract coords, values, weights from labels with ADAPTIVE confidence threshold.

        Strategy:
        1. Start with initial_confidence_threshold (0.8)
        2. If insufficient points, reduce threshold by 0.1 steps
        3. Stop when min_points_required is reached or threshold hits 0.0

        Returns:
            coords: Pixel positions (1D)
            values: Numeric values from labels (1D)
            weights: Confidences in [0,1], filtered by adaptive threshold
        """
        if not isinstance(scale_labels, list):
            raise TypeError("scale_labels must be a list of dictionaries")

        if axis_type.lower() not in ('x', 'y'):
            raise ValueError(f"axis_type must be 'x' or 'y', got {axis_type}")

        # Adaptive threshold loop
        confidence_threshold = initial_confidence_threshold
        coords_list = []
        values_list = []
        weights_list = []

        logger.debug(f"🔍 ADAPTIVE THRESHOLD CALIBRATION (axis={axis_type})")
        logger.debug(f"   ├─ Initial threshold: {confidence_threshold:.2f}")
        logger.debug(f"   ├─ Min points required: {min_points_required}")
        logger.debug(f"   └─ Total labels provided: {len(scale_labels)}")

        # Safety: limit iterations to prevent infinite loops
        max_threshold_iterations = 15  # From 0.8 down to 0.0 in 0.1 steps = max 9 iterations
        iteration_count = 0
        tried_zero_threshold = False

        while confidence_threshold >= 0.0 and iteration_count < max_threshold_iterations:
            iteration_count += 1
            coords_list = []
            values_list = []
            weights_list = []

            for lbl_idx, lbl in enumerate(scale_labels):
                if not isinstance(lbl, dict):
                    logger.warning(f"Skipping non-dictionary item at index {lbl_idx}")
                    continue

                xyxy = lbl.get('xyxy')
                if not xyxy or len(xyxy) < 4:
                    logger.warning(f"Skipping label {lbl_idx} with invalid or missing xyxy field")
                    continue

                # Extract confidence first to apply threshold
                try:
                    conf = float(lbl.get("ocr_confidence", confidence_threshold))
                    conf = max(0.0, min(1.0, conf))
                except (ValueError, TypeError):
                    logger.debug(f"   └─ Label {lbl_idx}: Invalid confidence, using threshold {confidence_threshold:.2f}")
                    conf = confidence_threshold

                # Apply threshold filter BEFORE processing
                if conf < confidence_threshold:
                    continue

                # Extract pixel coordinate
                try:
                    if axis_type.lower().startswith('y'):
                        pos = 0.5 * (float(xyxy[1]) + float(xyxy[3]))
                    else:
                        pos = 0.5 * (float(xyxy[0]) + float(xyxy[2]))
                except (ValueError, IndexError, TypeError):
                    logger.warning(f"Skipping label {lbl_idx} with invalid xyxy coordinates: {xyxy}")
                    continue

                # Extract numeric value
                val: Optional[float] = None

                # Try cleaned value first
                if prefer_cleaned and 'cleanedvalue' in lbl:
                    try:
                        val = float(lbl['cleanedvalue'])
                    except (ValueError, TypeError):
                        logger.debug(f"Failed to parse cleanedvalue for label {lbl_idx}: {lbl.get('cleanedvalue')}")
                        val = None

                # Fallback to text parsing
                if val is None:
                    txt = str(lbl.get('text', '')).strip()
                    if txt:
                        # Normalize number formats (handle commas as thousands separator or decimal)
                        t = txt.replace(' ', '').replace(',', '.')

                        # Regex: sign, digits, optional decimal, optional exponent
                        matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', t)

                        if matches:
                            try:
                                # Take first match (most common case)
                                val = float(matches[0])
                            except (ValueError, TypeError):
                                logger.debug(f"Failed to parse numeric value from text for label {lbl_idx}: {txt}")
                                val = None

                if val is None:
                    continue

                # FIXED: Add finite check
                if not np.isfinite(val):
                    logger.warning(f"Skipping label {lbl_idx} with non-finite value: {val}")
                    continue

                # Valid point found that passes threshold
                coords_list.append(pos)
                values_list.append(val)
                weights_list.append(conf)

                logger.debug(
                    f"   └─ Label {lbl_idx}: text='{lbl.get('text', 'N/A')}', val={val:.2f}, "
                    f"pos={pos:.1f}px, conf={conf:.3f} ≥ {confidence_threshold:.2f} ✓"
                )

            # Check if we have enough points
            num_points = len(coords_list)
            logger.debug(f"   ├─ Threshold {confidence_threshold:.2f}: Found {num_points} points")

            if num_points >= min_points_required:
                logger.info(
                    f"✅ ADAPTIVE THRESHOLD SUCCESS: {num_points} points at "
                    f"confidence >= {confidence_threshold:.2f}"
                )
                break

            # If we're at zero threshold, we've tried everything - break to avoid infinite loop
            if confidence_threshold <= 0.0:
                logger.debug(f"   ├─ Reached zero threshold, stopping search")
                break

            # Reduce threshold and try again
            confidence_threshold -= 0.1
            confidence_threshold = max(0.0, confidence_threshold)  # Clamp to 0.0

            logger.debug(f"   ├─ Insufficient points, lowering threshold to {confidence_threshold:.2f}")

        # Final validation
        if len(coords_list) < min_points_required:
            logger.warning(
                f"⚠️  ADAPTIVE THRESHOLD FAILED: Only {len(coords_list)} points found "
                f"after trying all thresholds down to 0.0"
            )
            return np.array([]), np.array([]), np.array([])

        # Convert to numpy arrays
        coords = np.array(coords_list, dtype=float)
        values = np.array(values_list, dtype=float)
        weights = np.array(weights_list, dtype=float)

        # Filter out NaN/inf
        valid_mask = np.isfinite(coords) & np.isfinite(values) & np.isfinite(weights)
        num_valid = np.sum(valid_mask)

        if num_valid < min_points_required:
            logger.warning(
                f"After NaN filtering: {num_valid} valid points (need {min_points_required})"
            )
            return np.array([]), np.array([]), np.array([])

        logger.debug(f"   └─ Final: {num_valid} valid points after filtering")

        return coords[valid_mask], values[valid_mask], weights[valid_mask]

    @staticmethod
    def _extract_points(
        scale_labels: List[Dict],
        axis_type: str,
        prefer_cleaned: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract coords, values, weights from labels with robust parsing.

        Returns:
            coords: Pixel positions (1D)
            values: Numeric values from labels (1D)
            weights: Confidences in [0,1], default 0.8 if missing
        """
        if not isinstance(scale_labels, list):
            raise TypeError("scale_labels must be a list of dictionaries")

        if axis_type.lower() not in ('x', 'y'):
            raise ValueError(f"axis_type must be 'x' or 'y', got {axis_type}")

        coords: List[float] = []
        values: List[float] = []
        weights: List[float] = []

        for lbl_idx, lbl in enumerate(scale_labels):
            if not isinstance(lbl, dict):
                logger.warning(f"Skipping non-dictionary item at index {lbl_idx}")
                continue

            xyxy = lbl.get('xyxy')
            if not xyxy or len(xyxy) < 4:
                logger.warning(f"Skipping label {lbl_idx} with invalid or missing xyxy field")
                continue

            # Extract pixel coordinate
            try:
                if axis_type.lower().startswith('y'):
                    pos = 0.5 * (float(xyxy[1]) + float(xyxy[3]))
                else:
                    pos = 0.5 * (float(xyxy[0]) + float(xyxy[2]))
            except (ValueError, IndexError, TypeError):
                logger.warning(f"Skipping label {lbl_idx} with invalid xyxy coordinates: {xyxy}")
                continue

            # Extract numeric value
            val: Optional[float] = None

            # Try cleaned value first
            if prefer_cleaned and 'cleanedvalue' in lbl:
                try:
                    val = float(lbl['cleanedvalue'])
                except (ValueError, TypeError):
                    logger.debug(f"Failed to parse cleanedvalue for label {lbl_idx}: {lbl.get('cleanedvalue')}")
                    val = None

            # Fallback to text parsing
            if val is None:
                txt = str(lbl.get('text', '')).strip()
                if txt:
                    # Normalize number formats (handle commas as thousands separator or decimal)
                    t = txt.replace(' ', '').replace(',', '.')

                    # Regex: sign, digits, optional decimal, optional exponent
                    matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', t)

                    if matches:
                        try:
                            # Take first match (most common case)
                            val = float(matches[0])
                        except (ValueError, TypeError):
                            logger.debug(f"Failed to parse numeric value from text for label {lbl_idx}: {txt}")
                            val = None

            if val is None:
                continue

            # FIXED: Add finite check
            if not np.isfinite(val):
                logger.warning(f"Skipping label {lbl_idx} with non-finite value: {val}")
                continue

            # Extract confidence
            try:
                conf = float(lbl.get('ocr_confidence', 0.8))
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                logger.debug(f"Invalid confidence value for label {lbl_idx}, using default 0.8")
                conf = 0.8

            coords.append(pos)
            values.append(val)
            weights.append(conf)

        return (
            np.asarray(coords, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            np.asarray(weights, dtype=np.float64),
        )
    
    @staticmethod
    def _make_func(slope: float, intercept: float) -> CalibrationFunc:
        """Create a callable function for the calibrated model."""
        def f(x: np.ndarray | float | List[float]) -> np.ndarray | float:
            if isinstance(x, (list, tuple)):
                xa = np.asarray(x, dtype=np.float64)
                return slope * xa + intercept
            
            if isinstance(x, np.ndarray):
                return slope * x.astype(np.float64, copy=False) + intercept
            
            return slope * float(x) + intercept
        
        return f
    
    @staticmethod
    def _r2(x: np.ndarray, y: np.ndarray, m: float, b: float) -> float:
        """
        Compute R² coefficient of determination.
        
        Returns 0.0 for degenerate cases (constant y).
        """
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("Input arrays must have the same length and not be empty")
        
        y_pred = m * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot > 0:
            return float(1.0 - ss_res / ss_tot)
        else:
            # FIXED: Add warning for degenerate data
            logger.warning("R² is undefined (constant y values); returning 0.0")
            return 0.0

    @staticmethod
    def _refit_linear(
        x: np.ndarray, 
        y: np.ndarray, 
        weights: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Unified robust linear regression solver (m*x + b).
        
        Handles:
        - Weighted/Unweighted least squares
        - Lstsq unpacking
        - Degenerate cases
        
        Args:
            x: Input coordinates
            y: Target values
            weights: Optional confidence weights [0,1]
            
        Returns:
            Tuple (slope, intercept)
        """
        if x.size < 2:
            raise ValueError("Not enough points to fit linear model")

        # Construct design matrix [x, 1]
        A = np.vstack([x, np.ones_like(x)]).T

        if weights is not None and weights.size == x.size:
            # Weighted least squares via sqrt(weights) transform
            # Clip minimal weight to avoid division by zero or effective point removal
            sw = np.sqrt(np.clip(weights, 1e-6, 1.0))
            A_w = A * sw[:, None]
            y_w = y * sw
            res = np.linalg.lstsq(A_w, y_w, rcond=None)
        else:
            # Standard OLS
            res = np.linalg.lstsq(A, y, rcond=None)

        sol = res[0]
        return float(sol[0]), float(sol[1])

    @staticmethod
    def _compute_mad_threshold(
        x: np.ndarray, 
        y: np.ndarray, 
        base_thr: float, 
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute adaptive inlier threshold using (Weighted) Median Absolute Deviation.
        
        Args:
            x: Points x
            y: Points y
            base_thr: Minimum threshold floor
            weights: Optional weights for weighted median calculation
            
        Returns:
            Adaptive threshold in pixels
        """
        if x.size < 3:
            return base_thr

        try:
            # Initial simple fit to get residuals
            m, b = BaseCalibration._refit_linear(x, y, weights)
            resid = np.abs(y - (m * x + b))

            if weights is not None:
                # Weighted MAD
                # Normalize weights
                w_norm = weights / np.maximum(1e-6, weights.sum())
                
                # Sort by residual size
                sorted_idx = np.argsort(resid)
                sorted_resid = resid[sorted_idx]
                sorted_weights = w_norm[sorted_idx]
                
                # Weighted median of residuals
                cumsum_weights = np.cumsum(sorted_weights)
                total_weight = cumsum_weights[-1]
                med_idx = np.searchsorted(cumsum_weights, total_weight / 2.0)
                med_idx = min(med_idx, len(sorted_resid) - 1)
                med = sorted_resid[med_idx]
                
                # Weighted MAD of deviation from median
                abs_dev = np.abs(sorted_resid - med)
                sorted_abs_dev_idx = np.argsort(abs_dev)
                sorted_abs_dev = abs_dev[sorted_abs_dev_idx]
                sorted_weights_mad = w_norm[sorted_abs_dev_idx]
                
                cumsum_mad = np.cumsum(sorted_weights_mad)
                mad_idx = np.searchsorted(cumsum_mad, total_weight / 2.0)
                mad_idx = min(mad_idx, len(sorted_abs_dev) - 1)
                mad = sorted_abs_dev[mad_idx]
                
                # Sigma estimate
                sigma = 1.4826 * mad if mad > 1e-9 else np.sqrt(np.average(resid**2, weights=weights))
                
            else:
                # Standard unweighted MAD
                median_resid = np.median(resid)
                mad = np.median(np.abs(resid - median_resid))
                sigma = 1.4826 * mad if mad > 1e-9 else np.std(resid)

            # Threshold logic: 2.5 sigma, floored at base_thr
            return max(base_thr, 2.5 * sigma)

        except Exception as e:
            logger.warning(f"Error in MAD threshold calculation: {e}", exc_info=False)
            return base_thr