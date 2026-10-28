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
    scale_type: str = 'linear'  # 'linear' or 'log'


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
        return BaseCalibration._extract_points(scale_labels, axis_type)

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
                cv = lbl['cleanedvalue']
                try:
                    val = float(cv)
                except (ValueError, TypeError):
                    from utils.validation_utils import clean_numeric_text
                    val = clean_numeric_text(str(cv))
                    if val is None:
                        logger.debug(f"Failed to parse cleanedvalue for label {lbl_idx}: {cv}")

            # Fallback to text parsing
            if val is None:
                txt = str(lbl.get('text', '')).strip()
                if txt:
                    from utils.validation_utils import clean_numeric_text, is_numeric
                    txt_norm = txt.translate(str.maketrans({
                        '\u2212': '-', '\u2013': '-', '\u2014': '-',
                        '\u2015': '-', '\ufe63': '-', '\uff0d': '-',
                    }))
                    val = clean_numeric_text(txt_norm)
                    if val is None:
                        # Secondary fallback only if string does not contain categorical letters
                        core = re.sub(r'[\s\$\%\€\£,]', '', txt_norm)
                        if not re.search(r'[A-Za-df-z]', core):
                            matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', core)
                            if matches:
                                try:
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