"""
Visual Tick Detection Calibration - OCR-Free Fallback

Provides calibration when OCR fails by detecting tick marks visually
and inferring scale from their uniform spacing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

from .calibration_base import BaseCalibration, CalibrationResult

logger = logging.getLogger(__name__)


@dataclass
class TickDetectionResult:
    """Result of visual tick detection."""
    tick_positions: List[float]  # Pixel positions of detected ticks
    spacing: float  # Average spacing between ticks
    confidence: float  # Detection confidence [0, 1]
    method: str  # Detection method used
    

class VisualTickCalibration(BaseCalibration):
    """
    Fallback calibration using visual tick mark detection.
    
    When OCR fails to read axis labels, this class detects tick marks
    visually and infers a linear scale based on uniform tick spacing.
    
    Detection methods:
    1. Edge detection (Sobel/Canny) to find tick mark edges
    2. Peak detection in projected intensity profile
    3. Hough line detection for longer tick marks
    
    Assumptions:
    - Tick marks are uniformly spaced (linear scale)
    - Ticks are visible as intensity changes
    - First tick represents value 0, subsequent ticks increment by 1
      (unless reference labels are provided)
    """
    
    def __init__(
        self,
        edge_threshold: float = 50.0,
        min_tick_spacing: float = 10.0,
        max_tick_spacing: float = 200.0,
        min_ticks: int = 3,
    ):
        """
        Initialize visual tick calibration.
        
        Args:
            edge_threshold: Threshold for edge detection (Canny)
            min_tick_spacing: Minimum spacing between ticks (pixels)
            max_tick_spacing: Maximum spacing between ticks (pixels)
            min_ticks: Minimum number of ticks required for valid calibration
        """
        self.edge_threshold = edge_threshold
        self.min_tick_spacing = min_tick_spacing
        self.max_tick_spacing = max_tick_spacing
        self.min_ticks = min_ticks
    
    def calibrate(self, scale_labels: List[Dict], axis_type: str) -> Optional[CalibrationResult]:
        """
        Standard calibration interface - delegates to parent for OCR-based.
        
        For visual-only calibration, use calibrate_from_image() instead.
        """
        logger.warning(
            "VisualTickCalibration.calibrate() called with labels - "
            "this is a fallback calibrator. Use calibrate_from_image() for visual detection."
        )
        # Fall back to simple linear fit if labels are available
        coords, values, weights = self._extract_points(scale_labels, axis_type)
        
        if len(coords) >= 2:
            try:
                m, b = self._refit_linear(coords, values, weights)
                r2 = self._r2(coords, values, m, b)
                return CalibrationResult(
                    func=self._make_func(m, b),
                    r2=r2,
                    coeffs=(m, b),
                )
            except Exception as e:
                logger.warning(f"Linear fit failed: {e}")
        
        return None
    
    def calibrate_from_image(
        self,
        img: np.ndarray,
        axis_region: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        axis_type: str,
        reference_values: Optional[Tuple[float, float]] = None,
    ) -> Optional[CalibrationResult]:
        """
        Calibrate from image by detecting tick marks visually.
        
        Args:
            img: Full chart image (BGR or grayscale)
            axis_region: Bounding box of axis area (x1, y1, x2, y2)
            axis_type: 'x' or 'y' 
            reference_values: Optional (first_tick_value, tick_increment)
                             Default is (0, 1) if not provided
        
        Returns:
            CalibrationResult or None if detection fails
        """
        if not HAS_CV2:
            logger.error("OpenCV (cv2) not available for visual tick detection")
            return None
        
        x1, y1, x2, y2 = axis_region
        
        # Extract axis region
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        else:
            gray = img[y1:y2, x1:x2].copy()
        
        # Detect ticks
        tick_result = self._detect_ticks(gray, axis_type)
        
        if tick_result is None or len(tick_result.tick_positions) < self.min_ticks:
            logger.warning(f"Insufficient ticks detected: {len(tick_result.tick_positions) if tick_result else 0}")
            return None
        
        # Convert to absolute coordinates
        if axis_type.lower() == 'x':
            tick_coords = [x1 + pos for pos in tick_result.tick_positions]
        else:
            tick_coords = [y1 + pos for pos in tick_result.tick_positions]
        
        # Assign values based on reference or default 0, 1, 2, ...
        if reference_values:
            first_val, increment = reference_values
        else:
            first_val, increment = 0.0, 1.0
        
        tick_values = [first_val + i * increment for i in range(len(tick_coords))]
        
        # Fit linear model
        coords = np.array(tick_coords, dtype=np.float64)
        values = np.array(tick_values, dtype=np.float64)
        
        try:
            m, b = self._refit_linear(coords, values)
            r2 = self._r2(coords, values, m, b)
            
            logger.info(
                f"Visual tick calibration: {len(tick_coords)} ticks detected, "
                f"spacing={tick_result.spacing:.1f}px, R²={r2:.4f}"
            )
            
            return CalibrationResult(
                func=self._make_func(m, b),
                r2=r2,
                coeffs=(m, b),
                inliers=np.ones(len(coords), dtype=bool),  # All ticks are "inliers"
            )
            
        except Exception as e:
            logger.error(f"Visual calibration fit failed: {e}")
            return None
    
    def _detect_ticks(self, gray: np.ndarray, axis_type: str) -> Optional[TickDetectionResult]:
        """
        Detect tick marks in grayscale axis region.
        
        Args:
            gray: Grayscale image of axis region
            axis_type: 'x' or 'y'
        
        Returns:
            TickDetectionResult or None
        """
        h, w = gray.shape
        
        # Apply edge detection
        edges = cv2.Canny(
            gray, 
            self.edge_threshold / 2, 
            self.edge_threshold,
            apertureSize=3
        )
        
        # Project edges to 1D profile
        if axis_type.lower() == 'x':
            # For X-axis, sum vertically to get horizontal position profile
            profile = np.sum(edges, axis=0).astype(np.float64)
        else:
            # For Y-axis, sum horizontally to get vertical position profile
            profile = np.sum(edges, axis=1).astype(np.float64)
        
        # Smooth profile
        kernel_size = max(3, int(self.min_tick_spacing / 4))
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(profile.reshape(1, -1), (kernel_size, 1), 0).flatten()
        
        # Find peaks (tick positions)
        tick_positions = self._find_peaks(
            smoothed,
            min_distance=int(self.min_tick_spacing),
            threshold=np.max(smoothed) * 0.3
        )
        
        if len(tick_positions) < 2:
            return TickDetectionResult(
                tick_positions=[],
                spacing=0.0,
                confidence=0.0,
                method="edge_projection"
            )
        
        # Compute spacing statistics
        spacings = np.diff(sorted(tick_positions))
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)
        
        # Confidence based on spacing uniformity
        if mean_spacing > 0:
            cv = std_spacing / mean_spacing  # Coefficient of variation
            confidence = max(0.0, min(1.0, 1.0 - cv))
        else:
            confidence = 0.0
        
        # Filter out ticks with anomalous spacing
        if confidence > 0.3:
            tick_positions = self._filter_uniform_ticks(
                tick_positions, 
                mean_spacing,
                tolerance=0.3
            )
        
        return TickDetectionResult(
            tick_positions=sorted(tick_positions),
            spacing=mean_spacing,
            confidence=confidence,
            method="edge_projection"
        )
    
    def _find_peaks(
        self, 
        signal: np.ndarray, 
        min_distance: int,
        threshold: float
    ) -> List[float]:
        """
        Find peaks in 1D signal.
        
        Args:
            signal: 1D array
            min_distance: Minimum distance between peaks
            threshold: Minimum peak height
        
        Returns:
            List of peak positions
        """
        peaks = []
        n = len(signal)
        
        for i in range(1, n - 1):
            # Check if local maximum
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                # Check threshold
                if signal[i] >= threshold:
                    # Check minimum distance from previous peaks
                    if not peaks or (i - peaks[-1]) >= min_distance:
                        peaks.append(float(i))
        
        return peaks
    
    def _filter_uniform_ticks(
        self,
        ticks: List[float],
        expected_spacing: float,
        tolerance: float = 0.3
    ) -> List[float]:
        """
        Filter tick positions to keep only uniformly spaced ones.
        
        Args:
            ticks: Raw tick positions
            expected_spacing: Expected spacing between ticks
            tolerance: Allowed deviation from expected spacing (fraction)
        
        Returns:
            Filtered tick positions
        """
        if len(ticks) < 2:
            return ticks
        
        sorted_ticks = sorted(ticks)
        filtered = [sorted_ticks[0]]
        
        for tick in sorted_ticks[1:]:
            last = filtered[-1]
            spacing = tick - last
            
            # Check if spacing is close to expected
            if abs(spacing - expected_spacing) <= expected_spacing * tolerance:
                filtered.append(tick)
            # Check if it's approximately a multiple of expected spacing
            elif spacing > expected_spacing * (1 - tolerance):
                n_intervals = round(spacing / expected_spacing)
                if abs(spacing - n_intervals * expected_spacing) <= expected_spacing * tolerance:
                    filtered.append(tick)
        
        return filtered
