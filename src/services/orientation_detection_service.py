"""
Orientation detection service with dimension variance as primary detector.
Centralizes orientation inference logic for all chart types.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from dataclasses import dataclass

from services.orientation_service import Orientation

@dataclass
class OrientationResult:
    """Result of orientation detection with diagnostics."""
    orientation: Orientation  # Type-safe enum
    confidence: float  # [0, 1]
    method: str  # 'variance', 'aspect_ratio', 'spatial_distribution', 'majority_vote', etc.
    aspect_ratio: float
    consistency: float  # Fraction of elements agreeing with detected orientation
    cv_width: float = 0.0  # Coefficient of variation for widths
    cv_height: float = 0.0  # Coefficient of variation for heights

class OrientationDetectionService:
    """
    Robust orientation detection using dimension variance as primary method.
    
    Mathematical foundation:
    ========================
    
    KEY INSIGHT:
    - VERTICAL charts: Width is CONSTANT, Height VARIES (data-bearing)
    - HORIZONTAL charts: Height is CONSTANT, Width VARIES (data-bearing)
    
    Coefficient of Variation (CV):
        CV_w = σ_w / μ_w  (width variability relative to mean)
        CV_h = σ_h / μ_h  (height variability relative to mean)
    
    Decision hierarchy:
    1. Variance-based (primary): CV_h >> CV_w → vertical, CV_w >> CV_h → horizontal
    2. Aspect ratio (fallback): median(h)/median(w) > 1.5 → vertical, < 0.67 → horizontal
    3. Spatial distribution (ultimate fallback): position variance
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def detect(
        self,
        elements: List[Dict],
        img_width: int,
        img_height: int,
        variance_threshold: float = 2.0,
        cv_threshold: float = 0.15,
        aspect_threshold: float = 1.5,
        confidence_threshold: float = 0.6,
        chart_type: str = 'bar'
    ) -> OrientationResult:
        """
        Multi-stage orientation detection with variance as primary method.
        
        Stage 1: Dimension variance (CV-based) - MOST ROBUST
        Stage 2: Aspect ratio (fallback for uniform elements)
        Stage 3: Spatial distribution (ultimate fallback)
        Stage 4: Majority vote (final fallback)
        
        Args:
            elements: List of chart elements with 'xyxy' bounding boxes
            img_width: Image width in pixels
            img_height: Image height in pixels
            variance_threshold: CV ratio threshold (default 2.0)
                - If CV_h > CV_w × 2.0 → vertical
                - If CV_w > CV_h × 2.0 → horizontal
            cv_threshold: Minimum CV to consider dimension as varying (default 0.15)
                - CV < 0.15 means dimension is essentially constant
            aspect_threshold: Aspect ratio threshold for fallback (default 1.5)
            confidence_threshold: Minimum consistency for aspect-based decision
            chart_type: Chart type hint ('bar', 'box', etc.)
        
        Returns:
            OrientationResult with orientation, confidence, and diagnostics
        """
        if not elements:
            self.logger.warning("No elements provided for orientation detection - defaulting to vertical")
            return OrientationResult(
                orientation=Orientation.VERTICAL,
                confidence=0.0,
                method='no_elements',
                cv_width=0.0,
                cv_height=0.0,
                aspect_ratio=0.0,
                consistency=0.0
            )
        
        if len(elements) < 2:
            # Single element: use aspect ratio
            el = elements[0]
            width = el['xyxy'][2] - el['xyxy'][0]
            height = el['xyxy'][3] - el['xyxy'][1]
            aspect = height / width if width > 0 else float('inf')
            
            orientation = Orientation.VERTICAL if aspect > 1.0 else Orientation.HORIZONTAL
            confidence = min(1.0, abs(aspect - 1.0))
            
            return OrientationResult(
                orientation=orientation,
                confidence=confidence,
                method='single_element',
                cv_width=0.0,
                cv_height=0.0,
                aspect_ratio=aspect,
                consistency=1.0
            )
        
        # Extract dimensions
        # Extract dimensions (Force float to avoid sequence/object types)
        widths = np.array([float(el['xyxy'][2]) - float(el['xyxy'][0]) for el in elements], dtype=np.float64)
        heights = np.array([float(el['xyxy'][3]) - float(el['xyxy'][1]) for el in elements], dtype=np.float64)
        
        # Calculate statistics
        mean_w = np.mean(widths)
        mean_h = np.mean(heights)
        std_w = np.std(widths)
        std_h = np.std(heights)
        
        # Coefficient of Variation (scale-invariant variance)
        cv_w = std_w / mean_w if mean_w > 0 else 0.0
        cv_h = std_h / mean_h if mean_h > 0 else 0.0
        
        # Median aspect ratio (for fallback)
        median_width = np.median(widths)
        median_height = np.median(heights)
        aspect_ratio = median_height / median_width if median_width > 0 else float('inf')
        
        # ============================================================
        # STAGE 1: VARIANCE-BASED DETECTION (PRIMARY METHOD)
        # ============================================================
        
        # Check if height varies significantly more than width
        if cv_h > cv_threshold and cv_h > cv_w * variance_threshold:
            # Height varies → VERTICAL chart
            # Confidence increases with CV ratio
            confidence = min(1.0, (cv_h / (cv_w + 1e-6)) / variance_threshold)
            
            self.logger.info(
                f"Variance detection: VERTICAL (CV_h={cv_h:.3f} >> CV_w={cv_w:.3f})"
            )
            
            return OrientationResult(
                orientation=Orientation.VERTICAL,
                confidence=confidence,
                method='variance',
                cv_width=cv_w,
                cv_height=cv_h,
                aspect_ratio=aspect_ratio,
                consistency=confidence
            )
        
        # Check if width varies significantly more than height
        if cv_w > cv_threshold and cv_w > cv_h * variance_threshold:
            # Width varies → HORIZONTAL chart
            confidence = min(1.0, (cv_w / (cv_h + 1e-6)) / variance_threshold)
            
            self.logger.info(
                f"Variance detection: HORIZONTAL (CV_w={cv_w:.3f} >> CV_h={cv_h:.3f})"
            )
            
            return OrientationResult(
                orientation=Orientation.HORIZONTAL,
                confidence=confidence,
                method='variance',
                cv_width=cv_w,
                cv_height=cv_h,
                aspect_ratio=aspect_ratio,
                consistency=confidence
            )
        
        # ============================================================
        # STAGE 2: ASPECT RATIO DETECTION (FALLBACK)
        # ============================================================
        
        # Variance is ambiguous (both vary similarly or neither varies much)
        # Fall back to aspect ratio with element-level consistency check
        
        self.logger.info(
            f"Variance ambiguous (CV_w={cv_w:.3f}, CV_h={cv_h:.3f}) - "
            f"falling back to aspect ratio"
        )
        
        # Calculate element-level consistency
        vertical_votes = np.sum(heights > widths * aspect_threshold)
        horizontal_votes = np.sum(widths > heights * aspect_threshold)
        total = len(elements)
        
        consistency = max(vertical_votes, horizontal_votes) / total
        
        if aspect_ratio > aspect_threshold:
            # Aspect suggests vertical
            if consistency > confidence_threshold:
                confidence = min(1.0, consistency * (aspect_ratio - aspect_threshold) / aspect_threshold)
                
                return OrientationResult(
                    orientation=Orientation.VERTICAL,
                    confidence=confidence,
                    method='aspect_ratio',
                    cv_width=cv_w,
                    cv_height=cv_h,
                    aspect_ratio=aspect_ratio,
                    consistency=consistency
                )
            else:
                self.logger.warning(
                    f"Aspect ratio suggests vertical ({aspect_ratio:.2f}), "
                    f"but low consistency ({consistency:.2f})"
                )
        
        elif aspect_ratio < 1.0 / aspect_threshold:
            # Aspect suggests horizontal
            if consistency > confidence_threshold:
                confidence = min(1.0, consistency * (1.0/aspect_ratio - aspect_threshold) / aspect_threshold)
                
                return OrientationResult(
                    orientation=Orientation.HORIZONTAL,
                    confidence=confidence,
                    method='aspect_ratio',
                    cv_width=cv_w,
                    cv_height=cv_h,
                    aspect_ratio=aspect_ratio,
                    consistency=consistency
                )
            else:
                self.logger.warning(
                    f"Aspect ratio suggests horizontal ({aspect_ratio:.2f}), "
                    f"but low consistency ({consistency:.2f})"
                )
        
        # ============================================================
        # STAGE 3: SPATIAL DISTRIBUTION (ULTIMATE FALLBACK)
        # ============================================================
        
        spatial_result = self._detect_spatial(elements, img_width, img_height)
        
        if spatial_result['confidence'] > 0.6:
            # Spatial distribution provides clear signal
            return OrientationResult(
                orientation=spatial_result['orientation'],
                confidence=spatial_result['confidence'],
                method='spatial_distribution',
                cv_width=cv_w,
                cv_height=cv_h,
                aspect_ratio=aspect_ratio,
                consistency=consistency
            )
        
        # ============================================================
        # STAGE 4: MAJORITY VOTE OR DEFAULT FALLBACK
        # ============================================================
        
        if vertical_votes > horizontal_votes:
            return OrientationResult(
                orientation=Orientation.VERTICAL,
                confidence=vertical_votes / total,
                method='majority_vote',
                cv_width=cv_w,
                cv_height=cv_h,
                aspect_ratio=aspect_ratio,
                consistency=vertical_votes / total
            )
        elif horizontal_votes > vertical_votes:
            return OrientationResult(
                orientation=Orientation.HORIZONTAL,
                confidence=horizontal_votes / total,
                method='majority_vote',
                cv_width=cv_w,
                cv_height=cv_h,
                aspect_ratio=aspect_ratio,
                consistency=horizontal_votes / total
            )
        
        # Ultimate fallback: prefer vertical if completely ambiguous
        self.logger.warning(
            f"Completely ambiguous orientation: CV_h={cv_h:.3f}, CV_w={cv_w:.3f}, "
            f"aspect={aspect_ratio:.2f}, votes={vertical_votes}/{horizontal_votes} - "
            f"defaulting to vertical"
        )
        
        return OrientationResult(
            orientation=Orientation.VERTICAL,
            confidence=0.5,
            method='default_fallback',
            cv_width=cv_w,
            cv_height=cv_h,
            aspect_ratio=aspect_ratio,
            consistency=0.5
        )
    
    def _detect_spatial(
        self,
        elements: List[Dict],
        img_width: int,
        img_height: int
    ) -> Dict[str, any]:
        """
        Detect orientation using spatial distribution of element centers.
        
        Heuristic:
        - Vertical bars: high X variance (spread horizontally), low Y variance
        - Horizontal bars: low X variance, high Y variance (spread vertically)
        
        Returns dict with 'orientation' and 'confidence'
        """
        if len(elements) < 2:
            return {'orientation': Orientation.VERTICAL, 'confidence': 0.0}
        
        centers_x = [(el['xyxy'][0] + el['xyxy'][2]) / 2.0 for el in elements]
        centers_y = [(el['xyxy'][1] + el['xyxy'][3]) / 2.0 for el in elements]
        
        # Normalized variance (prevents image size bias)
        var_x = np.var(centers_x) / (img_width ** 2) if img_width > 0 else 0
        var_y = np.var(centers_y) / (img_height ** 2) if img_height > 0 else 0
        
        # Avoid division by zero
        if var_y == 0 and var_x == 0:
            return {'orientation': Orientation.VERTICAL, 'confidence': 0.0}
        
        # Calculate spatial confidence
        if var_x > var_y * 2.0:
            # Elements spread horizontally → vertical bars
            spatial_confidence = min(1.0, var_x / (var_y + 1e-6) / 2.0)
            return {'orientation': Orientation.VERTICAL, 'confidence': spatial_confidence}
        elif var_y > var_x * 2.0:
            # Elements spread vertically → horizontal bars
            spatial_confidence = min(1.0, var_y / (var_x + 1e-6) / 2.0)
            return {'orientation': Orientation.HORIZONTAL, 'confidence': spatial_confidence}
        else:
            # No clear spatial pattern
            return {'orientation': Orientation.VERTICAL, 'confidence': 0.0}
    
    def detect_simple(
        self,
        elements: List[Dict],
        variance_threshold: float = 2.0,
        cv_threshold: float = 0.15
    ) -> Tuple[str, float]:
        """
        Simplified orientation detection using variance as primary method (backward compatibility).
        
        Returns (orientation, confidence) tuple.
        """
        if not elements or len(elements) < 2:
            return 'vertical', 0.0
        
        widths = np.array([el['xyxy'][2] - el['xyxy'][0] for el in elements])
        heights = np.array([el['xyxy'][3] - el['xyxy'][1] for el in elements])
        
        # Coefficient of Variation
        cv_w = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 0
        cv_h = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 0
        
        # Variance-based detection
        if cv_h > cv_threshold and cv_h > cv_w * variance_threshold:
            confidence = min(1.0, cv_h / (cv_w + 1e-6) / variance_threshold)
            return 'vertical', confidence
        
        if cv_w > cv_threshold and cv_w > cv_h * variance_threshold:
            confidence = min(1.0, cv_w / (cv_h + 1e-6) / variance_threshold)
            return 'horizontal', confidence
        
        # Fallback to aspect ratio
        median_aspect = np.median(heights) / np.median(widths) if np.median(widths) > 0 else 1.0
        
        if median_aspect > 1.5:
            return 'vertical', 0.6
        elif median_aspect < 0.67:
            return 'horizontal', 0.6
        else:
            return ('vertical' if median_aspect > 1.0 else 'horizontal'), 0.4