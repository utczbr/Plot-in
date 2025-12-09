"""
Dual-axis detection service as a single source of truth.

This service consolidates the duplicated logic from:
- label_classification_service._detect_dual_axis()
- ModularBaselineDetector.decide_dual_axis()
- spatial_classification_enhanced.detect_and_separate_dual_axis()

It provides unified dual-axis detection with consistent results.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans

from services.orientation_service import Orientation

@dataclass
class DualAxisDecision:
    """Unified dual-axis detection result."""
    has_dual_axis: bool
    primary_labels: List[Dict]
    secondary_labels: List[Dict]
    separation_px: float
    confidence: float
    method: str  # 'clustering', 'heuristic', 'metadata'
    diagnostics: Dict

class DualAxisDetectionService:
    """
    Single source of truth for dual-axis detection.
    
    Consolidates logic from:
    - label_classification_service._detect_dual_axis()
    - ModularBaselineDetector.decide_dual_axis()
    - spatial_classification_enhanced.detect_and_separate_dual_axis()
    
    Strategy:
    1. Use ModularBaselineDetector's clustering as primary method
    2. Validate with heuristics (edge proximity, balance)
    3. Override with metadata if explicitly provided
    """
    
    def __init__(self, 
                 clusterer_factory=None,
                 min_separation_ratio: float = 0.30,
                 min_balance: float = 0.35):
        """
        Args:
            clusterer_factory: Function(img_shape) -> Clusterer
            min_separation_ratio: Minimum separation as fraction of image width/height
            min_balance: Minimum balance (0-1) for valid dual-axis
        """
        self.clusterer_factory = clusterer_factory
        self.min_sep_ratio = min_separation_ratio
        self.min_balance = min_balance
    
    def detect(self,
               axis_labels: List[Dict],
               chart_elements: List[Dict],
               orientation: Orientation,
               image_size: Tuple[int, int],
               explicit_dual: Optional[bool] = None) -> DualAxisDecision:
        """
        Detect dual-axis configuration.
        
        Args:
            axis_labels: Numeric scale labels with 'xyxy', 'cleanedvalue'
            chart_elements: Chart elements (bars, points) with 'xyxy'
            orientation: Chart orientation enum
            image_size: (width, height)
            explicit_dual: If provided, override detection
        
        Returns:
            DualAxisDecision with primary/secondary splits
        """
        w, h = image_size
        
        # Priority 1: Explicit metadata (from chart detection model)
        if explicit_dual is not None:
            if explicit_dual:
                return self._force_dual_split(axis_labels, orientation, image_size)
            else:
                return DualAxisDecision(
                    has_dual_axis=False,
                    primary_labels=axis_labels,
                    secondary_labels=[],
                    separation_px=0.0,
                    confidence=1.0,
                    method='explicit_metadata',
                    diagnostics={}
                )
        
        # Priority 2: Clustering-based detection (most reliable)
        if len(axis_labels) >= 4:
            cluster_result = self._detect_via_clustering(
                axis_labels, chart_elements, orientation, image_size
            )
            if cluster_result.confidence > 0.7:
                return cluster_result
        
        # Priority 3: Heuristic fallback (for <4 labels)
        heuristic_result = self._detect_via_heuristics(
            axis_labels, chart_elements, orientation, image_size
        )
        
        return heuristic_result
    
    def _force_dual_split(self, 
                          axis_labels: List[Dict], 
                          orientation: Orientation,
                          image_size: Tuple[int, int]) -> DualAxisDecision:
        """Force split based on spatial position (left/right or top/bottom)."""
        w, h = image_size
        
        if orientation == Orientation.VERTICAL:  # Horizontal bars or vertical orientation
            # Split by x-coordinate (left/right)
            mid = w / 2
            primary = [lbl for lbl in axis_labels if (lbl['xyxy'][0] + lbl['xyxy'][2])/2 < mid]
            secondary = [lbl for lbl in axis_labels if (lbl['xyxy'][0] + lbl['xyxy'][2])/2 >= mid]
        else:  # Horizontal orientation
            # Split by y-coordinate (top/bottom)
            mid = h / 2
            primary = [lbl for lbl in axis_labels if (lbl['xyxy'][1] + lbl['xyxy'][3])/2 < mid]
            secondary = [lbl for lbl in axis_labels if (lbl['xyxy'][1] + lbl['xyxy'][3])/2 >= mid]
        
        separation = abs(len(primary) - len(secondary)) / len(axis_labels) if axis_labels else 0
        
        return DualAxisDecision(
            has_dual_axis=True,
            primary_labels=primary,
            secondary_labels=secondary,
            separation_px=separation,
            confidence=1.0,
            method='explicit_force',
            diagnostics={'n_primary': len(primary), 'n_secondary': len(secondary)}
        )
    
    def _detect_via_clustering(self, 
                                axis_labels: List[Dict],
                                chart_elements: List[Dict],
                                orientation: Orientation,
                                image_size: Tuple[int, int]) -> DualAxisDecision:
        """
        Use K-means clustering to detect dual-axis based on label positions.
        
        This is the ONLY place where clustering runs for dual-axis detection.
        """
        w, h = image_size
        
        # Extract perpendicular coordinates based on orientation
        if orientation == Orientation.VERTICAL:
            coords = np.array([
                (lbl['xyxy'][0] + lbl['xyxy'][2]) / 2.0 
                for lbl in axis_labels
            ]).reshape(-1, 1)
            threshold = self.min_sep_ratio * w
        else:
            coords = np.array([
                (lbl['xyxy'][1] + lbl['xyxy'][3]) / 2.0 
                for lbl in axis_labels
            ]).reshape(-1, 1)
            threshold = self.min_sep_ratio * h
        
        if len(coords) < 2:
            return DualAxisDecision(
                has_dual_axis=False,
                primary_labels=axis_labels,
                secondary_labels=[],
                separation_px=0.0,
                confidence=0.5,
                method='insufficient_data',
                diagnostics={}
            )
        
        # KMeans with k=2
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(coords)
        centers = kmeans.cluster_centers_.flatten()
        
        # Compute separation
        separation = abs(centers[0] - centers[1])
        
        # Compute balance
        n0 = int(np.sum(labels == 0))
        n1 = int(np.sum(labels == 1))
        balance = 1.0 - abs(n0 - n1) / float(max(n0 + n1, 1))
        
        # Edge proximity scores
        low_edge, high_edge = (0.0, float(w)) if orientation == Orientation.VERTICAL else (0.0, float(h))
        low_score = float(np.mean(np.exp(-(centers / high_edge - 0.0) / 0.08)))
        high_score = float(np.mean(np.exp(-(1.0 - centers / high_edge) / 0.08)))
        
        # Decision criteria
        dual = (
            separation > threshold and
            low_score > 0.5 and high_score > 0.5 and
            balance > self.min_balance
        )
        
        # Split labels
        if dual:
            primary_idx = 0 if centers[0] < centers[1] else 1
            primary = [axis_labels[i] for i in range(len(axis_labels)) if labels[i] == primary_idx]
            secondary = [axis_labels[i] for i in range(len(axis_labels)) if labels[i] != primary_idx]
        else:
            primary = axis_labels
            secondary = []
        
        confidence = float(
            0.5 + 
            0.25 * min(separation / threshold, 1.0) +
            0.25 * balance
        )
        
        return DualAxisDecision(
            has_dual_axis=dual,
            primary_labels=primary,
            secondary_labels=secondary,
            separation_px=separation,
            confidence=confidence,
            method='kmeans_clustering',
            diagnostics={
                'n_clusters': 2,
                'centers': centers.tolist(),
                'separation_px': float(separation),
                'threshold_px': float(threshold),
                'balance': float(balance),
                'low_score': float(low_score),
                'high_score': float(high_score),
                'n_primary': len(primary),
                'n_secondary': len(secondary)
            }
        )
    
    def _detect_via_heuristics(self,
                                axis_labels: List[Dict],
                                chart_elements: List[Dict],
                                orientation: Orientation,
                                image_size: Tuple[int, int]) -> DualAxisDecision:
        """
        Heuristic fallback for dual-axis detection when <4 labels.
        """
        w, h = image_size
        
        # Simple heuristic: if labels are spread across image, may be dual-axis
        if orientation == Orientation.VERTICAL:
            positions = [(lbl['xyxy'][0] + lbl['xyxy'][2]) / 2.0 for lbl in axis_labels]
            if positions:
                pos_range = max(positions) - min(positions)
                if pos_range > 0.6 * w:  # More than 60% of width
                    # Try to split by position
                    mid = (max(positions) + min(positions)) / 2
                    primary = [lbl for lbl in axis_labels 
                              if (lbl['xyxy'][0] + lbl['xyxy'][2])/2 < mid]
                    secondary = [lbl for lbl in axis_labels 
                                if (lbl['xyxy'][0] + lbl['xyxy'][2])/2 >= mid]
                    
                    return DualAxisDecision(
                        has_dual_axis=True,
                        primary_labels=primary,
                        secondary_labels=secondary,
                        separation_px=pos_range,
                        confidence=0.6,
                        method='heuristic_position',
                        diagnostics={'position_range': pos_range, 'width_ratio': pos_range / w}
                    )
        else:
            positions = [(lbl['xyxy'][1] + lbl['xyxy'][3]) / 2.0 for lbl in axis_labels]
            if positions:
                pos_range = max(positions) - min(positions)
                if pos_range > 0.6 * h:  # More than 60% of height
                    # Try to split by position
                    mid = (max(positions) + min(positions)) / 2
                    primary = [lbl for lbl in axis_labels 
                              if (lbl['xyxy'][1] + lbl['xyxy'][3])/2 < mid]
                    secondary = [lbl for lbl in axis_labels 
                                if (lbl['xyxy'][1] + lbl['xyxy'][3])/2 >= mid]
                    
                    return DualAxisDecision(
                        has_dual_axis=True,
                        primary_labels=primary,
                        secondary_labels=secondary,
                        separation_px=pos_range,
                        confidence=0.6,
                        method='heuristic_position',
                        diagnostics={'position_range': pos_range, 'height_ratio': pos_range / h}
                    )
        
        # Default: single axis
        return DualAxisDecision(
            has_dual_axis=False,
            primary_labels=axis_labels,
            secondary_labels=[],
            separation_px=0.0,
            confidence=0.5,
            method='heuristic_fallback',
            diagnostics={'n_labels': len(axis_labels)}
        )