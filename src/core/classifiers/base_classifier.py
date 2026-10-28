from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, ClassVar
from dataclasses import dataclass, field
import numpy as np
import logging

from services.orientation_service import Orientation

@dataclass
class ClassificationResult:
    """Structured output for classification results"""
    scale_labels: List[Dict]
    tick_labels: List[Dict]
    axis_titles: List[Dict]
    confidence: float
    metadata: Dict
    # Per-axis label pools populated by ScatterChartClassifier.
    # All other classifiers leave these as empty lists (default).
    x_scale_labels: List[Dict] = field(default_factory=list)
    y_scale_labels: List[Dict] = field(default_factory=list)

class BaseChartClassifier(ABC):
    """
    Abstract base class for chart-type specific classifiers.
    
    Design Pattern: Strategy Pattern
    Each chart type implements its own classification strategy while
    maintaining a consistent interface.
    """
    
    def __init__(self, params: Dict, logger: logging.Logger = None):
        self.params = params
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.performance_metrics = {}
        
    @classmethod
    @abstractmethod
    def get_default_params(cls) -> Dict:
        """
        Return default parameters for this classifier.
        Must be implemented by subclasses to provide a single source of truth.
        """
        pass
        
    @abstractmethod
    def classify(
        self,
        axis_labels: List[Dict],
        chart_elements: List[Dict],
        img_width: int,
        img_height: int,
        orientation: Orientation
    ) -> ClassificationResult:
        """
        Main classification method - must be implemented by subclasses.
        
        Args:
            axis_labels: Detected axis label bounding boxes with OCR text
            chart_elements: Primary chart elements (bars, points, boxes)
            img_width, img_height: Image dimensions
            orientation: Orientation enum
            
        Returns:
            ClassificationResult with labeled elements and confidence
        """
        pass
    
    def validate_inputs(self, axis_labels: List[Dict], chart_elements: List[Dict]) -> bool:
        """Input validation with comprehensive error checking"""
        if not axis_labels:
            self.logger.warning("No axis labels provided")
            return False
            
        required_keys = ['xyxy']
        for label in axis_labels:
            if not all(key in label for key in required_keys):
                self.logger.error(f"Invalid label format: {label}")
                return False
                
        return True
    
    def compute_confidence(
        self,
        classified: Dict[str, List[Dict]],
        scores: List[Dict[str, float]]
    ) -> float:
        """
        Compute classification confidence based on score distribution.
        
        Returns:
            Confidence score [0-1], used for fallback decisions
        """
        if not scores:
            return 0.5
            
        # Analyze score separation (margin between top-2 classes)
        margins = []
        for score_dict in scores:
            sorted_scores = sorted(score_dict.values(), reverse=True)
            if len(sorted_scores) >= 2:
                margin = sorted_scores[0] - sorted_scores[1]
                margins.append(margin)
        
        avg_margin = np.mean(margins) if margins else 0.0
        
        # High margin = high confidence
        confidence = min(1.0, avg_margin / 3.0)  # Normalize to [0, 1]
        
        return confidence
    
    def _compute_gaussian_region_scores(
        self,
        normalized_pos: Tuple[float, float],
        sigma_x: float = 0.09,
        sigma_y: float = 0.09,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute probabilistic scores using Gaussian kernels centered on typical axis regions.
        
        Ported from spatial_classification_enhanced.py for unified class-based architecture.
        
        Args:
            normalized_pos: (nx, ny) where nx, ny are in [0, 1]
            sigma_x, sigma_y: Gaussian spread parameters
            weights: Optional dict overriding default region weights
            
        Returns:
            Dict with keys: 'left_axis', 'right_axis', 'bottom_axis', 'top_title', 'center_plot'
        """
        nx, ny = normalized_pos
        
        # Default weights (can be overridden per chart type)
        w = weights or {}
        left_weight = w.get('left_axis_weight', 5.0)
        right_weight = w.get('right_axis_weight', 4.0)
        bottom_weight = w.get('bottom_axis_weight', 5.0)
        top_weight = w.get('top_title_weight', 4.0)
        center_weight = w.get('center_plot_weight', 2.0)
        
        scores = {}
        
        # Left Y-axis region (nx ~ 0.08, ny ~ 0.5)
        if nx < 0.20 and 0.1 < ny < 0.9:
            dx = (nx - 0.08) / sigma_x
            dy = (ny - 0.5) / sigma_y
            scores['left_axis'] = np.exp(-(dx**2 + dy**2) / 2) * left_weight
        else:
            scores['left_axis'] = 0.0
        
        # Right Y-axis region (nx ~ 0.92, ny ~ 0.5)
        if nx > 0.80 and 0.1 < ny < 0.9:
            dx = (nx - 0.92) / sigma_x
            dy = (ny - 0.5) / sigma_y
            scores['right_axis'] = np.exp(-(dx**2 + dy**2) / 2) * right_weight
        else:
            scores['right_axis'] = 0.0
        
        # Bottom X-axis region (nx ~ 0.5, ny ~ 0.92)
        if 0.15 < nx < 0.85 and ny > 0.80:
            dx = (nx - 0.5) / sigma_x
            dy = (ny - 0.92) / sigma_y
            scores['bottom_axis'] = np.exp(-(dx**2 + dy**2) / 2) * bottom_weight
        else:
            scores['bottom_axis'] = 0.0
        
        # Top title region (nx ~ 0.5, ny ~ 0.08)
        if 0.15 < nx < 0.85 and ny < 0.15:
            dx = (nx - 0.5) / sigma_x
            dy = (ny - 0.08) / sigma_y
            scores['top_title'] = np.exp(-(dx**2 + dy**2) / 2) * top_weight
        else:
            scores['top_title'] = 0.0
        
        # Center plot region (penalize labels here)
        if 0.2 < nx < 0.8 and 0.2 < ny < 0.8:
            center_dist = np.sqrt((nx - 0.5)**2 + (ny - 0.5)**2)
            scores['center_plot'] = np.exp(-(center_dist**2) / 0.08) * center_weight
        else:
            scores['center_plot'] = 0.0
        
        return scores

