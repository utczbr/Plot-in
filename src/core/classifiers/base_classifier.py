from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, ClassVar
from dataclasses import dataclass
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
    
    @abstractmethod
    def compute_feature_scores(
        self,
        label_features: Dict,
        region_scores: Dict,
        element_context: Optional[Dict]
    ) -> Dict[str, float]:
        """
        Chart-specific feature scoring logic.
        
        Returns:
            Dict with 'scale_label', 'tick_label', 'axis_title' scores
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
