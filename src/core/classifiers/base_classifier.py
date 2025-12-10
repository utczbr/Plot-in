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
    
    def _compute_element_context(
        self,
        chart_elements: List[Dict],
        chart_type: str,
        img_width: int,
        img_height: int,
        orientation: str
    ) -> Dict:
        """
        Extract structural features from chart elements for contextual classification.
        
        Ported from spatial_classification_enhanced.py.
        
        Returns:
            Dict with keys: extent, positions, orientation, num_elements, avg_spacing, etc.
        """
        if not chart_elements:
            return {
                'extent': None,
                'positions': np.array([]).reshape(0, 2),
                'orientation': orientation,
                'num_elements': 0,
                'avg_spacing': 0.0,
                'element_centers': np.array([]),
                'chart_type': chart_type,
                'median_box_width': 0.0,
                'median_box_height': 0.0,
                'point_density': 0.0,
                'x_spread': 0.0,
                'y_spread': 0.0
            }
        
        # Extract positions
        try:
            element_positions = np.array([
                ((el['xyxy'][0] + el['xyxy'][2])/2, (el['xyxy'][1] + el['xyxy'][3])/2)
                for el in chart_elements
                if 'xyxy' in el and len(el['xyxy']) >= 4
            ])
        except (KeyError, IndexError, TypeError):
            element_positions = np.array([]).reshape(0, 2)
        
        if len(element_positions) == 0:
            return {
                'extent': None,
                'positions': element_positions,
                'orientation': orientation,
                'num_elements': 0,
                'avg_spacing': 0.0,
                'element_centers': np.array([]),
                'chart_type': chart_type,
                'median_box_width': 0.0,
                'median_box_height': 0.0,
                'point_density': 0.0,
                'x_spread': 0.0,
                'y_spread': 0.0
            }
        
        # Compute extent
        x_coords = element_positions[:, 0]
        y_coords = element_positions[:, 1]
        
        x_range = np.max(x_coords) - np.min(x_coords) if len(x_coords) > 1 else 0.0
        y_range = np.max(y_coords) - np.min(y_coords) if len(y_coords) > 1 else 0.0
        
        padding = 0.02
        extent = {
            'left': np.min(x_coords) - x_range * padding,
            'right': np.max(x_coords) + x_range * padding,
            'top': np.min(y_coords) - y_range * padding,
            'bottom': np.max(y_coords) + y_range * padding
        }
        
        # Compute spacing
        if orientation == 'vertical':
            centers = x_coords
        else:
            centers = y_coords
        
        avg_spacing = float(np.mean(np.diff(np.sort(centers)))) if len(centers) > 1 else 0.0
        
        context = {
            'extent': extent,
            'positions': element_positions,
            'orientation': orientation,
            'num_elements': len(chart_elements),
            'avg_spacing': avg_spacing,
            'element_centers': centers,
            'chart_type': chart_type,
            'x_spread': float(x_range),
            'y_spread': float(y_range)
        }
        
        # Chart-type specific features
        if chart_type in ['box']:
            try:
                box_widths = [el['xyxy'][2] - el['xyxy'][0] for el in chart_elements if 'xyxy' in el]
                box_heights = [el['xyxy'][3] - el['xyxy'][1] for el in chart_elements if 'xyxy' in el]
                context['median_box_width'] = float(np.median(box_widths)) if box_widths else 0.0
                context['median_box_height'] = float(np.median(box_heights)) if box_heights else 0.0
            except:
                context['median_box_width'] = 0.0
                context['median_box_height'] = 0.0
        else:
            context['median_box_width'] = 0.0
            context['median_box_height'] = 0.0
        
        # Point density for scatter/line
        if chart_type in ['scatter', 'line', 'histogram']:
            if extent['right'] > extent['left'] and extent['bottom'] > extent['top']:
                total_area = (extent['right'] - extent['left']) * (extent['bottom'] - extent['top'])
                context['point_density'] = len(chart_elements) / max(total_area, 1.0)
            else:
                context['point_density'] = 0.0
        else:
            context['point_density'] = 0.0
        
        return context
