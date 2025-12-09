import json
import logging
from pathlib import Path
from typing import Dict, List
from .spatial_classification_enhanced import spatial_classify_axis_labels_enhanced

logger = logging.getLogger(__name__)

class HypertunedSpatialClassifier:
    """
    Production-ready classifier using optimized LYAA parameters.
    Loads hypertuning results and applies to spatial classification.
    """
    
    def __init__(self, hypertuning_results_path: str = 'lylaa_hypertuning_results.json'):
        """Initialize with hypertuned parameters"""
        results_path = Path(hypertuning_results_path)
        
        if not results_path.exists():
            logger.warning(f"Hypertuning results not found at {hypertuning_results_path}")
            logger.warning("Using default LYAA parameters")
            self.detection_settings = self._get_default_settings()
            return
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        optimal_params = results['optimal_parameters']
        logger.info(f"Loaded {len(optimal_params)} optimized parameters")
        logger.info(f"Training accuracy: {results.get('best_accuracy', 0):.4f}")
        
        self.detection_settings = {
            'classification_threshold': optimal_params['classification_threshold'],
            'size_threshold_width': optimal_params['size_threshold_width'],
            'size_threshold_height': optimal_params['size_threshold_height'],
            'aspect_ratio_min': optimal_params['aspect_ratio_min'],
            'aspect_ratio_max': optimal_params['aspect_ratio_max'],
            'eps_factor': optimal_params['eps_factor'],
            'left_y_axis_weight': optimal_params['left_y_axis_weight'],
            'right_y_axis_weight': optimal_params['right_y_axis_weight'],
            'bottom_x_axis_weight': optimal_params['bottom_x_axis_weight'],
            'top_title_weight': optimal_params['top_title_weight'],
            'center_data_weight': optimal_params['center_data_weight'],
            'size_constraint_primary': optimal_params['size_constraint_primary'],
            'size_constraint_secondary': optimal_params['size_constraint_secondary'],
            'aspect_ratio_weight': optimal_params['aspect_ratio_weight'],
            'position_weight_primary': optimal_params['position_weight_primary'],
            'position_weight_secondary': optimal_params['position_weight_secondary'],
            'distance_weight': optimal_params['distance_weight'],
            'context_weight_primary': optimal_params['context_weight_primary'],
            'context_weight_secondary': optimal_params['context_weight_secondary'],
            'ocr_numeric_boost': optimal_params['ocr_numeric_boost'],
            'ocr_numeric_penalty': optimal_params['ocr_numeric_penalty'],
            'sigma_x': optimal_params['sigma_x'],
            'sigma_y': optimal_params['sigma_y']
        }
    
    def _get_default_settings(self) -> Dict:
        """Fallback to default LYAA parameters"""
        return {
            'classification_threshold': 1.5,
            'size_threshold_width': 0.08,
            'size_threshold_height': 0.04,
            'aspect_ratio_min': 0.5,
            'aspect_ratio_max': 3.5,
            'eps_factor': 0.12,
            'left_y_axis_weight': 5.0,
            'right_y_axis_weight': 4.0,
            'bottom_x_axis_weight': 5.0,
            'top_title_weight': 4.0,
            'center_data_weight': 2.0,
            'size_constraint_primary': 3.0,
            'size_constraint_secondary': 2.5,
            'aspect_ratio_weight': 2.5,
            'position_weight_primary': 5.0,
            'position_weight_secondary': 4.0,
            'distance_weight': 2.0,
            'context_weight_primary': 4.0,
            'context_weight_secondary': 5.0,
            'ocr_numeric_boost': 2.0,
            'ocr_numeric_penalty': 1.0,
            'sigma_x': 0.09,
            'sigma_y': 0.09
        }
    
    def classify(self, axis_labels: List[Dict], chart_elements: List[Dict],
                 chart_type: str, image_width: int, image_height: int,
                 chart_orientation: str = 'vertical', mode: str = 'precise') -> Dict[str, List[Dict]]:
        """
        Classify axis labels using hypertuned parameters.
        
        Args:
            axis_labels: List of detected axis label bboxes
            chart_elements: List of detected chart data elements
            chart_type: 'bar', 'box', 'scatter', 'line'
            image_width, image_height: Image dimensions
            chart_orientation: 'vertical' or 'horizontal'
            mode: 'fast', 'optimized', or 'precise'
        
        Returns:
            Dict with keys 'scale_label', 'tick_label', 'axis_title'
        """
        return spatial_classify_axis_labels_enhanced(
            axis_labels=axis_labels,
            chart_elements=chart_elements,
            chart_type=chart_type,
            image_width=image_width,
            image_height=image_height,
            chart_orientation=chart_orientation,
            detection_settings=self.detection_settings,
            mode=mode
        )
