"""
Base abstract class for all chart extractors.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union
import logging


class BaseExtractor(ABC):
    """
    Abstract base class for specific chart type extractors.
    
    Provides standardized methods for result construction, logging setup,
    and scale model handling to reduce code duplication.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def extract(self, img, detections, scale_model, baseline_coord, img_dimensions, **kwargs):
        """
        Extract data from the chart image.
        
        Args:
            img: Input image
            detections: Dictionary of detected elements
            scale_model: Primary axis scale model (usually Y-axis)
            baseline_coord: Baseline pixel coordinate
            img_dimensions: Dictionary with image dimensions and metadata
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            Dictionary containing extracted data and metadata
        """
        pass

    def _create_result_template(self, chart_type: str, detections: Dict, num_elements: int) -> Dict:
        """
        Creates the standardized result dictionary template.
        
        Args:
            chart_type: String identifier for the chart type
            detections: The full detections dictionary
            num_elements: Number of primary elements detected
            
        Returns:
            Initialized result dictionary with standard fields
        """
        chart_titles_list = detections.get('chart_title', [])
        
        # Extract title text safely
        chart_title = ''
        if chart_titles_list:
            chart_title = chart_titles_list[0].get('text', '')
            
        # Extract axis titles
        axis_titles = [title.get('text', '') for title in detections.get('axis_title', [])]
        
        # Extract legend items
        legend_items = [item.get('text', '') for item in detections.get('legend', [])]
        
        # Determine element key and count key based on chart_type
        if chart_type == 'bar':
            element_key = 'bars'
            count_key = 'num_bars'
        elif chart_type == 'box':
            element_key = 'boxes'
            count_key = 'num_boxes'
        elif chart_type in ['scatter', 'line', 'data_point', 'area']:
            element_key = 'data_points'
            count_key = 'num_points'
        else:
            element_key = f"{chart_type}s"
            count_key = f"num_{chart_type}s"

        return {
            element_key: [],
            count_key: num_elements,
            'chart_type': chart_type,
            'titles': {
                'chart_title': chart_title,
                'axis_titles': axis_titles
            },
            'legend': legend_items
        }

    def _resolve_scale_func(self, scale_model: Union[Any, Callable]) -> Callable:
        """
        Helper to resolve the actual scaling function from a model object or function.
        
        Args:
            scale_model: Either a callable or an object with a .func attribute
            
        Returns:
            Callable function for scaling
        """
        if scale_model is None:
            return lambda x: x  # No-op identity if None
            
        if hasattr(scale_model, 'func'):
            return scale_model.func
        return scale_model

    def _add_calibration_info(self, result: Dict, r_squared: Optional[float], 
                             baseline_coord: Optional[float], orientation: str = 'vertical'):
        """Adds standard calibration quality metadata to the result."""
        if r_squared is not None:
            result['calibration_quality'] = {
                'r_squared': r_squared,
                'baseline_coord': baseline_coord,
                'orientation': orientation
            }
