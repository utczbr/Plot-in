import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time

from .bar_chart_classifier import BarChartClassifier
from .line_chart_classifier import LineChartClassifier
from .scatter_chart_classifier import ScatterChartClassifier
from .box_chart_classifier import BoxChartClassifier
from .histogram_chart_classifier import HistogramChartClassifier
from .heatmap_chart_classifier import HeatmapChartClassifier
from .pie_chart_classifier import PieChartClassifier
from .base_classifier import ClassificationResult, BaseChartClassifier
from services.orientation_service import Orientation

logger = logging.getLogger(__name__)

class ProductionSpatialClassifier:
    """
    Production classifier with hypertuned parameters and fallback logic.
    
    Usage:
        classifier = ProductionSpatialClassifier('hypertuned_params.json')
        result = classifier.classify(
            chart_type='bar',
            axis_labels=labels,
            chart_elements=bars,
            img_width=800,
            img_height=600,
            orientation=Orientation.VERTICAL
        )
    """
    
    def __init__(
        self,
        hyperparams_path: Optional[Path] = None,
        enable_profiling: bool = False
    ):
        self.enable_profiling = enable_profiling
        self.performance_stats = {}
        
        # Load hypertuned parameters
        self.hyperparams = self._load_hyperparameters(hyperparams_path)
        
        # Initialize classifiers (lazy loading)
        self._classifiers = {}
        
        logger.info("ProductionSpatialClassifier initialized")
        if hyperparams_path and hyperparams_path.exists():
            logger.info(f"Using hypertuned parameters from {hyperparams_path}")
        else:
            logger.warning("Using default parameters (hypertuning results not found)")
    
    def classify(
        self,
        chart_type: str,
        axis_labels: List[Dict],
        chart_elements: List[Dict],
        img_width: int,
        img_height: int,
        orientation: Orientation,
        classification_confidence: float = 1.0
    ) -> ClassificationResult:
        """
        Main classification method with confidence-based parameter selection.
        
        Args:
            chart_type: 'bar', 'line', 'scatter', 'box', 'pie'
            axis_labels: Detected axis label bboxes with OCR
            chart_elements: Primary chart elements (bars, points, boxes)
            img_width, img_height: Image dimensions
            orientation: 'vertical' or 'horizontal'
            classification_confidence: Chart type classifier confidence [0-1]
            
        Returns:
            ClassificationResult with labeled elements and metadata
        """
        
        start_time = time.time()
        
        # Select parameters based on confidence
        params = self._select_parameters(chart_type, classification_confidence)
        
        # Get or create classifier
        classifier = self._get_classifier(chart_type, params)
        
        # Run classification
        result = classifier.classify(
            axis_labels=axis_labels,
            chart_elements=chart_elements,
            img_width=img_width,
            img_height=img_height,
            orientation=orientation
        )
        
        # Update performance stats
        if self.enable_profiling:
            elapsed = time.time() - start_time
            self._update_stats(chart_type, elapsed, result)
        
        return result
    
    def _load_hyperparameters(self, path: Optional[Path]) -> Dict:
        """Load hypertuned parameters from JSON"""
        
        if not path or not path.exists():
            logger.warning(f"Hyperparameter file not found: {path}")
            return self._get_default_parameters()
        
        try:
            with open(path, 'r') as f:
                params = json.load(f)
            
            # Check if the file has the expected structure with chart-type keys
            # If not, use the optimal_parameters as universal parameters for all chart types
            if 'optimal_parameters' in params:
                universal_params = params['optimal_parameters']
                # Apply the universal parameters to all chart types, preserving chart-specific defaults
                chart_specific_params = self._get_default_parameters()
                
                # Update each chart type's parameters with the universal ones
                # but preserve chart-specific defaults that are not in universal parameters
                for chart_type in chart_specific_params:
                    # First store the original defaults
                    original_defaults = chart_specific_params[chart_type].copy()
                    # Then update with universal parameters (this overwrites overlapping keys)
                    chart_specific_params[chart_type].update(universal_params)
                    # Add back any missing keys from original defaults
                    for key, value in original_defaults.items():
                        if key not in chart_specific_params[chart_type]:
                            chart_specific_params[chart_type][key] = value
                
                logger.info(f"Loaded universal optimal parameters and applied to all chart types")
                return chart_specific_params
            else:
                # Original structure with chart-type keys
                logger.info(f"Loaded hyperparameters for {len(params)} chart types")
                return params
        
        except Exception as e:
            logger.error(f"Error loading hyperparameters: {e}")
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict:
        """Default parameters for all chart types, dynamically loaded from classifier classes"""
        
        # Universal base parameters (preserved for fallback/tuning context)
        default = {
            'scale_size_weight': 3.0,
            'scale_aspect_weight': 2.5,
            'scale_region_boost': 5.0,
            'scale_center_dist_weight': 2.0,
            'tick_size_weight': 2.5,
            'tick_spacing_weight': 5.0,
            'tick_alignment_boost': 4.0,
            'title_aspect_weight': 4.0,
            'title_size_weight': 3.0,
            'spacing_multiplier': 1.5,
            'numeric_boost': 2.0,
            'classification_threshold': 1.5,
            'eps_factor': 0.12
        }
        
        return {
            'bar': {**default, **BarChartClassifier.get_default_params()},
            'line': {**default, **LineChartClassifier.get_default_params()},
            'scatter': {**default, **ScatterChartClassifier.get_default_params()},
            'box': {**default, **BoxChartClassifier.get_default_params()},
            'histogram': {**default, **HistogramChartClassifier.get_default_params()},
            'heatmap': {**default, **HeatmapChartClassifier.get_default_params()},
            'pie': {**default, **PieChartClassifier.get_default_params()}
        }
    
    def _select_parameters(
        self,
        chart_type: str,
        confidence: float
    ) -> Dict:
        """
        Select parameters based on classification confidence.
        
        Logic:
        - confidence >= 0.7: Use chart-type specific parameters
        - confidence < 0.7: Use generic 'bar' parameters (most robust)
        """
        
        if confidence < 0.7:
            logger.info(
                f"Low confidence ({confidence:.2f}), using fallback parameters"
            )
            return self.hyperparams.get('bar', self._get_default_parameters()['bar'])
        
        params = self.hyperparams.get(
            chart_type,
            self._get_default_parameters().get(chart_type, {})
        )
        
        return params
    
    def _get_classifier(self, chart_type: str, params: Dict) -> BaseChartClassifier:
        """Get or create classifier instance"""
        
        # Map chart types to classifier classes
        classifier_map = {
            'bar': BarChartClassifier,
            'line': LineChartClassifier,
            'scatter': ScatterChartClassifier,
            'box': BoxChartClassifier,
            'histogram': HistogramChartClassifier,
            'heatmap': HeatmapChartClassifier,
            'pie': PieChartClassifier
        }
        
        classifier_class = classifier_map.get(chart_type)
        
        if not classifier_class:
            logger.warning(f"Unknown chart type: {chart_type}, using BarChartClassifier")
            classifier_class = BarChartClassifier
        
        # Create new instance with parameters
        return classifier_class(params, logger)
    
    def _update_stats(
        self,
        chart_type: str,
        elapsed: float,
        result: ClassificationResult
    ):
        """Update performance statistics"""
        
        if chart_type not in self.performance_stats:
            self.performance_stats[chart_type] = {
                'count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'avg_confidence': 0.0
            }
        
        stats = self.performance_stats[chart_type]
        stats['count'] += 1
        stats['total_time'] += elapsed
        stats['avg_time'] = stats['total_time'] / stats['count']
        
        # Running average of confidence
        n = stats['count']
        stats['avg_confidence'] = (
            (stats['avg_confidence'] * (n - 1) + result.confidence) / n
        )
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        
        if not self.enable_profiling:
            return {'profiling_disabled': True}
        
        return {
            'per_chart_type': self.performance_stats,
            'total_classifications': sum(
                s['count'] for s in self.performance_stats.values()
            )
        }
