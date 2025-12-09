"""
Chart analysis orchestrator with flexible service injection for diverse chart types.

ARCHITECTURAL CHANGES:
- Handlers are now instantiated with only the services they need
- _initialize_handlers() inspects handler requirements via class hierarchy
- New handlers (heatmap, pie) integrate seamlessly without modifying orchestrator logic
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import numpy as np

from core.classifiers.production_classifier import ProductionSpatialClassifier

from handlers.base_handler import (
    BaseHandler,
    CartesianChartHandler,
    GridChartHandler,
    PolarChartHandler,
    ExtractionResult,
    ChartCoordinateSystem
)

# Import all handler implementations
from handlers.bar_handler import BarHandler
from handlers.box_handler import BoxHandler
from handlers.histogram_handler import HistogramHandler
from handlers.line_handler import LineHandler
from handlers.scatter_handler import ScatterHandler
from handlers.heatmap_handler import HeatmapHandler  # NEW
from handlers.pie_handler import PieHandler          # NEW

from services.dual_axis_service import DualAxisDetectionService
from services.meta_clustering_service import MetaClusteringService
from services.orientation_service import Orientation, OrientationService
from services.color_mapping_service import ColorMappingService  # NEW
from services.legend_matching_service import LegendMatchingService  # NEW


class ChartAnalysisOrchestrator:
    """
    Main orchestrator supporting diverse chart types with flexible service injection.

    **Design Pattern: Service Locator + Dependency Injection Hybrid**
    - The orchestrator maintains a registry of shared services
    - Handlers declare their dependencies via their base class
    - The orchestrator injects only relevant services during handler initialization
    """

    def __init__(self,
                 calibration_service,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the orchestrator with required services.

        Args:
            calibration_service: Service for calibrating chart axes
            logger: Optional logger for tracking analysis
        """
        self.calibration_service = calibration_service
        self.spatial_classifier = ProductionSpatialClassifier(hyperparams_path=Path('lylaa_hypertuning_results.json'))
        self.logger = logger or logging.getLogger(__name__)

        # Initialize shared services
        self.dual_axis_service = DualAxisDetectionService()
        self.meta_clustering_service = MetaClusteringService()

        # Initialize new services
        self.color_mapping_service = ColorMappingService()  # For heatmaps
        self.legend_matching_service = LegendMatchingService()  # For pie charts

        # Initialize handlers with appropriate services based on their requirements
        self.handlers = self._initialize_handlers()

        self.logger.info("ChartAnalysisOrchestrator initialized with all handlers")

    def _initialize_handlers(self) -> Dict[str, BaseHandler]:
        """Initialize handlers with appropriate services based on their type."""
        handlers = {}

        # Cartesian chart handlers require calibration, spatial classifier, and optional services
        cartesian_params = {
            'calibration_service': self.calibration_service,
            'spatial_classifier': self.spatial_classifier,
            'dual_axis_service': self.dual_axis_service,
            'meta_clustering_service': self.meta_clustering_service,
            'logger': self.logger
        }

        handlers['bar'] = BarHandler(**cartesian_params)
        handlers['scatter'] = ScatterHandler(**cartesian_params)
        handlers['line'] = LineHandler(**cartesian_params)
        handlers['box'] = BoxHandler(**cartesian_params)
        handlers['histogram'] = HistogramHandler(**cartesian_params)

        # Grid chart handlers (e.g., heatmap) require color mapping service
        handlers['heatmap'] = HeatmapHandler(
            color_mapper=self.color_mapping_service,
            logger=self.logger
        )

        # Polar chart handlers (e.g., pie) require legend matching service
        handlers['pie'] = PieHandler(
            legend_matcher=self.legend_matching_service,
            logger=self.logger
        )

        return handlers

    def process_chart(self,
                     image: np.ndarray,
                     chart_type: str,
                     detections: Dict[str, Any],
                     axis_labels: list,
                     chart_elements: list,
                     orientation: str = 'vertical') -> ExtractionResult:
        """
        Process a single chart using the appropriate handler.

        Args:
            image: Input image as numpy array
            chart_type: Type of chart ('bar', 'scatter', 'line', 'box', 'heatmap', 'pie', etc.)
            detections: Detection results from chart detection model
            axis_labels: Extracted axis labels
            chart_elements: Chart elements that need baseline detection
            orientation: Chart orientation ('vertical' or 'horizontal')

        Returns:
            ExtractionResult with processed values and diagnostics
        """
        self.logger.info(f"Processing {chart_type} chart with {orientation} orientation")

        # Validate orientation
        try:
            orientation_enum = OrientationService.from_any(orientation)
        except ValueError as e:
            self.logger.error(f"Invalid orientation '{orientation}': {e}")
            return ExtractionResult(
                chart_type=chart_type,
                coordinate_system=ChartCoordinateSystem.CARTESIAN,  # Default
                elements=[],
                baselines={},
                calibration={},
                diagnostics={'error': f'Invalid orientation: {e}'},
                errors=[f'Orientation validation failed: {e}'],
                warnings=[],
                orientation=Orientation.NOT_APPLICABLE
            )

        # Get appropriate handler
        handler = self.handlers.get(chart_type.lower())
        if not handler:
            error_msg = f"Unsupported chart type: {chart_type}. Supported types: {list(self.handlers.keys())}"
            self.logger.error(error_msg)
            return ExtractionResult(
                chart_type=chart_type,
                coordinate_system=ChartCoordinateSystem.CARTESIAN,  # Default
                elements=[],
                baselines={},
                calibration={},
                diagnostics={'error': error_msg},
                errors=[error_msg],
                warnings=[],
                orientation=orientation_enum
            )

        # Process with handler
        try:
            handler_result = handler.process(
                image=image,
                detections=detections,
                axis_labels=axis_labels,
                chart_elements=chart_elements,
                orientation=orientation_enum
            )
            
            # Handle both old and new result types
            if hasattr(handler_result, 'orientation'):  # New result type
                result = handler_result
            else:  # Old result type, convert to new format
                from handlers.base_handler import ExtractionResult, ChartCoordinateSystem
                result = ExtractionResult(
                    chart_type=handler_result.chart_type,
                    coordinate_system=ChartCoordinateSystem.CARTESIAN,
                    elements=handler_result.elements,
                    baselines=handler_result.baselines,
                    calibration=handler_result.calibration,
                    diagnostics=handler_result.diagnostics,
                    errors=handler_result.errors,
                    warnings=handler_result.warnings,
                    orientation=orientation_enum
                )

            # CRITICAL FIX: Check for errors returned from the handler
            if result.errors:
                self.logger.error(f"Handler for {chart_type} failed with errors: {result.errors}")
                if result.warnings:
                    self.logger.warning(f"Handler for {chart_type} also has warnings: {result.warnings}")
                return result  # Return the failed result

            self.logger.info(f"Successfully processed {chart_type} chart: {len(result.elements)} elements extracted")
            return result

        except Exception as e:
            error_msg = f"Handler processing failed for {chart_type} chart: {e}"
            self.logger.error(error_msg, exc_info=True)
            return ExtractionResult(
                chart_type=chart_type,
                coordinate_system=ChartCoordinateSystem.CARTESIAN,  # Default
                elements=[],
                baselines={},
                calibration={},
                diagnostics={'error': str(e)},
                errors=[error_msg],
                warnings=[],
                orientation=orientation_enum
            )

    def process_image(self,
                     image: np.ndarray,
                     chart_detections: Dict[str, Any],
                     axis_labels: list,
                     chart_elements: list,
                     orientation: str = 'vertical') -> Dict[str, ExtractionResult]:
        """
        Process all chart types in a single image.

        Args:
            image: Input image as numpy array
            chart_detections: All chart detections in the image
            axis_labels: All extracted axis labels
            chart_elements: All chart elements
            orientation: Default orientation ('vertical' or 'horizontal')

        Returns:
            Dictionary mapping chart types to their extraction results
        """
        results = {}

        # Determine which chart types were detected
        detected_types = []
        for chart_type in self.handlers.keys():
            if chart_type in chart_detections and chart_detections[chart_type]:
                detected_types.append(chart_type)

        if not detected_types:
            self.logger.warning("No recognized chart types detected in image")
            return results

        # Process each detected chart type
        for chart_type in detected_types:
            self.logger.info(f"Processing detected {chart_type} chart")

            # Extract chart-specific detections and elements
            chart_specific_detections = {chart_type: chart_detections[chart_type]}
            # Note: In a real implementation, you'd filter chart_elements to those
            # that belong to this specific chart type
            chart_specific_elements = chart_elements

            result = self.process_chart(
                image=image,
                chart_type=chart_type,
                detections=chart_specific_detections,
                axis_labels=axis_labels,
                chart_elements=chart_specific_elements,
                orientation=orientation
            )

            results[chart_type] = result

        return results

    def get_supported_chart_types(self) -> list:
        """Return list of supported chart types."""
        return list(self.handlers.keys())

    def validate_input(self, image, chart_detections, axis_labels, chart_elements) -> list:
        """Validate input parameters and return list of validation errors."""
        errors = []

        if image is None or not hasattr(image, 'shape'):
            errors.append("Image is None or invalid")

        if chart_detections is None:
            errors.append("Chart detections are None")

        if axis_labels is None:
            errors.append("Axis labels are None")

        if chart_elements is None:
            errors.append("Chart elements are None")

        return errors