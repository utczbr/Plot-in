"""
Chart analysis orchestrator with flexible service injection for diverse chart types.

ARCHITECTURAL CHANGES:
- Handlers are now instantiated with only the services they need
- _initialize_handlers() inspects handler requirements via class hierarchy
- New handlers (heatmap, pie) integrate seamlessly without modifying orchestrator logic
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
import numpy as np

from core.classifiers.production_classifier import ProductionSpatialClassifier
from core.classifiers.heatmap_chart_classifier import HeatmapChartClassifier
from core.classifiers.pie_chart_classifier import PieChartClassifier
from core.chart_registry import normalize_chart_type

from handlers.base_handler import (
    BaseHandler,
    CartesianChartHandler,
    GridChartHandler,
    PolarChartHandler,
    ExtractionResult,
    ChartCoordinateSystem,
)
from handlers.types import HandlerContext

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

    _HANDLER_REGISTRY: Dict[str, Type[BaseHandler]] = {
        "bar": BarHandler,
        "scatter": ScatterHandler,
        "line": LineHandler,
        "box": BoxHandler,
        "histogram": HistogramHandler,
        "heatmap": HeatmapHandler,
        "pie": PieHandler,
    }

    _HANDLER_EXTRAS: Dict[str, Callable[["ChartAnalysisOrchestrator"], Dict[str, Any]]] = {
        "heatmap": lambda self: {
            "classifier": HeatmapChartClassifier(logger=self.logger),
        },
        "pie": lambda self: {
            "classifier": PieChartClassifier(logger=self.logger),
        },
    }

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
        hyperparams_path = Path(__file__).resolve().parent / 'lylaa_hypertuning_results.json'
        self.spatial_classifier = ProductionSpatialClassifier(hyperparams_path=hyperparams_path)
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
        """Initialize handlers using class registry + subclass-based DI."""
        handlers: Dict[str, BaseHandler] = {}
        for chart_type, handler_cls in self._HANDLER_REGISTRY.items():
            handlers[chart_type] = self._build_handler(chart_type, handler_cls)
        return handlers

    def _build_handler(self, chart_type: str, handler_cls: Type[BaseHandler]) -> BaseHandler:
        """Build a handler instance using DI rules based on handler hierarchy."""
        extras_factory = self._HANDLER_EXTRAS.get(chart_type)
        extras = extras_factory(self) if extras_factory else {}

        if issubclass(handler_cls, CartesianChartHandler):
            kwargs: Dict[str, Any] = {
                "calibration_service": self.calibration_service,
                "spatial_classifier": self.spatial_classifier,
                "dual_axis_service": self.dual_axis_service,
                "meta_clustering_service": self.meta_clustering_service,
                "logger": self.logger,
            }
        elif issubclass(handler_cls, GridChartHandler):
            kwargs = {
                "color_mapper": self.color_mapping_service,
                "logger": self.logger,
            }
        elif issubclass(handler_cls, PolarChartHandler):
            kwargs = {
                "legend_matcher": self.legend_matching_service,
                "logger": self.logger,
            }
        else:
            kwargs = {"logger": self.logger}

        kwargs.update(extras)
        return handler_cls(**kwargs)

    def process_chart(self,
                     image: Optional[np.ndarray] = None,
                     chart_type: str = 'bar',
                     detections: Optional[Dict[str, Any]] = None,
                     axis_labels: Optional[list] = None,
                     chart_elements: Optional[list] = None,
                     orientation: Any = 'vertical',
                     context: Optional[HandlerContext] = None) -> ExtractionResult:
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
        if context is not None:
            image = context.image
            chart_type = context.chart_type
            detections = context.detections
            axis_labels = context.axis_labels
            chart_elements = context.chart_elements
            orientation = context.orientation

        if image is None:
            raise ValueError("image is required for process_chart")

        detections = detections or {}
        axis_labels = axis_labels or []
        chart_elements = chart_elements or []
        chart_type = normalize_chart_type(chart_type)

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
            result = handler.process(
                image=image,
                detections=detections,
                axis_labels=axis_labels,
                chart_elements=chart_elements,
                orientation=orientation_enum
            )

            if not isinstance(result, ExtractionResult):
                raise TypeError(
                    f"Handler {handler.__class__.__name__} returned unexpected type: "
                    f"{type(result).__name__}. Expected ExtractionResult."
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
