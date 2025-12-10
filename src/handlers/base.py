"""
Minimal abstract base class for all chart handlers (new architecture).

This module implements the new clean hierarchy for chart handlers, separating
Cartesian, Polar, and Grid coordinate systems.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import logging

from handlers.types import ChartCoordinateSystem, ExtractionResult
from services.orientation_service import Orientation


class BaseHandler(ABC):
    """
    Minimal abstract base class for all chart handlers (new architecture).

    This class defines the core interface that all handlers must implement,
    while allowing optional injection of chart-type-specific services.

    **Design Philosophy:**
    - Services are injected via constructor but stored as Optional attributes
    - Handlers only use services they actually need
    - Coordinate system is declared at class level for type routing
    """

    # Class-level declaration of coordinate system (override in subclasses)
    COORDINATE_SYSTEM: ChartCoordinateSystem = ChartCoordinateSystem.CARTESIAN

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        # Cartesian-specific services (optional)
        calibration_service: Optional[Any] = None,
        spatial_classifier: Optional[Any] = None,
        dual_axis_service: Optional[Any] = None,
        meta_clustering_service: Optional[Any] = None,
        # Grid-specific services (optional, for heatmaps)
        color_mapper: Optional[Any] = None,
        # Polar-specific services (optional, for pie charts)
        legend_matcher: Optional[Any] = None,
    ):
        """
        Initializes the handler with optional service dependencies.

        Args:
            logger: Logger instance for tracking processing.
            calibration_service: Axis calibration for Cartesian charts.
            spatial_classifier: LYLAA classifier for Cartesian labels.
            dual_axis_service: Dual Y-axis detection for Cartesian charts.
            meta_clustering_service: Baseline clustering for Cartesian charts.
            color_mapper: Color-to-value mapping for heatmaps.
            legend_matcher: Legend-slice association for pie charts.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Store services as optional attributes
        self.calibration_service = calibration_service
        self.spatial_classifier = spatial_classifier
        self.dual_axis_service = dual_axis_service
        self.meta_clustering_service = meta_clustering_service
        self.color_mapper = color_mapper
        self.legend_matcher = legend_matcher

    @abstractmethod
    def process(
        self,
        image: np.ndarray,
        detections: Dict[str, Any],
        axis_labels: List[Dict],
        chart_elements: List[Dict],
        orientation: Orientation,
        **kwargs  # Allow handler-specific parameters
    ) -> ExtractionResult:
        """
        Processes a chart and extracts structured data.

        Args:
            image: Input image as NumPy array (H, W, C) in BGR format.
            detections: Detection results organized by class name.
            axis_labels: List of detected axis label bounding boxes with OCR text.
            chart_elements: Primary chart elements (bars, points, cells, slices).
            orientation: Chart orientation enum.
            **kwargs: Handler-specific optional parameters.

        Returns:
            ExtractionResult containing extracted data and metadata.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement process()")

    def get_coordinate_system(self) -> ChartCoordinateSystem:
        """Returns the coordinate system for this handler."""
        return self.COORDINATE_SYSTEM


class CartesianChartHandler(BaseHandler):
    """
    Base class for Cartesian coordinate system charts (bar, line, scatter, box, histogram).

    This intermediate class enforces that calibration_service and spatial_classifier
    are provided, as they are mandatory for Cartesian charts.
    """

    COORDINATE_SYSTEM = ChartCoordinateSystem.CARTESIAN

    def __init__(
        self,
        calibration_service: Any,  # Now required (not Optional)
        spatial_classifier: Any,   # Now required
        dual_axis_service: Optional[Any] = None,
        meta_clustering_service: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes a Cartesian handler with required services.

        Args:
            calibration_service: Axis calibration engine (REQUIRED).
            spatial_classifier: LYLAA spatial classifier (REQUIRED).
            dual_axis_service: Dual Y-axis detection (optional).
            meta_clustering_service: Baseline clustering (optional).
            logger: Logger instance.

        Raises:
            ValueError: If required services are None.
        """
        if calibration_service is None:
            raise ValueError(f"{self.__class__.__name__} requires calibration_service")
        if spatial_classifier is None:
            raise ValueError(f"{self.__class__.__name__} requires spatial_classifier")

        super().__init__(
            logger=logger,
            calibration_service=calibration_service,
            spatial_classifier=spatial_classifier,
            dual_axis_service=dual_axis_service,
            meta_clustering_service=meta_clustering_service,
        )


class GridChartHandler(BaseHandler):
    """
    Base class for grid-based charts (heatmaps).

    These charts organize data in a 2D grid indexed by categorical X/Y labels.
    """

    COORDINATE_SYSTEM = ChartCoordinateSystem.GRID

    def __init__(
        self,
        color_mapper: Any,  # Required for heatmaps
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes a grid-based chart handler.

        Args:
            color_mapper: Service to map cell colors to numeric values.
            logger: Logger instance.

        Raises:
            ValueError: If color_mapper is None.
        """
        if color_mapper is None:
            raise ValueError(f"{self.__class__.__name__} requires color_mapper")

        super().__init__(logger=logger, color_mapper=color_mapper)


class PolarChartHandler(BaseHandler):
    """
    Base class for polar coordinate charts (pie, radar).

    These charts represent data using angles and radii.
    """

    COORDINATE_SYSTEM = ChartCoordinateSystem.POLAR

    def __init__(
        self,
        legend_matcher: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes a polar chart handler.

        Args:
            legend_matcher: Service to associate slices with legend labels.
            logger: Logger instance.
        """
        super().__init__(logger=logger, legend_matcher=legend_matcher)
