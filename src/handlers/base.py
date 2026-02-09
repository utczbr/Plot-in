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
from services.orientation_service import Orientation, OrientationService
from services.dual_axis_service import DualAxisDecision

# Baseline detector components (compatibility facade path remains stable)
from core.baseline_detection import (
    ModularBaselineDetector,
    DetectorConfig,
    DBSCANClusterer,
    HDBSCANClusterer,
    KMeansGumbelClusterer,
)


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


class CartesianExtractionHandler(CartesianChartHandler, ABC):
    """
    Canonical runtime base for Cartesian extraction handlers.

    Implements the shared 7-stage Cartesian flow formerly hosted in legacy
    base classes, while preserving contracts and diagnostics shape.
    """

    CRITICAL_R2 = 0.85
    FAILURE_R2 = 0.40
    BASELINE_TOLERANCE = 5.0

    def __init__(
        self,
        calibration_service: Any,
        spatial_classifier: Any,
        dual_axis_service: Optional[Any] = None,
        meta_clustering_service: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            calibration_service=calibration_service,
            spatial_classifier=spatial_classifier,
            dual_axis_service=dual_axis_service,
            meta_clustering_service=meta_clustering_service,
            logger=logger,
        )
        # Compatibility aliases used by existing cartesian handlers.
        self.meta_clustering = self.meta_clustering_service
        self.clusterer = None
        self.clustering_recommendation = None
        self.last_image_size = None
        self.last_chart_elements = None
        self._axis_labels: List[Dict[str, Any]] = []

    @abstractmethod
    def get_chart_type(self) -> str:
        """Return the chart type this handler processes."""
        raise NotImplementedError

    @abstractmethod
    def extract_values(
        self,
        img: np.ndarray,
        detections: Dict[str, Any],
        calibration: Dict[str, Any],
        baselines: Any,
        orientation: Orientation,
    ) -> List[Dict[str, Any]]:
        """Extract chart-specific values using calibration and baselines."""
        raise NotImplementedError

    def process(
        self,
        image: np.ndarray,
        detections: Dict[str, Any],
        axis_labels: List[Dict],
        chart_elements: List[Dict],
        orientation: Orientation,
        **kwargs,
    ) -> ExtractionResult:
        """Shared 7-stage Cartesian processing pipeline."""
        h, w = image.shape[:2]
        self.last_image_size = (w, h)
        self.last_chart_elements = chart_elements
        self._axis_labels = axis_labels

        errors: List[str] = []
        warnings: List[str] = []
        diagnostics: Dict[str, Any] = {}

        # Stage 0: Validate orientation
        if not isinstance(orientation, Orientation):
            try:
                orientation = OrientationService.from_any(orientation)
            except ValueError as exc:
                errors.append(f"Orientation validation failed: {exc}")
                return self._fail_result(
                    "Orientation validation",
                    errors,
                    warnings,
                    Orientation.NOT_APPLICABLE,
                )

        # Stage 1: Meta-learning clustering selection
        features = self.meta_clustering.extract_features(
            chart_elements, axis_labels, orientation.value, self.get_chart_type(), (w, h)
        )
        self.clustering_recommendation = self.meta_clustering.recommend(
            features, self.get_chart_type(), orientation.value, (w, h)
        )
        self.logger.info(
            "Clustering: %s (conf=%.2f) - %s",
            self.clustering_recommendation.algorithm,
            self.clustering_recommendation.confidence,
            self.clustering_recommendation.rationale,
        )
        self.clusterer = self._create_clusterer_from_recommendation(
            self.clustering_recommendation, image.shape
        )
        diagnostics["clustering"] = {
            "algorithm": self.clustering_recommendation.algorithm,
            "confidence": self.clustering_recommendation.confidence,
            "rationale": self.clustering_recommendation.rationale,
            "parameters": self.clustering_recommendation.parameters,
        }

        # Stage 2: Label classification
        try:
            classified = self.spatial_classifier.classify(
                chart_type=self.get_chart_type(),
                axis_labels=axis_labels,
                chart_elements=chart_elements,
                img_width=w,
                img_height=h,
                orientation=orientation.value,
            )
            diagnostics["label_classification"] = classified
        except Exception as exc:
            errors.append(f"Label classification failed: {exc}")
            return self._fail_result("Label classification", errors, warnings, orientation)

        # Stage 3: Dual-axis detection
        try:
            dual_decision = self.dual_axis_service.detect(
                axis_labels=classified.scale_labels,
                chart_elements=chart_elements,
                orientation=orientation,
                image_size=(w, h),
            )
            diagnostics["dual_axis"] = dual_decision.__dict__
        except Exception as exc:
            errors.append(f"Dual-axis detection failed: {exc}")
            return self._fail_result("Dual-axis detection", errors, warnings, orientation)

        # Stage 4: Calibration with recovery
        try:
            calibrations = self._calibrate_axes(classified, dual_decision, orientation)
            for axis_id, cal in calibrations.items():
                if cal is None:
                    continue
                r2 = getattr(cal, "r_squared", getattr(cal, "r2", None))
                if r2 is None:
                    continue
                if r2 < self.FAILURE_R2:
                    errors.append(f"{axis_id} calibration catastrophic: R²={r2:.3f}")
                elif r2 < self.CRITICAL_R2:
                    warnings.append(f"{axis_id} calibration quality low: R²={r2:.3f}")
            if errors:
                return self._fail_result("Calibration failure", errors, warnings, orientation)
        except Exception as exc:
            errors.append(f"Calibration failed: {exc}")
            return self._fail_result("Calibration", errors, warnings, orientation)

        # Stage 5: Baseline detection
        try:
            baselines = self._detect_baselines(
                chart_elements, calibrations, image, orientation, dual_decision
            )
        except Exception as exc:
            errors.append(f"Baseline detection failed: {exc}")
            return self._fail_result("Baseline detection", errors, warnings, orientation)

        # Stage 6: Data extraction (chart-specific)
        try:
            elements = self.extract_values(
                image, detections, calibrations, baselines, orientation
            )
        except Exception as exc:
            errors.append(f"Value extraction failed: {exc}")
            return self._fail_result("Value extraction", errors, warnings, orientation)

        # Stage 7: Return standardized result
        return ExtractionResult(
            chart_type=self.get_chart_type(),
            coordinate_system=ChartCoordinateSystem.CARTESIAN,
            elements=elements,
            baselines=baselines,
            calibration=calibrations,
            diagnostics=diagnostics,
            errors=errors,
            warnings=warnings,
            orientation=orientation,
        )

    def _create_clusterer_from_recommendation(self, rec, img_shape):
        """Instantiate clustering backend based on recommendation."""
        if rec.algorithm == "hdbscan":
            try:
                return HDBSCANClusterer(
                    min_cluster_size=rec.parameters["min_cluster_size"],
                    min_samples=rec.parameters.get("min_samples"),
                    metric=rec.parameters.get("metric", "euclidean"),
                )
            except Exception:
                from sklearn.cluster import DBSCAN

                class FallbackClusterer:
                    def __init__(self, **kwargs):
                        self.dbscan = DBSCAN(
                            eps=kwargs.get("min_cluster_size", 5),
                            min_samples=kwargs.get("min_samples", 3),
                        )

                    def fit(self, X):
                        return self.dbscan.fit(X)

                return FallbackClusterer(
                    min_cluster_size=rec.parameters["min_cluster_size"],
                    min_samples=rec.parameters.get("min_samples"),
                )
        if rec.algorithm == "dbscan":
            try:
                return DBSCANClusterer(
                    eps=rec.parameters["eps"],
                    min_samples=rec.parameters["min_samples"],
                    metric=rec.parameters.get("metric", "euclidean"),
                )
            except Exception:
                from sklearn.cluster import DBSCAN

                class FallbackClusterer:
                    def __init__(self, **kwargs):
                        self.dbscan = DBSCAN(
                            eps=kwargs.get("eps", 5),
                            min_samples=kwargs.get("min_samples", 3),
                        )

                    def fit(self, X):
                        return self.dbscan.fit(X)

                return FallbackClusterer(
                    eps=rec.parameters["eps"],
                    min_samples=rec.parameters["min_samples"],
                )

        try:
            return KMeansGumbelClusterer(
                k_range=rec.parameters["k_range"],
                n_init=rec.parameters.get("n_init", 10),
                temperature=rec.parameters.get("temperature", 0.7),
            )
        except Exception:
            from sklearn.cluster import KMeans

            class FallbackClusterer:
                def __init__(self, **kwargs):
                    k_range = kwargs.get("k_range", (2, 10))
                    self.kmeans = KMeans(
                        n_clusters=max(2, min(5, sum(k_range) // 2)),
                        n_init=kwargs.get("n_init", 10),
                    )

                def fit(self, X):
                    return self.kmeans.fit(X)

            return FallbackClusterer(k_range=rec.parameters["k_range"])

    def _calibrate_axes(
        self,
        classified_labels,
        dual_decision: DualAxisDecision,
        orientation: Orientation,
    ) -> Dict[str, Any]:
        """Calibrate primary/secondary axes and expose canonical aliases."""
        calibrations: Dict[str, Any] = {"primary": None, "secondary": None, "x": None, "y": None}
        primary_labels = dual_decision.primary_labels
        axis_type = "y" if orientation == Orientation.VERTICAL else "x"

        cal_primary = self.calibration_service.calibrate(primary_labels, axis_type=axis_type)
        calibrations["primary"] = cal_primary
        calibrations[axis_type] = cal_primary

        if dual_decision.has_dual_axis and dual_decision.secondary_labels:
            try:
                cal_secondary = self.calibration_service.calibrate(
                    dual_decision.secondary_labels, axis_type=axis_type
                )
                calibrations["secondary"] = cal_secondary
            except Exception as exc:
                self.logger.warning("Secondary axis calibration failed: %s", exc)

        return calibrations

    def _detect_baselines(
        self,
        chart_elements: List[Dict],
        calibrations: Dict[str, Any],
        image: np.ndarray,
        orientation: Orientation,
        dual_decision: DualAxisDecision,
    ):
        """Detect baselines through ModularBaselineDetector composition."""
        primary_calibration = calibrations.get("primary")
        primary_calibration_zero = None
        if primary_calibration is not None and hasattr(primary_calibration, "coeffs"):
            if primary_calibration.coeffs and len(primary_calibration.coeffs) >= 2:
                m, b = primary_calibration.coeffs[0], primary_calibration.coeffs[1]
                if abs(m) > 1e-6:
                    primary_calibration_zero = -b / m
                    self.logger.info(
                        "✅ Primary calibration zero-crossing: %.1fpx (R²=%.4f)",
                        primary_calibration_zero,
                        getattr(primary_calibration, "r2", 0.0),
                    )

        config = DetectorConfig(
            cluster_backend=self.clustering_recommendation.algorithm,
            dbscan_eps_px=0.04 * image.shape[0],
            stack_band_frac=0.02,
        )

        if self.get_chart_type() in {"bar", "histogram"}:
            config.dbscan_eps_px = 0.04 * image.shape[0]
        elif self.get_chart_type() in {"scatter", "line"}:
            config.hdbscan_min_cluster_size = 3
            config.hdbscan_min_samples = 1

        detector = ModularBaselineDetector(config=config)
        from core.enums import ChartType

        return detector.detect(
            img=image,
            chart_elements=chart_elements,
            axis_labels=self._axis_labels,
            orientation=orientation,
            chart_type=ChartType(self.get_chart_type()),
            primary_calibration_zero=primary_calibration_zero,
            primary_calibration_result=primary_calibration,
        )

    def _fail_result(
        self,
        stage: str,
        errors: List[str],
        warnings: List[str],
        orientation: Orientation = Orientation.NOT_APPLICABLE,
    ) -> ExtractionResult:
        """Create standardized failure result."""
        return ExtractionResult(
            chart_type=self.get_chart_type(),
            coordinate_system=ChartCoordinateSystem.CARTESIAN,
            elements=[],
            baselines={},
            calibration={},
            diagnostics={"stage_failed": stage},
            errors=errors,
            warnings=warnings,
            orientation=orientation,
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
