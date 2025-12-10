"""
Original abstract base with full service composition.

This is maintained for backward compatibility with existing handlers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging

from handlers.types import OldExtractionResult
from services.orientation_service import Orientation, OrientationService
from services.dual_axis_service import DualAxisDetectionService, DualAxisDecision
from services.meta_clustering_service import MetaClusteringService
# ChartType import moved to method level to avoid circular dependency
# from core.enums import ChartType

# Import baseline detection components
try:
    # Try common locations for baseline detection
    from core.baseline_detection import (
        ModularBaselineDetector,
        DetectorConfig,
        DBSCANClusterer,
        HDBSCANClusterer,
        KMeansGumbelClusterer
    )
except ImportError:
    # If baseline detection is in a different location, fallback
    pass

class BaseChartHandler(ABC):
    """
    Original abstract base with full service composition.

    This is maintained for backward compatibility with existing handlers.
    """

    CRITICAL_R2 = 0.85
    FAILURE_R2 = 0.40
    BASELINE_TOLERANCE = 5.0

    def __init__(self,
                 calibration_service,
                 spatial_classifier,
                 dual_axis_service: DualAxisDetectionService,
                 meta_clustering_service: MetaClusteringService,
                 logger: logging.Logger):
        self.calibration_service = calibration_service
        self.spatial_classifier = spatial_classifier
        self.dual_axis_service = dual_axis_service
        self.meta_clustering = meta_clustering_service
        self.logger = logger
        self.clusterer = None
        self.clustering_recommendation = None
        self.last_image_size = None
        self.last_chart_elements = None
        self._axis_labels = []

    @abstractmethod
    def get_chart_type(self) -> str:
        """Return the chart type this handler processes."""
        pass

    @abstractmethod
    def extract_values(self, img, detections, calibration,
                      baselines, orientation) -> List[Dict]:
        """Extract chart-specific values using baseline and calibration."""
        pass

    def process(self, image, detections, axis_labels,
               chart_elements, orientation) -> OldExtractionResult:
        """Seven-stage processing pipeline (old format)."""
        h, w = image.shape[:2]
        self.last_image_size = (w, h)
        self.last_chart_elements = chart_elements
        self._axis_labels = axis_labels

        errors, warnings, diagnostics = [], [], {}

        # Stage 0: Validate orientation
        if not isinstance(orientation, Orientation):
            try:
                orientation = OrientationService.from_any(orientation)
            except ValueError as e:
                errors.append(f"Orientation validation failed: {e}")
                return self._fail_result("Orientation validation", errors, warnings)

        # Stage 1: Meta-learning clustering selection
        features = self.meta_clustering.extract_features(
            chart_elements, axis_labels, orientation.value,
            self.get_chart_type(), (w, h)
        )

        self.clustering_recommendation = self.meta_clustering.recommend(
            features, self.get_chart_type(), orientation.value, (w, h)
        )

        self.logger.info(
            f"Clustering: {self.clustering_recommendation.algorithm} "
            f"(conf={self.clustering_recommendation.confidence:.2f}) - "
            f"{self.clustering_recommendation.rationale}"
        )

        self.clusterer = self._create_clusterer_from_recommendation(
            self.clustering_recommendation, image.shape
        )

        diagnostics['clustering'] = {
            'algorithm': self.clustering_recommendation.algorithm,
            'confidence': self.clustering_recommendation.confidence,
            'rationale': self.clustering_recommendation.rationale,
            'parameters': self.clustering_recommendation.parameters
        }

        # Stage 2: Label classification
        try:
            # from core.classifiers.production_classifier import ProductionSpatialClassifier
            classified = self.spatial_classifier.classify(
                chart_type=self.get_chart_type(),
                axis_labels=axis_labels,
                chart_elements=chart_elements,
                img_width=w,
                img_height=h,
                orientation=orientation.value
            )
            diagnostics['label_classification'] = classified
        except Exception as e:
            errors.append(f"Label classification failed: {e}")
            return self._fail_result("Label classification", errors, warnings)

        # Stage 3: Dual-axis detection (single source of truth)
        try:
            dual_decision = self.dual_axis_service.detect(
                axis_labels=classified.scale_labels,
                chart_elements=chart_elements,
                orientation=orientation,
                image_size=(w, h)
            )
            diagnostics['dual_axis'] = dual_decision.__dict__
        except Exception as e:
            errors.append(f"Dual-axis detection failed: {e}")
            return self._fail_result("Dual-axis detection", errors, warnings)

        # Stage 4: Calibration with recovery
        try:
            calibrations = self._calibrate_axes(classified, dual_decision, orientation)

            for axis_id, cal in calibrations.items():
                if hasattr(cal, 'r_squared') and cal.r_squared < self.FAILURE_R2:
                    errors.append(f"{axis_id} calibration catastrophic: R²={getattr(cal, 'r_squared', 0):.3f}")
                elif hasattr(cal, 'r_squared') and cal.r_squared < self.CRITICAL_R2:
                    warnings.append(f"{axis_id} calibration quality low: R²={getattr(cal, 'r_squared', 0):.3f}")

            if errors:
                return self._fail_result("Calibration failure", errors, warnings)
        except Exception as e:
            errors.append(f"Calibration failed: {e}")
            return self._fail_result("Calibration", errors, warnings)

        # Stage 5: Baseline detection (composition with ModularBaselineDetector)
        try:
            baselines = self._detect_baselines(
                chart_elements, calibrations, image, orientation, dual_decision
            )
        except Exception as e:
            errors.append(f"Baseline detection failed: {e}")
            return self._fail_result("Baseline detection", errors, warnings)

        # Stage 6: Data extraction (chart-specific)
        try:
            elements = self.extract_values(
                image, detections, calibrations, baselines, orientation.value
            )
        except Exception as e:
            errors.append(f"Value extraction failed: {e}")
            return self._fail_result("Value extraction", errors, warnings)

        # Stage 7: Return structured result
        return OldExtractionResult(
            chart_type=self.get_chart_type(),
            orientation=orientation.value,
            elements=elements,
            baselines=baselines,
            calibration=calibrations,
            diagnostics=diagnostics,
            errors=errors,
            warnings=warnings
        )

    def _create_clusterer_from_recommendation(self, rec, img_shape):
        """Factory: instantiate clusterer from baseline_detection.py."""
        h, w = img_shape[:2]

        if rec.algorithm == 'hdbscan':
            try:
                return HDBSCANClusterer(
                    min_cluster_size=rec.parameters['min_cluster_size'],
                    min_samples=rec.parameters.get('min_samples'),
                    metric=rec.parameters.get('metric', 'euclidean')
                )
            except Exception:
                # Fallback clusterer implementation
                from sklearn.cluster import DBSCAN
                class FallbackClusterer:
                    def __init__(self, **kwargs):
                        self.dbscan = DBSCAN(eps=kwargs.get('min_cluster_size', 5), min_samples=kwargs.get('min_samples', 3))
                    def fit(self, X):
                        return self.dbscan.fit(X)
                return FallbackClusterer(min_cluster_size=rec.parameters['min_cluster_size'])
        elif rec.algorithm == 'dbscan':
            try:
                return DBSCANClusterer(
                    eps=rec.parameters['eps'],
                    min_samples=rec.parameters['min_samples'],
                    metric=rec.parameters.get('metric', 'euclidean')
                )
            except Exception:
                # Fallback
                from sklearn.cluster import DBSCAN
                class FallbackClusterer:
                    def __init__(self, **kwargs):
                        self.dbscan = DBSCAN(eps=kwargs.get('eps', 5), min_samples=kwargs.get('min_samples', 3))
                    def fit(self, X):
                        return self.dbscan.fit(X)
                return FallbackClusterer(eps=rec.parameters['eps'])
        else:  # kmeans
            try:
                return KMeansGumbelClusterer(
                    k_range=rec.parameters['k_range'],
                    n_init=rec.parameters.get('n_init', 10),
                    temperature=rec.parameters.get('temperature', 0.7)
                )
            except Exception:
                # Fallback to regular KMeans
                from sklearn.cluster import KMeans
                class FallbackClusterer:
                    def __init__(self, **kwargs):
                        k_range = kwargs.get('k_range', (2, 10))
                        self.kmeans = KMeans(n_clusters=max(2, min(5, sum(k_range)//2)), n_init=kwargs.get('n_init', 10))
                    def fit(self, X):
                        return self.kmeans.fit(X)
                return FallbackClusterer(k_range=rec.parameters['k_range'])

    def _calibrate_axes(self, classified_labels, dual_decision: DualAxisDecision,
                       orientation: Orientation):
        """Calibrate primary and secondary axes if dual-axis detected."""
        calibrations = {}

        # Calibrate primary axis
        primary_labels = dual_decision.primary_labels
        axis_type = 'y' if orientation == Orientation.VERTICAL else 'x'

        try:
            cal_primary = self.calibration_service.calibrate(
                primary_labels,
                axis_type=axis_type
            )
            calibrations['primary'] = cal_primary
        except Exception as e:
            self.logger.error(f"Primary axis calibration failed: {e}")
            raise

        # Calibrate secondary axis if dual-axis
        if dual_decision.has_dual_axis and dual_decision.secondary_labels:
            try:
                cal_secondary = self.calibration_service.calibrate(
                    dual_decision.secondary_labels,
                    axis_type=axis_type
                )
                calibrations['secondary'] = cal_secondary
            except Exception as e:
                self.logger.warning(f"Secondary axis calibration failed: {e}")

        return calibrations

    def _detect_baselines(self, chart_elements, calibrations, image,
                         orientation: Orientation, dual_decision: DualAxisDecision):
        """Detect baselines using ModularBaselineDetector with composition."""

        # DEBUG: Log what we're passing to baseline detection
        self.logger.info(f"_detect_baselines: chart_elements count = {len(chart_elements)}, orientation = {orientation.value}")
        if orientation.value == 'vertical':
            self.logger.info(f"_detect_baselines: sample chart_elements = {chart_elements[:2] if chart_elements else 'None'}")

        # ✅ FIX: Pass the actual CalibrationResult, not the wrapper
        primary_calibration = calibrations.get('primary')

        # Extract zero-crossing from the CalibrationResult
        primary_calibration_zero = None
        if primary_calibration is not None and hasattr(primary_calibration, 'coeffs'):
            if primary_calibration.coeffs and len(primary_calibration.coeffs) >= 2:
                m, b = primary_calibration.coeffs[0], primary_calibration.coeffs[1]
                if abs(m) > 1e-6:  # Avoid division by zero
                    primary_calibration_zero = -b / m
                    self.logger.info(
                        f"✅ Primary calibration zero-crossing: {primary_calibration_zero:.1f}px "
                        f"(R²={getattr(primary_calibration, 'r2', 0):.4f})"
                    )

        # Configure detector
        try:
            config = DetectorConfig(
                cluster_backend=self.clustering_recommendation.algorithm,
                dbscan_eps_px=0.04 * image.shape[0],  # Default for bars
                stack_band_frac=0.02  # Stack-aware aggregation
            )
        except Exception:
            # Fallback config
            class FallbackConfig:
                def __init__(self):
                    self.cluster_backend = 'dbscan'
                    self.dbscan_eps_px = 0.04 * image.shape[0]
                    self.stack_band_frac = 0.02
            config = FallbackConfig()

        # Update config based on chart type
        if self.get_chart_type() in ['bar', 'histogram']:
            config.dbscan_eps_px = 0.04 * image.shape[0]  # 4% of image height for bars
        elif self.get_chart_type() in ['scatter', 'line']:
            try:
                config.hdbscan_min_cluster_size = 3
                config.hdbscan_min_samples = 1
            except Exception:
                pass  # Use defaults

        try:
            detector = ModularBaselineDetector(config=config)
        except Exception:
            # Fallback detector
            class FallbackDetector:
                def detect(self, **kwargs):
                    return {'baselines': []}
            detector = FallbackDetector()

        from core.enums import ChartType
        # ✅ FIX: Pass the actual CalibrationResult, not DetectorCalibrationInput wrapper
        result = detector.detect(
            img=image,
            chart_elements=chart_elements,
            axis_labels=self._axis_labels,
            orientation=orientation,
            chart_type=ChartType(self.get_chart_type()),
            primary_calibration_zero=primary_calibration_zero,  # Zero-crossing coordinate
            primary_calibration_result=primary_calibration  # ✅ Pass the actual CalibrationResult!
        )

        return result

    def _fail_result(self, stage: str, errors: List[str], warnings: List[str]):
        """Create a failure result with error information."""
        return OldExtractionResult(
            chart_type=self.get_chart_type(),
            orientation='unknown',
            elements=[],
            baselines={},
            calibration={},
            diagnostics={'stage_failed': stage},
            errors=errors,
            warnings=warnings
        )
