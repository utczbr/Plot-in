"""
Box plot handler with complete extraction implementation.

This handler addresses the issue where box plots have 0% extraction rate
due to missing append in BoxExtractor.
"""
from typing import List, Dict, Any
import numpy as np

from handlers.base_handler import ExtractionResult, ChartCoordinateSystem
from handlers.legacy import BaseChartHandler
from extractors.box_extractor import BoxExtractor, group_box_plot_elements
from services.meta_clustering_service import ClusteringRecommendation


class BoxHandler(BaseChartHandler):
    """
    Box plot handler.
    CRITICAL FIX: Complete implementation of box extraction to address 0% extraction rate.
    """
    
    def get_chart_type(self) -> str:
        return "box"
    
    def extract_values(self, img, detections, calibration, 
                      baselines, orientation) -> List[Dict]:
        """Extract box plot values using the dedicated BoxExtractor."""
        
        # The calibration result is stored under the 'primary' key by the BaseChartHandler
        calibration_result = calibration.get('primary')

        if not calibration_result or not hasattr(calibration_result, 'func'):
            self.logger.warning(f"Missing or invalid primary calibration for box plot.")
            return []

        # The BoxExtractor now expects the full CalibrationResult object to handle coordinate system metadata
        scale_model = calibration_result  # Pass the full CalibrationResult object
        
        # The 'baselines' object is a BaselineResult instance.
        # We need to access the 'value' from the first BaselineLine in its 'baselines' list.
        baseline_coord = None
        if baselines and baselines.baselines:
            baseline_coord = baselines.baselines[0].value
        
        # Image dimensions and R² for quality reporting
        img_height, img_width = img.shape[:2]
        img_dimensions = {
            'height': img_height,
            'width': img_width,
            'r_squared': calibration_result.r2 if hasattr(calibration_result, 'r2') else None
        }

        # Instantiate the extractor and perform extraction
        extractor = BoxExtractor()
        extraction_data = extractor.extract(
            img=img,
            detections=detections,
            scale_model=scale_model,
            baseline_coord=baseline_coord,
            img_dimensions=img_dimensions,
            mode='optimized', # or other modes as needed
            axis_labels=self._axis_labels  # Pass stored axis labels for tick label grouping
        )

        # The extractor returns a dictionary, we need to return the list of boxes
        return extraction_data.get('boxes', [])
    
    def process(self, image, detections, axis_labels,
               chart_elements, orientation) -> ExtractionResult:
        """Override process() to use custom grouping for box plots."""
        h, w = image.shape[:2]
        self.last_image_size = (w, h)
        self.last_chart_elements = chart_elements
        self._axis_labels = axis_labels

        errors, warnings, diagnostics = [], [], {}

        # Stage 0: Validate orientation
        from services.orientation_service import Orientation
        if not isinstance(orientation, Orientation):
            try:
                from services.orientation_service import OrientationService
                orientation = OrientationService.from_any(orientation)
            except ValueError as e:
                errors.append(f"Orientation validation failed: {e}")
                return self._fail_result_new("Orientation validation", errors, warnings, orientation)

        # Stage 1: Special handling for box plots - skip generic clustering
        # Check meta_clustering to see if it recommends intersection alignment
        features = self.meta_clustering.extract_features(
            chart_elements, axis_labels, orientation.value,
            self.get_chart_type(), (w, h)
        )

        self.clustering_recommendation = self.meta_clustering.recommend(
            features, self.get_chart_type(), orientation.value, (w, h)
        )

        # If using intersection_alignment algorithm, set up diagnostics without clusterer
        if self.clustering_recommendation.algorithm == 'intersection_alignment':
            self.logger.info(
                f"Box plot topology-aware grouping: {self.clustering_recommendation.algorithm} "
                f"(conf={self.clustering_recommendation.confidence:.2f}) - "
                f"{self.clustering_recommendation.rationale}"
            )

            diagnostics['clustering'] = {
                'algorithm': self.clustering_recommendation.algorithm,
                'confidence': self.clustering_recommendation.confidence,
                'rationale': self.clustering_recommendation.rationale,
                'parameters': self.clustering_recommendation.parameters
            }
        else:
            # For any other algorithm, use the standard approach
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
            return self._fail_result_new("Label classification", errors, warnings, orientation)

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
            return self._fail_result_new("Dual-axis detection", errors, warnings, orientation)

        # Stage 4: Calibration with recovery
        try:
            calibrations = self._calibrate_axes(classified, dual_decision, orientation)

            for axis_id, cal in calibrations.items():
                if hasattr(cal, 'r_squared') and cal.r_squared < self.FAILURE_R2:
                    errors.append(f"{axis_id} calibration catastrophic: R²={getattr(cal, 'r_squared', 0):.3f}")
                elif hasattr(cal, 'r_squared') and cal.r_squared < self.CRITICAL_R2:
                    warnings.append(f"{axis_id} calibration quality low: R²={getattr(cal, 'r_squared', 0):.3f}")

            if errors:
                return self._fail_result_new("Calibration failure", errors, warnings, orientation)
        except Exception as e:
            errors.append(f"Calibration failed: {e}")
            return self._fail_result_new("Calibration", errors, warnings, orientation)

        # Stage 5: Baseline detection (composition with ModularBaselineDetector)
        try:
            baselines = self._detect_baselines(
                chart_elements, calibrations, image, orientation, dual_decision
            )
        except Exception as e:
            errors.append(f"Baseline detection failed: {e}")
            return self._fail_result_new("Baseline detection", errors, warnings, orientation)

        # Stage 6: Data extraction (chart-specific)
        try:
            elements = self.extract_values(
                image, detections, calibrations, baselines, orientation.value
            )
        except Exception as e:
            errors.append(f"Value extraction failed: {e}")
            return self._fail_result_new("Value extraction", errors, warnings, orientation)

        # Stage 7: Return structured result with NEW format
        return ExtractionResult(
            chart_type=self.get_chart_type(),
            coordinate_system=ChartCoordinateSystem.CARTESIAN,  # ✅ NEW: Required parameter
            elements=elements,
            baselines=baselines,
            calibration=calibrations,
            diagnostics=diagnostics,
            errors=errors,
            warnings=warnings,
            orientation=orientation  # ✅ Use orientation enum object, not string
        )

    def _fail_result_new(self, stage: str, errors: List[str], warnings: List[str], orientation):
        """Create a failure result in the NEW ExtractionResult format."""
        from handlers.base_handler import ChartCoordinateSystem
        from services.orientation_service import Orientation

        # Convert string orientation to enum if needed
        if isinstance(orientation, str):
            from services.orientation_service import OrientationService
            try:
                orientation_enum = OrientationService.from_any(orientation)
            except:
                orientation_enum = Orientation.VERTICAL  # Default
        else:
            orientation_enum = orientation

        return ExtractionResult(
            chart_type=self.get_chart_type(),
            coordinate_system=ChartCoordinateSystem.CARTESIAN,  # ✅ NEW: Required parameter
            elements=[],
            baselines={},
            calibration={},
            diagnostics={'stage_failed': stage},
            errors=errors,
            warnings=warnings,
            orientation=orientation_enum
        )