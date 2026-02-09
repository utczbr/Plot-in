"""
Histogram chart handler with composition pattern.

This handler composes with ModularBaselineDetector to avoid re-implementing
the 1,500+ lines of baseline detection logic, while providing histogram-specific
configuration and value extraction.
"""
from typing import List, Dict, Any

from handlers.base_handler import CartesianExtractionHandler
from services.orientation_service import Orientation, OrientationService
from extractors.histogram_extractor import HistogramExtractor


class HistogramHandler(CartesianExtractionHandler):
    """Histogram chart handler with composition (NOT re-implementation)."""
    
    def get_chart_type(self) -> str:
        return "histogram"
    
    def extract_values(self, img, detections, calibration,
                      baselines, orientation) -> List[Dict]:
        """Extract histogram values using baseline and calibration."""
        from extractors.histogram_extractor import HistogramExtractor

        try:
            orientation_enum = OrientationService.from_any(orientation)
        except ValueError:
            self.logger.warning(
                "Invalid orientation '%s' for histogram extraction. Defaulting to vertical.",
                orientation,
            )
            orientation_enum = Orientation.VERTICAL
        
        # Try multiple possible keys for histogram bins
        bars = detections.get('bar', []) or detections.get('histogram', []) or detections.get('data', [])

        if not bars:
            self.logger.warning(f"No histogram bars found in detections. Available keys: {list(detections.keys())}")
            return []
        
        baseline_coord = None
        axis_id = 'y' if orientation_enum == Orientation.VERTICAL else 'x'
        for baseline in baselines.baselines:
            if baseline.axis_id == axis_id:
                baseline_coord = baseline.value
                break

        cal_model = None
        if 'primary' in calibration and hasattr(calibration['primary'], 'func'):
            cal_model = calibration['primary'].func
        
        # For horizontal histograms, we also need x-scale model
        x_cal_model = None
        if 'x' in calibration and hasattr(calibration['x'], 'func'):
            x_cal_model = calibration['x'].func

        if baseline_coord is None or cal_model is None:
            self.logger.warning("Missing baseline or calibration for histogram extraction")
            return []
        
        extractor = HistogramExtractor()
        h, w = img.shape[:2]

        # Pass the bars to the extractor - they should already be in the detections['bar']
        extraction_result = extractor.extract(
            img=img,
            detections=detections,
            scale_model=cal_model,
            baseline_coord=baseline_coord,
            img_dimensions={'width': w, 'height': h},
            mode='optimized',
            x_scale_model=x_cal_model
            # Note: axis_labels parameter not supported in HistogramExtractor
        )

        # Return list of histogram bin dictionaries
        return extraction_result.get('bars', [])
