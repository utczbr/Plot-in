"""
Bar chart handler with composition pattern.

This handler composes with ModularBaselineDetector to avoid re-implementing
the 1,500+ lines of baseline detection logic, while providing bar-specific
configuration and value extraction.
"""
from typing import List, Dict, Any
import numpy as np

from handlers.base_handler import BaseChartHandler, ExtractionResult

# NEW: Import BarExtractor to use the new topological association
from extractors.bar_extractor import BarExtractor


class BarHandler(BaseChartHandler):
    """Bar chart handler with composition (NOT re-implementation)."""
    
    def get_chart_type(self) -> str:
        return "bar"
    
    def extract_values(self, img, detections, calibration,
                      baselines, orientation) -> List[Dict]:
        """Extract bar values using baseline and calibration."""
        from extractors.bar_extractor import BarExtractor
        
        bars = detections.get('bar', [])
        if not bars:
            return []
        
        baseline_coord = None
        axis_id = 'y' if orientation == 'vertical' else 'x'
        for baseline in baselines.baselines:
            if baseline.axis_id == axis_id:
                baseline_coord = baseline.value
                break

        cal_model = None
        if 'primary' in calibration and hasattr(calibration['primary'], 'func'):
            cal_model = calibration['primary'].func
        
        if baseline_coord is None or cal_model is None:
            self.logger.warning("Missing baseline or calibration for bar extraction")
            return []
        
        # NEW: Use BarExtractor with axis_labels
        extractor = BarExtractor()
        h, w = img.shape[:2]
        
        # The axis_labels need to be the properly classified tick_labels
        # Get them from the label_classification metadata provided by the orchestrator
        # For now, let's pass the full detections which should contain the necessary metadata
        # We'll extract the classified tick_labels in the extractor
        
        # Try to get the classified tick_labels from metadata if available
        axis_labels = detections.get('axis_labels', [])
        
        extraction_result = extractor.extract(
            img=img,
            detections=detections,
            scale_model=cal_model,
            baseline_coord=baseline_coord,
            img_dimensions={'width': w, 'height': h},
            mode='optimized',
            axis_labels=axis_labels  # Pass axis_labels for extraction
        )
        
        # Return list of bar dictionaries
        return extraction_result.get('bars', [])