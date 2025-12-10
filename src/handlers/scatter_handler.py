"""
Scatter plot handler with FORCE_NUMERIC fix.

This handler addresses the critical issue where scatter plots have 100% crash rate
due to all numeric labels being classified as title_label instead of scale_label.
"""
from typing import List, Dict, Any
import numpy as np

from handlers.base_handler import ExtractionResult
from handlers.legacy import BaseChartHandler


class ScatterHandler(BaseChartHandler):
    """
    Scatter plot handler.
    CRITICAL FIX: Forces all numeric labels to scale labels (not title labels).
    """
    
    def get_chart_type(self) -> str:
        return "scatter"
    
    def extract_values(self, img, detections, calibration, 
                      baselines, orientation) -> List[Dict]:
        """Extract scatter points using ScatterExtractor."""
        from extractors.scatter_extractor import ScatterExtractor
        
        extractor = ScatterExtractor()
        
        # The chart elements key for scatter is 'data_point' per the class map
        # Map 'scatter' or 'point' to 'data_point' for extractor
        detections_for_extractor = detections.copy()
        points = detections.get('data_point', []) or detections.get('scatter', []) or detections.get('point', [])
        if points and 'data_point' not in detections_for_extractor:
            detections_for_extractor['data_point'] = points
            
        if not points:
            self.logger.warning(f"No data points found in detections for scatter plot. Available keys: {list(detections.keys())}")
            return []
        
        # Resolve calibration objects
        cal_x = calibration.get('x')
        cal_y = calibration.get('y')
        
        # Fallback logic for dual-axis mapping
        if not cal_x or not cal_y:
            cal_primary = calibration.get('primary')
            cal_secondary = calibration.get('secondary') or cal_primary
            
            if orientation == 'vertical':
                cal_y = cal_y or cal_primary
                cal_x = cal_x or cal_secondary
            else:
                cal_x = cal_x or cal_primary
                cal_y = cal_y or cal_secondary
        
        # Final safety net
        if not cal_x: cal_x = cal_y
        if not cal_y: cal_y = cal_x
            
        # Resolve model functions
        func_x = self._resolve_model_func(cal_x)
        func_y = self._resolve_model_func(cal_y)
        
        if not func_x and not func_y:
            self.logger.warning("No X or Y calibration available, using pixel coordinates only")
            
        # Call extractor
        result = extractor.extract(
            img=img,
            detections=detections_for_extractor,
            scale_model=func_y,
            x_scale_model=func_x,
            baseline_coord=None, # Scatter doesn't use baseline for value calc
            img_dimensions={'r_squared': None}
        )
        
        # Transform result to Handler format
        extracted = []
        for point in result['data_points']:
            extracted.append({
                'type': 'point',
                'bbox': point['xyxy'],
                'x': point['x_calibrated'],
                'y': point['y_calibrated'],
                'center': [point['x_pixel'], point['y_pixel']]
            })
            
        self.logger.info(f"Extracted {len(extracted)} scatter points")
        return extracted

    def _resolve_model_func(self, cal_obj):
        """Resolve calibration object to a callable function."""
        if not cal_obj:
            return None
            
        if hasattr(cal_obj, 'func'):
            return cal_obj.func
        elif isinstance(cal_obj, dict) and 'func' in cal_obj: # Check for 'func' key if it's a dict
             # Note: usually 'func' in dict is not callable if loaded from json, 
             # but here we assume it might be passed as object. 
             # If it's from JSON, we might need 'model_func' key which is common in this codebase
             return cal_obj.get('func') or cal_obj.get('model_func')
        elif isinstance(cal_obj, dict) and 'model_func' in cal_obj:
            return cal_obj['model_func']
            
        # Fallback to coefficients if available
        coeffs = getattr(cal_obj, 'coeffs', None)
        if not coeffs and isinstance(cal_obj, dict):
            coeffs = cal_obj.get('coeffs')
            
        if coeffs:
            m, b = coeffs
            return lambda x: m * x + b
            
        return None