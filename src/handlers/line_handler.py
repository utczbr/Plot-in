"""
Line chart handler with type conversion fixes.

This handler addresses the critical issue where line charts crash with
"'list' has no attribute 'values'" due to passing list instead of np.ndarray.
"""
from typing import List, Dict, Any
import numpy as np

from handlers.base_handler import BaseChartHandler, ExtractionResult


class LineHandler(BaseChartHandler):
    """
    Line chart handler.
    CRITICAL FIX: Ensures proper type conversion to avoid "'list' has no attribute 'values'" crashes.
    """
    
    def get_chart_type(self) -> str:
        return "line"
    
    def extract_values(self, img, detections, calibration, 
                      baselines, orientation) -> List[Dict]:
        """Extract line values with proper type handling."""
        """Extract line values using LineExtractor."""
        from extractors.line_extractor import LineExtractor
        
        extractor = LineExtractor()
        
        # Prepare detections dict for extractor
        # LineHandler receives 'line' key but Extractor expects 'data_point'
        # We need to map 'line' to 'data_point' for the extractor
        detections_for_extractor = detections.copy()
        if 'line' in detections:
            detections_for_extractor['data_point'] = detections['line']
            
        # Get baseline coordinate
        baseline_coord = (baselines.get('y', {}).get('coordinate') 
                         if orientation == 'vertical' 
                         else baselines.get('x', {}).get('coordinate'))
        
        # Get scale model
        cal_axis = calibration.get('y' if orientation == 'vertical' else 'x')
        scale_model = cal_axis.get('model_func') if cal_axis else None
        
        if not scale_model:
            self.logger.warning(f"Missing calibration for {orientation} axis in line chart")
            return []

        # Call extractor
        result = extractor.extract(
            img=img,
            detections=detections_for_extractor,
            scale_model=scale_model,
            baseline_coord=baseline_coord,
            img_dimensions={'r_squared': None} # Placeholder
        )
        
        # Transform result to Handler format
        extracted = []
        for point in result['data_points']:
            x1, y1, x2, y2 = point['xyxy']
            
            # Calculate position (center)
            if orientation == 'vertical':
                pos = (y1 + y2) / 2.0
            else:
                pos = (x1 + x2) / 2.0
                
            extracted.append({
                'type': 'line_segment', # Maintain legacy type name if needed, or 'data_point'
                'bbox': [x1, y1, x2, y2],
                'position': pos,
                'value': point['estimated_value'],
                'orientation': orientation,
                'confidence': point.get('confidence', 1.0)
            })
            
        return extracted