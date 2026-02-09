"""
Line chart handler with type conversion fixes.

This handler addresses the critical issue where line charts crash with
"'list' has no attribute 'values'" due to passing list instead of np.ndarray.
"""
from typing import List, Dict, Any
import numpy as np

from handlers.base_handler import CartesianExtractionHandler
from services.orientation_service import Orientation, OrientationService


class LineHandler(CartesianExtractionHandler):
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

        try:
            orientation_enum = OrientationService.from_any(orientation)
        except ValueError:
            self.logger.warning(f"Invalid orientation '{orientation}' for line extraction. Defaulting to vertical.")
            orientation_enum = Orientation.VERTICAL

        axis_key = 'y' if orientation_enum == Orientation.VERTICAL else 'x'

        # Resolve baseline from BaselineResult contract.
        baseline_coord = None
        baseline_lines = getattr(baselines, 'baselines', None)
        if baseline_lines:
            for baseline in baseline_lines:
                if baseline.axis_id in {axis_key, f"{axis_key}1", "primary"}:
                    baseline_coord = baseline.value
                    break
            if baseline_coord is None:
                baseline_coord = baseline_lines[0].value

        # Resolve scale model from standardized calibration contract.
        cal_axis = calibration.get(axis_key) or calibration.get('primary')
        scale_model = None
        r_squared = None
        if cal_axis is not None:
            if hasattr(cal_axis, 'func'):
                scale_model = cal_axis.func
                r_squared = getattr(cal_axis, 'r2', getattr(cal_axis, 'r_squared', None))
            elif isinstance(cal_axis, dict):
                scale_model = cal_axis.get('model_func') or cal_axis.get('func')
                r_squared = cal_axis.get('r2', cal_axis.get('r_squared'))
        
        if not scale_model:
            self.logger.warning(f"Missing calibration for {axis_key} axis in line chart")
            return []

        # Call extractor
        result = extractor.extract(
            img=img,
            detections=detections_for_extractor,
            scale_model=scale_model,
            baseline_coord=baseline_coord,
            img_dimensions={'r_squared': r_squared}
        )
        
        # Transform result to Handler format
        extracted = []
        for point in result['data_points']:
            x1, y1, x2, y2 = point['xyxy']
            
            # Calculate position (center)
            if orientation_enum == Orientation.VERTICAL:
                pos = (y1 + y2) / 2.0
            else:
                pos = (x1 + x2) / 2.0
                
            extracted.append({
                'type': 'line_segment', # Maintain legacy type name if needed, or 'data_point'
                'bbox': [x1, y1, x2, y2],
                'position': pos,
                'value': point['estimated_value'],
                'orientation': orientation_enum.value,
                'confidence': point.get('confidence', 1.0)
            })
            
        return extracted
