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
        """Extract line values using LineSegExtractor (or LineExtractor fallback)."""
        if detections.get('line_series'):
            from extractors.line_seg_extractor import LineSegExtractor
            extractor = LineSegExtractor()
        else:
            from extractors.line_extractor import LineExtractor
            extractor = LineExtractor()
        
        # Prepare detections dict for extractor
        # Map 'line' to 'data_point' if needed for legacy path
        detections_for_extractor = detections.copy()
        if 'line' in detections and 'data_point' not in detections:
            detections_for_extractor['data_point'] = detections['line']

        try:
            orientation_enum = OrientationService.from_any(orientation)
        except ValueError:
            self.logger.warning(f"Invalid orientation '{orientation}' for line extraction. Defaulting to vertical.")
            orientation_enum = Orientation.VERTICAL

        axis_key = 'y' if orientation_enum == Orientation.VERTICAL else 'x'
        x_axis_key = 'x' if orientation_enum == Orientation.VERTICAL else 'y'

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

        # Resolve primary (Y) scale model from standardized calibration contract.
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

        # Resolve secondary (X) scale model from calibration
        x_cal_axis = calibration.get(x_axis_key)
        x_scale_model = None
        if x_cal_axis is not None:
            if hasattr(x_cal_axis, 'func'):
                x_scale_model = x_cal_axis.func
            elif isinstance(x_cal_axis, dict):
                x_scale_model = x_cal_axis.get('model_func') or x_cal_axis.get('func')

        # Call extractor
        result = extractor.extract(
            img=img,
            detections=detections_for_extractor,
            scale_model=scale_model,
            baseline_coord=baseline_coord,
            img_dimensions={'r_squared': r_squared},
            x_scale_model=x_scale_model,
        )
        
        # Transform result to Handler format
        extracted = []
        for point in result.get('data_points', []):
            x1, y1, x2, y2 = point['xyxy']
            
            # Calculate position (center)
            if orientation_enum == Orientation.VERTICAL:
                pos = (y1 + y2) / 2.0
            else:
                pos = (x1 + x2) / 2.0
                
            entry = {
                'type': 'line_segment',
                'bbox': [x1, y1, x2, y2],
                'position': pos,
                'value': point['estimated_value'],
                'x_value': point.get('x_value'),
                'series_id': point.get('series_id', 0),
                'orientation': orientation_enum.value,
                'confidence': point.get('confidence', 1.0),
            }
            if point.get('data_label'):
                entry['data_label'] = point['data_label']
            if point.get('error_bar'):
                entry['error_bar'] = point['error_bar']
            extracted.append(entry)
            
        error_bars = detections.get('error_bar', [])
        for point in extracted:
            px = (point['bbox'][0] + point['bbox'][2]) / 2
            nearest = min(error_bars, key=lambda e: abs((e['xyxy'][0]+e['xyxy'][2])/2 - px), default=None)
            if nearest:
                point['error_bar'] = {'bbox': nearest['xyxy'], 'margin': None}  # calibrate separately
                
        from extractors.legend_associator import LegendAssociator
        return LegendAssociator.associate(extracted, detections)
