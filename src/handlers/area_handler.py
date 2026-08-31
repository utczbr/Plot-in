"""
Area chart handler.

Extends CartesianExtractionHandler for the shared 7-stage pipeline
(meta-learning clustering, label classification, dual-axis detection,
calibration, baseline detection, value extraction, result formatting).

Uses the line detection model (shared via config) and adds AUC computation
per series relative to the baseline.
"""
from typing import List, Dict

from handlers.base_handler import CartesianExtractionHandler
from services.orientation_service import Orientation, OrientationService


class AreaHandler(CartesianExtractionHandler):
    """Area chart handler with AUC extraction."""

    def get_chart_type(self) -> str:
        return "area"

    def extract_values(self, img, detections, calibration,
                       baselines, orientation) -> List[Dict]:
        """Extract area chart values using AreaSegExtractor (or AreaExtractor fallback)."""
        if detections.get('area_series'):
            from extractors.area_seg_extractor import AreaSegExtractor
            extractor = AreaSegExtractor()
        else:
            from extractors.area_extractor import AreaExtractor
            extractor = AreaExtractor()

        # Map 'area' to 'data_point' if needed for legacy path
        detections_for_extractor = detections.copy()
        if 'area' in detections and 'data_point' not in detections:
            detections_for_extractor['data_point'] = detections['area']

        try:
            orientation_enum = OrientationService.from_any(orientation)
        except ValueError:
            self.logger.warning(
                f"Invalid orientation '{orientation}' for area extraction. "
                "Defaulting to vertical."
            )
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
            self.logger.warning(f"Missing calibration for {axis_key} axis in area chart")
            return []

        # Resolve secondary (X) scale model from calibration
        x_cal_axis = calibration.get(x_axis_key)
        x_scale_model = None
        if x_cal_axis is not None:
            if hasattr(x_cal_axis, 'func'):
                x_scale_model = x_cal_axis.func
            elif isinstance(x_cal_axis, dict):
                x_scale_model = x_cal_axis.get('model_func') or x_cal_axis.get('func')

        result = extractor.extract(
            img=img,
            detections=detections_for_extractor,
            scale_model=scale_model,
            baseline_coord=baseline_coord,
            img_dimensions={'r_squared': r_squared},
            x_scale_model=x_scale_model,
        )

        # Transform result to handler format
        extracted = []
        auc_info = result.get('auc', {})

        for point in result.get('data_points', []):
            x1, y1, x2, y2 = point['xyxy']

            if orientation_enum == Orientation.VERTICAL:
                pos = (y1 + y2) / 2.0
            else:
                pos = (x1 + x2) / 2.0

            entry = {
                'type': 'area_point',
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

        # Append series summary entries with AUC
        if isinstance(auc_info, list):
            for layer_auc in auc_info:
                extracted.append({
                    'type': 'area_series_summary',
                    'series_id': layer_auc.get('series_id', 0),
                    'layer_index': layer_auc.get('layer_index', 0),
                    'auc': layer_auc.get('auc', 0.0),
                    'confidence': layer_auc.get('confidence', 1.0),
                })
        elif isinstance(auc_info, dict) and auc_info.get('total_auc') is not None:
            extracted.append({
                'type': 'area_series_summary',
                'auc': auc_info['total_auc'],
                'baseline_value': auc_info.get('baseline_value'),
                'num_points': auc_info.get('num_points', 0),
            })

        return extracted
