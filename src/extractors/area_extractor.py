"""
Extractor for area charts with AUC computation.

Reuses line chart extraction logic for per-point values, then computes
area under curve (AUC) per series using the trapezoidal rule.
"""
from typing import Dict, List, Optional, Callable
import numpy as np
import logging

from extractors.base_extractor import BaseExtractor
from utils.geometry_utils import find_closest_element


class AreaExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract(self, img, detections, scale_model, baseline_coord,
                img_dimensions, mode='optimized', x_scale_model=None,
                y_baseline_coord=None, x_baseline_coord=None):
        """Extract area chart data points and compute AUC per series."""
        raw_points = detections.get('data_point', [])
        data_points = []
        if isinstance(raw_points, list):
            for point in raw_points:
                if isinstance(point, dict) and 'xyxy' in point:
                    data_points.append(point)
                elif isinstance(point, (list, tuple)) and len(point) >= 4:
                    data_points.append({'xyxy': list(point), 'conf': 1.0})

        data_labels = detections.get('data_label', [])
        error_bars = detections.get('error_bar', [])

        result = self._create_result_template('area', detections, len(data_points))
        r_squared = img_dimensions.get('r_squared', None)
        scale_func = self._resolve_scale_func(scale_model)

        for i, point in enumerate(data_points):
            x1, y1, x2, y2 = point['xyxy']
            y_center = (y1 + y2) / 2
            x_center = (x1 + x2) / 2
            estimated_value = None
            pixel_distance = 0

            if scale_model is not None and baseline_coord is not None:
                pixel_distance = abs(baseline_coord - y_center)
                try:
                    value_at_center = float(scale_func(y_center))
                    value_at_baseline = float(scale_func(baseline_coord))
                    estimated_value = abs(value_at_baseline - value_at_center)
                except Exception as e:
                    logging.warning(f"Scale model failed for area point: {e}")
                    estimated_value = pixel_distance
            else:
                if baseline_coord is not None:
                    pixel_distance = abs(baseline_coord - y_center)
                estimated_value = pixel_distance

            point_info = {
                'index': i,
                'xyxy': point['xyxy'],
                'x_center': x_center,
                'y_center': y_center,
                'confidence': point.get('conf', 0.0),
                'pixel_distance': pixel_distance,
                'estimated_value': estimated_value,
                'data_label': None,
                'error_bar': None,
            }

            assoc_data_label = find_closest_element(point, data_labels, orientation='vertical')
            if assoc_data_label:
                point_info['data_label'] = {
                    'text': assoc_data_label.get('text', ''),
                    'value': assoc_data_label.get('cleanedvalue'),
                    'bbox': assoc_data_label.get('xyxy'),
                }
                if assoc_data_label.get('cleanedvalue') is not None:
                    point_info['estimated_value'] = assoc_data_label['cleanedvalue']

            assoc_error_bar = find_closest_element(point, error_bars, orientation='vertical')
            if assoc_error_bar and scale_model:
                eb_y1, eb_y2 = assoc_error_bar['xyxy'][1], assoc_error_bar['xyxy'][3]
                try:
                    error_margin = abs(float(scale_func(eb_y1)) - float(scale_func(eb_y2)))
                    point_info['error_bar'] = {'margin': error_margin, 'bbox': assoc_error_bar['xyxy']}
                except Exception as e:
                    logging.warning(f"Could not calculate error bar value for area point: {e}")

            result['data_points'].append(point_info)

        # Compute AUC per series
        auc_info = self._compute_auc(result['data_points'], scale_func, baseline_coord)
        result['auc'] = auc_info

        self._add_calibration_info(result, r_squared, baseline_coord)
        return result

    def _compute_auc(
        self,
        data_points: List[Dict],
        scale_func: Optional[Callable],
        baseline_coord: Optional[float],
    ) -> Dict:
        """Compute area under curve using the trapezoidal rule.

        Points are sorted by x_center. The AUC is calculated relative to the
        baseline value using calibrated y-values when available.
        """
        if not data_points or scale_func is None or baseline_coord is None:
            return {'total_auc': None, 'num_points': len(data_points)}

        sorted_points = sorted(data_points, key=lambda p: p.get('x_center', 0))

        x_coords = []
        y_values = []
        baseline_value = None

        try:
            baseline_value = float(scale_func(baseline_coord))
        except Exception:
            baseline_value = 0.0

        for point in sorted_points:
            x_coords.append(point['x_center'])
            val = point.get('estimated_value')
            if val is not None:
                y_values.append(float(val))
            else:
                y_values.append(0.0)

        if len(x_coords) < 2:
            return {
                'total_auc': None,
                'baseline_value': baseline_value,
                'num_points': len(x_coords),
            }

        x_arr = np.array(x_coords)
        y_arr = np.array(y_values)
        total_auc = float(np.trapz(y_arr, x_arr))

        return {
            'total_auc': total_auc,
            'baseline_value': baseline_value,
            'num_points': len(x_coords),
        }
