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

            if scale_model is not None:
                if baseline_coord is not None:
                    pixel_distance = abs(baseline_coord - y_center)
                try:
                    bottom_y = point.get('bottom_y')
                    if bottom_y is not None:
                        # Stacked layer: delta between upper and lower boundary
                        val_top = float(scale_func(y_center))
                        val_bottom = float(scale_func(bottom_y))
                        estimated_value = abs(val_top - val_bottom)
                    else:
                        # Direct coordinate evaluation on continuous axis
                        estimated_value = float(scale_func(y_center))
                except Exception as e:
                    logging.warning(f"Scale model failed for area point: {e}")
                    estimated_value = pixel_distance if baseline_coord is not None else y_center
            else:
                if baseline_coord is not None:
                    pixel_distance = abs(baseline_coord - y_center)
                estimated_value = pixel_distance if baseline_coord is not None else y_center

            point_info = {
                'index': i,
                'xyxy': point['xyxy'],
                'x_center': x_center,
                'y_center': y_center,
                'series_id': point.get('series_id', 0),
                'x_value': float(self._resolve_scale_func(x_scale_model)(x_center)) if x_scale_model is not None else None,
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
        auc_info = self._compute_auc(
            result['data_points'],
            scale_func if scale_model is not None else None,
            baseline_coord,
            x_scale_func=self._resolve_scale_func(x_scale_model) if x_scale_model is not None else None,
        )
        result['auc'] = auc_info

        self._add_calibration_info(result, r_squared, baseline_coord)
        return result

    def _compute_auc(
        self,
        data_points: List[Dict],
        scale_func: Optional[Callable],
        baseline_coord: Optional[float],
        x_scale_func: Optional[Callable] = None,
    ) -> Dict:
        """Compute area under curve using the trapezoidal rule.

        Points are sorted by x_center. The AUC is calculated relative to the
        baseline value using calibrated y-values when available.
        """
        if not data_points or scale_func is None or baseline_coord is None:
            return {'total_auc': None, 'num_points': len(data_points)}

        try:
            baseline_value = float(scale_func(baseline_coord))
        except Exception:
            baseline_value = 0.0

        # Integrate each series separately (never interleave series into one curve)
        by_series: Dict = {}
        for point in data_points:
            by_series.setdefault(point.get('series_id', 0), []).append(point)

        _trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        series_auc: Dict = {}
        total = 0.0
        n_integrated = 0
        for sid, pts in by_series.items():
            pts = sorted(pts, key=lambda p: p.get('x_center', 0))
            if len(pts) < 2:
                continue
            if x_scale_func is not None:
                try:
                    x_arr = np.array([float(x_scale_func(p['x_center'])) for p in pts])
                except Exception:
                    x_arr = np.array([p['x_center'] for p in pts], dtype=float)
            else:
                x_arr = np.array([p['x_center'] for p in pts], dtype=float)
            y_arr = np.array([float(p.get('estimated_value') or 0.0) for p in pts])
            series_auc[sid] = float(_trapezoid(y_arr, x_arr))
            total += series_auc[sid]
            n_integrated += 1

        return {
            'total_auc': total if n_integrated else None,
            'series_auc': series_auc,
            'auc_units': 'value*x' if x_scale_func is not None else 'value*pixel_x',
            'baseline_value': baseline_value,
            'num_points': len(data_points),
        }
