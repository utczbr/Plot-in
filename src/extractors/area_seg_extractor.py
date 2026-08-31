"""
Area Segmentation Extractor with Topological Sorting and Dynamic Envelope Subtraction.

Implements BaseExtractor contract using:
- area_series instance segmentation masks (YOLO-seg)
- Upper & lower boundary envelope decomposition per instance polygon
- Topological vertical sorting for stacked & overlapping area layers
- Dynamic lower-envelope baseline subtraction
- Calibrated Area Under Curve (AUC) integration via cross-version trapezoidal rule
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from extractors.base_extractor import BaseExtractor
from utils.geometry_utils import find_closest_element

logger = logging.getLogger(__name__)

# Cross-version numpy trapezoidal integration alias (NumPy 1.x / 2.x safe)
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


class AreaSegExtractor(BaseExtractor):
    """Segmentation extractor for stacked and standard area charts."""

    def __init__(self):
        super().__init__()

    def extract(
        self,
        img: np.ndarray,
        detections: Dict[str, Any],
        scale_model: Any,
        baseline_coord: Optional[float],
        img_dimensions: Dict[str, Any],
        mode: str = 'optimized',
        x_scale_model: Optional[Any] = None,
        y_baseline_coord: Optional[float] = None,
        x_baseline_coord: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Extracts area chart data points and computes AUC per series."""
        area_series_list = detections.get('area_series', [])
        data_labels = detections.get('data_label', [])
        error_bars = detections.get('error_bar', [])

        # Fallback if no area_series masks were detected: try legacy data_point/area keys
        if not area_series_list:
            legacy_points = detections.get('data_point', []) or detections.get('area', [])
            if legacy_points:
                from extractors.area_extractor import AreaExtractor
                return AreaExtractor().extract(
                    img, detections, scale_model, baseline_coord, img_dimensions,
                    mode=mode, x_scale_model=x_scale_model,
                    y_baseline_coord=y_baseline_coord, x_baseline_coord=x_baseline_coord,
                )

        scale_func = self._resolve_scale_func(scale_model)
        x_scale_func = self._resolve_scale_func(x_scale_model) if x_scale_model is not None else None

        # 1. Decompose each mask into upper and lower envelopes
        layer_envelopes = []
        for series_idx, series_det in enumerate(area_series_list):
            sx1, sy1, sx2, sy2 = series_det['xyxy']
            local_mask = series_det.get('mask')
            conf = float(series_det.get('conf', 1.0))

            if local_mask is None or local_mask.size == 0 or not np.any(local_mask):
                continue

            h_mask, w_mask = local_mask.shape[:2]
            cols = []
            y_top_list = []
            y_bottom_list = []

            for cx in range(w_mask):
                col_pixels = np.where(local_mask[:, cx] > 0)[0]
                if len(col_pixels) == 0:
                    continue
                gx = sx1 + cx
                top_y = sy1 + float(col_pixels.min())
                bottom_y = sy1 + float(col_pixels.max())
                cols.append(gx)
                y_top_list.append(top_y)
                y_bottom_list.append(bottom_y)

            if len(cols) < 2:
                continue

            mean_y = float(np.mean(y_top_list))
            layer_envelopes.append({
                'series_id': series_idx,
                'confidence': conf,
                'x_cols': np.array(cols),
                'y_top': np.array(y_top_list),
                'y_bottom': np.array(y_bottom_list),
                'mean_y': mean_y,
                'bbox': [sx1, sy1, sx2, sy2],
            })

        # 2. Sort layers vertically from bottom to top (in image pixel space, larger Y is lower on screen)
        layer_envelopes.sort(key=lambda item: item['mean_y'], reverse=True)

        result_data_points = []
        auc_results = []

        for layer_k, layer in enumerate(layer_envelopes):
            x_cols = layer['x_cols']
            y_top = layer['y_top']
            conf = layer['confidence']

            # Dynamic baseline: layer below it or chart baseline
            if layer_k == 0:
                # Bottom-most layer: baseline is the detected chart baseline
                dynamic_base_y = np.full_like(y_top, baseline_coord if baseline_coord is not None else float(np.max(layer['y_bottom'])))
            else:
                # Stacked layer: dynamic baseline is the top envelope of layer (k-1) below it
                prev_layer = layer_envelopes[layer_k - 1]
                dynamic_base_y = np.interp(x_cols, prev_layer['x_cols'], prev_layer['y_top'])

            # Net pixel thickness per column
            pixel_thickness = np.maximum(0.0, dynamic_base_y - y_top)

            # Sample discrete points along envelope for output representation
            step = max(1, len(x_cols) // 25)
            sample_indices = range(0, len(x_cols), step)

            layer_points = []
            for idx in sample_indices:
                px = float(x_cols[idx])
                py = float(y_top[idx])
                thick_px = float(pixel_thickness[idx])

                estimated_val = None
                if scale_model is not None:
                    try:
                        val_top = float(scale_func(py))
                        val_base = float(scale_func(dynamic_base_y[idx]))
                        estimated_val = abs(val_base - val_top)
                    except Exception:
                        estimated_val = thick_px
                else:
                    estimated_val = thick_px

                real_x = None
                if x_scale_func is not None:
                    try:
                        real_x = float(x_scale_func(px))
                    except Exception:
                        real_x = px

                pt_info = {
                    'index': len(result_data_points),
                    'series_id': layer['series_id'],
                    'xyxy': [int(px - 2), int(py - 2), int(px + 2), int(py + 2)],
                    'x_center': px,
                    'y_center': py,
                    'confidence': conf,
                    'pixel_distance': thick_px,
                    'estimated_value': estimated_val,
                    'x_value': real_x,
                    'data_label': None,
                    'error_bar': None,
                }

                assoc_label = find_closest_element({'xyxy': pt_info['xyxy']}, data_labels, orientation='vertical')
                if assoc_label:
                    pt_info['data_label'] = {
                        'text': assoc_label.get('text', ''),
                        'value': assoc_label.get('cleanedvalue'),
                        'bbox': assoc_label.get('xyxy'),
                    }
                    if assoc_label.get('cleanedvalue') is not None:
                        pt_info['estimated_value'] = assoc_label['cleanedvalue']

                result_data_points.append(pt_info)
                layer_points.append(pt_info)

            # 3. Compute AUC using cross-version trapezoidal integration
            try:
                if x_scale_func is not None and scale_model is not None:
                    x_real_arr = np.array([float(x_scale_func(x)) for x in x_cols])
                    y_real_arr = np.array([abs(float(scale_func(dynamic_base_y[i])) - float(scale_func(y_top[i]))) for i in range(len(x_cols))])
                    layer_auc = float(_trapz(y_real_arr, x_real_arr))
                else:
                    layer_auc = float(_trapz(pixel_thickness, x_cols))
            except Exception as e:
                logger.warning(f"AUC trapezoid integration failed for layer {layer_k}: {e}")
                layer_auc = 0.0

            auc_results.append({
                'series_id': layer['series_id'],
                'layer_index': layer_k,
                'auc': layer_auc,
                'confidence': conf,
            })

        result = self._create_result_template('area', detections, len(result_data_points))
        result['data_points'] = result_data_points
        result['auc'] = auc_results

        r_squared = img_dimensions.get('r_squared', None)
        self._add_calibration_info(result, r_squared, baseline_coord)
        return result
