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

    @staticmethod
    def _independent_tick_centers(axis_labels, layers) -> List[float]:
        """X centres of tick labels lying below every area layer (excludes scale labels/titles)."""
        if not axis_labels or not layers:
            return []
        plot_bottom = max(float(np.max(l['y_bottom'])) for l in layers)
        centers = []
        for lbl in axis_labels:
            if not isinstance(lbl, dict):
                continue
            if lbl.get('label_type') in ('scale_label', 'axis_title'):
                continue
            box = lbl.get('xyxy') or lbl.get('bbox')
            if not box or len(box) < 4:
                continue
            if (box[1] + box[3]) / 2.0 >= plot_bottom - 2:
                centers.append((box[0] + box[2]) / 2.0)
        return sorted(centers)

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

        # Independent-axis tick centres (genuine tick labels below the plot only)
        tick_centers = self._independent_tick_centers(
            detections.get('axis_labels', []), layer_envelopes
        )

        for layer_k, layer in enumerate(layer_envelopes):
            x_cols = layer['x_cols']
            y_top = layer['y_top']
            conf = layer['confidence']

            # Stacked vs overlapping: a stacked layer's lower boundary tracks the
            # top envelope of the layer below it; an overlapping one sits on the baseline.
            is_stacked = False
            if layer_k > 0:
                prev_layer = layer_envelopes[layer_k - 1]
                lo, hi = prev_layer['x_cols'][0], prev_layer['x_cols'][-1]
                in_rng = (x_cols >= lo) & (x_cols <= hi)
                if np.count_nonzero(in_rng) >= 2:
                    prev_top = np.interp(x_cols[in_rng], prev_layer['x_cols'], prev_layer['y_top'])
                    is_stacked = float(np.median(np.abs(layer['y_bottom'][in_rng] - prev_top))) <= 8.0

            if is_stacked:
                dynamic_base_y = np.interp(x_cols, prev_layer['x_cols'], prev_layer['y_top'])
            else:
                base = baseline_coord if baseline_coord is not None else float(np.max(layer['y_bottom']))
                dynamic_base_y = np.full_like(y_top, base, dtype=float)

            # Net pixel thickness per column (pixel-space diagnostic only)
            pixel_thickness = np.maximum(0.0, dynamic_base_y - y_top)

            # Sample at independent-axis ticks when available (1-to-1 with tick labels)
            in_range_ticks = [t for t in tick_centers if x_cols[0] <= t <= x_cols[-1]]
            if len(in_range_ticks) >= 2:
                sample_indices = sorted({int(np.argmin(np.abs(x_cols - t))) for t in in_range_ticks})
            else:
                step = max(1, len(x_cols) // 25)
                sample_indices = list(range(0, len(x_cols), step))

            layer_points = []
            for idx in sample_indices:
                px = float(x_cols[idx])
                py = float(y_top[idx])
                thick_px = float(pixel_thickness[idx])

                estimated_val = None
                if scale_model is not None:
                    try:
                        val_top = float(scale_func(py))
                        if is_stacked:
                            # signed layer delta over the layer below
                            estimated_val = val_top - float(scale_func(dynamic_base_y[idx]))
                        else:
                            # absolute value on the axis (preserves sign and non-zero minima)
                            estimated_val = val_top
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
            auc_units = 'pixel^2'
            try:
                if scale_model is not None:
                    y_real_arr = np.array([
                        float(scale_func(y_top[i])) - float(scale_func(dynamic_base_y[i]))
                        for i in range(len(x_cols))
                    ])
                    if x_scale_func is not None:
                        x_arr = np.array([float(x_scale_func(x)) for x in x_cols])
                        auc_units = 'value*x'
                    else:
                        x_arr = x_cols.astype(float)
                        auc_units = 'value*pixel_x'
                    layer_auc = float(_trapz(y_real_arr, x_arr))
                else:
                    layer_auc = float(_trapz(pixel_thickness, x_cols))
            except Exception as e:
                logger.warning(f"AUC trapezoid integration failed for layer {layer_k}: {e}")
                layer_auc = 0.0

            auc_results.append({
                'series_id': layer['series_id'],
                'layer_index': layer_k,
                'auc': layer_auc,
                'auc_units': auc_units,
                'stacked': is_stacked,
                'confidence': conf,
            })

        result = self._create_result_template('area', detections, len(result_data_points))
        result['data_points'] = result_data_points
        result['auc'] = auc_results

        r_squared = img_dimensions.get('r_squared', None)
        self._add_calibration_info(result, r_squared, baseline_coord)
        return result
