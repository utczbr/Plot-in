"""
Line Segmentation Extractor with Anchor-Guided Spline Reconstruction.

Implements BaseExtractor contract using:
- line_series instance segmentation masks (YOLO-seg)
- data_marker detection centroids as high-weight anchors (YOLO-detect)
- Anchor-guided parametric B-spline reconstruction (splprep)
- Dual-axis PROSAC pixel-to-value calibration mapping
- Categorical X-axis tick snapping to OCR axis_labels
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from extractors.base_extractor import BaseExtractor
from extractors.series_linker import skeletonize_mask, fit_anchor_guided_curve
from utils.geometry_utils import find_closest_element

logger = logging.getLogger(__name__)


class LineSegExtractor(BaseExtractor):
    """Anchor-guided segmentation extractor for line charts."""

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
        """Extracts continuous series curves using instance masks and marker anchors."""
        line_series_list = detections.get('line_series', [])
        marker_list = detections.get('data_marker', [])
        data_labels = detections.get('data_label', [])
        error_bars = detections.get('error_bar', [])
        axis_labels = detections.get('axis_labels', [])

        # Fallback if no line_series masks were detected: try legacy data_point/line keys
        if not line_series_list:
            legacy_points = detections.get('data_point', []) or detections.get('line', [])
            if legacy_points:
                from extractors.line_extractor import LineExtractor
                return LineExtractor().extract(
                    img, detections, scale_model, baseline_coord, img_dimensions,
                    mode=mode, x_scale_model=x_scale_model,
                    y_baseline_coord=y_baseline_coord, x_baseline_coord=x_baseline_coord,
                )

        scale_func = self._resolve_scale_func(scale_model)
        x_scale_func = self._resolve_scale_func(x_scale_model) if x_scale_model is not None else None

        # Extract all marker centroids in global coordinates
        marker_coords = []
        for m in marker_list:
            mx1, my1, mx2, my2 = m['xyxy']
            marker_coords.append([(mx1 + mx2) / 2.0, (my1 + my2) / 2.0])
        marker_pts_array = np.array(marker_coords, dtype=float) if marker_coords else None

        result_data_points = []
        series_curves = []

        for series_idx, series_det in enumerate(line_series_list):
            sx1, sy1, sx2, sy2 = series_det['xyxy']
            local_mask = series_det.get('mask')
            conf = float(series_det.get('conf', 1.0))

            if local_mask is None or local_mask.size == 0 or not np.any(local_mask):
                continue

            # 1. Skeletonize local mask
            skel = skeletonize_mask(local_mask)
            skel_y, skel_x = np.where(skel > 0)
            if len(skel_x) == 0:
                continue

            # 2. Map skeleton coordinates to full image coordinates
            global_skel_pts = np.column_stack([skel_x + sx1, skel_y + sy1]).astype(float)

            # 3. Anchor-Guided Spline Curve Fitting (w=1.0 for skel, w=50.0 for markers)
            curve_pts = fit_anchor_guided_curve(
                skeleton_pts=global_skel_pts,
                marker_pts=marker_pts_array,
                max_marker_dist=15.0,
                smoothing=1.0,
                num_samples=max(50, int((sx2 - sx1) / 2)),
            )

            if len(curve_pts) == 0:
                continue

            series_curves.append({
                'series_id': series_idx,
                'confidence': conf,
                'curve_pts': curve_pts,
            })

            # 4. Generate discrete data points from the continuous curve
            # If categorical X-axis labels exist, snap curve evaluation to label centers
            x_tick_centers = []
            if axis_labels:
                # Find horizontal center of axis labels lying below the chart
                for lbl in axis_labels:
                    lx1, ly1, lx2, ly2 = lbl.get('xyxy', [0, 0, 0, 0])
                    x_center = (lx1 + lx2) / 2.0
                    if sx1 - 10 <= x_center <= sx2 + 10:
                        x_tick_centers.append(x_center)
                x_tick_centers.sort()

            # Determine query X points (snapped to ticks or resampled along curve)
            if len(x_tick_centers) >= 2:
                # Interpolate curve Y at discrete tick centers
                query_x = np.array(x_tick_centers)
                query_y = np.interp(query_x, curve_pts[:, 0], curve_pts[:, 1])
                sample_pts = np.column_stack([query_x, query_y])
            else:
                # Subsample curve points for discrete output representation
                step = max(1, len(curve_pts) // 30)
                sample_pts = curve_pts[::step]

            for pt_idx, (px, py) in enumerate(sample_pts):
                estimated_val = None
                pixel_dist = 0
                if scale_model is not None and baseline_coord is not None:
                    pixel_dist = abs(baseline_coord - py)
                    try:
                        val_center = float(scale_func(py))
                        val_base = float(scale_func(baseline_coord))
                        estimated_val = abs(val_base - val_center)
                    except Exception:
                        estimated_val = pixel_dist
                elif scale_model is not None:
                    try:
                        estimated_val = float(scale_func(py))
                    except Exception:
                        estimated_val = py
                else:
                    estimated_val = py

                real_x_val = None
                if x_scale_func is not None:
                    try:
                        real_x_val = float(x_scale_func(px))
                    except Exception:
                        real_x_val = px

                pt_dict = {
                    'index': len(result_data_points),
                    'series_id': series_idx,
                    'xyxy': [int(px - 2), int(py - 2), int(px + 2), int(py + 2)],
                    'x_center': float(px),
                    'y_center': float(py),
                    'confidence': conf,
                    'pixel_distance': pixel_dist,
                    'estimated_value': estimated_val,
                    'x_value': real_x_val,
                    'data_label': None,
                    'error_bar': None,
                }

                # Associate nearest data label if any
                assoc_label = find_closest_element({'xyxy': pt_dict['xyxy']}, data_labels, orientation='vertical')
                if assoc_label:
                    pt_dict['data_label'] = {
                        'text': assoc_label.get('text', ''),
                        'value': assoc_label.get('cleanedvalue'),
                        'bbox': assoc_label.get('xyxy'),
                    }
                    if assoc_label.get('cleanedvalue') is not None:
                        pt_dict['estimated_value'] = assoc_label['cleanedvalue']

                result_data_points.append(pt_dict)

        result = self._create_result_template('line', detections, len(result_data_points))
        result['data_points'] = result_data_points
        result['series_curves'] = series_curves

        r_squared = img_dimensions.get('r_squared', None)
        self._add_calibration_info(result, r_squared, baseline_coord)
        return result
