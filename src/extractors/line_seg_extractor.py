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
from extractors.series_linker import (
    skeletonize_mask,
    fit_anchor_guided_curve,
    find_skeleton_junctions,
    estimate_endpoint_tangent_pca,
    compute_cielab_color_distance,
)
from utils.geometry_utils import find_closest_element

logger = logging.getLogger(__name__)


class LineSegExtractor(BaseExtractor):
    """Anchor-guided segmentation extractor for line charts."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _assign_markers_to_series(img, marker_pts, series_data, max_dist: float = 15.0) -> Dict[int, List[List[float]]]:
        """Assign each marker to at most one series (nearest skeleton within max_dist).

        Near-ties (< 1 px) are broken by CIELAB distance between the marker pixel
        and the series stroke colour.
        """
        from scipy.spatial import cKDTree

        assigned: Dict[int, List[List[float]]] = {}
        if marker_pts is None or len(marker_pts) == 0 or not series_data:
            return assigned
        trees = [cKDTree(sd['global_skel_pts']) for sd in series_data]

        def _color(pt):
            if img is None or getattr(img, 'size', 0) == 0:
                return None
            h, w = img.shape[:2]
            return img[int(np.clip(pt[1], 0, h - 1)), int(np.clip(pt[0], 0, w - 1))]

        def _stroke_color(sd):
            if img is None or getattr(img, 'size', 0) == 0:
                return None
            h, w = img.shape[:2]
            pts = sd['global_skel_pts']
            mid = pts[len(pts) // 2]
            return img[int(np.clip(mid[1], 0, h - 1)), int(np.clip(mid[0], 0, w - 1))]

        for pt in marker_pts:
            cands = []
            for sd, tree in zip(series_data, trees):
                d, _ = tree.query(pt)
                if d <= max_dist:
                    cands.append((float(d), sd))
            if not cands:
                continue
            cands.sort(key=lambda c: c[0])
            best_d = cands[0][0]
            tied = [c for c in cands if c[0] - best_d < 1.0]
            winner = tied[0][1]
            mc = _color(pt)
            if len(tied) > 1 and mc is not None:
                try:
                    winner = min(
                        tied,
                        key=lambda c: compute_cielab_color_distance(mc, _stroke_color(c[1])),
                    )[1]
                except Exception:
                    pass
            assigned.setdefault(winner['series_idx'], []).append([float(pt[0]), float(pt[1])])
        return assigned

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

        # Pass 1: skeletonize every series so each marker can be owned by exactly one series
        series_data = []
        for series_idx, series_det in enumerate(line_series_list):
            sx1, sy1, sx2, sy2 = series_det['xyxy']
            local_mask = series_det.get('mask')
            if local_mask is None or local_mask.size == 0 or not np.any(local_mask):
                continue
            skel = skeletonize_mask(local_mask)
            skel_y, skel_x = np.where(skel > 0)
            if len(skel_x) == 0:
                continue
            series_data.append({
                'series_idx': series_idx,
                'det': series_det,
                'skel': skel,
                'global_skel_pts': np.column_stack([skel_x + sx1, skel_y + sy1]).astype(float),
            })

        assigned_markers = self._assign_markers_to_series(
            img, marker_pts_array, series_data, max_dist=15.0
        )

        result_data_points = []
        series_curves = []

        for sd in series_data:
            series_idx = sd['series_idx']
            series_det = sd['det']
            skel = sd['skel']
            global_skel_pts = sd['global_skel_pts']
            sx1, sy1, sx2, sy2 = series_det['xyxy']
            conf = float(series_det.get('conf', 1.0))

            junctions, endpoints = find_skeleton_junctions(skel)

            # Resolve multi-segment tracklets when endpoints > 2 (occlusion/branching)
            endpoint_tangents = []
            cielab_color_dist = None
            if len(endpoints) > 2 and len(global_skel_pts) >= 2:
                for ep in endpoints:
                    g_ep = ep.astype(float) + np.array([sx1, sy1], dtype=float)
                    tangent = estimate_endpoint_tangent_pca(global_skel_pts, g_ep)
                    endpoint_tangents.append(tangent.tolist())

                if img is not None and getattr(img, 'size', 0) > 0 and len(endpoints) >= 2:
                    h_img, w_img = img.shape[:2]
                    ep0_y = int(np.clip(endpoints[0][1] + sy1, 0, h_img - 1))
                    ep0_x = int(np.clip(endpoints[0][0] + sx1, 0, w_img - 1))
                    ep1_y = int(np.clip(endpoints[1][1] + sy1, 0, h_img - 1))
                    ep1_x = int(np.clip(endpoints[1][0] + sx1, 0, w_img - 1))
                    cielab_color_dist = compute_cielab_color_distance(
                        img[ep0_y, ep0_x], img[ep1_y, ep1_x]
                    )

            series_det['junction_count'] = int(len(junctions))
            series_det['endpoint_count'] = int(len(endpoints))

            own_markers = assigned_markers.get(series_idx, [])
            own_marker_pts = (
                np.array(own_markers, dtype=float) if own_markers else None
            )

            # 3. Anchor-Guided Spline Curve Fitting (only this series' own markers)
            curve_pts = fit_anchor_guided_curve(
                skeleton_pts=global_skel_pts,
                marker_pts=own_marker_pts,
                max_marker_dist=15.0,
                smoothing=1.0,
                num_samples=max(50, int((sx2 - sx1) / 2)),
            )

            if len(curve_pts) == 0:
                continue
            curve_pts = curve_pts[np.argsort(curve_pts[:, 0])]

            curve_metadata = {
                'series_id': series_idx,
                'confidence': conf,
                'curve_pts': curve_pts,
                'junction_count': int(len(junctions)),
                'endpoint_count': int(len(endpoints)),
            }
            if endpoint_tangents:
                curve_metadata['endpoint_tangents'] = endpoint_tangents
            if cielab_color_dist is not None:
                curve_metadata['cielab_color_dist'] = cielab_color_dist

            series_curves.append(curve_metadata)

            # 4. Generate discrete data points from the continuous curve.
            # Only genuine independent-axis tick labels below the plot are used as X ticks.
            x_tick_centers = []
            for lbl in axis_labels:
                if not isinstance(lbl, dict):
                    continue
                if lbl.get('label_type') in ('scale_label', 'axis_title'):
                    continue
                lx1, ly1, lx2, ly2 = lbl.get('xyxy', [0, 0, 0, 0])
                x_center = (lx1 + lx2) / 2.0
                y_center = (ly1 + ly2) / 2.0
                if sx1 - 10 <= x_center <= sx2 + 10 and y_center >= sy2:
                    x_tick_centers.append(x_center)
            x_tick_centers.sort()

            if own_marker_pts is not None and len(own_marker_pts) >= 2:
                # Real markers are the series' true data points
                sample_pts = own_marker_pts[np.argsort(own_marker_pts[:, 0])]
            else:
                lo, hi = curve_pts[0, 0], curve_pts[-1, 0]
                in_range = [x for x in x_tick_centers if lo <= x <= hi]
                if len(in_range) >= 2:
                    query_x = np.array(in_range)
                    query_y = np.interp(query_x, curve_pts[:, 0], curve_pts[:, 1])
                    sample_pts = np.column_stack([query_x, query_y])
                else:
                    step = max(1, len(curve_pts) // 30)
                    sample_pts = curve_pts[::step]

            for pt_idx, (px, py) in enumerate(sample_pts):
                estimated_val = None
                pixel_dist = 0
                if baseline_coord is not None:
                    pixel_dist = abs(baseline_coord - py)
                if scale_model is not None:
                    try:
                        estimated_val = float(scale_func(py))
                    except Exception:
                        estimated_val = pixel_dist if baseline_coord is not None else py
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
