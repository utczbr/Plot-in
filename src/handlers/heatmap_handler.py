"""
Heatmap handler implementing grid-based chart processing.

This handler processes heatmaps by mapping cell colors to numeric values
using color space analysis and spatial classification.
"""
from typing import List, Dict, Any
import numpy as np
import cv2
from handlers.base_handler import GridChartHandler, ExtractionResult, ChartCoordinateSystem
from services.orientation_service import Orientation
from utils.clustering_utils import cluster_1d_dbscan


class HeatmapHandler(GridChartHandler):
    """Heatmap handler with grid-based coordinate processing."""

    COORDINATE_SYSTEM = ChartCoordinateSystem.GRID

    def __init__(self, classifier=None, **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier

    def get_chart_type(self) -> str:
        return "heatmap"

    def process(
        self,
        image: np.ndarray,
        detections: Dict[str, Any],
        axis_labels: List[Dict],
        chart_elements: List[Dict],
        orientation: Orientation,
        **kwargs
    ) -> ExtractionResult:
        """Process heatmap and extract value matrix."""
        try:
            # Extract heatmap cells from detections
            heatmap_cells = detections.get('heatmap_cell', []) or detections.get('cell', []) or chart_elements
            
            if not heatmap_cells:
                self.logger.warning("No heatmap cells detected")
                return ExtractionResult(
                    chart_type=self.get_chart_type(),
                    coordinate_system=self.get_coordinate_system(),
                    elements=[],
                    orientation=orientation
                )

            # Classify axis labels using specialized classifier if available
            classified_labels = {'x_labels': [], 'y_labels': []}
            if hasattr(self, 'classifier'):
                h, w = image.shape[:2]
                clf_result = self.classifier.classify(
                    axis_labels, heatmap_cells, w, h, orientation
                )
                classified_labels = clf_result.metadata
                # Log classification results
                self.logger.info(f"Heatmap classification: {len(classified_labels.get('x_labels', []))} x-labels, {len(classified_labels.get('y_labels', []))} y-labels")
            
            # --- Color Calibration ---
            if self.color_mapper:
                color_bars = detections.get('color_bar', [])
                if color_bars:
                    self._calibrate_color_mapper(image, color_bars[0], axis_labels)
            
            # --- Dynamic Grid Detection ---
            # 1. Collect all cell centers
            centers = []
            for cell in heatmap_cells:
                bbox = cell['xyxy']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                centers.append({'cx': cx, 'cy': cy, 'cell': cell})
            
            # 2. Cluster to find unique Rows (y) and Cols (x) using DBSCAN
            # §4.3: 2-pass DBSCAN — coarse pass for cell geometry, then geometry-aware eps
            h, w = image.shape[:2]
            cy_vals = [c['cy'] for c in centers]
            cx_vals = [c['cx'] for c in centers]

            # Pass 1: Coarse clustering with legacy eps to estimate cell geometry
            coarse_rows = cluster_1d_dbscan(cy_vals, h * 0.015)
            coarse_cols = cluster_1d_dbscan(cx_vals, w * 0.015)

            # Estimate median cell dimensions from coarse grid
            if len(coarse_rows) >= 2 and len(coarse_cols) >= 2:
                row_diffs = np.diff(sorted(coarse_rows))
                col_diffs = np.diff(sorted(coarse_cols))
                median_cell_h = float(np.median(row_diffs)) if len(row_diffs) > 0 else h * 0.015
                median_cell_w = float(np.median(col_diffs)) if len(col_diffs) > 0 else w * 0.015

                # Pass 2: Re-cluster with geometry-aware eps = 0.5 × cell dimension
                eps_y = median_cell_h * 0.5
                eps_x = median_cell_w * 0.5
                self._row_centers = cluster_1d_dbscan(cy_vals, eps_y)
                self._col_centers = cluster_1d_dbscan(cx_vals, eps_x)
                self.logger.info(
                    f"2-pass DBSCAN: cell geometry {median_cell_w:.0f}×{median_cell_h:.0f}px, "
                    f"eps_x={eps_x:.1f}, eps_y={eps_y:.1f}"
                )
            else:
                # Fallback: use coarse pass results directly
                self._row_centers = coarse_rows
                self._col_centers = coarse_cols

            # Warn if degenerate grid
            if len(self._row_centers) < 2 or len(self._col_centers) < 2:
                self.logger.warning(f"Degenerate grid detected: {len(self._row_centers)} rows x {len(self._col_centers)} cols")
            else:
                self.logger.info(f"Detected Grid: {len(self._row_centers)} rows x {len(self._col_centers)} cols")
            
            # 3. Align Text Labels to Rows/Cols
            row_labels = self._align_labels_to_grid(classified_labels.get('y_labels', []), self._row_centers, is_vertical=True)
            col_labels = self._align_labels_to_grid(classified_labels.get('x_labels', []), self._col_centers, is_vertical=False)

            # Process heatmap cells to extract values
            elements = []
            for cell_data in centers:
                cell = cell_data['cell']
                try:
                    # Assign row/col index by finding closest center
                    row_idx = self._find_closest_index(cell_data['cy'], self._row_centers)
                    col_idx = self._find_closest_index(cell_data['cx'], self._col_centers)
                    
                    value = self._extract_cell_value(image, cell)

                    if value is not None:
                        element = {
                            'type': 'heatmap_cell',
                            'bbox': cell['xyxy'],
                            'value': value,
                            'confidence': cell.get('conf', 1.0),
                            'row': row_idx,
                            'col': col_idx,
                            'row_label': row_labels.get(row_idx, ''),
                            'col_label': col_labels.get(col_idx, '')
                        }
                        # §4.2.4: Surface value_confidence and value_source from color mapper
                        if self.color_mapper:
                            element['value_confidence'] = getattr(self.color_mapper, 'last_confidence', None)
                            element['value_source'] = getattr(self.color_mapper, 'last_value_source', None)
                        elements.append(element)
                except Exception as e:
                    self.logger.warning(f"Error processing heatmap cell: {e}")
                    continue

            # §4.2.6: Count clamped cells for diagnostics (guard against non-numeric)
            clamped_count = sum(
                1 for e in elements
                if isinstance(e.get('value_confidence'), (int, float))
                and e['value_confidence'] < 0.1
            )

            diagnostics = {
                'cell_count': len(heatmap_cells),
                'grid_rows': len(self._row_centers),
                'grid_cols': len(self._col_centers),
            }
            if clamped_count > 0:
                diagnostics['low_confidence_cells'] = clamped_count

            return ExtractionResult(
                chart_type=self.get_chart_type(),
                coordinate_system=self.get_coordinate_system(),
                elements=elements,
                diagnostics=diagnostics,
                orientation=orientation
            )
        except Exception as e:
            self.logger.error(f"Error in HeatmapHandler.process: {e}")
            return ExtractionResult.from_error(self.get_chart_type(), e)

    def _extract_cell_value(self, image: np.ndarray, cell: Dict[str, Any]) -> float:
        """Extract numeric value from heatmap cell based on color."""
        x1, y1, x2, y2 = [int(coord) for coord in cell['xyxy']]
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0  # Invalid bounding box
            
        cell_img = image[y1:y2, x1:x2]
        
        if cell_img.size == 0:
            return 0.0

        # Use color mapping service if available
        if self.color_mapper:
            try:
                return self.color_mapper.map_color_to_value(cell_img)
            except Exception:
                # Fallback to average color analysis
                pass
        
        # Fallback: Use average HSV value for color-to-value mapping
        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        # For heatmaps, typically the V (value) channel or color intensity represents the data value
        avg_hsv = np.mean(hsv, axis=(0, 1))
        
        # Map HSV to a normalized value (0-1 range)
        # This is a simplified approach; in practice, a proper color scale mapping would be used
        intensity = avg_hsv[2] / 255.0  # V channel (brightness)
        
        return float(intensity)

    def _calibrate_color_mapper(self, image: np.ndarray, color_bar: Dict[str, Any], labels: List[Dict]) -> None:
        """
        Calibrate the color mapper using dense 100-point sampling along the color bar.
        
        Strategy:
        1. Extract numeric labels near the color bar and their positions
        2. Sample 100 evenly-spaced pixels along the color bar axis
        3. Interpolate values for each sample based on label positions
        4. If no labels found, fall back to 50 uniform samples with [0, 1] range
        """
        if not self.color_mapper:
            return
            
        bbox = color_bar['xyxy']
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Determine orientation of color bar
        w, h = x2 - x1, y2 - y1
        is_vertical = h > w
        bar_length = h if is_vertical else w
        
        # --- Phase 1: Extract label positions and values ---
        label_anchors = []  # [(position_ratio, value), ...]
        
        for label in labels:
            if not label.get('text'):
                continue
                
            try:
                value = float(label['text'].replace(',', '.'))
            except ValueError:
                continue
            
            l_bbox = label['xyxy']
            l_cx = (l_bbox[0] + l_bbox[2]) / 2
            l_cy = (l_bbox[1] + l_bbox[3]) / 2
            
            # Check proximity to color bar (within 2x bar width/height)
            if is_vertical:
                if x1 - w * 2 < l_cx < x2 + w * 2:
                    # Position ratio along bar (0 = top, 1 = bottom)
                    pos_ratio = (l_cy - y1) / max(h, 1)
                    pos_ratio = max(0.0, min(1.0, pos_ratio))
                    label_anchors.append((pos_ratio, value))
            else:
                if y1 - h * 2 < l_cy < y2 + h * 2:
                    # Position ratio along bar (0 = left, 1 = right)
                    pos_ratio = (l_cx - x1) / max(w, 1)
                    pos_ratio = max(0.0, min(1.0, pos_ratio))
                    label_anchors.append((pos_ratio, value))
        
        # Sort anchors by position
        label_anchors.sort(key=lambda x: x[0])
        
        # --- Phase 2: Dense sampling (100 points) ---
        n_samples = 100
        samples = []
        
        # Center line of the color bar
        bar_cx = int((x1 + x2) / 2)
        bar_cy = int((y1 + y2) / 2)
        
        if label_anchors:
            # We have labels - interpolate values
            min_val = min(a[1] for a in label_anchors)
            max_val = max(a[1] for a in label_anchors)
            
            for i in range(n_samples):
                t = i / (n_samples - 1)  # 0.0 to 1.0
                
                # Sample pixel position
                if is_vertical:
                    s_y = int(y1 + t * (y2 - y1))
                    s_x = bar_cx
                else:
                    s_x = int(x1 + t * (x2 - x1))
                    s_y = bar_cy
                
                # Bounds check
                if not (0 <= s_y < image.shape[0] and 0 <= s_x < image.shape[1]):
                    continue
                
                # Sample 3x3 patch for noise reduction
                patch = image[max(0, s_y-1):min(image.shape[0], s_y+2),
                              max(0, s_x-1):min(image.shape[1], s_x+2)]
                
                if patch.size == 0:
                    continue
                
                # Interpolate value from label anchors
                value = self._interpolate_value(t, label_anchors, min_val, max_val)
                samples.append((patch, value))
            
            self.color_mapper.min_value = min_val
            self.color_mapper.max_value = max_val
            self.color_mapper.value_range = max_val - min_val
            
            self.logger.info(f"Dense calibration: {len(samples)} samples from {len(label_anchors)} labels (range: {min_val:.2f} to {max_val:.2f})")
        
        else:
            # --- Fallback: No labels found - uniform sampling with [0, 1] range ---
            n_fallback = 50
            self.logger.warning(f"No label anchors found, using {n_fallback}-point uniform fallback")
            
            for i in range(n_fallback):
                t = i / (n_fallback - 1)
                
                if is_vertical:
                    s_y = int(y1 + t * (y2 - y1))
                    s_x = bar_cx
                else:
                    s_x = int(x1 + t * (x2 - x1))
                    s_y = bar_cy
                
                if not (0 <= s_y < image.shape[0] and 0 <= s_x < image.shape[1]):
                    continue
                
                patch = image[max(0, s_y-1):min(image.shape[0], s_y+2),
                              max(0, s_x-1):min(image.shape[1], s_x+2)]
                
                if patch.size > 0:
                    # Value proportional to position (0 at start, 1 at end)
                    samples.append((patch, t))
            
            self.color_mapper.min_value = 0.0
            self.color_mapper.max_value = 1.0
            self.color_mapper.value_range = 1.0
        
        # --- Phase 3: Calibrate ---
        if len(samples) >= 2:
            self.color_mapper.calibrate_from_known_values(samples)
        else:
            self.logger.error("Color bar sampling failed - insufficient samples")

    def _interpolate_value(self, t: float, anchors: List[tuple], min_val: float, max_val: float) -> float:
        """
        Interpolate value at position t (0-1) using label anchors.
        
        Uses piecewise linear interpolation between anchor points.
        Extrapolates linearly outside anchor range.
        """
        if not anchors:
            return min_val + t * (max_val - min_val)
        
        if len(anchors) == 1:
            return anchors[0][1]
        
        # Find bracketing anchors
        for i in range(len(anchors) - 1):
            p1, v1 = anchors[i]
            p2, v2 = anchors[i + 1]
            
            if p1 <= t <= p2:
                # Linear interpolation
                if abs(p2 - p1) < 1e-6:
                    return v1
                local_t = (t - p1) / (p2 - p1)
                return v1 + local_t * (v2 - v1)
        
        # Extrapolate
        if t < anchors[0][0]:
            # Before first anchor - extrapolate from first segment
            p1, v1 = anchors[0]
            p2, v2 = anchors[1]
            if abs(p2 - p1) < 1e-6:
                return v1
            slope = (v2 - v1) / (p2 - p1)
            return v1 + slope * (t - p1)
        else:
            # After last anchor - extrapolate from last segment
            p1, v1 = anchors[-2]
            p2, v2 = anchors[-1]
            if abs(p2 - p1) < 1e-6:
                return v2
            slope = (v2 - v1) / (p2 - p1)
            return v2 + slope * (t - p2)

    def _compute_robust_bounds(self, cells: List[Dict]) -> Dict:
        """Compute grid bounds using percentile trimming to exclude outliers."""
        if not cells:
            return {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
            
        x1s = [c['xyxy'][0] for c in cells]
        y1s = [c['xyxy'][1] for c in cells]
        x2s = [c['xyxy'][2] for c in cells]
        y2s = [c['xyxy'][3] for c in cells]
        
        # Use 5th/95th percentile to trim outliers
        return {
            'left': np.percentile(x1s, 5),
            'top': np.percentile(y1s, 5),
            'right': np.percentile(x2s, 95),
            'bottom': np.percentile(y2s, 95)
        }

    def _find_closest_index(self, value: float, centers: List[float]) -> int:
        """Find index of the closest center value."""
        if not centers:
            return 0
        return int(np.argmin([abs(c - value) for c in centers]))

    def _align_labels_to_grid(self, labels: List[Dict], grid_centers: List[float], is_vertical: bool) -> Dict[int, str]:
        """Align text labels to grid row/col indices using IoU projection and Hungarian matching."""
        alignment = {}
        if not labels or not grid_centers:
            return alignment
        
        # Compute spacing between grid lines
        if len(grid_centers) > 1:
            spacing = np.mean(np.diff(grid_centers))
        else:
            spacing = 50  # default
        
        # Compute IoU matrix: labels x grid_indices
        n_labels = len(labels)
        n_grid = len(grid_centers)
        iou_matrix = np.zeros((n_labels, n_grid))
        
        for i, label in enumerate(labels):
            bbox = label['xyxy']
            
            for j, center in enumerate(grid_centers):
                # Create projection band for this grid line
                if is_vertical:
                    # Row projection: horizontal band at y=center
                    band_min = center - spacing / 2
                    band_max = center + spacing / 2
                    label_min = bbox[1]  # y1
                    label_max = bbox[3]  # y2
                else:
                    # Col projection: vertical band at x=center
                    band_min = center - spacing / 2
                    band_max = center + spacing / 2
                    label_min = bbox[0]  # x1
                    label_max = bbox[2]  # x2
                
                # Compute 1D IoU
                intersection = max(0, min(label_max, band_max) - max(label_min, band_min))
                union = (label_max - label_min) + (band_max - band_min) - intersection
                iou = intersection / max(union, 1e-6)
                iou_matrix[i, j] = iou
        
        # Hungarian matching to find optimal assignment
        from scipy.optimize import linear_sum_assignment
        cost_matrix = -iou_matrix  # Negate for minimization
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Assign only if IoU > threshold
        iou_threshold = 0.3
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] > iou_threshold:
                alignment[c] = labels[r].get('text', '')
        
        return alignment

    # extract_values method removed (dead code)