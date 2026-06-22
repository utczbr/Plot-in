"""
Heatmap handler implementing grid-based chart processing.

This handler processes heatmaps by mapping cell colors to numeric values
using color space analysis and spatial classification.

Phase 1-3 optimisations are gated by HeatmapConfig feature flags so that
the default (all flags=False) reproduces the existing 2-pass DBSCAN / HSV
behaviour exactly.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from handlers.base_handler import GridChartHandler, ExtractionResult, ChartCoordinateSystem
from services.orientation_service import Orientation
from utils.clustering_utils import cluster_1d_dbscan
from services.heatmap.config import HeatmapConfig


class HeatmapHandler(GridChartHandler):
    """Heatmap handler with grid-based coordinate processing."""

    COORDINATE_SYSTEM = ChartCoordinateSystem.GRID

    def __init__(
        self,
        classifier=None,
        heatmap_config: Optional[HeatmapConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classifier = classifier
        self.cfg = heatmap_config or HeatmapConfig()

        # Lazily instantiated — created only when the matching flag is True
        self._lattice_detector = None
        self._hybrid_anchor    = None
        self._rectifier        = None
        self._artifact_rejector = None
        self._sequence_aligner = None
        self._label_interpolator = None

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
                    calib_labels = detections.get('color_bar_label', [])
                    if not calib_labels:
                        calib_labels = axis_labels
                    self._calibrate_color_mapper(image, color_bars[0], calib_labels)
            
            # --- Dynamic Grid Detection ---
            # 1. Collect all cell centres
            centers = []
            for cell in heatmap_cells:
                bbox = cell['xyxy']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                centers.append({'cx': cx, 'cy': cy, 'cell': cell})

            # 2. Reconstruct grid (FFT Goertzel path or legacy 2-pass DBSCAN)
            h, w = image.shape[:2]
            grid_diagnostics = self._reconstruct_grid(image, heatmap_cells, centers, h, w, detections)
            
            # 3. Align Text Labels to Rows/Cols
            row_labels = self._align_labels_to_grid(
                classified_labels.get('y_labels', []), self._row_centers, is_vertical=True
            )
            col_labels = self._align_labels_to_grid(
                classified_labels.get('x_labels', []), self._col_centers, is_vertical=False
            )

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
            # Enrich with grid-method and color-bar-type info (Task 4.4)
            diagnostics.update(grid_diagnostics)
            if hasattr(self.color_mapper, 'is_discrete'):
                diagnostics['color_bar_type'] = 'discrete' if self.color_mapper.is_discrete else 'continuous'

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

    def _reconstruct_grid(
        self,
        image: np.ndarray,
        heatmap_cells: List[Dict],
        centers: List[Dict],
        h: int,
        w: int,
        detections: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Determine row_centers and col_centers for the heatmap grid.

        Two modes selected by cfg.use_fft_grid:
          True  → GoertzelLatticeDetector + HybridGridAnchor (Phase 1),
                   with 2-pass DBSCAN as inner fallback.
          False → Legacy 2-pass DBSCAN (unchanged behaviour).

        When a 'chart' detection from the Macro expert model is available,
        it is used as a tighter ROI for FFT analysis (instead of computing
        the union of cell bboxes).

        Returns a diagnostics dict for inclusion in ExtractionResult.
        """
        diagnostics: Dict = {}

        cy_vals = [c['cy'] for c in centers]
        cx_vals = [c['cx'] for c in centers]

        # ── Phase 1: Goertzel + HybridGridAnchor path ────────────────────────
        if self.cfg.use_fft_grid and len(heatmap_cells) >= 3:
            try:
                # Lazy-init detector and anchor
                if self._lattice_detector is None:
                    from services.heatmap.lattice_detector import GoertzelLatticeDetector
                    self._lattice_detector = GoertzelLatticeDetector(
                        num_harmonics=self.cfg.fft_num_harmonics,
                        dc_mask_radius=self.cfg.fft_dc_mask_radius,
                        freq_count=self.cfg.goertzel_freq_count,
                    )
                if self._hybrid_anchor is None:
                    from services.heatmap.hybrid_grid_anchor import HybridGridAnchor
                    self._hybrid_anchor = HybridGridAnchor(
                        confidence_threshold=self.cfg.hybrid_conf_threshold,
                        snap_tolerance_ratio=self.cfg.hybrid_snap_ratio,
                        circular_coherence_min=self.cfg.hybrid_circular_coherence_min,
                    )

                # Determine ROI for FFT analysis.
                # Prefer Macro 'chart' detection (tight heatmap-only crop)
                # over cell-union bbox (may include noise from titles/legend).
                chart_dets = (detections or {}).get('chart', [])
                if chart_dets:
                    chart_bbox = chart_dets[0]['xyxy']
                    roi_x1 = max(0, int(chart_bbox[0]))
                    roi_y1 = max(0, int(chart_bbox[1]))
                    roi_x2 = min(w, int(chart_bbox[2]))
                    roi_y2 = min(h, int(chart_bbox[3]))
                    diagnostics['fft_roi_source'] = 'macro_chart'
                else:
                    # Fallback: union bbox of YOLO cell detections
                    x1s = [c['cell']['xyxy'][0] for c in centers]
                    y1s = [c['cell']['xyxy'][1] for c in centers]
                    x2s = [c['cell']['xyxy'][2] for c in centers]
                    y2s = [c['cell']['xyxy'][3] for c in centers]
                    roi_x1 = max(0, int(min(x1s)))
                    roi_y1 = max(0, int(min(y1s)))
                    roi_x2 = min(w,  int(max(x2s)))
                    roi_y2 = min(h,  int(max(y2s)))
                    diagnostics['fft_roi_source'] = 'cell_union'

                if roi_x2 > roi_x1 + 6 and roi_y2 > roi_y1 + 6:
                    roi_gray = cv2.cvtColor(
                        image[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY
                    )
                    T_x, T_y = self._lattice_detector.extract_rectangular_periods(roi_gray)

                    if T_x is not None and T_y is not None:
                        col_centers, row_centers = self._hybrid_anchor.align_grid_to_detections(
                            yolo_cells=heatmap_cells,
                            T_x=T_x,
                            T_y=T_y,
                            image_shape=(h, w),
                        )

                        if col_centers is not None and row_centers is not None:
                            self._col_centers = col_centers.tolist()
                            self._row_centers = row_centers.tolist()
                            diagnostics.update({
                                'grid_method': 'fft_hybrid',
                                'fft_periods': {'T_x': round(T_x, 2), 'T_y': round(T_y, 2)},
                            })
                            self.logger.info(
                                "FFT Goertzel grid: %d cols × %d rows "
                                "(T_x=%.1fpx T_y=%.1fpx)",
                                len(self._col_centers), len(self._row_centers), T_x, T_y,
                            )
                            return diagnostics
            except Exception as exc:
                self.logger.debug("FFT grid failed — falling back to DBSCAN: %s", exc)

        # ── Legacy fallback: 2-pass DBSCAN ────────────────────────────────────
        coarse_rows = cluster_1d_dbscan(cy_vals, h * 0.015)
        coarse_cols = cluster_1d_dbscan(cx_vals, w * 0.015)

        if len(coarse_rows) >= 2 and len(coarse_cols) >= 2:
            row_diffs = np.diff(sorted(coarse_rows))
            col_diffs = np.diff(sorted(coarse_cols))
            median_cell_h = float(np.median(row_diffs)) if len(row_diffs) > 0 else h * 0.015
            median_cell_w = float(np.median(col_diffs)) if len(col_diffs) > 0 else w * 0.015
            eps_y = median_cell_h * 0.5
            eps_x = median_cell_w * 0.5
            self._row_centers = cluster_1d_dbscan(cy_vals, eps_y)
            self._col_centers = cluster_1d_dbscan(cx_vals, eps_x)
            self.logger.info(
                "2-pass DBSCAN: cell geometry %d×%dpx, eps_x=%.1f, eps_y=%.1f",
                int(median_cell_w), int(median_cell_h), eps_x, eps_y,
            )
        else:
            self._row_centers = coarse_rows
            self._col_centers = coarse_cols

        if len(self._row_centers) < 2 or len(self._col_centers) < 2:
            self.logger.warning(
                "Degenerate grid detected: %d rows × %d cols",
                len(self._row_centers), len(self._col_centers),
            )
        else:
            self.logger.info(
                "Detected Grid: %d rows × %d cols",
                len(self._row_centers), len(self._col_centers),
            )

        diagnostics['grid_method'] = 'dbscan'
        return diagnostics

    def _extract_cell_value(self, image: np.ndarray, cell: Dict[str, Any]) -> float:
        """Extract numeric value from heatmap cell based on color."""
        x1, y1, x2, y2 = [int(coord) for coord in cell['xyxy']]

        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        cell_img = image[y1:y2, x1:x2]
        if cell_img.size == 0:
            return 0.0

        # Phase 2: Artifact rejection before colour averaging
        if self.cfg.use_artifact_rejector and (x2 - x1) > 15 and (y2 - y1) > 15:
            if self._artifact_rejector is None:
                from services.heatmap.artifact_rejector import HeatmapArtifactRejector
                self._artifact_rejector = HeatmapArtifactRejector(
                    d=self.cfg.bilateral_d,
                    sigma_color=self.cfg.bilateral_sigma,
                    sigma_space=self.cfg.bilateral_sigma,
                )
            try:
                cell_img = self._artifact_rejector.process_cell(cell_img)
            except Exception as exc:
                self.logger.debug("ArtifactRejector skipped for cell: %s", exc)

        # Use colour mapping service if available
        if self.color_mapper:
            try:
                return self.color_mapper.map_color_to_value(cell_img)
            except Exception:
                pass

        # Legacy fallback: HSV V-channel brightness
        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        intensity = float(np.mean(hsv[:, :, 2])) / 255.0
        return intensity

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

        # ── DEBUG: Log color bar bbox and sample a few pixels ───────────
        bar_cx_dbg = int((x1 + x2) / 2)
        self.logger.info(
            "CALIB_DEBUG color_bar bbox=[%d,%d,%d,%d] w=%d h=%d is_vert=%s center_x=%d",
            x1, y1, x2, y2, w, h, is_vertical, bar_cx_dbg,
        )
        for t_dbg in (0.0, 0.25, 0.5, 0.75, 1.0):
            if is_vertical:
                sy = int(y1 + t_dbg * (y2 - y1))
                sx = bar_cx_dbg
            else:
                sx = int(x1 + t_dbg * (x2 - x1))
                sy = int((y1 + y2) / 2)
            sy = max(0, min(sy, image.shape[0] - 1))
            sx = max(0, min(sx, image.shape[1] - 1))
            px = image[sy, sx]
            self.logger.info(
                "CALIB_DEBUG sample t=%.2f pos=(%d,%d) BGR=%s", t_dbg, sx, sy, px.tolist()
            )
        # ── END DEBUG ─────────────────────────────────────────────────────
        
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

        # ── DEBUG: Log label anchors ──────────────────────────────────────
        self.logger.info(
            "CALIB_DEBUG label_anchors (%d): %s",
            len(label_anchors),
            [(f"{p:.3f}", f"{v:.2f}") for p, v in label_anchors[:15]],
        )
        # ── END DEBUG ─────────────────────────────────────────────────────
        
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
                # Restrict sampling strictly to the range of known labels to prevent 
                # sampling background pixels if the color bar bounding box is imprecise.
                min_t = label_anchors[0][0]
                max_t = label_anchors[-1][0]
                # If all labels are bunched up, allow a little bit of breathing room, but clamp to 0-1
                if max_t - min_t < 0.1:
                    min_t, max_t = 0.0, 1.0
                    
                t = min_t + (i / (n_samples - 1)) * (max_t - min_t)
                
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

        # ── DEBUG: Log calibration curve quality ──────────────────────────
        if samples:
            bgrs = [np.mean(s[0], axis=(0, 1)) for s in samples]
            bgr_std = np.std(bgrs)
            self.logger.info(
                "CALIB_DEBUG %d samples, BGR std=%.2f, first_bgr=%s, last_bgr=%s",
                len(samples), bgr_std,
                np.mean(samples[0][0], axis=(0, 1)).astype(int).tolist(),
                np.mean(samples[-1][0], axis=(0, 1)).astype(int).tolist(),
            )
            if bgr_std < 5:
                self.logger.warning(
                    "CALIB_DEBUG *** Low color variance (%.2f) — likely sampling background! ***",
                    bgr_std,
                )
        # ── END DEBUG ─────────────────────────────────────────────────────
        
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

    def _align_labels_to_grid(
        self, labels: List[Dict], grid_centers: List[float], is_vertical: bool
    ) -> Dict[int, str]:
        """
        Align text labels to grid row/col indices.

        When cfg.use_nw_aligner=True: BandedGotohAligner (Phase 3).
        Otherwise: legacy IoU + Hungarian matching.

        Label interpolation (cfg.use_label_interpolator=True) fills any
        unmatched slots provided numeric_density > 30% (Patch 5).
        """
        alignment: Dict[int, str] = {}
        if not labels or not grid_centers:
            return alignment

        # ── Legacy IoU + Hungarian (default path) ─────────────────────────────
        spacing = float(np.mean(np.diff(grid_centers))) if len(grid_centers) > 1 else 50.0
        n_labels, n_grid = len(labels), len(grid_centers)
        iou_matrix = np.zeros((n_labels, n_grid))

        for i, label in enumerate(labels):
            bbox = label['xyxy']
            for j, center in enumerate(grid_centers):
                if is_vertical:
                    band_min, band_max = center - spacing / 2, center + spacing / 2
                    label_min, label_max = bbox[1], bbox[3]
                else:
                    band_min, band_max = center - spacing / 2, center + spacing / 2
                    label_min, label_max = bbox[0], bbox[2]
                inter = max(0.0, min(label_max, band_max) - max(label_min, band_min))
                union = (label_max - label_min) + (band_max - band_min) - inter
                iou_matrix[i, j] = inter / max(union, 1e-6)

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        iou_threshold = 0.3
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] > iou_threshold:
                alignment[c] = labels[r].get('text', '')

        # ── Phase 3: Banded Gotoh refinement (optional) ───────────────────────
        if self.cfg.use_nw_aligner:
            try:
                if self._sequence_aligner is None:
                    from services.heatmap.sequence_aligner import BandedGotohAligner
                    self._sequence_aligner = BandedGotohAligner(
                        gap_open=self.cfg.nw_gap_open,
                        gap_extend=self.cfg.nw_gap_extend,
                        max_dist=self.cfg.nw_max_dist,
                        band=self.cfg.nw_band_width,
                    )
                label_positions = np.array([
                    (lb['xyxy'][1] + lb['xyxy'][3]) / 2 if is_vertical
                    else (lb['xyxy'][0] + lb['xyxy'][2]) / 2
                    for lb in labels
                ])
                grid_arr = np.array(grid_centers)
                pairs = self._sequence_aligner.align_sequences(label_positions, grid_arr)
                gotoh_alignment: Dict[int, str] = {}
                for ocr_i, grid_j in pairs:
                    gotoh_alignment[grid_j] = labels[ocr_i].get('text', '')
                # Merge: Gotoh wins over Hungarian on matched positions
                alignment.update(gotoh_alignment)
            except Exception as exc:
                self.logger.debug("BandedGotohAligner failed — keeping Hungarian result: %s", exc)

        # ── Phase 3: Label interpolation (Patch 5: numeric density guard) ─────
        if self.cfg.use_label_interpolator and alignment:
            numeric_vals = []
            for text in alignment.values():
                try:
                    numeric_vals.append(float(str(text).replace(',', '.')))
                except (ValueError, TypeError):
                    pass
            numeric_density = len(numeric_vals) / max(len(alignment), 1)

            if numeric_density > 0.30 and len(numeric_vals) >= 2:
                try:
                    if self._label_interpolator is None:
                        from services.heatmap.label_interpolator import AxisLabelInterpolator
                        self._label_interpolator = AxisLabelInterpolator(
                            variance_tolerance=self.cfg.interp_variance_tol,
                        )
                    valid_idx  = sorted(alignment.keys())
                    valid_vals = []
                    for k in valid_idx:
                        try:
                            valid_vals.append(float(str(alignment[k]).replace(',', '.')))
                        except (ValueError, TypeError):
                            valid_vals.append(float('nan'))

                    # Filter to truly numeric entries
                    clean_idx, clean_vals = [], []
                    for i, v in zip(valid_idx, valid_vals):
                        if not np.isnan(v):
                            clean_idx.append(i)
                            clean_vals.append(v)

                    if len(clean_vals) >= 2:
                        filled = self._label_interpolator.fill_missing_labels(
                            clean_idx, clean_vals, len(grid_centers)
                        )
                        for j, val in enumerate(filled):
                            if j not in alignment:
                                alignment[j] = str(round(val, 6))
                except Exception as exc:
                    self.logger.debug("AxisLabelInterpolator failed: %s", exc)

        return alignment

    # extract_values method removed (dead code)