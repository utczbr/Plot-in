import numpy as np
from typing import Dict, List, Optional
import logging

from .base_classifier import BaseChartClassifier, ClassificationResult
from services.orientation_service import Orientation
from utils.clustering_utils import cluster_1d_dbscan

class HeatmapChartClassifier(BaseChartClassifier):
    """
    Specialized classifier for heatmaps that leverages grid structure for label identification.
    
    It classifies labels into:
    - scale_label (x-axis labels and y-axis labels)
    - axis_title (titles for x/y axes)
    - tick_label (rarely used in heatmaps, usually same as scale labels)
    
    And distinctly separates x-axis vs y-axis candidates for downstream alignment.
    """
    
    def __init__(self, params: Dict = None, logger: logging.Logger = None):
        super().__init__(params or self.get_default_params(), logger)
    
    @classmethod
    def get_default_params(cls) -> Dict:
        return {
            # Heatmap specific weights
            'grid_alignment_weight': 5.0,  # Specific to heatmaps
            'relational_weight': 4.0,      # Above/Left/Right/Bottom of grid
            
            # Standard params but tuned for dense grids
            'scale_size_max_width': 0.15,  # Labels can be wide (e.g. dates)
            'scale_size_max_height': 0.08,
            
            # Gaussian weights (Phase 14 style)
            'gaussian_kernel_weight': 1.0,
            'gaussian_sigma': 0.05,        # Tighter sigma for tight grid alignment
            'left_axis_weight': 6.0,
            'right_axis_weight': 3.0,
            'bottom_axis_weight': 6.0,
            'top_axis_weight': 4.0,       # Sometimes x-axis is on top
            'top_title_weight': 4.0,
            'center_plot_weight': 3.0,     # Heavily penalized
            
            # Thresholds
            'classification_threshold': 1.8,
            'edge_threshold': 0.25
        }
    
    def classify(
        self,
        axis_labels: List[Dict],
        chart_elements: List[Dict], # These are heatmap_cells
        img_width: int,
        img_height: int,
        orientation: Orientation = Orientation.VERTICAL
    ) -> ClassificationResult:
        """
        Classify labels based on their relationship to the heatmap grid.
        """
        if not self.validate_inputs(axis_labels, chart_elements):
            return self._empty_result()
            
        # 1. Derive Grid Structure
        grid_context = self._derive_grid_structure(chart_elements, img_width, img_height)
        
        classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
        all_scores = []
        
        # 2. Score each label
        for label in axis_labels:
            # Basic features
            features = self._extract_features(label, img_width, img_height)
            
            # Compute scores
            scores = self._compute_heatmap_scores(features, grid_context, orientation)
            all_scores.append(scores)
            
            # Decision
            best_class = max(scores, key=scores.get)
            
            # Thresholding
            if scores[best_class] > self.params['classification_threshold']:
                classified[best_class].append(label)
            else:
                # Fallback based on simple geometry
                self._apply_fallback(label, classified, grid_context)
                
        # 3. Post-processing / Separation (Metadata)
        # We store the separated x/y labels in metadata for the handler to use
        x_labels, y_labels = self._separate_xy_labels(classified['scale_label'], grid_context)
        
        metadata = {
            'chart_type': 'heatmap',
            'grid_rows': len(grid_context.get('row_centers', [])),
            'grid_cols': len(grid_context.get('col_centers', [])),
            'x_labels': x_labels,
            'y_labels': y_labels
        }
        
        confidence = self.compute_confidence(classified, all_scores)
        
        return ClassificationResult(
            scale_labels=classified['scale_label'],
            tick_labels=classified['tick_label'],
            axis_titles=classified['axis_title'],
            confidence=confidence,
            metadata=metadata
        )

    def _derive_grid_structure(self, cells: List[Dict], w: int, h: int) -> Dict:
        """Analyze cells to find grid boundaries and structure."""
        if not cells:
            return {'exists': False, 'bounds': (0,0,w,h)}
            
        # Collect centers
        cw, ch = [], []
        x1s, y1s, x2s, y2s = [], [], [], []
        
        for c in cells:
            bbox = c['xyxy']
            x1s.append(bbox[0])
            y1s.append(bbox[1])
            x2s.append(bbox[2])
            y2s.append(bbox[3])
            cw.append((bbox[0] + bbox[2])/2)
            ch.append((bbox[1] + bbox[3])/2)
            
        bounds = {
            'left': min(x1s),
            'top': min(y1s),
            'right': max(x2s),
            'bottom': max(y2s)
        }
        
        # Simple Clustering for unique rows/cols (1D clustering)
        # tolerance=h*0.02 is equivalent to eps since cluster_1d_dbscan uses tolerance directly as eps
        # The original code used eps = tolerance * 0.75.
        # Let's preserve the exact math:
        row_eps = (h*0.02) * 0.75
        col_eps = (w*0.02) * 0.75
        
        row_centers = cluster_1d_dbscan(ch, row_eps)
        col_centers = cluster_1d_dbscan(cw, col_eps)
        
        return {
            'exists': True,
            'bounds': bounds,
            'row_centers': sorted(row_centers),
            'col_centers': sorted(col_centers),
            'avg_row_h': (bounds['bottom'] - bounds['top']) / len(row_centers) if row_centers else 0,
            'avg_col_w': (bounds['right'] - bounds['left']) / len(col_centers) if col_centers else 0
        }

    def _extract_features(self, label: Dict, w: int, h: int) -> Dict:
        """Extract geometric features."""
        bbox = label['xyxy']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return {
            'cx': cx, 'cy': cy,
            'nx': cx/w, 'ny': cy/h,
            'w': bbox[2] - bbox[0],
            'h': bbox[3] - bbox[1],
            'bbox': bbox,
            'text': label.get('text', '')
        }

    def _compute_heatmap_scores(self, feat: Dict, grid: Dict, orientation: Orientation) -> Dict[str, float]:
        """Compute probability scores with fractional overlap."""
        scores = {'scale_label': 0.0, 'tick_label': 0.0, 'axis_title': 0.0}
        
        if not grid['exists']:
            return scores
            
        bounds = grid['bounds']
        
        # 1. Fractional Overlap (instead of binary flags)
        # Compute how much the label overlaps with regions adjacent to grid edges
        label_bbox = feat['bbox']
        
        # Fractional overlap with left region
        # Use simple coordinate arithmetic for robustness
        # Left region: x < left bounding
        left_region = (0, bounds['top'], bounds['left'], bounds['bottom'])
        frac_left = self._compute_bbox_iou(label_bbox, left_region)
        
        # Fractional overlap with right region
        # Ensure right region doesn't exceed reasonable image bounds (though IoU handles overlap securely)
        right_boundary = bounds['right']
        right_region_width = bounds['right']  # roughly another width
        right_region = (right_boundary, bounds['top'], right_boundary + right_region_width, bounds['bottom'])
        frac_right = self._compute_bbox_iou(label_bbox, right_region)
        
        # Fractional overlap with bottom region
        bottom_boundary = bounds['bottom']
        bottom_region_height = bounds['bottom']
        bottom_region = (bounds['left'], bottom_boundary, bounds['right'], bottom_boundary + bottom_region_height)
        frac_bottom = self._compute_bbox_iou(label_bbox, bottom_region)
        
        # Fractional overlap with top region
        top_region = (bounds['left'], 0, bounds['right'], bounds['top'])
        frac_top = self._compute_bbox_iou(label_bbox, top_region)
        
        # Scale Label Logic (weighted by position)
        scale_score = (frac_left + frac_right) * self.params['left_axis_weight']
        scale_score += (frac_bottom + frac_top) * self.params['bottom_axis_weight']
        scores['scale_label'] += scale_score
        
        # Title Logic (very far from grid = likely title)
        dist_from_grid = min(
            bounds['left'] - feat['cx'] if feat['cx'] < bounds['left'] else 0,
            feat['cx'] - bounds['right'] if feat['cx'] > bounds['right'] else 0,
            bounds['top'] - feat['cy'] if feat['cy'] < bounds['top'] else 0,
            feat['cy'] - bounds['bottom'] if feat['cy'] > bounds['bottom'] else 0
        )
        if dist_from_grid > 50:  # Far from grid
            scores['axis_title'] += self.params['top_title_weight'] * 0.5
        
        # 2. Gaussian Scoring (Recenter Gaussians on Grid Edges)
        gaussian_weights = {
            'left_axis_weight': self.params['left_axis_weight'],
            'right_axis_weight': self.params['right_axis_weight'],
            'bottom_axis_weight': self.params['bottom_axis_weight'],
            'top_title_weight': self.params['top_title_weight'],
            'center_plot_weight': self.params['center_plot_weight']
        }
        
        sigma = self.params.get('gaussian_sigma', 0.05)
        region_scores = self._compute_gaussian_region_scores(
            (feat['nx'], feat['ny']),
            sigma_x=sigma, sigma_y=sigma,
            weights=gaussian_weights
        )
        
        k = self.params['gaussian_kernel_weight']
        scores['scale_label'] += (region_scores['left_axis'] + region_scores['right_axis'] + region_scores['bottom_axis']) * k
        scores['axis_title'] += region_scores['top_title'] * k
        scores['scale_label'] -= region_scores['center_plot'] * k
        
        return scores

    def _compute_bbox_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)

    def _apply_fallback(self, label: Dict, classified: Dict, grid: Dict):
        """Fallback: if close to grid edge -> scale, else title"""
        # Simple distance check
        scores = self._compute_heatmap_scores(
             self._extract_features(label, 1000, 1000), # dummy dims, relational relies on relative
             grid, Orientation.VERTICAL
        )
        if scores['scale_label'] > scores['axis_title']:
            classified['scale_label'].append(label)
        else:
            classified['axis_title'].append(label)

    def _separate_xy_labels(self, scale_labels: List[Dict], grid: Dict):
        """Separate scale labels into X and Y sets based on geometric relation to grid."""
        if not grid['exists']:
            return [], []
            
        x_labels = []
        y_labels = []
        bounds = grid['bounds']
        
        for label in scale_labels:
            bbox = label['xyxy']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Determine if more horizontally or vertically aligned
            # Vertical span check (for Y-axis)
            in_v_span = bounds['top'] <= cy <= bounds['bottom']
            # Horizontal span check (for X-axis)
            in_h_span = bounds['left'] <= cx <= bounds['right']
            
            if in_v_span and not in_h_span:
                y_labels.append(label)
            elif in_h_span and not in_v_span:
                x_labels.append(label)
            else:
                # Ambiguous: use proximity
                dist_x = min(abs(cx - bounds['left']), abs(cx - bounds['right']))
                dist_y = min(abs(cy - bounds['top']), abs(cy - bounds['bottom']))
                
                if dist_x < dist_y:
                    y_labels.append(label) # Closer to side edge
                else:
                    x_labels.append(label) # Closer to top/bottom edge
                    
        return x_labels, y_labels

    def compute_feature_scores(self, label_features: Dict, region_scores: Dict, element_context: Optional[Dict]) -> Dict[str, float]:
        """Abstract method implementation - not used directly as we override classify."""
        return {}

    def _empty_result(self) -> ClassificationResult:
        return ClassificationResult([], [], [], 0.0, {})
