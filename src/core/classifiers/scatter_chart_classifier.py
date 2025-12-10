import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .base_classifier import BaseChartClassifier, ClassificationResult
from utils.validation_utils import is_numeric
from services.orientation_service import Orientation

class ScatterChartClassifier(BaseChartClassifier):
    """Specialized classifier optimized exclusively for scatter plots"""
    
    def __init__(self, params: Dict = None, logger: logging.Logger = None):
        super().__init__(params or self.get_default_params(), logger)
    
    @classmethod
    def get_default_params(cls) -> Dict:
        return {
            # Scatter-specific thresholds
            'scale_size_max_width': 0.075,
            'scale_size_max_height': 0.04,
            'numeric_boost': 7.0,  # INCREASED: Scatter labels are almost always numeric
            
            # Position weights
            'left_edge_weight': 7.0,
            'right_edge_weight': 6.0,
            'bottom_edge_weight': 6.5,
            
            # Scatter-specific
            'point_cloud_proximity_weight': 3.0,  # REDUCED: Less false positives near dense clusters
            'dual_axis_support': True,
            
            # Title detection
            'title_size_min': 0.14,
            'title_aspect_min': 7.0,
            
            # Thresholds
            'classification_threshold': 3.0,
            'edge_threshold': 0.25,  # INCREASED: Wider tolerance for tight layouts
            
            # NEW: Gaussian kernel weights (Phase 14)
            'gaussian_kernel_weight': 1.5,  # 1.5x boost for scatter
            'center_penalty': 0.0,  # DISABLED: Scatter can have internal axes
            'left_axis_weight': 6.0,
            'right_axis_weight': 5.0,
            'bottom_axis_weight': 5.0,
            'top_title_weight': 4.0,
            'center_plot_weight': 0.0  # DISABLED
        }
    
    def classify(
        self,
        axis_labels: List[Dict],
        chart_elements: List[Dict],
        img_width: int,
        img_height: int,
        orientation: Orientation = Orientation.VERTICAL
    ) -> ClassificationResult:
        """
        Scatter plot specific classification
        """
        scatter_points = chart_elements
        if not self.validate_inputs(axis_labels, scatter_points):
            return self._empty_result()
        
        # Extract scatter-specific features
        label_features = self._extract_scatter_features(axis_labels, img_width, img_height)
        scatter_context = self._compute_scatter_context(scatter_points, img_width, img_height)
        
        # Classification
        classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
        all_scores = []
        
        for feat in label_features:
            scores = self._compute_scatter_scores(feat, scatter_context)
            all_scores.append(scores)
            
            best_class = max(scores, key=scores.get)
            threshold = self.params['classification_threshold']
            
            if scores[best_class] > threshold:
                if best_class == 'scale_label' and feat['is_numeric']:
                    try:
                        numeric_val = float(feat['text'].replace(',', '').replace('%', ''))
                        feat['label']['cleaned_value'] = numeric_val
                        classified[best_class].append(feat['label'])
                    except:
                        classified['axis_title'].append(feat['label'])
                else:
                    classified[best_class].append(feat['label'])
            else:
                # Scatter default: strong bias toward scale labels
                if feat['is_numeric']:
                    classified['scale_label'].append(feat['label'])
                else:
                    classified['axis_title'].append(feat['label'])
        
        # Separate X and Y scales
        x_scales, y_scales = self._separate_xy_scales_scatter(
            classified['scale_label'], img_width, img_height, scatter_context
        )
        
        confidence = self._compute_confidence(all_scores)
        
        metadata = {
            'chart_type': 'scatter',
            'num_points': len(scatter_points),
            'x_scales': len(x_scales),
            'y_scales': len(y_scales),
            'point_density': scatter_context.get('density', 0)
        }
        
        return ClassificationResult(
            scale_labels=classified['scale_label'],
            tick_labels=[],  # Scatter plots don't have tick labels
            axis_titles=classified['axis_title'],
            confidence=confidence,
            metadata=metadata
        )
    
    def _extract_scatter_features(self, labels: List[Dict], w: int, h: int) -> List[Dict]:
        features = []
        for label in labels:
            x1, y1, x2, y2 = label['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            
            text = label.get('text', '')
            is_num = is_numeric(text)
            
            features.append({
                'label': label,
                'cx': cx, 'cy': cy,
                'nx': cx / w, 'ny': cy / h,
                'width': width, 'height': height,
                'rel_w': width / w, 'rel_h': height / h,
                'aspect': width / (height + 1e-6),
                'is_numeric': is_numeric,
                'text': text
            })
        return features
    
    def _compute_scatter_context(self, points: List[Dict], w: int, h: int) -> Dict:
        if not points:
            return {}
        
        positions = []
        for pt in points:
            x1, y1, x2, y2 = pt['xyxy']
            positions.append(((x1 + x2) / 2, (y1 + y2) / 2))
        
        positions = np.array(positions)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        x_spread = x_max - x_min
        y_spread = y_max - y_min
        
        extent = {
            'left': x_min - x_spread * 0.05,
            'right': x_max + x_spread * 0.05,
            'top': y_min - y_spread * 0.05,
            'bottom': y_max + y_spread * 0.05
        }
        
        area = x_spread * y_spread
        density = len(points) / (area + 1e-6)
        
        return {
            'positions': positions,
            'extent': extent,
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'density': density,
            'num_points': len(points)
        }
    
    def _compute_scatter_scores(self, feat: Dict, scatter_ctx: Dict) -> Dict[str, float]:
        scores = {'scale_label': 0.0, 'tick_label': 0.0, 'axis_title': 0.0}
        
        nx, ny = feat['nx'], feat['ny']
        rel_w, rel_h = feat['rel_w'], feat['rel_h']
        aspect = feat['aspect']
        is_numeric = feat['is_numeric']
        
        # === GAUSSIAN REGION SCORING (Phase 14) ===
        gaussian_weights = {
            'left_axis_weight': self.params.get('left_axis_weight', 6.0),
            'right_axis_weight': self.params.get('right_axis_weight', 5.0),
            'bottom_axis_weight': self.params.get('bottom_axis_weight', 5.0),
            'top_title_weight': self.params.get('top_title_weight', 4.0),
            'center_plot_weight': self.params.get('center_plot_weight', 0.0)  # Disabled for scatter
        }
        
        region_scores = self._compute_gaussian_region_scores(
            (nx, ny),
            sigma_x=0.09,
            sigma_y=0.09,
            weights=gaussian_weights
        )
        
        kernel_weight = self.params.get('gaussian_kernel_weight', 1.5)
        
        # Apply Gaussian scores to scale_label
        gaussian_scale = (
            region_scores['left_axis'] + 
            region_scores['right_axis'] + 
            region_scores['bottom_axis']
        ) * kernel_weight
        scores['scale_label'] += gaussian_scale
        
        # Apply Gaussian to title
        scores['axis_title'] += region_scores['top_title'] * kernel_weight
        
        # Center penalty (disabled via center_plot_weight=0 for scatter)
        scores['scale_label'] -= region_scores['center_plot']
        
        # === SCALE LABEL SCORING (Original + Enhanced) ===
        # Small size
        if rel_w < self.params['scale_size_max_width'] and rel_h < self.params['scale_size_max_height']:
            scores['scale_label'] += 5.0
        
        # Edge positions (critical for scatter)
        if nx < self.params['edge_threshold']:
            scores['scale_label'] += self.params['left_edge_weight']
        elif nx > (1.0 - self.params['edge_threshold']):
            scores['scale_label'] += self.params['right_edge_weight']
        
        if ny > (1.0 - self.params['edge_threshold']):
            scores['scale_label'] += self.params['bottom_edge_weight']
        
        # Numeric boost (very strong for scatter)
        if is_numeric:
            scores['scale_label'] += self.params['numeric_boost']
        
        # Proximity to point cloud
        if scatter_ctx:
            proximity_score = self._compute_point_cloud_proximity(feat, scatter_ctx)
            scores['scale_label'] += proximity_score * self.params['point_cloud_proximity_weight']
        
        # === TICK LABEL SCORING (Not used in scatter) ===
        scores['tick_label'] = 0.0
        
        # === AXIS TITLE SCORING ===
        # Large size
        if rel_w > self.params['title_size_min'] or rel_h > 0.09:
            scores['axis_title'] += 6.0
        
        # Extreme aspect ratio
        if aspect > self.params['title_aspect_min'] or aspect < 0.14:
            scores['axis_title'] += 5.0
        
        # Non-numeric
        if not is_numeric:
            scores['axis_title'] += 3.0
        
        # Top or center position
        if ny < 0.15 or (0.3 < ny < 0.7):
            scores['axis_title'] += 2.0
        
        return scores
    
    def _compute_point_cloud_proximity(self, feat: Dict, scatter_ctx: Dict) -> float:
        """Compute proximity to scatter point cloud"""
        extent = scatter_ctx.get('extent')
        if not extent:
            return 0.0
        cx, cy = feat['cx'], feat['cy']
        
        # Compute distance to cloud extent
        if cx < extent['left']:
            dist_x = extent['left'] - cx
        elif cx > extent['right']:
            dist_x = cx - extent['right']
        else:
            dist_x = 0
        
        if cy < extent['top']:
            dist_y = extent['top'] - cy
        elif cy > extent['bottom']:
            dist_y = cy - extent['bottom']
        else:
            dist_y = 0
        
        total_dist = np.sqrt(dist_x**2 + dist_y**2)
        
        # Normalize
        max_dim = max(extent['right'] - extent['left'], extent['bottom'] - extent['top'])
        proximity = 1.0 - (total_dist / (max_dim * 0.5 + 1e-6))
        
        return max(0.0, min(1.0, proximity))
    
    def _separate_xy_scales_scatter(
        self, scale_labels: List[Dict], w: int, h: int, scatter_ctx: Dict
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Separate into X and Y scales using robust geometric clustering.
        Replaces naïve hard thresholds with alignment detection.
        """
        if not scale_labels:
            self.logger.warning("No scale labels provided to sensitive separation")
            return [], []
        
        self.logger.info(f"Separating {len(scale_labels)} labels into X/Y axes")

        x_scales = []
        y_scales = []
        
        # 1. Collect Centroids
        points = []
        for i, label in enumerate(scale_labels):
            x1, y1, x2, y2 = label['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            points.append({'cx': cx, 'cy': cy, 'label': label, 'idx': i})
            
        # 2. Score affinity to X-axis (Horizontal) vs Y-axis (Vertical)
        # X-axis labels form a horizontal line: cy ~= constant
        # Y-axis labels form a vertical line: cx ~= constant
        
        # We can simply group by finding the dominant Horizontal line (bottom usually)
        # and dominant Vertical line (left usually)
        
        # Simple RANSAC-like line finding:
        # Bin 'cy' values to find potential X-axis rows
        cy_vals = [p['cy'] for p in points]
        cx_vals = [p['cx'] for p in points]
        
        # Initial Assignments based on alignment
        # Detect rows (potential X axes)
        # We assume X axis is usually at bottom
        y_bins = {}
        bin_size = h * 0.05 # 5% height tolerance
        for p in points:
            bin_idx = int(p['cy'] / bin_size)
            if bin_idx not in y_bins: y_bins[bin_idx] = []
            y_bins[bin_idx].append(p)
            
        # Find dominant bottom row
        best_x_row = []
        max_y_val = -1
        
        for b_idx, row_points in y_bins.items():
            # Filter: Must have mostly distinct X values (spread out horizontally)
            if len(row_points) < 2: continue
            
            # Check if it's the "lowest" (max cy) significant row
            avg_cy = np.mean([p['cy'] for p in row_points])
            
            # Prefer rows lower in image (higher y) for standard charts
            # Heuristic: Score = count * 10 + (position_score)
            # We want to favor rows with MANY points (the axis) over rows with 1-2 points (artifacts)
            # taking into account that the axis is usually near the bottom.
            
            # Current score
            current_count = len(row_points)
            best_count = len(best_x_row)
            
            # If strictly more points (with some buffer against noise), take it
            if current_count > best_count:
                max_y_val = avg_cy
                best_x_row = row_points
            elif current_count == best_count:
                 # If counts are equal, prefer the one lower in image (higher y)
                 if avg_cy > max_y_val:
                     max_y_val = avg_cy
                     best_x_row = row_points

        self.logger.debug(f"Best X-row candidate: {len(best_x_row)} points at cy~={max_y_val:.1f}")

        # Detect columns (potential Y axes)
        # We assume Y axis is usually at left
        x_bins = {}
        bin_size_w = w * 0.05
        for p in points:
            bin_idx = int(p['cx'] / bin_size_w)
            if bin_idx not in x_bins: x_bins[bin_idx] = []
            x_bins[bin_idx].append(p)
            
        # Find dominant left column
        best_y_col = []
        min_x_val = w * 999
        
        for b_idx, col_points in x_bins.items():
            if len(col_points) < 2: continue
            
            avg_cx = np.mean([p['cx'] for p in col_points])
            
            # Prefer columns further left (lower x)
            # Similar logic for Y-axis (prefer Left)
            current_count = len(col_points)
            best_count = len(best_y_col)
            
            if current_count > best_count:
                min_x_val = avg_cx
                best_y_col = col_points
            elif current_count == best_count:
                # If equal, prefer left-most
                if avg_cx < min_x_val:
                     min_x_val = avg_cx
                     best_y_col = col_points

        self.logger.debug(f"Best Y-col candidate: {len(best_y_col)} points at cx~={min_x_val:.1f}")
                
        # 3. Assign and Resolve Conflicts
        assigned_indices = set()
        
        # Limit X candidates to those with high cy (bottom 40% usually, or just below plot)
        # Limit Y candidates to those with low cx (left 40% usually)
        
        # But wait, scientific plots might have axes in center.
        # Let's rely on the "Line" quality.
        
        # Assign best row to X
        for p in best_x_row:
            p['label']['axis'] = 'x'
            x_scales.append(p['label'])
            assigned_indices.add(p['idx'])
            
        # Assign best col to Y
        for p in best_y_col:
            if p['idx'] in assigned_indices:
                # Conflict (Corner point?): Assign to based on monotonicity or exclude
                # Usually corner point is 0. If it fits both, we might duplicate or pick one.
                # For now, let's keep it in X if already there? 
                # Better: Check aspect ratio of the label text?
                # or check local neighbors.
                pass
            else:
                p['label']['axis'] = 'y'
                y_scales.append(p['label'])
                assigned_indices.add(p['idx'])
                
        # 4. Handle remaining/outlier points
        # If we missed some, try to assign to closest group if close enough
        for p in points:
            if p['idx'] in assigned_indices: continue
            
            # Distance to X-line (cy = max_y_val)
            dist_to_x = abs(p['cy'] - max_y_val) if max_y_val > 0 else 9999
            
            # Distance to Y-line (cx = min_x_val)
            dist_to_y = abs(p['cx'] - min_x_val) if min_x_val < 9999 else 9999
            
            threshold = h * 0.05
            
            if dist_to_x < threshold and dist_to_x < dist_to_y:
                p['label']['axis'] = 'x'
                x_scales.append(p['label'])
            elif dist_to_y < threshold:
                p['label']['axis'] = 'y'
                y_scales.append(p['label'])
            else:
                 # Fallback to old positional logic for strays
                 nx, ny = p['cx']/w, p['cy']/h
                 if ny > 0.75: 
                     p['label']['axis'] = 'x'
                     x_scales.append(p['label'])
                 elif nx < 0.2:
                     p['label']['axis'] = 'y'
                     y_scales.append(p['label'])
        
        self.logger.info(f"Separation Result: X={len(x_scales)}, Y={len(y_scales)} labels")
        return x_scales, y_scales
    
    
    def _compute_confidence(self, all_scores: List[Dict[str, float]]) -> float:
        if not all_scores:
            return 0.5
        
        margins = []
        for score_dict in all_scores:
            sorted_scores = sorted(score_dict.values(), reverse=True)
            if len(sorted_scores) >= 2:
                margin = sorted_scores[0] - sorted_scores[1]
                margins.append(margin)
        
        avg_margin = np.mean(margins) if margins else 0.0
        return min(1.0, max(0.0, avg_margin * 0.25))
    
    def _empty_result(self) -> ClassificationResult:
        return ClassificationResult(
            scale_labels=[], tick_labels=[], axis_titles=[],
            confidence=0.0, metadata={'error': 'Invalid input'}
        )

    def compute_feature_scores(self, label_features: Dict, region_scores: Dict, element_context: Optional[Dict]) -> Dict[str, float]:
        """
        This is a placeholder to satisfy the abstract method requirement.
        The main logic is in _compute_scatter_scores.
        """
        return self._compute_scatter_scores(label_features, element_context)
