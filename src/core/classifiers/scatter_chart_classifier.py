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
            'numeric_boost': 5.0,
            
            # Position weights
            'left_edge_weight': 7.0,
            'right_edge_weight': 6.0,
            'bottom_edge_weight': 6.5,
            
            # Scatter-specific
            'point_cloud_proximity_weight': 5.0,
            'dual_axis_support': True,
            
            # Title detection
            'title_size_min': 0.14,
            'title_aspect_min': 7.0,
            
            # Thresholds
            'classification_threshold': 3.0,
            'edge_threshold': 0.18
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
        
        # === SCALE LABEL SCORING (Dominant in scatter plots) ===
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
        
        # Dual-axis penalty for center positions
        center_dist = np.sqrt((nx - 0.5)**2 + (ny - 0.5)**2)
        if center_dist < 0.3:
            scores['scale_label'] -= 2.0
        
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
        """Separate into X and Y scales using scatter-specific logic"""
        x_scales = []
        y_scales = []
        
        extent = scatter_ctx.get('extent', {})
        
        for label in scale_labels:
            x1, y1, x2, y2 = label['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            nx, ny = cx / w, cy / h
            
            # X-axis: bottom region and below point cloud
            if ny > 0.75 or (extent and cy > extent.get('bottom', h)):
                label['axis'] = 'x'
                x_scales.append(label)
            # Y-axis: left/right edges
            elif nx < 0.20 or nx > 0.80:
                label['axis'] = 'y'
                y_scales.append(label)
            else:
                # Fallback
                if ny > nx:
                    label['axis'] = 'x'
                    x_scales.append(label)
                else:
                    label['axis'] = 'y'
                    y_scales.append(label)
        
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
