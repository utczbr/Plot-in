import numpy as np
from typing import Dict, List, Optional
import logging

from .base_classifier import BaseChartClassifier, ClassificationResult
from utils.validation_utils import is_numeric
from services.orientation_service import Orientation

class BoxChartClassifier(BaseChartClassifier):
    """Specialized classifier optimized exclusively for box plots"""
    
    def __init__(self, params: Dict = None, logger: logging.Logger = None):
        super().__init__(params or self.get_default_params(), logger)
    
    @classmethod
    def get_default_params(cls) -> Dict:
        return {
            # Box-specific thresholds
            'scale_size_max_width': 0.065,
            'scale_size_max_height': 0.038,
            'numeric_boost': 3.5,
            
            # Position weights
            'scale_edge_weight': 6.0,
            'tick_alignment_weight': 5.5,
            
            # Box-specific
            'box_spacing_weight': 4.0,
            'category_weight': 3.5,
            
            # Title detection
            'title_size_min': 0.13,
            'title_aspect_min': 6.5,
            
            # Thresholds
            'classification_threshold': 2.2,
            'edge_threshold': 0.22
        }
    
    def classify(
        self,
        axis_labels: List[Dict],
        chart_elements: List[Dict],
        img_width: int,
        img_height: int,
        orientation: Orientation
    ) -> ClassificationResult:
        """
        Box plot specific classification
        """
        boxes = chart_elements
        if not self.validate_inputs(axis_labels, boxes):
            return self._empty_result()
        
        # Extract box-specific features
        label_features = self._extract_box_features(axis_labels, img_width, img_height)
        box_context = self._compute_box_context(boxes, img_width, img_height, orientation)
        
        # Classification
        classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
        all_scores = []
        
        for feat in label_features:
            scores = self._compute_box_scores(feat, box_context, orientation)
            all_scores.append(scores)
            
            best_class = max(scores, key=scores.get)
            threshold = self.params['classification_threshold']
            
            if scores[best_class] > threshold:
                classified[best_class].append(feat['label'])
            else:
                # Box default: numeric = scale, non-numeric = tick
                if feat['is_numeric']:
                    classified['scale_label'].append(feat['label'])
                else:
                    classified['tick_label'].append(feat['label'])
        
        # Post-process: align tick labels with boxes (similar to bar classifier)
        # Use the BoxElementAssociator for consistent architecture with other chart types
        from extractors.box_associator import BoxElementAssociator
        associator = BoxElementAssociator()
        classified['tick_label'] = associator.align_tick_labels_with_boxes(
            classified['tick_label'], boxes, orientation, img_width, img_height
        )
        
        # Post-process: align tick labels with boxes (similar to bar classifier)
        # Use the BoxElementAssociator for consistent architecture with other chart types
        from extractors.box_associator import BoxElementAssociator
        associator = BoxElementAssociator()
        classified['tick_label'] = associator.align_tick_labels_with_boxes(
            classified['tick_label'], boxes, orientation, img_width, img_height
        )
        
        confidence = self._compute_confidence(all_scores)
        
        metadata = {
            'chart_type': 'box',
            'orientation': orientation,
            'num_boxes': len(boxes),
            'box_spacing': box_context.get('avg_spacing', 0)
        }
        
        return ClassificationResult(
            scale_labels=classified['scale_label'],
            tick_labels=classified['tick_label'],
            axis_titles=classified['axis_title'],
            confidence=confidence,
            metadata=metadata
        )
    
    def _extract_box_features(self, labels: List[Dict], w: int, h: int) -> List[Dict]:
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
    
    def _compute_box_context(self, boxes: List[Dict], w: int, h: int, orientation: Orientation) -> Dict:
        if not boxes:
            return {'orientation': orientation}
        
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers.append((cx, cy))
        
        centers = np.array(centers)
        
        if orientation == Orientation.VERTICAL:
            primary_coords = centers[:, 0]
        else:
            primary_coords = centers[:, 1]
        
        sorted_coords = np.sort(primary_coords)
        spacings = np.diff(sorted_coords) if len(sorted_coords) > 1 else [0]
        avg_spacing = np.mean(spacings) if len(spacings) > 0 else 0
        
        return {
            'boxes': boxes,
            'centers': centers,
            'avg_spacing': avg_spacing,
            'orientation': orientation,
            'extent': self._compute_extent(boxes)
        }
    
    def _compute_extent(self, elements: List[Dict]) -> Dict:
        all_x = []
        all_y = []
        for elem in elements:
            x1, y1, x2, y2 = elem['xyxy']
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])
        
        return {
            'left': min(all_x) if all_x else 0,
            'right': max(all_x) if all_x else 0,
            'top': min(all_y) if all_y else 0,
            'bottom': max(all_y) if all_y else 0
        }
    
    def _compute_box_scores(self, feat: Dict, box_ctx: Dict, orientation: Orientation) -> Dict[str, float]:
        scores = {'scale_label': 0.0, 'tick_label': 0.0, 'axis_title': 0.0}
        
        nx, ny = feat['nx'], feat['ny']
        rel_w, rel_h = feat['rel_w'], feat['rel_h']
        aspect = feat['aspect']
        is_numeric = feat['is_numeric']
        
        # === SCALE LABEL SCORING ===
        # For box plots:
        # - Vertical: Y-axis has scale (left/right)
        # - Horizontal: X-axis has scale (bottom)
        
        # Small size
        if rel_w < self.params['scale_size_max_width'] and rel_h < self.params['scale_size_max_height']:
            scores['scale_label'] += 3.5
        
        # Position based on orientation
        if orientation == Orientation.VERTICAL:
            # Y-axis scale on left or right
            if nx < self.params['edge_threshold']:
                scores['scale_label'] += self.params['scale_edge_weight']
            elif nx > (1.0 - self.params['edge_threshold']):
                scores['scale_label'] += self.params['scale_edge_weight'] * 0.8
        else:  # horizontal
            # X-axis scale on bottom
            if ny > (1.0 - self.params['edge_threshold']):
                scores['scale_label'] += self.params['scale_edge_weight']
        
        # Numeric boost
        if is_numeric:
            scores['scale_label'] += self.params['numeric_boost']
        
        # === TICK LABEL SCORING ===
        # For box plots:
        # - Vertical: X-axis has ticks (bottom, categories)
        # - Horizontal: Y-axis has ticks (left, categories)
        
        if orientation == Orientation.VERTICAL:
            # Bottom region for tick labels
            if ny > 0.78:
                scores['tick_label'] += self.params['tick_alignment_weight']
            
            # Alignment with box centers
            if box_ctx:
                alignment_score = self._compute_box_alignment(feat, box_ctx, Orientation.VERTICAL)
                scores['tick_label'] += alignment_score * self.params['box_spacing_weight']
        else:  # horizontal
            # Left region for tick labels
            if nx < 0.22:
                scores['tick_label'] += self.params['tick_alignment_weight']
            
            # Alignment with box centers
            if box_ctx:
                alignment_score = self._compute_box_alignment(feat, box_ctx, Orientation.HORIZONTAL)
                scores['tick_label'] += alignment_score * self.params['box_spacing_weight']
        
        # Non-numeric boost for tick labels
        if not is_numeric:
            scores['tick_label'] += self.params['category_weight']
        
        # === AXIS TITLE SCORING ===
        # Large size
        if rel_w > self.params['title_size_min'] or rel_h > 0.09:
            scores['axis_title'] += 5.0
        
        # Extreme aspect ratio
        if aspect > self.params['title_aspect_min'] or aspect < 0.15:
            scores['axis_title'] += 4.5
        
        return scores
    
    def _compute_box_alignment(self, feat: Dict, box_ctx: Dict, orientation: Orientation) -> float:
        """Compute alignment score with box centers"""
        if 'centers' not in box_ctx or len(box_ctx['centers']) == 0:
            return 0.0
        centers = box_ctx['centers']
        avg_spacing = box_ctx['avg_spacing']
        cx, cy = feat['cx'], feat['cy']
        
        if orientation == Orientation.VERTICAL:
            box_coords = centers[:, 0]
            label_coord = cx
        else:
            box_coords = centers[:, 1]
            label_coord = cy
        
        distances = np.abs(box_coords - label_coord)
        min_dist = np.min(distances)
        
        # Score based on proximity to nearest box
        if min_dist < avg_spacing * 0.4:
            return 1.0
        elif min_dist < avg_spacing * 0.8:
            return 0.6
        elif min_dist < avg_spacing * 1.2:
            return 0.3
        else:
            return 0.0
    
    
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
        return min(1.0, max(0.0, avg_margin * 0.35))
    
    def _empty_result(self) -> ClassificationResult:
        return ClassificationResult(
            scale_labels=[], tick_labels=[], axis_titles=[],
            confidence=0.0, metadata={'error': 'Invalid input'}
        )
    
    def compute_feature_scores(self, label_features: Dict, region_scores: Dict, element_context: Optional[Dict]) -> Dict[str, float]:
        """
        This is a placeholder to satisfy the abstract method requirement.
        The main logic is in _compute_box_scores.
        """
        return self._compute_box_scores(label_features, element_context, element_context.get('orientation', Orientation.VERTICAL))