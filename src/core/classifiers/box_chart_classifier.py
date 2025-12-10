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
            'classification_threshold': 2.5,  # Increased as requested (was 2.2)
            'edge_threshold': 0.22,
            
            # NEW: Gaussian kernel weights (Phase 14)
            'gaussian_kernel_weight': 1.0,
            'left_axis_weight': 5.0,
            'right_axis_weight': 4.0,
            'bottom_axis_weight': 5.0,
            'top_title_weight': 4.0,
            'center_plot_weight': 2.5,
            
            # User Recommended Updates
            'tick_bottom_vertical_weight': 7.0,  # New param for vertical boxt plots
            'numeric_boost': 2.5,  # Reduced from 3.5
            'scale_edge_weight': 5.0,  # Reduced from 6.0
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
        Box plot specific classification with adaptive thresholding.
        """
        boxes = chart_elements
        if not self.validate_inputs(axis_labels, boxes):
            return self._empty_result()
        
        # Extract box-specific features (now includes distance features)
        label_features = self._extract_box_features(axis_labels, img_width, img_height)
        box_context = self._compute_box_context(boxes, img_width, img_height, orientation)
        
        # Classification with adaptive thresholding
        classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
        all_scores = []
        
        for feat in label_features:
            scores = self._compute_box_scores(feat, box_context, orientation)
            all_scores.append(scores)
            
            best_class = max(scores, key=scores.get)
            
            # IMPROVEMENT: Adaptive threshold based on margin analysis
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else float('inf')
            
            base_threshold = self.params['classification_threshold']
            if margin < 0.5:  # Ambiguous case
                threshold = base_threshold * 1.3  # Require higher confidence
            elif margin > 2.0:  # Clear case
                threshold = base_threshold * 0.9  # Allow lower threshold
            else:
                threshold = base_threshold
            
            if scores[best_class] > threshold:
                classified[best_class].append(feat['label'])
            else:
                # IMPROVEMENT: Enhanced fallback with constraint checking (now context-aware)
                self._apply_fallback_classification(feat, classified, box_context)
        
        # Post-process: align tick labels with boxes
        from extractors.box_associator import BoxElementAssociator
        associator = BoxElementAssociator()
        classified['tick_label'] = associator.align_tick_labels_with_boxes(
            classified['tick_label'], boxes, orientation, img_width, img_height
        )
        
        # IMPROVEMENT: Apply constraint-based post-processing
        # Pass box_context for relative positioning checks
        classified = self._apply_constraints(classified, label_features, img_width, img_height, box_context)
        
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
    
    def _apply_fallback_classification(self, feat: Dict, classified: Dict, box_context: Dict = None):
        """
        Apply multi-stage fallback when primary classification is ambiguous.
        Uses position (relative to boxes), size, and content features hierarchically.
        """
        # Stage 0: Priority bottom check for vertical plots
        # Robust: Check if label is below the lowest box
        is_below_boxes = False
        if box_context and 'extent' in box_context:
            box_bottom = box_context['extent']['bottom']
            # If label center is below box bottom (with small margin)
            if feat['cy'] > box_bottom + 5:
                is_below_boxes = True
        
        # Use either absolute position (backup) or relative position (primary)
        if is_below_boxes or feat.get('ny', 0) > 0.82:
            classified['tick_label'].append(feat['label'])
            return
            
        # Stage 1: Position-based fallback (continuous distance)
        min_edge_dist = feat.get('min_edge_dist', 0.5)
        
        if min_edge_dist < 0.15:  # Very close to edge
            classified['scale_label'].append(feat['label'])
            return
        elif min_edge_dist > 0.3:  # Far from edges
            classified['tick_label'].append(feat['label'])
            return
        
        # Stage 2: Size-based fallback
        is_small = feat.get('is_small', False)
        if is_small:
            classified['scale_label'].append(feat['label'])
            return
        
        # Stage 3: Content-based fallback (original logic)
        if feat['is_numeric']:
            classified['scale_label'].append(feat['label'])
        else:
            classified['tick_label'].append(feat['label'])
    
    def _apply_constraints(self, classified: Dict, features: List[Dict], 
                           img_width: int, img_height: int, box_context: Dict = None) -> Dict:
        """
        Apply domain-specific constraints to refine classifications.
        
        Constraints:
        1. Scale labels should be on axis edges (min_edge_dist < 0.2)
        2. Tick labels should NOT be on edges AND should align with elements
        """
        # Build feature lookup
        label_to_feat = {}
        for f in features:
            xyxy = tuple(f['label']['xyxy'])
            label_to_feat[xyxy] = f
        
        refined_scale = []
        refined_tick = []
        ambiguous = []
        
        # Check scale labels
        for label in classified.get('scale_label', []):
            xyxy = tuple(label['xyxy'])
            feat = label_to_feat.get(xyxy)
            if feat and feat.get('min_edge_dist', 0) >= 0.25:  # Violates edge constraint
                ambiguous.append((feat, label, 'scale'))
            else:
                refined_scale.append(label)
        
        # Check tick labels
        for label in classified.get('tick_label', []):
            xyxy = tuple(label['xyxy'])
            feat = label_to_feat.get(xyxy)
            if feat:
                ny = feat.get('ny', 0)
                cy = feat.get('cy', 0)
                min_edge_dist = feat.get('min_edge_dist', 0.5)
                
                # Check if truly below boxes
                is_below_boxes = False
                if box_context and 'extent' in box_context:
                    if cy > box_context['extent']['bottom']:
                        is_below_boxes = True
                
                # IMPROVEMENT: Stricter edge constraint for tick labels, especially at bottom
                # If NOT below boxes and on edge -> swap to scale
                if not is_below_boxes and ny > 0.8 and min_edge_dist < 0.25:
                    ambiguous.append((feat, label, 'tick'))
                elif not is_below_boxes and min_edge_dist < 0.1:  # General edge constraint
                    ambiguous.append((feat, label, 'tick'))
                else:
                    refined_tick.append(label)
            else:
                refined_tick.append(label)
        
        # Resolve ambiguous cases
        for feat, label, original_type in ambiguous:
            if original_type == 'scale':
                # Was scale but not on edge -> should be tick
                refined_tick.append(label)
            else:
                # Was tick but too close to edge -> should be scale
                refined_scale.append(label)
        
        return {
            'scale_label': refined_scale,
            'tick_label': refined_tick,
            'axis_title': classified.get('axis_title', [])
        }
    
    def _extract_box_features(self, labels: List[Dict], w: int, h: int) -> List[Dict]:
        """
        Extract comprehensive features including continuous distance metrics.
        """
        features = []
        for label in labels:
            x1, y1, x2, y2 = label['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            
            text = label.get('text', '')
            is_num = is_numeric(text)  # FIX: Properly call the function
            
            # Normalized positions
            nx, ny = cx / w, cy / h
            
            # IMPROVEMENT: Continuous distance features (not binary)
            dist_left = nx
            dist_right = 1.0 - nx
            dist_top = ny
            dist_bottom = 1.0 - ny
            min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            
            # Size analysis
            rel_w = width / w
            rel_h = height / h
            is_small = (rel_w < self.params['scale_size_max_width'] and 
                       rel_h < self.params['scale_size_max_height'])
            
            features.append({
                'label': label,
                'cx': cx, 'cy': cy,
                'nx': nx, 'ny': ny,
                'width': width, 'height': height,
                'rel_w': rel_w, 'rel_h': rel_h,
                'aspect': width / (height + 1e-6),
                'is_numeric': is_num,  # FIX: Use the computed is_num, not the function
                'text': text,
                # NEW: Distance features
                'dist_left': dist_left,
                'dist_right': dist_right,
                'dist_top': dist_top,
                'dist_bottom': dist_bottom,
                'min_edge_dist': min_edge_dist,
                'is_small': is_small
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
        
        # === GAUSSIAN REGION SCORING (Phase 14) ===
        gaussian_weights = {
            'left_axis_weight': self.params.get('left_axis_weight', 5.0),
            'right_axis_weight': self.params.get('right_axis_weight', 4.0),
            'bottom_axis_weight': self.params.get('bottom_axis_weight', 5.0),
            'top_title_weight': self.params.get('top_title_weight', 4.0),
            'center_plot_weight': self.params.get('center_plot_weight', 2.5)
        }
        
        region_scores = self._compute_gaussian_region_scores(
            (nx, ny),
            sigma_x=0.09,
            sigma_y=0.09,
            weights=gaussian_weights
        )
        
        kernel_weight = self.params.get('gaussian_kernel_weight', 1.0)
        
        # Apply Gaussian scores
        if orientation == Orientation.VERTICAL:
            scores['scale_label'] += (region_scores['left_axis'] + region_scores['right_axis']) * kernel_weight
            scores['tick_label'] += region_scores['bottom_axis'] * kernel_weight
        else:
            scores['scale_label'] += region_scores['bottom_axis'] * kernel_weight
            scores['tick_label'] += region_scores['left_axis'] * kernel_weight
        
        scores['axis_title'] += region_scores['top_title'] * kernel_weight
        scores['scale_label'] -= region_scores['center_plot']
        
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
        
        # IMPROVEMENT: Continuous edge scoring (smooth gradient instead of cliff)
        min_edge_dist = feat.get('min_edge_dist', 0.5)
        edge_score = max(0, (0.2 - min_edge_dist) / 0.2)  # Normalize to [0, 1]
        scores['scale_label'] += edge_score * 2.0  # Additional edge-based score
        
        # === TICK LABEL SCORING ===
        # For box plots:
        # - Vertical: X-axis has ticks (bottom, categories)
        # - Horizontal: Y-axis has ticks (left, categories)
        
        if orientation == Orientation.VERTICAL:
            # Bottom region for tick labels
            if ny > 0.78:
                scores['tick_label'] += self.params['tick_alignment_weight']
            
            # IMPROVEMENT: Increased boost for very low (bottom) labels in vertical plots
            # Use relative checks if available, falling back to absolute
            is_below_boxes = False
            if box_ctx and 'extent' in box_ctx:
                 if feat['cy'] > box_ctx['extent']['bottom']:
                     is_below_boxes = True
            
            if is_below_boxes or ny > 0.82:
                scores['tick_label'] += self.params.get('tick_bottom_vertical_weight', 7.0)
                # Reduce scale score to avoid Gaussian interference from left axis
                scores['scale_label'] -= region_scores.get('left_axis', 0) * 0.5
            
            # Alignment with box centers
            if box_ctx:
                alignment_score = self._compute_box_alignment(feat, box_ctx, Orientation.VERTICAL)
                scores['tick_label'] += alignment_score * self.params['box_spacing_weight']
                
                # Extra boost for good alignment
                if alignment_score > 0.6:
                    scores['tick_label'] += 3.0
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