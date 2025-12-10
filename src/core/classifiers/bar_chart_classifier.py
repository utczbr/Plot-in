import numpy as np
from typing import Dict, List, Optional
import logging

from .base_classifier import BaseChartClassifier, ClassificationResult
from utils.validation_utils import is_numeric
from services.orientation_service import Orientation

class BarChartClassifier(BaseChartClassifier):
    """Specialized classifier optimized exclusively for bar charts"""
    
    def __init__(self, params: Dict = None, logger: logging.Logger = None):
        super().__init__(params or self.get_default_params(), logger)
        
    @classmethod
    def get_default_params(cls) -> Dict:
        return {
            # Size thresholds
            'scale_size_max_width': 0.08,
            'scale_size_max_height': 0.04,
            'tick_size_min_width': 0.03,
            'title_size_min_width': 0.15,
            
            # Position weights
            'scale_left_weight': 6.0,
            'scale_right_weight': 5.0,
            'scale_bottom_weight': 5.5,
            'tick_bottom_weight': 7.0,
            'tick_left_weight': 6.5,
            
            # Context weights
            'bar_alignment_weight': 5.0,
            'bar_spacing_weight': 4.5,
            'numeric_boost': 3.0,
            
            # Aspect ratio
            'aspect_ratio_min': 0.4,
            'aspect_ratio_max': 4.0,
            'title_aspect_min': 5.0,
            
            # Decision thresholds
            'classification_threshold': 2.0,
            'confidence_margin_factor': 0.4,
            
            # NEW: Gaussian kernel weights (Phase 14)
            'gaussian_kernel_weight': 1.0,  # Default for bar
            'left_axis_weight': 5.0,
            'right_axis_weight': 4.0,
            'bottom_axis_weight': 6.0,  # Boosted for tick labels
            'top_title_weight': 4.0,
            'center_plot_weight': 3.0  # Penalty for center labels
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
        Bar chart specific classification with adaptive thresholding.
        """
        bars = chart_elements
        if not self.validate_inputs(axis_labels, bars):
            return self._empty_result()
        
        # Extract bar-specific features (includes distance features)
        label_features = self._extract_bar_features(axis_labels, img_width, img_height)
        bar_context = self._compute_bar_context(bars, img_width, img_height, orientation)
        
        # Classification with adaptive thresholding
        classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
        all_scores = []
        
        for feat in label_features:
            scores = self._compute_bar_scores(feat, bar_context, orientation)
            all_scores.append(scores)
            
            best_class = max(scores, key=scores.get)
            
            # IMPROVEMENT: Adaptive threshold based on margin analysis
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else float('inf')
            
            base_threshold = self.params['classification_threshold']
            if margin < 0.5:  # Ambiguous
                threshold = base_threshold * 1.25
            elif margin > 2.0:  # Clear
                threshold = base_threshold * 0.9
            else:
                threshold = base_threshold
            
            if scores[best_class] > threshold:
                classified[best_class].append(feat['label'])
            else:
                # IMPROVEMENT: Enhanced fallback
                self._apply_fallback_classification(feat, classified)
        
        # Post-process: align tick labels with bars
        classified['tick_label'] = self._align_ticks_with_bars(
            classified['tick_label'], bars, orientation, img_width, img_height
        )
        
        # IMPROVEMENT: Apply constraint-based post-processing
        classified = self._apply_constraints(classified, label_features)
        
        confidence = self._compute_confidence(all_scores)
        
        metadata = {
            'chart_type': 'bar',
            'orientation': orientation,
            'num_bars': len(bars),
            'bar_spacing': bar_context.get('avg_spacing', 0),
            'num_scale': len(classified['scale_label']),
            'num_tick': len(classified['tick_label']),
            'num_title': len(classified['axis_title'])
        }
        
        return ClassificationResult(
            scale_labels=classified['scale_label'],
            tick_labels=classified['tick_label'],
            axis_titles=classified['axis_title'],
            confidence=confidence,
            metadata=metadata
        )
    
    def _apply_fallback_classification(self, feat: Dict, classified: Dict):
        """Multi-stage fallback when primary classification is ambiguous."""
        min_edge_dist = feat.get('min_edge_dist', 0.5)
        
        if min_edge_dist < 0.15:
            classified['scale_label'].append(feat['label'])
        elif min_edge_dist > 0.3:
            classified['tick_label'].append(feat['label'])
        elif feat.get('is_small', False):
            classified['scale_label'].append(feat['label'])
        elif feat['is_numeric']:
            classified['scale_label'].append(feat['label'])
        else:
            classified['tick_label'].append(feat['label'])
    
    def _apply_constraints(self, classified: Dict, features: List[Dict]) -> Dict:
        """Apply domain-specific constraints to refine classifications."""
        label_to_feat = {tuple(f['label']['xyxy']): f for f in features}
        
        refined_scale = []
        refined_tick = []
        swaps = []
        
        for label in classified.get('scale_label', []):
            feat = label_to_feat.get(tuple(label['xyxy']))
            if feat and feat.get('min_edge_dist', 0) >= 0.25:
                swaps.append((label, 'scale_to_tick'))
            else:
                refined_scale.append(label)
        
        for label in classified.get('tick_label', []):
            feat = label_to_feat.get(tuple(label['xyxy']))
            if feat and feat.get('min_edge_dist', 0.5) < 0.1:
                swaps.append((label, 'tick_to_scale'))
            else:
                refined_tick.append(label)
        
        for label, swap_type in swaps:
            if swap_type == 'scale_to_tick':
                refined_tick.append(label)
            else:
                refined_scale.append(label)
        
        return {
            'scale_label': refined_scale,
            'tick_label': refined_tick,
            'axis_title': classified.get('axis_title', [])
        }
    
    def _extract_bar_features(self, labels: List[Dict], w: int, h: int) -> List[Dict]:
        """Extract comprehensive features including distance metrics."""
        features = []
        for label in labels:
            x1, y1, x2, y2 = label['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            
            text = label.get('text', '')
            is_num = is_numeric(text)  # FIX: Call function properly
            
            # Normalized positions
            nx, ny = cx / w, cy / h
            
            # IMPROVEMENT: Continuous distance features
            dist_left = nx
            dist_right = 1.0 - nx
            dist_top = ny
            dist_bottom = 1.0 - ny
            min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            
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
                'is_numeric': is_num,  # FIX: Use computed value, not function
                'text': text,
                'dist_left': dist_left,
                'dist_right': dist_right,
                'dist_top': dist_top,
                'dist_bottom': dist_bottom,
                'min_edge_dist': min_edge_dist,
                'is_small': is_small
            })
        return features
    
    def _compute_bar_context(self, bars: List[Dict], w: int, h: int, orientation: Orientation) -> Dict:
        if not bars:
            return {}
        
        centers = []
        for bar in bars:
            x1, y1, x2, y2 = bar['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers.append((cx, cy))
        
        centers = np.array(centers)
        
        if orientation == Orientation.VERTICAL:
            primary_coords = centers[:, 0]  # X-coords
        else:
            primary_coords = centers[:, 1]  # Y-coords
        
        sorted_coords = np.sort(primary_coords)
        spacings = np.diff(sorted_coords) if len(sorted_coords) > 1 else [0]
        avg_spacing = np.mean(spacings) if len(spacings) > 0 else 0
        
        return {
            'bars': bars,
            'centers': centers,
            'avg_spacing': avg_spacing,
            'orientation': orientation,
            'extent': self._compute_extent(bars)
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
    
    def _compute_bar_scores(self, feat: Dict, bar_ctx: Dict, orientation: Orientation) -> Dict[str, float]:
        scores = {'scale_label': 0.0, 'tick_label': 0.0, 'axis_title': 0.0}
        
        nx, ny = feat['nx'], feat['ny']
        rel_w, rel_h = feat['rel_w'], feat['rel_h']
        aspect = feat['aspect']
        is_numeric = feat['is_numeric']
        
        # === GAUSSIAN REGION SCORING (Phase 14) ===
        gaussian_weights = {
            'left_axis_weight': self.params.get('left_axis_weight', 5.0),
            'right_axis_weight': self.params.get('right_axis_weight', 4.0),
            'bottom_axis_weight': self.params.get('bottom_axis_weight', 6.0),
            'top_title_weight': self.params.get('top_title_weight', 4.0),
            'center_plot_weight': self.params.get('center_plot_weight', 3.0)
        }
        
        region_scores = self._compute_gaussian_region_scores(
            (nx, ny),
            sigma_x=0.09,
            sigma_y=0.09,
            weights=gaussian_weights
        )
        
        kernel_weight = self.params.get('gaussian_kernel_weight', 1.0)
        
        # Apply Gaussian scores
        scores['scale_label'] += (region_scores['left_axis'] + region_scores['right_axis']) * kernel_weight
        scores['tick_label'] += region_scores['bottom_axis'] * kernel_weight  # Ticks at bottom for bar
        scores['axis_title'] += region_scores['top_title'] * kernel_weight
        scores['scale_label'] -= region_scores['center_plot']  # Penalty
        scores['tick_label'] -= region_scores['center_plot'] * 0.5
        
        # === SCALE LABEL SCORING ===
        # Small size
        if rel_w < self.params['scale_size_max_width'] and rel_h < self.params['scale_size_max_height']:
            scores['scale_label'] += 3.0
        
        # Position: left/right for vertical bars, bottom for horizontal
        if orientation == Orientation.VERTICAL:
            if nx < 0.15:
                scores['scale_label'] += self.params['scale_left_weight']
            elif nx > 0.85:
                scores['scale_label'] += self.params['scale_right_weight']
        else:
            if ny > 0.85:
                scores['scale_label'] += self.params['scale_bottom_weight']
        
        # Numeric boost
        if is_numeric:
            scores['scale_label'] += self.params['numeric_boost']
        
        # IMPROVEMENT: Continuous edge scoring
        min_edge_dist = feat.get('min_edge_dist', 0.5)
        edge_score = max(0, (0.2 - min_edge_dist) / 0.2)
        scores['scale_label'] += edge_score * 2.0
        
        # === TICK LABEL SCORING ===
        if orientation == Orientation.VERTICAL:
            # Bottom region for vertical bars
            if ny > 0.75:
                scores['tick_label'] += self.params['tick_bottom_weight']
            
            # Alignment with bar centers
            if bar_ctx:
                alignment_score = self._compute_bar_alignment_score(
                    feat['cx'], feat['cy'], bar_ctx, Orientation.VERTICAL
                )
                scores['tick_label'] += alignment_score * self.params['bar_alignment_weight']
        else:
            # Left region for horizontal bars
            if nx < 0.25:
                scores['tick_label'] += self.params['tick_left_weight']
            
            # Alignment with bar centers
            if bar_ctx:
                alignment_score = self._compute_bar_alignment_score(
                    feat['cx'], feat['cy'], bar_ctx, Orientation.HORIZONTAL
                )
                scores['tick_label'] += alignment_score * self.params['bar_alignment_weight']
        
        # Non-numeric boost for tick labels
        if not is_numeric:
            scores['tick_label'] += 2.0
        
        # === AXIS TITLE SCORING ===
        # Large size
        if rel_w > self.params['title_size_min_width'] or rel_h > 0.1:
            scores['axis_title'] += 4.0
        
        # Extreme aspect ratio
        if aspect > self.params['title_aspect_min'] or aspect < 0.2:
            scores['axis_title'] += 3.5
        
        return scores
    
    def _compute_bar_alignment_score(self, cx: float, cy: float, bar_ctx: Dict, orientation: Orientation) -> float:
        if 'centers' not in bar_ctx or len(bar_ctx['centers']) == 0:
            return 0.0
        centers = bar_ctx['centers']
        avg_spacing = bar_ctx['avg_spacing']
        
        if orientation == Orientation.VERTICAL:
            bar_x_coords = centers[:, 0]
            distances = np.abs(bar_x_coords - cx)
            min_dist = np.min(distances)
            
            if min_dist < avg_spacing * 0.5:
                return 1.0
            elif min_dist < avg_spacing * 1.0:
                return 0.5
            else:
                return 0.0
        else:
            bar_y_coords = centers[:, 1]
            distances = np.abs(bar_y_coords - cy)
            min_dist = np.min(distances)
            
            if min_dist < avg_spacing * 0.5:
                return 1.0
            elif min_dist < avg_spacing * 1.0:
                return 0.5
            else:
                return 0.0
    
    def _align_ticks_with_bars(self, ticks: List[Dict], bars: List[Dict], 
                                orientation: Orientation, w: int, h: int) -> List[Dict]:
        """Ensure tick labels are properly aligned with bars"""
        if not ticks or not bars:
            return ticks
        
        bar_centers = []
        for bar in bars:
            x1, y1, x2, y2 = bar['xyxy']
            bar_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        
        aligned_ticks = []
        for tick in ticks:
            x1, y1, x2, y2 = tick['xyxy']
            tx, ty = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Check alignment
            min_dist = float('inf')
            for bx, by in bar_centers:
                if orientation == Orientation.VERTICAL:
                    dist = abs(tx - bx)
                else:
                    dist = abs(ty - by)
                min_dist = min(min_dist, dist)
            
            # Only keep if reasonably aligned
            threshold = 0.15 * (w if orientation == Orientation.VERTICAL else h)
            if min_dist < threshold:
                aligned_ticks.append(tick)
        
        return aligned_ticks
    
    
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
        confidence = min(1.0, avg_margin * self.params['confidence_margin_factor'])
        return max(0.0, confidence)
    
    def _empty_result(self) -> ClassificationResult:
        return ClassificationResult(
            scale_labels=[], tick_labels=[], axis_titles=[],
            confidence=0.0, metadata={'error': 'Invalid input'}
        )

    def compute_feature_scores(self, label_features: Dict, region_scores: Dict, element_context: Optional[Dict]) -> Dict[str, float]:
        """
        This is a placeholder to satisfy the abstract method requirement.
        The main logic is in _compute_bar_scores.
        """
        return self._compute_bar_scores(label_features, element_context, element_context.get('orientation', 'vertical'))
