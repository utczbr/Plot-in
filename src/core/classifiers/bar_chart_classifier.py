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
            'confidence_margin_factor': 0.4
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
        Bar chart specific classification with context-aware scoring
        """
        bars = chart_elements
        if not self.validate_inputs(axis_labels, bars):
            return self._empty_result()
        
        # Extract bar-specific features
        label_features = self._extract_bar_features(axis_labels, img_width, img_height)
        bar_context = self._compute_bar_context(bars, img_width, img_height, orientation)
        
        # Classification with bar-specific logic
        classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
        all_scores = []
        
        for feat in label_features:
            scores = self._compute_bar_scores(feat, bar_context, orientation)
            all_scores.append(scores)
            
            # Decision logic
            best_class = max(scores, key=scores.get)
            threshold = self.params['classification_threshold']
            
            if scores[best_class] > threshold:
                classified[best_class].append(feat['label'])
            else:
                # Default for bars: numeric = scale, non-numeric = tick
                if feat['is_numeric']:
                    classified['scale_label'].append(feat['label'])
                else:
                    classified['tick_label'].append(feat['label'])
        
        # Post-process: align tick labels with bars
        classified['tick_label'] = self._align_ticks_with_bars(
            classified['tick_label'], bars, orientation, img_width, img_height
        )
        
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
    
    def _extract_bar_features(self, labels: List[Dict], w: int, h: int) -> List[Dict]:
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
