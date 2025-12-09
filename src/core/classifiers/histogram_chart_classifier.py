import numpy as np
from typing import Dict, List, Optional
import logging

from .base_classifier import BaseChartClassifier, ClassificationResult
from utils.validation_utils import is_numeric, is_continuous_scale
from services.orientation_service import Orientation

class HistogramChartClassifier(BaseChartClassifier):
    """Specialized classifier optimized exclusively for histograms"""
    
    def __init__(self, params: Dict = None, logger: logging.Logger = None):
        super().__init__(params or self.get_default_params(), logger)
    
    @classmethod
    def get_default_params(cls) -> Dict:
        return {
            # Histogram-specific thresholds
            'scale_size_max_width': 0.07,
            'scale_size_max_height': 0.04,
            'numeric_boost': 4.0,
            
            # Position weights
            'left_edge_weight': 6.5,
            'bottom_edge_weight': 6.0,
            'frequency_axis_weight': 5.5,
            
            # Histogram-specific
            'bin_alignment_weight': 5.0,
            'continuous_scale_weight': 3.5,
            
            # Title detection
            'title_size_min': 0.13,
            'title_aspect_min': 6.0,
            
            # Thresholds
            'classification_threshold': 2.3,
            'edge_threshold': 0.19
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
        Histogram specific classification
        """
        bins = chart_elements
        if not self.validate_inputs(axis_labels, bins):
            return self._empty_result()
        
        # Extract histogram-specific features
        label_features = self._extract_histogram_features(axis_labels, img_width, img_height)
        histogram_context = self._compute_histogram_context(bins, img_width, img_height, orientation)
        
        # Classification
        classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
        all_scores = []
        
        for feat in label_features:
            scores = self._compute_histogram_scores(feat, histogram_context, orientation)
            all_scores.append(scores)
            
            best_class = max(scores, key=scores.get)
            threshold = self.params['classification_threshold']
            
            if scores[best_class] > threshold:
                classified[best_class].append(feat['label'])
            else:
                # Histogram default: all numeric = scale
                if feat['is_numeric']:
                    classified['scale_label'].append(feat['label'])
                else:
                    classified['axis_title'].append(feat['label'])
        
        confidence = self._compute_confidence(all_scores)
        
        metadata = {
            'chart_type': 'histogram',
            'orientation': orientation,
            'num_bins': len(bins),
            'bin_width': histogram_context.get('bin_width', 0)
        }
        
        return ClassificationResult(
            scale_labels=classified['scale_label'],
            tick_labels=classified['tick_label'],
            axis_titles=classified['axis_title'],
            confidence=confidence,
            metadata=metadata
        )
    
    def _extract_histogram_features(self, labels: List[Dict], w: int, h: int) -> List[Dict]:
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
    
    def _compute_histogram_context(self, bins: List[Dict], w: int, h: int, orientation: Orientation) -> Dict:
        if not bins:
            return {'orientation': orientation}
        
        centers = []
        widths = []
        for bin_elem in bins:
            x1, y1, x2, y2 = bin_elem['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers.append((cx, cy))
            
            if orientation == Orientation.VERTICAL:
                widths.append(x2 - x1)
            else:
                widths.append(y2 - y1)
        
        centers = np.array(centers)
        avg_bin_width = np.mean(widths) if widths else 0
        
        if orientation == Orientation.VERTICAL:
            primary_coords = centers[:, 0]
        else:
            primary_coords = centers[:, 1]
        
        sorted_coords = np.sort(primary_coords)
        spacings = np.diff(sorted_coords) if len(sorted_coords) > 1 else [0]
        avg_spacing = np.mean(spacings) if len(spacings) > 0 else 0
        
        return {
            'bins': bins,
            'centers': centers,
            'bin_width': avg_bin_width,
            'avg_spacing': avg_spacing,
            'orientation': orientation,
            'extent': self._compute_extent(bins)
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
    
    def _compute_histogram_scores(self, feat: Dict, hist_ctx: Dict, orientation: Orientation) -> Dict[str, float]:
        scores = {'scale_label': 0.0, 'tick_label': 0.0, 'axis_title': 0.0}
        
        nx, ny = feat['nx'], feat['ny']
        rel_w, rel_h = feat['rel_w'], feat['rel_h']
        aspect = feat['aspect']
        is_numeric = feat['is_numeric']
        
        # === SCALE LABEL SCORING ===
        # Histograms have:
        # - Vertical: Y-axis = frequency (left), X-axis = bins (bottom)
        # - Both axes typically have numeric scales
        
        # Small size
        if rel_w < self.params['scale_size_max_width'] and rel_h < self.params['scale_size_max_height']:
            scores['scale_label'] += 4.0
        
        # Position based on orientation
        if orientation == Orientation.VERTICAL:
            # Y-axis (frequency) on left
            if nx < self.params['edge_threshold']:
                scores['scale_label'] += self.params['frequency_axis_weight']
            # X-axis (bins) on bottom
            elif ny > (1.0 - self.params['edge_threshold']):
                scores['scale_label'] += self.params['bottom_edge_weight']
        else:  # horizontal
            # X-axis (frequency) on bottom
            if ny > (1.0 - self.params['edge_threshold']):
                scores['scale_label'] += self.params['frequency_axis_weight']
            # Y-axis (bins) on left
            elif nx < self.params['edge_threshold']:
                scores['scale_label'] += self.params['left_edge_weight']
        
        # Numeric boost (very strong for histograms)
        if is_numeric:
            scores['scale_label'] += self.params['numeric_boost']
        
        # Continuous scale detection
        if is_numeric and is_continuous_scale(feat['text']):
            scores['scale_label'] += self.params['continuous_scale_weight']
        
        # === TICK LABEL SCORING ===
        # Histograms typically don't have tick labels (both axes are scales)
        # Only if categorical bins are detected
        if not is_numeric:
            if orientation == Orientation.VERTICAL and ny > 0.8:
                scores['tick_label'] += 2.0
            elif orientation == Orientation.HORIZONTAL and nx < 0.2:
                scores['tick_label'] += 2.0
        
        # === AXIS TITLE SCORING ===
        # Large size
        if rel_w > self.params['title_size_min'] or rel_h > 0.08:
            scores['axis_title'] += 5.5
        
        # Extreme aspect ratio
        if aspect > self.params['title_aspect_min'] or aspect < 0.16:
            scores['axis_title'] += 5.0
        
        # Non-numeric
        if not is_numeric:
            scores['axis_title'] += 2.5
        
        return scores
    
    
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
        return min(1.0, max(0.0, avg_margin * 0.32))
    
    def _empty_result(self) -> ClassificationResult:
        return ClassificationResult(
            scale_labels=[], tick_labels=[], axis_titles=[],
            confidence=0.0, metadata={'error': 'Invalid input'}
        )
    
    def compute_feature_scores(self, label_features: Dict, region_scores: Dict, element_context: Optional[Dict]) -> Dict[str, float]:
        """
        This is a placeholder to satisfy the abstract method requirement.
        The main logic is in _compute_histogram_scores.
        """
        return self._compute_histogram_scores(label_features, element_context, element_context.get('orientation', Orientation.VERTICAL))
