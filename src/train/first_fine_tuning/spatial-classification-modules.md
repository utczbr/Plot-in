```python
# bar_chart_classifier.py
"""
Bar Chart Specialized Classifier - Optimized for vertical/horizontal bar charts
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class BarClassificationResult:
    scale_labels: List[Dict]
    tick_labels: List[Dict]
    axis_titles: List[Dict]
    confidence: float
    metadata: Dict

class BarChartClassifier:
    """Specialized classifier optimized exclusively for bar charts"""
    
    def __init__(self, params: Dict = None):
        self.params = params or self._get_default_params()
        
    def _get_default_params(self) -> Dict:
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
        bars: List[Dict],
        img_width: int,
        img_height: int,
        orientation: str
    ) -> BarClassificationResult:
        """
        Bar chart specific classification with context-aware scoring
        """
        if not axis_labels:
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
        
        return BarClassificationResult(
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
            is_numeric = self._is_numeric(text)
            
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
    
    def _compute_bar_context(self, bars: List[Dict], w: int, h: int, orientation: str) -> Dict:
        if not bars:
            return {}
        
        centers = []
        for bar in bars:
            x1, y1, x2, y2 = bar['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers.append((cx, cy))
        
        centers = np.array(centers)
        
        if orientation == 'vertical':
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
            'left': min(all_x),
            'right': max(all_x),
            'top': min(all_y),
            'bottom': max(all_y)
        }
    
    def _compute_bar_scores(self, feat: Dict, bar_ctx: Dict, orientation: str) -> Dict[str, float]:
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
        if orientation == 'vertical':
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
        if orientation == 'vertical':
            # Bottom region for vertical bars
            if ny > 0.75:
                scores['tick_label'] += self.params['tick_bottom_weight']
            
            # Alignment with bar centers
            if bar_ctx:
                alignment_score = self._compute_bar_alignment_score(
                    feat['cx'], feat['cy'], bar_ctx, 'vertical'
                )
                scores['tick_label'] += alignment_score * self.params['bar_alignment_weight']
        else:
            # Left region for horizontal bars
            if nx < 0.25:
                scores['tick_label'] += self.params['tick_left_weight']
            
            # Alignment with bar centers
            if bar_ctx:
                alignment_score = self._compute_bar_alignment_score(
                    feat['cx'], feat['cy'], bar_ctx, 'horizontal'
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
    
    def _compute_bar_alignment_score(self, cx: float, cy: float, bar_ctx: Dict, orientation: str) -> float:
        centers = bar_ctx['centers']
        avg_spacing = bar_ctx['avg_spacing']
        
        if orientation == 'vertical':
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
                                orientation: str, w: int, h: int) -> List[Dict]:
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
                if orientation == 'vertical':
                    dist = abs(tx - bx)
                else:
                    dist = abs(ty - by)
                min_dist = min(min_dist, dist)
            
            # Only keep if reasonably aligned
            threshold = 0.15 * (w if orientation == 'vertical' else h)
            if min_dist < threshold:
                aligned_ticks.append(tick)
        
        return aligned_ticks
    
    def _is_numeric(self, text: str) -> bool:
        if not text:
            return False
        try:
            float(text.replace(',', '').replace('%', '').replace('$', ''))
            return True
        except (ValueError, TypeError):
            return False
    
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
    
    def _empty_result(self) -> BarClassificationResult:
        return BarClassificationResult(
            scale_labels=[], tick_labels=[], axis_titles=[],
            confidence=0.0, metadata={'error': 'Invalid input'}
        )
```

```python
# line_chart_classifier.py
"""
Line Chart Specialized Classifier - Optimized for continuous line plots
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class LineClassificationResult:
    scale_labels: List[Dict]
    tick_labels: List[Dict]
    axis_titles: List[Dict]
    confidence: float
    metadata: Dict

class LineChartClassifier:
    """Specialized classifier optimized exclusively for line charts"""
    
    def __init__(self, params: Dict = None):
        self.params = params or self._get_default_params()
    
    def _get_default_params(self) -> Dict:
        return {
            # Line-specific thresholds
            'scale_size_max_width': 0.07,
            'scale_size_max_height': 0.035,
            'numeric_boost': 4.0,
            
            # Position weights
            'left_edge_weight': 6.5,
            'right_edge_weight': 5.0,
            'bottom_edge_weight': 6.0,
            'top_edge_weight': 3.0,
            
            # Line-specific features
            'line_proximity_weight': 4.5,
            'value_range_weight': 3.5,
            
            # Title detection
            'title_size_min': 0.12,
            'title_aspect_min': 6.0,
            
            # Thresholds
            'classification_threshold': 2.5,
            'edge_threshold_x': 0.20,
            'edge_threshold_y': 0.85
        }
    
    def classify(
        self,
        axis_labels: List[Dict],
        line_points: List[Dict],
        img_width: int,
        img_height: int,
        orientation: str = 'vertical'
    ) -> LineClassificationResult:
        """
        Line chart specific classification
        """
        if not axis_labels:
            return self._empty_result()
        
        # Extract line-specific features
        label_features = self._extract_line_features(axis_labels, img_width, img_height)
        line_context = self._compute_line_context(line_points, img_width, img_height)
        
        # Classification
        classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
        all_scores = []
        
        for feat in label_features:
            scores = self._compute_line_scores(feat, line_context)
            all_scores.append(scores)
            
            best_class = max(scores, key=scores.get)
            threshold = self.params['classification_threshold']
            
            if scores[best_class] > threshold:
                # Additional validation for scale labels
                if best_class == 'scale_label' and feat['is_numeric']:
                    # Extract numeric value for calibration
                    try:
                        numeric_val = float(feat['text'].replace(',', '').replace('%', ''))
                        feat['label']['cleaned_value'] = numeric_val
                        classified[best_class].append(feat['label'])
                    except:
                        classified['axis_title'].append(feat['label'])
                else:
                    classified[best_class].append(feat['label'])
            else:
                # Default: numeric = scale, non-numeric = title
                if feat['is_numeric']:
                    classified['scale_label'].append(feat['label'])
                else:
                    classified['axis_title'].append(feat['label'])
        
        # Separate X and Y axis scales
        x_scales, y_scales = self._separate_xy_scales(
            classified['scale_label'], img_width, img_height
        )
        
        confidence = self._compute_confidence(all_scores)
        
        metadata = {
            'chart_type': 'line',
            'num_points': len(line_points),
            'x_scales': len(x_scales),
            'y_scales': len(y_scales),
            'line_extent': line_context.get('extent', {})
        }
        
        return LineClassificationResult(
            scale_labels=classified['scale_label'],
            tick_labels=classified['tick_label'],
            axis_titles=classified['axis_title'],
            confidence=confidence,
            metadata=metadata
        )
    
    def _extract_line_features(self, labels: List[Dict], w: int, h: int) -> List[Dict]:
        features = []
        for label in labels:
            x1, y1, x2, y2 = label['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            
            text = label.get('text', '')
            is_numeric = self._is_numeric(text)
            
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
    
    def _compute_line_context(self, points: List[Dict], w: int, h: int) -> Dict:
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
        
        extent = {
            'left': x_min,
            'right': x_max,
            'top': y_min,
            'bottom': y_max
        }
        
        return {
            'positions': positions,
            'extent': extent,
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'num_points': len(points)
        }
    
    def _compute_line_scores(self, feat: Dict, line_ctx: Dict) -> Dict[str, float]:
        scores = {'scale_label': 0.0, 'tick_label': 0.0, 'axis_title': 0.0}
        
        nx, ny = feat['nx'], feat['ny']
        rel_w, rel_h = feat['rel_w'], feat['rel_h']
        aspect = feat['aspect']
        is_numeric = feat['is_numeric']
        
        # === SCALE LABEL SCORING (Primary for line charts) ===
        # Small size
        if rel_w < self.params['scale_size_max_width'] and rel_h < self.params['scale_size_max_height']:
            scores['scale_label'] += 4.0
        
        # Edge positions
        if nx < self.params['edge_threshold_x']:
            scores['scale_label'] += self.params['left_edge_weight']
        elif nx > (1.0 - self.params['edge_threshold_x']):
            scores['scale_label'] += self.params['right_edge_weight']
        
        if ny > self.params['edge_threshold_y']:
            scores['scale_label'] += self.params['bottom_edge_weight']
        elif ny < 0.15:
            scores['scale_label'] += self.params['top_edge_weight']
        
        # Numeric boost (critical for line charts)
        if is_numeric:
            scores['scale_label'] += self.params['numeric_boost']
        
        # Proximity to line extent
        if line_ctx:
            proximity_score = self._compute_line_proximity(feat, line_ctx)
            scores['scale_label'] += proximity_score * self.params['line_proximity_weight']
        
        # === TICK LABEL SCORING (Rare in line charts) ===
        # Line charts typically don't have tick labels
        # Only score if non-numeric and near data region
        if not is_numeric and ny > 0.7:
            scores['tick_label'] += 1.0
        
        # === AXIS TITLE SCORING ===
        # Large size
        if rel_w > self.params['title_size_min'] or rel_h > 0.08:
            scores['axis_title'] += 5.0
        
        # Extreme aspect ratio
        if aspect > self.params['title_aspect_min'] or aspect < 0.15:
            scores['axis_title'] += 4.0
        
        # Non-numeric
        if not is_numeric:
            scores['axis_title'] += 2.0
        
        return scores
    
    def _compute_line_proximity(self, feat: Dict, line_ctx: Dict) -> float:
        """Compute how close label is to line extent"""
        extent = line_ctx['extent']
        cx, cy = feat['cx'], feat['cy']
        
        # Distance to line extent boundaries
        dist_left = abs(cx - extent['left'])
        dist_right = abs(cx - extent['right'])
        dist_top = abs(cy - extent['top'])
        dist_bottom = abs(cy - extent['bottom'])
        
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        
        # Normalize and invert (closer = higher score)
        max_dim = max(extent['right'] - extent['left'], extent['bottom'] - extent['top'])
        proximity = 1.0 - (min_dist / (max_dim + 1e-6))
        
        return max(0.0, proximity)
    
    def _separate_xy_scales(
        self, scale_labels: List[Dict], w: int, h: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Separate into X-axis and Y-axis scales"""
        x_scales = []
        y_scales = []
        
        for label in scale_labels:
            x1, y1, x2, y2 = label['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            nx, ny = cx / w, cy / h
            
            # X-axis: bottom region
            if ny > 0.75:
                label['axis'] = 'x'
                x_scales.append(label)
            # Y-axis: left or right edges
            elif nx < 0.25 or nx > 0.75:
                label['axis'] = 'y'
                y_scales.append(label)
            else:
                # Ambiguous: use position bias
                if ny > nx:
                    label['axis'] = 'x'
                    x_scales.append(label)
                else:
                    label['axis'] = 'y'
                    y_scales.append(label)
        
        return x_scales, y_scales
    
    def _is_numeric(self, text: str) -> bool:
        if not text:
            return False
        try:
            text_clean = text.replace(',', '').replace('%', '').replace('$', '').strip()
            float(text_clean)
            return True
        except (ValueError, TypeError):
            return False
    
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
        return min(1.0, max(0.0, avg_margin * 0.3))
    
    def _empty_result(self) -> LineClassificationResult:
        return LineClassificationResult(
            scale_labels=[], tick_labels=[], axis_titles=[],
            confidence=0.0, metadata={'error': 'Invalid input'}
        )
```

```python
# scatter_chart_classifier.py
"""
Scatter Chart Specialized Classifier - Optimized for scatter plots
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ScatterClassificationResult:
    scale_labels: List[Dict]
    tick_labels: List[Dict]
    axis_titles: List[Dict]
    confidence: float
    metadata: Dict

class ScatterChartClassifier:
    """Specialized classifier optimized exclusively for scatter plots"""
    
    def __init__(self, params: Dict = None):
        self.params = params or self._get_default_params()
    
    def _get_default_params(self) -> Dict:
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
        scatter_points: List[Dict],
        img_width: int,
        img_height: int,
        orientation: str = 'vertical'
    ) -> ScatterClassificationResult:
        """
        Scatter plot specific classification
        """
        if not axis_labels:
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
        
        return ScatterClassificationResult(
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
            is_numeric = self._is_numeric(text)
            
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
        extent = scatter_ctx['extent']
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
    
    def _is_numeric(self, text: str) -> bool:
        if not text:
            return False
        try:
            text_clean = text.replace(',', '').replace('%', '').replace('$', '').strip()
            float(text_clean)
            return True
        except (ValueError, TypeError):
            return False
    
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
    
    def _empty_result(self) -> ScatterClassificationResult:
        return ScatterClassificationResult(
            scale_labels=[], tick_labels=[], axis_titles=[],
            confidence=0.0, metadata={'error': 'Invalid input'}
        )
```

```python
# box_chart_classifier.py
"""
Box Chart Specialized Classifier - Optimized for box plots
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class BoxClassificationResult:
    scale_labels: List[Dict]
    tick_labels: List[Dict]
    axis_titles: List[Dict]
    confidence: float
    metadata: Dict

class BoxChartClassifier:
    """Specialized classifier optimized exclusively for box plots"""
    
    def __init__(self, params: Dict = None):
        self.params = params or self._get_default_params()
    
    def _get_default_params(self) -> Dict:
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
        boxes: List[Dict],
        img_width: int,
        img_height: int,
        orientation: str
    ) -> BoxClassificationResult:
        """
        Box plot specific classification
        """
        if not axis_labels:
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
        
        confidence = self._compute_confidence(all_scores)
        
        metadata = {
            'chart_type': 'box',
            'orientation': orientation,
            'num_boxes': len(boxes),
            'box_spacing': box_context.get('avg_spacing', 0)
        }
        
        return BoxClassificationResult(
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
            is_numeric = self._is_numeric(text)
            
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
    
    def _compute_box_context(self, boxes: List[Dict], w: int, h: int, orientation: str) -> Dict:
        if not boxes:
            return {'orientation': orientation}
        
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers.append((cx, cy))
        
        centers = np.array(centers)
        
        if orientation == 'vertical':
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
            'left': min(all_x),
            'right': max(all_x),
            'top': min(all_y),
            'bottom': max(all_y)
        }
    
    def _compute_box_scores(self, feat: Dict, box_ctx: Dict, orientation: str) -> Dict[str, float]:
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
        if orientation == 'vertical':
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
        
        if orientation == 'vertical':
            # Bottom region for tick labels
            if ny > 0.78:
                scores['tick_label'] += self.params['tick_alignment_weight']
            
            # Alignment with box centers
            if box_ctx:
                alignment_score = self._compute_box_alignment(feat, box_ctx, 'vertical')
                scores['tick_label'] += alignment_score * self.params['box_spacing_weight']
        else:  # horizontal
            # Left region for tick labels
            if nx < 0.22:
                scores['tick_label'] += self.params['tick_alignment_weight']
            
            # Alignment with box centers
            if box_ctx:
                alignment_score = self._compute_box_alignment(feat, box_ctx, 'horizontal')
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
    
    def _compute_box_alignment(self, feat: Dict, box_ctx: Dict, orientation: str) -> float:
        """Compute alignment score with box centers"""
        centers = box_ctx['centers']
        avg_spacing = box_ctx['avg_spacing']
        cx, cy = feat['cx'], feat['cy']
        
        if orientation == 'vertical':
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
    
    def _is_numeric(self, text: str) -> bool:
        if not text:
            return False
        try:
            text_clean = text.replace(',', '').replace('%', '').replace('$', '').strip()
            float(text_clean)
            return True
        except (ValueError, TypeError):
            return False
    
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
    
    def _empty_result(self) -> BoxClassificationResult:
        return BoxClassificationResult(
            scale_labels=[], tick_labels=[], axis_titles=[],
            confidence=0.0, metadata={'error': 'Invalid input'}
        )
```

```python
# histogram_chart_classifier.py
"""
Histogram Specialized Classifier - Optimized for histogram charts
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class HistogramClassificationResult:
    scale_labels: List[Dict]
    tick_labels: List[Dict]
    axis_titles: List[Dict]
    confidence: float
    metadata: Dict

class HistogramChartClassifier:
    """Specialized classifier optimized exclusively for histograms"""
    
    def __init__(self, params: Dict = None):
        self.params = params or self._get_default_params()
    
    def _get_default_params(self) -> Dict:
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
        bins: List[Dict],
        img_width: int,
        img_height: int,
        orientation: str = 'vertical'
    ) -> HistogramClassificationResult:
        """
        Histogram specific classification
        """
        if not axis_labels:
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
        
        return HistogramClassificationResult(
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
            is_numeric = self._is_numeric(text)
            
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
    
    def _compute_histogram_context(self, bins: List[Dict], w: int, h: int, orientation: str) -> Dict:
        if not bins:
            return {'orientation': orientation}
        
        centers = []
        widths = []
        for bin_elem in bins:
            x1, y1, x2, y2 = bin_elem['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centers.append((cx, cy))
            
            if orientation == 'vertical':
                widths.append(x2 - x1)
            else:
                widths.append(y2 - y1)
        
        centers = np.array(centers)
        avg_bin_width = np.mean(widths) if widths else 0
        
        if orientation == 'vertical':
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
            'left': min(all_x),
            'right': max(all_x),
            'top': min(all_y),
            'bottom': max(all_y)
        }
    
    def _compute_histogram_scores(self, feat: Dict, hist_ctx: Dict, orientation: str) -> Dict[str, float]:
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
        if orientation == 'vertical':
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
        if is_numeric and self._is_continuous_scale(feat['text']):
            scores['scale_label'] += self.params['continuous_scale_weight']
        
        # === TICK LABEL SCORING ===
        # Histograms typically don't have tick labels (both axes are scales)
        # Only if categorical bins are detected
        if not is_numeric:
            if orientation == 'vertical' and ny > 0.8:
                scores['tick_label'] += 2.0
            elif orientation == 'horizontal' and nx < 0.2:
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
    
    def _is_continuous_scale(self, text: str) -> bool:
        """Check if text represents a continuous scale value (e.g., decimal, range)"""
        if not text:
            return False
        
        # Check for decimal points, ranges, or scientific notation
        has_decimal = '.' in text
        has_range = '-' in text and text.count('-') == 1 and text.index('-') > 0
        has_scientific = 'e' in text.lower() or 'E' in text
        
        return has_decimal or has_range or has_scientific
    
    def _is_numeric(self, text: str) -> bool:
        if not text:
            return False
        try:
            text_clean = text.replace(',', '').replace('%', '').replace('$', '').strip()
            float(text_clean)
            return True
        except (ValueError, TypeError):
            return False
    
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
    
    def _empty_result(self) -> HistogramClassificationResult:
        return HistogramClassificationResult(
            scale_labels=[], tick_labels=[], axis_titles=[],
            confidence=0.0, metadata={'error': 'Invalid input'}
        )
```

```python
# hypertuning_optimizer.py
"""
Hyperparameter optimization for chart-specific classifiers
"""
import numpy as np
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import itertools

@dataclass
class HypertuningResult:
    chart_type: str
    optimal_params: Dict
    best_accuracy: float
    best_f1_score: float
    evaluation_metrics: Dict

class ChartClassifierHypertuner:
    """Hyperparameter optimizer for each chart type"""
    
    def __init__(self, chart_type: str):
        self.chart_type = chart_type
        self.param_space = self._get_param_space(chart_type)
    
    def _get_param_space(self, chart_type: str) -> Dict:
        """Define search space for each chart type"""
        
        if chart_type == 'bar':
            return {
                'scale_left_weight': [5.0, 5.5, 6.0, 6.5, 7.0],
                'scale_bottom_weight': [5.0, 5.5, 6.0, 6.5],
                'tick_bottom_weight': [6.0, 6.5, 7.0, 7.5, 8.0],
                'bar_alignment_weight': [4.0, 4.5, 5.0, 5.5, 6.0],
                'numeric_boost': [2.5, 3.0, 3.5, 4.0],
                'classification_threshold': [1.5, 2.0, 2.5, 3.0]
            }
        
        elif chart_type == 'line':
            return {
                'left_edge_weight': [6.0, 6.5, 7.0, 7.5],
                'bottom_edge_weight': [5.5, 6.0, 6.5, 7.0],
                'numeric_boost': [3.5, 4.0, 4.5, 5.0],
                'line_proximity_weight': [4.0, 4.5, 5.0, 5.5],
                'classification_threshold': [2.0, 2.5, 3.0, 3.5]
            }
        
        elif chart_type == 'scatter':
            return {
                'left_edge_weight': [6.5, 7.0, 7.5, 8.0],
                'bottom_edge_weight': [6.0, 6.5, 7.0, 7.5],
                'numeric_boost': [4.5, 5.0, 5.5, 6.0],
                'point_cloud_proximity_weight': [4.5, 5.0, 5.5, 6.0],
                'classification_threshold': [2.5, 3.0, 3.5, 4.0]
            }
        
        elif chart_type == 'box':
            return {
                'scale_edge_weight': [5.5, 6.0, 6.5, 7.0],
                'tick_alignment_weight': [5.0, 5.5, 6.0, 6.5],
                'numeric_boost': [3.0, 3.5, 4.0, 4.5],
                'box_spacing_weight': [3.5, 4.0, 4.5, 5.0],
                'classification_threshold': [2.0, 2.2, 2.5, 2.8]
            }
        
        elif chart_type == 'histogram':
            return {
                'frequency_axis_weight': [5.0, 5.5, 6.0, 6.5],
                'bottom_edge_weight': [5.5, 6.0, 6.5, 7.0],
                'numeric_boost': [3.5, 4.0, 4.5, 5.0],
                'continuous_scale_weight': [3.0, 3.5, 4.0, 4.5],
                'classification_threshold': [2.0, 2.3, 2.6, 3.0]
            }
        
        else:
            return {}
    
    def optimize_grid_search(
        self,
        training_data: List[Dict],
        validation_data: List[Dict],
        max_combinations: int = 100
    ) -> HypertuningResult:
        """
        Grid search optimization with early stopping
        """
        # Generate parameter combinations
        param_names = list(self.param_space.keys())
        param_values = [self.param_space[name] for name in param_names]
        
        # Sample combinations if too many
        all_combinations = list(itertools.product(*param_values))
        if len(all_combinations) > max_combinations:
            np.random.shuffle(all_combinations)
            combinations = all_combinations[:max_combinations]
        else:
            combinations = all_combinations
        
        best_score = 0.0
        best_params = None
        best_metrics = None
        
        print(f"Optimizing {self.chart_type} classifier...")
        print(f"Testing {len(combinations)} parameter combinations")
        
        for i, param_combo in enumerate(combinations):
            # Build parameter dict
            params = {name: val for name, val in zip(param_names, param_combo)}
            
            # Evaluate on training data
            train_metrics = self._evaluate_params(params, training_data)
            
            # Evaluate on validation data
            val_metrics = self._evaluate_params(params, validation_data)
            
            # Combined score (weighted average)
            combined_score = 0.7 * val_metrics['f1_score'] + 0.3 * train_metrics['f1_score']
            
            if combined_score > best_score:
                best_score = combined_score
                best_params = params
                best_metrics = val_metrics
                
                print(f"  Iteration {i+1}: New best F1={combined_score:.4f}")
        
        print(f"Optimization complete. Best F1: {best_score:.4f}")
        
        return HypertuningResult(
            chart_type=self.chart_type,
            optimal_params=best_params,
            best_accuracy=best_metrics['accuracy'],
            best_f1_score=best_metrics['f1_score'],
            evaluation_metrics=best_metrics
        )
    
    def optimize_random_search(
        self,
        training_data: List[Dict],
        validation_data: List[Dict],
        n_iterations: int = 50
    ) -> HypertuningResult:
        """
        Random search optimization
        """
        best_score = 0.0
        best_params = None
        best_metrics = None
        
        print(f"Optimizing {self.chart_type} classifier with random search...")
        print(f"Running {n_iterations} iterations")
        
        for i in range(n_iterations):
            # Sample random parameters
            params = {}
            for param_name, param_values in self.param_space.items():
                params[param_name] = np.random.choice(param_values)
            
            # Evaluate
            train_metrics = self._evaluate_params(params, training_data)
            val_metrics = self._evaluate_params(params, validation_data)
            
            combined_score = 0.7 * val_metrics['f1_score'] + 0.3 * train_metrics['f1_score']
            
            if combined_score > best_score:
                best_score = combined_score
                best_params = params
                best_metrics = val_metrics
                
                print(f"  Iteration {i+1}: New best F1={combined_score:.4f}")
        
        print(f"Optimization complete. Best F1: {best_score:.4f}")
        
        return HypertuningResult(
            chart_type=self.chart_type,
            optimal_params=best_params,
            best_accuracy=best_metrics['accuracy'],
            best_f1_score=best_metrics['f1_score'],
            evaluation_metrics=best_metrics
        )
    
    def _evaluate_params(self, params: Dict, data: List[Dict]) -> Dict:
        """
        Evaluate parameter configuration on dataset
        """
        from bar_chart_classifier import BarChartClassifier
        from line_chart_classifier import LineChartClassifier
        from scatter_chart_classifier import ScatterChartClassifier
        from box_chart_classifier import BoxChartClassifier
        from histogram_chart_classifier import HistogramChartClassifier
        
        # Initialize classifier with params
        if self.chart_type == 'bar':
            classifier = BarChartClassifier(params)
        elif self.chart_type == 'line':
            classifier = LineChartClassifier(params)
        elif self.chart_type == 'scatter':
            classifier = ScatterChartClassifier(params)
        elif self.chart_type == 'box':
            classifier = BoxChartClassifier(params)
        elif self.chart_type == 'histogram':
            classifier = HistogramChartClassifier(params)
        else:
            raise ValueError(f"Unknown chart type: {self.chart_type}")
        
        # Evaluate on data
        correct = 0
        total = 0
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        for sample in data:
            result = classifier.classify(
                axis_labels=sample['axis_labels'],
                chart_elements=sample['chart_elements'],
                img_width=sample['img_width'],
                img_height=sample['img_height'],
                orientation=sample.get('orientation', 'vertical')
            )
            
            # Compare with ground truth
            gt_scale = set([str(lbl['xyxy']) for lbl in sample['gt_scale_labels']])
            gt_tick = set([str(lbl['xyxy']) for lbl in sample['gt_tick_labels']])
            gt_title = set([str(lbl['xyxy']) for lbl in sample['gt_axis_titles']])
            
            pred_scale = set([str(lbl['xyxy']) for lbl in result.scale_labels])
            pred_tick = set([str(lbl['xyxy']) for lbl in result.tick_labels])
            pred_title = set([str(lbl['xyxy']) for lbl in result.axis_titles])
            
            # Compute metrics
            tp += len(gt_scale & pred_scale) + len(gt_tick & pred_tick) + len(gt_title & pred_title)
            fp += len(pred_scale - gt_scale) + len(pred_tick - gt_tick) + len(pred_title - gt_title)
            fn += len(gt_scale - pred_scale) + len(gt_tick - pred_tick) + len(gt_title - pred_title)
            
            total += len(gt_scale) + len(gt_tick) + len(gt_title)
        
        # Compute metrics
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def save_results(self, result: HypertuningResult, output_path: str):
        """Save hypertuning results to JSON"""
        output = {
            'chart_type': result.chart_type,
            'optimal_parameters': result.optimal_params,
            'best_accuracy': result.best_accuracy,
            'best_f1_score': result.best_f1_score,
            'evaluation_metrics': result.evaluation_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {output_path}")
```

All classifier modules are now complete with specialized, non-reused logic for each chart type (bar, line, scatter, box, histogram) and a comprehensive hypertuning system.
