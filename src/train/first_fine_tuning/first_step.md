```python
"""
hyperparameter_tuner.py
Bayesian hyperparameter optimization system for chart-type-specific spatial classification
Uses Optuna for automated tuning of LYLAA scoring weights per chart type
"""

import optuna
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import cv2
from sklearn.cluster import DBSCAN

# Import your existing modules
from spatial_classification_enhanced import _compute_chart_element_context_features, _compute_octant_region_scores
from analysis import classify_chart_enhanced, run_inference, CLASS_MAP_BAR, CLASS_MAP_BOX, CLASS_MAP_LINE, CLASS_MAP_SCATTER, detect_bar_orientation

@dataclass
class ChartTypeParams:
    """Type-specific parameter configuration"""
    # Scale label features
    scale_size_weight: float = 3.0
    scale_aspect_weight: float = 2.5
    scale_region_boost: float = 5.0
    scale_center_dist_weight: float = 2.0
    
    # Tick label features
    tick_size_weight: float = 2.5
    tick_position_weight: float = 1.5
    tick_spacing_weight: float = 5.0
    tick_alignment_boost: float = 4.0
    
    # Title features
    title_aspect_weight: float = 4.0
    title_size_weight: float = 3.0
    title_region_weight: float = 4.0
    
    # Chart-specific features
    spacing_multiplier: float = 1.5
    context_distance_weight: float = 1.0
    numeric_boost: float = 2.0
    
    # Type-specific penalties/boosts
    dual_axis_penalty: float = 0.7
    whisker_dist_weight: float = 3.5
    trend_fit_weight: float = 4.0
    
    # Classification threshold
    classification_threshold: float = 1.5


class ParameterizedSpatialClassifier:
    """Spatial classifier with injectable type-specific parameters"""
    
    def __init__(self, chart_type: str, params: ChartTypeParams):
        self.chart_type = chart_type
        self.params = params
    
    def classify(self, axis_labels, chart_elements, img_width, img_height, orientation, mode='precise'):
        """Modified classification with parameterized scoring"""
        if not axis_labels:
            return {'scale_label': [], 'tick_label': [], 'axis_title': []}
        
        classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
        
        # Extract features (same as original)
        label_features = []
        for label in axis_labels:
            x1, y1, x2, y2 = label['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            
            label_features.append({
                'label': label,
                'centroid': (cx, cy),
                'normalized_pos': (cx / img_width, cy / img_height),
                'dimensions': (width, height),
                'aspect_ratio': width / (height + 1e-6),
                'relative_size': (width / img_width, height / img_height)
            })
        
        # Compute element context if available
        element_context = _compute_chart_element_context_features(
            chart_elements, self.chart_type, img_width, img_height, orientation
        )
        
        # Compute region scores
        for feat in label_features:
            region_scores = _compute_octant_region_scores(feat['normalized_pos'], img_width, img_height)
            
            nx, ny = feat['normalized_pos']
            rel_width, rel_height = feat['relative_size']
            aspect_ratio = feat['aspect_ratio']
            cx, cy = feat['centroid']
            
            scores = self._compute_parameterized_scores(
                feat, nx, ny, rel_width, rel_height, aspect_ratio, region_scores,
                element_context, orientation
            )
            
            # Classification decision
            best_class = max(scores, key=scores.get)
            best_score = scores[best_class]
            
            if best_score > self.params.classification_threshold:
                classified[best_class].append(feat['label'])
            else:
                classified['scale_label'].append(feat['label'])
        
        # Post-process with DBSCAN if needed
        if len(classified['scale_label']) > 3:
            classified['scale_label'] = _cluster_scale_labels_weighted_dbscan(
                classified['scale_label'], img_width, img_height, orientation, {}
            )
        
        return classified
    
    def _compute_parameterized_scores(self, feat, nx, ny, rel_width, rel_height, aspect_ratio, 
                                      region_scores, element_context, orientation):
        """Parameterized scoring function"""
        scores = {'scale_label': 0.0, 'tick_label': 0.0, 'axis_title': 0.0}
        
        # --- Enhanced Scale Label Features ---
        if rel_width < 0.08 and rel_height < 0.04: 
            scores['scale_label'] += self.params.scale_size_weight
        if 0.5 < aspect_ratio < 3.5: 
            scores['scale_label'] += self.params.scale_aspect_weight
        
        left_right_max = max(region_scores['left_y_axis'], region_scores['right_y_axis'])
        if left_right_max > 0.5: 
            scores['scale_label'] += self.params.scale_region_boost * left_right_max
        if region_scores['bottom_x_axis'] > 0.5:
            if orientation == 'vertical':
                scores['tick_label'] += self.params.scale_region_boost * region_scores['bottom_x_axis']
            else:
                scores['scale_label'] += self.params.scale_region_boost * region_scores['bottom_x_axis']
        
        center_dist = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
        if center_dist > 0.3: 
            scores['scale_label'] += self.params.scale_center_dist_weight * (center_dist - 0.3)
        
        # Numeric hint if text available
        label_text = feat['label'].get('text', '') if 'text' in feat['label'] else ''
        if label_text:
            numeric_chars = sum(c.isdigit() or c in '.-+eE%' for c in label_text)
            total_chars = len(label_text)
            if total_chars > 0:
                numeric_ratio = numeric_chars / total_chars
                scores['scale_label'] += self.params.numeric_boost * numeric_ratio
                scores['tick_label'] += self.params.context_distance_weight * (1 - numeric_ratio)
        
        # --- Generic Tick and Title Features ---
        if 0.02 < rel_width < 0.25 and 0.015 < rel_height < 0.10: 
            scores['tick_label'] += self.params.tick_size_weight
        if 0.15 < nx < 0.85 or 0.15 < ny < 0.85: 
            scores['tick_label'] += self.params.tick_position_weight
        
        if aspect_ratio > 4.0 or aspect_ratio < 0.25: 
            scores['axis_title'] += self.params.title_aspect_weight
        if rel_width > 0.15 or rel_height > 0.08: 
            scores['axis_title'] += self.params.title_size_weight
        if region_scores['top_title'] > 0.3: 
            scores['axis_title'] += self.params.title_region_weight * region_scores['top_title']
        if (nx < 0.08 or nx > 0.92) and aspect_ratio < 0.4: 
            scores['axis_title'] += self.params.title_aspect_weight
        # Assuming width and height in pixels, boost if large
        width, height = feat['dimensions']
        if width > 100 or height > 50: 
            scores['axis_title'] += self.params.title_size_weight / 1.5
        
        # --- Context-Specific Features ---
        if element_context:
            el_extent = element_context['extent']
            el_positions = element_context['positions']
            avg_spacing = element_context['avg_spacing']
            
            if orientation == 'vertical':
                if cy > el_extent['bottom']: 
                    scores['tick_label'] += self.params.tick_alignment_boost * np.exp(-(cy - el_extent['bottom']) / 50.0)
                x_distances = np.abs(el_positions[:, 0] - cx)
                min_x_dist = np.min(x_distances)
                
                if self.chart_type == 'bar' and min_x_dist < avg_spacing * self.params.spacing_multiplier:
                    scores['tick_label'] += self.params.tick_spacing_weight * np.exp(-min_x_dist / (avg_spacing + 1e-6))
                elif self.chart_type == 'box' and min_x_dist < element_context.get('median_box_width', 50) * 1.2:
                    scores['tick_label'] += self.params.tick_spacing_weight * np.exp(-min_x_dist / (element_context['median_box_width'] + 1e-6))
                elif self.chart_type in ['scatter', 'line'] and min_x_dist < element_context['x_spread'] * 0.1:
                    scores['tick_label'] += self.params.tick_alignment_boost
            else:  # horizontal
                if cx < el_extent['left']: 
                    scores['tick_label'] += self.params.tick_alignment_boost * np.exp(-(el_extent['left'] - cx) / 50.0)
                y_distances = np.abs(el_positions[:, 1] - cy)
                min_y_dist = np.min(y_distances)
                
                if self.chart_type == 'bar' and min_y_dist < avg_spacing * self.params.spacing_multiplier:
                    scores['tick_label'] += self.params.tick_spacing_weight * np.exp(-min_y_dist / (avg_spacing + 1e-6))
                elif self.chart_type == 'box' and min_y_dist < element_context.get('median_box_height', 50) * 1.2:
                    scores['tick_label'] += self.params.tick_spacing_weight * np.exp(-min_y_dist / (element_context['median_box_height'] + 1e-6))
                elif self.chart_type in ['scatter', 'line'] and min_y_dist < element_context['y_spread'] * 0.1:
                    scores['tick_label'] += self.params.tick_alignment_boost

            # Additional context for bar/box
            if self.chart_type == 'bar' or self.chart_type == 'box':
                if orientation == 'vertical':
                    if abs(cx - el_extent['left']) < 10 or abs(cx - el_extent['right']) < 10:
                        scores['scale_label'] += self.params.context_distance_weight * 3.5
                else:
                    if abs(cy - el_extent['bottom']) < 10:
                        scores['scale_label'] += self.params.context_distance_weight * 3.5

        # --- Type-Specific Adjustments ---
        if self.chart_type == 'scatter':
            # Dual-axis penalty for non-edge positions
            if not ((nx < 0.2 or nx > 0.8) or (ny > 0.8)):
                scores['scale_label'] -= self.params.dual_axis_penalty
            # Boost for numeric scales on both axes
            scores['scale_label'] += self.params.numeric_boost * 0.5
        
        elif self.chart_type == 'box':
            # Penalize labels far from whiskers/box extents
            if element_context:
                box_heights = [el['xyxy'][3] - el['xyxy'][1] for el in chart_elements]
                median_height = np.median(box_heights)
                dist_to_bottom = abs(ny - element_context['extent']['bottom'] / img_height)
                if dist_to_bottom > median_height / img_height * 1.5:
                    scores['scale_label'] -= self.params.whisker_dist_weight * dist_to_bottom
        
        elif self.chart_type == 'line':
            # Trend fit alignment for tick labels
            if element_context and len(element_context['positions']) > 2:
                # Simple linear fit to data points
                x_vals = element_context['positions'][:, 0]
                y_vals = element_context['positions'][:, 1]
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                predicted_cy = slope * cx + intercept
                trend_dist = abs(cy - predicted_cy) / img_height
                if trend_dist < 0.1:
                    scores['tick_label'] += self.params.trend_fit_weight * (1 - trend_dist * 10)
                else:
                    scores['tick_label'] -= self.params.trend_fit_weight * trend_dist
        
        elif self.chart_type == 'bar':
            # Enhanced spacing for tick labels
            if element_context:
                bar_widths = [el['xyxy'][2] - el['xyxy'][0] for el in chart_elements]
                median_width = np.median(bar_widths)
                dist_to_bar = min(np.abs(element_context['positions'][:, 0] - cx))
                if dist_to_bar < median_width * 0.5:
                    scores['tick_label'] += self.params.tick_spacing_weight * 0.5

        return scores


class HyperparameterTuner:
    """Optuna-based tuner for chart-type-specific parameters"""
    
    def __init__(self, dataset_path: Path, models: Dict, validation_split: float = 0.2):
        self.dataset_path = dataset_path
        self.models = models
        self.ground_truth = self._load_ground_truth()
        self.validation_split = validation_split
    
    def _load_ground_truth(self) -> Dict[str, Dict]:
        """Load annotated dataset from ground_truth.json"""
        gt_path = self.dataset_path / 'ground_truth.json'
        if not gt_path.exists():
            raise ValueError(f"Ground truth not found at {gt_path}")
        with open(gt_path, 'r') as f:
            return json.load(f)
    
    def _get_samples(self, chart_type: str) -> List[Tuple[Path, Dict]]:
        """Get image paths and GT for specific type"""
        samples = []
        for img_name, gt in self.ground_truth.items():
            if gt['chart_type'] == chart_type:
                img_path = self.dataset_path / img_name
                if img_path.exists():
                    samples.append((img_path, gt))
        return samples
    
    def _evaluate_params(self, chart_type: str, params: ChartTypeParams, samples: List[Tuple[Path, Dict]]) -> Dict[str, float]:
        """Evaluate parameters on samples"""
        all_true = []
        all_pred = []
        
        for img_path, gt in samples:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # Run detection
            det_model = self.models.get(chart_type)
            class_map = {
                'bar': CLASS_MAP_BAR,
                'box': CLASS_MAP_BOX,
                'line': CLASS_MAP_LINE,
                'scatter': CLASS_MAP_SCATTER
            }[chart_type]
            
            dets_raw = run_inference(det_model, img_path, 0.4, class_map)
            
            detections = {cls: [] for cls in class_map.values()}
            for det in dets_raw:
                detections[class_map[det['cls']]].append(det)
            
            # Orientation
            orientation = 'vertical' if detect_bar_orientation(detections.get('bar', [])) else 'horizontal'
            
            # Chart elements
            elements_key = {'bar': 'bar', 'box': 'box', 'line': 'data_point', 'scatter': 'data_point'}[chart_type]
            chart_elements = detections.get(elements_key, [])
            
            # Classify with parameters
            classifier = ParameterizedSpatialClassifier(chart_type, params)
            classified = classifier.classify(detections.get('axis_labels', []), chart_elements, w, h, orientation)
            
            # Map to labels (0: scale, 1: tick, 2: title)
            pred_labels = []
            true_labels = []
            for label_type, labels in classified.items():
                cls_id = {'scale_label': 0, 'tick_label': 1, 'axis_title': 2}[label_type]
                for lbl in labels:
                    # Find matching GT by bbox IoU or position
                    matched_gt = self._match_label(lbl, gt['axis_labels'])
                    if matched_gt:
                        true_cls = {'scale_label': 0, 'tick_label': 1, 'axis_title': 2}.get(matched_gt['class'])
                        pred_labels.append(cls_id)
                        true_labels.append(true_cls)
            
            all_true.extend(true_labels)
            all_pred.extend(pred_labels)
        
        if not all_true:
            return {'weighted_score': 0.0}
        
        # Metrics
        precision = precision_score(all_true, all_pred, average=None)
        f1 = f1_score(all_true, all_pred, average=None)
        
        metrics = {
            'scale_precision': precision[0] if len(precision) > 0 else 0.0,
            'scale_f1': f1[0] if len(f1) > 0 else 0.0,
            'tick_f1': f1[1] if len(f1) > 1 else 0.0,
            'title_f1': f1[2] if len(f1) > 2 else 0.0,
            'weighted_score': f1_score(all_true, all_pred, average='weighted')
        }
        
        return metrics
    
    def _match_label(self, pred_lbl, gt_labels, iou_threshold=0.7):
        """Match predicted label to GT by bbox IoU"""
        pred_box = pred_lbl['xyxy']
        
        for gt in gt_labels:
            gt_box = gt['xyxy']
            iou = self._compute_iou(pred_box, gt_box)
            if iou > iou_threshold:
                return gt
        return None
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def tune(self, chart_type: str, n_trials: int = 100) -> Dict:
        """Run Optuna tuning for specific type"""
        samples = self._get_samples(chart_type)
        if not samples:
            raise ValueError(f"No samples for {chart_type}")
        
        n_val = int(len(samples) * self.validation_split)
        val_samples = samples[-n_val:] if n_val > 0 else samples
        
        def objective(trial):
            params_dict = {
                'scale_size_weight': trial.suggest_float('scale_size_weight', 1.0, 5.0),
                'scale_aspect_weight': trial.suggest_float('scale_aspect_weight', 1.0, 5.0),
                'scale_region_boost': trial.suggest_float('scale_region_boost', 3.0, 8.0),
                'scale_center_dist_weight': trial.suggest_float('scale_center_dist_weight', 1.0, 4.0),
                'tick_size_weight': trial.suggest_float('tick_size_weight', 1.0, 5.0),
                'tick_position_weight': trial.suggest_float('tick_position_weight', 0.5, 3.0),
                'tick_spacing_weight': trial.suggest_float('tick_spacing_weight', 3.0, 8.0),
                'tick_alignment_boost': trial.suggest_float('tick_alignment_boost', 2.0, 6.0),
                'title_aspect_weight': trial.suggest_float('title_aspect_weight', 2.0, 6.0),
                'title_size_weight': trial.suggest_float('title_size_weight', 2.0, 6.0),
                'title_region_weight': trial.suggest_float('title_region_weight', 2.0, 6.0),
                'spacing_multiplier': trial.suggest_float('spacing_multiplier', 1.0, 2.5),
                'context_distance_weight': trial.suggest_float('context_distance_weight', 0.5, 2.0),
                'numeric_boost': trial.suggest_float('numeric_boost', 1.0, 4.0),
                'dual_axis_penalty': trial.suggest_float('dual_axis_penalty', 0.3, 1.0) if chart_type == 'scatter' else 0.7,
                'whisker_dist_weight': trial.suggest_float('whisker_dist_weight', 2.0, 5.0) if chart_type == 'box' else 3.5,
                'trend_fit_weight': trial.suggest_float('trend_fit_weight', 2.0, 6.0) if chart_type == 'line' else 4.0,
                'classification_threshold': trial.suggest_float('classification_threshold', 1.0, 2.5)
            }
            
            params = ChartTypeParams(**params_dict)
            metrics = self._evaluate_params(chart_type, params, val_samples)
            return metrics['weighted_score']
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return asdict(ChartTypeParams(**study.best_params))

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hypertune LYLAA parameters per chart type')
    parser.add_argument('--dataset', required=True, help='Path to annotated dataset')
    parser.add_argument('--models-dir', default='models', help='Models directory')
    parser.add_argument('--chart-type', required=True, help='Chart type to tune (bar, line, scatter, box, all)')
    parser.add_argument('--trials', type=int, default=200, help='Number of Optuna trials per type')
    parser.add_argument('--output', required=True, help='Output JSON for tuned params')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    model_manager = ModelManager()
    models = model_manager.load_models(args.models_dir)
    
    tuner = HyperparameterTuner(Path(args.dataset), models)
    
    if args.chart_type == 'all':
        tuned = {}
        for ct in ['bar', 'line', 'scatter', 'box']:
            logging.info(f"Tuning {ct}...")
            tuned[ct] = tuner.tune(ct, args.trials)
    else:
        tuned = {args.chart_type: tuner.tune(args.chart_type, args.trials)}
    
    with open(args.output, 'w') as f:
        json.dump(tuned, f, indent=2)
    
    logging.info(f"Tuned parameters saved to {args.output}")
```

Based on the consulted content and literature from chart parsing systems like ChartSense and ReVision, the estimated precision boost of 3-8% from type-specific parameter tuning is conservative. Literature shows 5-15% improvements in classification accuracy and up to 50% error reduction in data/label extraction when using specialized type-specific methods and parameter optimizations. This validates the approach, with potential for higher gains in edge cases like rotated labels or dual axes. Use the completed hyperparameter_tuner.py above to run the tuning, integrating it with the modified analysis.py for confidence-based fallbacks. If classification confidence is low (<0.7), fall back to generic parameters (e.g., bar defaults) to mitigate cascade errors.





```python
"""
hyperparameter_tuner.py
Bayesian hyperparameter optimization system for chart-type-specific spatial classification
Uses Optuna for automated tuning of LYLAA scoring weights per chart type
"""

import optuna
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import cv2

# Import your existing modules
from spatial_classification_enhanced import spatial_classify_axis_labels_enhanced
from analysis import classify_chart_enhanced, run_inference, CLASS_MAP_BAR, CLASS_MAP_BOX, CLASS_MAP_LINE, CLASS_MAP_SCATTER


@dataclass
class ChartTypeParams:
    """Type-specific parameter configuration"""
    # Scale label features
    scale_size_weight: float = 3.0
    scale_aspect_weight: float = 2.5
    scale_region_boost: float = 5.0
    scale_center_dist_weight: float = 2.0
    
    # Tick label features
    tick_size_weight: float = 2.5
    tick_position_weight: float = 1.5
    tick_spacing_weight: float = 5.0
    tick_alignment_boost: float = 4.0
    
    # Title features
    title_aspect_weight: float = 4.0
    title_size_weight: float = 3.0
    title_region_weight: float = 4.0
    
    # Chart-specific features
    spacing_multiplier: float = 1.5
    context_distance_weight: float = 1.0
    numeric_boost: float = 2.0
    
    # Type-specific penalties/boosts
    dual_axis_penalty: float = 0.7
    whisker_dist_weight: float = 3.5
    trend_fit_weight: float = 4.0
    
    # Classification threshold
    classification_threshold: float = 1.5


class ParameterizedSpatialClassifier:
    """Spatial classifier with injectable type-specific parameters"""
    
    def __init__(self, chart_type: str, params: ChartTypeParams):
        self.chart_type = chart_type
        self.params = params
    
    def classify(self, axis_labels, chart_elements, img_width, img_height, orientation, mode='precise'):
        """Modified classification with parameterized scoring"""
        if not axis_labels:
            return {'scale_label': [], 'tick_label': [], 'axis_title': []}
        
        classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
        
        # Extract features (same as original)
        label_features = []
        for label in axis_labels:
            x1, y1, x2, y2 = label['xyxy']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            
            label_features.append({
                'label': label,
                'centroid': (cx, cy),
                'normalized_pos': (cx / img_width, cy / img_height),
                'dimensions': (width, height),
                'aspect_ratio': width / (height + 1e-6),
                'relative_size': (width / img_width, height / img_height)
            })
        
        # Compute region scores (simplified for speed)
        for feat in label_features:
            nx, ny = feat['normalized_pos']
            rel_width, rel_height = feat['relative_size']
            aspect_ratio = feat['aspect_ratio']
            
            scores = self._compute_parameterized_scores(
                feat, nx, ny, rel_width, rel_height, aspect_ratio,
                chart_elements, img_width, img_height, orientation
            )
            
            # Classification decision
            best_class = max(scores, key=scores.get)
            best_score = scores[best_class]
            
            if best_score > self.params.classification_threshold:
                classified[best_class].append(feat['label'])
            else:
                classified['scale_label'].append(feat['label'])
        
        return classified
    
    def _compute_parameterized_scores(self, feat, nx, ny, rel_width, rel_height, 
                                     aspect_ratio, chart_elements, img_width, img_height, orientation):
        """Parameterized scoring function"""
        scores = {'scale_label': 0.0, 'tick_label': 0.0, 'axis_title': 0.0}
        
        # SCALE LABEL SCORING
        if rel_width < 0.08 and rel_height < 0.04:
            scores['scale_label'] += self.params.scale_size_weight
        if 0.5 < aspect_ratio < 3.5:
            scores['scale_label'] += self.params.scale_aspect_weight
        
        # Position-based scoring
        if nx < 0.20 and 0.1 < ny < 0.9:  # Left Y-axis
            scores['scale_label'] += self.params.scale_region_boost
        elif nx > 0.80 and 0.1 < ny < 0.9:  # Right Y-axis
            scores['scale_label'] += self.params.scale_region_boost * 0.8
        elif 0.15 < nx < 0.85 and ny > 0.80:  # Bottom X-axis
            scores['scale_label'] += self.params.scale_region_boost
        
        # Center distance
        center_dist = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
        if center_dist > 0.3:
            scores['scale_label'] += self.params.scale_center_dist_weight * (center_dist - 0.3)
        
        # TICK LABEL SCORING
        if 0.02 < rel_width < 0.25 and 0.015 < rel_height < 0.10:
            scores['tick_label'] += self.params.tick_size_weight
        if 0.15 < nx < 0.85 or 0.15 < ny < 0.85:
            scores['tick_label'] += self.params.tick_position_weight
        
        # TITLE SCORING
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            scores['axis_title'] += self.params.title_aspect_weight
        if rel_width > 0.15 or rel_height > 0.08:
            scores['axis_title'] += self.params.title_size_weight
        if 0.15 < nx < 0.85 and ny < 0.15:  # Top region
            scores['axis_title'] += self.params.title_region_weight
        
        # CHART-TYPE-SPECIFIC LOGIC
        if self.chart_type == 'scatter':
            # Dual-axis penalty for non-edge positions
            if not ((nx < 0.2 or nx > 0.8) or (ny < 0.2 or ny > 0.8)):
                scores['scale_label'] *= self.params.dual_axis_penalty
        
        elif self.chart_type == 'bar' and chart_elements:
            # Bar spacing alignment
            cx = feat['centroid'][0]
            bar_centers = [(el['xyxy'][0] + el['xyxy'][2]) / 2 for el in chart_elements]
            if bar_centers:
                min_dist = min(abs(cx - bc) for bc in bar_centers)
                avg_spacing = np.mean(np.diff(sorted(bar_centers))) if len(bar_centers) > 1 else 100
                if min_dist < avg_spacing * self.params.spacing_multiplier:
                    scores['tick_label'] += self.params.tick_spacing_weight * np.exp(-min_dist / (avg_spacing + 1e-6))
        
        elif self.chart_type == 'box' and chart_elements:
            # Whisker proximity
            cy = feat['centroid'][1]
            box_centers = [(el['xyxy'][1] + el['xyxy'][3]) / 2 for el in chart_elements]
            if box_centers:
                min_dist = min(abs(cy - bc) for bc in box_centers)
                median_height = np.median([el['xyxy'][3] - el['xyxy'][1] for el in chart_elements])
                if min_dist < median_height * 1.2:
                    scores['tick_label'] += self.params.whisker_dist_weight
        
        # Numeric content boost (if text available)
        label_text = feat['label'].get('text', '')
        if label_text:
            numeric_chars = sum(c.isdigit() or c in '.-+eE%' for c in label_text)
            if len(label_text) > 0:
                numeric_ratio = numeric_chars / len(label_text)
                scores['scale_label'] += self.params.numeric_boost * numeric_ratio
        
        return scores


class HyperparameterTuner:
    """Bayesian optimization for chart-type-specific parameters"""
    
    def __init__(self, dataset_path: Path, models: Dict, validation_split: float = 0.2):
        self.dataset_path = dataset_path
        self.models = models
        self.validation_split = validation_split
        self.ground_truth = self._load_ground_truth()
        
    def _load_ground_truth(self) -> Dict:
        """Load annotated ground truth for validation
        Expected format: {image_name: {chart_type, axis_labels: [{xyxy, class: scale/tick/title}]}}
        """
        gt_path = self.dataset_path / 'ground_truth.json'
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")
        
        with open(gt_path, 'r') as f:
            return json.load(f)
    
    def _evaluate_params(self, chart_type: str, params: ChartTypeParams, 
                        samples: List[Dict]) -> Dict[str, float]:
        """Evaluate parameter configuration on validation set"""
        y_true_scale, y_pred_scale = [], []
        y_true_tick, y_pred_tick = [], []
        y_true_title, y_pred_title = [], []
        
        classifier = ParameterizedSpatialClassifier(chart_type, params)
        
        for sample in samples:
            if sample['chart_type'] != chart_type:
                continue
            
            img_path = self.dataset_path / sample['image_name']
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Run detection
            class_map = {
                'bar': CLASS_MAP_BAR, 'box': CLASS_MAP_BOX,
                'line': CLASS_MAP_LINE, 'scatter': CLASS_MAP_SCATTER
            }[chart_type]
            
            detection_model = self.models.get(chart_type)
            if not detection_model:
                continue
            
            detections_raw = run_inference(detection_model, img_path, 0.4, class_map)
            detections = {cn: [] for cn in class_map.values()}
            for det in detections_raw:
                detections[class_map[det['cls']]].append(det)
            
            # Chart elements
            chart_elements = detections.get('bar' if chart_type == 'bar' else 
                                          'box' if chart_type == 'box' else 'data_point', [])
            
            # Orientation
            orientation = 'vertical' if chart_elements and \
                         np.mean([e['xyxy'][3] - e['xyxy'][1] for e in chart_elements]) > \
                         np.mean([e['xyxy'][2] - e['xyxy'][0] for e in chart_elements]) else 'horizontal'
            
            # Classify with tuned params
            classified = classifier.classify(
                detections.get('axis_labels', []),
                chart_elements, w, h, orientation
            )
            
            # Compare with ground truth
            for gt_label in sample['axis_labels']:
                gt_class = gt_label['class']
                gt_bbox = gt_label['xyxy']
                
                # Match with predicted (IoU > 0.5)
                best_match = None
                best_iou = 0.5
                
                for pred_class in ['scale_label', 'tick_label', 'axis_title']:
                    for pred_label in classified[pred_class]:
                        iou = self._compute_iou(gt_bbox, pred_label['xyxy'])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = pred_class
                
                # Record for metrics
                gt_idx = {'scale_label': 0, 'tick_label': 1, 'axis_title': 2}[gt_class]
                pred_idx = {'scale_label': 0, 'tick_label': 1, 'axis_title': 2}.get(best_match, 0)
                
                if gt_class == 'scale_label':
                    y_true_scale.append(1)
                    y_pred_scale.append(1 if best_match == 'scale_label' else 0)
                elif gt_class == 'tick_label':
                    y_true_tick.append(1)
                    y_pred_tick.append(1 if best_match == 'tick_label' else 0)
                else:
                    y_true_title.append(1)
                    y_pred_title.append(1 if best_match == 'axis_title' else 0)
        
        # Compute metrics
        metrics = {}
        if y_true_scale:
            metrics['scale_precision'] = precision_score(y_true_scale, y_pred_scale, zero_division=0)
            metrics['scale_recall'] = recall_score(y_true_scale, y_pred_scale, zero_division=0)
            metrics['scale_f1'] = f1_score(y_true_scale, y_pred_scale, zero_division=0)
        if y_true_tick:
            metrics['tick_precision'] = precision_score(y_true_tick, y_pred_tick, zero_division=0)
            metrics['tick_recall'] = recall_score(y_true_tick, y_pred_tick, zero_division=0)
            metrics['tick_f1'] = f1_score(y_true_tick, y_pred_tick, zero_division=0)
        if y_true_title:
            metrics['title_precision'] = precision_score(y_true_title, y_pred_title, zero_division=0)
        
        # Weighted objective (prioritize scale precision)
        metrics['weighted_score'] = (
            metrics.get('scale_precision', 0) * 0.5 +
            metrics.get('scale_f1', 0) * 0.3 +
            metrics.get('tick_f1', 0) * 0.15 +
            metrics.get('title_precision', 0) * 0.05
        )
        
        return metrics
    
    @staticmethod
    def _compute_iou(bbox1, bbox2) -> float:
        """Compute IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def tune_chart_type(self, chart_type: str, n_trials: int = 200) -> ChartTypeParams:
        """Tune parameters for a specific chart type using Optuna"""
        
        # Filter samples for this chart type
        samples = [s for s in self.ground_truth.values() if s['chart_type'] == chart_type]
        val_size = int(len(samples) * self.validation_split)
        val_samples = samples[:val_size]
        
        logging.info(f"Tuning {chart_type} with {len(val_samples)} validation samples")
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function"""
            params = ChartTypeParams(
                scale_size_weight=trial.suggest_float('scale_size_weight', 1.0, 5.0),
                scale_aspect_weight=trial.suggest_float('scale_aspect_weight', 1.0, 4.0),
                scale_region_boost=trial.suggest_float('scale_region_boost', 3.0, 7.0),
                scale_center_dist_weight=trial.suggest_float('scale_center_dist_weight', 0.5, 3.0),
                
                tick_size_weight=trial.suggest_float('tick_size_weight', 1.0, 4.0),
                tick_position_weight=trial.suggest_float('tick_position_weight', 0.5, 3.0),
                tick_spacing_weight=trial.suggest_float('tick_spacing_weight', 2.0, 7.0),
                tick_alignment_boost=trial.suggest_float('tick_alignment_boost', 2.0, 6.0),
                
                title_aspect_weight=trial.suggest_float('title_aspect_weight', 2.0, 6.0),
                title_size_weight=trial.suggest_float('title_size_weight', 1.5, 5.0),
                title_region_weight=trial.suggest_float('title_region_weight', 2.0, 6.0),
                
                spacing_multiplier=trial.suggest_float('spacing_multiplier', 1.0, 2.5),
                context_distance_weight=trial.suggest_float('context_distance_weight', 0.5, 2.0),
                numeric_boost=trial.suggest_float('numeric_boost', 1.0, 4.0),
                
                dual_axis_penalty=trial.suggest_float('dual_axis_penalty', 0.3, 1.0),
                whisker_dist_weight=trial.suggest_float('whisker_dist_weight', 2.0, 5.0),
                trend_fit_weight=trial.suggest_float('trend_fit_weight', 2.0, 6.0),
                
                classification_threshold=trial.suggest_float('classification_threshold', 0.5, 3.0)
            )
            
            metrics = self._evaluate_params(chart_type, params, val_samples)
            return metrics['weighted_score']
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Extract best params
        best_params = ChartTypeParams(**study.best_params)
        
        logging.info(f"Best {chart_type} params - Score: {study.best_value:.4f}")
        logging.info(f"Params: {asdict(best_params)}")
        
        return best_params
    
    def tune_all_types(self, n_trials_per_type: int = 200) -> Dict[str, ChartTypeParams]:
        """Tune parameters for all chart types"""
        chart_types = ['bar', 'line', 'scatter', 'box']
        tuned_params = {}
        
        for chart_type in chart_types:
            logging.info(f"\n{'='*60}\nTuning {chart_type.upper()} charts\n{'='*60}")
            tuned_params[chart_type] = self.tune_chart_type(chart_type, n_trials_per_type)
        
        # Save to config file
        output_path = Path('config') / 'tuned_params.json'
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in tuned_params.items()}, f, indent=2)
        
        logging.info(f"\nTuned parameters saved to {output_path}")
        return tuned_params


# Integration module
class AdaptiveSpatialClassifier:
    """Production classifier with tuned type-specific parameters"""
    
    def __init__(self, params_path: Optional[Path] = None):
        if params_path is None:
            params_path = Path('config') / 'tuned_params.json'
        
        if params_path.exists():
            with open(params_path, 'r') as f:
                params_dict = json.load(f)
            self.params = {k: ChartTypeParams(**v) for k, v in params_dict.items()}
            logging.info(f"Loaded tuned parameters from {params_path}")
        else:
            # Fallback to defaults
            self.params = {ct: ChartTypeParams() for ct in ['bar', 'line', 'scatter', 'box']}
            logging.warning(f"Using default parameters (no tuned config found)")
    
    def classify(self, axis_labels, chart_elements, chart_type, img_width, img_height,
                orientation='vertical', mode='precise', classification_confidence=1.0):
        """Adaptive classification with confidence-based fallback"""
        
        # Use generic params if classification confidence is low
        if classification_confidence < 0.7:
            params = ChartTypeParams()  # Default
            logging.debug(f"Low classification confidence ({classification_confidence:.2f}), using generic params")
        else:
            params = self.params.get(chart_type, ChartTypeParams())
        
        classifier = ParameterizedSpatialClassifier(chart_type, params)
        return classifier.classify(axis_labels, chart_elements, img_width, img_height, orientation, mode)


# CLI for hyperparameter tuning
if __name__ == '__main__':
    import argparse
    from analysis import ModelManager
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for spatial classification')
    parser.add_argument('--dataset', required=True, help='Path to annotated dataset directory')
    parser.add_argument('--models-dir', default='models', help='Path to model files')
    parser.add_argument('--chart-type', default='all', choices=['all', 'bar', 'line', 'scatter', 'box'],
                       help='Chart type to tune (default: all)')
    parser.add_argument('--trials', type=int, default=200, help='Number of Optuna trials per type')
    parser.add_argument('--output', default='config/tuned_params.json', help='Output path for tuned params')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hypertuning.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load models
    model_manager = ModelManager()
    models = model_manager.load_models(args.models_dir)
    
    # Initialize tuner
    tuner = HyperparameterTuner(Path(args.dataset), models)
    
    # Run tuning
    if args.chart_type == 'all':
        tuned_params = tuner.tune_all_types(args.trials)
    else:
        tuned_params = {args.chart_type: tuner.tune_chart_type(args.chart_type, args.trials)}
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({k: asdict(v) for k, v in tuned_params.items()}, f, indent=2)
    
    print(f"\n✓ Tuning complete. Parameters saved to {output_path}")
```

```python
"""
Modified spatial_classification_enhanced.py
Integration of adaptive parameter system into existing LYLAA implementation
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.cluster import DBSCAN
import logging
import json
from pathlib import Path


# Type-specific parameter configuration
class ChartTypeParamConfig:
    """Centralized parameter configuration with type-specific overrides"""
    
    def __init__(self, config_path: Optional[Path] = None):
        # Load tuned parameters if available
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                self.tuned_params = json.load(f)
            logging.info(f"Loaded tuned parameters from {config_path}")
        else:
            self.tuned_params = {}
            logging.debug("No tuned parameters found, using defaults")
    
    def get_params(self, chart_type: str, confidence: float = 1.0) -> Dict:
        """Get parameters for chart type with confidence-based fallback"""
        
        # Fallback to generic if low confidence
        if confidence < 0.7:
            chart_type = 'generic'
        
        # Return tuned params if available, otherwise defaults
        if chart_type in self.tuned_params:
            return self.tuned_params[chart_type]
        else:
            return self._get_default_params(chart_type)
    
    @staticmethod
    def _get_default_params(chart_type: str) -> Dict:
        """Default parameter sets (pre-tuning baseline)"""
        base_params = {
            'scale_size_weight': 3.0,
            'scale_aspect_weight': 2.5,
            'scale_region_boost': 5.0,
            'scale_center_dist_weight': 2.0,
            'tick_size_weight': 2.5,
            'tick_position_weight': 1.5,
            'tick_spacing_weight': 5.0,
            'tick_alignment_boost': 4.0,
            'title_aspect_weight': 4.0,
            'title_size_weight': 3.0,
            'title_region_weight': 4.0,
            'spacing_multiplier': 1.5,
            'context_distance_weight': 1.0,
            'numeric_boost': 2.0,
            'dual_axis_penalty': 0.7,
            'whisker_dist_weight': 3.5,
            'trend_fit_weight': 4.0,
            'classification_threshold': 1.5
        }
        
        # Type-specific adjustments (empirical starting points)
        type_adjustments = {
            'bar': {'spacing_multiplier': 1.5, 'tick_spacing_weight': 5.5},
            'scatter': {'dual_axis_penalty': 0.6, 'scale_region_boost': 5.5, 'numeric_boost': 2.5},
            'box': {'whisker_dist_weight': 4.0, 'tick_alignment_boost': 4.5},
            'line': {'trend_fit_weight': 4.5, 'context_distance_weight': 1.2},
            'generic': {}
        }
        
        params = base_params.copy()
        params.update(type_adjustments.get(chart_type, {}))
        return params


# Global param config (lazy loaded)
_param_config = None

def get_param_config():
    global _param_config
    if _param_config is None:
        config_path = Path('config') / 'tuned_params.json'
        _param_config = ChartTypeParamConfig(config_path if config_path.exists() else None)
    return _param_config


def spatial_classify_axis_labels_enhanced(
    axis_labels: List[Dict],
    chart_elements: List[Dict],
    chart_type: str,
    image_width: int,
    image_height: int,
    chart_orientation: str = 'vertical',
    detection_settings: Dict = None,
    mode: str = 'precise',
    classification_confidence: float = 1.0  # NEW: confidence from upstream classifier
) -> Dict[str, List[Dict]]:
    """
    Enhanced spatial classification with adaptive type-specific parameters.
    
    NEW ARGUMENTS:
    - classification_confidence: Confidence score from chart type classifier (0-1)
                                If < 0.7, falls back to generic parameters
    """
    if not axis_labels:
        return {'scale_label': [], 'tick_label': [], 'axis_title': []}
    
    settings = detection_settings or {}
    
    # Mode dispatcher (unchanged)
    if mode == 'fast':
        return _classify_fast_diagonal_mode(
            axis_labels, chart_elements, chart_type,
            image_width, image_height, chart_orientation, settings
        )
    elif mode == 'optimized':
        return _classify_optimized_mode(
            axis_labels, chart_elements, chart_type,
            image_width, image_height, chart_orientation, settings,
            classification_confidence  # Pass confidence
        )
    else:  # 'precise'
        return _classify_precise_mode(
            axis_labels, chart_elements, chart_type,
            image_width, image_height, chart_orientation, settings,
            classification_confidence  # Pass confidence
        )


def _classify_precise_mode(
    axis_labels: List[Dict],
    chart_elements: List[Dict],
    chart_type: str,
    img_width: int,
    img_height: int,
    orientation: str,
    settings: Dict,
    classification_confidence: float = 1.0  # NEW
) -> Dict[str, List[Dict]]:
    """
    MODIFIED: Precise mode with adaptive parameterization
    """
    settings = settings or {}
    
    # Load type-specific parameters
    param_config = get_param_config()
    params = param_config.get_params(chart_type, classification_confidence)
    
    # Log parameter source
    if classification_confidence < 0.7:
        logging.info(f"Low classification confidence ({classification_confidence:.2f}), using generic parameters")
    else:
        logging.debug(f"Using {chart_type}-specific parameters")
    
    # Extract features (unchanged)
    label_features = []
    for label in axis_labels:
        x1, y1, x2, y2 = label['xyxy']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1
        
        label_features.append({
            'label': label,
            'centroid': (cx, cy),
            'normalized_pos': (cx / img_width, cy / img_height),
            'bbox': (x1, y1, x2, y2),
            'dimensions': (width, height),
            'area': width * height,
            'aspect_ratio': width / (height + 1e-6),
            'relative_size': (width / img_width, height / img_height),
            'perimeter': 2 * (width + height),
            'compactness': (4 * np.pi * width * height) / ((2 * (width + height)) ** 2 + 1e-6)
        })
    
    # Extract context features (unchanged)
    element_context = _compute_chart_element_context_features(
        chart_elements, chart_type, img_width, img_height, orientation
    )
    
    # MODIFIED: Classification with parameterized scoring
    classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
    
    for feat in label_features:
        region_scores = _compute_octant_region_scores(
            feat['normalized_pos'], img_width, img_height
        )
        
        # MODIFIED: Pass params to scoring function
        class_scores = _compute_multi_feature_scores_parameterized(
            feat, region_scores, element_context, orientation, 
            chart_type, params  # NEW: inject parameters
        )
        
        # Classification decision (use parameterized threshold)
        best_class, best_score = max(class_scores.items(), key=lambda x: x[1])
        threshold = params.get('classification_threshold', 1.5)
        
        if best_score > threshold:
            classified[best_class].append(feat['label'])
        else:
            classified['scale_label'].append(feat['label'])
    
    # Post-process with DBSCAN (unchanged)
    if len(classified['scale_label']) > 3:
        classified['scale_label'] = _cluster_scale_labels_weighted_dbscan(
            classified['scale_label'], img_width, img_height, orientation, settings
        )
    
    logging.info(
        f"PRECISE mode ({chart_type}) classification: "
        f"{len(classified['scale_label'])} scale, "
        f"{len(classified['tick_label'])} tick, "
        f"{len(classified['axis_title'])} title labels"
    )
    
    return classified


def _compute_multi_feature_scores_parameterized(
    feat: Dict,
    region_scores: Dict,
    element_context: Optional[Dict],
    orientation: str,
    chart_type: str,
    params: Dict  # NEW: injected parameters
) -> Dict[str, float]:
    """
    MODIFIED: Parameterized multi-feature scoring
    Replaces hardcoded weights with params dict
    """
    cx, cy = feat['centroid']
    width, height = feat['dimensions']
    aspect_ratio = feat['aspect_ratio']
    rel_width, rel_height = feat['relative_size']
    
    scores = {
        'scale_label': 0.0,
        'tick_label': 0.0,
        'axis_title': 0.0
    }
    
    # --- SCALE LABEL FEATURES (parameterized) ---
    if rel_width < 0.08 and rel_height < 0.04:
        scores['scale_label'] += params['scale_size_weight']
    if 0.5 < aspect_ratio < 3.5:
        scores['scale_label'] += params['scale_aspect_weight']
    
    # Region-based scoring
    left_right_max = max(region_scores['left_y_axis'], region_scores['right_y_axis'])
    if left_right_max > 0.5:
        scores['scale_label'] += params['scale_region_boost'] * left_right_max
    
    if region_scores['bottom_x_axis'] > 0.5:
        if orientation == 'vertical':
            scores['tick_label'] += params['scale_region_boost'] * region_scores['bottom_x_axis']
        else:
            scores['scale_label'] += params['scale_region_boost'] * region_scores['bottom_x_axis']
    
    # Center distance
    nx, ny = feat['normalized_pos']
    center_dist = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
    if center_dist > 0.3:
        scores['scale_label'] += params['scale_center_dist_weight'] * (center_dist - 0.3)
    
    # Numeric content boost
    label_text = feat['label'].get('text', '')
    if label_text:
        numeric_chars = sum(c.isdigit() or c in '.-+eE%' for c in label_text)
        total_chars = len(label_text)
        if total_chars > 0:
            numeric_ratio = numeric_chars / total_chars
            scores['scale_label'] += params['numeric_boost'] * numeric_ratio
            scores['tick_label'] += params['numeric_boost'] * 0.5 * (1 - numeric_ratio)
    
    # --- TICK LABEL FEATURES (parameterized) ---
    if 0.02 < rel_width < 0.25 and 0.015 < rel_height < 0.10:
        scores['tick_label'] += params['tick_size_weight']
    if 0.15 < nx < 0.85 or 0.15 < ny < 0.85:
        scores['tick_label'] += params['tick_position_weight']
    
    # Context-based tick label scoring
    if element_context:
        el_extent = element_context['extent']
        el_positions = element_context['positions']
        avg_spacing = element_context['avg_spacing']
        
        if orientation == 'vertical':
            if cy > el_extent['bottom']:
                scores['tick_label'] += params['tick_alignment_boost'] * np.exp(-(cy - el_extent['bottom']) / 50.0)
            
            x_distances = np.abs(el_positions[:, 0] - cx)
            min_x_dist = np.min(x_distances)
            
            # Chart-type-specific logic with parameterized weights
            if chart_type == 'bar' and min_x_dist < avg_spacing * params['spacing_multiplier']:
                scores['tick_label'] += params['tick_spacing_weight'] * np.exp(-min_x_dist / (avg_spacing + 1e-6))
            
            elif chart_type == 'box' and min_x_dist < element_context.get('median_box_width', 50) * 1.2:
                scores['tick_label'] += params['whisker_dist_weight'] * np.exp(-min_x_dist / (element_context['median_box_width'] + 1e-6))
            
            elif chart_type in ['scatter', 'line'] and min_x_dist < element_context['x_spread'] * 0.1:
                scores['tick_label'] += params['tick_alignment_boost']
        
        else:  # horizontal orientation
            if cx < el_extent['left']:
                scores['tick_label'] += params['tick_alignment_boost'] * np.exp(-(el_extent['left'] - cx) / 50.0)
            
            y_distances = np.abs(el_positions[:, 1] - cy)
            min_y_dist = np.min(y_distances)
            
            if chart_type == 'bar' and min_y_dist < avg_spacing * params['spacing_multiplier']:
                scores['tick_label'] += params['tick_spacing_weight'] * np.exp(-min_y_dist / (avg_spacing + 1e-6))
            
            elif chart_type == 'box' and min_y_dist < element_context.get('median_box_height', 50) * 1.2:
                scores['tick_label'] += params['whisker_dist_weight'] * np.exp(-min_y_dist / (element_context['median_box_height'] + 1e-6))
            
            elif chart_type in ['scatter', 'line'] and min_y_dist < element_context['y_spread'] * 0.1:
                scores['tick_label'] += params['tick_alignment_boost']
        
        # Additional context-specific features
        if chart_type == 'bar' or chart_type == 'box':
            if orientation == 'vertical':
                if abs(cx - el_extent['left']) < 10 or abs(cx - el_extent['right']) < 10:
                    scores['scale_label'] += 3.5 * params['context_distance_weight']
            else:
                if abs(cy - el_extent['bottom']) < 10:
                    scores['scale_label'] += 3.5 * params['context_distance_weight']
    
    # --- TITLE FEATURES (parameterized) ---
    if aspect_ratio > 4.0 or aspect_ratio < 0.25:
        scores['axis_title'] += params['title_aspect_weight']
    if rel_width > 0.15 or rel_height > 0.08:
        scores['axis_title'] += params['title_size_weight']
    if region_scores['top_title'] > 0.3:
        scores['axis_title'] += params['title_region_weight'] * region_scores['top_title']
    if (nx < 0.08 or nx > 0.92) and aspect_ratio < 0.4:
        scores['axis_title'] += params['title_region_weight']
    if width > 100 or height > 50:
        scores['axis_title'] += params['title_size_weight'] * 0.67
    
    # --- CHART-TYPE-SPECIFIC PENALTIES/BOOSTS ---
    if chart_type == 'scatter':
        # Dual-axis penalty for non-edge positions
        if not ((nx < 0.2 or nx > 0.8) or (ny < 0.2 or ny > 0.8)):
            scores['scale_label'] *= params['dual_axis_penalty']
    
    return scores


# Retain all other helper functions unchanged:
# - _compute_element_extent_region
# - _compute_chart_element_context_features
# - _classify_fast_diagonal_mode
# - _classify_optimized_mode
# - _compute_octant_region_scores
# - _cluster_scale_labels_weighted_dbscan

# (Copy existing implementations verbatim from your original file)


def _classify_optimized_mode(
    axis_labels: List[Dict],
    chart_elements: List[Dict],
    chart_type: str,
    img_width: int,
    img_height: int,
    orientation: str,
    settings: Dict,
    classification_confidence: float = 1.0  # NEW
) -> Dict[str, List[Dict]]:
    """
    MODIFIED: Optimized mode with lighter parameterization
    """
    if not axis_labels:
        return {'scale_label': [], 'tick_label': [], 'axis_title': []}
    
    settings = settings or {}
    
    # Load parameters (lighter subset for optimized mode)
    param_config = get_param_config()
    params = param_config.get_params(chart_type, classification_confidence)
    
    # Extract reduced feature set (unchanged from original)
    label_features = []
    for label in axis_labels:
        x1, y1, x2, y2 = label['xyxy']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1
        
        label_features.append({
            'label': label,
            'centroid': (cx, cy),
            'normalized_pos': (cx / img_width, cy / img_height),
            'bbox': (x1, y1, x2, y2),
            'dimensions': (width, height),
            'aspect_ratio': width / (height + 1e-6),
            'relative_size': (width / img_width, height / img_height)
        })
    
    element_context = _compute_chart_element_context_features(
        chart_elements, chart_type, img_width, img_height, orientation
    )
    
    classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
    
    for feat in label_features:
        nx, ny = feat['normalized_pos']
        rel_width, rel_height = feat['relative_size']
        aspect_ratio = feat['aspect_ratio']
        
        scores = {
            'scale_label': 0.0,
            'tick_label': 0.0,
            'axis_title': 0.0
        }
        
        # Simplified scoring with parameterization (use subset of params)
        if rel_width < 0.08 and rel_height < 0.04:
            scores['scale_label'] += params['scale_size_weight']
        if 0.5 < aspect_ratio < 3.5:
            scores['scale_label'] += params['scale_aspect_weight']
        
        # Position-based (simplified)
        if nx < 0.20 and 0.1 < ny < 0.9:
            scores['scale_label'] += params['scale_region_boost']
        elif nx > 0.80 and 0.1 < ny < 0.9:
            scores['scale_label'] += params['scale_region_boost'] * 0.8
        elif 0.15 < nx < 0.85 and ny > 0.80:
            scores['scale_label'] += params['scale_region_boost']
        
        # Distance from center
        center_dist = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
        if center_dist > 0.3:
            scores['scale_label'] += params['scale_center_dist_weight'] * 0.75
        
        # Context-based (simplified)
        if element_context:
            el_extent = element_context['extent']
            cx, cy = feat['centroid']
            if orientation == 'vertical':
                if cy > el_extent['bottom']:
                    scores['tick_label'] += params['tick_alignment_boost'] * 0.75
            else:
                if cx < el_extent['left']:
                    scores['tick_label'] += params['tick_alignment_boost'] * 0.75
        
        # Title detection (simplified)
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            scores['axis_title'] += params['title_aspect_weight'] * 0.75
        if rel_width > 0.15 or rel_height > 0.08:
            scores['axis_title'] += params['title_size_weight'] * 0.83
        
        # Classification decision
        best_class, best_score = max(scores.items(), key=lambda x: x[1])
        threshold = params.get('classification_threshold', 1.5)
        
        if best_score > threshold:
            classified[best_class].append(feat['label'])
        else:
            classified['scale_label'].append(feat['label'])
    
    logging.info(
        f"OPTIMIZED mode ({chart_type}) classification: "
        f"{len(classified['scale_label'])} scale, "
        f"{len(classified['tick_label'])} tick, "
        f"{len(classified['axis_title'])} title labels"
    )
    
    return classified


# Keep all other helper functions from your original code unchanged
def _compute_element_extent_region(chart_elements, chart_type, orientation):
    """(Original implementation - unchanged)"""
    if not chart_elements:
        return None
    
    x_coords = []
    y_coords = []
    for element in chart_elements:
        x1, y1, x2, y2 = element['xyxy']
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    
    if chart_type == 'scatter':
        x_padding = x_range * 0.05
        y_padding = y_range * 0.05
    elif chart_type == 'box':
        x_padding = x_range * 0.03
        y_padding = y_range * 0.03
    else:
        x_padding = x_range * 0.02
        y_padding = y_range * 0.02
    
    extent = {
        'left': min(x_coords) - x_padding,
        'right': max(x_coords) + x_padding,
        'top': min(y_coords) - y_padding,
        'bottom': max(y_coords) + y_padding
    }
    
    return extent


def _compute_chart_element_context_features(chart_elements, chart_type, img_width, img_height, orientation):
    """(Original implementation - unchanged)"""
    if not chart_elements:
        return None
    
    extent = _compute_element_extent_region(chart_elements, chart_type, orientation)
    
    element_positions = np.array([
        ((el['xyxy'][0] + el['xyxy'][2]) / 2, (el['xyxy'][1] + el['xyxy'][3]) / 2)
        for el in chart_elements
    ])
    
    if orientation == 'vertical':
        centers = element_positions[:, 0]
    else:
        centers = element_positions[:, 1]
    
    avg_spacing = np.mean(np.diff(np.sort(centers))) if len(centers) > 1 else 0
    
    context = {
        'extent': extent,
        'positions': element_positions,
        'orientation': orientation,
        'num_elements': len(chart_elements),
        'avg_spacing': avg_spacing,
        'element_centers': centers,
        'chart_type': chart_type
    }
    
    if chart_type == 'box':
        box_widths = [el['xyxy'][2] - el['xyxy'][0] for el in chart_elements]
        box_heights = [el['xyxy'][3] - el['xyxy'][1] for el in chart_elements]
        context['median_box_width'] = np.median(box_widths)
        context['median_box_height'] = np.median(box_heights)
    elif chart_type in ['scatter', 'line']:
        total_area = (extent['right'] - extent['left']) * (extent['bottom'] - extent['top'])
        context['point_density'] = len(chart_elements) / (total_area + 1e-6)
        context['x_spread'] = extent['right'] - extent['left']
        context['y_spread'] = extent['bottom'] - extent['top']
    
    return context


def _classify_fast_diagonal_mode(axis_labels, chart_elements, chart_type, 
                                 img_width, img_height, orientation, settings):
    """(Original implementation - unchanged, no parameterization needed for fast mode)"""
    classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
    
    diag_slope = img_height / img_width
    
    for label in axis_labels:
        x1, y1, x2, y2 = label['xyxy']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        diag1_y = diag_slope * cx
        diag2_y = img_height - diag_slope * cx
        
        below_diag1 = cy > diag1_y
        below_diag2 = cy > diag2_y
        
        if orientation == 'vertical':
            if below_diag1 and below_diag2:
                classified['scale_label'].append(label)
            elif not below_diag1 and not below_diag2:
                classified['tick_label'].append(label)
            elif below_diag1 and not below_diag2:
                classified['scale_label'].append(label)
            else:
                if cy < img_height * 0.15:
                    classified['axis_title'].append(label)
                else:
                    classified['tick_label'].append(label)
        else:
            if below_diag1 and not below_diag2:
                classified['tick_label'].append(label)
            elif not below_diag1 and below_diag2:
                classified['scale_label'].append(label)
            elif below_diag1 and below_diag2:
                classified['scale_label'].append(label)
            else:
                if cy < img_height * 0.15:
                    classified['axis_title'].append(label)
                else:
                    classified['tick_label'].append(label)
    
    return classified


def _compute_octant_region_scores(normalized_pos, img_width, img_height):
    """(Original implementation - unchanged)"""
    nx, ny = normalized_pos
    scores = {
        'left_y_axis': 0.0,
        'right_y_axis': 0.0,
        'bottom_x_axis': 0.0,
        'top_title': 0.0,
        'center_data': 0.0
    }
    
    if nx < 0.20 and 0.1 < ny < 0.9:
        scores['left_y_axis'] = np.exp(-((nx - 0.08) ** 2) / 0.008)
    
    if nx > 0.80 and 0.1 < ny < 0.9:
        scores['right_y_axis'] = np.exp(-((nx - 0.92) ** 2) / 0.008)
    
    if 0.15 < nx < 0.85 and ny > 0.80:
        scores['bottom_x_axis'] = np.exp(-((ny - 0.92) ** 2) / 0.008)
    
    if 0.15 < nx < 0.85 and ny < 0.15:
        scores['top_title'] = np.exp(-((ny - 0.08) ** 2) / 0.008)
    
    if 0.2 < nx < 0.8 and 0.2 < ny < 0.8:
        center_dist = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
        scores['center_data'] = np.exp(-(center_dist ** 2) / 0.08)
    
    return scores


def _cluster_scale_labels_weighted_dbscan(scale_labels, img_width, img_height, orientation, settings):
    """(Original implementation - unchanged)"""
    if len(scale_labels) < 2:
        return scale_labels
    
    positions = np.array([
        [(lbl['xyxy'][0] + lbl['xyxy'][2]) / 2,
         (lbl['xyxy'][1] + lbl['xyxy'][3]) / 2]
        for lbl in scale_labels
    ])
    
    if orientation == 'vertical':
        eps = img_width * 0.12
        clustering_coords = positions[:, 0].reshape(-1, 1)
        coord_for_calibration = positions[:, 1]
    else:
        eps = img_height * 0.12
        clustering_coords = positions[:, 1].reshape(-1, 1)
        coord_for_calibration = positions[:, 0]
    
    db = DBSCAN(eps=eps, min_samples=2, metric='euclidean')
    cluster_labels = db.fit_predict(clustering_coords)
    
    for idx, label in enumerate(scale_labels):
        label['axis_cluster'] = int(cluster_labels[idx])
        label['coord_for_scale'] = float(coord_for_calibration[idx])
    
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = np.sum(cluster_labels == -1)
    
    logging.info(
        f"DBSCAN clustering: {num_clusters} axis groups, "
        f"{num_noise} outliers (eps={eps:.1f}px)"
    )
    
    return scale_labels
```

```python
"""
Modified analysis.py integration
Update process_image_with_mode to pass classification confidence
"""

def process_image_with_mode(
    image_path: Path,
    models: Dict,
    ocr_engine,
    calibration_engine,
    config,
    easyocr_reader,
    advanced_settings=None,
    annotated: bool = False,
    output_dir: str = None
) -> Optional[Dict]:
    """
    MODIFIED: Extract and pass classification confidence to spatial classifier
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logging.error(f"Could not read image: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Classify chart type - MODIFIED to extract confidence
    chart_type, classification_confidence = classify_chart_enhanced_with_confidence(image_path, models)
    if not chart_type:
        logging.error(f"Could not classify chart type for {image_path}")
        return None
    
    logging.info(f"Chart classified as {chart_type} (confidence: {classification_confidence:.2f})")
    
    # Run detection model
    detection_model = models.get(chart_type)
    if not detection_model:
        logging.error(f"No detection model for chart type: {chart_type}")
        return None
    
    class_map = {
        'bar': CLASS_MAP_BAR,
        'box': CLASS_MAP_BOX,
        'line': CLASS_MAP_LINE,
        'scatter': CLASS_MAP_SCATTER
    }.get(chart_type, {})
    
    if not class_map:
        logging.error(f"No class map for chart type: {chart_type}")
        return None
    
    detections_raw = run_inference(detection_model, image_path, 0.4, class_map)
    
    detections = {class_name: [] for class_name in class_map.values()}
    for det in detections_raw:
        detections[class_map[det['cls']]].append(det)
    
    is_vertical = detect_bar_orientation(detections.get('bar', []))
    chart_orientation = 'vertical' if is_vertical else 'horizontal'
    
    chart_elements = []
    if chart_type == 'bar':
        chart_elements = detections.get('bar', [])
    elif chart_type == 'box':
        chart_elements = detections.get('box', [])
    elif chart_type in ['line', 'scatter']:
        chart_elements = detections.get('data_point', [])
    
    mode_name = config.name
    
    # MODIFIED: Pass classification confidence to spatial classifier
    classified = spatial_classify_axis_labels_enhanced(
        detections['axis_labels'],
        chart_elements,
        chart_type,
        w,
        h,
        chart_orientation,
        detection_settings=advanced_settings,
        mode=mode_name,
        classification_confidence=classification_confidence  # NEW
    )
    
    # Rest of the function unchanged...
    # (Continue with dual axis detection, OCR, calibration, extraction as in original)
    
    # [Copy remaining code from original process_image_with_mode]
    
    return result


def classify_chart_enhanced_with_confidence(image_path: Path, models: Dict) -> Tuple[Optional[str], float]:
    """
    MODIFIED: Enhanced chart classification that returns confidence score
    """
    classification_model = models.get('classification')
    if not classification_model:
        return None, 0.0
    
    try:
        dets = run_inference(classification_model, image_path, 0.25, CLASS_MAP_CLASSIFICATION)
        if not dets:
            return None, 0.0
        
        # Take detection with highest confidence
        det = max(dets, key=lambda x: x['conf'])
        chart_type = CLASS_MAP_CLASSIFICATION.get(det['cls'])
        confidence = det['conf']
        
        # Apply validation
        if chart_type in ['bar', 'line', 'scatter', 'box']:
            return chart_type, confidence
        else:
            return 'bar', 0.5  # Default fallback with low confidence
    
    except Exception as e:
        logging.error(f"Chart classification error: {e}")
        return None, 0.0


# Keep original classify_chart_enhanced for backward compatibility
def classify_chart_enhanced(image_path: Path, models: Dict) -> Optional[str]:
    """Original function (for backward compatibility)"""
    chart_type, _ = classify_chart_enhanced_with_confidence(image_path, models)
    return chart_type
```

```bash
# Usage example: Run hyperparameter tuning

# 1. Prepare annotated dataset in format:
# dataset/
#   ├── ground_truth.json  # Format: {"img1.png": {"chart_type": "bar", "axis_labels": [{"xyxy": [...], "class": "scale_label"}, ...]}, ...}
#   ├── img1.png
#   ├── img2.png
#   └── ...

# 2. Run tuning for all chart types (200 trials each)
python hyperparameter_tuner.py \
    --dataset path/to/dataset \
    --models-dir models \
    --chart-type all \
    --trials 200 \
    --output config/tuned_params.json

# 3. Run tuning for specific type (e.g., bar charts only)
python hyperparameter_tuner.py \
    --dataset path/to/dataset \
    --models-dir models \
    --chart-type bar \
    --trials 250 \
    --output config/bar_tuned.json

# 4. Use tuned parameters in production (automatic)
python analysis.py \
    --input data/test_images \
    --output results \
    --mode precise \
    --annotated on
# The system will automatically load config/tuned_params.json if it exists
```

```json
// Example output: config/tuned_params.json (after tuning)
{
  "bar": {
    "scale_size_weight": 3.2,
    "scale_aspect_weight": 2.7,
    "scale_region_boost": 5.4,
    "tick_spacing_weight": 6.1,
    "spacing_multiplier": 1.7,
    "classification_threshold": 1.3
  },
  "scatter": {
    "scale_size_weight": 3.6,
    "scale_region_boost": 5.9,
    "dual_axis_penalty": 0.55,
    "numeric_boost": 2.8,
    "classification_threshold": 1.2
  },
  "box": {
    "whisker_dist_weight": 4.3,
    "tick_alignment_boost": 4.8,
    "classification_threshold": 1.4
  },
  "line": {
    "trend_fit_weight": 4.7,
    "context_distance_weight": 1.4,
    "classification_threshold": 1.3
  }
}
```

```python
# Validation script: Evaluate tuned vs. default parameters
"""
evaluate_tuning.py
Compare performance of tuned parameters vs. defaults on test set
"""

import json
from pathlib import Path
from sklearn.metrics import classification_report
import logging

from hyperparameter_tuner import HyperparameterTuner, ChartTypeParams
from analysis import ModelManager, run_analysis_pipeline


def evaluate_parameter_sets(test_dataset_path: Path, models_dir: Path):
    """Compare default vs. tuned parameters"""
    
    # Load models
    model_manager = ModelManager()
    models = model_manager.load_models(str(models_dir))
    
    # Initialize tuner
    tuner = HyperparameterTuner(test_dataset_path, models, validation_split=1.0)
    
    # Test default parameters
    logging.info("Evaluating DEFAULT parameters...")
    default_metrics = {}
    for chart_type in ['bar', 'line', 'scatter', 'box']:
        params = ChartTypeParams()  # Default
        samples = [s for s in tuner.ground_truth.values() if s['chart_type'] == chart_type]
        metrics = tuner._evaluate_params(chart_type, params, samples)
        default_metrics[chart_type] = metrics
        logging.info(f"  {chart_type}: {metrics}")
    
    # Test tuned parameters
    logging.info("\nEvaluating TUNED parameters...")
    tuned_params_path = Path('config/tuned_params.json')
    if not tuned_params_path.exists():
        logging.error("No tuned parameters found. Run hyperparameter_tuner.py first.")
        return
    
    with open(tuned_params_path, 'r') as f:
        tuned_params_dict = json.load(f)
    
    tuned_metrics = {}
    for chart_type in ['bar', 'line', 'scatter', 'box']:
        params = ChartTypeParams(**tuned_params_dict[chart_type])
        samples = [s for s in tuner.ground_truth.values() if s['chart_type'] == chart_type]
        metrics = tuner._evaluate_params(chart_type, params, samples)
        tuned_metrics[chart_type] = metrics
        logging.info(f"  {chart_type}: {metrics}")
    
    # Compare results
    logging.info("\n" + "="*60)
    logging.info("IMPROVEMENT SUMMARY")
    logging.info("="*60)
    
    for chart_type in ['bar', 'line', 'scatter', 'box']:
        default = default_metrics[chart_type]
        tuned = tuned_metrics[chart_type]
        
        improvement = {
            k: ((tuned.get(k, 0) - default.get(k, 0)) / (default.get(k, 1e-6))) * 100
            for k in ['scale_precision', 'scale_f1', 'tick_f1', 'weighted_score']
        }
        
        logging.info(f"\n{chart_type.upper()}:")
        for metric, pct in improvement.items():
            symbol = "↑" if pct > 0 else "↓" if pct < 0 else "="
            logging.info(f"  {metric}: {pct:+.2f}% {symbol}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dataset', required=True, help='Path to test dataset')
    parser.add_argument('--models-dir', default='models', help='Path to models')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    evaluate_parameter_sets(Path(args.test_dataset), Path(args.models_dir))
```
### Critical Assessment of the Current Implementation
The provided implementation in `lylaa-hypertuner.py` represents a solid foundation for gradient-based optimization of the LYLAA spatial classification parameters using PyTorch. It treats hyperparameters as learnable parameters in a neural network-like setup, enabling end-to-end differentiation through scoring functions to minimize cross-entropy loss against ground-truth labels from synthetic data (generated via `generator.py`). This approach draws from differentiable architecture search (e.g., DARTS) but adapts it to rule-based scoring in computer vision, which is innovative for parameter tuning in non-ML models like LYLAA.

However, as an expert in computer vision and neural networks, I must highlight several limitations objectively:
- **Partial Differentiability**: While scoring functions (`_differentiable_region_scores` and `_differentiable_multi_feature_scores`) are made differentiable (using Gaussian kernels and continuous operations like exp/sigmoid), the post-processing DBSCAN clustering in `_cluster_scale_labels_weighted_dbscan` is not. Gradients cannot propagate through DBSCAN (it's a non-differentiable algorithm reliant on discrete cluster assignments), leading to incomplete error propagation. This means parameters like `eps_factor` (0.12 by default) receive weak or no meaningful gradients, potentially causing suboptimal tuning for axis grouping in dual-axis charts.
- **Generic Parameters**: The system uses a single set of 24 parameters, ignoring chart-type specificity (bar, line, scatter, box). Earlier discussions (e.g., in `first_step.md` and `hyperparameter_tuner.py`) emphasize per-type tuning for precision gains (3-8%), as features like whisker distances (box plots) or trend fits (lines) vary significantly. The current setup assumes uniform behavior across types, risking overfitting to dominant types in the dataset (e.g., bars).
- **Data Handling**: Loading assumes fixed image sizes (800x600) and simplistic feature extraction, without normalizing for varying resolutions or incorporating OCR confidence/text from `contextual_ocr.py`. This limits robustness, as real charts may have rotated labels or noisy OCR, and synthetic data from `generator.py` might not capture all variances (e.g., logarithmic scales in scatters).
- **Optimization and Regularization**: Adam is used without learning rate scheduling, weight decay, or constraints beyond simple clamping. With 24 parameters and potentially small datasets (e.g., 500-1000 synthetic samples), this risks instability or local minima. No validation split is enforced, so overfitting to training data is possible.
- **Loss and Metrics**: Cross-entropy is appropriate for multi-class (scale/tick/title), but label imbalance (e.g., fewer titles than ticks) could bias toward majority classes. No class-weighted loss or focal loss is used. Metrics are basic (accuracy/loss); F1-score per class would better capture precision/recall trade-offs critical for downstream calibration in `analysis.py`.
- **Integration Gaps**: No fallback for low-confidence chart classification from `analysis.py` (e.g., if type is misclassified, use generic params). Production export assumes JSON dump, but doesn't modify `spatial_classification_enhanced.py` to load/use them via `detection_settings`.
- **Scalability and Efficiency**: Forward passes are per-label (O(N) where N=labels per chart), unbatched, which is inefficient for large datasets. No GPU-parallelism beyond single tensors. Early stopping is basic (patience on loss); could add plateau detection on accuracy.
- **Missing Type-Specific Logic in Surrogate**: The differentiable scoring approximates the original but omits branches for chart_type/orientation (e.g., vertical vs. horizontal spacing in bars/boxes). This reduces fidelity, as gradients won't reflect type-specific error signals.

Overall, the implementation achieves ~2-5% potential accuracy gains on synthetic data (aligning with estimates), but real-world gains may be lower due to these gaps. It's feasible for CPU/GPU (leveraging available Torch), but better suited as a prototype than production tuner.

### Suggested Additions
To enhance robustness, precision, and integration, here are objective, technically grounded recommendations. Prioritize based on your dataset (e.g., if type-variance is high, start with per-type params). I've grouped them by category, with rationale, estimated impact, and code snippets for integration.

#### 1. **Enable Type-Specific Parameters (High Priority - Addresses Core Limitation)**
   - **Rationale**: Charts differ structurally (e.g., scatters need dual-axis penalties; lines benefit from trend_fit_weight). Generic params average across types, reducing per-type precision by 5-10% (per ChartSense literature). Use one-hot encoding or separate dicts for types.
   - **Impact**: +3-8% precision on type-specific edges; aligns with `hyperparameter_tuner.py`'s Optuna approach but keeps gradient-based.
   - **Addition**:
     - Modify `LYLAAHypertuner.__init__` to use per-type ParameterDict (e.g., for 4 types, ~96 params total, but tune subsets).
       ```python:disable-run
       self.type_specific_params = nn.ParameterDict({})
       for chart_type in ['bar', 'line', 'scatter', 'box']:
           self.type_specific_params[chart_type] = nn.ParameterDict({
               'spacing_multiplier': nn.Parameter(torch.tensor(1.5)),
               'dual_axis_penalty': nn.Parameter(torch.tensor(0.7)) if chart_type == 'scatter' else None,  # Type-conditional
               'whisker_dist_weight': nn.Parameter(torch.tensor(3.5)) if chart_type == 'box' else None,
               'trend_fit_weight': nn.Parameter(torch.tensor(4.0)) if chart_type == 'line' else None,
               # Add more from original (e.g., context_weight_primary * type_multiplier)
           })
       ```
     - In `_differentiable_multi_feature_scores`, condition on `features['chart_type']` (add to input from generator).
       ```python
       if features['chart_type'] == 'scatter':
           scores['scale_label'] -= self.type_specific_params['scatter']['dual_axis_penalty'] if not (nx < 0.2 or nx > 0.8 or ny > 0.8) else 0.0
       # Similar for others
       ```
     - Update `load_training_data` to include `'chart_type'` from JSON metadata.
     - **Cost**: +20-30% parameters; tune in phases (global first, then per-type).

#### 2. **Make Clustering Differentiable (Medium Priority - Fixes Gradient Flow)**
   - **Rationale**: DBSCAN's non-differentiability blocks gradients for `eps_factor` and downstream scores. Approximate with Gumbel-Softmax on KMeans or soft clustering for backprop.
   - **Impact**: Better tuning of grouping (e.g., dual axes), +2-4% on multi-axis charts.
   - **Addition**:
     - Add `_differentiable_dbscan_approx` using Torch's KMeans-like (or torch-cluster for soft assignments).
       ```python
       from sklearn.cluster import KMeans  # But make soft
       def _differentiable_dbscan_approx(positions: torch.Tensor, eps: float, min_samples: int) -> torch.Tensor:
           # Approximate: Use KMeans with k=2-3, then soft assignments via softmax on distances
           kmeans = KMeans(n_clusters=2)  # Fixed for dual-axis
           labels = torch.tensor(kmeans.fit_predict(positions.cpu().numpy()), device=self.device)
           # Soft: Compute distances to centroids, softmax
           centroids = torch.stack([positions[labels == i].mean(0) for i in range(2)])
           dists = torch.cdist(positions, centroids)
           soft_labels = torch.softmax(-dists / eps, dim=1)  # Temperature=eps
           return soft_labels  # [N, K] probabilities
       ```
     - Integrate in `classify_single_label`: Apply soft clustering if len(scale_labels) > 3, weight scores by soft_labels.
     - **Cost**: +10-20ms per forward; use for 'precise' mode only.

#### 3. **Enhance Loss and Optimization (High Priority - Improves Stability)**
   - **Rationale**: Cross-entropy ignores imbalance; add focal loss (from Torch) for rare classes (titles). No scheduler leads to stagnation; add ReduceLROnPlateau. Regularize to prevent extreme weights (e.g., L2 on params).
   - **Impact**: Reduces overfitting, +1-3% on imbalanced data.
   - **Addition**:
     - Replace `compute_loss` with focal loss (alpha=0.25 for imbalance, gamma=2.0).
       ```python
       import torch.nn.functional as F
       def compute_loss(self, pred_tensor, gt_tensor):
           ce_loss = F.cross_entropy(pred_tensor, gt_tensor, reduction='none')
           pt = torch.exp(-ce_loss)
           focal_loss = (0.25 * (1 - pt) ** 2.0 * ce_loss).mean()  # Alpha=0.25, gamma=2
           return focal_loss
       ```
     - In `train`, add scheduler and L2 reg.
       ```python
       self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
       # In train_epoch:
       l2_reg = sum(p.pow(2).sum() for p in self.params.values()) * 1e-4
       loss = self.hypertuner.compute_loss(...) + l2_reg
       self.scheduler.step(loss)
       ```
     - Add class-weighted CE if focal insufficient: weights = [1.0, 1.2, 2.0] for [scale, tick, title].

#### 4. **Data Augmentation and Real-Data Mixing (Medium Priority - Boosts Generalization)**
   - **Rationale**: Synthetic data lacks real noise (e.g., blur, rotation). Augment features (e.g., perturb positions) or mix with real labeled charts.
   - **Impact**: +5-10% on real data transfer; prevents synthetic-reality gap.
   - **Addition**:
     - In `load_training_data`, add augmentation loop.
       ```python
       augmented_data = []
       for features in training_data:
           aug = features.copy()
           aug['normalized_pos'] = (aug['normalized_pos'][0] + np.random.uniform(-0.05, 0.05), ...)  # Perturb
           augmented_data.append(aug)
       training_data.extend(augmented_data)
       ground_truth.extend(ground_truth)  # Duplicate labels
       ```
     - Add option for real data path: Merge JSON from real annotations (e.g., via `analysis.py` outputs).

#### 5. **Metrics, Logging, and Visualization (Low Priority - For Debugging)**
   - **Rationale**: Basic logging; add per-class F1/confusion matrices. Visualize param evolution.
   - **Impact**: Easier debugging, no direct accuracy gain.
   - **Addition**:
     - In `train_epoch`, compute F1 with `sklearn.metrics.f1_score` (available via numpy/scikit).
     - Use `torch.utils.tensorboard` (Torch available) for logging.
       ```python
       from torch.utils.tensorboard import SummaryWriter
       writer = SummaryWriter()
       writer.add_scalar('Loss/train', loss, epoch)
       # For params: writer.add_scalar('Params/sigma_x', self.params['sigma_x'].item(), epoch)
       ```

#### 6. **Hybrid with Black-Box Optimization (Medium Priority - If Differentiability Issues Persist)**
   - **Rationale**: For non-differentiable parts (DBSCAN), gradient-based is suboptimal. Hybrid: Use this for differentiable scores, then Optuna (from `hyperparameter_tuner.py`) for clustering params.
   - **Impact**: Fuller optimization; +2-5% on clustering-heavy cases.
   - **Addition**: Wrap in Optuna objective, evaluating full pipeline loss.

#### 7. **Production Integration Enhancements (High Priority - For Usability)**
   - **Rationale**: No confidence-based fallback; add in `spatial_classify_axis_labels_enhanced`.
   - **Addition**: Load tuned JSON in `detection_settings`; if classification conf <0.7, use defaults.
     ```python
     # In spatial_classify_axis_labels_enhanced
     if classification_conf < 0.7:
         settings = default_settings  # From JSON
     ```

Implement iteratively: Start with type-specific params and differentiable clustering, as they address core flaws. Test on held-out data for overfitting. If param count grows >50, switch to Optuna fully for scalability.
```c