# ============================================================================
# COMPLETE INTEGRATION SOLUTION FOR LYLAA HYPERTUNING WITH GENERATOR
# ============================================================================

# ============================================================================
# FILE 1: generator.py - ADD THESE FUNCTIONS
# ============================================================================

def extract_label_features_for_hypertuning(fig, chart_info_map, img_w, img_h):
    """
    Extract complete feature vectors for hypertuning optimization.
    Returns list of feature dicts matching LYLAAHypertuner input format.
    """
    renderer = fig.canvas.get_renderer()
    label_features = []
    
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        
        chart_info = chart_info_map.get(ax, {})
        chart_type = chart_info.get('chart_type_str', 'unknown')
        orientation = chart_info.get('orientation', 'vertical')
        
        # Determine which axis is scale axis (numeric values)
        scale_axis_info = chart_info.get('scale_axis_info', {})
        primary_scale_axis = scale_axis_info.get('primary_scale_axis', 'y')
        
        # Extract X-axis labels with complete features
        for label in ax.get_xticklabels():
            if label.get_visible() and label.get_text():
                txt = label.get_text().strip()
                bbox = label.get_window_extent(renderer)
                if bbox.width > 1 and bbox.height > 1:
                    x0, y0 = bbox.x0, img_h - bbox.y1
                    x1, y1 = bbox.x1, img_h - bbox.y0
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    width, height = x1 - x0, y1 - y0
                    
                    # Determine ground truth class
                    is_numeric = is_float(txt.replace('%', '').replace(',', ''))
                    is_scale_axis = (primary_scale_axis == 'x')
                    
                    if is_numeric and is_scale_axis:
                        true_class = 0  # scale_label
                        class_name = 'scale_label'
                    elif not is_numeric:
                        true_class = 1  # tick_label (category)
                        class_name = 'tick_label'
                    else:
                        true_class = 0  # default to scale
                        class_name = 'scale_label'
                    
                    features = {
                        'text': txt,
                        'xyxy': [int(x0), int(y0), int(x1), int(y1)],
                        'normalized_pos': (cx / img_w, cy / img_h),
                        'relative_size': (width / img_w, height / img_h),
                        'aspect_ratio': width / (height + 1e-6),
                        'area': width * height,
                        'centroid': (cx, cy),
                        'bbox': [x0, y0, x1, y1],
                        'dimensions': (width, height),
                        'axis': 'x',
                        'is_numeric': is_numeric,
                        'chart_type': chart_type,
                        'orientation': orientation,
                        'true_class': true_class,
                        'class_name': class_name
                    }
                    label_features.append(features)
        
        # Extract Y-axis labels with complete features
        for label in ax.get_yticklabels():
            if label.get_visible() and label.get_text():
                txt = label.get_text().strip()
                bbox = label.get_window_extent(renderer)
                if bbox.width > 1 and bbox.height > 1:
                    x0, y0 = bbox.x0, img_h - bbox.y1
                    x1, y1 = bbox.x1, img_h - bbox.y0
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    width, height = x1 - x0, y1 - y0
                    
                    is_numeric = is_float(txt.replace('%', '').replace(',', ''))
                    is_scale_axis = (primary_scale_axis == 'y')
                    
                    if is_numeric and is_scale_axis:
                        true_class = 0
                        class_name = 'scale_label'
                    elif not is_numeric:
                        true_class = 1
                        class_name = 'tick_label'
                    else:
                        true_class = 0
                        class_name = 'scale_label'
                    
                    features = {
                        'text': txt,
                        'xyxy': [int(x0), int(y0), int(x1), int(y1)],
                        'normalized_pos': (cx / img_w, cy / img_h),
                        'relative_size': (width / img_w, height / img_h),
                        'aspect_ratio': width / (height + 1e-6),
                        'area': width * height,
                        'centroid': (cx, cy),
                        'bbox': [x0, y0, x1, y1],
                        'dimensions': (width, height),
                        'axis': 'y',
                        'is_numeric': is_numeric,
                        'chart_type': chart_type,
                        'orientation': orientation,
                        'true_class': true_class,
                        'class_name': class_name
                    }
                    label_features.append(features)
        
        # Extract axis titles (class 2)
        if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text():
            txt = ax.xaxis.label.get_text().strip()
            bbox = ax.xaxis.label.get_window_extent(renderer)
            if bbox.width > 1 and bbox.height > 1:
                x0, y0 = bbox.x0, img_h - bbox.y1
                x1, y1 = bbox.x1, img_h - bbox.y0
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                width, height = x1 - x0, y1 - y0
                
                features = {
                    'text': txt,
                    'xyxy': [int(x0), int(y0), int(x1), int(y1)],
                    'normalized_pos': (cx / img_w, cy / img_h),
                    'relative_size': (width / img_w, height / img_h),
                    'aspect_ratio': width / (height + 1e-6),
                    'area': width * height,
                    'centroid': (cx, cy),
                    'bbox': [x0, y0, x1, y1],
                    'dimensions': (width, height),
                    'axis': 'x',
                    'is_numeric': False,
                    'chart_type': chart_type,
                    'orientation': orientation,
                    'true_class': 2,
                    'class_name': 'axis_title'
                }
                label_features.append(features)
        
        if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text():
            txt = ax.yaxis.label.get_text().strip()
            bbox = ax.yaxis.label.get_window_extent(renderer)
            if bbox.width > 1 and bbox.height > 1:
                x0, y0 = bbox.x0, img_h - bbox.y1
                x1, y1 = bbox.x1, img_h - bbox.y0
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                width, height = x1 - x0, y1 - y0
                
                features = {
                    'text': txt,
                    'xyxy': [int(x0), int(y0), int(x1), int(y1)],
                    'normalized_pos': (cx / img_w, cy / img_h),
                    'relative_size': (width / img_w, height / img_h),
                    'aspect_ratio': width / (height + 1e-6),
                    'area': width * height,
                    'centroid': (cx, cy),
                    'bbox': [x0, y0, x1, y1],
                    'dimensions': (width, height),
                    'axis': 'y',
                    'is_numeric': False,
                    'chart_type': chart_type,
                    'orientation': orientation,
                    'true_class': 2,
                    'class_name': 'axis_title'
                }
                label_features.append(features)
    
    return label_features


# ============================================================================
# MODIFY main() IN generator.py - ADD AFTER LINE ~1480 (after ocr extraction)
# ============================================================================

# Inside main() generation loop, after:
# ocr_ground_truth = extract_ocr_ground_truth(fig, chart_info_map)

# ADD THIS:
label_features = extract_label_features_for_hypertuning(fig, chart_info_map, img_w, img_h)

hypertuning_data = {
    'image_id': base_filename,
    'image_dimensions': {'width': img_w, 'height': img_h},
    'chart_type': chart_info_map[axes[0]]['chart_type_str'] if axes else 'unknown',
    'orientation': chart_info_map[axes[0]]['orientation'] if axes else 'vertical',
    'label_features': label_features,
    'num_labels': len(label_features),
    'class_distribution': {
        'scale_label': sum(1 for f in label_features if f['true_class'] == 0),
        'tick_label': sum(1 for f in label_features if f['true_class'] == 1),
        'axis_title': sum(1 for f in label_features if f['true_class'] == 2)
    }
}

hypertuning_file = os.path.join(labels_dir, f"{base_filename}_hypertuning.json")
with open(hypertuning_file, 'w') as f:
    json.dump(convert_numpy_types(hypertuning_data), f, indent=2)

print(f" ✓ Saved hypertuning data with {len(label_features)} label features")


# ============================================================================
# FILE 2: lylaa-hypertuner.py - REPLACE load_training_data() METHOD
# ============================================================================

def load_training_data(self) -> Tuple[List[Dict], List[int]]:
    """
    Load ground truth data from generator.py hypertuning outputs.
    CRITICAL FIX: Reads _hypertuning.json files with complete feature vectors.
    """
    training_data = []
    ground_truth_labels = []
    
    labels_dir = self.generator_output_dir / 'labels'
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return [], []
    
    for hypertuning_file in sorted(labels_dir.glob('*_hypertuning.json')):
        try:
            with open(hypertuning_file, 'r') as f:
                data = json.load(f)
            
            label_features = data.get('label_features', [])
            
            for features in label_features:
                # Extract ground truth class
                true_class = features.get('true_class', 0)
                
                # Ensure all required features are present
                if 'normalized_pos' in features and 'relative_size' in features:
                    training_data.append(features)
                    ground_truth_labels.append(true_class)
                    
        except Exception as e:
            logger.warning(f"Error processing {hypertuning_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(training_data)} training samples from {len(list(labels_dir.glob('*_hypertuning.json')))} files")
    
    # Class distribution analysis
    class_counts = {0: 0, 1: 0, 2: 0}
    for label in ground_truth_labels:
        class_counts[label] += 1
    
    logger.info(f"Class distribution: scale_label={class_counts[0]}, tick_label={class_counts[1]}, axis_title={class_counts[2]}")
    
    # Check for class imbalance
    total = len(ground_truth_labels)
    if total > 0:
        for cls, count in class_counts.items():
            ratio = count / total
            if ratio < 0.05:
                logger.warning(f"Class {cls} is underrepresented ({ratio*100:.1f}% of data)")
    
    return training_data, ground_truth_labels


# ============================================================================
# FILE 3: spatial_classification_enhanced.py - CRITICAL MODIFICATIONS
# ============================================================================

# MODIFY _compute_octant_region_scores() - ADD settings PARAMETER

def _compute_octant_region_scores(
    normalized_pos: Tuple[float, float],
    img_width: int,
    img_height: int,
    settings: Dict = None  # CRITICAL ADDITION
) -> Dict[str, float]:
    """
    Compute Gaussian-kernel probabilistic scores with hypertuned parameters.
    """
    nx, ny = normalized_pos
    settings = settings or {}
    
    # Get hypertuned parameters or defaults
    sigma_x = settings.get('sigma_x', 0.09)
    sigma_y = settings.get('sigma_y', 0.09)
    left_weight = settings.get('left_y_axis_weight', 5.0)
    right_weight = settings.get('right_y_axis_weight', 4.0)
    bottom_weight = settings.get('bottom_x_axis_weight', 5.0)
    top_weight = settings.get('top_title_weight', 4.0)
    center_weight = settings.get('center_data_weight', 2.0)
    
    scores = {}
    
    # Left Y-axis region with hypertuned Gaussian
    if nx < 0.20 and 0.1 < ny < 0.9:
        dx = (nx - 0.08) / sigma_x
        dy = (ny - 0.5) / sigma_y
        scores['left_y_axis'] = np.exp(-(dx**2 + dy**2) / 2) * left_weight
    else:
        scores['left_y_axis'] = 0.0
    
    # Right Y-axis region
    if nx > 0.80 and 0.1 < ny < 0.9:
        dx = (nx - 0.92) / sigma_x
        dy = (ny - 0.5) / sigma_y
        scores['right_y_axis'] = np.exp(-(dx**2 + dy**2) / 2) * right_weight
    else:
        scores['right_y_axis'] = 0.0
    
    # Bottom X-axis region
    if 0.15 < nx < 0.85 and ny > 0.80:
        dx = (nx - 0.5) / sigma_x
        dy = (ny - 0.92) / sigma_y
        scores['bottom_x_axis'] = np.exp(-(dx**2 + dy**2) / 2) * bottom_weight
    else:
        scores['bottom_x_axis'] = 0.0
    
    # Top title region
    if 0.15 < nx < 0.85 and ny < 0.15:
        dx = (nx - 0.5) / sigma_x
        dy = (ny - 0.08) / sigma_y
        scores['top_title'] = np.exp(-(dx**2 + dy**2) / 2) * top_weight
    else:
        scores['top_title'] = 0.0
    
    # Center data region
    if 0.2 < nx < 0.8 and 0.2 < ny < 0.8:
        center_dist = np.sqrt((nx - 0.5)**2 + (ny - 0.5)**2)
        scores['center_data'] = np.exp(-(center_dist**2) / 0.08) * center_weight
    else:
        scores['center_data'] = 0.0
    
    return scores


# MODIFY _classify_precise_mode() - PASS settings TO _compute_octant_region_scores

def _classify_precise_mode(
    axis_labels: List[Dict],
    chart_elements: List[Dict],
    chart_type: str,
    img_width: int,
    img_height: int,
    orientation: str,
    settings: Dict
) -> Dict[str, List[Dict]]:
    """PRECISE MODE with hypertuned parameters"""
    settings = settings or {}
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
    
    element_context = _compute_chart_element_context_features(
        chart_elements, chart_type, img_width, img_height, orientation
    )
    
    classified = {'scale_label': [], 'tick_label': [], 'axis_title': []}
    
    for feat in label_features:
        # CRITICAL: Pass settings to region scoring
        region_scores = _compute_octant_region_scores(
            feat['normalized_pos'],
            img_width,
            img_height,
            settings  # PASS HYPERTUNED PARAMS
        )
        
        class_scores = _compute_multi_feature_scores(
            feat,
            region_scores,
            element_context,
            orientation,
            settings  # PASS HYPERTUNED PARAMS
        )
        
        best_class, best_score = max(class_scores.items(), key=lambda x: x[1])
        threshold = settings.get('classification_threshold', 1.5)
        
        if best_score > threshold:
            classified[best_class].append(feat['label'])
        else:
            classified['scale_label'].append(feat['label'])
    
    if len(classified['scale_label']) > 3:
        classified['scale_label'] = _cluster_scale_labels_weighted_dbscan(
            classified['scale_label'],
            img_width,
            img_height,
            orientation,
            settings
        )
    
    return classified


# ============================================================================
# FILE 4: hypertuned_spatial_classifier.py - NEW FILE FOR INTEGRATION
# ============================================================================

import json
import logging
from pathlib import Path
from typing import Dict, List
from spatial_classification_enhanced import spatial_classify_axis_labels_enhanced

logger = logging.getLogger(__name__)

class HypertunedSpatialClassifier:
    """
    Production-ready classifier using optimized LYAA parameters.
    Loads hypertuning results and applies to spatial classification.
    """
    
    def __init__(self, hypertuning_results_path: str = 'lylaa_hypertuning_results.json'):
        """Initialize with hypertuned parameters"""
        results_path = Path(hypertuning_results_path)
        
        if not results_path.exists():
            logger.warning(f"Hypertuning results not found at {hypertuning_results_path}")
            logger.warning("Using default LYAA parameters")
            self.detection_settings = self._get_default_settings()
            return
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        optimal_params = results['optimal_parameters']
        logger.info(f"Loaded {len(optimal_params)} optimized parameters")
        logger.info(f"Training accuracy: {results.get('best_accuracy', 0):.4f}")
        
        self.detection_settings = {
            'classification_threshold': optimal_params['classification_threshold'],
            'size_threshold_width': optimal_params['size_threshold_width'],
            'size_threshold_height': optimal_params['size_threshold_height'],
            'aspect_ratio_min': optimal_params['aspect_ratio_min'],
            'aspect_ratio_max': optimal_params['aspect_ratio_max'],
            'eps_factor': optimal_params['eps_factor'],
            'left_y_axis_weight': optimal_params['left_y_axis_weight'],
            'right_y_axis_weight': optimal_params['right_y_axis_weight'],
            'bottom_x_axis_weight': optimal_params['bottom_x_axis_weight'],
            'top_title_weight': optimal_params['top_title_weight'],
            'center_data_weight': optimal_params['center_data_weight'],
            'size_constraint_primary': optimal_params['size_constraint_primary'],
            'size_constraint_secondary': optimal_params['size_constraint_secondary'],
            'aspect_ratio_weight': optimal_params['aspect_ratio_weight'],
            'position_weight_primary': optimal_params['position_weight_primary'],
            'position_weight_secondary': optimal_params['position_weight_secondary'],
            'distance_weight': optimal_params['distance_weight'],
            'context_weight_primary': optimal_params['context_weight_primary'],
            'context_weight_secondary': optimal_params['context_weight_secondary'],
            'ocr_numeric_boost': optimal_params['ocr_numeric_boost'],
            'ocr_numeric_penalty': optimal_params['ocr_numeric_penalty'],
            'sigma_x': optimal_params['sigma_x'],
            'sigma_y': optimal_params['sigma_y']
        }
    
    def _get_default_settings(self) -> Dict:
        """Fallback to default LYAA parameters"""
        return {
            'classification_threshold': 1.5,
            'size_threshold_width': 0.08,
            'size_threshold_height': 0.04,
            'aspect_ratio_min': 0.5,
            'aspect_ratio_max': 3.5,
            'eps_factor': 0.12,
            'left_y_axis_weight': 5.0,
            'right_y_axis_weight': 4.0,
            'bottom_x_axis_weight': 5.0,
            'top_title_weight': 4.0,
            'center_data_weight': 2.0,
            'size_constraint_primary': 3.0,
            'size_constraint_secondary': 2.5,
            'aspect_ratio_weight': 2.5,
            'position_weight_primary': 5.0,
            'position_weight_secondary': 4.0,
            'distance_weight': 2.0,
            'context_weight_primary': 4.0,
            'context_weight_secondary': 5.0,
            'ocr_numeric_boost': 2.0,
            'ocr_numeric_penalty': 1.0,
            'sigma_x': 0.09,
            'sigma_y': 0.09
        }
    
    def classify(self, axis_labels: List[Dict], chart_elements: List[Dict],
                 chart_type: str, image_width: int, image_height: int,
                 chart_orientation: str = 'vertical', mode: str = 'precise') -> Dict[str, List[Dict]]:
        """
        Classify axis labels using hypertuned parameters.
        
        Args:
            axis_labels: List of detected axis label bboxes
            chart_elements: List of detected chart data elements
            chart_type: 'bar', 'box', 'scatter', 'line'
            image_width, image_height: Image dimensions
            chart_orientation: 'vertical' or 'horizontal'
            mode: 'fast', 'optimized', or 'precise'
        
        Returns:
            Dict with keys 'scale_label', 'tick_label', 'axis_title'
        """
        return spatial_classify_axis_labels_enhanced(
            axis_labels=axis_labels,
            chart_elements=chart_elements,
            chart_type=chart_type,
            image_width=image_width,
            image_height=image_height,
            chart_orientation=chart_orientation,
            detection_settings=self.detection_settings,
            mode=mode
        )


# ============================================================================
# EXECUTION WORKFLOW
# ============================================================================

"""
STEP 1: Generate training data
-------------------------------
python generator.py --num 500

This creates:
- test_generation/images/*.png
- test_generation/labels/*_hypertuning.json (NEW - contains label features + ground truth)

STEP 2: Run hyperparameter optimization
---------------------------------------
python lylaa-hypertuner.py --data-dir test_generation --epochs 200 --lr 0.01

This creates:
- lylaa_hypertuning_results.json (optimized parameters)

STEP 3: Use hypertuned classifier in production
-----------------------------------------------
from hypertuned_spatial_classifier import HypertunedSpatialClassifier

classifier = HypertunedSpatialClassifier('lylaa_hypertuning_results.json')
classified = classifier.classify(
    axis_labels=detected_labels,
    chart_elements=detected_bars,
    chart_type='bar',
    image_width=800,
    image_height=600,
    chart_orientation='vertical',
    mode='precise'
)

scale_labels = classified['scale_label']
tick_labels = classified['tick_label']
axis_titles = classified['axis_title']
"""



The provided content outlines a comprehensive, gradient-based hypertuning system for the LYLAA spatial classification module, building directly on the recommendations from our previous discussion. It maintains the current system's structure (e.g., geometric, contextual, and OCR-based features) while making the parameters optimizable via PyTorch, using synthetic ground truth data from `generator.py`. This aligns with my earlier advice to start with hypertuning the existing parameters before adding new type-specific scoring methods—it's lower-risk and targets the empirical hardcoded weights (e.g., +4.0, +5.0 in `_compute_multi_feature_scores`).

### Key Highlights of the Implementation
- **Parameter Optimization Scope**: Covers 23 tunable parameters (close to the 24 listed in `script_1.py`), categorized into Gaussian kernels, region weights, feature weights, clustering (eps_factor), and thresholds. This captures most of the empirical values in the original `spatial_classification_enhanced.py`.
  - Example: Replaces hardcodes like `scores['scale_label'] += 3.5` with `self.params['size_constraint_primary']` in the differentiable surrogate model.
- **Differentiable Surrogate Model**: The `LYLAAHypertuner` class reimplements key functions (`differentiable_gaussian_score`, `_differentiable_region_scores`, `_differentiable_multi_feature_scores`) in PyTorch for auto-differentiation. This enables gradient descent on cross-entropy loss, treating classification as a 3-class problem (scale_label=0, tick_label=1, axis_title=2).
  - Handles non-differentiable elements gracefully (e.g., sigmoid approximation for thresholds).
  - Constraints prevent invalid values (e.g., sigmas >0.01).
- **Integration with Generator**: `extract_label_features_for_hypertuning` in `generator.py` extracts rich features (e.g., normalized_pos, aspect_ratio, is_numeric) and ground truth classes from Matplotlib renderings. `LYLAATrainer` loads this for training, assuming JSON outputs in `test_generation/labels`.
- **Training Workflow**: Adam optimizer, early stopping, history tracking. Runs on CPU/GPU, with progress logging.
- **Production Use**: `HypertunedSpatialClassifier` loads optimized params from JSON and passes them as `detection_settings` to `spatial_classify_axis_labels_enhanced`, which must be modified to use these (e.g., `scores['tick_label'] += settings['context_weight_primary'] * np.exp(...)` instead of fixed multipliers).
- **Modes Compatibility**: Hypertuning focuses on 'precise' mode but benefits all, as params are mode-agnostic.

### Strengths
- **Efficiency**: Gradient-based (analytical via autograd) over finite differences, as recommended in `script_1.py`. Suitable for 500+ synthetic samples (e.g., `--num 500` in generator).
- **Robustness**: Includes OCR hints via `ocr_numeric_boost/penalty` and `is_numeric`, aligning with numeric validation in `contextual_ocr.py`.
- **Modularity**: Easy to extend for type-specific params (e.g., add `bar_context_weight` if needed later).
- **Error Propagation**: Full backprop path: features → region scores → multi-feature scores → logits → loss vs. ground truth.

### Potential Issues and Fixes
- **Incomplete Type-Specific Logic**: The differentiable scoring in `_differentiable_multi_feature_scores` is generic and misses some chart_type branches from the original (e.g., bar/box spacing checks). **Fix**: Add conditional logic in the surrogate model, using one-hot encoded `chart_type` from features (e.g., if features['chart_type'] == 'bar': scores['tick_label'] += self.params['bar_spacing_weight'] * ...). Add corresponding params/constraints.
- **Missing Clustering in Forward Pass**: DBSCAN isn't simulated differentiably, so `eps_factor` gradients might be weak. **Fix**: Approximate clustering with differentiable KMeans (from `torch`) or tune it separately post-optimization.
- **Data Assumptions**: Load assumes fixed 800x600 size and `_detailed.json` format; real generator outputs might vary. **Fix**: Load `img_w`, `img_h` from JSON metadata; add error handling for missing files.
- **OCR Integration**: Features include `is_numeric`, but scoring doesn't use it yet (truncated code). **Fix**: In `_differentiable_multi_feature_scores`, add:
  ```
  if features.get('is_numeric', False):
      scores['scale_label'] += self.params['ocr_numeric_boost']
  else:
      scores['tick_label'] += self.params['ocr_numeric_penalty']
  ```
- **Scalability**: For large datasets, batch the forward pass in `train_epoch` (e.g., use DataLoader).
- **Evaluation**: Only accuracy/loss; add F1 per class in history for imbalanced labels (titles rarer than ticks).
- **Dependencies**: Relies on torch (available in your env); ensure numpy/torch compatibility for features.

### Recommendations for Next Steps
1. **Modify Original Classifier**: Update `spatial_classification_enhanced.py` to use `settings` for all weights/thresholds. Example for a score line:
   ```
   scores['tick_label'] += settings.get('context_weight_primary', 4.0) * np.exp(-(cy - el_extent['bottom']) / 50.0)
   ```
   Fallback to defaults if not in settings.

2. **Generate Data**: Run `python generator.py --num 1000` for diverse charts (bar/line/scatter/box, vertical/horizontal, dual-axis). Ensure even distribution across types/classes.

3. **Run Hypertuning**: Execute `python lylaa-hypertuner.py --data-dir test_generation --epochs 500 --lr 0.005`. Monitor history for convergence (plot losses/accuracies with matplotlib if needed).

4. **Validate**: After tuning, test on held-out synthetic data and real charts (e.g., via `analysis.py`). Compare accuracy vs. defaults.

5. **If Adding Type-Specific Scores**: If benchmarks show per-type gaps (e.g., <95% on scatters), add 4-6 new params (e.g., `scatter_x_spread_weight`). Retrain.

This setup should yield 2-5% accuracy gains as estimated. If you provide sample data or specific errors, I can refine further!