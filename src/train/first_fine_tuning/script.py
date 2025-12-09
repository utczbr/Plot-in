# Let me analyze the LYLAA implementation and identify the scoring parameters and error propagation opportunities

# First, let's examine the key components of the LYLAA system from the spatial_classification_enhanced.py
print("=== ANALYSIS OF LYLAA IMPLEMENTATION ===")
print()

print("1. SCORING PARAMETERS IDENTIFIED IN LYLAA SYSTEM:")
print("   - Gaussian kernel parameters (σ_x = 0.09, σ_y = 0.09)")
print("   - Region probability weights for different zones:")
print("     * left_y_axis, right_y_axis, bottom_x_axis, top_title, center_data")
print("   - Multi-feature scoring weights:")
print("     * Size constraints (3.0, 2.5 weight factors)")
print("     * Position-based scoring (5.0, 4.0 weight factors)")
print("     * Distance from center scoring (2.0 weight factor)")
print("     * Context-based scoring (4.0, 5.0 weight factors)")
print("     * OCR-based numeric ratio scoring (2.0, 1.0 weight factors)")
print("   - DBSCAN clustering parameters (eps = img_width * 0.12)")
print("   - Classification threshold (default 1.5)")
print()

print("2. CURRENT ERROR PROPAGATION PATHS:")
print("   - Gaussian kernel errors → region score errors")
print("   - Region score errors → multi-feature classification errors")
print("   - Classification errors → final label assignment errors")
print("   - DBSCAN parameter errors → clustering quality errors")
print()

print("3. FEASIBILITY ASSESSMENT FOR HYPERTUNING:")
print("✓ HIGHLY FEASIBLE - The LYLAA system has:")
print("  • Clear numerical parameters that can be optimized")
print("  • Differentiable scoring functions")
print("  • Ground truth labels from generator.py for supervision")
print("  • Existing error metrics (accuracy, precision, recall)")
print()

print("4. IMPLEMENTATION STRATEGY:")
print("   A. Parameterize all scoring weights and thresholds")
print("   B. Define loss function based on classification accuracy")
print("   C. Implement gradient computation for error backpropagation")
print("   D. Use ground truth from generator.py as supervision signal")
print("   E. Apply gradient-based optimization (Adam/SGD)")


"""
Advanced chart axis/scale/label clustering and analysis system with orientation-aware heuristics.
Implements LYLAA (Label-You-Label-Alignment-Accuracy) metric and robust clustering for various chart types.
"""


'''
Newest LYLAA implementation
'''
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.cluster import DBSCAN
from collections import defaultdict
import math
import logging

def directional_region_scores(feats: List[Dict], orientation: str, chart_bounds: Dict) -> np.ndarray:
    """
    Compute orientation-aware region scores for label elements.
    
    Args:
        feats: List of feature dictionaries with centroid, dimensions, etc.
        orientation: 'vertical' or 'horizontal'
        chart_bounds: Dict with 'left', 'right', 'top', 'bottom' boundaries
        
    Returns:
        Array of directional scores [0,1] for each feature
    """
    scores = np.zeros(len(feats))
    
    for i, feat in enumerate(feats):
        cx, cy = feat['centroid']
        w, h = feat['dimensions']
        
        if orientation == 'vertical':
            # For vertical charts, Y-axis labels should be on left/right margins
            # Score based on proximity to left/right edges and text aspect ratio
            left_dist = abs(cx - chart_bounds['left']) / (chart_bounds['right'] - chart_bounds['left'])
            right_dist = abs(cx - chart_bounds['right']) / (chart_bounds['right'] - chart_bounds['left'])
            edge_score = 1.0 - min(left_dist, right_dist)
            
            # Prefer horizontal text layout for Y-axis labels
            aspect_score = min(w/h, 2.0) / 2.0 if h > 0 else 0.5
            
            # Vertical position should span reasonable chart range
            y_coverage = (cy - chart_bounds['top']) / (chart_bounds['bottom'] - chart_bounds['top'])
            coverage_score = 1.0 - abs(0.5 - y_coverage)
            
            scores[i] = 0.5 * edge_score + 0.3 * aspect_score + 0.2 * coverage_score
            
        else:  # horizontal
            # For horizontal charts, X-axis labels should be on top/bottom margins
            top_dist = abs(cy - chart_bounds['top']) / (chart_bounds['bottom'] - chart_bounds['top'])
            bottom_dist = abs(cy - chart_bounds['bottom']) / (chart_bounds['bottom'] - chart_bounds['top'])
            edge_score = 1.0 - min(top_dist, bottom_dist)
            
            # Prefer compact/square text for X-axis labels
            aspect_score = min(h/w, 1.5) / 1.5 if w > 0 else 0.5
            
            # Horizontal position coverage
            x_coverage = (cx - chart_bounds['left']) / (chart_bounds['right'] - chart_bounds['left'])
            coverage_score = 1.0 - abs(0.5 - x_coverage)
            
            scores[i] = 0.5 * edge_score + 0.3 * aspect_score + 0.2 * coverage_score
    
    return np.clip(scores, 0.0, 1.0)

def directional_feature_adjustments(feats: List[Dict], orientation: str, base_scores: np.ndarray) -> np.ndarray:
    """
    Transform per-element scores for DBSCAN weighting based on directional heuristics.
    
    Args:
        feats: Feature list
        orientation: Chart orientation
        base_scores: Base directional scores from directional_region_scores
        
    Returns:
        Adjusted weights for DBSCAN clustering
    """
    weights = base_scores.copy()
    
    for i, feat in enumerate(feats):
        # OCR confidence boost
        ocr_conf = feat['label'].get('ocr_conf', 0.5)
        ocr_boost = 0.2 * ocr_conf
        
        # Numeric content detection
        text = feat['label'].get('text', '')
        is_numeric = any(c.isdigit() or c in '.-+' for c in text)
        numeric_boost = 0.3 if is_numeric else 0.0
        
        # Size consistency - prefer medium-sized labels
        w, h = feat['dimensions']
        size_score = 1.0 - abs(0.5 - min(feat['relative_size'][0], 1.0))
        size_boost = 0.1 * size_score
        
        # Apply adjustments
        weights[i] = min(1.0, weights[i] + ocr_boost + numeric_boost + size_boost)
    
    return weights

def adaptive_dbscan_params(positions: np.ndarray, image_dims: Tuple[int, int], 
                          orientation: str, element_context: Dict, 
                          feat_scores: np.ndarray) -> Tuple[float, int, np.ndarray]:
    """
    Compute adaptive DBSCAN parameters based on data statistics and context.
    
    Args:
        positions: Nx2 array of label positions
        image_dims: (width, height) of image
        orientation: Chart orientation
        element_context: Context dictionary with spreads, spacing etc.
        feat_scores: Feature quality scores
        
    Returns:
        (eps, min_samples, position_weights)
    """
    if len(positions) < 2:
        return 10.0, 2, np.ones(len(positions))
    
    # Base eps from image dimensions (12% baseline, adapted to spread)
    img_w, img_h = image_dims
    base_eps = 0.12 * min(img_w, img_h)
    
    # Adapt to data spread and variance
    if orientation == 'vertical':
        # For Y-axis labels, cluster primarily on Y coordinate
        coord_std = np.std(positions[:, 1])
        coord_range = np.ptp(positions[:, 1])
    else:
        # For X-axis labels, cluster primarily on X coordinate  
        coord_std = np.std(positions[:, 0])
        coord_range = np.ptp(positions[:, 0])
    
    # Variance-based eps adjustment
    if coord_std > 0:
        variance_factor = min(2.0, coord_std / (coord_range * 0.1)) if coord_range > 0 else 1.0
        eps = base_eps * (0.6 + 0.9 * variance_factor)
    else:
        eps = base_eps * 0.8
    
    # Local density estimation for min_samples
    n_labels = len(positions)
    density_factor = element_context.get('avg_spacing', eps) / eps
    min_samples = max(2, min(5, int(0.02 * n_labels * density_factor)))
    
    # Position weights based on feature scores and local density
    weights = feat_scores.copy()
    
    # Boost weights for positions with high local density
    for i, pos in enumerate(positions):
        distances = np.linalg.norm(positions - pos, axis=1)
        k_nearest = min(5, len(distances))
        knn_dist = np.partition(distances, k_nearest)[k_nearest]
        density_boost = 0.2 * (1.0 - min(1.0, knn_dist / eps))
        weights[i] = min(1.0, weights[i] + density_boost)
    
    return eps, min_samples, weights

def variance_sensitive_heuristic(positions: np.ndarray, feats: List[Dict], 
                               orientation: str) -> Dict[str, float]:
    """
    Analyze position variance and provide sensitivity adjustments for clustering.
    
    Args:
        positions: Label positions
        feats: Feature dictionaries
        orientation: Chart orientation
        
    Returns:
        Dictionary with variance analysis and adjustment parameters
    """
    if len(positions) < 2:
        return {'variance': 0.0, 'eps_multiplier': 1.0, 'min_samples_adjustment': 0}
    
    # Select primary clustering coordinate
    if orientation == 'vertical':
        primary_coord = positions[:, 1]  # Y for vertical charts
        secondary_coord = positions[:, 0]  # X
    else:
        primary_coord = positions[:, 0]  # X for horizontal charts
        secondary_coord = positions[:, 1]  # Y
    
    # Variance analysis
    primary_std = np.std(primary_coord)
    primary_range = np.ptp(primary_coord)
    variance_ratio = primary_std / primary_range if primary_range > 0 else 0.0
    
    # Secondary coordinate spread (detect dual-axis scenarios)
    secondary_std = np.std(secondary_coord)
    secondary_clusters = len(np.unique(np.round(secondary_coord / max(1, secondary_std))))
    
    # Adjustments based on variance profile
    if variance_ratio < 0.1:  # Low variance (tight clusters)
        eps_multiplier = 0.7
        min_samples_adj = 1
    elif variance_ratio > 0.4:  # High variance (spread out)
        eps_multiplier = 1.4
        min_samples_adj = -1
    else:  # Medium variance
        eps_multiplier = 1.0
        min_samples_adj = 0
    
    # Special handling for potential dual-axis (high secondary spread)
    if secondary_clusters >= 2 and secondary_std > 0.2 * np.ptp(secondary_coord):
        eps_multiplier *= 0.8  # Tighter clustering to separate axes
    
    return {
        'variance': variance_ratio,
        'eps_multiplier': eps_multiplier,
        'min_samples_adjustment': min_samples_adj,
        'secondary_clusters': secondary_clusters,
        'dual_axis_likely': secondary_clusters >= 2
    }

def dual_axis_similarity_score(cluster1_feats: List[Dict], cluster2_feats: List[Dict]) -> float:
    """
    Compute similarity score between two axis label clusters for dual-axis detection.
    
    Args:
        cluster1_feats: Features from first cluster
        cluster2_feats: Features from second cluster
        
    Returns:
        Similarity score [0,1] where higher indicates likely dual-axis relationship
    """
    if len(cluster1_feats) < 2 or len(cluster2_feats) < 2:
        return 0.0
    
    # Extract numeric values where possible
    def extract_numeric_values(feats):
        values = []
        for feat in feats:
            text = feat['label'].get('text', '')
            try:
                # Try to extract numeric value
                cleaned = ''.join(c for c in text if c.isdigit() or c in '.-+')
                if cleaned and cleaned not in '.-+':
                    values.append(float(cleaned))
            except:
                pass
        return np.array(values)
    
    vals1 = extract_numeric_values(cluster1_feats)
    vals2 = extract_numeric_values(cluster2_feats)
    
    similarity_scores = []
    
    # Numeric progression similarity
    if len(vals1) >= 2 and len(vals2) >= 2:
        # Compare step sizes after normalization
        diffs1 = np.diff(np.sort(vals1))
        diffs2 = np.diff(np.sort(vals2))
        
        if len(diffs1) > 0 and len(diffs2) > 0:
            # Normalize by range
            range1 = np.ptp(vals1)
            range2 = np.ptp(vals2)
            if range1 > 0 and range2 > 0:
                norm_diffs1 = diffs1 / range1
                norm_diffs2 = diffs2 / range2
                
                # Compare normalized step consistency
                std1 = np.std(norm_diffs1)
                std2 = np.std(norm_diffs2)
                step_similarity = 1.0 - min(1.0, abs(std1 - std2))
                similarity_scores.append(step_similarity)
    
    # Geometric arrangement similarity
    pos1 = np.array([feat['centroid'] for feat in cluster1_feats])
    pos2 = np.array([feat['centroid'] for feat in cluster2_feats])
    
    # Y-spread comparison (for dual Y-axes)
    y_spread1 = np.ptp(pos1[:, 1])
    y_spread2 = np.ptp(pos2[:, 1])
    if max(y_spread1, y_spread2) > 0:
        spread_similarity = 1.0 - abs(y_spread1 - y_spread2) / max(y_spread1, y_spread2)
        similarity_scores.append(spread_similarity)
    
    # Label count similarity
    count_similarity = 1.0 - abs(len(cluster1_feats) - len(cluster2_feats)) / max(len(cluster1_feats), len(cluster2_feats))
    similarity_scores.append(count_similarity)
    
    # OCR confidence similarity
    conf1 = np.mean([feat['label'].get('ocr_conf', 0.5) for feat in cluster1_feats])
    conf2 = np.mean([feat['label'].get('ocr_conf', 0.5) for feat in cluster2_feats])
    conf_similarity = 1.0 - abs(conf1 - conf2)
    similarity_scores.append(conf_similarity)
    
    return np.mean(similarity_scores) if similarity_scores else 0.0

def postprocess_axis_clusters(clustered_labels: Dict[int, List[Dict]], 
                            feats: List[Dict], element_context: Dict) -> List[Dict]:
    """
    Post-process DBSCAN clusters to merge/split based on heuristics.
    
    Args:
        clustered_labels: Dict mapping cluster_id to list of label features
        feats: All feature dictionaries
        element_context: Context information
        
    Returns:
        List of axis objects with 'side', 'labels', 'scale_model'
    """
    orientation = element_context.get('chart_type', 'vertical')
    axes = []
    
    for cluster_id, labels in clustered_labels.items():
        if cluster_id == -1 or len(labels) < 2:  # Skip noise and tiny clusters
            continue
            
        # Extract positions for analysis
        positions = np.array([label['centroid'] for label in labels])
        
        # Determine primary clustering coordinate
        if 'vertical' in orientation.lower():
            primary_coords = positions[:, 1]  # Y coordinates
            secondary_coords = positions[:, 0]  # X coordinates
        else:
            primary_coords = positions[:, 0]  # X coordinates  
            secondary_coords = positions[:, 1]  # Y coordinates
        
        # Sort labels by primary coordinate
        sorted_indices = np.argsort(primary_coords)
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_coords = primary_coords[sorted_indices]
        
        # Detect large gaps that indicate cluster splits
        coord_diffs = np.diff(sorted_coords)
        median_diff = np.median(coord_diffs) if len(coord_diffs) > 0 else 0
        large_gap_threshold = 3.0 * median_diff if median_diff > 0 else np.inf
        
        split_points = []
        for i, diff in enumerate(coord_diffs):
            if diff > large_gap_threshold:
                split_points.append(i + 1)
        
        # Split cluster if large gaps found
        if split_points:
            split_points = [0] + split_points + [len(sorted_labels)]
            for i in range(len(split_points) - 1):
                start_idx = split_points[i]
                end_idx = split_points[i + 1]
                sub_labels = sorted_labels[start_idx:end_idx]
                
                if len(sub_labels) >= 2:
                    axis = create_axis_object(sub_labels, orientation, element_context)
                    axes.append(axis)
        else:
            # Keep cluster as single axis
            axis = create_axis_object(sorted_labels, orientation, element_context)
            axes.append(axis)
    
    # Merge nearby axes with compatible numeric progressions
    axes = merge_compatible_axes(axes, element_context)
    
    return axes

def create_axis_object(labels: List[Dict], orientation: str, element_context: Dict) -> Dict:
    """Create axis object from clustered labels."""
    positions = np.array([label['centroid'] for label in labels])
    
    # Determine axis side based on position
    chart_bounds = element_context.get('extent', {})
    avg_x = np.mean(positions[:, 0])
    avg_y = np.mean(positions[:, 1])
    
    if 'vertical' in orientation.lower():
        # Y-axis labels - determine left or right
        chart_center_x = (chart_bounds.get('left', 0) + chart_bounds.get('right', 1000)) / 2
        side = 'left' if avg_x < chart_center_x else 'right'
    else:
        # X-axis labels - determine top or bottom
        chart_center_y = (chart_bounds.get('top', 0) + chart_bounds.get('bottom', 1000)) / 2
        side = 'bottom' if avg_y > chart_center_y else 'top'
    
    # Fit linear scale model (position -> value mapping)
    scale_model = fit_scale_model(labels, orientation)
    
    return {
        'side': side,
        'labels': labels,
        'scale_model': scale_model,
        'label_count': len(labels),
        'span': np.ptp(positions[:, 1 if 'vertical' in orientation.lower() else 0])
    }

def fit_scale_model(labels: List[Dict], orientation: str) -> Dict:
    """Fit linear scale model from label positions to values."""
    positions = []
    values = []
    
    for label in labels:
        pos = label['centroid'][1 if 'vertical' in orientation.lower() else 0]
        text = label['label'].get('text', '')
        
        try:
            # Extract numeric value
            cleaned = ''.join(c for c in text if c.isdigit() or c in '.-+')
            if cleaned and cleaned not in '.-+':
                value = float(cleaned)
                positions.append(pos)
                values.append(value)
        except:
            continue
    
    if len(positions) >= 2:
        # Linear regression: value = slope * position + intercept
        positions = np.array(positions)
        values = np.array(values)
        
        A = np.vstack([positions, np.ones(len(positions))]).T
        slope, intercept = np.linalg.lstsq(A, values, rcond=None)[0]
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': calculate_r_squared(positions, values, slope, intercept)
        }
    
    return {'slope': 0.0, 'intercept': 0.0, 'r_squared': 0.0}

def calculate_r_squared(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    """Calculate R-squared for linear fit."""
    if len(y) < 2:
        return 0.0
    
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def merge_compatible_axes(axes: List[Dict], element_context: Dict) -> List[Dict]:
    """Merge axes that are nearby and have compatible numeric progressions."""
    if len(axes) <= 1:
        return axes
    
    merged_axes = []
    used_indices = set()
    
    for i, axis1 in enumerate(axes):
        if i in used_indices:
            continue
            
        candidates_to_merge = [axis1]
        used_indices.add(i)
        
        for j, axis2 in enumerate(axes[i+1:], i+1):
            if j in used_indices:
                continue
                
            # Check if axes are compatible for merging
            if should_merge_axes(axis1, axis2, element_context):
                candidates_to_merge.append(axis2)
                used_indices.add(j)
        
        # Merge if multiple candidates found
        if len(candidates_to_merge) > 1:
            merged_axis = merge_axis_group(candidates_to_merge, element_context)
            merged_axes.append(merged_axis)
        else:
            merged_axes.append(axis1)
    
    return merged_axes

def should_merge_axes(axis1: Dict, axis2: Dict, element_context: Dict) -> bool:
    """Check if two axes should be merged."""
    # Must be on same side
    if axis1['side'] != axis2['side']:
        return False
    
    # Check proximity
    pos1 = np.array([label['centroid'] for label in axis1['labels']])
    pos2 = np.array([label['centroid'] for label in axis2['labels']])
    
    orientation = element_context.get('chart_type', 'vertical')
    if 'vertical' in orientation.lower():
        # Check X proximity for Y-axis labels
        x_dist = abs(np.mean(pos1[:, 0]) - np.mean(pos2[:, 0]))
        proximity_threshold = 0.1 * element_context.get('x_spread', 100)
    else:
        # Check Y proximity for X-axis labels
        y_dist = abs(np.mean(pos1[:, 1]) - np.mean(pos2[:, 1]))
        proximity_threshold = 0.1 * element_context.get('y_spread', 100)
        x_dist = y_dist  # Reuse variable name
    
    if x_dist > proximity_threshold:
        return False
    
    # Check numeric progression compatibility
    model1 = axis1['scale_model']
    model2 = axis2['scale_model']
    
    if model1['r_squared'] > 0.5 and model2['r_squared'] > 0.5:
        # Compare slopes (should be similar)
        slope_ratio = model1['slope'] / model2['slope'] if model2['slope'] != 0 else 0
        if not (0.7 <= slope_ratio <= 1.3):
            return False
    
    return True

def merge_axis_group(axes: List[Dict], element_context: Dict) -> Dict:
    """Merge a group of compatible axes into a single axis."""
    all_labels = []
    for axis in axes:
        all_labels.extend(axis['labels'])
    
    # Sort labels by position
    orientation = element_context.get('chart_type', 'vertical')
    coord_idx = 1 if 'vertical' in orientation.lower() else 0
    all_labels.sort(key=lambda x: x['centroid'][coord_idx])
    
    # Create merged axis
    return create_axis_object(all_labels, orientation, element_context)

def compute_LYLAA(axis_objects: List[Dict], element_context: Dict) -> Dict:
    """
    Compute Label-You-Label-Alignment-Accuracy (LYLAA) metric.
    
    LYLAA combines multiple factors:
    - Alignment quality (geometric consistency)
    - Numeric consistency (monotonic progression, regular spacing)
    - Coverage (how much of chart space is covered by axis labels)
    - Isolation (separation between different axes)
    
    Args:
        axis_objects: List of detected axis objects
        element_context: Chart context information
        
    Returns:
        Dict with overall LYLAA score [0,1] and component breakdowns
    """
    if not axis_objects:
        return {
            'lylaa_score': 0.0,
            'alignment_score': 0.0,
            'numeric_consistency': 0.0,
            'coverage_score': 0.0,
            'isolation_score': 0.0,
            'axis_count': 0
        }
    
    # Component scores
    alignment_scores = []
    numeric_scores = []
    coverage_scores = []
    
    chart_bounds = element_context.get('extent', {})
    chart_width = chart_bounds.get('right', 1000) - chart_bounds.get('left', 0)
    chart_height = chart_bounds.get('bottom', 1000) - chart_bounds.get('top', 0)
    
    for axis in axis_objects:
        labels = axis['labels']
        if len(labels) < 2:
            continue
            
        positions = np.array([label['centroid'] for label in labels])
        
        # 1. Alignment Score - geometric consistency
        orientation = element_context.get('chart_type', 'vertical')
        if 'vertical' in orientation.lower():
            # Y-axis: check X-alignment (should be vertically aligned)
            x_coords = positions[:, 0]
            x_std = np.std(x_coords)
            x_alignment = 1.0 - min(1.0, x_std / (0.05 * chart_width))
        else:
            # X-axis: check Y-alignment (should be horizontally aligned)
            y_coords = positions[:, 1]
            y_std = np.std(y_coords)
            x_alignment = 1.0 - min(1.0, y_std / (0.05 * chart_height))
        
        alignment_scores.append(x_alignment)
        
        # 2. Numeric Consistency - regular progression and spacing
        scale_model = axis['scale_model']
        r_squared = scale_model.get('r_squared', 0.0)
        
        # Check for regular spacing in numeric values
        numeric_values = []
        for label in labels:
            text = label['label'].get('text', '')
            try:
                cleaned = ''.join(c for c in text if c.isdigit() or c in '.-+')
                if cleaned and cleaned not in '.-+':
                    numeric_values.append(float(cleaned))
            except:
                pass
        
        if len(numeric_values) >= 3:
            numeric_values = np.array(sorted(numeric_values))
            diffs = np.diff(numeric_values)
            spacing_consistency = 1.0 - min(1.0, np.std(diffs) / np.mean(diffs)) if np.mean(diffs) > 0 else 0.5
            numeric_consistency = 0.6 * r_squared + 0.4 * spacing_consistency
        else:
            numeric_consistency = 0.3 * r_squared  # Partial credit for linear fit
        
        numeric_scores.append(numeric_consistency)
        
        # 3. Coverage Score - how much chart space is covered
        if 'vertical' in orientation.lower():
            axis_span = np.ptp(positions[:, 1])  # Y-span for Y-axis
            coverage = min(1.0, axis_span / (0.8 * chart_height))
        else:
            axis_span = np.ptp(positions[:, 0])  # X-span for X-axis
            coverage = min(1.0, axis_span / (0.8 * chart_width))
        
        coverage_scores.append(coverage)
    
    # 4. Isolation Score - separation between different axes
    isolation_score = 1.0
    if len(axis_objects) > 1:
        isolation_scores = []
        for i, axis1 in enumerate(axis_objects):
            for j, axis2 in enumerate(axis_objects[i+1:], i+1):
                pos1 = np.array([label['centroid'] for label in axis1['labels']])
                pos2 = np.array([label['centroid'] for label in axis2['labels']])
                
                # Measure separation distance
                min_dist = np.min([np.linalg.norm(p1 - p2) for p1 in pos1 for p2 in pos2])
                separation = min(1.0, min_dist / (0.1 * min(chart_width, chart_height)))
                isolation_scores.append(separation)
        
        isolation_score = np.mean(isolation_scores) if isolation_scores else 1.0
    
    # Combine component scores into LYLAA
    avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
    avg_numeric = np.mean(numeric_scores) if numeric_scores else 0.0
    avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
    
    # LYLAA formula: weighted combination of components
    lylaa_score = (0.3 * avg_alignment + 
                   0.3 * avg_numeric + 
                   0.25 * avg_coverage + 
                   0.15 * isolation_score)
    
    return {
        'lylaa_score': lylaa_score,
        'alignment_score': avg_alignment,
        'numeric_consistency': avg_numeric,
        'coverage_score': avg_coverage,
        'isolation_score': isolation_score,
        'axis_count': len(axis_objects)
    }

def _cluster_scale_labels_weighted_dbscan(scale_labels: List[Dict], orientation: str, 
                                        element_context: Dict, image_dims: Tuple[int, int]) -> List[Dict]:
    """
    Main clustering pipeline using all the above functions.
    
    Args:
        scale_labels: List of label feature dictionaries
        orientation: Chart orientation ('vertical' or 'horizontal')
        element_context: Chart context information
        image_dims: Image dimensions (width, height)
        
    Returns:
        List of axis objects
    """
    if len(scale_labels) < 2:
        return []
    
    # Extract positions and chart bounds
    positions = np.array([label['centroid'] for label in scale_labels])
    chart_bounds = element_context.get('extent', {
        'left': 0, 'right': image_dims[0], 'top': 0, 'bottom': image_dims[1]
    })
    
    # 1. Compute directional region scores
    region_scores = directional_region_scores(scale_labels, orientation, chart_bounds)
    
    # 2. Apply directional feature adjustments
    adjusted_weights = directional_feature_adjustments(scale_labels, orientation, region_scores)
    
    # 3. Variance sensitivity analysis
    variance_info = variance_sensitive_heuristic(positions, scale_labels, orientation)
    
    # 4. Compute adaptive DBSCAN parameters
    eps, min_samples, pos_weights = adaptive_dbscan_params(
        positions, image_dims, orientation, element_context, adjusted_weights
    )
    
    # Apply variance-based adjustments
    eps *= variance_info['eps_multiplier']
    min_samples = max(2, min_samples + variance_info['min_samples_adjustment'])
    
    # 5. Weighted DBSCAN clustering
    # Use primary coordinate for clustering (Y for vertical, X for horizontal)
    if orientation == 'vertical':
        cluster_coords = positions[:, [1]]  # Y coordinates only
    else:
        cluster_coords = positions[:, [0]]  # X coordinates only
    
    # Weight the coordinates by feature quality
    weighted_coords = cluster_coords * pos_weights.reshape(-1, 1)
    
    # Perform DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(weighted_coords)
    
    # 6. Group labels by cluster
    clustered_labels = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        clustered_labels[cluster_id].append(scale_labels[i])
    
    # 7. Post-process clusters
    axis_objects = postprocess_axis_clusters(clustered_labels, scale_labels, element_context)
    
    # 8. Handle dual-axis scenarios
    if variance_info['dual_axis_likely'] and len(axis_objects) >= 2:
        # Check for dual-axis relationships
        for i, axis1 in enumerate(axis_objects):
            for j, axis2 in enumerate(axis_objects[i+1:], i+1):
                similarity = dual_axis_similarity_score(axis1['labels'], axis2['labels'])
                # Store similarity info for potential dual-axis handling
                axis1['dual_axis_similarity'] = max(axis1.get('dual_axis_similarity', 0), similarity)
                axis2['dual_axis_similarity'] = max(axis2.get('dual_axis_similarity', 0), similarity)
    
    return axis_objects

# Test harness with synthetic scenarios
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def create_synthetic_feat(centroid, dimensions, text, ocr_conf=0.9):
        """Create synthetic feature dictionary for testing."""
        cx, cy = centroid
        w, h = dimensions
        return {
            'centroid': (cx, cy),
            'dimensions': (w, h),
            'aspect_ratio': w / h if h > 0 else 1.0,
            'relative_size': (w / 800, h / 600),  # Assuming 800x600 image
            'normalized_pos': (cx / 800, cy / 600),
            'label': {'text': text, 'ocr_conf': ocr_conf},
            'xyxy': (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
        }
    
    # Test Scenario 1: Vertical dual-Y bar chart
    print("=== Test Scenario 1: Vertical Dual-Y Bar Chart ===")
    
    # Left Y-axis labels (0, 20, 40, 60, 80, 100)
    left_labels = [
        create_synthetic_feat((50, 500), (30, 15), "0"),
        create_synthetic_feat((50, 400), (30, 15), "20"),
        create_synthetic_feat((50, 300), (30, 15), "40"),
        create_synthetic_feat((50, 200), (30, 15), "60"),
        create_synthetic_feat((50, 100), (30, 15), "80"),
        create_synthetic_feat((50, 50), (30, 15), "100"),
    ]
    
    # Right Y-axis labels (0.0, 0.5, 1.0, 1.5, 2.0)
    right_labels = [
        create_synthetic_feat((750, 500), (25, 15), "0.0"),
        create_synthetic_feat((750, 375), (25, 15), "0.5"),
        create_synthetic_feat((750, 250), (25, 15), "1.0"),
        create_synthetic_feat((750, 125), (25, 15), "1.5"),
        create_synthetic_feat((750, 50), (25, 15), "2.0"),
    ]
    
    all_labels_1 = left_labels + right_labels
    element_context_1 = {
        'extent': {'left': 100, 'right': 700, 'top': 30, 'bottom': 520},
        'positions': np.array([[label['centroid'][0], label['centroid'][1]] for label in all_labels_1]),
        'avg_spacing': 50.0,
        'chart_type': 'vertical',
        'x_spread': 600,
        'y_spread': 470
    }
    
    axes_1 = _cluster_scale_labels_weighted_dbscan(all_labels_1, 'vertical', element_context_1, (800, 600))
    lylaa_1 = compute_LYLAA(axes_1, element_context_1)
    
    print(f"Detected {len(axes_1)} axes")
    for i, axis in enumerate(axes_1):
        print(f"  Axis {i+1}: {axis['side']}, {axis['label_count']} labels, R²={axis['scale_model']['r_squared']:.3f}")
    print(f"LYLAA Score: {lylaa_1['lylaa_score']:.3f}")
    print(f"  - Alignment: {lylaa_1['alignment_score']:.3f}")
    print(f"  - Numeric: {lylaa_1['numeric_consistency']:.3f}")
    print(f"  - Coverage: {lylaa_1['coverage_score']:.3f}")
    print(f"  - Isolation: {lylaa_1['isolation_score']:.3f}")
    print()
    
    # Test Scenario 2: Horizontal bar chart
    print("=== Test Scenario 2: Horizontal Bar Chart ===")
    
    x_labels = [
        create_synthetic_feat((150, 550), (40, 20), "Q1"),
        create_synthetic_feat((300, 550), (40, 20), "Q2"),
        create_synthetic_feat((450, 550), (40, 20), "Q3"),
        create_synthetic_feat((600, 550), (40, 20), "Q4"),
    ]
    
    element_context_2 = {
        'extent': {'left': 100, 'right': 650, 'top': 50, 'bottom': 500},
        'positions': np.array([[label['centroid'][0], label['centroid'][1]] for label in x_labels]),
        'avg_spacing': 150.0,
        'chart_type': 'horizontal',
        'x_spread': 550,
        'y_spread': 450
    }
    
    axes_2 = _cluster_scale_labels_weighted_dbscan(x_labels, 'horizontal', element_context_2, (800, 600))
    lylaa_2 = compute_LYLAA(axes_2, element_context_2)
    
    print(f"Detected {len(axes_2)} axes")
    for i, axis in enumerate(axes_2):
        print(f"  Axis {i+1}: {axis['side']}, {axis['label_count']} labels, R²={axis['scale_model']['r_squared']:.3f}")
    print(f"LYLAA Score: {lylaa_2['lylaa_score']:.3f}")
    print()
    
    # Test Scenario 3: Scatter plot with low variance (clustered points)
    print("=== Test Scenario 3: Scatter Plot with Low Variance ===")
    
    # Tight clustering around center
    scatter_labels = [
        create_synthetic_feat((80, 480), (25, 15), "10", 0.85),
        create_synthetic_feat((85, 460), (25, 15), "15", 0.80),
        create_synthetic_feat((75, 440), (25, 15), "20", 0.90),
        create_synthetic_feat((90, 420), (25, 15), "25", 0.75),
        create_synthetic_feat((70, 400), (25, 15), "30", 0.85),
    ]
    
    element_context_3 = {
        'extent': {'left': 120, 'right': 720, 'top': 50, 'bottom': 500},
        'positions': np.array([[label['centroid'][0], label['centroid'][1]] for label in scatter_labels]),
        'avg_spacing': 20.0,
        'chart_type': 'vertical',
        'x_spread': 600,
        'y_spread': 450
    }
    
    # Test variance sensitivity
    positions_3 = np.array([[label['centroid'][0], label['centroid'][1]] for label in scatter_labels])
    variance_info_3 = variance_sensitive_heuristic(positions_3, scatter_labels, 'vertical')
    
    axes_3 = _cluster_scale_labels_weighted_dbscan(scatter_labels, 'vertical', element_context_3, (800, 600))
    lylaa_3 = compute_LYLAA(axes_3, element_context_3)
    
    print(f"Variance analysis: ratio={variance_info_3['variance']:.3f}, eps_mult={variance_info_3['eps_multiplier']:.2f}")
    print(f"Detected {len(axes_3)} axes")
    for i, axis in enumerate(axes_3):
        print(f"  Axis {i+1}: {axis['side']}, {axis['label_count']} labels, R²={axis['scale_model']['r_squared']:.3f}")
    print(f"LYLAA Score: {lylaa_3['lylaa_score']:.3f}")
    print()
    
    # Test dual-axis similarity
    if len(axes_1) >= 2:
        print("=== Dual-Axis Similarity Test ===")
        similarity = dual_axis_similarity_score(axes_1[0]['labels'], axes_1[1]['labels'])
        print(f"Dual-axis similarity between first two axes: {similarity:.3f}")
    
    print("=== Test Complete ===")