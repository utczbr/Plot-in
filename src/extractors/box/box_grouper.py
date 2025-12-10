"""
BoxGrouper module for grouping box plot elements.
"""
from typing import List, Dict, Tuple

def compute_aabb_intersection(box1_xyxy: Tuple[float, float, float, float], 
                            box2_xyxy: Tuple[float, float, float, float]) -> bool:
    """
    Axis-Aligned Bounding Box intersection test.
    Returns True if boxes overlap (shared area > 0).
    
    Computational complexity: O(1)
    """
    x1_min, y1_min, x1_max, y1_max = box1_xyxy
    x2_min, y2_min, x2_max, y2_max = box2_xyxy
    
    # Separating axis theorem: no intersection if separated on any axis
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True


def group_box_plot_elements(boxes: List[Dict], 
                          range_indicators: List[Dict], 
                          median_lines: List[Dict], 
                          outliers: List[Dict], 
                          tick_labels: List[Dict], 
                          orientation: str = 'vertical') -> List[Dict]:
    """
    Topology-aware grouping for box plots using intersection + coordinate alignment.
    
    Algorithm stages:
    1. Intersection detection: Group range_indicator and median_line that 
       spatially intersect with each box (IoU > 0 or bounding box overlap)
    2. Coordinate proximity: Group outliers and tick_labels based on 
       alignment with box center coordinates
       
    Mathematical foundation:
    - Intersection: Axis-Aligned Bounding Box (AABB) collision detection
    - Proximity: |coord_element - coord_box_center| < threshold
      where threshold = box_dimension × α, α ∈ [0.3, 0.5]
    """
    grouped = []
    
    for box in boxes:
        group = {
            'box': box,
            'range_indicator': None,
            'median_line': None,
            'outliers': [],
            'tick_label': None
        }
        
        # Extract box geometry
        x1, y1, x2, y2 = box['xyxy']
        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Stage 1: Intersection-based grouping for range_indicator and median_line
        # Collect ALL intersecting indicators (may have separate upper/lower whiskers)
        intersecting_indicators = [
            indicator for indicator in range_indicators
            if compute_aabb_intersection(box['xyxy'], indicator['xyxy'])
        ]
        
        if len(intersecting_indicators) == 1:
            # Single whisker element - standard case
            group['range_indicator'] = intersecting_indicators[0]
        elif len(intersecting_indicators) >= 2:
            # Multiple whisker elements detected - merge them
            # Compute bounding box covering all whisker elements
            all_x1 = min(ind['xyxy'][0] for ind in intersecting_indicators)
            all_y1 = min(ind['xyxy'][1] for ind in intersecting_indicators)
            all_x2 = max(ind['xyxy'][2] for ind in intersecting_indicators)
            all_y2 = max(ind['xyxy'][3] for ind in intersecting_indicators)
            
            # Create merged indicator
            merged_indicator = {
                'xyxy': (all_x1, all_y1, all_x2, all_y2),
                'confidence': max(ind.get('confidence', 1.0) for ind in intersecting_indicators),
                'merged_from': len(intersecting_indicators)
            }
            group['range_indicator'] = merged_indicator
        # else: no intersection, will fall through to proximity-based
        
        # If no intersecting indicator found, use proximity-based assignment
        if group['range_indicator'] is None:
            x1, y1, x2, y2 = box['xyxy']
            box_center_x = (x1 + x2) / 2.0
            box_center_y = (y1 + y2) / 2.0
            box_width = x2 - x1
            box_height = y2 - y1
            
            best_indicator = None
            min_distance = float('inf')
            
            for indicator in range_indicators:
                ind_x1, ind_y1, ind_x2, ind_y2 = indicator['xyxy']
                ind_center_x = (ind_x1 + ind_x2) / 2.0
                ind_center_y = (ind_y1 + ind_y2) / 2.0
                
                # Calculate distance based on orientation
                if orientation == 'vertical':
                    # For vertical: check x-alignment and vertical proximity
                    x_alignment = abs(ind_center_x - box_center_x)
                    if x_alignment < (box_width * 0.5):  # Reasonable x-alignment
                        y_distance = abs(ind_center_y - box_center_y)
                        if y_distance < min_distance:
                            min_distance = y_distance
                            best_indicator = indicator
                else:  # horizontal
                    # For horizontal: check y-alignment and horizontal proximity
                    y_alignment = abs(ind_center_y - box_center_y)
                    if y_alignment < (box_height * 0.5):  # Reasonable y-alignment
                        x_distance = abs(ind_center_x - box_center_x)
                        if x_distance < min_distance:
                            min_distance = x_distance
                            best_indicator = indicator
            
            if best_indicator is not None:
                group['range_indicator'] = best_indicator

        # Do the same for median lines
        for median in median_lines:
            if compute_aabb_intersection(box['xyxy'], median['xyxy']):
                group['median_line'] = median
                break  # Use first intersecting median
        
        # If no intersecting median found, use proximity-based assignment
        if group['median_line'] is None:
            x1, y1, x2, y2 = box['xyxy']
            box_center_x = (x1 + x2) / 2.0
            box_center_y = (y1 + y2) / 2.0
            box_width = x2 - x1
            box_height = y2 - y1
            
            best_median = None
            min_distance = float('inf')
            
            for median in median_lines:
                med_x1, med_y1, med_x2, med_y2 = median['xyxy']
                med_center_x = (med_x1 + med_x2) / 2.0
                med_center_y = (med_y1 + med_y2) / 2.0
                
                # Calculate distance based on orientation
                if orientation == 'vertical':
                    # For vertical: check x-alignment and proximity to box center
                    x_alignment = abs(med_center_x - box_center_x)
                    if x_alignment < (box_width * 0.5):  # Reasonable x-alignment
                        y_distance = abs(med_center_y - box_center_y)
                        if y_distance < min_distance:
                            min_distance = y_distance
                            best_median = median
                else:  # horizontal
                    # For horizontal: check y-alignment and proximity to box center
                    y_alignment = abs(med_center_y - box_center_y)
                    if y_alignment < (box_height * 0.5):  # Reasonable y-alignment
                        x_distance = abs(med_center_x - box_center_x)
                        if x_distance < min_distance:
                            min_distance = x_distance
                            best_median = median
            
            if best_median is not None:
                group['median_line'] = best_median
        
        # Stage 2: Coordinate-aligned grouping for outliers
        # Vertical: match x-coordinates, Horizontal: match y-coordinates
        if orientation == 'vertical':
            threshold = box_width * 0.4  # 40% box width tolerance
            for outlier in outliers:
                outlier_cx = (outlier['xyxy'][0] + outlier['xyxy'][2]) / 2.0
                if abs(outlier_cx - box_center_x) < threshold:
                    group['outliers'].append(outlier)
        else:  # horizontal
            threshold = box_height * 0.4
            for outlier in outliers:
                outlier_cy = (outlier['xyxy'][1] + outlier['xyxy'][3]) / 2.0
                if abs(outlier_cy - box_center_y) < threshold:
                    group['outliers'].append(outlier)
        
        # Stage 3: Coordinate-aligned tick label assignment
        if orientation == 'vertical':
            threshold = box_width * 0.5
            for label in tick_labels:
                label_cx = (label['xyxy'][0] + label['xyxy'][2]) / 2.0
                if abs(label_cx - box_center_x) < threshold:
                    group['tick_label'] = label
                    break
        else:  # horizontal
            threshold = box_height * 0.5
            for label in tick_labels:
                label_cy = (label['xyxy'][1] + label['xyxy'][3]) / 2.0
                if abs(label_cy - box_center_y) < threshold:
                    group['tick_label'] = label
                    break
        
        grouped.append(group)
    
    return grouped
