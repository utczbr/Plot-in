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
    assigned_indicators = set()
    assigned_medians = set()

    # Pass 1: Intersection-based grouping + outlier & tick coordinate alignment
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
        intersecting_indicators = [
            (idx, indicator)
            for idx, indicator in enumerate(range_indicators)
            if idx not in assigned_indicators and compute_aabb_intersection(box['xyxy'], indicator['xyxy'])
        ]

        if len(intersecting_indicators) == 1:
            idx, ind = intersecting_indicators[0]
            group['range_indicator'] = ind
            assigned_indicators.add(idx)
        elif len(intersecting_indicators) >= 2:
            for idx, _ in intersecting_indicators:
                assigned_indicators.add(idx)
            inds = [ind for _, ind in intersecting_indicators]
            all_x1 = min(ind['xyxy'][0] for ind in inds)
            all_y1 = min(ind['xyxy'][1] for ind in inds)
            all_x2 = max(ind['xyxy'][2] for ind in inds)
            all_y2 = max(ind['xyxy'][3] for ind in inds)

            merged_indicator = {
                'xyxy': (all_x1, all_y1, all_x2, all_y2),
                'confidence': max(ind.get('confidence', 1.0) for ind in inds),
                'merged_from': len(inds)
            }
            group['range_indicator'] = merged_indicator

        for med_idx, median in enumerate(median_lines):
            if med_idx not in assigned_medians and compute_aabb_intersection(box['xyxy'], median['xyxy']):
                group['median_line'] = median
                assigned_medians.add(med_idx)
                break

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

    # Pass 2: Greedy min-distance exclusive proximity fallback for unassigned indicators and medians
    ind_candidates = []
    med_candidates = []
    for g_idx, group in enumerate(grouped):
        box = group['box']
        x1, y1, x2, y2 = box['xyxy']
        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0
        box_width = x2 - x1
        box_height = y2 - y1

        if group['range_indicator'] is None:
            for idx, indicator in enumerate(range_indicators):
                if idx in assigned_indicators:
                    continue
                ind_x1, ind_y1, ind_x2, ind_y2 = indicator['xyxy']
                ind_center_x = (ind_x1 + ind_x2) / 2.0
                ind_center_y = (ind_y1 + ind_y2) / 2.0
                if orientation == 'vertical':
                    if abs(ind_center_x - box_center_x) < (box_width * 0.5):
                        ind_candidates.append((abs(ind_center_y - box_center_y), g_idx, idx))
                else:
                    if abs(ind_center_y - box_center_y) < (box_height * 0.5):
                        ind_candidates.append((abs(ind_center_x - box_center_x), g_idx, idx))

        if group['median_line'] is None:
            for idx, median in enumerate(median_lines):
                if idx in assigned_medians:
                    continue
                med_x1, med_y1, med_x2, med_y2 = median['xyxy']
                med_center_x = (med_x1 + med_x2) / 2.0
                med_center_y = (med_y1 + med_y2) / 2.0
                if orientation == 'vertical':
                    if abs(med_center_x - box_center_x) < (box_width * 0.5):
                        med_candidates.append((abs(med_center_y - box_center_y), g_idx, idx))
                else:
                    if abs(med_center_y - box_center_y) < (box_height * 0.5):
                        med_candidates.append((abs(med_center_x - box_center_x), g_idx, idx))

    for _, g_idx, idx in sorted(ind_candidates):
        if grouped[g_idx]['range_indicator'] is None and idx not in assigned_indicators:
            grouped[g_idx]['range_indicator'] = range_indicators[idx]
            assigned_indicators.add(idx)

    for _, g_idx, idx in sorted(med_candidates):
        if grouped[g_idx]['median_line'] is None and idx not in assigned_medians:
            grouped[g_idx]['median_line'] = median_lines[idx]
            assigned_medians.add(idx)

    return grouped
