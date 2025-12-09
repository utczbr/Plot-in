"""
Centralized geometry utilities for chart analysis.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def calculate_pixel_distance(
    coord1: Union[float, Tuple[float, float]], 
    coord2: Union[float, Tuple[float, float]]
) -> float:
    """
    Calculate Euclidean distance between two coordinates.
    Supports both 1D (float) and 2D (tuple) coordinates.
    """
    if isinstance(coord1, (int, float)) and isinstance(coord2, (int, float)):
        return abs(coord1 - coord2)
    
    c1 = np.array(coord1)
    c2 = np.array(coord2)
    return float(np.linalg.norm(c1 - c2))

def compute_aabb_intersection(
    box1_xyxy: Tuple[float, float, float, float], 
    box2_xyxy: Tuple[float, float, float, float]
) -> bool:
    """
    Axis-Aligned Bounding Box intersection test.
    Returns True if boxes overlap (shared area > 0).
    """
    x1_min, y1_min, x1_max, y1_max = box1_xyxy
    x2_min, y2_min, x2_max, y2_max = box2_xyxy
    
    # Separating axis theorem: no intersection if separated on any axis
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False
    
    return True

def get_center(bbox: List[float]) -> Tuple[float, float]:
    """Get center (x, y) of a bounding box [x1, y1, x2, y2]."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

def find_closest_element(
    target: Dict, 
    candidates: List[Dict], 
    orientation: str = 'vertical',
    threshold_multiplier: float = 3.0
) -> Optional[Dict]:
    """
    Find the closest candidate element to the target element.
    
    Args:
        target: Dict with 'xyxy' key.
        candidates: List of Dicts with 'xyxy' key.
        orientation: 'vertical' or 'horizontal'.
        threshold_multiplier: Multiplier for target width/height to set max distance threshold.
        
    Returns:
        Closest element Dict or None.
    """
    if not candidates:
        return None

    target_center = get_center(target['xyxy'])
    target_w = target['xyxy'][2] - target['xyxy'][0]
    target_h = target['xyxy'][3] - target['xyxy'][1]
    
    closest_element = None
    min_dist = float('inf')

    for element in candidates:
        el_center = get_center(element['xyxy'])
        
        if orientation == 'vertical':
            # For vertical association (e.g., bar to label below), 
            # we might prioritize horizontal alignment but calculate full distance
            dist = calculate_pixel_distance(target_center, el_center)
        else:
            dist = calculate_pixel_distance(target_center, el_center)
        
        if dist < min_dist:
            min_dist = dist
            closest_element = element
    
    # Threshold check
    # Use the relevant dimension based on orientation or just max dimension
    ref_dim = target_w if orientation == 'vertical' else target_h
    # Fallback if dimension is 0 (point)
    if ref_dim == 0: 
        ref_dim = 10.0 
        
    if min_dist < ref_dim * threshold_multiplier:
        return closest_element
        
    return None
