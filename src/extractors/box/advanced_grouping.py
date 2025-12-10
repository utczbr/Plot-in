"""
Advanced element grouping utilities for box plot analysis.

Implements:
- Color consistency checking for element grouping
- Alignment-based scoring (beyond Euclidean distance)
- Global layout pattern detection
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def extract_dominant_color(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    sample_ratio: float = 0.3
) -> Optional[Tuple[int, int, int]]:
    """
    Extract the dominant color from a bounding box region.
    
    Uses center sampling to avoid edge artifacts.
    
    Args:
        image: BGR image (numpy array)
        bbox: Bounding box (x1, y1, x2, y2)
        sample_ratio: Ratio of center region to sample (0.3 = 30% center)
        
    Returns:
        Dominant color as (R, G, B) tuple or None if extraction fails
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure valid bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Sample center region to avoid edges
        width = x2 - x1
        height = y2 - y1
        
        margin_x = int(width * (1 - sample_ratio) / 2)
        margin_y = int(height * (1 - sample_ratio) / 2)
        
        cx1 = x1 + margin_x
        cx2 = x2 - margin_x
        cy1 = y1 + margin_y
        cy2 = y2 - margin_y
        
        if cx2 <= cx1 or cy2 <= cy1:
            # Region too small, use full box
            region = image[y1:y2, x1:x2]
        else:
            region = image[cy1:cy2, cx1:cx2]
        
        if region.size == 0:
            return None
        
        # Get median color (robust to outliers)
        if len(region.shape) == 3 and region.shape[2] >= 3:
            # BGR image
            median_b = int(np.median(region[:, :, 0]))
            median_g = int(np.median(region[:, :, 1]))
            median_r = int(np.median(region[:, :, 2]))
            return (median_r, median_g, median_b)
        else:
            # Grayscale
            median_gray = int(np.median(region))
            return (median_gray, median_gray, median_gray)
            
    except Exception as e:
        logger.warning(f"Color extraction failed: {e}")
        return None


def compute_color_similarity(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    method: str = 'euclidean'
) -> float:
    """
    Compute similarity between two colors.
    
    Args:
        color1: RGB tuple
        color2: RGB tuple
        method: 'euclidean' or 'hsv'
        
    Returns:
        Similarity score in [0, 1] where 1 is identical
    """
    if method == 'euclidean':
        # Euclidean distance in RGB space, normalized
        diff = np.array(color1) - np.array(color2)
        distance = np.sqrt(np.sum(diff ** 2))
        # Max distance is sqrt(3 * 255^2) ≈ 441.7
        max_distance = 441.7
        return 1.0 - (distance / max_distance)
    else:
        # Simple Euclidean for now
        return compute_color_similarity(color1, color2, 'euclidean')


def color_consistency_score(
    image: np.ndarray,
    element_bbox: Tuple[float, float, float, float],
    candidate_bboxes: List[Tuple[float, float, float, float]]
) -> List[float]:
    """
    Score candidates based on color consistency with the element.
    
    Args:
        image: BGR image
        element_bbox: Reference element bounding box
        candidate_bboxes: List of candidate bounding boxes
        
    Returns:
        List of similarity scores [0, 1] for each candidate
    """
    element_color = extract_dominant_color(image, element_bbox)
    
    if element_color is None:
        return [0.5] * len(candidate_bboxes)  # Neutral score
    
    scores = []
    for bbox in candidate_bboxes:
        candidate_color = extract_dominant_color(image, bbox)
        if candidate_color is None:
            scores.append(0.5)
        else:
            scores.append(compute_color_similarity(element_color, candidate_color))
    
    return scores


def compute_alignment_score(
    box_bbox: Tuple[float, float, float, float],
    candidate_bbox: Tuple[float, float, float, float],
    orientation: str = 'vertical'
) -> float:
    """
    Compute alignment score between a box and a candidate element.
    
    For vertical box plots:
    - Whiskers should be x-aligned with box center
    - Median lines should be fully contained in box x-range
    
    For horizontal box plots:
    - Whiskers should be y-aligned with box center
    - Median lines should be fully contained in box y-range
    
    Args:
        box_bbox: Box bounding box (x1, y1, x2, y2)
        candidate_bbox: Candidate element bounding box
        orientation: 'vertical' or 'horizontal'
        
    Returns:
        Alignment score in [0, 1] where 1 is perfectly aligned
    """
    bx1, by1, bx2, by2 = box_bbox
    cx1, cy1, cx2, cy2 = candidate_bbox
    
    box_center_x = (bx1 + bx2) / 2
    box_center_y = (by1 + by2) / 2
    box_width = bx2 - bx1
    box_height = by2 - by1
    
    candidate_center_x = (cx1 + cx2) / 2
    candidate_center_y = (cy1 + cy2) / 2
    
    if orientation == 'vertical':
        # X-alignment is important
        x_offset = abs(candidate_center_x - box_center_x)
        
        # Perfect alignment: offset = 0
        # Poor alignment: offset > box_width
        if box_width > 0:
            alignment_ratio = 1.0 - min(1.0, x_offset / box_width)
        else:
            alignment_ratio = 0.5
        
        # Bonus for containment in x-range
        if cx1 >= bx1 and cx2 <= bx2:
            alignment_ratio = min(1.0, alignment_ratio + 0.2)
        
        return alignment_ratio
    else:
        # Y-alignment is important for horizontal
        y_offset = abs(candidate_center_y - box_center_y)
        
        if box_height > 0:
            alignment_ratio = 1.0 - min(1.0, y_offset / box_height)
        else:
            alignment_ratio = 0.5
        
        # Bonus for containment in y-range
        if cy1 >= by1 and cy2 <= by2:
            alignment_ratio = min(1.0, alignment_ratio + 0.2)
        
        return alignment_ratio


def compute_composite_grouping_score(
    image: np.ndarray,
    box_bbox: Tuple[float, float, float, float],
    candidate_bbox: Tuple[float, float, float, float],
    orientation: str = 'vertical',
    weights: Dict[str, float] = None
) -> float:
    """
    Compute composite grouping score using multiple factors.
    
    Combines:
    - Euclidean distance (closer = better)
    - Alignment score (aligned = better)
    - Color consistency (same color = better)
    
    Args:
        image: BGR image
        box_bbox: Box bounding box
        candidate_bbox: Candidate element bounding box
        orientation: 'vertical' or 'horizontal'
        weights: Optional dict with 'distance', 'alignment', 'color' weights
        
    Returns:
        Composite score in [0, 1] where 1 is best match
    """
    if weights is None:
        weights = {'distance': 0.4, 'alignment': 0.4, 'color': 0.2}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Distance score
    bx1, by1, bx2, by2 = box_bbox
    cx1, cy1, cx2, cy2 = candidate_bbox
    
    box_center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
    candidate_center = np.array([(cx1 + cx2) / 2, (cy1 + cy2) / 2])
    
    distance = np.linalg.norm(box_center - candidate_center)
    
    # Normalize distance by image diagonal (approximate)
    img_diagonal = max(image.shape[0], image.shape[1]) if image is not None else 1000
    distance_score = 1.0 - min(1.0, distance / (0.3 * img_diagonal))
    
    # Alignment score
    alignment_score = compute_alignment_score(box_bbox, candidate_bbox, orientation)
    
    # Color score
    if image is not None:
        color_scores = color_consistency_score(image, box_bbox, [candidate_bbox])
        color_score = color_scores[0]
    else:
        color_score = 0.5
    
    # Composite
    composite = (
        weights['distance'] * distance_score +
        weights['alignment'] * alignment_score +
        weights['color'] * color_score
    )
    
    return composite


def detect_grid_layout(
    boxes: List[Dict],
    orientation: str = 'vertical',
    tolerance_ratio: float = 0.1
) -> Dict[str, any]:
    """
    Detect if boxes are arranged in a regular grid pattern.
    
    Useful for grouped box plots where boxes are evenly spaced.
    
    Args:
        boxes: List of box dictionaries with 'xyxy' keys
        orientation: 'vertical' or 'horizontal'
        tolerance_ratio: Tolerance for spacing consistency
        
    Returns:
        Dict with 'is_grid', 'spacing', 'num_groups', 'groups'
    """
    if len(boxes) < 2:
        return {'is_grid': False, 'spacing': None, 'num_groups': 1, 'groups': [boxes]}
    
    # Extract positions along the grouping axis
    positions = []
    for box in boxes:
        x1, y1, x2, y2 = box['xyxy']
        if orientation == 'vertical':
            # For vertical, group by x position
            positions.append((x1 + x2) / 2)
        else:
            # For horizontal, group by y position
            positions.append((y1 + y2) / 2)
    
    positions = np.array(positions)
    sort_idx = np.argsort(positions)
    sorted_positions = positions[sort_idx]
    
    # Compute spacing between adjacent boxes
    spacings = np.diff(sorted_positions)
    
    if len(spacings) == 0:
        return {'is_grid': False, 'spacing': None, 'num_groups': 1, 'groups': [boxes]}
    
    median_spacing = np.median(spacings)
    
    if median_spacing < 1e-6:
        return {'is_grid': False, 'spacing': 0, 'num_groups': 1, 'groups': [boxes]}
    
    # Check consistency
    spacing_variation = np.std(spacings) / median_spacing
    is_grid = spacing_variation < tolerance_ratio
    
    # Group boxes by position clusters
    # Find large gaps (> 2x median spacing)
    gap_threshold = 2.0 * median_spacing
    gap_indices = np.where(spacings > gap_threshold)[0]
    
    # Split into groups
    groups = []
    start_idx = 0
    for gap_idx in gap_indices:
        group = [boxes[sort_idx[i]] for i in range(start_idx, gap_idx + 1)]
        groups.append(group)
        start_idx = gap_idx + 1
    
    # Add last group
    last_group = [boxes[sort_idx[i]] for i in range(start_idx, len(boxes))]
    if last_group:
        groups.append(last_group)
    
    return {
        'is_grid': is_grid,
        'spacing': float(median_spacing),
        'spacing_variation': float(spacing_variation),
        'num_groups': len(groups),
        'groups': groups
    }


class EnhancedBoxGrouper:
    """
    Enhanced grouper that combines spatial, color, and alignment signals.
    """
    
    def __init__(
        self,
        use_color: bool = True,
        use_alignment: bool = True,
        distance_weight: float = 0.4,
        alignment_weight: float = 0.4,
        color_weight: float = 0.2
    ):
        self.use_color = use_color
        self.use_alignment = use_alignment
        self.weights = {
            'distance': distance_weight,
            'alignment': alignment_weight,
            'color': color_weight if use_color else 0.0
        }
        self.logger = logging.getLogger(__name__)
    
    def find_best_match(
        self,
        image: np.ndarray,
        box: Dict,
        candidates: List[Dict],
        orientation: str = 'vertical',
        threshold: float = 0.3
    ) -> Optional[Dict]:
        """
        Find the best matching candidate for a box element.
        
        Args:
            image: BGR image
            box: Box dictionary with 'xyxy'
            candidates: List of candidate dictionaries with 'xyxy'
            orientation: 'vertical' or 'horizontal'
            threshold: Minimum score to accept a match
            
        Returns:
            Best matching candidate or None if no match above threshold
        """
        if not candidates:
            return None
        
        box_bbox = box['xyxy']
        
        scores = []
        for candidate in candidates:
            score = compute_composite_grouping_score(
                image,
                box_bbox,
                candidate['xyxy'],
                orientation,
                self.weights
            )
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        if best_score >= threshold:
            self.logger.debug(
                f"Best match found with score {best_score:.3f} "
                f"(threshold={threshold})"
            )
            return candidates[best_idx]
        else:
            self.logger.debug(
                f"No match above threshold: best={best_score:.3f}, "
                f"threshold={threshold}"
            )
            return None
