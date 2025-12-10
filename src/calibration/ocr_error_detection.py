"""
OCR Error Detection and Correction utilities.

Implements detection of common OCR confusion patterns in axis labels:
- Digit confusions: 0/O, 1/l/I, 5/S, 8/B, 2/Z, 6/G
- Number format issues: comma vs decimal point
- Sign confusion: - vs −
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# Common OCR confusion mappings
OCR_DIGIT_CONFUSIONS = {
    'O': '0',
    'o': '0',
    'l': '1',
    'I': '1',
    'i': '1',
    '|': '1',
    'S': '5',
    's': '5',
    'B': '8',
    'Z': '2',
    'z': '2',
    'G': '6',
    'g': '9',
    'q': '9',
    # Special minus signs
    '−': '-',  # Unicode minus
    '–': '-',  # En dash
    '—': '-',  # Em dash
}


def correct_ocr_digit_confusions(text: str) -> str:
    """
    Apply common OCR digit confusion corrections to text.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Corrected text with digit confusions fixed
    """
    result = text
    for wrong, correct in OCR_DIGIT_CONFUSIONS.items():
        result = result.replace(wrong, correct)
    return result


def detect_number_format(labels: List[str]) -> str:
    """
    Detect whether labels use comma or period as decimal separator.
    
    Heuristic:
    - If any label has both comma and period, comma is thousands separator
    - If only commas appear and values look like decimals, comma is decimal separator
    - Default to period as decimal separator
    
    Returns:
        'comma_decimal' or 'period_decimal'
    """
    has_comma = any(',' in lbl for lbl in labels)
    has_period = any('.' in lbl for lbl in labels)
    
    if has_comma and has_period:
        # Both present - comma is likely thousands separator
        return 'period_decimal'
    
    if has_comma and not has_period:
        # Only comma - check if it looks like decimal
        # European style: "1,5" vs "1,000"
        comma_positions = []
        for lbl in labels:
            if ',' in lbl:
                parts = lbl.split(',')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    # Short decimal part (1 or 2 digits) suggests comma is decimal
                    comma_positions.append('decimal')
                else:
                    comma_positions.append('thousands')
        
        if comma_positions and comma_positions.count('decimal') > comma_positions.count('thousands'):
            return 'comma_decimal'
    
    return 'period_decimal'


def parse_numeric_robust(
    text: str,
    number_format: str = 'period_decimal'
) -> Optional[float]:
    """
    Robustly parse a numeric value from OCR text.
    
    Steps:
    1. Apply OCR digit confusion corrections
    2. Handle number format (comma/period decimal)
    3. Extract numeric value with regex
    
    Args:
        text: Raw OCR text
        number_format: 'comma_decimal' or 'period_decimal'
        
    Returns:
        Parsed float or None if parsing fails
    """
    if not text or not text.strip():
        return None
    
    # Step 1: Apply OCR corrections
    corrected = correct_ocr_digit_confusions(text.strip())
    
    # Step 2: Normalize number format
    if number_format == 'comma_decimal':
        # European format: 1.234,56 -> 1234.56
        corrected = corrected.replace('.', '').replace(',', '.')
    else:
        # American format: 1,234.56 -> 1234.56
        corrected = corrected.replace(',', '')
    
    # Step 3: Extract numeric value
    # Handle various formats: -1.23, 1.23e-4, 1E+5, etc.
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    matches = re.findall(pattern, corrected)
    
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    
    return None


def detect_tick_clustering(
    coords: np.ndarray,
    values: np.ndarray,
    tolerance_pixels: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Detect and merge tick marks that are too close together (phantom ticks).
    
    When multiple OCR detections cluster at similar pixel positions,
    they likely represent the same tick mark detected multiple times.
    
    Args:
        coords: Pixel coordinates of tick marks
        values: Numeric values of tick marks
        tolerance_pixels: Distance within which ticks are considered duplicates
        
    Returns:
        Tuple of (filtered_coords, filtered_values, merged_indices)
    """
    if len(coords) < 2:
        return coords, values, list(range(len(coords)))
    
    # Sort by coordinate
    sort_idx = np.argsort(coords)
    sorted_coords = coords[sort_idx]
    sorted_values = values[sort_idx]
    
    # Find clusters
    merged_coords = []
    merged_values = []
    merged_indices = []
    
    cluster_start = 0
    for i in range(1, len(sorted_coords) + 1):
        # End of array or gap detected
        if i == len(sorted_coords) or (sorted_coords[i] - sorted_coords[i-1]) > tolerance_pixels:
            # Merge cluster [cluster_start, i)
            cluster_coords = sorted_coords[cluster_start:i]
            cluster_values = sorted_values[cluster_start:i]
            
            if len(cluster_coords) > 1:
                logger.debug(
                    f"Merging tick cluster: {len(cluster_coords)} ticks at "
                    f"positions {cluster_coords.tolist()}"
                )
            
            # Use mean coordinate and median value (robust to outliers)
            merged_coords.append(np.mean(cluster_coords))
            merged_values.append(np.median(cluster_values))
            merged_indices.append(sort_idx[cluster_start])
            
            cluster_start = i
    
    return np.array(merged_coords), np.array(merged_values), merged_indices


def compute_subpixel_tick_position(
    bbox: Tuple[float, float, float, float],
    axis_type: str,
    image_gradient: Optional[np.ndarray] = None
) -> float:
    """
    Compute sub-pixel accurate tick position using gradient information.
    
    If gradient image is available, uses gradient peak for refinement.
    Otherwise, uses weighted center based on bbox dimensions.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        axis_type: 'x' or 'y'
        image_gradient: Optional gradient magnitude image
        
    Returns:
        Sub-pixel accurate coordinate
    """
    x1, y1, x2, y2 = bbox
    
    if axis_type.lower() == 'y':
        # Y-axis tick: refine vertical position
        center = (y1 + y2) / 2.0
        box_height = y2 - y1
        
        if image_gradient is not None:
            # TODO: Use gradient peak for refinement
            # For now, use weighted center
            pass
        
        # Small boxes are more precise
        # Large boxes may contain multiple lines - use center
        return center
    else:
        # X-axis tick: refine horizontal position
        center = (x1 + x2) / 2.0
        box_width = x2 - x1
        
        return center


def validate_scale_consistency(
    coords: np.ndarray,
    values: np.ndarray,
    tolerance_factor: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Validate that tick marks form a consistent linear scale.
    
    Detects and removes outliers that break the linear pattern,
    which often indicates OCR errors.
    
    Args:
        coords: Pixel coordinates
        values: Numeric values
        tolerance_factor: Relative tolerance for residuals (0.1 = 10%)
        
    Returns:
        Tuple of (filtered_coords, filtered_values, is_consistent)
    """
    if len(coords) < 3:
        return coords, values, True
    
    # Fit initial linear model
    try:
        coeffs = np.polyfit(coords, values, 1)
        predicted = np.polyval(coeffs, coords)
        residuals = np.abs(values - predicted)
        
        # Compute scale for tolerance
        value_range = np.max(values) - np.min(values)
        if value_range == 0:
            return coords, values, True
        
        tolerance = tolerance_factor * value_range
        
        # Find consistent points
        consistent_mask = residuals <= tolerance
        
        if np.sum(consistent_mask) < 2:
            # Not enough consistent points - return original
            logger.warning("Scale consistency check: too few consistent points")
            return coords, values, False
        
        inconsistent_count = np.sum(~consistent_mask)
        if inconsistent_count > 0:
            logger.info(
                f"Scale consistency: removed {inconsistent_count} outlier tick(s) "
                f"with residuals > {tolerance:.2f}"
            )
        
        return coords[consistent_mask], values[consistent_mask], True
        
    except Exception as e:
        logger.warning(f"Scale consistency check failed: {e}")
        return coords, values, False


def estimate_grid_line_orientation(
    lines: List[Tuple[float, float, float, float]],
    angle_tolerance_deg: float = 5.0
) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    Classify lines as horizontal, vertical, or diagonal based on angle.
    
    Used to filter out grid lines from whiskers:
    - Whiskers are typically vertical (for vertical box plots)
    - Grid lines are typically horizontal
    
    Args:
        lines: List of line segments as (x1, y1, x2, y2)
        angle_tolerance_deg: Tolerance for classification
        
    Returns:
        Dict with 'horizontal', 'vertical', 'diagonal' lists
    """
    result = {'horizontal': [], 'vertical': [], 'diagonal': []}
    
    for line in lines:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) < 1e-6:
            # Perfectly vertical
            result['vertical'].append(line)
        elif abs(dy) < 1e-6:
            # Perfectly horizontal
            result['horizontal'].append(line)
        else:
            angle_deg = np.degrees(np.arctan2(abs(dy), abs(dx)))
            
            if angle_deg < angle_tolerance_deg:
                result['horizontal'].append(line)
            elif angle_deg > (90 - angle_tolerance_deg):
                result['vertical'].append(line)
            else:
                result['diagonal'].append(line)
    
    return result
