"""
Morphological refinement utilities for whisker detection.

Implements multi-pass morphological operations to handle:
- Fragmented whiskers (broken by compression artifacts)
- Grid line interference
- Antialiasing artifacts
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Optional cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available, morphological refinement disabled")


def create_line_kernel(length: int, angle: float) -> np.ndarray:
    """
    Create a line-shaped morphological kernel at given angle.
    
    Args:
        length: Kernel length in pixels (should be odd)
        angle: Angle in degrees (0 = horizontal, 90 = vertical)
        
    Returns:
        Binary kernel array
    """
    if length % 2 == 0:
        length += 1
    
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    
    # Convert angle to radians
    rad = np.radians(angle)
    
    # Draw line through center
    for i in range(-center, center + 1):
        x = int(center + i * np.cos(rad))
        y = int(center - i * np.sin(rad))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    
    return kernel


def enhance_whisker_lines(
    image: np.ndarray,
    orientation: str = 'vertical',
    line_thickness: int = 3
) -> np.ndarray:
    """
    Enhance whisker lines using directional morphological operations.
    
    For vertical box plots, enhances vertical lines while suppressing horizontal.
    
    Args:
        image: Grayscale or BGR image
        orientation: 'vertical' or 'horizontal'
        line_thickness: Expected line thickness
        
    Returns:
        Enhanced binary image
    """
    if not CV2_AVAILABLE:
        logger.warning("OpenCV not available, returning original image")
        return image
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Invert if needed (assume dark lines on light background)
    if np.mean(gray) > 128:
        gray = 255 - gray
    
    # Binary threshold with Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create directional kernels
    if orientation == 'vertical':
        # Vertical line enhancement
        enhance_kernel = create_line_kernel(line_thickness * 3, 90)
        suppress_kernel = create_line_kernel(line_thickness * 5, 0)
    else:
        # Horizontal line enhancement
        enhance_kernel = create_line_kernel(line_thickness * 3, 0)
        suppress_kernel = create_line_kernel(line_thickness * 5, 90)
    
    # Morphological operations
    # 1. Close small gaps in whisker lines
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, enhance_kernel)
    
    # 2. Remove perpendicular lines (grid interference)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, suppress_kernel)
    
    # 3. Final cleanup
    result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    return result


def repair_fragmented_line(
    binary_image: np.ndarray,
    start_point: Tuple[int, int],
    direction: str = 'vertical',
    max_gap: int = 10
) -> List[Tuple[int, int]]:
    """
    Repair a fragmented line by bridging small gaps.
    
    Uses scanline search to find continuation of fragmented lines.
    
    Args:
        binary_image: Binary image with line segments
        start_point: Starting point (x, y)
        direction: 'vertical' or 'horizontal'
        max_gap: Maximum gap to bridge in pixels
        
    Returns:
        List of points forming the repaired line
    """
    if not CV2_AVAILABLE:
        return [start_point]
    
    h, w = binary_image.shape[:2]
    x, y = start_point
    points = [(x, y)]
    
    if direction == 'vertical':
        # Scan downward
        current_y = y
        while current_y < h - 1:
            current_y += 1
            
            # Look for line continuation
            found = False
            for offset in range(max_gap + 1):
                check_y = current_y + offset
                if check_y >= h:
                    break
                
                # Search left and right for line pixel
                for dx in range(-3, 4):
                    check_x = x + dx
                    if 0 <= check_x < w and binary_image[check_y, check_x] > 128:
                        points.append((check_x, check_y))
                        current_y = check_y
                        x = check_x
                        found = True
                        break
                
                if found:
                    break
            
            if not found:
                break
    else:
        # Scan rightward
        current_x = x
        while current_x < w - 1:
            current_x += 1
            
            found = False
            for offset in range(max_gap + 1):
                check_x = current_x + offset
                if check_x >= w:
                    break
                
                for dy in range(-3, 4):
                    check_y = y + dy
                    if 0 <= check_y < h and binary_image[check_y, check_x] > 128:
                        points.append((check_x, check_y))
                        current_x = check_x
                        y = check_y
                        found = True
                        break
                
                if found:
                    break
            
            if not found:
                break
    
    return points


def multipass_whisker_refinement(
    image: np.ndarray,
    box_bbox: Tuple[float, float, float, float],
    orientation: str = 'vertical',
    passes: int = 2
) -> Dict:
    """
    Multi-pass morphological refinement for whisker detection.
    
    Pass 1: Coarse enhancement with aggressive gap closing
    Pass 2: Fine refinement with directional filtering
    
    Args:
        image: Input image
        box_bbox: Box bounding box to focus search
        orientation: 'vertical' or 'horizontal'
        passes: Number of refinement passes
        
    Returns:
        Dict with 'whisker_low_pixel', 'whisker_high_pixel', 'confidence', 'enhanced_image'
    """
    if not CV2_AVAILABLE:
        return {
            'whisker_low_pixel': None,
            'whisker_high_pixel': None,
            'confidence': 0.0,
            'enhanced_image': None
        }
    
    x1, y1, x2, y2 = map(int, box_bbox)
    h, w = image.shape[:2]
    
    # Expand search region beyond box
    margin = int(max(y2 - y1, x2 - x1) * 0.5)
    
    if orientation == 'vertical':
        # Search above and below box
        search_y1 = max(0, y1 - margin)
        search_y2 = min(h, y2 + margin)
        search_x1 = max(0, x1 - 5)
        search_x2 = min(w, x2 + 5)
    else:
        # Search left and right of box
        search_x1 = max(0, x1 - margin)
        search_x2 = min(w, x2 + margin)
        search_y1 = max(0, y1 - 5)
        search_y2 = min(h, y2 + 5)
    
    # Extract region
    region = image[search_y1:search_y2, search_x1:search_x2]
    
    if region.size == 0:
        return {
            'whisker_low_pixel': None,
            'whisker_high_pixel': None,
            'confidence': 0.0,
            'enhanced_image': None
        }
    
    # Multi-pass enhancement
    enhanced = region.copy()
    for p in range(passes):
        line_thickness = 3 + p  # Increase kernel size each pass
        enhanced = enhance_whisker_lines(enhanced, orientation, line_thickness)
    
    # Find extreme points of lines
    if len(enhanced.shape) == 3:
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        enhanced_gray = enhanced
    
    # Find line pixels
    line_pixels = np.where(enhanced_gray > 128)
    
    if len(line_pixels[0]) == 0:
        return {
            'whisker_low_pixel': None,
            'whisker_high_pixel': None,
            'confidence': 0.0,
            'enhanced_image': enhanced
        }
    
    if orientation == 'vertical':
        # Find topmost and bottommost pixels for whiskers
        min_y = np.min(line_pixels[0]) + search_y1
        max_y = np.max(line_pixels[0]) + search_y1
        
        # Confidence based on line density
        line_density = len(line_pixels[0]) / float((search_y2 - search_y1) * (search_x2 - search_x1) + 1)
        confidence = min(1.0, line_density * 100)  # Rough normalization
        
        return {
            'whisker_high_pixel': float(min_y),  # Top = high value
            'whisker_low_pixel': float(max_y),   # Bottom = low value  
            'confidence': confidence,
            'enhanced_image': enhanced
        }
    else:
        # Find leftmost and rightmost pixels
        min_x = np.min(line_pixels[1]) + search_x1
        max_x = np.max(line_pixels[1]) + search_x1
        
        line_density = len(line_pixels[0]) / float((search_y2 - search_y1) * (search_x2 - search_x1) + 1)
        confidence = min(1.0, line_density * 100)
        
        return {
            'whisker_low_pixel': float(min_x),
            'whisker_high_pixel': float(max_x),
            'confidence': confidence,
            'enhanced_image': enhanced
        }


def suppress_grid_lines(
    image: np.ndarray,
    orientation: str = 'vertical',
    line_length_threshold: int = 50
) -> np.ndarray:
    """
    Suppress grid lines that interfere with whisker detection.
    
    Uses line length analysis to distinguish grid lines from whiskers:
    - Grid lines: span entire chart width/height
    - Whiskers: limited to box region
    
    Args:
        image: Binary or grayscale image
        orientation: 'vertical' or 'horizontal'
        line_length_threshold: Minimum length to be considered grid line
        
    Returns:
        Image with grid lines suppressed
    """
    if not CV2_AVAILABLE:
        return image
    
    # Convert if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    
    # Create mask for grid lines
    grid_mask = np.zeros_like(gray)
    
    # Use Hough lines to detect long straight lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=line_length_threshold,
        maxLineGap=10
    )
    
    if lines is None:
        return image
    
    # Filter lines by orientation and draw to mask
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if orientation == 'vertical':
            # Suppress horizontal grid lines
            if dx > dy * 3 and dx > line_length_threshold:
                cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 3)
        else:
            # Suppress vertical grid lines
            if dy > dx * 3 and dy > line_length_threshold:
                cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 3)
    
    # Remove grid lines from image
    result = gray.copy()
    result[grid_mask > 0] = 0 if np.mean(gray) > 128 else 255
    
    return result
