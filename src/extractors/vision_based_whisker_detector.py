"""
Computer vision-based whisker detection as fallback when object detector misses them.
Uses line detection and morphological operations.
"""
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import cv2
import logging


class VisionBasedWhiskerDetector:
    """
    Computer vision-based whisker detection as fallback when object detector misses them.
    Uses line detection and morphological operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_whiskers_in_region(
        self,
        img: np.ndarray,
        box_bbox: Tuple[int, int, int, int],
        orientation: str,
        scale_model: Callable
    ) -> Optional[Tuple[float, float]]:
        """
        Detect whiskers using computer vision in the expanded region around the box.
        
        Approach:
        1. Define search regions above/below (vertical) or left/right (horizontal) of box
        2. Apply edge detection and line detection (Hough transform)
        3. Identify vertical lines (vertical charts) or horizontal lines (horizontal charts)
        4. Filter by alignment with box center
        5. Find line extents and convert to data values
        
        Returns:
            (whisker_low_value, whisker_high_value) if detected, None otherwise
        """
        x1, y1, x2, y2 = box_bbox
        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Define search regions
        if orientation == 'vertical':
            # Search above and below the box
            search_margin = int(box_height * 2.0)  # Generous search region
            search_width = int(box_width * 1.5)
            
            # Top search region (for high whisker)
            top_region_x1 = max(0, int(box_center_x - search_width / 2))
            top_region_x2 = min(gray.shape[1], int(box_center_x + search_width / 2))
            top_region_y1 = max(0, int(y1 - search_margin))
            top_region_y2 = int(y1)
            
            # Bottom search region (for low whisker)
            bottom_region_x1 = max(0, int(box_center_x - search_width / 2))
            bottom_region_x2 = min(gray.shape[1], int(box_center_x + search_width / 2))
            bottom_region_y1 = int(y2)
            bottom_region_y2 = min(gray.shape[0], int(y2 + search_margin))
            
            # Extract regions
            top_region = gray[top_region_y1:top_region_y2, top_region_x1:top_region_x2]
            bottom_region = gray[bottom_region_y1:bottom_region_y2, bottom_region_x1:bottom_region_x2]
            
            # Detect lines using Hough transform
            high_whisker_pixel = self._detect_whisker_extent_in_region(
                top_region, 'vertical', 'high', (top_region_x1, top_region_y1)
            )
            low_whisker_pixel = self._detect_whisker_extent_in_region(
                bottom_region, 'vertical', 'low', (bottom_region_x1, bottom_region_y1)
            )
            
        else:  # horizontal
            # Search left and right of the box
            search_margin = int(box_width * 2.0)
            search_height = int(box_height * 1.5)
            
            # Left search region (for low whisker)
            left_region_x1 = max(0, int(x1 - search_margin))
            left_region_x2 = int(x1)
            left_region_y1 = max(0, int(box_center_y - search_height / 2))
            left_region_y2 = min(gray.shape[0], int(box_center_y + search_height / 2))
            
            # Right search region (for high whisker)
            right_region_x1 = int(x2)
            right_region_x2 = min(gray.shape[1], int(x2 + search_margin))
            right_region_y1 = max(0, int(box_center_y - search_height / 2))
            right_region_y2 = min(gray.shape[0], int(box_center_y + search_height / 2))
            
            left_region = gray[left_region_y1:left_region_y2, left_region_x1:left_region_x2]
            right_region = gray[right_region_y1:right_region_y2, right_region_x1:right_region_x2]
            
            low_whisker_pixel = self._detect_whisker_extent_in_region(
                left_region, 'horizontal', 'low', (left_region_x1, left_region_y1)
            )
            high_whisker_pixel = self._detect_whisker_extent_in_region(
                right_region, 'horizontal', 'high', (right_region_x1, right_region_y1)
            )
        
        # Convert pixels to values using scale_model
        if low_whisker_pixel is not None and high_whisker_pixel is not None:
            try:
                whisker_low_value = float(scale_model(low_whisker_pixel))
                whisker_high_value = float(scale_model(high_whisker_pixel))
                
                self.logger.info(
                    f"Vision-based whisker detection successful: "
                    f"low={whisker_low_value:.2f} (pixel={low_whisker_pixel:.1f}), "
                    f"high={whisker_high_value:.2f} (pixel={high_whisker_pixel:.1f})"
                )
                
                return whisker_low_value, whisker_high_value
            except Exception as e:
                self.logger.warning(f"Failed to convert whisker pixels to values: {e}")
                return None
        else:
            self.logger.warning("Vision-based whisker detection failed to find lines")
            return None
    
    def _detect_whisker_extent_in_region(
        self,
        region: np.ndarray,
        orientation: str,
        whisker_type: str,  # 'low' or 'high'
        region_offset: Tuple[int, int]
    ) -> Optional[float]:
        """
        Detect whisker extent in a specific search region using line detection.
        
        Returns:
            Pixel coordinate of whisker extent in original image coordinates
        """
        if region.size == 0:
            return None
        
        # Apply edge detection
        edges = cv2.Canny(region, 50, 150, apertureSize=3)
        
        # Apply Hough line transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=10,
            minLineLength=5,
            maxLineGap=3
        )
        
        if lines is None:
            return None
        
        # Filter lines by orientation
        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            if x2 - x1 != 0:
                angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            else:
                angle = 90  # Vertical line
            
            if orientation == 'vertical':
                # For vertical whiskers, we want near-vertical lines (angle close to 90°)
                if angle > 80 or angle < 10:  # Nearly vertical
                    valid_lines.append((x1, y1, x2, y2))
            else:  # horizontal
                # For horizontal whiskers, we want near-horizontal lines (angle close to 0°)
                if angle < 10 or angle > 170:  # Nearly horizontal
                    valid_lines.append((x1, y1, x2, y2))
        
        if not valid_lines:
            return None
        
        # Find the extent (furthest point from box)
        if orientation == 'vertical':
            if whisker_type == 'high':
                # Find topmost point (smallest y)
                extent_pixel = min(min(y1, y2) for x1, y1, x2, y2 in valid_lines)
            else:  # low
                # Find bottommost point (largest y)
                extent_pixel = max(max(y1, y2) for x1, y1, x2, y2 in valid_lines)
            
            # Convert to original image coordinates
            return float(region_offset[1] + extent_pixel)
        else:  # horizontal
            if whisker_type == 'high':
                # Find rightmost point (largest x)
                extent_pixel = max(max(x1, x2) for x1, y1, x2, y2 in valid_lines)
            else:  # low
                # Find leftmost point (smallest x)
                extent_pixel = min(min(x1, x2) for x1, y1, x2, y2 in valid_lines)
            
            # Convert to original image coordinates
            return float(region_offset[0] + extent_pixel)