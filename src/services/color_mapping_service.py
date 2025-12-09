"""
Color mapping service for heatmaps.

Maps pixel colors to numeric values based on a defined color scale.
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional


class ColorMappingService:
    """
    Service to map colors to numeric values for heatmaps.
    
    Supports various color mapping strategies including:
    - Predefined color scales (viridis, plasma, etc.)
    - Custom color ranges
    - HSV-based mapping
    """
    
    def __init__(self, color_scale: Optional[str] = None, 
                 min_value: float = 0.0, 
                 max_value: float = 1.0):
        """
        Initialize the color mapping service.
        
        Args:
            color_scale: Predefined color scale ('viridis', 'plasma', 'hot', etc.) or None
            min_value: Minimum value in the data range
            max_value: Maximum value in the data range
        """
        self.color_scale = color_scale
        self.min_value = min_value
        self.max_value = max_value
        self.value_range = max_value - min_value
        
    def map_color_to_value(self, cell_image: np.ndarray) -> float:
        """
        Map the color of a cell to a numeric value.
        
        Args:
            cell_image: Image array of the heatmap cell
            
        Returns:
            Numeric value corresponding to the cell's color
        """
        if cell_image.size == 0:
            return 0.0
            
        # Convert to HSV for better color-based value estimation
        hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
        
        # For different color scales, different approaches work better
        if self.color_scale == 'viridis':
            # For viridis, brightness (V) often correlates with value
            avg_v = np.mean(hsv[:,:,2])
            return self.min_value + (avg_v / 255.0) * self.value_range
        elif self.color_scale == 'hot':
            # For hot colormap, use a combination of HSV channels
            avg_h = np.mean(hsv[:,:,0])
            avg_s = np.mean(hsv[:,:,1])
            avg_v = np.mean(hsv[:,:,2])
            # Weight the value channel heavily, with some consideration for hue/saturation
            weighted_value = (0.2 * (avg_h / 179.0) + 0.2 * (avg_s / 255.0) + 0.6 * (avg_v / 255.0))
            return self.min_value + weighted_value * self.value_range
        else:
            # General approach: use brightness as proxy for value
            avg_brightness = np.mean(hsv[:,:,2])
            # Normalize to range [0, 1] then scale to our desired range
            normalized = avg_brightness / 255.0
            return self.min_value + normalized * self.value_range
    
    def calibrate_from_known_values(self, color_samples: List[Tuple[np.ndarray, float]]) -> None:
        """
        Calibrate the color mapping using known color-value pairs.
        
        Args:
            color_samples: List of (color_sample, true_value) tuples
        """
        if not color_samples:
            return
            
        # Calculate the relationship between color intensity and actual values
        intensities = []
        true_values = []
        
        for color_sample, true_value in color_samples:
            if color_sample.size > 0:
                hsv = cv2.cvtColor(color_sample, cv2.COLOR_BGR2HSV)
                avg_brightness = np.mean(hsv[:,:,2])
                intensities.append(avg_brightness)
                true_values.append(true_value)
        
        if len(intensities) > 1:
            # Fit a simple linear relationship
            # In a real implementation, we'd use a more sophisticated model
            self.min_value = min(true_values)
            self.max_value = max(true_values)
            self.value_range = self.max_value - self.min_value