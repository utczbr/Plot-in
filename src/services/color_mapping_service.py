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
        
    def calibrate_from_known_values(self, color_samples: List[Tuple[np.ndarray, float]]) -> None:
        """
        Calibrate using 3D RGB trajectory mapping.
        
        Learns the piecewise linear curve through RGB space that defines the color scale.
        """
        if not color_samples or len(color_samples) < 2:
            return
            
        # 1. Extract mean RGB vectors for each sample
        points = []
        for color_sample, val in color_samples:
            if color_sample.size > 0:
                # Average BGR color (opencv uses BGR)
                avg_bgr = np.mean(color_sample, axis=(0, 1))
                points.append({
                    'val': val,
                    'vec': avg_bgr.astype(float) # [B, G, R]
                })
        
        if len(points) < 2:
            return

        # 2. Sort by value to define the trajectory order
        points.sort(key=lambda x: x['val'])
        
        # 3. Store calibration curve
        self.calibration_curve = points
        self.is_calibrated = True
        
        self.min_value = points[0]['val']
        self.max_value = points[-1]['val']
        self.value_range = self.max_value - self.min_value

    def map_color_to_value(self, cell_image: np.ndarray) -> float:
        """
        Map color to value using orthogonal projection onto the calibrated 3D curve.
        
        Fallback hierarchy:
        1. Calibrated 3D RGB curve projection (most accurate)
        2. LAB lightness mapping (good for grayscale/intensity scales)
        3. HSV hue mapping (good for colorscale heatmaps like red→blue)
        4. HSV brightness (final fallback)
        """
        if cell_image.size == 0:
            return 0.0
            
        # Extract query vector
        query_vec = np.mean(cell_image, axis=(0, 1)).astype(float) # [B, G, R]

        # Tier 1: Calibrated curve projection
        if hasattr(self, 'is_calibrated') and self.is_calibrated:
            try:
                return self._project_onto_curve(query_vec)
            except Exception:
                pass  # Fall through to fallbacks

        # Tier 2: LAB lightness mapping (good for grayscale/intensity colorscales)
        try:
            lab = cv2.cvtColor(cell_image, cv2.COLOR_BGR2LAB)
            avg_l = np.mean(lab[:,:,0])  # L channel (lightness)
            normalized = avg_l / 255.0
            return self.min_value + normalized * self.value_range
        except Exception:
            pass

        # Tier 3: HSV hue mapping (good for rainbow/colorscale heatmaps)
        try:
            hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
            avg_h = np.mean(hsv[:,:,0])  # H channel (hue, 0-179 in OpenCV)
            avg_s = np.mean(hsv[:,:,1])  # S channel (saturation)
            
            # Only use hue if saturation is high enough (colored, not grayscale)
            if avg_s > 30:  # Threshold for "has color"
                # Normalize hue: OpenCV hue is 0-179
                # Common heatmap: blue(120) -> cyan(90) -> green(60) -> yellow(30) -> red(0/180)
                # We'll map hue linearly but handle the red wrap-around
                
                # Shift so that blue (120) is 0 and red (0/180) is 1
                # hue_shift = (180 - hue) mod 180 gives: red=0, blue=60 (inverted)
                # Better: hue_norm = (120 - hue) / 120 for blue→green→red
                
                if avg_h <= 120:
                    # Blue(120) to Red(0): linear mapping
                    hue_normalized = 1.0 - (avg_h / 120.0)
                else:
                    # Red wraps at 180, so hue 121-179 is also "red-ish"
                    # Map 120-180 range to ~0 (close to red)
                    hue_normalized = 1.0 - ((180 - avg_h) / 120.0)
                    hue_normalized = max(0.0, min(1.0, hue_normalized))
                
                return self.min_value + hue_normalized * self.value_range
        except Exception:
            pass

        # Tier 4: HSV brightness (final fallback)
        try:
            hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
            avg_v = np.mean(hsv[:,:,2])
            normalized = avg_v / 255.0
            return self.min_value + normalized * self.value_range
        except Exception:
            return self.min_value  # Absolute fallback

    def _project_onto_curve(self, query: np.ndarray) -> float:
        """
        Find closest point on the piecewise linear curve and return its interpolated value.
        """
        curve = self.calibration_curve
        best_dist_sq = float('inf')
        best_val = curve[0]['val']
        
        # Check each segment
        for i in range(len(curve) - 1):
            p1 = curve[i]
            p2 = curve[i+1]
            
            v = p2['vec'] - p1['vec'] # Segment vector
            w = query - p1['vec']     # Vector from p1 to query
            
            # Project w onto v
            # t = (w . v) / (v . v)
            v_len_sq = np.dot(v, v)
            if v_len_sq < 1e-6:
                # Points are identical, treat as single point
                t = 0.0
            else:
                t = np.dot(w, v) / v_len_sq
                t = max(0.0, min(1.0, t)) # Clamp to segment
            
            # Closest point on segment
            closest = p1['vec'] + t * v
            
            # Distance squared
            d = closest - query
            dist_sq = np.dot(d, d)
            
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                # Linear interpolation of value
                best_val = p1['val'] + t * (p2['val'] - p1['val'])
                
        return float(best_val)