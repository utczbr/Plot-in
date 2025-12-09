"""
Image Manager - Service for handling image operations including loading, caching, and processing.
"""
import threading
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image, ImageDraw
import numpy as np
from .data_manager import DataManager

class ImageManager:
    """Service for handling all image-related operations."""
    
    def __init__(self, data_manager: DataManager, max_cache_size: int = 50):
        self.data_manager = data_manager
        self._pixmap_cache = {}
        self._cache_lock = threading.Lock()
        self.max_cache_size = max_cache_size
        
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load image from file with proper error handling."""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
            
    def create_image_with_detections(self, image_path: str, result: dict, 
                                   target_size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
        """Create image with detection overlays drawn."""
        original_img = self.load_image(image_path)
        if not original_img:
            return None
            
        img = original_img.copy()
        
        # Calculate scaling if target_size is provided
        if target_size and target_size != img.size:
            original_width, original_height = img.size
            target_width, target_height = target_size
            scale_x = target_width / original_width
            scale_y = target_height / original_height
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        else:
            scale_x = scale_y = 1.0
            
        draw = ImageDraw.Draw(img)
        
        # Draw detections
        if result and 'detections' in result:
            for class_name, items in result['detections'].items():
                for item in items:
                    bbox = item.get('xyxy')
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        # Scale bbox coordinates
                        scaled_bbox = [
                            int(x1 * scale_x),
                            int(y1 * scale_y),
                            int(x2 * scale_x),
                            int(y2 * scale_y)
                        ]
                        # Use a simple color map
                        color_map = {
                            'bar': (0, 120, 255),
                            'line': (255, 0, 0),
                            'scatter': (0, 128, 0),
                            'box': (128, 0, 128),
                            'data_point': (255, 165, 60),
                            'axis_title': (255, 165, 0),
                            'chart_title': (50, 50, 220),
                            'legend': (210, 105, 30),
                            'axis_labels': (255, 0, 255),
                            'scale_label': (255, 117, 24),
                            'tick_label': (0, 255, 255),
                            'other': (128, 128, 128),
                            'baseline': (240, 240, 240)
                        }
                        color = color_map.get(class_name, (128, 128, 128))
                        draw.rectangle(scaled_bbox, outline=color, width=2)
        
        # Draw baseline if present
        baseline_coord = result.get('baseline_coord') if result else None
        if baseline_coord is not None:
            y = int(baseline_coord * scale_y)
            draw.line([(0, y), (img.width, y)], fill=(255, 215, 0), width=2)
            
        return img
        
    def cache_pixmap(self, key: str, pixmap):
        """Cache a pixmap with LRU eviction."""
        with self._cache_lock:
            if key in self._pixmap_cache:
                self._pixmap_cache[key] = pixmap
                return
                
            # Check if cache is full
            if len(self._pixmap_cache) >= self.max_cache_size:
                # Remove oldest entries
                oldest_keys = list(self._pixmap_cache.keys())
                for old_key in oldest_keys[:5]:  # Remove 5 oldest to make space
                    if old_key in self._pixmap_cache:
                        self._pixmap_cache.pop(old_key)
                        
            self._pixmap_cache[key] = pixmap
            
    def get_cached_pixmap(self, key: str):
        """Get cached pixmap."""
        with self._cache_lock:
            return self._pixmap_cache.get(key)
            
    def clear_pixmap_cache(self):
        """Clear pixmap cache."""
        with self._cache_lock:
            self._pixmap_cache.clear()
            
    def get_image_dimensions(self, image_path: str) -> Optional[Tuple[int, int]]:
        """Get image dimensions without loading full image."""
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception:
            return None