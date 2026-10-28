"""
Image Manager - Service for handling image operations including loading, caching, and processing.
"""
from typing import Optional
from PIL import Image
from .data_manager import DataManager

class ImageManager:
    """Service for handling image loading operations."""
    
    def __init__(self, data_manager: DataManager, max_cache_size: int = 50):
        self.data_manager = data_manager
        
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