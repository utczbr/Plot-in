"""
Data Manager - Centralized service for managing analysis results, caching, and data operations.
"""
import json
import gc
from pathlib import Path
from typing import Dict, Any, Optional
from collections import OrderedDict
import threading
from PIL import Image

class DataManager:
    """Centralized service for managing analysis data and caching."""
    
    def __init__(self, max_cache_size: int = 10):
        self._analysis_results = {}
        self._image_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self.max_cache_size = max_cache_size
        self._results_lock = threading.Lock()
        
    def store_analysis_result(self, image_path: str, result: Dict[str, Any]):
        """Store analysis results for an image."""
        with self._results_lock:
            self._analysis_results[image_path] = result
            
    def get_analysis_result(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results for an image."""
        with self._results_lock:
            return self._analysis_results.get(image_path)
            
    def clear_analysis_result(self, image_path: str):
        """Remove analysis results for an image."""
        with self._results_lock:
            if image_path in self._analysis_results:
                del self._analysis_results[image_path]
                
    def clear_all_results(self):
        """Clear all stored analysis results."""
        with self._results_lock:
            self._analysis_results.clear()
            
    def cache_image(self, key: str, image: Image.Image):
        """Cache an image with LRU eviction policy."""
        with self._cache_lock:
            if key in self._image_cache:
                self._image_cache.move_to_end(key)
                return
                
            # Check if cache is full
            if len(self._image_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key, oldest_image = self._image_cache.popitem(last=False)
                if oldest_image:
                    oldest_image.close()
                    
            self._image_cache[key] = image
            
    def get_cached_image(self, key: str) -> Optional[Image.Image]:
        """Retrieve cached image and mark as recently used."""
        with self._cache_lock:
            if key in self._image_cache:
                self._image_cache.move_to_end(key)
                return self._image_cache[key]
            return None
            
    def clear_image_cache(self):
        """Clear all cached images."""
        with self._cache_lock:
            for img in self._image_cache.values():
                if img:
                    img.close()
            self._image_cache.clear()
            gc.collect()
            
    def clear_cache_for_key(self, key: str):
        """Clear specific cache entry."""
        with self._cache_lock:
            if key in self._image_cache:
                img = self._image_cache.pop(key)
                if img:
                    img.close()
                    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._results_lock:
            results_count = len(self._analysis_results)
        with self._cache_lock:
            cache_count = len(self._image_cache)
        return {
            'analysis_results_count': results_count,
            'image_cache_count': cache_count,
            'max_cache_size': self.max_cache_size
        }