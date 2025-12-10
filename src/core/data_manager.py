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
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._results_lock:
            results_count = len(self._analysis_results)
        return {
            'analysis_results_count': results_count
        }
                    