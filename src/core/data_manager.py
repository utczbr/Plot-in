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
        self._context: Optional[Dict[str, Any]] = None
        
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
            
    _CONTEXT_FORMAT_HINT = (
        'Expected JSON format:\n'
        '{\n'
        '  "outcomes": ["outcome1", "outcome2"],\n'
        '  "groups": ["group1", "group2"],\n'
        '  "units": {"outcome1": "kg"},\n'
        '  "error_bar_type": "SE"\n'
        '}'
    )

    def load_context(self, path: str) -> Dict[str, Any]:
        """Load and validate a context-of-interest JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                ctx = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in context file: {e}\n\n"
                f"{self._CONTEXT_FORMAT_HINT}"
            ) from e
        if not isinstance(ctx, dict):
            raise ValueError(
                f"Context file must contain a JSON object (dict), "
                f"got {type(ctx).__name__}.\n\n"
                f"{self._CONTEXT_FORMAT_HINT}"
            )
        if not isinstance(ctx.get('outcomes'), list):
            ctx['outcomes'] = []
        if not isinstance(ctx.get('groups'), list):
            ctx['groups'] = []
        self._context = ctx
        return ctx

    def get_context(self) -> Optional[Dict[str, Any]]:
        """Return the loaded context-of-interest, or None."""
        return self._context

    def clear_context(self):
        """Clear the loaded context."""
        self._context = None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._results_lock:
            results_count = len(self._analysis_results)
        return {
            'analysis_results_count': results_count
        }
                    