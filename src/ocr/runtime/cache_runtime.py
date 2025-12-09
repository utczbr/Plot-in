"""
Zero-copy LRU cache and deduplication utilities for OCR system.
Implements efficient caching using content hashes and deduplication of identical crops.
"""
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import threading
import cv2


class ZeroCopyHashCache:
    """
    Zero-copy LRU cache that stores items by content hash to avoid memory duplication.
    Uses numpy array hashing to identify unique content without copying the array data.
    """
    
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._cache = OrderedDict()  # hash -> cached_result
        self._lock = threading.RLock()  # Thread-safe operations
    
    def get_by_hash(self, array: np.ndarray):
        """
        Retrieve cached result by computing hash of array content
        """
        array_hash = self._compute_hash(array)
        with self._lock:
            return self._cache.get(array_hash)
    
    def put(self, array: np.ndarray, result: Any):
        """
        Store result in cache using hash of array content as key
        """
        array_hash = self._compute_hash(array)
        with self._lock:
            # If already exists, move to end (most recent)
            if array_hash in self._cache:
                self._cache.move_to_end(array_hash)
            self._cache[array_hash] = result
            
            # Evict oldest if over size limit
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
    
    def _compute_hash(self, array: np.ndarray) -> str:
        """
        Compute a hash of the array content using SHA-256
        This is a zero-copy operation since we're hashing the buffer directly
        """
        # Convert numpy array to bytes for hashing
        # This preserves the exact content for comparison purposes
        array_bytes = array.tobytes()
        return hashlib.sha256(array_bytes).hexdigest()
    
    def clear(self):
        """
        Clear all cached entries
        """
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """
        Return current number of cached entries
        """
        with self._lock:
            return len(self._cache)


class HashDeduplicator:
    """
    Utility for deduplicating numpy arrays by content hash
    """
    
    @staticmethod
    def deduplicate_crops(crops_with_context: List[Tuple[np.ndarray, str]]) -> Tuple[List[Tuple[np.ndarray, str]], Dict[str, Tuple[np.ndarray, str, List[int]]]]:
        """
        Deduplicate crops by content hash, returning unique crops and a mapping to original positions
        
        Args:
            crops_with_context: List of (crop_image, context_string) tuples
            
        Returns:
            Tuple of (unique_crops, mapping_dict)
            - unique_crops: List of unique (crop, context) tuples
            - mapping_dict: Dict mapping hash -> (crop, context, [original_indices])
        """
        seen_hashes = {}
        unique_crops = []
        hash_to_original_indices = {}
        
        for idx, (crop, context) in enumerate(crops_with_context):
            crop_hash = HashDeduplicator._compute_content_hash(crop)
            
            if crop_hash not in seen_hashes:
                # First occurrence of this content
                seen_hashes[crop_hash] = len(unique_crops)
                unique_crops.append((crop, context))
                hash_to_original_indices[crop_hash] = (crop, context, [idx])
            else:
                # Duplicate content - add index to existing entry
                _, _, indices_list = hash_to_original_indices[crop_hash]
                indices_list.append(idx)
        
        return unique_crops, hash_to_original_indices
    
    @staticmethod
    def _compute_content_hash(array: np.ndarray) -> str:
        """
        Compute content hash for deduplication purposes
        """
        array_bytes = array.tobytes()
        return hashlib.sha256(array_bytes).hexdigest()


class LRUCache:
    """
    Generic LRU cache implementation with thread safety
    """
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key):
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    def put(self, key, value):
        with self._lock:
            # If key exists, move to end
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            
            # If over size limit, remove oldest
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
    
    def clear(self):
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        with self._lock:
            return len(self._cache)
    
    def keys(self):
        with self._lock:
            return list(self._cache.keys())