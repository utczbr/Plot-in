"""
Smart Pixmap Cache - Memory-Bounded LRU Cache with Thread Safety

This module provides a memory-aware pixmap cache with automatic LRU eviction,
replacing the unbounded OrderedDict cache in main_modern.py (Line 221).

Author: Chart Analysis Tool Team
Date: November 24, 2025
"""

from collections import OrderedDict
from typing import Optional, Dict
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QObject, pyqtSignal
import gc
import logging

logger = logging.getLogger(__name__)


class SmartPixmapCache(QObject):
    """
    LRU cache with memory-bounded eviction and thread safety.
    
    Features:
    - Memory-based eviction (not just entry count)
    - LRU (Least Recently Used) eviction policy
    - Thread-safe via ThreadSafetyManager integration
    - Explicit cleanup with gc.collect()
    - Hit/miss statistics tracking
    
    Replaces the unbounded OrderedDict cache (main_modern.py:221) that caused
    memory leaks when processing 100+ images.
    
    Example:
        >>> cache = SmartPixmapCache(max_memory_mb=150)
        >>> cache.insert('image1.png', pixmap)
        >>> pixmap = cache.get('image1.png')
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']}")
    """
    
    # Signals
    cache_full = pyqtSignal()  # Emitted when cache reaches capacity
    eviction_occurred = pyqtSignal(str)  # Emitted when key evicted
    
    def __init__(self, max_memory_mb: int = 150, thread_safety_manager=None):
        """
        Initialize smart pixmap cache.
        
        Args:
            max_memory_mb: Maximum memory usage in megabytes (default: 150MB)
            thread_safety_manager: Optional ThreadSafetyManager for locking
        """
        super().__init__()
        self._cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._max_bytes = max_memory_mb * 1024 * 1024
        self._current_bytes = 0
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        
        # Thread safety (optional, can be added later)
        self._thread_safety = thread_safety_manager
        
        logger.info(f"SmartPixmapCache initialized with {max_memory_mb}MB limit")
    
    def _estimate_pixmap_size(self, pixmap: QPixmap) -> int:
        """
        Calculate memory footprint of pixmap.
        
        Formula: width * height * 4 bytes (RGBA 32-bit)
        
        Args:
            pixmap: QPixmap to estimate
        
        Returns:
            Estimated size in bytes
        """
        if pixmap.isNull():
            return 0
        return pixmap.width() * pixmap.height() * 4
    
    def insert(self, key: str, pixmap: QPixmap) -> None:
        """
        Insert pixmap with automatic LRU eviction.
        
        If cache is full, evicts least recently used entries until
        there's space for the new pixmap.
        
        Args:
            key: Cache key (typically image path)
            pixmap: QPixmap to cache
        
        Example:
            >>> cache.insert('path/to/image.png', pixmap)
        """
        if pixmap.isNull():
            logger.warning(f"Attempted to cache null pixmap for key: {key}")
            return
        
        size = self._estimate_pixmap_size(pixmap)
        
        # Use thread safety if available
        context = self._thread_safety.cache_write() if self._thread_safety else self._null_context()
        
        with context:
            # Remove key if already exists (update case)
            if key in self._cache:
                old_pixmap = self._cache.pop(key)
                old_size = self._estimate_pixmap_size(old_pixmap)
                self._current_bytes -= old_size
                del old_pixmap
            
            # Evict oldest entries until we have space
            while (self._current_bytes + size > self._max_bytes and len(self._cache) > 0):
                oldest_key, oldest_pixmap = self._cache.popitem(last=False)
                evicted_size = self._estimate_pixmap_size(oldest_pixmap)
                self._current_bytes -= evicted_size
                self._eviction_count += 1
                
                # Explicitly delete to trigger Qt cleanup
                del oldest_pixmap
                
                logger.debug(f"Evicted '{oldest_key}' ({evicted_size / 1024:.1f} KB)")
                self.eviction_occurred.emit(oldest_key)
            
            # Insert new pixmap
            self._cache[key] = pixmap
            self._current_bytes += size
            
            logger.debug(f"Cached '{key}' ({size / 1024:.1f} KB, total: {self._current_bytes / 1024 / 1024:.1f} MB)")
            
            # Emit signal if we're near capacity
            if self._current_bytes > self._max_bytes * 0.9:
                self.cache_full.emit()
    
    def get(self, key: str) -> Optional[QPixmap]:
        """
        Retrieve pixmap and mark as recently used.
        
        Args:
            key: Cache key
        
        Returns:
            Cached QPixmap if found, None otherwise
        
        Example:
            >>> pixmap = cache.get('path/to/image.png')
            >>> if pixmap:
            ...     display(pixmap)
        """
        context = self._thread_safety.cache_read() if self._thread_safety else self._null_context()
        
        with context:
            if key in self._cache:
                self._hit_count += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            
            self._miss_count += 1
            return None
    
    def remove(self, key: str) -> bool:
        """
        Explicitly remove a key from cache.
        
        Args:
            key: Cache key to remove
        
        Returns:
            True if key was removed, False if not found
        """
        context = self._thread_safety.cache_write() if self._thread_safety else self._null_context()
        
        with context:
            if key in self._cache:
                pixmap = self._cache.pop(key)
                size = self._estimate_pixmap_size(pixmap)
                self._current_bytes -= size
                del pixmap
                logger.debug(f"Removed '{key}' from cache")
                return True
            return False
    
    def clear(self) -> None:
        """
        Flush entire cache and force garbage collection.
        
        Use this when switching projects or on application exit.
        
        Example:
            >>> cache.clear()
            >>> gc.collect()  # Called automatically
        """
        context = self._thread_safety.cache_write() if self._thread_safety else self._null_context()
        
        with context:
            num_entries = len(self._cache)
            self._cache.clear()
            self._current_bytes = 0
            
            # Force garbage collection to immediately free memory
            gc.collect()
            
            logger.info(f"Cache cleared ({num_entries} entries freed)")
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        context = self._thread_safety.cache_read() if self._thread_safety else self._null_context()
        
        with context:
            return key in self._cache
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary with statistics:
            - entries: Number of cached pixmaps
            - memory_mb: Current memory usage in MB
            - hit_rate: Cache hit rate percentage
            - hits: Total cache hits
            - misses: Total cache misses
            - evictions: Total evictions
            - capacity_pct: Percentage of max capacity used
        
        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']}, Memory: {stats['memory_mb']:.1f} MB")
        """
        total = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total * 100) if total > 0 else 0
        capacity_pct = (self._current_bytes / self._max_bytes * 100) if self._max_bytes > 0 else 0
        
        return {
            'entries': len(self._cache),
            'memory_mb': self._current_bytes / (1024 * 1024),
            'max_memory_mb': self._max_bytes / (1024 * 1024),
            'hit_rate': f'{hit_rate:.1f}%',
            'hits': self._hit_count,
            'misses': self._miss_count,
            'evictions': self._eviction_count,
            'capacity_pct': f'{capacity_pct:.1f}%'
        }
    
    def reset_stats(self) -> None:
        """Reset hit/miss/eviction counters."""
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        logger.debug("Cache statistics reset")
    
    @staticmethod
    def _null_context():
        """Null context manager for when thread_safety is not available."""
        from contextlib import nullcontext
        return nullcontext()
    
    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return self.contains(key)
