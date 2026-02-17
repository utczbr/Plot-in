"""
Unit tests for SmartPixmapCache

Tests for core/pixmap_cache.py - memory-bounded LRU cache.
"""

import pytest
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from core.pixmap_cache import SmartPixmapCache
import time


def create_test_pixmap(width: int, height: int) -> QPixmap:
    """Helper to create test pixmap of specific size."""
    image = QImage(width, height, QImage.Format.Format_RGB888)
    image.fill(Qt.GlobalColor.blue)
    return QPixmap.fromImage(image)


def test_pixmap_cache_initialization():
    """Test SmartPixmapCache initializes correctly."""
    cache = SmartPixmapCache(max_memory_mb=100)
    assert cache is not None
    assert len(cache) == 0
    
    stats = cache.get_stats()
    assert stats['entries'] == 0
    assert stats['memory_mb'] == 0


def test_pixmap_size_estimation():
    """Test pixmap size estimation is accurate."""
    cache = SmartPixmapCache()
    
    # 100x100 RGBA = 100 * 100 * 4 = 40,000 bytes
    pixmap = create_test_pixmap(100, 100)
    size = cache._estimate_pixmap_size(pixmap)
    assert size == 40000


def test_insert_and_get():
    """Test basic insert and retrieval."""
    cache = SmartPixmapCache(max_memory_mb=10)
    
    pixmap = create_test_pixmap(100, 100)
    cache.insert('test_key', pixmap)
    
    retrieved = cache.get('test_key')
    assert retrieved is not None
    assert retrieved.width() == 100
    assert retrieved.height() == 100


def test_cache_hit_and_miss():
    """Test hit/miss statistics are tracked correctly."""
    cache = SmartPixmapCache()
    
    pixmap = create_test_pixmap(50, 50)
    cache.insert('key1', pixmap)
    
    # Hit
    cache.get('key1')
    stats = cache.get_stats()
    assert stats['hits'] == 1
    assert stats['misses'] == 0
    
    # Miss
    cache.get('nonexistent')
    stats = cache.get_stats()
    assert stats['hits'] == 1
    assert stats['misses'] == 1


def test_lru_behavior():
    """Test LRU (Least Recently Used) eviction."""
    cache = SmartPixmapCache()
    
    pixmap = create_test_pixmap(10, 10)
    
    cache.insert('key1', pixmap)
    cache.insert('key2', pixmap)
    cache.insert('key3', pixmap)
    
    # Access key1 to make it most recently used
    cache.get('key1')
    
    # Access order: key2 (oldest), key3, key1 (newest)
    # Verify with internal cache order
    assert list(cache._cache.keys()) == ['key2', 'key3', 'key1']


def test_automatic_eviction():
    """Test automatic LRU eviction when memory limit reached."""
    # Small cache: ~0.01 MB = 10,240 bytes
    cache = SmartPixmapCache(max_memory_mb=0.01)
    
    # Each 50x50 pixmap = 50*50*4 = 10,000 bytes
    pixmap = create_test_pixmap(50, 50)
    
    # First pixmap should fit
    cache.insert('key1', pixmap)
    assert 'key1' in cache
    
    # Second pixmap should trigger eviction of first
    cache.insert('key2', pixmap)
    assert 'key2' in cache
    assert 'key1' not in cache  # Evicted
    
    stats = cache.get_stats()
    assert stats['evictions'] == 1


def test_update_existing_key():
    """Test updating existing key doesn't increase memory."""
    cache = SmartPixmapCache(max_memory_mb=10)
    
    pixmap1 = create_test_pixmap(100, 100)
    pixmap2 = create_test_pixmap(100, 100)
    
    cache.insert('key', pixmap1)
    initial_mem = cache._current_bytes
    
    # Update with same-sized pixmap
    cache.insert('key', pixmap2)
    
    # Memory should stay the same (replaced, not added)
    assert cache._current_bytes == initial_mem
    assert len(cache) == 1


def test_remove():
    """Test explicit removal."""
    cache = SmartPixmapCache()
    
    pixmap = create_test_pixmap(50, 50)
    cache.insert('key', pixmap)
    
    assert cache.remove('key') is True
    assert 'key' not in cache
    assert cache.remove('key') is False  # Already removed


def test_clear():
    """Test clearing entire cache."""
    cache = SmartPixmapCache()
    
    for i in range(10):
        pixmap = create_test_pixmap(50, 50)
        cache.insert(f'key{i}', pixmap)
    
    assert len(cache) == 10
    
    cache.clear()
    
    assert len(cache) == 0
    assert cache._current_bytes == 0


def test_contains():
    """Test cache containment check."""
    cache = SmartPixmapCache()
    
    pixmap = create_test_pixmap(50, 50)
    cache.insert('key', pixmap)
    
    assert cache.contains('key') is True
    assert 'key' in cache  # __contains__ magic method
    assert 'nonexistent' not in cache


def test_stats_accuracy():
    """Test statistics are accurate."""
    cache = SmartPixmapCache(max_memory_mb=10)
    
    # Insert 5 pixmaps of 100x100 (40KB each)
    for i in range(5):
        pixmap = create_test_pixmap(100, 100)
        cache.insert(f'key{i}', pixmap)
    
    stats = cache.get_stats()
    
    assert stats['entries'] == 5
    # 5 * 40,000 bytes = 200,000 bytes ≈ 0.19 MB
    assert 0.18 < stats['memory_mb'] < 0.21


def test_reset_stats():
    """Test statistics reset."""
    cache = SmartPixmapCache()
    
    pixmap = create_test_pixmap(50, 50)
    cache.insert('key', pixmap)
    cache.get('key')
    cache.get('nonexistent')
    
    cache.reset_stats()
    
    stats = cache.get_stats()
    assert stats['hits'] == 0
    assert stats['misses'] == 0
    assert stats['evictions'] == 0


def test_null_pixmap_handling():
    """Test null pixmap is rejected."""
    cache = SmartPixmapCache()
    
    null_pixmap = QPixmap()
    assert null_pixmap.isNull()
    
    # Should not insert null pixmap
    cache.insert('null_key', null_pixmap)
    assert 'null_key' not in cache


def test_cache_signals(qtbot):
    """Test cache emits signals correctly."""
    cache = SmartPixmapCache(max_memory_mb=0.01)
    
    # Track eviction signal
    evicted_keys = []
    cache.eviction_occurred.connect(lambda key: evicted_keys.append(key))
    
    # Insert two pixmaps, second should evict first
    pixmap = create_test_pixmap(50, 50)
    cache.insert('key1', pixmap)
    cache.insert('key2', pixmap)
    
    # Check eviction signal was emitted
    assert 'key1' in evicted_keys


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
