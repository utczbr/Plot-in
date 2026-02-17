"""
Unit tests for ThreadSafetyManager

Tests for core/thread_safety.py - unified Qt threading primitives.
"""

import pytest
from PyQt6.QtCore import QThread, pyqtSignal
from core.thread_safety import ThreadSafetyManager, get_thread_safety_manager
import time


class WorkerThread(QThread):
    """Test worker thread for concurrent access testing."""
    
    result_ready = pyqtSignal(int)
    
    def __init__(self, manager, operation, iterations=100):
        super().__init__()
        self.manager = manager
        self.operation = operation
        self.iterations = iterations
        self.counter = 0
    
    def run(self):
        for _ in range(self.iterations):
            self.operation()
        self.result_ready.emit(self.counter)


def test_thread_safety_manager_initialization():
    """Test ThreadSafetyManager initializes correctly."""
    manager = ThreadSafetyManager(max_concurrent_analyses=4)
    assert manager is not None
    stats = manager.get_statistics()
    assert stats['slots_available'] == 4


def test_model_access_context_manager():
    """Test model_access context manager for basic operation."""
    manager = ThreadSafetyManager()
    
    with manager.model_access():
        # Should not raise exception
        pass


def test_model_access_recursive():
    """Test model_access allows re-entrant calls (recursive mutex)."""
    manager = ThreadSafetyManager()
    
    def recursive_operation(depth=3):
        with manager.model_access():
            if depth > 0:
                recursive_operation(depth - 1)
    
    # Should not deadlock on recursive calls
    recursive_operation()


def test_cache_read_concurrent():
    """Test cache_read allows concurrent access."""
    manager = ThreadSafetyManager()
    shared_data = {'value': 0}
    
    def read_operation():
        with manager.cache_read():
            # Multiple threads should be able to read simultaneously
            _ = shared_data['value']
            time.sleep(0.001)  # Simulate read time
    
    threads = [WorkerThread(manager, read_operation, iterations=10) for _ in range(5)]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.wait()
    
    # All threads should complete without deadlock


def test_cache_write_exclusive():
    """Test cache_write provides exclusive access."""
    manager = ThreadSafetyManager()
    shared_counter = [0]  # Use list for mutability in closure
    
    def write_operation():
        with manager.cache_write():
            current = shared_counter[0]
            time.sleep(0.001)  # Simulate write time
            shared_counter[0] = current + 1
    
    threads = [WorkerThread(manager, write_operation, iterations=10) for _ in range(5)]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.wait()
    
    # If writes were truly exclusive, counter should be exactly 50
    assert shared_counter[0] == 50


def test_analysis_slot_acquisition():
    """Test analysis slot semaphore limits concurrent access."""
    manager = ThreadSafetyManager(max_concurrent_analyses=2)
    
    # First two should succeed
    assert manager.acquire_analysis_slot() is True
    assert manager.acquire_analysis_slot() is True
    
    # Third should timeout (all slots occupied)
    assert manager.acquire_analysis_slot(timeout_ms=10) is False
    
    # Release one slot
    manager.release_analysis_slot()
    
    # Now acquisition should succeed
    assert manager.acquire_analysis_slot() is True


def test_analysis_slot_context_manager():
    """Test analysis_slot context manager with automatic release."""
    manager = ThreadSafetyManager(max_concurrent_analyses=1)
    
    with manager.analysis_slot() as acquired:
        assert acquired is True
        # Slot should be occupied
        assert manager.acquire_analysis_slot(timeout_ms=10) is False
    
    # After context exit, slot should be released
    assert manager.acquire_analysis_slot() is True


def test_exception_safety():
    """Test that locks are released even on exceptions."""
    manager = ThreadSafetyManager()
    
    try:
        with manager.model_access():
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Lock should be released, so this should not deadlock
    with manager.model_access():
        pass


def test_singleton_instance():
    """Test get_thread_safety_manager returns singleton."""
    manager1 = get_thread_safety_manager()
    manager2 = get_thread_safety_manager()
    
    assert manager1 is manager2


def test_statistics_tracking():
    """Test statistics are tracked correctly."""
    manager = ThreadSafetyManager(max_concurrent_analyses=3)
    
    # Acquire slots
    manager.acquire_analysis_slot()
    manager.acquire_analysis_slot()
    
    stats = manager.get_statistics()
    assert stats['slots_acquired'] == 2
    assert stats['slots_available'] == 1
    
    # Release one
    manager.release_analysis_slot()
    
    stats = manager.get_statistics()
    assert stats['slots_available'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
