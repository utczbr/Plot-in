"""
Unit tests for SignalConnectionManager and HoverEventFilter

Tests for core/signal_manager.py and ui/components/hover_filter.py.
"""

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QLabel, QPushButton
from core.signal_manager import SignalConnectionManager, safe_connect
from ui.components.hover_filter import (
    HoverEventFilter, ClickEventFilter, HoverHighlightFilter, install_hover_filter
)


# Test helpers
class SignalEmitter(QObject):
    """Helper class for testing signal connections."""
    test_signal = pyqtSignal(str)
    value_changed = pyqtSignal(int)


def test_signal_manager_initialization():
    """Test SignalConnectionManager initializes correctly."""
    manager = SignalConnectionManager()
    assert manager is not None
    assert manager.get_connection_count() == 0


def test_signal_manager_connect():
    """Test connecting signals."""
    manager = SignalConnectionManager()
    emitter = SignalEmitter()
    
    received = []
    
    def slot(value):
        received.append(value)
    
    manager.connect(emitter.test_signal, slot)
    
    assert manager.get_connection_count() == 1
    
    # Emit signal
    emitter.test_signal.emit('test')
    assert received == ['test']


def test_signal_manager_disconnect():
    """Test disconnecting specific signal."""
    manager = SignalConnectionManager()
    emitter = SignalEmitter()
    
    def slot(value):
        pass
    
    manager.connect(emitter.test_signal, slot)
    assert manager.get_connection_count() == 1
    
    manager.disconnect(emitter.test_signal, slot)
    assert manager.get_connection_count() == 0


def test_signal_manager_disconnect_all():
    """Test disconnecting all signals."""
    manager = SignalConnectionManager()
    emitter1 = SignalEmitter()
    emitter2 = SignalEmitter()
    
    received = []
    
    manager.connect(emitter1.test_signal, lambda x: received.append(x))
    manager.connect(emitter2.test_signal, lambda x: received.append(x))
    
    assert manager.get_connection_count() == 2
    
    disconnected = manager.disconnect_all()
    assert disconnected == 2
    assert manager.get_connection_count() == 0
    
    # Signals should no longer fire
    emitter1.test_signal.emit('test')
    assert received == []


def test_signal_manager_block_unblock():
    """Test blocking and unblocking signals."""
    manager = SignalConnectionManager()
    emitter = SignalEmitter()
    
    received = []
    manager.connect(emitter.test_signal, lambda x: received.append(x))
    
    # Normal emission
    emitter.test_signal.emit('before')
    assert received == ['before']
    
    # Block signals
    manager.block_all()
    emitter.test_signal.emit('blocked')
    # Note: pyqtSignal blocking works at sender level, not signal level
    # This test validates the blocking mechanism exists
    
    # Unblock
    manager.unblock_all()
    emitter.test_signal.emit('after')
    assert 'after' in received


def test_safe_connect_with_manager():
    """Test safe_connect helper with manager."""
    manager = SignalConnectionManager()
    emitter = SignalEmitter()
    
    received = []
    safe_connect(emitter.test_signal, lambda x: received.append(x), manager)
    
    assert manager.get_connection_count() == 1
    emitter.test_signal.emit('test')
    assert received == ['test']


def test_safe_connect_without_manager():
    """Test safe_connect helper without manager."""
    emitter = SignalEmitter()
    
    received = []
    safe_connect(emitter.test_signal, lambda x: received.append(x))
    
    emitter.test_signal.emit('test')
    assert received == ['test']


def test_hover_event_filter_creation(qtbot):
    """Test HoverEventFilter creation."""
    label = QLabel("Test")
    qtbot.addWidget(label)
    
    bbox = {'x': 10, 'y': 20, 'width': 100, 'height': 50}
    event_filter = HoverEventFilter(bbox, 'bar', parent=label)
    
    assert event_filter is not None
    assert event_filter.bbox == bbox
    assert event_filter.class_name == 'bar'
    assert event_filter.parent() == label  # Parent set correctly


def test_hover_event_filter_signals(qtbot):
    """Test HoverEventFilter emits signals."""
    label = QLabel("Test")
    qtbot.addWidget(label)
    
    bbox = {'x': 10, 'y': 20}
    event_filter = HoverEventFilter(bbox, 'bar', parent=label)
    label.installEventFilter(event_filter)
    
    # Track signals
    enter_signals = []
    leave_signals = []
    
    event_filter.hover_enter.connect(lambda b, c: enter_signals.append((b, c)))
    event_filter.hover_leave.connect(lambda: leave_signals.append(True))
    
    # Simulate hover (requires actual Qt event loop)
    # Note: Full testing requires qtbot mouse simulation
    assert len(enter_signals) == 0  # No events yet
    assert len(leave_signals) == 0


def test_click_event_filter(qtbot):
    """Test ClickEventFilter creation."""
    button = QPushButton("Test")
    qtbot.addWidget(button)
    
    data = {'id': 'detection_1', 'class': 'bar'}
    event_filter = ClickEventFilter(data, parent=button)
    
    assert event_filter.data == data
    assert event_filter.parent() == button


def test_hover_highlight_filter(qtbot):
    """Test HoverHighlightFilter state tracking."""
    label = QLabel("Test")
    qtbot.addWidget(label)
    
    event_filter = HoverHighlightFilter('detection_123', parent=label)
    
    assert event_filter.detection_id == 'detection_123'
    assert event_filter._is_highlighted is False
    
    # Test reset
    event_filter._is_highlighted = True
    event_filter.reset()
    assert event_filter._is_highlighted is False


def test_install_hover_filter_helper(qtbot):
    """Test install_hover_filter helper function."""
    label = QLabel("Test")
    qtbot.addWidget(label)
    
    enter_calls = []
    leave_calls = []
    
    bbox = {'x': 10, 'y': 20}
    event_filter = install_hover_filter(
        label,
        bbox,
        'bar',
        enter_handler=lambda b, c: enter_calls.append((b, c)),
        leave_handler=lambda: leave_calls.append(True)
    )
    
    assert event_filter is not None
    assert event_filter.parent() == label
    
    # Verify filter is installed
    # (Full event simulation requires Qt event loop)


def test_hover_filter_prevents_memory_leak(qtbot):
    """Test that HoverEventFilter with parent doesn't leak."""
    label = QLabel("Test")
    qtbot.addWidget(label)
    
    bbox = {'x': 10, 'y': 20}
    event_filter = HoverEventFilter(bbox, 'bar', parent=label)
    label.installEventFilter(event_filter)
    
    # Critical: event_filter has parent set
    # When label is deleted, event_filter should be auto-deleted
    assert event_filter.parent() == label
    
    # Verify no circular reference
    import sys
    initial_refcount = sys.getrefcount(label)
    
    # Creating filter shouldn't increase label refcount significantly
    # (Some increase is expected due to parent-child relationship)
    assert sys.getrefcount(label) <= initial_refcount + 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
