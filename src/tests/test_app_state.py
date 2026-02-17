"""
Unit tests for AppState and StateManager

Tests for core/app_state.py - immutable state management.
"""

import pytest
from pathlib import Path
from PyQt6.QtCore import QObject, QRect
from core.app_state import (
    VisualizationState, CanvasState, AppState, StateManager
)


def test_visualization_state_immutability():
    """Test VisualizationState is truly immutable."""
    state = VisualizationState()
    
    # Should raise FrozenInstanceError if trying to modify
    with pytest.raises(Exception):  # dataclass FrozenInstanceError
        state.show_baseline = False


def test_visualization_state_class_visibility():
    """Test class visibility toggling creates new state."""
    state = VisualizationState()
    
    # Add class
    new_state = state.with_class_visibility('bar', True)
    assert 'bar' in new_state.visible_classes
    assert 'bar' not in state.visible_classes  # Original unchanged
    
    # Remove class
    newer_state = new_state.with_class_visibility('bar', False)
    assert 'bar' not in newer_state.visible_classes


def test_visualization_state_opacity():
    """Test opacity updates with validation."""
    state = VisualizationState()
    
    # Valid opacity
    new_state = state.with_opacity('detections', 0.5)
    assert new_state.opacity_map['detections'] == 0.5
    
    # Opacity clamping
    clamped = state.with_opacity('detections', 1.5)
    assert clamped.opacity_map['detections'] == 1.0  # Clamped to max
    
    clamped = state.with_opacity('detections', -0.5)
    assert clamped.opacity_map['detections'] == 0.0  # Clamped to min


def test_canvas_state_zoom():
    """Test zoom operations with validation."""
    state = CanvasState()
    
    # Zoom by factor
    zoomed = state.zoom_by(2.0)
    assert zoomed.zoom_level == 2.0
    
    # Zoom clamping (max 10.0)
    zoomed = state.zoom_by(20.0)
    assert zoomed.zoom_level == 10.0
    
    # Zoom to absolute level
    zoomed = state.zoom_to(0.5)
    assert zoomed.zoom_level == 0.5


def test_canvas_state_pan():
    """Test pan operations."""
    state = CanvasState()
    
    # Pan by delta
    panned = state.pan(10.0, 20.0)
    assert panned.pan_offset == (10.0, 20.0)
    
    # Incremental pan
    panned = panned.pan(5.0, -10.0)
    assert panned.pan_offset == (15.0, 10.0)
    
    # Reset pan
    reset = panned.reset_pan()
    assert reset.pan_offset == (0.0, 0.0)


def test_app_state_immutability():
    """Test AppState is truly immutable."""
    state = AppState()
    
    with pytest.raises(Exception):
        state.current_image_path = Path('/test')


def test_app_state_computed_properties():
    """Test computed properties work correctly."""
    state = AppState()
    
    # No image
    assert state.has_image is False
    assert state.has_analysis is False
    
    # With image
    state_with_image = state.with_update(current_image_path=Path('/test.png'))
    assert state_with_image.has_image is True
    
    # With analysis
    state_with_analysis = state.with_update(current_analysis={'detections': []})
    assert state_with_analysis.has_analysis is True


def test_app_state_navigation():
    """Test navigation state checks."""
    state = AppState(
        image_files=('img1.png', 'img2.png', 'img3.png'),
        current_image_index=1
    )
    
    assert state.can_navigate_prev is True
    assert state.can_navigate_next is True
    
    # At beginning
    state_at_start = state.with_update(current_image_index=0)
    assert state_at_start.can_navigate_prev is False
    assert state_at_start.can_navigate_next is True
    
    # At end
    state_at_end = state.with_update(current_image_index=2)
    assert state_at_end.can_navigate_prev is True
    assert state_at_end.can_navigate_next is False


def test_app_state_serialization():
    """Test state can be serialized and deserialized."""
    state = AppState(
        current_image_path=Path('/test.png'),
        current_image_index=5,
        canvas=CanvasState(zoom_level=1.5, pan_offset=(10.0, 20.0))
    )
    
    # Serialize
    data = state.to_dict()
    assert data['current_image_path'] == '/test.png'
    assert data['zoom_level'] == 1.5
    assert data['pan_offset'] == (10.0, 20.0)
    
    # Deserialize
    restored = AppState.from_dict(data)
    assert restored.current_image_path == Path('/test.png')
    assert restored.canvas.zoom_level == 1.5


def test_state_manager_initialization():
    """Test StateManager initializes correctly."""
    manager = StateManager()
    
    state = manager.get_state()
    assert isinstance(state, AppState)
    assert manager.can_undo() is False
    assert manager.can_redo() is False


def test_state_manager_update():
    """Test state updates work and emit signals."""
    manager = StateManager()
    
    # Track signal emissions
    emitted_states = []
    
    def on_state_changed(state):
        emitted_states.append(state)
    
    manager.state_changed.connect(on_state_changed)
    
    # Update state
    manager.update(current_image_index=5, is_processing=True)
    
    # Signal should be emitted
    assert len(emitted_states) == 1
    assert emitted_states[0].current_image_index == 5
    assert emitted_states[0].is_processing is True


def test_state_manager_undo_redo():
    """Test undo/redo functionality."""
    manager = StateManager()
    
    # Make changes
    manager.update(current_image_index=1)
    manager.update(current_image_index=2)
    manager.update(current_image_index=3)
    
    assert manager.get_state().current_image_index == 3
    
    # Undo twice
    assert manager.undo() is True
    assert manager.get_state().current_image_index == 2
    
    assert manager.undo() is True
    assert manager.get_state().current_image_index == 1
    
    # Redo once
    assert manager.redo() is True
    assert manager.get_state().current_image_index == 2
    
    # Make new change (should truncate redo history)
    manager.update(current_image_index=10)
    assert manager.can_redo() is False


def test_state_manager_undo_limits():
    """Test undo cannot go beyond beginning of history."""
    manager = StateManager()
    
    manager.update(current_image_index=1)
    
    # Undo to beginning
    assert manager.undo() is True
    
    # Cannot undo further
    assert manager.undo() is False


def test_state_manager_redo_limits():
    """Test redo cannot go beyond end of history."""
    manager = StateManager()
    
    manager.update(current_image_index=1)
    manager.undo()
    
    # Redo to end
    assert manager.redo() is True
    
    # Cannot redo further
    assert manager.redo() is False


def test_state_manager_history_limit():
    """Test history is limited to max_history states."""
    manager = StateManager()
    manager._max_history = 5  # Set low for testing
    
    # Add more states than max
    for i in range(10):
        manager.update(current_image_index=i)
    
    # History should be capped
    history_info = manager.get_history_info()
    assert history_info['total_states'] == 5


def test_state_manager_replace_state():
    """Test replacing entire state."""
    manager = StateManager()
    
    new_state = AppState(
        current_image_index=100,
        is_processing=True
    )
    
    manager.replace_state(new_state)
    
    assert manager.get_state().current_image_index == 100
    assert manager.get_state().is_processing is True


def test_state_manager_clear_history():
    """Test clearing undo/redo history."""
    manager = StateManager()
    
    manager.update(current_image_index=1)
    manager.update(current_image_index=2)
    
    assert manager.can_undo() is True
    
    manager.clear_history()
    
    assert manager.can_undo() is False
    assert manager.get_history_info()['total_states'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
