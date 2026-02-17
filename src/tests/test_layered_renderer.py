"""
Unit tests for LayeredRenderer

Tests for ui/rendering/layered_renderer.py - GPU-accelerated rendering.
"""

import pytest
import numpy as np
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from ui.rendering.layered_renderer import LayeredRenderer, LayerType, Layer


def create_test_array(width: int, height: int, channels: int = 3) -> np.ndarray:
    """Create test numpy array."""
    if channels == 1:
        return np.random.randint(0, 255, (height, width), dtype=np.uint8)
    else:
        return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)


def test_layered_renderer_initialization():
    """Test LayeredRenderer initializes correctly."""
    renderer = LayeredRenderer(target_fps=60)
    assert renderer is not None
    
    # Check all layers exist
    for layer_type in LayerType:
        assert layer_type in renderer._layers


def test_numpy_to_qimage_rgb():
    """Test direct numpy → QImage conversion for RGB."""
    renderer = LayeredRenderer()
    
    # Create RGB array
    array = create_test_array(100, 50, channels=3)
    qimage = renderer._numpy_to_qimage(array)
    
    assert qimage.width() == 100
    assert qimage.height() == 50
    assert qimage.format() == QImage.Format.Format_RGB888


def test_numpy_to_qimage_grayscale():
    """Test direct numpy → QImage conversion for grayscale."""
    renderer = LayeredRenderer()
    
    # Create grayscale array
    array = create_test_array(100, 50, channels=1)
    qimage = renderer._numpy_to_qimage(array)
    
    assert qimage.width() == 100
    assert qimage.height() == 50
    assert qimage.format() == QImage.Format.Format_Grayscale8


def test_set_base_image():
    """Test setting base image."""
    renderer = LayeredRenderer()
    
    array = create_test_array(200, 150, channels=3)
    renderer.set_base_image(array)
    
    assert renderer._base_image_size == (200, 150)
    assert renderer._layers[LayerType.BASE_IMAGE].pixmap is not None
    assert not renderer._layers[LayerType.BASE_IMAGE].dirty


def test_layer_visibility():
    """Test toggling layer visibility."""
    renderer = LayeredRenderer()
    
    # Initially visible
    assert renderer._layers[LayerType.DETECTIONS].visible is True
    
    # Toggle off
    renderer.set_layer_visible(LayerType.DETECTIONS, False)
    assert renderer._layers[LayerType.DETECTIONS].visible is False
    
    # Toggle on
    renderer.set_layer_visible(LayerType.DETECTIONS, True)
    assert renderer._layers[LayerType.DETECTIONS].visible is True


def test_layer_opacity():
    """Test setting layer opacity with clamping."""
    renderer = LayeredRenderer()
    
    # Valid opacity
    renderer.set_layer_opacity(LayerType.DETECTIONS, 0.5)
    assert renderer._layers[LayerType.DETECTIONS].opacity == 0.5
    
    # Clamp to max
    renderer.set_layer_opacity(LayerType.DETECTIONS, 1.5)
    assert renderer._layers[LayerType.DETECTIONS].opacity == 1.0
    
    # Clamp to min
    renderer.set_layer_opacity(LayerType.DETECTIONS, -0.5)
    assert renderer._layers[LayerType.DETECTIONS].opacity == 0.0


def test_dirty_flags():
    """Test dirty flag system."""
    renderer = LayeredRenderer()
    
    array = create_test_array(100, 100, channels=3)
    renderer.set_base_image(array)
    
    # Base layer should not be dirty
    assert not renderer._layers[LayerType.BASE_IMAGE].dirty
    
    # Mark layer dirty
    renderer.mark_layer_dirty(LayerType.DETECTIONS)
    assert renderer._layers[LayerType.DETECTIONS].dirty


def test_render_basic():
    """Test basic rendering."""
    renderer = LayeredRenderer()
    
    # Set base image
    array = create_test_array(200, 150, channels=3)
    renderer.set_base_image(array)
    
    # Render
    pixmap = renderer.render()
    
    assert pixmap.width() == 200
    assert pixmap.height() == 150
    assert renderer._render_count == 1


def test_render_caching():
    """Test render caching (no changes = use cache)."""
    renderer = LayeredRenderer()
    
    array = create_test_array(100, 100, channels=3)
    renderer.set_base_image(array)
    
    # First render
    pixmap1 = renderer.render()
    render_count1 = renderer._render_count
    
    # Second render (should use cache)
    pixmap2 = renderer.render()
    
    assert pixmap1 is pixmap2  # Same object (cached)
    assert renderer._cache_hits == 1


def test_render_invalidation():
    """Test cache invalidation on changes."""
    renderer = LayeredRenderer()
    
    array = create_test_array(100, 100, channels=3)
    renderer.set_base_image(array)
    
    # First render
    pixmap1 = renderer.render()
    
    # Change visibility (invalidates cache)
    renderer.set_layer_visible(LayerType.GRID, False)
    
    # Second render (should re-render)
    pixmap2 = renderer.render()
    
    assert renderer._render_count == 2  # Not cached


def test_layer_update():
    """Test updating layer content."""
    renderer = LayeredRenderer()
    
    array = create_test_array(100, 100, channels=3)
    renderer.set_base_image(array)
    
    # Create overlay pixmap
    overlay = QPixmap(100, 100)
    overlay.fill(Qt.GlobalColor.red)
    
    # Update layer
    renderer.update_layer(LayerType.DETECTIONS, overlay)
    
    assert renderer._layers[LayerType.DETECTIONS].pixmap == overlay
    assert not renderer._layers[LayerType.DETECTIONS].dirty


def test_render_stats():
    """Test rendering statistics tracking."""
    renderer = LayeredRenderer()
    
    array = create_test_array(100, 100, channels=3)
    renderer.set_base_image(array)
    
    # Perform multiple renders
    for _ in range(5):
        renderer.render()
    
    stats = renderer.get_render_stats()
    
    assert stats['total_renders'] >= 5
    assert 'avg_render_time_ms' in stats
    assert 'actual_fps' in stats
    assert stats['base_image_size'] == (100, 100)


def test_reset_stats():
    """Test statistics reset."""
    renderer = LayeredRenderer()
    
    array = create_test_array(100, 100, channels=3)
    renderer.set_base_image(array)
    renderer.render()
    
    # Reset
    renderer.reset_stats()
    
    assert renderer._render_count == 0
    assert renderer._total_render_time_ms == 0
    assert renderer._cache_hits == 0


def test_clear():
    """Test clearing renderer."""
    renderer = LayeredRenderer()
    
    array = create_test_array(100, 100, channels=3)
    renderer.set_base_image(array)
    renderer.render()
    
    # Clear
    renderer.clear()
    
    assert renderer._base_image_array is None
    assert renderer._base_image_size == (0, 0)
    assert renderer._cached_result is None


def test_layer_z_index():
    """Test layers are rendered in z-index order."""
    renderer = LayeredRenderer()
    
    # Verify z-index ordering
    z_indices = [
        renderer._layers[layer_type].z_index
        for layer_type in LayerType
    ]
    
    # Should be in ascending order (base at bottom, annotations on top)
    assert z_indices == sorted(z_indices)


def test_empty_render():
    """Test rendering without base image."""
    renderer = LayeredRenderer()
    
    # Should return empty pixmap
    pixmap = renderer.render()
    assert pixmap.isNull()


def test_layer_dataclass():
    """Test Layer dataclass."""
    layer = Layer(z_index=5, opacity=0.7, visible=False)
    
    assert layer.z_index == 5
    assert layer.opacity == 0.7
    assert layer.visible is False
    assert layer.dirty is True  # Default
    
    # Test dirty flag methods
    layer.clear_dirty()
    assert layer.dirty is False
    
    layer.mark_dirty()
    assert layer.dirty is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
