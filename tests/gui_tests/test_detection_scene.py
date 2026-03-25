# -*- coding: utf-8 -*-
"""
Tests for the DetectionScene and DetectionCanvasView components.

These tests verify:
- Scene items are created at original-pixel coordinates
- export_detections() returns original-space coords regardless of view zoom
- Visibility toggles work correctly
- Highlight/unhighlight preserves pen width and z-value semantics
- Baseline item is managed correctly
"""

import pytest
import sys
import os

# Ensure src is on sys.path
_src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

# Guard: skip entire module if PyQt6 is not available for headless CI
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

# Need a QApplication instance for any QGraphics* work
_app = QApplication.instance() or QApplication(sys.argv)

from visual.detection_scene import (
    DetectionScene,
    DetectionCanvasView,
    EditableRectItem,
    BaselineItem,
    EditorMode,
)


# ── Fixtures ──

def _make_pixmap(w=400, h=300):
    """Create a simple test pixmap."""
    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(Qt.GlobalColor.white)
    return QPixmap.fromImage(img)


def _make_detections():
    """Standard test detections dict."""
    return {
        'bar': [
            {'xyxy': [10, 20, 50, 100], 'conf': 0.95, 'cls': 0},
            {'xyxy': [60, 20, 100, 100], 'conf': 0.90, 'cls': 0},
        ],
        'chart_title': [
            {'xyxy': [100, 5, 300, 25], 'conf': 0.99, 'cls': 1},
        ],
        'scale_label': [
            {'xyxy': [0, 50, 8, 60], 'conf': 0.80, 'cls': 2},
        ],
    }


# ── Tests ──

class TestDetectionScene:
    """Unit tests for DetectionScene."""

    def test_load_image_sets_scene_rect(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap(400, 300))
        assert scene.sceneRect().width() == 400
        assert scene.sceneRect().height() == 300

    def test_set_detections_creates_rect_items(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        scene.set_detections(_make_detections())

        # 2 bars + 1 title + 1 scale_label = 4 items + 1 base pixmap
        rect_count = len(scene._rect_items)
        assert rect_count == 4, f"Expected 4 rect items, got {rect_count}"

    def test_export_detections_returns_original_coords(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        dets = _make_detections()
        scene.set_detections(dets)

        exported = scene.export_detections()

        # bar[0] should have exact same xyxy
        assert 'bar' in exported
        assert len(exported['bar']) == 2
        assert exported['bar'][0]['xyxy'] == [10.0, 20.0, 50.0, 100.0]

    def test_export_invariant_under_zoom(self):
        """export_detections() MUST return original coords regardless of view zoom."""
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        scene.set_detections(_make_detections())

        view = DetectionCanvasView(scene)
        view.set_zoom(3.5)  # Zoom to 350%

        exported = scene.export_detections()
        assert exported['bar'][0]['xyxy'] == [10.0, 20.0, 50.0, 100.0], \
            "Coordinates must be in original pixel space regardless of view zoom"

    def test_set_class_visible_hides_items(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        scene.set_detections(_make_detections())

        scene.set_class_visible('bar', False)

        bar_items = [item for item in scene._rect_items if item.class_name == 'bar']
        assert all(not item.isVisible() for item in bar_items)

        # Other classes should still be visible
        title_items = [item for item in scene._rect_items if item.class_name == 'chart_title']
        assert all(item.isVisible() for item in title_items)

    def test_set_class_visible_shows_items(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        scene.set_detections(_make_detections())

        scene.set_class_visible('bar', False)
        scene.set_class_visible('bar', True)

        bar_items = [item for item in scene._rect_items if item.class_name == 'bar']
        assert all(item.isVisible() for item in bar_items)

    def test_highlight_and_clear(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        scene.set_detections(_make_detections())

        bbox = [10, 20, 50, 100]
        scene.highlight_item_by_bbox(bbox, 'bar')

        assert scene._highlighted_item is not None
        assert scene._highlighted_item.class_name == 'bar'
        # Highlighted item should have boosted z-value
        assert scene._highlighted_item.zValue() > 50

        scene.clear_highlight()
        assert scene._highlighted_item is None

    def test_baseline_visibility_toggle(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        scene.set_baseline(150.0, "horizontal")

        assert scene._baseline_item is not None
        assert scene._baseline_item.isVisible()

        scene.set_class_visible('baseline', False)
        assert not scene._baseline_item.isVisible()

        scene.set_class_visible('baseline', True)
        assert scene._baseline_item.isVisible()

    def test_empty_detections(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        scene.set_detections({})

        assert len(scene._rect_items) == 0
        exported = scene.export_detections()
        assert exported == {}

    def test_detections_without_xyxy_are_skipped(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        scene.set_detections({
            'layout_text_regions': [
                {'text': 'some text', 'conf': 0.5},  # no xyxy
            ],
        })
        assert len(scene._rect_items) == 0

    def test_set_detections_replaces_previous(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        scene.set_detections(_make_detections())
        assert len(scene._rect_items) == 4

        scene.set_detections({'bar': [{'xyxy': [0, 0, 10, 10], 'conf': 0.5}]})
        assert len(scene._rect_items) == 1


class TestDetectionCanvasView:
    """Unit tests for DetectionCanvasView."""

    def test_zoom_in_out(self):
        scene = DetectionScene()
        scene.load_image(_make_pixmap())
        view = DetectionCanvasView(scene)

        initial = view.zoom_level
        view.zoom_in()
        assert view.zoom_level > initial

        view.zoom_out()
        view.zoom_out()
        assert view.zoom_level < initial

    def test_zoom_clamped(self):
        scene = DetectionScene()
        view = DetectionCanvasView(scene)

        view.set_zoom(100.0)
        assert view.zoom_level <= view.ZOOM_MAX

        view.set_zoom(0.001)
        assert view.zoom_level >= view.ZOOM_MIN

    def test_mode_switching(self):
        scene = DetectionScene()
        view = DetectionCanvasView(scene)

        view.set_mode(EditorMode.VIEW)
        assert view.mode == EditorMode.VIEW

        view.set_mode(EditorMode.EDIT_BOXES)
        assert view.mode == EditorMode.EDIT_BOXES


class TestEditableRectItem:
    """Unit tests for EditableRectItem."""

    def test_current_xyxy(self):
        det = {'xyxy': [10, 20, 50, 80], 'conf': 0.9}
        item = EditableRectItem([10, 20, 50, 80], 'bar', det)

        # Without adding to a scene, sceneBoundingRect = local rect
        xyxy = item.current_xyxy()
        assert abs(xyxy[0] - 10) < 1
        assert abs(xyxy[1] - 20) < 1
        assert abs(xyxy[2] - 50) < 1
        assert abs(xyxy[3] - 80) < 1

    def test_highlight_changes_pen_width(self):
        det = {'xyxy': [0, 0, 100, 100], 'conf': 0.9}
        item = EditableRectItem([0, 0, 100, 100], 'bar', det)

        assert item.pen().width() == 2  # normal
        item.set_highlighted(True)
        assert item.pen().width() == 3  # highlighted
        item.set_highlighted(False)
        assert item.pen().width() == 2  # back to normal
