# -*- coding: utf-8 -*-
"""
Tests for the DetectionEditorState — undo/redo commands and state management.
"""

import pytest
import sys
import os

_src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QRectF

_app = QApplication.instance() or QApplication(sys.argv)

from visual.detection_scene import DetectionScene, EditableRectItem
from visual.detection_editor_state import (
    EditorStateManager,
    MoveCommand,
    ResizeCommand,
    DeleteCommand,
    CreateCommand,
    ChangeClassCommand,
)

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt


def _make_scene_with_items():
    scene = DetectionScene()
    img = QImage(400, 300, QImage.Format.Format_RGB888)
    img.fill(Qt.GlobalColor.white)
    scene.load_image(QPixmap.fromImage(img))
    scene.set_detections({
        'bar': [
            {'xyxy': [10, 20, 50, 100], 'conf': 0.95},
            {'xyxy': [60, 20, 100, 100], 'conf': 0.90},
        ],
    })
    return scene


class TestEditorStateManager:
    def test_initial_state(self):
        mgr = EditorStateManager()
        assert not mgr.is_dirty
        assert not mgr.can_undo
        assert not mgr.can_redo
        assert mgr.edit_count == 0

    def test_push_and_undo(self):
        scene = _make_scene_with_items()
        mgr = EditorStateManager()
        item = scene._rect_items[0]

        old_xyxy = [10, 20, 50, 100]
        new_xyxy = [15, 25, 55, 105]
        cmd = MoveCommand(item, old_xyxy, new_xyxy)
        mgr.push(cmd)

        assert mgr.is_dirty
        assert mgr.can_undo
        assert mgr.edit_count == 1

        # Check that item was moved
        rect = item.rect()
        assert abs(rect.x() - 15) < 0.1

        # Undo
        mgr.undo()
        rect = item.rect()
        assert abs(rect.x() - 10) < 0.1
        assert not mgr.is_dirty

    def test_redo(self):
        scene = _make_scene_with_items()
        mgr = EditorStateManager()
        item = scene._rect_items[0]

        cmd = MoveCommand(item, [10, 20, 50, 100], [15, 25, 55, 105])
        mgr.push(cmd)
        mgr.undo()
        assert mgr.can_redo

        mgr.redo()
        rect = item.rect()
        assert abs(rect.x() - 15) < 0.1

    def test_clear(self):
        mgr = EditorStateManager()
        scene = _make_scene_with_items()
        item = scene._rect_items[0]
        mgr.push(MoveCommand(item, [10, 20, 50, 100], [15, 25, 55, 105]))
        assert mgr.is_dirty

        mgr.clear()
        assert not mgr.is_dirty
        assert not mgr.can_undo

    def test_status_text(self):
        scene = _make_scene_with_items()
        mgr = EditorStateManager()
        item0 = scene._rect_items[0]
        item1 = scene._rect_items[1]

        mgr.push(MoveCommand(item0, [10, 20, 50, 100], [15, 25, 55, 105]))
        mgr.push(DeleteCommand(scene, item1))

        status = mgr.get_status_text()
        assert "1 moved" in status
        assert "1 deleted" in status


class TestMoveCommand:
    def test_original_xyxy_snapshot(self):
        scene = _make_scene_with_items()
        item = scene._rect_items[0]
        assert 'original_xyxy' not in item.detection

        cmd = MoveCommand(item, [10, 20, 50, 100], [15, 25, 55, 105])
        assert item.detection['original_xyxy'] == [10, 20, 50, 100]

        # Second move should NOT overwrite original
        cmd2 = MoveCommand(item, [15, 25, 55, 105], [20, 30, 60, 110])
        assert item.detection['original_xyxy'] == [10, 20, 50, 100]


class TestDeleteCommand:
    def test_delete_and_undo(self):
        scene = _make_scene_with_items()
        assert len(scene._rect_items) == 2

        item = scene._rect_items[0]
        cmd = DeleteCommand(scene, item)
        cmd.redo()
        assert len(scene._rect_items) == 1

        cmd.undo()
        assert len(scene._rect_items) == 2


class TestCreateCommand:
    def test_create_and_undo(self):
        scene = _make_scene_with_items()
        assert len(scene._rect_items) == 2

        det = {'xyxy': [200, 200, 250, 250], 'conf': 1.0}
        new_item = EditableRectItem([200, 200, 250, 250], 'bar', det)
        cmd = CreateCommand(scene, new_item)
        cmd.redo()
        assert len(scene._rect_items) == 3

        cmd.undo()
        assert len(scene._rect_items) == 2


class TestChangeClassCommand:
    def test_change_class(self):
        scene = _make_scene_with_items()
        item = scene._rect_items[0]
        assert item.class_name == 'bar'

        from visual.detection_scene import DEFAULT_COLORS
        cmd = ChangeClassCommand(item, 'bar', 'scatter', DEFAULT_COLORS)
        cmd.redo()
        assert item.class_name == 'scatter'

        cmd.undo()
        assert item.class_name == 'bar'
