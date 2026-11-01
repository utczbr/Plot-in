# -*- coding: utf-8 -*-
"""
Tests for Canvas Shortcuts, Digit-Key Class Assignment, ChangeClassCommand OCR restore,
Mnemonic Escape, and Context Menu Cleanup (T-016.4, NF-16.11, T-013.2).
"""

from unittest.mock import MagicMock, patch
import pytest
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QContextMenuEvent, QKeyEvent
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox, QPushButton, QWidget

from visual.canvas_editor_controller import CanvasEditorMixin, ChangeClassCommand
from visual.detection_editor_state import EditorStateManager
from visual.detection_scene import DEFAULT_COLORS, DetectionCanvasView, DetectionScene, EditableRectItem, EditorMode


class DummyEditorHost(CanvasEditorMixin, QWidget):
    """Host widget for testing CanvasEditorMixin canvas interactions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._det_scene = DetectionScene()
        self._det_scene.setSceneRect(QRectF(0, 0, 800, 600))
        self._det_canvas_view = DetectionCanvasView(self._det_scene)
        self._editor_state = EditorStateManager()
        self._editor_toolbar = MagicMock()
        self._editor_toolbar._current_mode = EditorMode.EDIT_BOXES
        self._editor_toolbar.sync_class_to_selection = MagicMock()
        self._editor_toolbar._btn_apply = QPushButton("Apply & Re-Extract")
        self.current_analysis_result = {
            "chart_type": "bar",
            "detections": {},
            "protocol_rows": [],
        }
        self.analysis_results_widgets = {}
        self.analysis_thread = None
        self.is_processing = False
        self.current_image_path = "test.png"
        self.output_path_edit = MagicMock()
        self.output_path_edit.text.return_value = "out"
        self.models_dir_edit = MagicMock()
        self.models_dir_edit.text.return_value = "models"
        self.context = None
        self.advanced_settings = {}
        self.progress_bar = MagicMock()
        self.highlighted_bbox = None
        self.original_pil_image = None

    def update_status(self, msg):
        pass

    def _update_scene_display(self):
        pass

    def _on_analysis_complete(self, res):
        pass

    def statusBar(self):
        return MagicMock()


class TestDigitClassAssignmentT0164:
    """T-016.4: Chart-type-aware digit class assignment (1-9)."""

    def test_chart_type_classes_frequency_ordering(self):
        """
        Verify ADR-016 frequency order:
        Bar chart: 1 -> bar, 2 -> axis_labels, 3 -> chart_title, 4 -> axis_title, 5 -> legend
        """
        bar_classes = CanvasEditorMixin._CHART_TYPE_CLASSES["bar"]
        assert bar_classes[0] == "bar"
        assert bar_classes[1] == "axis_labels"
        assert bar_classes[2] == "chart_title"
        assert bar_classes[3] == "axis_title"
        assert bar_classes[4] == "legend"

        line_classes = CanvasEditorMixin._CHART_TYPE_CLASSES["line"]
        assert line_classes[0] == "data_point"
        assert line_classes[1] == "axis_labels"

        box_classes = CanvasEditorMixin._CHART_TYPE_CLASSES["box"]
        assert box_classes[0] == "box"
        assert box_classes[1] == "axis_labels"

    def test_rect_item_key_2_assigns_axis_labels_on_bar_chart(self, qapp):
        """
        ADR-016 Acceptance: Pressing '2' on a bar chart assigns 'axis_labels'.
        """
        scene = DetectionScene()
        scene.setSceneRect(QRectF(0, 0, 800, 600))
        # Configure scene with bar chart classes
        scene.set_available_classes(CanvasEditorMixin._CHART_TYPE_CLASSES["bar"])

        det = {"xyxy": [10, 10, 50, 100], "conf": 0.95, "text": "Bar 1"}
        item = EditableRectItem(xyxy=[10, 10, 50, 100], class_name="bar", detection=det)
        item.set_editor_mode(EditorMode.EDIT_BOXES)
        item.setSelected(True)
        scene.addItem(item)
        scene.add_rect_item(item)

        item_class_changed_spy = []
        scene.item_class_changed.connect(lambda it, cls: item_class_changed_spy.append((it, cls)))

        # Send Key_2 event to the item
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_2, Qt.KeyboardModifier.NoModifier)
        item.keyPressEvent(event)

        assert len(item_class_changed_spy) == 1
        emitted_item, new_class = item_class_changed_spy[0]
        assert emitted_item is item
        assert new_class == "axis_labels"

    def test_controller_routes_digit_keys_to_selected_item(self, qapp):
        """
        Pressing digit keys 1-9 on controller window routes to selected item.
        """
        host = DummyEditorHost()
        host._on_toolbar_chart_type_changed("bar")

        det = {"xyxy": [20, 20, 60, 120], "conf": 0.9, "text": "Header"}
        item = EditableRectItem(xyxy=[20, 20, 60, 120], class_name="bar", detection=det)
        host._det_scene.addItem(item)
        host._det_scene.add_rect_item(item)
        item.setSelected(True)

        # Press key 3 -> index 2 in bar chart is 'chart_title'
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_3, Qt.KeyboardModifier.NoModifier)
        with patch.object(host, "_re_ocr_detection_item"):
            host.keyPressEvent(event)

        assert item.class_name == "chart_title"
        assert host._editor_state.can_undo


class TestChangeClassCommandOCRRestoreT0164:
    """T-016.4: ChangeClassCommand preserves and restores OCR text across Undo/Redo."""

    def test_change_class_undo_restores_original_ocr_text(self, qapp):
        host = DummyEditorHost()
        det = {"xyxy": [30, 30, 80, 140], "conf": 0.88, "text": "Original Title", "ocr_confidence": 0.88}
        item = EditableRectItem(xyxy=[30, 30, 80, 140], class_name="bar", detection=det)
        host._det_scene.addItem(item)
        host._det_scene.add_rect_item(item)

        # Mock re_ocr to simulate OCR changing the text to "2024 Revenue"
        def mock_re_ocr(it):
            it.detection["text"] = "2024 Revenue"
            it.detection["ocr_confidence"] = 0.99

        with patch.object(host, "_re_ocr_detection_item", side_effect=mock_re_ocr):
            host._on_scene_item_class_changed(item, "chart_title")

        assert item.class_name == "chart_title"
        assert item.detection["text"] == "2024 Revenue"
        assert host._editor_toolbar.sync_class_to_selection.called

        # Undo class change
        host._on_editor_undo()

        # Both class and original OCR text must be fully restored!
        assert item.class_name == "bar"
        assert item.detection["text"] == "Original Title"
        assert item.detection["ocr_confidence"] == 0.88

        # Redo class change
        host._on_editor_redo()
        assert item.class_name == "chart_title"
        assert item.detection["text"] == "2024 Revenue"
        assert item.detection["ocr_confidence"] == 0.99


class TestContextMenuCleanupT0132:
    """T-013.2: Context menu X before Y, and actionable uncalibrated entry."""

    def test_context_menu_order_x_before_y(self, qapp):
        scene = DetectionScene()
        scene.setSceneRect(QRectF(0, 0, 800, 600))
        view = DetectionCanvasView(scene)

        # Intercept QMenu.exec to inspect action order
        recorded_actions = []

        def mock_exec(menu_self, pos=None):
            recorded_actions.extend([a.text() for a in menu_self.actions()])
            return None

        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(100, 100), QPoint(100, 100))
        with patch.object(QMenu, "exec", mock_exec):
            view.contextMenuEvent(event)

        assert "Copy Calculated X" in recorded_actions
        assert "Copy Calculated Y" in recorded_actions

        idx_x = recorded_actions.index("Copy Calculated X")
        idx_y = recorded_actions.index("Copy Calculated Y")
        assert idx_x < idx_y, f"Copy Calculated X (idx={idx_x}) must appear before Copy Calculated Y (idx={idx_y})"

    def test_uncalibrated_menu_has_placeholder_action_and_disabled_copy(self, qapp):
        scene = DetectionScene()
        scene.setSceneRect(QRectF(0, 0, 800, 600))
        view = DetectionCanvasView(scene)

        actions_captured = []

        def mock_exec(menu_self, pos=None):
            for a in menu_self.actions():
                actions_captured.append((a.text(), a.isEnabled()))
            return None

        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(150, 150), QPoint(150, 150))
        with patch.object(QMenu, "exec", mock_exec):
            view.contextMenuEvent(event)

        action_dict = dict(actions_captured)

        # Must contain actionable disabled placeholder
        assert "Not calibrated — Calibrate axes…" in action_dict
        assert action_dict["Not calibrated — Calibrate axes…"] is False

        # Copy actions must be disabled and not copy fake 0.000
        assert action_dict.get("Copy Calculated X") is False
        assert action_dict.get("Copy Calculated Y") is False


class TestMnemonicEscapeNF1611:
    """NF-16.11: Escape mnemonic 'Apply && Re-Extract' to render literal &."""

    def test_format_apply_button_label_escapes_ampersand(self):
        escaped = CanvasEditorMixin.format_apply_button_label("Apply & Re-Extract")
        assert escaped == "Apply && Re-Extract"

        # Already escaped should not double-escape
        assert CanvasEditorMixin.format_apply_button_label("Apply && Re-Extract") == "Apply && Re-Extract"

    def test_create_apply_confirmation_dialog_has_escaped_mnemonic(self, qapp):
        host = DummyEditorHost()
        dialog = host.create_apply_confirmation_dialog(total_edits=3)

        btn_texts = [b.text() for b in dialog.buttons()]
        assert "Apply && Re-Extract" in btn_texts
        # Must not have single unescaped ampersand
        assert "Apply & Re-Extract" not in btn_texts
