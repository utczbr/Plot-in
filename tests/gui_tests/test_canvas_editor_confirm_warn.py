"""
Tests for T-008.3 (Confirm Reset) and T-008.4 (Warn before Apply discards text/table edits).
"""
from unittest.mock import MagicMock, patch
import pytest
from PyQt6.QtWidgets import QWidget, QMessageBox, QLineEdit

from visual.canvas_editor_controller import CanvasEditorMixin


class DummyController(CanvasEditorMixin, QWidget):
    """Minimal host widget implementing CanvasEditorMixin for testing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._det_scene = MagicMock()
        self._det_scene.get_incomplete_slices.return_value = []
        self._det_canvas_view = MagicMock()
        self._editor_toolbar = MagicMock()
        self._editor_state = MagicMock()
        self.current_analysis_result = {}
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

    def update_status(self, msg):
        pass

    def _update_scene_display(self):
        pass

    def _on_analysis_complete(self, res):
        pass

    def statusBar(self):
        sb = MagicMock()
        return sb


class TestConfirmResetT0083:
    """T-008.3: _on_editor_reset asks confirmation when dirty; Cancel retains state."""

    def test_reset_dirty_boxes_cancelled_preserves_state(self, qapp):
        ctrl = DummyController()
        ctrl._editor_state.is_dirty = True
        ctrl._editor_state.edit_count = 3

        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel) as mock_q:
            ctrl._on_editor_reset()
            mock_q.assert_called_once()
            args, _ = mock_q.call_args
            assert "Discard 3 box edits and clear undo history?" in args[2]
            ctrl._editor_state.clear.assert_not_called()

    def test_reset_clean_boxes_no_prompt(self, qapp):
        ctrl = DummyController()
        ctrl._editor_state.is_dirty = False
        ctrl._editor_state.edit_count = 0

        with patch.object(QMessageBox, "question") as mock_q:
            ctrl._on_editor_reset()
            mock_q.assert_not_called()
            ctrl._editor_state.clear.assert_called_once()

    def test_reset_dirty_boxes_confirmed_clears_state(self, qapp):
        ctrl = DummyController()
        ctrl._editor_state.is_dirty = True
        ctrl._editor_state.edit_count = 5

        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard) as mock_q:
            ctrl._on_editor_reset()
            mock_q.assert_called_once()
            ctrl._editor_state.clear.assert_called_once()


class TestWarnBeforeApplyT0084:
    """T-008.4: Warn before Apply discards text and Results-table edits."""

    def test_apply_warns_on_results_table_edits(self, qapp):
        ctrl = DummyController()
        ctrl.current_analysis_result = {
            "chart_type": "bar",
            "protocol_rows": [
                {"review_status": "corrected", "_original": {"value": 10.0, "label": "A"}, "value": 12.0, "label": "B"},
            ],
        }
        ctrl.analysis_results_widgets = {}

        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel) as mock_q:
            ctrl._on_editor_apply()
            mock_q.assert_called_once()
            args, _ = mock_q.call_args
            assert "2 cell and text edits will be replaced" in args[2]
            assert ctrl.analysis_thread is None

    def test_apply_warns_on_ocr_text_edits(self, qapp):
        ctrl = DummyController()
        ctrl.current_analysis_result = {"chart_type": "bar", "protocol_rows": []}
        entry = QLineEdit("Edited Text")
        ctrl.analysis_results_widgets = {
            "w1": {
                "entry": entry,
                "original_item": {"text": "Original Text"},
                "type": "text",
            }
        }

        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel) as mock_q:
            ctrl._on_editor_apply()
            mock_q.assert_called_once()
            args, _ = mock_q.call_args
            assert "1 cell and text edits will be replaced" in args[2]
            assert ctrl.analysis_thread is None

    def test_apply_no_prompt_when_clean(self, qapp):
        ctrl = DummyController()
        ctrl.current_analysis_result = {
            "chart_type": "bar",
            "protocol_rows": [
                {"review_status": "auto", "_original": None, "value": 10.0},
            ],
        }
        entry = QLineEdit("Unchanged Text")
        ctrl.analysis_results_widgets = {
            "w1": {
                "entry": entry,
                "original_item": {"text": "Unchanged Text"},
                "type": "text",
            }
        }
        ctrl._det_scene.export_detections.return_value = {"bar": [{"xyxy": [10, 10, 50, 50]}]}

        with patch.object(QMessageBox, "question") as mock_q, \
             patch("visual.canvas_editor_controller.ModernAnalysisThread") as mock_thread:
            instance = MagicMock()
            mock_thread.return_value = instance
            ctrl._on_editor_apply()
            mock_q.assert_not_called()
            assert ctrl.analysis_thread is not None
