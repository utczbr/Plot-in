# -*- coding: utf-8 -*-
"""
Characterization and Integration Tests for Reversible Dirty Tracking and Unsaved-Edit Guards
(ADR-008 T-008.1, T-008.2).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QCloseEvent, QUndoCommand
from PyQt6.QtWidgets import QMessageBox, QPushButton

from main_modern import ModernChartAnalysisApp


class DummyUndoCommand(QUndoCommand):
    """Simple undo command for testing QUndoStack reversibility."""

    def __init__(self):
        super().__init__("Test Box Edit")

    def undo(self):
        pass

    def redo(self):
        pass


def test_box_undo_reversibility(offscreen_window: ModernChartAnalysisApp):
    """
    T-008.1: Box edits on canvas push undo commands; cleanChanged(isClean)
    wires reversibly to session.set_source_dirty("boxes", not isClean).
    """
    session = offscreen_window.session
    assert "boxes" not in session.dirty_sources
    assert session.is_dirty is False

    # Push a box edit command
    cmd = DummyUndoCommand()
    offscreen_window._editor_state.push(cmd)

    assert "boxes" in session.dirty_sources
    assert session.is_dirty is True

    # Undo back to clean index
    offscreen_window._editor_state.undo()

    assert "boxes" not in session.dirty_sources
    assert session.is_dirty is False


def test_numeric_parse_failure_does_not_mark_dirty_or_corrected(offscreen_window: ModernChartAnalysisApp):
    """
    T-008.1: Failed numeric parse in _on_protocol_cell_changed MUST NOT mark
    results_table dirty and MUST NOT flip review_status to 'corrected'.
    """
    rows = [
        {"outcome": "A", "group": "G1", "review_status": "auto", "value": 15.0},
    ]
    offscreen_window.current_analysis_result = {
        "chart_type": "bar",
        "protocol_rows": rows,
    }
    offscreen_window._populate_protocol_tab()

    session = offscreen_window.session
    session.mark_clean()
    assert session.is_dirty is False

    # Set invalid text with signals blocked, then invoke handler
    val_item = offscreen_window.protocol_table.item(0, 4)
    assert val_item is not None
    offscreen_window.protocol_table.blockSignals(True)
    val_item.setText("invalid_abc")
    offscreen_window.protocol_table.blockSignals(False)

    # Call handler to process user change
    offscreen_window._on_protocol_cell_changed(0, 4)

    # Must NOT be marked dirty, and review_status must NOT be 'corrected'
    assert "results_table" not in session.dirty_sources
    assert session.is_dirty is False
    assert rows[0]["review_status"] == "auto"
    status_item = offscreen_window.protocol_table.item(0, 9)
    assert status_item is not None and status_item.text() != "corrected"


def test_valid_results_table_edit_marks_dirty_and_corrected(offscreen_window: ModernChartAnalysisApp):
    """
    T-008.1: Valid edit in Results table marks results_table dirty and flips status to 'corrected'.
    """
    rows = [
        {"outcome": "A", "group": "G1", "review_status": "auto", "value": 15.0},
    ]
    offscreen_window.current_analysis_result = {
        "chart_type": "bar",
        "protocol_rows": rows,
    }
    offscreen_window._populate_protocol_tab()

    session = offscreen_window.session
    session.mark_clean()
    assert session.is_dirty is False

    val_item = offscreen_window.protocol_table.item(0, 4)
    assert val_item is not None
    val_item.setText("42.5")
    offscreen_window._on_protocol_cell_changed(0, 4)

    assert "results_table" in session.dirty_sources
    assert session.is_dirty is True
    assert rows[0]["value"] == 42.5
    assert rows[0]["review_status"] == "corrected"


def test_data_table_edit_marks_dirty(offscreen_window: ModernChartAnalysisApp):
    """
    T-008.1: Editing cell in Data tab marks data_table dirty.
    """
    offscreen_window.current_analysis_result = {
        "chart_type": "bar",
        "data_tab_model": {
            "columns": ["Label", "Value"],
            "rows": [["Control", "10.0"]],
        },
    }
    offscreen_window.data_tab_bindings[(0, 1)] = {"type": "bar", "index": 0, "field": "value"}
    offscreen_window.session.mark_clean()

    item = MagicMock()
    item.text.return_value = "12.5"
    offscreen_window.data_table = MagicMock()
    offscreen_window.data_table.item.return_value = item

    offscreen_window._on_data_cell_changed(0, 1)

    assert "data_table" in offscreen_window.session.dirty_sources
    assert offscreen_window.session.is_dirty is True


def test_ocr_text_edit_reversibility(offscreen_window: ModernChartAnalysisApp):
    """
    T-008.1: User text edits in OCR tab mark ocr_text dirty, and reverting to original marks clean.
    """
    entry = MagicMock()
    entry.text.return_value = "Modified Title"

    offscreen_window.analysis_results_widgets = {
        "w1": {
            "entry": entry,
            "original_item": {"text": "Original Title"},
            "type": "text",
        }
    }
    offscreen_window.session.mark_clean()

    # User edited text -> dirty
    offscreen_window._on_ocr_text_edited()
    assert "ocr_text" in offscreen_window.session.dirty_sources
    assert offscreen_window.session.is_dirty is True

    # User restored original text -> clean
    entry.text.return_value = "Original Title"
    offscreen_window._on_ocr_text_edited()
    assert "ocr_text" not in offscreen_window.session.dirty_sources
    assert offscreen_window.session.is_dirty is False


def test_navigation_clean_triggers_no_dialog(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    T-008.2: When session is clean, image navigation triggers zero dialog prompts.
    """
    dialog_called = False

    def mock_question(*args, **kwargs):
        nonlocal dialog_called
        dialog_called = True
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    offscreen_window.image_files = ["/tmp/img0.png", "/tmp/img1.png"]
    offscreen_window.current_image_index = 0
    offscreen_window.session.mark_clean()

    # Mock load helpers to avoid filesystem operations
    monkeypatch.setattr(offscreen_window, "cleanup_image_resources", lambda: None)
    monkeypatch.setattr(offscreen_window, "highlight_selected_file", lambda idx: None)
    monkeypatch.setattr(offscreen_window, "load_image_for_assisted_analysis", lambda: None)

    offscreen_window.load_image_by_index(1)

    assert dialog_called is False
    assert offscreen_window.current_image_index == 1


def test_navigation_box_dirty_prompts_discard_cancel(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    T-008.2: Box-dirty edits prompt Discard / Cancel (no Save).
    Cancel keeps reviewer on current image and restores selection highlight.
    Discard cleans box state and navigates.
    """
    buttons_passed = None

    def mock_question(parent, title, text, buttons, default_btn):
        nonlocal buttons_passed
        buttons_passed = buttons
        # First test Cancel
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    offscreen_window.image_files = ["/tmp/img0.png", "/tmp/img1.png"]
    offscreen_window.current_image_index = 0
    offscreen_window.session.set_source_dirty("boxes", True)

    highlight_calls = []
    monkeypatch.setattr(
        offscreen_window, "highlight_selected_file", lambda idx: highlight_calls.append(idx)
    )

    # 1. User attempts navigation to index 1, clicks Cancel
    offscreen_window.load_image_by_index(1)

    # Assert dialog buttons were Discard | Cancel (NO Save)
    assert QMessageBox.StandardButton.Discard in buttons_passed
    assert QMessageBox.StandardButton.Cancel in buttons_passed
    assert QMessageBox.StandardButton.Save not in buttons_passed

    # Navigation aborted, index preserved, selection highlight restored to index 0
    assert offscreen_window.current_image_index == 0
    assert highlight_calls == [0]
    assert offscreen_window.session.is_dirty is True

    # 2. User attempts navigation and clicks Discard
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Discard
    )
    monkeypatch.setattr(offscreen_window, "cleanup_image_resources", lambda: None)
    monkeypatch.setattr(offscreen_window, "load_image_for_assisted_analysis", lambda: None)

    offscreen_window.load_image_by_index(1)

    assert offscreen_window.current_image_index == 1
    assert offscreen_window.session.is_dirty is False


def test_navigation_table_dirty_save_success(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    T-008.2: Table/text dirty prompts Save / Discard / Cancel.
    Clicking Save calls _save_analysis_results, marks clean, and proceeds with navigation.
    """
    buttons_passed = None

    def mock_question(parent, title, text, buttons, default_btn):
        nonlocal buttons_passed
        buttons_passed = buttons
        return QMessageBox.StandardButton.Save

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    save_called = False

    def mock_save():
        nonlocal save_called
        save_called = True
        return True

    monkeypatch.setattr(offscreen_window, "_save_analysis_results", mock_save)
    monkeypatch.setattr(offscreen_window, "cleanup_image_resources", lambda: None)
    monkeypatch.setattr(offscreen_window, "highlight_selected_file", lambda idx: None)
    monkeypatch.setattr(offscreen_window, "load_image_for_assisted_analysis", lambda: None)

    offscreen_window.image_files = ["/tmp/img0.png", "/tmp/img1.png"]
    offscreen_window.current_image_index = 0
    offscreen_window.session.set_source_dirty("results_table", True)

    offscreen_window.load_image_by_index(1)

    assert QMessageBox.StandardButton.Save in buttons_passed
    assert QMessageBox.StandardButton.Discard in buttons_passed
    assert QMessageBox.StandardButton.Cancel in buttons_passed
    assert save_called is True
    assert offscreen_window.session.is_dirty is False
    assert offscreen_window.current_image_index == 1


def test_navigation_table_dirty_save_failure_stays_on_image(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    T-008.2: If Save fails (_save_analysis_results returns False), navigation aborts
    and user remains on current image.
    """
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Save
    )
    monkeypatch.setattr(offscreen_window, "_save_analysis_results", lambda: False)

    highlight_calls = []
    monkeypatch.setattr(
        offscreen_window, "highlight_selected_file", lambda idx: highlight_calls.append(idx)
    )

    offscreen_window.image_files = ["/tmp/img0.png", "/tmp/img1.png"]
    offscreen_window.current_image_index = 0
    offscreen_window.session.set_source_dirty("results_table", True)

    offscreen_window.load_image_by_index(1)

    assert offscreen_window.current_image_index == 0
    assert highlight_calls == [0]
    assert offscreen_window.session.is_dirty is True


def test_close_event_guarded(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    T-008.2: closeEvent prompts on dirty session. If Cancel is chosen, event is ignored.
    """
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Cancel
    )
    offscreen_window.session.set_source_dirty("ocr_text", True)

    event = QCloseEvent()
    offscreen_window.closeEvent(event)

    assert event.isAccepted() is False

    # When user chooses Discard, event is accepted
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Discard
    )
    event2 = QCloseEvent()
    offscreen_window.closeEvent(event2)
    assert event2.isAccepted() is True


def test_browse_input_guarded(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    T-008.2: browse_input prompts on dirty session. If Cancel is chosen, file dialog does not open.
    """
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Cancel
    )
    offscreen_window.session.set_source_dirty("data_table", True)

    file_dialog_opened = False

    def mock_get_existing_directory(*args, **kwargs):
        nonlocal file_dialog_opened
        file_dialog_opened = True
        return "/tmp/new_dir"

    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getExistingDirectory", mock_get_existing_directory
    )

    edit = MagicMock()
    offscreen_window.browse_input(edit)

    assert file_dialog_opened is False


def test_start_batch_analysis_thread_guarded(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    T-008.2: start_batch_analysis_thread prompts on dirty session. If Cancel is chosen, batch does not start.
    """
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Cancel
    )
    offscreen_window.session.set_source_dirty("results_table", True)

    batch_started = False

    def mock_start():
        nonlocal batch_started
        batch_started = True

    monkeypatch.setattr(offscreen_window, "validate_paths", lambda: True)
    offscreen_window.start_batch_analysis_thread()

    assert batch_started is False
    assert offscreen_window.is_processing is False


def test_next_image_results_table_dirty_prompts_guard(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    T-008.0 / T-008.2: Calling next_image() with unsaved table edits triggers guard dialog.
    When user cancels, navigation is blocked and reviewer remains on current image.
    """
    dialog_called = False
    buttons_passed = None

    def mock_question(parent, title, text, buttons, default_btn):
        nonlocal dialog_called, buttons_passed
        dialog_called = True
        buttons_passed = buttons
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    offscreen_window.image_files = ["/tmp/img0.png", "/tmp/img1.png"]
    offscreen_window.current_image_index = 0
    offscreen_window.session.set_source_dirty("results_table", True)

    monkeypatch.setattr(offscreen_window, "highlight_selected_file", lambda idx: None)

    offscreen_window.next_image()

    assert dialog_called is True
    assert QMessageBox.StandardButton.Save in buttons_passed
    assert QMessageBox.StandardButton.Discard in buttons_passed
    assert QMessageBox.StandardButton.Cancel in buttons_passed
    assert offscreen_window.current_image_index == 0
    assert offscreen_window.session.is_dirty is True


def test_prev_image_results_table_dirty_prompts_guard(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    T-008.0 / T-008.2: Calling prev_image() with unsaved table edits triggers guard dialog.
    """
    dialog_called = False

    def mock_question(parent, title, text, buttons, default_btn):
        nonlocal dialog_called
        dialog_called = True
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    offscreen_window.image_files = ["/tmp/img0.png", "/tmp/img1.png"]
    offscreen_window.current_image_index = 1
    offscreen_window.session.set_source_dirty("results_table", True)

    monkeypatch.setattr(offscreen_window, "highlight_selected_file", lambda idx: None)

    offscreen_window.prev_image()

    assert dialog_called is True
    assert offscreen_window.current_image_index == 1
    assert offscreen_window.session.is_dirty is True


def test_legacy_unpatched_navigation_silent_discard_fixed(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    T-008.0 -> T-008.2: Flipped from xfail(strict=True) to normal passing assertion.
    Verifies that next_image() intercepts navigation with the unsaved-changes dialog
    and keeps the reviewer on index 0 when cancelled.
    """
    dialog_called = False

    def mock_question(*args, **kwargs):
        nonlocal dialog_called
        dialog_called = True
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    offscreen_window.image_files = ["/tmp/img0.png", "/tmp/img1.png"]
    offscreen_window.current_image_index = 0
    offscreen_window.session.set_source_dirty("results_table", True)

    offscreen_window.next_image()

    assert dialog_called is True
    assert offscreen_window.current_image_index == 0
    assert offscreen_window.session.is_dirty is True


def test_save_and_next_box_dirty_prompts_and_blocks_on_cancel(
    offscreen_window: ModernChartAnalysisApp,
    sample_result_payload,
    monkeypatch,
):
    """
    Defects 1 & 2 regression test: save_and_next() and _save_analysis_results()
    prompt Discard | Cancel when 'boxes' is dirty and do not silently discard box edits.
    """
    offscreen_window.image_files = ["/tmp/img0.png", "/tmp/img1.png"]
    offscreen_window.current_image_index = 0
    offscreen_window.current_image_path = "/tmp/img0.png"
    offscreen_window.current_analysis_result = sample_result_payload("bar")
    offscreen_window.session.set_source_dirty("boxes", True)

    dialog_prompted = False
    buttons_seen = None

    def mock_question(parent, title, text, buttons, default_btn):
        nonlocal dialog_prompted, buttons_seen
        dialog_prompted = True
        buttons_seen = buttons
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    offscreen_window.save_and_next()

    assert dialog_prompted is True
    assert QMessageBox.StandardButton.Discard in buttons_seen
    assert QMessageBox.StandardButton.Cancel in buttons_seen
    assert offscreen_window.current_image_index == 0
    assert "boxes" in offscreen_window.session.dirty_sources


def test_protocol_cell_changed_null_cell_and_non_finite_guards(
    offscreen_window: ModernChartAnalysisApp,
    sample_result_payload,
):
    """
    Defects 4 & 5 regression test: _on_protocol_cell_changed handles None cell items
    without AttributeError and rejects 'nan'/'inf' without marking dirty or mutating _original.
    """
    offscreen_window.current_analysis_result = sample_result_payload("bar")
    offscreen_window._populate_protocol_tab()
    offscreen_window.session.mark_clean()

    # Defect 5: Reject 'nan' and 'inf' without dirtying or adding _original
    val_item = offscreen_window.protocol_table.item(0, 4)
    assert val_item is not None
    orig_val = offscreen_window.current_analysis_result["protocol_rows"][0]["value"]
    for bad_input in ("nan", "NaN", "inf", "-inf", "infinity", "1e500"):
        val_item.setText(bad_input)
        assert offscreen_window.session.is_dirty is False
        assert "_original" not in offscreen_window.current_analysis_result["protocol_rows"][0]
        assert offscreen_window.current_analysis_result["protocol_rows"][0]["value"] == orig_val

    # Defect 4: Taking item and calling _on_protocol_cell_changed does not crash
    offscreen_window.protocol_table.takeItem(0, 4)
    offscreen_window._on_protocol_cell_changed(0, 4)

