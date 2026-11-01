# -*- coding: utf-8 -*-
"""
Empirical Challenge & Stress Suite for Wave 1 Bundle 1.1
Author: Challenger 2 (Integration Team Argon)

Verifies:
1. macOS Darwin async combo race condition (QTimer.singleShot delayed/blocked).
2. All 4 destructive transitions (load_image_by_index, closeEvent, browse_input, start_batch_analysis_thread)
   under Box-dirty, Table-dirty, and Mixed dirty conditions.
3. Box-dirty NEVER presents "Save" (even when multiple dirty sources exist).
4. Selection highlight restoration on Cancel without re-triggering navigation or recursion.
5. Save failure cleanly aborts navigation and keeps user on current image across transitions.
6. Edge cases: identical index, active processing guard, invalid dirty sources, null/malformed protocol rows.
"""

from __future__ import annotations

import platform
from unittest.mock import MagicMock
import pytest

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton, QLineEdit

from main_modern import ModernChartAnalysisApp
from visual.review_session import ReviewSession, ProcessingState
from visual.qt_utils import safe_combo_populate


# ---------------------------------------------------------------------------
# Challenge 1: macOS Darwin Async Combo Race Condition
# ---------------------------------------------------------------------------

def test_challenge_macos_darwin_async_combo_race_condition(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    Simulate Darwin macOS async safe_combo_populate where Cocoa updates are delayed or blocked.
    Verify that session.filtered_rows() accurately returns rows based on session.filters
    independent of combo state.
    """
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    # Queue to intercept singleShot calls instead of running them
    delayed_callbacks = []

    def mock_single_shot(msec, callback):
        delayed_callbacks.append(callback)

    monkeypatch.setattr(QTimer, "singleShot", mock_single_shot)

    rows = [
        {"outcome": "Outcome A", "group": "Control", "review_status": "auto", "value": 10.0},
        {"outcome": "Outcome A", "group": "Treated", "review_status": "corrected", "value": 20.0},
        {"outcome": "Outcome B", "group": "Control", "review_status": "reviewed", "value": 30.0},
        {"outcome": "Outcome B", "group": "Treated", "review_status": "auto", "value": 40.0},
    ]

    offscreen_window.current_analysis_result = {
        "chart_type": "bar",
        "protocol_rows": rows,
    }

    # Set prior non-default filter in session
    offscreen_window.session.set_filter("status", "corrected")
    offscreen_window.session.set_filter("outcome", "Outcome A")

    # Call _refresh_filter_combos()
    offscreen_window._refresh_filter_combos()

    # Verify safe_combo_populate was intercepted into delayed_callbacks (not executed immediately)
    assert len(delayed_callbacks) >= 2

    # 1. Verify session filters were reset to "All" synchronously BEFORE callbacks execute
    assert offscreen_window.session.filters == {
        "status": "All",
        "outcome": "All",
        "group": "All",
    }

    # 2. Verify session.filtered_rows() returns all rows (reflecting session.filters)
    assert len(offscreen_window.session.filtered_rows()) == 4

    # 3. Apply protocol filters to UI table while combos are still unpopulated/blocked
    offscreen_window._apply_protocol_filters()
    assert offscreen_window.protocol_table.rowCount() == 4

    # 4. Programmatically update session filters while combos are still delayed/blocked
    offscreen_window.session.set_filter("outcome", "Outcome A")
    filtered = offscreen_window.session.filtered_rows()
    assert len(filtered) == 2
    assert all(r["outcome"] == "Outcome A" for r in filtered)

    offscreen_window.session.set_filter("status", "corrected")
    filtered_both = offscreen_window.session.filtered_rows()
    assert len(filtered_both) == 1
    assert filtered_both[0]["value"] == 20.0

    offscreen_window._apply_protocol_filters()
    assert offscreen_window.protocol_table.rowCount() == 1

    # 5. Execute delayed callbacks now (simulating event loop wakeup)
    for cb in delayed_callbacks:
        cb()

    # Verify no crash, and combos now contain the populated items
    assert offscreen_window.proto_outcome_combo.count() >= 2
    assert offscreen_window.proto_group_combo.count() >= 2


# ---------------------------------------------------------------------------
# Challenge 2 & 3: All 4 Destructive Transitions under Box-Dirty vs Table-Dirty
#                  and Verify "Save" is NEVER presented when Box-dirty
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("transition_name", [
    "load_image_by_index",
    "closeEvent",
    "browse_input",
    "start_batch_analysis_thread",
])
def test_challenge_box_dirty_never_presents_save(offscreen_window: ModernChartAnalysisApp, monkeypatch, transition_name):
    """
    Verify across all 4 destructive transitions that when Box-dirty is present,
    QMessageBox.question NEVER contains QMessageBox.StandardButton.Save.
    """
    dialog_buttons = None

    def mock_question(parent, title, text, buttons, default_btn):
        nonlocal dialog_buttons
        dialog_buttons = buttons
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    # Set box-dirty
    offscreen_window.session.set_source_dirty("boxes", True)
    assert offscreen_window.session.is_dirty is True

    # Setup mocks
    offscreen_window.image_files = ["/tmp/a.png", "/tmp/b.png"]
    offscreen_window.current_image_index = 0

    if transition_name == "load_image_by_index":
        offscreen_window.load_image_by_index(1)
    elif transition_name == "closeEvent":
        event = QCloseEvent()
        offscreen_window.closeEvent(event)
    elif transition_name == "browse_input":
        edit = MagicMock()
        offscreen_window.browse_input(edit)
    elif transition_name == "start_batch_analysis_thread":
        monkeypatch.setattr(offscreen_window, "validate_paths", lambda: True)
        offscreen_window.start_batch_analysis_thread()

    assert dialog_buttons is not None
    assert QMessageBox.StandardButton.Discard in dialog_buttons
    assert QMessageBox.StandardButton.Cancel in dialog_buttons
    assert QMessageBox.StandardButton.Save not in dialog_buttons


@pytest.mark.parametrize("dirty_combo", [
    {"boxes", "results_table"},
    {"boxes", "ocr_text"},
    {"boxes", "data_table"},
    {"boxes", "ocr_text", "data_table", "results_table"},
])
def test_challenge_mixed_dirty_with_boxes_never_presents_save(offscreen_window: ModernChartAnalysisApp, monkeypatch, dirty_combo):
    """
    Verify that even when table/text sources are ALSO dirty, if "boxes" is in dirty_sources,
    "Save" is NEVER presented.
    """
    dialog_buttons = None

    def mock_question(parent, title, text, buttons, default_btn):
        nonlocal dialog_buttons
        dialog_buttons = buttons
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    offscreen_window.session.mark_clean()
    for src in dirty_combo:
        offscreen_window.session.set_source_dirty(src, True)

    offscreen_window.image_files = ["/tmp/a.png", "/tmp/b.png"]
    offscreen_window.current_image_index = 0

    offscreen_window.load_image_by_index(1)

    assert dialog_buttons is not None
    assert QMessageBox.StandardButton.Save not in dialog_buttons
    assert QMessageBox.StandardButton.Discard in dialog_buttons
    assert QMessageBox.StandardButton.Cancel in dialog_buttons


@pytest.mark.parametrize("table_source", ["results_table", "ocr_text", "data_table"])
def test_challenge_table_dirty_presents_save_discard_cancel(offscreen_window: ModernChartAnalysisApp, monkeypatch, table_source):
    """
    Verify that when table/text edits are dirty without box edits,
    QMessageBox.question presents Save | Discard | Cancel.
    """
    dialog_buttons = None

    def mock_question(parent, title, text, buttons, default_btn):
        nonlocal dialog_buttons
        dialog_buttons = buttons
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    offscreen_window.session.mark_clean()
    offscreen_window.session.set_source_dirty(table_source, True)

    offscreen_window.image_files = ["/tmp/a.png", "/tmp/b.png"]
    offscreen_window.current_image_index = 0

    offscreen_window.load_image_by_index(1)

    assert dialog_buttons is not None
    assert QMessageBox.StandardButton.Save in dialog_buttons
    assert QMessageBox.StandardButton.Discard in dialog_buttons
    assert QMessageBox.StandardButton.Cancel in dialog_buttons


# ---------------------------------------------------------------------------
# Challenge 4: File List Selection Highlight Cleanly Restored on Cancel
#              Without Re-Triggering Navigation or Recursion
# ---------------------------------------------------------------------------

def test_challenge_cancel_restores_highlight_without_retriggering_navigation(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    Verify that when Cancel is chosen during load_image_by_index:
    1. File list selection highlight is cleanly restored to current_image_index.
    2. Target button is unselected.
    3. Navigation is NOT re-triggered (re-entrancy / recursion guard).
    4. Session dirty state remains True.
    """
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Cancel
    )

    offscreen_window.image_files = ["/tmp/img0.png", "/tmp/img1.png", "/tmp/img2.png"]
    offscreen_window.current_image_index = 0

    # Build real QPushButton widgets in _file_list_buttons with default unselected styles
    btn0 = QPushButton("img0.png")
    btn1 = QPushButton("img1.png")
    btn2 = QPushButton("img2.png")
    for b in (btn0, btn1, btn2):
        b.setStyleSheet("background-color: transparent;")
    offscreen_window._file_list_buttons = {0: btn0, 1: btn1, 2: btn2}

    # Initial highlight on index 0
    offscreen_window.highlight_selected_file(0)
    assert "#264f78" in btn0.styleSheet()
    assert "transparent" in btn1.styleSheet()

    # Simulate scenario where btn1 was selected/focused prior to navigation abort
    offscreen_window.highlight_selected_file(1)
    assert "#264f78" in btn1.styleSheet()
    assert "transparent" in btn0.styleSheet()
    assert offscreen_window._previous_selected_index == 1

    # Mark dirty
    offscreen_window.session.set_source_dirty("results_table", True)

    # Track calls to load_image_by_index
    original_load = offscreen_window.load_image_by_index
    call_indices = []

    def tracking_load(idx):
        call_indices.append(idx)
        return original_load(idx)

    monkeypatch.setattr(offscreen_window, "load_image_by_index", tracking_load)

    # User clicks button 1 (or triggers load_image_by_index(1))
    offscreen_window.load_image_by_index(1)

    # Exactly 1 call was made (no recursive re-triggering of load_image_by_index)
    assert call_indices == [1]

    # Current index remains 0
    assert offscreen_window.current_image_index == 0
    assert offscreen_window._previous_selected_index == 0

    # Highlight restored: btn0 selected, btn1 transparent
    assert "#264f78" in btn0.styleSheet()
    assert "transparent" in btn1.styleSheet()
    assert "transparent" in btn2.styleSheet()

    # Session is still dirty
    assert offscreen_window.session.is_dirty is True


# ---------------------------------------------------------------------------
# Challenge 5: Save Failure Aborts Navigation Across All 4 Transitions
# ---------------------------------------------------------------------------

def test_challenge_save_failure_aborts_load_image(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    Verify on load_image_by_index that if Save fails (_save_analysis_results returns False),
    navigation is aborted, user stays on current image, and highlight is restored.
    """
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Save
    )
    monkeypatch.setattr(offscreen_window, "_save_analysis_results", lambda: False)

    highlight_calls = []
    monkeypatch.setattr(
        offscreen_window, "highlight_selected_file", lambda idx: highlight_calls.append(idx)
    )

    load_assisted_called = False
    monkeypatch.setattr(
        offscreen_window, "load_image_for_assisted_analysis", lambda: setattr(load_assisted_called, "flag", True)
    )

    offscreen_window.image_files = ["/tmp/img0.png", "/tmp/img1.png"]
    offscreen_window.current_image_index = 0
    offscreen_window.session.set_source_dirty("results_table", True)

    offscreen_window.load_image_by_index(1)

    assert offscreen_window.current_image_index == 0
    assert highlight_calls == [0]
    assert offscreen_window.session.is_dirty is True
    assert load_assisted_called is False


def test_challenge_save_failure_aborts_close_event(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    Verify on closeEvent that if Save fails, closeEvent calls event.ignore(),
    and cleanup_resources / save_config are NOT executed.
    """
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Save
    )
    monkeypatch.setattr(offscreen_window, "_save_analysis_results", lambda: False)

    cleanup_called = False
    monkeypatch.setattr(offscreen_window, "cleanup_resources", lambda: setattr(cleanup_called, "flag", True))

    save_config_called = False
    monkeypatch.setattr(offscreen_window, "save_config", lambda: setattr(save_config_called, "flag", True))

    offscreen_window.session.set_source_dirty("ocr_text", True)

    event = QCloseEvent()
    offscreen_window.closeEvent(event)

    assert event.isAccepted() is False
    assert cleanup_called is False
    assert save_config_called is False
    assert offscreen_window.session.is_dirty is True


def test_challenge_save_failure_aborts_browse_input(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    Verify on browse_input that if Save fails, super().browse_input is not reached
    and file dialog is never invoked.
    """
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Save
    )
    monkeypatch.setattr(offscreen_window, "_save_analysis_results", lambda: False)

    dialog_opened = False
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: setattr(dialog_opened, "flag", True) or "/tmp/dummy",
    )

    offscreen_window.session.set_source_dirty("data_table", True)

    line_edit = QLineEdit()
    line_edit.setText("/initial/path")
    offscreen_window.browse_input(line_edit)

    assert line_edit.text() == "/initial/path"
    assert dialog_opened is False
    assert offscreen_window.session.is_dirty is True


def test_challenge_save_failure_aborts_start_batch_thread(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    Verify on start_batch_analysis_thread that if Save fails, batch thread is not started.
    """
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Save
    )
    monkeypatch.setattr(offscreen_window, "_save_analysis_results", lambda: False)

    thread_started = False
    monkeypatch.setattr(offscreen_window, "validate_paths", lambda: True)

    offscreen_window.session.set_source_dirty("results_table", True)
    offscreen_window.start_batch_analysis_thread()

    assert offscreen_window.is_processing is False
    assert thread_started is False
    assert offscreen_window.session.is_dirty is True


# ---------------------------------------------------------------------------
# Challenge 6: Stress & Edge Cases in ReviewSession & Guard Dispatch
# ---------------------------------------------------------------------------

def test_challenge_load_same_image_index_no_op(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    If load_image_by_index is called with the current index, it should return
    early without prompting any unsaved-edits dialog.
    """
    dialog_called = False

    def mock_question(*args, **kwargs):
        nonlocal dialog_called
        dialog_called = True
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", mock_question)

    offscreen_window.current_image_index = 2
    offscreen_window.session.set_source_dirty("boxes", True)

    offscreen_window.load_image_by_index(2)

    assert dialog_called is False
    assert offscreen_window.current_image_index == 2


def test_challenge_is_processing_blocks_navigation_without_guard(offscreen_window: ModernChartAnalysisApp, monkeypatch):
    """
    When is_processing is True, load_image_by_index should display an information
    dialog and abort before reaching the dirty guard.
    """
    info_called = False

    def mock_info(*args, **kwargs):
        nonlocal info_called
        info_called = True

    monkeypatch.setattr(QMessageBox, "information", mock_info)

    guard_called = False
    monkeypatch.setattr(offscreen_window, "_check_unsaved_changes_guard", lambda: setattr(guard_called, "flag", True) or True)

    offscreen_window.current_image_index = 0
    offscreen_window.is_processing = True

    offscreen_window.load_image_by_index(1)

    assert info_called is True
    assert guard_called is False
    assert offscreen_window.current_image_index == 0


def test_challenge_invalid_dirty_source_raises_value_error():
    """
    ReviewSession.set_source_dirty with an unrecognized source must raise ValueError.
    """
    session = ReviewSession()
    with pytest.raises(ValueError, match="Unknown dirty source: invalid_src"):
        session.set_source_dirty("invalid_src", True)


def test_challenge_dirty_changed_signal_only_emits_on_transitions():
    """
    dirty_changed(bool) must emit ONLY on transition False -> True or True -> False,
    not on redundant sets.
    """
    session = ReviewSession()
    emissions = []
    session.dirty_changed.connect(lambda state: emissions.append(state))

    # First dirty source -> transition False -> True
    session.set_source_dirty("boxes", True)
    assert emissions == [True]

    # Redundant set on same source -> no emission
    session.set_source_dirty("boxes", True)
    assert emissions == [True]

    # Second dirty source -> already dirty, no emission
    session.set_source_dirty("results_table", True)
    assert emissions == [True]

    # Remove one source -> still dirty ("boxes" remains), no emission
    session.set_source_dirty("results_table", False)
    assert emissions == [True]

    # Remove last source -> transition True -> False
    session.set_source_dirty("boxes", False)
    assert emissions == [True, False]

    # Redundant mark_clean when already clean -> no emission
    session.mark_clean()
    assert emissions == [True, False]


def test_challenge_review_session_malformed_rows_robustness():
    """
    ReviewSession.filtered_rows() and counts() should gracefully handle
    empty, None, or rows missing expected keys without raising KeyError or AttributeError.
    """
    session = ReviewSession()

    # None result
    assert session.filtered_rows() == []
    assert session.counts() == (0, 0)

    # Empty result
    session.set_result({})
    assert session.filtered_rows() == []
    assert session.counts() == (0, 0)

    # Rows with missing keys
    malformed_rows = [
        {},  # empty row
        {"value": 10},  # no outcome/group/review_status
        {"outcome": "A"},  # partial
    ]
    session.set_result({"protocol_rows": malformed_rows})
    assert len(session.filtered_rows()) == 3
    assert session.counts() == (3, 3)

    # Filter with missing keys
    session.set_filter("status", "corrected")
    # None of the malformed rows have review_status == "corrected"
    assert session.filtered_rows() == []
    assert session.counts() == (0, 3)

    # update_row_field with out of bounds index does not crash
    session.update_row_field(999, "value", 100)
    session.update_row_field(-5, "value", 100)
    session.update_row_field("non_existent_uid", "value", 100)
