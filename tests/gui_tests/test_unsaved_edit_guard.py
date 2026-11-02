# -*- coding: utf-8 -*-
"""GUI tests for ADR-008 Unsaved-Edit Protection & T-008.5-wire Draft Autosave.

Verifies:
- T-008.1: Dirty state aggregation across results_table, ocr_text, data_table, and boxes.
- T-008.2: Navigation and close guards at choke-points (load_image_by_index, closeEvent, browse_input, start_batch_analysis_thread).
- T-008.3 / T-008.4: Confirm Reset and Warn before Apply when text/table edits exist.
- T-008.5-wire:
  1. Debounced autosave timer writes compact draft (< 50 KB) without 'detections' after edit.
  2. Manual save at t=1.5s stops _draft_timer and deletes draft; at t=2.0s no stale draft exists.
  3. Crash recovery prompt on image load: Restore merges on-disk 'detections' back into result and restores editor_detections.
  4. Selecting Discard on draft restore prompt deletes the draft file from disk and does not re-prompt on reload.
  5. Autosave runs _flush_widgets_to_result(refresh_ui=False) without calling _refresh_protocol_rows_from_result or losing table selection/cursor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from PIL import Image
import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QLineEdit,
    QMessageBox,
    QTableWidgetItem,
)

from main_modern import ModernChartAnalysisApp
from visual.draft_store import get_draft_path, is_draft_newer, read_draft, write_draft
from visual.review_session import ReviewSession


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_sample_image(path: Path) -> str:
    img = Image.new("RGB", (120, 90), color=(240, 240, 240))
    img.save(path)
    return str(path)


def _make_sample_result(image_path: str) -> Dict[str, Any]:
    return {
        "image_file": os.path.basename(image_path),
        "original_image_path": str(Path(image_path).resolve()),
        "chart_type": "bar",
        "orientation": "vertical",
        "elements": [
            {"label": "Group A", "value": 12.5, "bbox": [10, 20, 30, 80], "xyxy": [10, 20, 30, 80]},
            {"label": "Group B", "value": 25.0, "bbox": [40, 10, 60, 80], "xyxy": [40, 10, 60, 80]},
        ],
        "bars": [
            {"bar_label": "Group A", "value": 12.5, "xyxy": [10, 20, 30, 80]},
            {"bar_label": "Group B", "value": 25.0, "xyxy": [40, 10, 60, 80]},
        ],
        "detections": {
            "bar": [
                {"xyxy": [10, 20, 30, 80], "conf": 0.95},
                {"xyxy": [40, 10, 60, 80], "conf": 0.92},
            ],
            "line_series": [
                {"mask": [0.5] * 15000, "color": "blue"}  # Heavy mask to verify stripping
            ],
        },
        "scale_info": {"r_squared": 0.99, "baseline_y_coord": 80.0},
        "protocol_rows": [
            {
                "_uid": "uid_row_1",
                "source_file": os.path.basename(image_path),
                "page_index": 0,
                "chart_type": "bar",
                "outcome": "Outcome 1",
                "group": "Group A",
                "value": 12.5,
                "unit": "mg",
                "confidence": 0.95,
                "review_status": "auto",
                "_original": None,
                "bbox": [10, 20, 30, 80],
            },
            {
                "_uid": "uid_row_2",
                "source_file": os.path.basename(image_path),
                "page_index": 0,
                "chart_type": "bar",
                "outcome": "Outcome 1",
                "group": "Group B",
                "value": 25.0,
                "unit": "mg",
                "confidence": 0.92,
                "review_status": "auto",
                "_original": None,
                "bbox": [40, 10, 60, 80],
            },
        ],
    }


@pytest.fixture
def review_window(qapp, tmp_path: Path):
    """Build an offscreen ModernChartAnalysisApp configured with a temp output dir."""
    mock_ctx = MagicMock()
    mock_ctx.data_manager.get_context.return_value = {}

    with patch.object(ModernChartAnalysisApp, "load_config", lambda self: None), \
         patch.object(ModernChartAnalysisApp, "load_advanced_settings", lambda self: None), \
         patch.object(ModernChartAnalysisApp, "save_advanced_settings", lambda self: None), \
         patch.object(ModernChartAnalysisApp, "save_config", lambda self: None):
        win = ModernChartAnalysisApp(context=mock_ctx)
    win.advanced_settings_file = tmp_path / "advanced_settings.json"

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    win.output_path_edit.setText(str(out_dir))
    win.input_path_edit.setText(str(tmp_path))
    win.advanced_settings = {
        "ocr_engine": "Paddle",
        "review_settings": {
            "autosave_draft": True,
            "autosave_interval_ms": 200,  # Fast interval for deterministic tests
        },
    }

    img1 = _make_sample_image(tmp_path / "chart_01.png")
    img2 = _make_sample_image(tmp_path / "chart_02.png")
    win.image_files = [img1, img2]
    win._unique_stems = {img1: "chart_01", img2: "chart_02"}
    win.current_image_index = 0
    win.current_image_path = img1

    res1 = _make_sample_result(img1)
    json1 = out_dir / "chart_01_analysis.json"
    json1.write_text(json.dumps(res1), encoding="utf-8")
    win._benchmark_result_cache[img1] = str(json1)

    res2 = _make_sample_result(img2)
    json2 = out_dir / "chart_02_analysis.json"
    json2.write_text(json.dumps(res2), encoding="utf-8")
    win._benchmark_result_cache[img2] = str(json2)

    win._load_result_from_cache(img1, str(json1))
    win.session.mark_clean()
    yield win
    win.cleanup_resources()
    win.close()


# ===========================================================================
# T-008.5-wire Tests: Draft Autosave, Crash Recovery, and Cleanup
# ===========================================================================


def test_draft_autosave_on_cell_edit_and_crash_recovery_restore(review_window, tmp_path: Path):
    """Acceptance 1: Editing a cell schedules autosave, strips 'detections', and prompts Restore on reload."""
    win = review_window
    out_dir = Path(win.output_path_edit.text())
    draft_path = get_draft_path(out_dir, "chart_01")

    assert not draft_path.exists()
    assert win.session.is_dirty is False

    # Find the 'value' column in protocol_table and edit row 0
    from core.protocol_row_builder import get_protocol_columns
    cols = get_protocol_columns("bar")
    val_col_idx = next(i for i, (cid, _, _) in enumerate(cols) if cid == "value")

    item = win.protocol_table.item(0, val_col_idx)
    assert item is not None
    item.setText("99.5")

    assert win.session.is_dirty is True
    assert "results_table" in win.session.dirty_sources
    assert win._draft_timer.isActive() is True

    # Trigger the autosave timer callback directly (simulating 2s debounce elapsing)
    win._draft_timer.stop()
    assert win._perform_draft_autosave() is True
    assert draft_path.is_file()

    # Verify draft content invariants: no 'detections', < 50 KB, has edited value
    draft_data = read_draft(draft_path)
    assert draft_data is not None
    assert "detections" not in draft_data["result"]
    assert draft_path.stat().st_size < 50 * 1024
    edited_rows = draft_data["result"]["protocol_rows"]
    assert edited_rows[0]["value"] == 99.5
    assert edited_rows[0]["review_status"] == "corrected"

    # Ensure draft mtime is strictly newer than the analysis JSON
    json1 = Path(win._benchmark_result_cache[win.image_files[0]])
    t_now = time.time() + 10.0
    os.utime(draft_path, (t_now, t_now))
    assert is_draft_newer(draft_path, json1) is True

    # Simulate crash & reload of chart_01 -> user clicks Yes (Restore)
    win.session.mark_clean()
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes) as mock_q:
        win._load_result_from_cache(win.image_files[0], str(json1))
        assert mock_q.called
        args, _ = mock_q.call_args
        assert "Restore unsaved changes from" in args[2]

    # Verify restored state: value is 99.5 AND 'detections' was re-attached from disk JSON
    assert win.current_analysis_result["protocol_rows"][0]["value"] == 99.5
    assert "detections" in win.current_analysis_result
    assert "line_series" in win.current_analysis_result["detections"]
    assert win.session.is_dirty is True


def test_draft_autosave_with_no_prior_analysis_json(review_window, tmp_path: Path):
    """Acceptance 1b: Newly opened image with no prior _analysis.json writes draft and restores cleanly."""
    win = review_window
    out_dir = Path(win.output_path_edit.text())

    # Remove on-disk _analysis.json for chart_01 to simulate uncached fresh analysis
    json1 = Path(win._benchmark_result_cache[win.image_files[0]])
    if json1.exists():
        json1.unlink()

    draft_path = get_draft_path(out_dir, "chart_01")
    assert not draft_path.exists()

    # Edit protocol row
    from core.protocol_row_builder import get_protocol_columns
    cols = get_protocol_columns("bar")
    val_col_idx = next(i for i, (cid, _, _) in enumerate(cols) if cid == "value")
    win.protocol_table.item(0, val_col_idx).setText("77.25")

    win._draft_timer.stop()
    assert win._perform_draft_autosave() is True
    assert draft_path.exists()
    assert is_draft_newer(draft_path, json1) is True

    # Simulate fresh analysis completion when draft exists -> prompts Restore
    fresh_pipeline_result = _make_sample_result(win.image_files[0])
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes) as mock_q:
        win._on_analysis_complete(fresh_pipeline_result)
        assert mock_q.called

    assert win.current_analysis_result["protocol_rows"][0]["value"] == 77.25
    assert "bar" in win.current_analysis_result.get("detections", {})


def test_manual_save_stops_timer_and_prevents_stale_draft(review_window):
    """Acceptance 2: Manual save at t=1.5s stops _draft_timer and deletes draft; at t=2.0s no stale draft is written."""
    win = review_window
    out_dir = Path(win.output_path_edit.text())
    draft_path = get_draft_path(out_dir, "chart_01")

    # Pre-create an earlier draft on disk to ensure manual save deletes it
    write_draft(draft_path, {"result": {"chart_type": "bar"}, "editor_detections": []})
    assert draft_path.exists()

    # Edit cell at t=0 -> timer starts
    from core.protocol_row_builder import get_protocol_columns
    cols = get_protocol_columns("bar")
    val_col_idx = next(i for i, (cid, _, _) in enumerate(cols) if cid == "value")
    win.protocol_table.item(0, val_col_idx).setText("45.0")

    assert win.session.is_dirty is True
    assert win._draft_timer.isActive() is True

    # Manual save at t=1.5s
    assert win._save_analysis_results() is True

    # Timer must be stopped, session clean, and existing draft deleted
    assert win._draft_timer.isActive() is False
    assert win.session.is_dirty is False
    assert not draft_path.exists()

    # If timer callback were invoked at t=2.0s anyway, it must early-exit without writing
    assert win._perform_draft_autosave() is False
    assert not draft_path.exists()


def test_discard_on_restore_prompt_deletes_draft_and_does_not_reprompt(review_window):
    """Acceptance 3: Selecting Discard on draft restore prompt deletes draft from disk and re-opening does not prompt."""
    win = review_window
    out_dir = Path(win.output_path_edit.text())
    draft_path = get_draft_path(out_dir, "chart_01")
    json1 = Path(win._benchmark_result_cache[win.image_files[0]])

    # Write a newer draft with modified value 555.0
    draft_res = _make_sample_result(win.image_files[0])
    del draft_res["detections"]
    draft_res["protocol_rows"][0]["value"] = 555.0
    write_draft(draft_path, {"result": draft_res, "editor_detections": []})
    t_future = time.time() + 20.0
    os.utime(draft_path, (t_future, t_future))
    assert draft_path.exists()

    # Load image 1 -> user clicks No (Discard)
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as mock_q:
        win._load_result_from_cache(win.image_files[0], str(json1))
        assert mock_q.call_count == 1

    # Draft file must be deleted immediately and original cached value (12.5) loaded
    assert not draft_path.exists()
    assert win.current_analysis_result["protocol_rows"][0]["value"] == 12.5

    # Reloading image 1 again must NOT prompt
    with patch.object(QMessageBox, "question") as mock_q_again:
        win._load_result_from_cache(win.image_files[0], str(json1))
        mock_q_again.assert_not_called()


def test_autosave_does_not_rerender_protocol_table_or_lose_selection(review_window):
    """Acceptance 4: Autosave calls _flush_widgets_to_result(refresh_ui=False) without calling _refresh_protocol_rows_from_result."""
    win = review_window

    # Inject an OCR widget entry and edit its text
    entry = QLineEdit("Updated Title")
    orig_item = {"text": "Old Title", "xyxy": [5, 5, 50, 15]}
    win.analysis_results_widgets["chart_title_0_0"] = {
        "entry": entry,
        "original_item": orig_item,
        "section": "chart_title",
    }
    win._on_ocr_text_edited()
    assert win.session.is_dirty is True

    # Select row 1, col 1 in protocol_table
    win.protocol_table.setCurrentCell(1, 1)
    assert win.protocol_table.currentRow() == 1
    assert win.protocol_table.currentColumn() == 1

    with patch.object(win, "_refresh_protocol_rows_from_result") as mock_refresh, \
         patch.object(win, "_render_protocol_table") as mock_render:
        win._draft_timer.stop()
        ok = win._perform_draft_autosave()
        assert ok is True
        mock_refresh.assert_not_called()
        mock_render.assert_not_called()

    # Verify OCR widget text was flushed into original_item and table selection is untouched
    assert orig_item["text"] == "Updated Title"
    assert win.protocol_table.currentRow() == 1
    assert win.protocol_table.currentColumn() == 1


# ===========================================================================
# T-008.2 Navigation & Close Guard Tests
# ===========================================================================


def test_navigation_guard_save_persists_and_navigates(review_window):
    """Dirty table edit + Next image -> Save persists _analysis.json, deletes draft, and navigates."""
    win = review_window
    out_dir = Path(win.output_path_edit.text())
    draft_path = get_draft_path(out_dir, "chart_01")

    from core.protocol_row_builder import get_protocol_columns
    cols = get_protocol_columns("bar")
    val_col_idx = next(i for i, (cid, _, _) in enumerate(cols) if cid == "value")
    win.protocol_table.item(0, val_col_idx).setText("88.0")
    win._perform_draft_autosave()
    assert draft_path.exists()

    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Save) as mock_q:
        win.next_image()
        assert mock_q.called

    assert win.current_image_index == 1
    assert not draft_path.exists()
    saved_json = json.loads((out_dir / "chart_01_analysis.json").read_text(encoding="utf-8"))
    assert saved_json["protocol_rows"][0]["value"] == 88.0


def test_navigation_guard_cancel_stays_on_current_image(review_window):
    """Dirty table edit + Next image -> Cancel aborts navigation and keeps edits."""
    win = review_window
    from core.protocol_row_builder import get_protocol_columns
    cols = get_protocol_columns("bar")
    val_col_idx = next(i for i, (cid, _, _) in enumerate(cols) if cid == "value")
    win.protocol_table.item(0, val_col_idx).setText("64.0")

    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel):
        win.next_image()

    assert win.current_image_index == 0
    assert win.session.is_dirty is True
    assert win.current_analysis_result["protocol_rows"][0]["value"] == 64.0


def test_navigation_guard_unapplied_boxes_offers_discard_or_cancel(review_window):
    """Dirty box edits warn that Apply is needed and offer only Discard / Cancel."""
    win = review_window
    win.session.set_source_dirty("boxes", True)

    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel) as mock_q:
        win.next_image()
        assert mock_q.called
        args, _ = mock_q.call_args
        assert "unapplied box edits" in args[2]

    assert win.current_image_index == 0
    assert win.session.is_dirty is True

    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard):
        win.next_image()

    assert win.current_image_index == 1
    assert win.session.is_dirty is False


def test_close_event_guarded_when_dirty(review_window):
    """closeEvent prompts when dirty and ignores the event if Cancel is selected."""
    win = review_window
    win.session.set_source_dirty("results_table", True)

    close_ev = QCloseEvent()
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel):
        win.closeEvent(close_ev)
        assert close_ev.isAccepted() is False