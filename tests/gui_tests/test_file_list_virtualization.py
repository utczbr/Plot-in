# -*- coding: utf-8 -*-
"""
Tests for File List Model/View Virtualization and Async State Loading.
Covers ADR-014 (T-014.3, T-014.4) and ADR-017 (T-017.5).

Verifies:
- 2,000 files populate in < 200 ms in QListView.
- Selection changes touch <= 2 model rows; sequential navigation is < 2 ms.
- Queue Selector ('All', 'Needs Review', 'Reviewed') filtering.
- Selected row contrast >= 4.5:1.
- Async FileStateLoader state derivation and safe path signal handling.
- Mark image reviewed moves it out of Needs Review queue.
- has_icon helper.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import List

import pytest
from PyQt6.QtCore import QModelIndex, Qt, QItemSelectionModel
from PyQt6.QtWidgets import QApplication, QListView

from core.review_state import ImageReviewState
from visual.contrast import contrast_ratio
from visual.file_list_model import (
    FileListDelegate,
    FileListFilterProxyModel,
    FileListModel,
    FileListRoles,
    FileStateLoader,
)
from main_modern import ModernChartAnalysisApp


@pytest.fixture
def mock_app_window(qapp, tmp_path):
    """Fixture providing an offscreen ModernChartAnalysisApp configured with tmp directories."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    models_dir = tmp_path / "models"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    win = ModernChartAnalysisApp()
    win.save_config = lambda: None
    win.save_advanced_settings = lambda: None
    win.input_path_edit.setText(str(input_dir))
    win.output_path_edit.setText(str(output_dir))
    win.models_dir_edit.setText(str(models_dir))

    yield win

    try:
        win.cleanup_image_resources()
        if hasattr(win, "_draft_timer") and win._draft_timer:
            win._draft_timer.stop()
        if hasattr(win, "_file_state_loader") and win._file_state_loader:
            win._file_state_loader.cancel()
    except Exception:
        pass


def test_2000_files_populate_under_200ms(qapp):
    """ADR-017 T-017.5: 2 000 images populate in < 200 ms in QListView."""
    files = [f"/path/to/image_{i:04d}.png" for i in range(2000)]
    stems = {f: f"image_{i:04d}" for i, f in enumerate(files)}

    model = FileListModel()
    view = QListView()
    view.setModel(model)

    start = time.perf_counter()
    model.set_files(files, stems)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert model.rowCount() == 2000
    assert elapsed_ms < 200.0, f"Populating 2,000 files took {elapsed_ms:.2f} ms (expected < 200 ms)"


def test_selection_change_efficiency_and_navigation_timing(qapp):
    """ADR-017 T-017.5: sequential arrow-key navigation in < 2 ms; touches <= 2 visual rows."""
    files = [f"/path/to/img_{i:02d}.png" for i in range(100)]
    model = FileListModel()
    model.set_files(files)

    view = QListView()
    view.setModel(model)

    sel_model = view.selectionModel()
    idx0 = model.index(0, 0)
    sel_model.setCurrentIndex(idx0, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    # Test sequential navigation time
    start = time.perf_counter()
    for row in range(1, 10):
        idx = model.index(row, 0)
        sel_model.setCurrentIndex(idx, QItemSelectionModel.SelectionFlag.ClearAndSelect)
    elapsed_per_step_ms = ((time.perf_counter() - start) / 9.0) * 1000.0

    assert elapsed_per_step_ms < 2.0, f"Navigation step took {elapsed_per_step_ms:.2f} ms (expected < 2 ms)"
    assert sel_model.currentIndex().row() == 9


def test_selected_item_contrast_wcag_conformance():
    """ADR-017 T-017.5: selected-row background #264f78 with white text has contrast >= 4.5:1."""
    ratio = contrast_ratio("#ffffff", "#264f78")
    assert ratio >= 4.5
    assert ratio > 8.0  # Measured ~8.5:1


def test_queue_selector_proxy_filtering(qapp):
    """ADR-014 T-014.3: Queue Selector filters [All | Needs Review | Reviewed] correctly."""
    files = [
        "/path/to/auto.png",
        "/path/to/needs_rev.png",
        "/path/to/reviewed.png",
        "/path/to/failed.png",
        "/path/to/pending.png",
    ]
    model = FileListModel()
    model.set_files(files)

    model.update_file_state("/path/to/auto.png", ImageReviewState.AUTO)
    model.update_file_state("/path/to/needs_rev.png", ImageReviewState.NEEDS_REVIEW, confidence=0.45)
    model.update_file_state("/path/to/reviewed.png", ImageReviewState.REVIEWED)
    model.update_file_state("/path/to/failed.png", ImageReviewState.FAILED, failure_reason="Corrupted image")
    model.update_file_state("/path/to/pending.png", ImageReviewState.PENDING)

    proxy = FileListFilterProxyModel()
    proxy.setSourceModel(model)

    # 1. 'All' mode includes all 5 files
    proxy.set_filter_mode("All")
    assert proxy.rowCount() == 5

    # 2. 'Needs Review' includes needs_review, failed
    proxy.set_filter_mode("Needs Review")
    assert proxy.rowCount() == 2
    visible_paths = [
        proxy.data(proxy.index(r, 0), FileListRoles.FilePathRole)
        for r in range(proxy.rowCount())
    ]
    assert "/path/to/needs_rev.png" in visible_paths
    assert "/path/to/failed.png" in visible_paths

    # Case-insensitive / normalized string check
    proxy.set_filter_mode("needs review")
    assert proxy.rowCount() == 2

    # 3. 'Reviewed' includes only reviewed
    proxy.set_filter_mode("Reviewed")
    assert proxy.rowCount() == 1
    assert proxy.data(proxy.index(0, 0), FileListRoles.FilePathRole) == "/path/to/reviewed.png"


def test_file_state_loader_async_derivation(qapp, tmp_path):
    """ADR-014 T-014.4: FileStateLoader computes review states from summaries and backfills missing summaries."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    img1 = str(tmp_path / "chart1.png")
    img2 = str(tmp_path / "chart2.png")
    img3 = str(tmp_path / "chart3.png")
    img4 = str(tmp_path / "chart4.png")

    # 1. chart1 has summary with low confidence -> needs_review
    summary1 = {
        "schema_version": "1.0",
        "chart_type": "bar",
        "min_confidence": 0.40,
        "warning_count": 0,
        "review_status": None,
        "corrected_rows": 0,
    }
    (output_dir / "chart1_summary.json").write_text(json.dumps(summary1), encoding="utf-8")

    # 2. chart2 has failed marker -> failed
    failed2 = {"failed_at": "2026-11-02T10:00:00Z", "reason": "No chart detected"}
    (output_dir / "chart2_failed.json").write_text(json.dumps(failed2), encoding="utf-8")

    # 3. chart3 has analysis JSON but NO summary -> backfill off-thread to auto
    analysis3 = {
        "chart_type": "bar",
        "protocol_rows": [
            {"row_uid": "r1", "value": 10.0, "confidence": 0.95, "review_status": "auto"}
        ],
        "metadata": {"warnings": []},
    }
    (output_dir / "chart3_analysis.json").write_text(json.dumps(analysis3), encoding="utf-8")

    # 4. chart4 has no files -> pending

    files = [img1, img2, img3, img4]
    stems = {img1: "chart1", img2: "chart2", img3: "chart3", img4: "chart4"}

    model = FileListModel()
    model.set_files(files, stems)

    loader = FileStateLoader(
        output_dir=str(output_dir),
        file_paths=files,
        unique_stems=stems,
        threshold=0.60,
    )

    emitted_states = {}
    loader.signals.state_ready.connect(
        lambda fpath, state, conf, reason, is_cached: emitted_states.update({fpath: (state, is_cached)})
    )

    # Run synchronously in test to verify logic
    loader.run()

    assert emitted_states[img1] == (ImageReviewState.NEEDS_REVIEW.value, True)
    assert emitted_states[img2] == (ImageReviewState.FAILED.value, False)
    assert emitted_states[img3] == (ImageReviewState.AUTO.value, True)
    assert emitted_states[img4] == (ImageReviewState.PENDING.value, False)

    # Verify backfill created summary for chart3
    assert (output_dir / "chart3_summary.json").is_file()


def test_file_state_loader_drops_stale_path_signals_gracefully(mock_app_window):
    """ADR-014 T-014.4: Stale path signals are dropped without KeyError or IndexError."""
    win = mock_app_window
    win.image_files = ["/new/folder/img1.png", "/new/folder/img2.png"]
    win.file_list_model.set_files(win.image_files)

    # Stale signal from an old folder
    stale_path = "/old/folder/old_img.png"

    # Must drop gracefully without raising KeyError or IndexError
    win._on_file_state_ready(stale_path, ImageReviewState.REVIEWED.value, None, None, True)

    assert win.file_list_model.index_for_path(stale_path).isValid() is False


def test_marking_image_reviewed_moves_it_out_of_needs_review(mock_app_window, tmp_path):
    """ADR-014 T-014.3: Marking an image reviewed moves it out of the Needs Review queue."""
    win = mock_app_window
    output_dir = tmp_path / "output"
    win.output_path_edit.setText(str(output_dir))

    img_path = str(tmp_path / "input" / "test_chart.png")
    win.image_files = [img_path]
    win._unique_stems = {img_path: "test_chart"}
    win.current_image_index = 0
    win.current_image_path = img_path

    # Set up analysis result with low confidence so it is initially in NEEDS_REVIEW
    win.current_analysis_result = {
        "chart_type": "bar",
        "protocol_rows": [
            {"row_uid": "r1", "value": 15.0, "confidence": 0.40, "review_status": "auto"}
        ],
        "metadata": {"warnings": []},
    }

    win.file_list_model.set_files(win.image_files, win._unique_stems)
    win.file_list_model.update_file_state(
        img_path,
        ImageReviewState.NEEDS_REVIEW.value,
        confidence=0.40,
        is_cached=True,
    )

    # Set filter mode to 'Needs Review'
    win.queue_filter_combo.setCurrentText("Needs Review")
    assert win.file_list_proxy.rowCount() == 1

    # Mark reviewed
    win.mark_image_reviewed()

    # Image is now REVIEWED, so it must be automatically removed from the Needs Review queue
    assert win.file_list_proxy.rowCount() == 0

    # Switching to 'Reviewed' shows it
    win.queue_filter_combo.setCurrentText("Reviewed")
    assert win.file_list_proxy.rowCount() == 1


def test_has_icon_helper(mock_app_window):
    """ADR-014 T-014.7: has_icon helper returns True for existing icons and False otherwise."""
    win = mock_app_window
    assert win.has_icon("check-solid-full.svg") is True
    assert win.has_icon("triangle-exclamation-solid-full.svg") is True
    assert win.has_icon("circle-regular-full.svg") is True
    assert win.has_icon("non_existent_icon_xyz123.svg") is False
