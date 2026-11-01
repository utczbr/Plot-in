"""Tests for T-010.1 BatchResult, failure markers, cancel handling, and UI reset."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.analysis_manager import AnalysisManager
from core.input_resolver import ResolvedAsset
from core.result_utils import FailedItem
from visual.review_session import ProcessingState, ReviewSession
from visual.workers import BatchAnalysisThread, BatchResult


def _make_asset(tmp_path: Path, name: str) -> ResolvedAsset:
    img_path = tmp_path / name
    img_path.write_bytes(b"fake_image_bytes")
    return ResolvedAsset(
        image_path=img_path,
        source_document=None,
        page_index=None,
        figure_id=img_path.stem,
        confidence_info={"classification": 0.9, "average": 0.9, "is_low_confidence": False},
    )


def test_batch_file_2_of_3_fails_and_writes_failure_marker(tmp_path: Path):
    """
    T-010.1 Acceptance:
    Given file 2 of 3 fails (returns None or raises), then processed == 2,
    failed == [FailedItem(file2, reason)], and <stem>_failed.json is written.
    A subsequent successful run clears the stale <stem>_failed.json marker.
    """
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    assets = [
        _make_asset(in_dir, "chart1.png"),
        _make_asset(in_dir, "chart2.png"),
        _make_asset(in_dir, "chart3.png"),
    ]

    mock_pipeline = MagicMock()

    def _run_side_effect(image_input, **kwargs):
        if Path(image_input).name == "chart2.png":
            return None  # Failure mode: None returned
        return {"chart_type": "bar", "elements": [{"label": "A", "value": 1.0}]}

    mock_pipeline.run.side_effect = _run_side_effect

    am = AnalysisManager(data_manager=MagicMock(), image_manager=MagicMock())
    am.set_models(MagicMock())

    ctx = MagicMock()
    ctx.analysis_manager = am
    ctx.model_manager = MagicMock()

    with patch.object(am, "_create_pipeline", return_value=mock_pipeline), \
         patch("core.input_resolver.resolve_input_assets", return_value=assets):
        bat = BatchAnalysisThread(
            input_path=str(in_dir),
            output_path=str(out_dir),
            models_dir=str(tmp_path / "models"),
            context=ctx,
        )
        finished_results = []
        bat.batch_finished.connect(lambda res: finished_results.append(res))
        bat.run()

    assert len(finished_results) == 1
    res: BatchResult = finished_results[0]
    assert res.processed == 2
    assert res.total == 3
    assert res.cancelled is False
    assert res.fatal_error is None
    assert len(res.failed) == 1
    assert res.failed[0] == FailedItem(
        file=str(assets[1].image_path),
        reason="Analysis returned no results",
    )

    # Verify failure marker was persisted on disk for chart2
    failed_marker = out_dir / "chart2_failed.json"
    assert failed_marker.exists()

    # Now chart2 succeeds on a subsequent run -> failure marker is cleared
    mock_pipeline.run.side_effect = lambda image_input, **kwargs: {
        "chart_type": "bar",
        "elements": [{"label": "A", "value": 1.0}],
    }
    with patch.object(am, "_create_pipeline", return_value=mock_pipeline), \
         patch("core.input_resolver.resolve_input_assets", return_value=[assets[1]]):
        bat2 = BatchAnalysisThread(
            input_path=str(in_dir),
            output_path=str(out_dir),
            models_dir=str(tmp_path / "models"),
            context=ctx,
        )
        bat2.run()

    assert not failed_marker.exists()


def test_batch_cancel_after_file_1_resets_ui(tmp_path: Path):
    """
    T-010.1 Acceptance:
    Given cancel after file 1, then cancelled is True, and the UI resets
    (is_processing == False, session.processing == IDLE, progress_bar hidden, Run enabled).
    """
    from visual.file_list_controller import FileListCacheMixin

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    assets = [
        _make_asset(in_dir, "chart1.png"),
        _make_asset(in_dir, "chart2.png"),
        _make_asset(in_dir, "chart3.png"),
    ]

    am = AnalysisManager(data_manager=MagicMock(), image_manager=MagicMock())
    am.set_models(MagicMock())

    ctx = MagicMock()
    ctx.analysis_manager = am
    ctx.model_manager = MagicMock()

    bat = BatchAnalysisThread(
        input_path=str(in_dir),
        output_path=str(out_dir),
        models_dir=str(tmp_path / "models"),
        context=ctx,
    )

    mock_pipeline = MagicMock()

    def _run_and_cancel_after_first(image_input, **kwargs):
        bat.cancel()
        return {"chart_type": "bar", "elements": [{"label": "A", "value": 1.0}]}

    mock_pipeline.run.side_effect = _run_and_cancel_after_first

    class _ControllerHost(FileListCacheMixin):
        def __init__(self):
            self.session = ReviewSession()
            self.session.processing = ProcessingState.BATCH
            self.is_processing = True
            self.progress_bar = MagicMock()
            self.run_batch_btn = MagicMock()
            self.update_status = MagicMock()
            self.image_files = [str(a.image_path) for a in assets]
            self.current_image_index = 0

        def populate_file_list(self):
            pass

        def _refresh_cache_from_output_dir(self):
            pass

        def _refresh_file_list_button_states(self):
            pass

        def refresh_low_confidence_tab(self):
            pass

        def load_image_by_index(self, idx):
            pass

    host = _ControllerHost()
    bat.batch_finished.connect(host.on_batch_finished)

    with patch.object(am, "_create_pipeline", return_value=mock_pipeline), \
         patch("core.input_resolver.resolve_input_assets", return_value=assets):
        bat.run()

    res: BatchResult = host._last_batch_result
    assert res.cancelled is True
    assert res.processed == 1
    assert res.total == 3
    assert host.is_processing is False
    assert host.session.processing == ProcessingState.IDLE
    host.progress_bar.setVisible.assert_called_with(False)
    host.run_batch_btn.setEnabled.assert_called_with(True)
    host.update_status.assert_called_with("Cancelled after 1 of 3")


def test_single_image_failure_and_recovery_markers(offscreen_window, tmp_path: Path, monkeypatch):
    """
    T-010.1 Acceptance:
    Single-image failure path calls write_failure_marker(output_dir, stem, reason),
    and a subsequent successful single-image analysis clears the stale marker.
    """
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.critical", lambda *a, **kw: None)
    img_path = tmp_path / "single_chart.png"
    img_path.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    offscreen_window.current_image_path = str(img_path)
    offscreen_window.output_path_edit.setText(str(out_dir))

    # 1. Error result writes failure marker
    offscreen_window._on_analysis_complete({"error": "Model failure"})
    marker = out_dir / "single_chart_failed.json"
    assert marker.exists()

    # 2. Successful result clears failure marker
    offscreen_window.current_image_path = str(img_path)
    monkeypatch.setattr(offscreen_window, "_update_ui_with_results", lambda: None)
    monkeypatch.setattr(offscreen_window, "update_displayed_image", lambda *a, **kw: None)
    monkeypatch.setattr(offscreen_window, "_setup_navigation_controls", lambda: None)
    offscreen_window._on_analysis_complete({"chart_type": "bar", "elements": []})
    assert not marker.exists()
