"""
Tests for image-level cache uniqueness and Clear Cache functionality.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from core.export_manager import ExportManager
from visual.draft_store import clear_drafts, get_draft_path, write_draft
from visual.file_list_controller import (
    FileListCacheMixin,
    build_image_identity_metadata,
    compute_unique_image_stem,
    verify_identity_metadata,
)
from visual.file_list_model import FileItem, FileListModel, ImageReviewState
from visual.settings_dialog import SettingsDialog


def _create_dummy_image(path: Path, size=(120, 80), color="blue") -> Path:
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# 1. compute_unique_image_stem & identity metadata
# ---------------------------------------------------------------------------

def test_compute_unique_image_stem_real_image(tmp_path: Path):
    img_path = _create_dummy_image(tmp_path / "chart_sample.png", size=(150, 90))
    stem = compute_unique_image_stem(img_path)
    file_size = img_path.stat().st_size
    assert stem == f"chart_sample_{file_size}_150x90"


def test_compute_unique_image_stem_non_existent(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.png"
    assert compute_unique_image_stem(missing_path) == "does_not_exist"


def test_build_and_verify_identity_metadata(tmp_path: Path):
    img_path = _create_dummy_image(tmp_path / "identity_test.png", size=(200, 100))
    meta = build_image_identity_metadata(img_path)
    assert meta["file_name"] == "identity_test.png"
    assert meta["width"] == 200
    assert meta["height"] == 100
    assert meta["file_size"] == img_path.stat().st_size
    assert meta["unique_stem"] == f"identity_test_{meta['file_size']}_200x100"

    # Verification passes with matching metadata
    assert verify_identity_metadata(img_path, meta) is True

    # Verification fails if file size is altered
    tampered_meta = dict(meta, file_size=999999)
    assert verify_identity_metadata(img_path, tampered_meta) is False

    # Verification fails if dimensions are altered
    tampered_dims = dict(meta, width=300, height=300)
    assert verify_identity_metadata(img_path, tampered_dims) is False

    # Verification returns True if stored_identity is None or empty dict
    assert verify_identity_metadata(img_path, None) is True
    assert verify_identity_metadata(img_path, {}) is True


# ---------------------------------------------------------------------------
# 2. FileListModel.reset_all_states
# ---------------------------------------------------------------------------

def test_file_list_model_reset_all_states(qapp):
    model = FileListModel()
    files = ["/path/a.png", "/path/b.png"]
    stems = {"/path/a.png": "a", "/path/b.png": "b"}
    model.set_files(files, stems)

    model.update_file_state("/path/a.png", state="reviewed", confidence=0.95, is_cached=True)
    model.update_file_state("/path/b.png", state="failed", failure_reason="Corrupt", is_cached=False)

    assert model.get_item(0).state == "reviewed"
    assert model.get_item(0).is_cached is True
    assert model.get_item(1).state == "failed"

    model.reset_all_states()

    item0 = model.get_item(0)
    item1 = model.get_item(1)
    assert item0.state == ImageReviewState.PENDING.value
    assert item0.confidence is None
    assert item0.is_cached is False
    assert item1.state == ImageReviewState.PENDING.value
    assert item1.failure_reason is None
    assert item1.is_cached is False


# ---------------------------------------------------------------------------
# 3. draft_store.clear_drafts
# ---------------------------------------------------------------------------

def test_clear_drafts(tmp_path: Path):
    out_dir = tmp_path / "out"
    d1 = get_draft_path(out_dir, "stem_01")
    d2 = get_draft_path(out_dir, "stem_02")
    payload = {
        "schema_version": "1.0",
        "image_file": "chart.png",
        "saved_at": "2026-11-02T12:00:00Z",
        "result": {"chart_type": "bar"},
        "editor_detections": [],
    }
    write_draft(d1, payload)
    write_draft(d2, payload)

    assert d1.is_file()
    assert d2.is_file()

    deleted = clear_drafts(out_dir)
    assert deleted == 2
    assert not (out_dir / ".drafts").exists()

    # Calling on non-existent dir returns 0
    assert clear_drafts(out_dir) == 0


# ---------------------------------------------------------------------------
# 4. SettingsDialog Clear Cache signal
# ---------------------------------------------------------------------------

def test_settings_dialog_clear_cache_signal(qapp):
    dialog = SettingsDialog()
    received = []
    dialog.clear_cache_requested.connect(lambda: received.append(True))

    assert hasattr(dialog, "basic_clear_cache_btn")
    assert hasattr(dialog, "clear_cache_btn")

    dialog.basic_clear_cache_btn.click()
    assert len(received) == 1

    dialog.clear_cache_btn.click()
    assert len(received) == 2


# ---------------------------------------------------------------------------
# 5. FileListCacheMixin.clear_cache double confirmation workflow
# ---------------------------------------------------------------------------

class DummyApp(FileListCacheMixin):
    def __init__(self, out_dir: str):
        self._benchmark_result_cache = {}
        self.output_path_edit = MagicMock()
        self.output_path_edit.text.return_value = out_dir
        self.input_path_edit = MagicMock()
        self.input_path_edit.text.return_value = ""
        self.file_list_model = FileListModel()
        self.image_files = []
        self._file_state_loader = None
        self._file_state_loader_gen = 0
        self.current_image_index = 0
        self.current_image_path = "some_path.png"
        self.current_analysis_result = {"chart_type": "bar"}
        self._status_messages = []

    def update_status(self, msg: str):
        self._status_messages.append(msg)

    def _refresh_file_list_button_states(self):
        pass

    def refresh_low_confidence_tab(self):
        pass

    def _clear_display(self):
        pass

    def _sync_navigation_bar(self):
        pass

    def populate_file_list(self, auto_select=False):
        pass


def test_clear_cache_cancelled_on_step1(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    app = DummyApp(str(out_dir))

    with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.No):
        result = app.clear_cache()
        assert result is False


def test_clear_cache_cancelled_on_step2(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    app = DummyApp(str(out_dir))

    with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Yes):
        with patch.object(QMessageBox, "exec", return_value=None):
            with patch.object(QMessageBox, "clickedButton", return_value=None):
                result = app.clear_cache()
                assert result is False


def test_clear_cache_confirmed_deletes_sidecars(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Pre-populate cached files
    (out_dir / "img1_analysis.json").write_text("{}", encoding="utf-8")
    (out_dir / "img1_summary.json").write_text("{}", encoding="utf-8")
    (out_dir / "img1_protocol_corrections.json").write_text("{}", encoding="utf-8")
    (out_dir / "img1_edited_detections.json").write_text("{}", encoding="utf-8")
    (out_dir / "img1_annotated.png").write_bytes(b"dummy_png")

    # Draft
    draft_p = get_draft_path(out_dir, "img1")
    write_draft(draft_p, {
        "schema_version": "1.0",
        "image_file": "img1.png",
        "saved_at": "2026-11-02T12:00:00Z",
        "result": {"chart_type": "bar"},
        "editor_detections": [],
    })
    assert draft_p.is_file()

    app = DummyApp(str(out_dir))
    app._benchmark_result_cache["img1.png"] = str(out_dir / "img1_analysis.json")

    mock_delete_all_btn = MagicMock()

    with patch("visual.file_list_controller.QMessageBox") as MockMsgBox:
        mock_box_inst = MagicMock()
        MockMsgBox.return_value = mock_box_inst
        MockMsgBox.warning.return_value = MockMsgBox.StandardButton.Yes
        mock_box_inst.addButton.return_value = mock_delete_all_btn
        mock_box_inst.clickedButton.return_value = mock_delete_all_btn

        res = app.clear_cache()
        assert res is True

    # Files must be deleted
    assert not (out_dir / "img1_analysis.json").exists()
    assert not (out_dir / "img1_summary.json").exists()
    assert not (out_dir / "img1_protocol_corrections.json").exists()
    assert not (out_dir / "img1_edited_detections.json").exists()
    assert not (out_dir / "img1_annotated.png").exists()
    assert not draft_p.exists()

    # In-memory caches cleared
    assert len(app._benchmark_result_cache) == 0
    assert app.current_image_index == -1
    assert app.current_image_path is None
    assert app.current_analysis_result is None
    assert "Cache cleared" in app._status_messages


# ---------------------------------------------------------------------------
# 6. ExportManager stem resolution
# ---------------------------------------------------------------------------

def test_export_manager_unique_stem_resolution(tmp_path: Path):
    result = {
        "_image_identity": {
            "unique_stem": "chart_999_800x600",
            "file_size": 999,
        },
        "image_file": "chart.png",
        "chart_type": "bar",
        "review_status": "corrected",
        "protocol_rows": [{"row_index": 0, "review_status": "corrected"}],
    }

    out_dir = tmp_path / "out"
    path = ExportManager.write_protocol_corrections_sidecar(result, out_dir)
    assert path is not None
    assert path.name == "chart_999_800x600_protocol_corrections.json"
    assert (out_dir / "chart_999_800x600_edited_detections.json").is_file()
