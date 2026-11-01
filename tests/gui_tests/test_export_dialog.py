"""GUI tests for Export Dialog Helper (src/visual/export_dialog_helper.py).

Resolves T-011.1 per ADR-011 and DP-04:
- Default filename derived as <image_stem>_protocol.csv or <image_stem>_protocol_filtered.csv.
- QFileDialog.Option.DontConfirmOverwrite is False (native overwrite confirmation prompt).
- User cancellation or declined overwrite safely aborts without error (resolving NameError: dest is not defined).
- Button text updates reflect active filter scope.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QPushButton, QWidget

from visual.export_dialog_helper import (
    derive_default_export_filename,
    export_protocol_csv_dialog,
    update_export_button_text,
)


class DummyWindow(QWidget):
    """Headless dummy window simulating ModernChartAnalysisApp state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = "/path/to/chart_00001.png"
        self.output_dir = "/tmp/test_out"
        self.status_messages: List[str] = []
        self.current_analysis_result: Dict[str, Any] = {
            "chart_type": "bar",
            "image_file": "chart_00001.png",
            "protocol_rows": [
                {
                    "source_file": "chart_00001.png",
                    "group": "A",
                    "outcome": "Response",
                    "value": 10.0,
                    "review_status": "auto",
                },
                {
                    "source_file": "chart_00001.png",
                    "group": "B",
                    "outcome": "Response",
                    "value": 20.0,
                    "review_status": "corrected",
                },
            ],
        }
        self.session = None

    def update_status(self, msg: str) -> None:
        self.status_messages.append(msg)


class DummySession:
    """Mock ReviewSession v0."""

    def __init__(self, rows: List[Dict[str, Any]], is_filtered: bool = False):
        self._rows = rows
        self._is_filtered = is_filtered
        self.current_analysis_result = {"protocol_rows": rows, "chart_type": "bar"}
        self.current_image_path = "/path/to/session_chart.png"

    def filtered_rows(self) -> List[Dict[str, Any]]:
        return self._rows

    def has_active_filters(self) -> bool:
        return self._is_filtered


# ---------------------------------------------------------------------------
# 1. Filename Derivation Tests
# ---------------------------------------------------------------------------

class TestDefaultFilenameDerivation:
    def test_unfiltered_filename_derivation(self, qapp):
        win = DummyWindow()
        fname = derive_default_export_filename(win, is_filtered=False)
        assert fname == "chart_00001_protocol.csv"

    def test_filtered_filename_derivation(self, qapp):
        win = DummyWindow()
        fname = derive_default_export_filename(win, is_filtered=True)
        assert fname == "chart_00001_protocol_filtered.csv"

    def test_fallback_when_no_image_path(self, qapp):
        win = DummyWindow()
        win.current_image_path = None
        win.current_analysis_result = {}
        fname = derive_default_export_filename(win, is_filtered=False)
        assert fname == "export_protocol.csv"

        fname_filtered = derive_default_export_filename(win, is_filtered=True)
        assert fname_filtered == "export_protocol_filtered.csv"

    def test_session_image_path_derivation(self, qapp):
        win = DummyWindow()
        win.current_image_path = None
        win.session = DummySession([], is_filtered=False)
        fname = derive_default_export_filename(win, is_filtered=False)
        assert fname == "session_chart_protocol.csv"


# ---------------------------------------------------------------------------
# 2. Overwrite Confirmation & Safe Dialog Execution Tests
# ---------------------------------------------------------------------------

class TestExportProtocolCsvDialog:
    def test_dont_confirm_overwrite_is_false(self, qapp, monkeypatch, tmp_path):
        """Native overwrite confirmation prompt must be active: DontConfirmOverwrite is False."""
        win = DummyWindow()
        out_csv = tmp_path / "saved.csv"

        captured_options = []

        def mock_get_save_file_name(parent, caption, directory, filter_str, options=None):
            captured_options.append(options)
            return (str(out_csv), filter_str)

        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_file_name)

        res = export_protocol_csv_dialog(win, default_dir=tmp_path)
        assert res == out_csv
        assert len(captured_options) == 1

        opts = captured_options[0]
        # Verify DontConfirmOverwrite option is NOT set
        if opts is not None:
            assert not bool(opts & QFileDialog.Option.DontConfirmOverwrite)

    def test_export_cancel_safely_aborts_without_name_error(self, qapp, monkeypatch):
        """When user cancels, helper safely aborts and returns None without NameError: dest is not defined."""
        win = DummyWindow()

        # Simulate user cancelling QFileDialog
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: ("", "CSV Files (*.csv)"),
        )

        res = export_protocol_csv_dialog(win)
        assert res is None
        assert len(win.status_messages) == 0

    def test_successful_export_writes_csv_and_updates_status(self, qapp, monkeypatch, tmp_path):
        win = DummyWindow()
        target_csv = tmp_path / "exported_results.csv"

        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(target_csv), "CSV Files (*.csv)"),
        )

        res = export_protocol_csv_dialog(win, default_dir=tmp_path)
        assert res == target_csv
        assert target_csv.exists()

        # Read back CSV
        with open(target_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        # Bar chart dynamic columns
        assert "source_file" in header
        assert "value" in header
        assert len(rows) == 2

        # Status updated
        assert any(str(target_csv) in msg for msg in win.status_messages)

    def test_session_filtered_rows_used_when_session_present(self, qapp, monkeypatch, tmp_path):
        win = DummyWindow()
        session_rows = [
            {"source_file": "chart_00001.png", "group": "OnlyOne", "outcome": "A", "value": 99.0}
        ]
        win.session = DummySession(session_rows, is_filtered=True)
        target_csv = tmp_path / "session_out.csv"

        captured_initial_path = []

        def mock_get_save_file(parent, caption, directory, filter_str, options=None):
            captured_initial_path.append(directory)
            return (str(target_csv), filter_str)

        monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_get_save_file)

        res = export_protocol_csv_dialog(win, default_dir=tmp_path)
        assert res == target_csv
        assert "filtered.csv" in captured_initial_path[0]

        with open(target_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["group"] == "OnlyOne"

    def test_no_rows_shows_info_and_returns_none(self, qapp, monkeypatch):
        win = DummyWindow()
        win.current_analysis_result["protocol_rows"] = []

        info_called = []
        monkeypatch.setattr(
            QMessageBox,
            "information",
            lambda parent, title, text: info_called.append((title, text)),
        )

        save_dialog_called = []
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: save_dialog_called.append(True),
        )

        res = export_protocol_csv_dialog(win)
        assert res is None
        assert len(info_called) == 1
        assert len(save_dialog_called) == 0


# ---------------------------------------------------------------------------
# 3. Button Text Update Tests
# ---------------------------------------------------------------------------

class TestButtonTextUpdate:
    def test_update_export_button_text_filtered(self, qapp):
        btn = QPushButton()
        update_export_button_text(btn, visible_count=12, total_count=48)
        assert btn.text() == "Export CSV (12 of 48 rows)"

    def test_update_export_button_text_unfiltered(self, qapp):
        btn = QPushButton()
        update_export_button_text(btn, visible_count=48, total_count=48)
        assert btn.text() == "Export Results CSV"

    def test_update_export_button_text_none_safe(self, qapp):
        # Should not raise exception
        update_export_button_text(None, visible_count=10, total_count=20)
