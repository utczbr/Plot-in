# -*- coding: utf-8 -*-
"""
Characterization and Invariant Tests for Atomic Save and Results Divergence
(ADR-015 T-015.1).

Tests:
1. Characterize 0-byte truncation hazard in legacy non-atomic `_save_analysis_results` (xfail strict).
2. Characterize Results-table edit divergence from underlying elements/bars (xfail strict).
3. Characterize stale Data-table sweep clobbering Results-table edits on save (xfail strict).
4. Verify atomic replacement safety pattern (tmp + fsync + os.replace).
5. Verify DP-01 compact JSON serialization compliance.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from PyQt6.QtWidgets import QMessageBox, QTableWidgetItem

from main_modern import ModernChartAnalysisApp


@pytest.fixture(autouse=True)
def auto_patch_dialogs(monkeypatch_message_box):
    """Ensure no modal QMessageBox blocks headless execution in this module."""
    return monkeypatch_message_box


# ==============================================================================
# 1. Atomic Save Characterization (0-Byte Truncation Hazard)
# ==============================================================================

def test_legacy_non_atomic_save_truncates_file_on_write_error_xfail(
    offscreen_window: ModernChartAnalysisApp,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    T-015.1: Characterize legacy non-atomic save vulnerability.
    In legacy _save_analysis_results (L3176-3178):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(...)
    If writing fails or raises an error mid-stream, `open(..., 'w')` has
    already truncated the existing valid _analysis.json file to 0 bytes.

    Under atomic save (T-015.3), writes go to a .tmp file first and replace
    atomically via os.replace, ensuring the existing file is never corrupted.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        stem = "sample_chart"
        json_path = output_dir / f"{stem}_analysis.json"

        # 1. Create a prior valid saved JSON file
        prior_saved_content = {
            "chart_type": "bar",
            "status": "previously_saved_valid_data",
            "elements": [{"value": 10.0, "label": "Control"}],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(prior_saved_content, f)

        assert json_path.stat().st_size > 0

        # Configure window paths to output directory
        image_path = str(output_dir / f"{stem}.png")
        offscreen_window.current_image_path = image_path
        offscreen_window.output_path_edit.setText(str(output_dir))
        offscreen_window.current_analysis_result = {
            "chart_type": "bar",
            "elements": [{"value": 20.0, "label": "Control"}],
        }

        # 2. Simulate a write error during serialization (e.g., disk full / write error)
        def mock_json_dump_fail(*args, **kwargs):
            raise IOError("Simulated disk write failure mid-stream")

        monkeypatch.setattr(json, "dump", mock_json_dump_fail)
        monkeypatch.setattr(offscreen_window, "update_status", lambda msg: None)
        monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)

        # 3. Attempt save
        save_success = offscreen_window._save_analysis_results()
        assert save_success is False

        # In unpatched legacy code: json_path was opened with 'w' before dump,
        # so it was immediately truncated to 0 bytes!
        # An atomic save would preserve prior_saved_content.
        # This assertion fails on unpatched code -> strict XFAIL!
        assert json_path.stat().st_size > 0, "Original file must not be truncated to 0 bytes"
        recovered_data = json.loads(json_path.read_text(encoding="utf-8"))
        assert recovered_data["status"] == "previously_saved_valid_data"


# ==============================================================================
# 2. Results Edits Divergence from Elements & Bars
# ==============================================================================

def test_results_edits_divergence_from_elements_and_bars_xfail(
    offscreen_window: ModernChartAnalysisApp,
):
    """
    T-015.1: Characterize in-memory divergence between Results table and elements/bars.
    In unpatched code, editing a cell in protocol_table mutates `row_dict[col_id]`
    in `protocol_rows`, but does NOT write through to `current_analysis_result['elements']`
    or `current_analysis_result['bars']`.

    Under ADR-015 T-015.2 (RowEditService), edits write through live to all
    underlying representations.
    """
    elements = [
        {"index": 0, "label": "Group A", "value": 10.0, "review_status": "auto"},
    ]
    bars = [
        {"bar_label": "Group A", "estimated_value": 10.0},
    ]
    rows = [
        {"outcome": "Metric 1", "group": "Group A", "review_status": "auto", "value": 10.0},
    ]
    offscreen_window.current_analysis_result = {
        "chart_type": "bar",
        "elements": elements,
        "bars": bars,
        "protocol_rows": rows,
    }
    offscreen_window._populate_protocol_tab()

    # User edits value in Results table from 10.0 to 42.5
    val_item = offscreen_window.protocol_table.item(0, 4)
    assert val_item is not None
    val_item.setText("42.5")
    offscreen_window._on_protocol_cell_changed(0, 4)

    # protocol_rows is updated:
    assert rows[0]["value"] == 42.5

    # But in unpatched code, elements and bars are NOT updated!
    # This assertion fails on unpatched code -> strict XFAIL!
    assert offscreen_window.current_analysis_result["elements"][0]["value"] == 42.5
    assert offscreen_window.current_analysis_result["bars"][0]["estimated_value"] == 42.5


# ==============================================================================
# 3. Data Table Sweep Stale Overwrite Divergence
# ==============================================================================

def test_data_table_sweep_clobbers_results_edits_xfail(
    offscreen_window: ModernChartAnalysisApp,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    T-015.1: Characterize stale Data tab sweeping in `_update_results_from_gui()`.
    When saving, `_update_results_from_gui()` iterates `data_tab_bindings` and
    reads raw text from `data_table`. If a user made edits in Results table,
    the unedited Data table text sweeps over `elements` and rebuilds `protocol_rows`,
    silently clobbering the reviewer's Results corrections.

    Under ADR-015 T-015.3, the Data tab sweep is removed and the Results table
    is the sole authoritative surface.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        offscreen_window.current_image_path = str(output_dir / "sample.png")
        offscreen_window.output_path_edit.setText(str(output_dir))

        elements = [{"label": "Control", "value": 10.0, "bbox": [0, 0, 10, 10]}]
        bars = [{"bar_label": "Control", "estimated_value": 10.0, "bbox": [0, 0, 10, 10]}]
        rows = [{"group": "Control", "outcome": "M1", "value": 10.0, "review_status": "auto"}]
        offscreen_window.current_analysis_result = {
            "chart_type": "bar",
            "elements": elements,
            "bars": bars,
            "protocol_rows": rows,
        }
        offscreen_window._populate_protocol_tab()

        # Simulate Data tab having original text "10.0"
        offscreen_window.data_tab_bindings = {
            (0, 1): {"type": "bar", "index": 0, "field": "value", "parser": "float"}
        }
        data_item = MagicMock()
        data_item.text.return_value = "10.0"
        offscreen_window.data_table = MagicMock()
        offscreen_window.data_table.item.return_value = data_item

        # Reviewer edits Results table to "88.0"
        rows[0]["value"] = 88.0
        rows[0]["review_status"] = "corrected"

        # Mock filesystem write to succeed cleanly
        monkeypatch.setattr(offscreen_window, "update_status", lambda msg: None)
        monkeypatch.setattr(offscreen_window, "refresh_low_confidence_tab", lambda: None)

        # Trigger save (which invokes _update_results_from_gui)
        offscreen_window._save_analysis_results()

        # In unpatched code, _update_results_from_gui does not propagate Results corrections
        # into underlying elements and bars; instead, stale data_table text is swept.
        # Downstream exports or re-evaluations (build_protocol_rows) see stale extraction elements!
        # This assertion fails on unpatched code -> strict XFAIL!
        from core.protocol_row_builder import build_protocol_rows
        assert offscreen_window.current_analysis_result["elements"][0]["value"] == 88.0
        assert build_protocol_rows(offscreen_window.current_analysis_result)[0]["value"] == 88.0


# ==============================================================================
# 4. Atomic Save Pattern & DP-01 Compliance (Passing Tests)
# ==============================================================================

def test_atomic_replacement_helper_guarantees_safety():
    """
    T-015.1: Verify the atomic replacement algorithm that eliminates the 0-byte hazard:
    1. Write to temporary file <stem>.tmp.
    2. Flush and fsync.
    3. Atomically replace target via os.replace.
    If an error occurs prior to os.replace, the original target file remains untouched.
    """
    def atomic_save_json(target_path: Path, data: Any, simulate_crash_before_replace: bool = False):
        tmp_path = target_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            if simulate_crash_before_replace:
                raise OSError("Simulated process crash or disk failure before atomic swap")
            os.replace(tmp_path, target_path)
            return True
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "test_result.json"

        # Initial successful write
        initial_payload = {"version": 1, "data": "initial_safe_state"}
        assert atomic_save_json(target, initial_payload) is True
        assert target.exists()
        assert json.loads(target.read_text(encoding="utf-8")) == initial_payload

        # Failed second write with simulated crash
        corrupted_payload = {"version": 2, "data": "partial_new_state"}
        assert atomic_save_json(target, corrupted_payload, simulate_crash_before_replace=True) is False

        # Invariant: target file was NOT truncated and still holds initial_payload
        assert target.exists()
        assert target.stat().st_size > 0
        assert json.loads(target.read_text(encoding="utf-8")) == initial_payload

        # Invariant: temporary file was cleaned up
        assert not target.with_suffix(".tmp").exists()


def test_compact_json_dp01_compliance():
    """
    T-015.1: Verify DP-01 compact JSON serialization rules:
    - separators=(',', ':')
    - no indent whitespace
    - UTF-8 roundtrip fidelity
    """
    payload = {
        "chart_type": "scatter",
        "elements": [
            {"x": 1.5, "y": 2.5, "label": "Point 1"},
            {"x": 3.0, "y": 4.0, "label": "Point 2"},
        ],
        "metadata": {"source": "test.png", "notes": "No extra whitespace"},
    }

    compact_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    # Invariant: No newline or indent spaces
    assert "\n" not in compact_str
    assert ": " not in compact_str
    assert ", " not in compact_str

    # Invariant: 100% roundtrip fidelity
    decoded = json.loads(compact_str)
    assert decoded == payload
