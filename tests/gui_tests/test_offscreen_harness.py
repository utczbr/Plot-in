"""
Unit tests for the Offscreen GUI Test Harness (Wave 0 Task T-007.0).

Verifies:
1. Real offscreen ModernChartAnalysisApp creation, window title, attributes.
2. Result injection, _refresh_protocol_rows_from_result(), and proto_status_combo filtering.
3. Dummy window fallback defaults, method binding, and safe session access (no AttributeError).
4. Legacy _DummyWindow compatibility (test_gui_error_flow._DummyWindow.session is None).
5. Fast window creation (< 3.0s threshold per ADR-007).
6. QMessageBox static dialog interceptor.
7. Canonical result payload fixtures for bar, line, pie, box.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QMessageBox, QTableWidget

from main_modern import ModernChartAnalysisApp


def test_offscreen_window_exposes_required_attributes(offscreen_window: ModernChartAnalysisApp):
    """
    AC-1 & AC-2: Offscreen window instantiates cleanly, exposes title 'Plot-in',
    and exposes proto_status_combo, protocol_table, and current_analysis_result.
    """
    assert offscreen_window.windowTitle() == "Plot-in"
    assert hasattr(offscreen_window, "proto_status_combo")
    assert isinstance(offscreen_window.proto_status_combo, QComboBox)
    assert hasattr(offscreen_window, "protocol_table")
    assert isinstance(offscreen_window.protocol_table, QTableWidget)
    assert hasattr(offscreen_window, "current_analysis_result")
    assert offscreen_window.current_analysis_result is None

    # Status combo items must match canonical choices
    items = [offscreen_window.proto_status_combo.itemText(i) for i in range(offscreen_window.proto_status_combo.count())]
    assert items == ["All", "auto", "corrected", "reviewed"]


def test_offscreen_window_inject_and_filter_protocol_rows(
    offscreen_window: ModernChartAnalysisApp,
    sample_result_payload,
):
    """
    AC-3: Injecting a result dict with protocol_rows and calling
    _refresh_protocol_rows_from_result() populates protocol_table
    and allows filtering via proto_status_combo.
    """
    # Use canonical bar payload containing 3 rows: 'auto', 'corrected', 'reviewed'
    payload = sample_result_payload("bar")
    offscreen_window.current_analysis_result = payload

    # Refresh protocol rows in the window
    offscreen_window._refresh_protocol_rows_from_result()

    # Total rows rendered initially (combo is "All")
    table = offscreen_window.protocol_table
    assert table.rowCount() == 3

    # Filter to 'corrected'
    combo = offscreen_window.proto_status_combo
    corrected_idx = combo.findText("corrected")
    assert corrected_idx >= 0
    combo.setCurrentIndex(corrected_idx)
    assert table.rowCount() == 1

    # Filter to 'auto'
    auto_idx = combo.findText("auto")
    assert auto_idx >= 0
    combo.setCurrentIndex(auto_idx)
    assert table.rowCount() == 1

    # Filter to 'reviewed'
    reviewed_idx = combo.findText("reviewed")
    assert reviewed_idx >= 0
    combo.setCurrentIndex(reviewed_idx)
    assert table.rowCount() == 1

    # Filter back to 'All'
    all_idx = combo.findText("All")
    assert all_idx >= 0
    combo.setCurrentIndex(all_idx)
    assert table.rowCount() == 3


def test_dummy_window_session_access_no_attribute_error(dummy_window):
    """
    AC-4: make_dummy_window() / dummy_window provides typed fallbacks
    and stub session = None to prevent AttributeError.
    """
    assert hasattr(dummy_window, "session")
    assert dummy_window.session is None

    assert dummy_window.is_processing is False
    assert dummy_window.current_analysis_result is None
    assert dummy_window.current_image_path is None
    assert dummy_window.current_image_index == -1
    assert dummy_window.image_files == []
    assert dummy_window.status_messages == []

    # Test dummy widgets
    assert dummy_window.progress_bar.visible is False
    dummy_window.progress_bar.setVisible(True)
    assert dummy_window.progress_bar.visible is True

    assert dummy_window.nav_frame.deleted is False
    dummy_window.nav_frame.deleteLater()
    assert dummy_window.nav_frame.deleted is True

    assert dummy_window.proto_status_combo.count() == 4
    assert dummy_window.proto_status_combo.currentText() == "All"

    # Test bound methods on dummy window
    dummy_window.update_status("Hello test")
    assert "Hello test" in dummy_window.status_messages

    assert dummy_window._is_error_result({}) is False
    assert dummy_window._is_error_result({"error": "Failed"}) is True

    # Verify presence of UI helper stubs
    assert hasattr(dummy_window, "_refresh_filter_combos")
    assert hasattr(dummy_window, "_apply_protocol_filters")
    assert hasattr(dummy_window, "_render_protocol_table")


def test_legacy_dummy_window_class_session_protection():
    """
    Ensures _DummyWindow in test_gui_error_flow has session = None so future
    wave migrations touching self.session do not break legacy tests.
    """
    import tests.gui_tests.test_gui_error_flow as tgef

    assert hasattr(tgef, "_DummyWindow")
    dummy = tgef._DummyWindow()
    assert hasattr(dummy, "session")
    assert dummy.session is None


def test_dummy_low_confidence_host_session_protection(qapp):
    """
    AC-4 / Iteration 1 Gate Defect #1 verification:
    Ensures DummyLowConfidenceHost in test_low_confidence_tab has session = None
    so future wave migrations (e.g. T-008.2 dirty guards on load_image_by_index)
    do not raise AttributeError on legacy dummy hosts (ADR-007 line 48).
    """
    import tests.gui_tests.test_low_confidence_tab as tlct

    assert hasattr(tlct, "DummyLowConfidenceHost"), "DummyLowConfidenceHost not found in test_low_confidence_tab"
    assert hasattr(tlct.DummyLowConfidenceHost, "session"), "DummyLowConfidenceHost class missing 'session' attribute"
    assert tlct.DummyLowConfidenceHost.session is None

    # Verify instance access (requires qapp fixture because DummyLowConfidenceHost is a QWidget)
    host = tlct.DummyLowConfidenceHost()
    assert hasattr(host, "session"), "DummyLowConfidenceHost instance missing 'session' attribute"
    assert host.session is None, f"Expected host.session to be None, got {host.session!r}"
    assert getattr(host, "session", None) is None


def test_dummy_window_refresh_protocol_rows_executes_cleanly(dummy_window, sample_result_payload):
    """
    AC-4 / Iteration 1 Gate Defect #2 verification:
    Verifies that make_dummy_window() / dummy_window executes
    _refresh_protocol_rows_from_result() without raising AttributeError,
    properly transforming current_analysis_result['protocol_rows'].
    """
    from tests.gui_tests.conftest import make_dummy_window

    # 1. Calling with None current_analysis_result returns early without error
    dummy_window.current_analysis_result = None
    dummy_window._refresh_protocol_rows_from_result()
    assert dummy_window.current_analysis_result is None

    # 2. Calling with empty dict returns early without error
    dummy_window.current_analysis_result = {}
    dummy_window._refresh_protocol_rows_from_result()
    assert dummy_window.current_analysis_result == {}

    # 3. Calling with populated canonical bar payload builds protocol_rows
    payload = sample_result_payload("bar")
    dummy_window.current_analysis_result = payload
    dummy_window._refresh_protocol_rows_from_result()

    assert "protocol_rows" in dummy_window.current_analysis_result
    assert len(dummy_window.current_analysis_result["protocol_rows"]) == 3

    # 4. Explicit verification of helper stub existence and callable signatures
    assert hasattr(dummy_window, "_refresh_filter_combos")
    assert hasattr(dummy_window, "_apply_protocol_filters")
    assert hasattr(dummy_window, "_render_protocol_table")

    dummy_window._refresh_filter_combos()
    dummy_window._apply_protocol_filters()
    dummy_window._render_protocol_table()
    dummy_window._render_protocol_table([])

    # 5. Verification on factory function make_dummy_window across other chart types
    for chart_type, expected_count in [("line", 2), ("pie", 2), ("box", 1)]:
        fresh_dummy = make_dummy_window()
        fresh_dummy.current_analysis_result = sample_result_payload(chart_type)
        fresh_dummy._refresh_protocol_rows_from_result()
        assert len(fresh_dummy.current_analysis_result["protocol_rows"]) == expected_count


def test_window_creation_speed_under_threshold(window_factory):
    """
    ADR-007 performance constraint: offscreen window creation must run < 3.0s in CI.
    (Typical benchmark is ~0.2s).
    """
    start_time = time.perf_counter()
    window = window_factory()
    duration = time.perf_counter() - start_time

    assert duration < 3.0, f"Window creation took {duration:.3f}s, exceeding 3.0s limit"
    assert window is not None


def test_window_factory_multiple_instances(window_factory):
    """
    Verifies creating multiple isolated windows sequentially or together works cleanly.
    """
    win1 = window_factory(test_tag="win1")
    win2 = window_factory(test_tag="win2")

    assert win1.test_tag == "win1"
    assert win2.test_tag == "win2"
    assert win1 is not win2


def test_monkeypatch_message_box_interceptor(monkeypatch_message_box):
    """
    Verifies monkeypatch_message_box intercepts static dialogs and records calls.
    """
    res_info = QMessageBox.information(None, "Info Title", "Info Text")
    assert res_info == QMessageBox.StandardButton.Ok

    res_warn = QMessageBox.warning(None, "Warn Title", "Warn Text")
    assert res_warn == QMessageBox.StandardButton.Ok

    res_crit = QMessageBox.critical(None, "Crit Title", "Crit Text")
    assert res_crit == QMessageBox.StandardButton.Ok

    res_quest = QMessageBox.question(None, "Quest Title", "Do you want to proceed?")
    assert res_quest == QMessageBox.StandardButton.Yes

    assert len(monkeypatch_message_box) == 4
    assert monkeypatch_message_box[0]["method"] == "information"
    assert monkeypatch_message_box[0]["title"] == "Info Title"
    assert monkeypatch_message_box[1]["method"] == "warning"
    assert monkeypatch_message_box[2]["method"] == "critical"
    assert monkeypatch_message_box[3]["method"] == "question"


def test_canonical_sample_result_payloads(sample_result_payload):
    """
    Verifies sample_result_payload provides valid canonical payloads for bar, line, pie, box.
    """
    # Test dictionary default (bar)
    assert sample_result_payload["chart_type"] == "bar"
    assert len(sample_result_payload["protocol_rows"]) == 3

    # Test callable access
    bar_payload = sample_result_payload("bar")
    assert bar_payload["chart_type"] == "bar"

    line_payload = sample_result_payload("line")
    assert line_payload["chart_type"] == "line"
    assert "series_id" in line_payload["protocol_rows"][0]

    pie_payload = sample_result_payload("pie")
    assert pie_payload["chart_type"] == "pie"
    assert "percent" in pie_payload["elements"][0]

    box_payload = sample_result_payload("box")
    assert box_payload["chart_type"] == "box"
    assert "median" in box_payload["elements"][0]

    with pytest.raises(KeyError):
        sample_result_payload("unsupported_type")


def test_mock_pipeline_and_context_fixtures(mock_pipeline, stub_app_context, mock_context):
    """
    Verifies mock_pipeline and stub_app_context behave as expected.
    """
    assert stub_app_context is mock_context
    assert stub_app_context.data_manager.get_context() == {}

    res = mock_pipeline.run_single_analysis()
    assert res["chart_type"] == "bar"
