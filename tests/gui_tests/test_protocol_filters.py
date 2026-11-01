# -*- coding: utf-8 -*-
"""
Tests for Protocol Filters Lifecycle and Strangler Wiring (ADR-007 T-007.1, T-007.2).
"""

from __future__ import annotations

import platform
from unittest.mock import MagicMock
import pytest

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from main_modern import ModernChartAnalysisApp
from visual.qt_utils import safe_combo_populate


# Use offscreen_window fixture from conftest


def test_status_combo_survives_clear_all_results(offscreen_window):
    """
    T-007.1: _clear_all_results must NOT clear or disable proto_status_combo.
    It must reset index to 0 ('All') and keep all 4 static options enabled.
    """
    assert offscreen_window.proto_status_combo.isEnabled() is True
    assert offscreen_window.proto_status_combo.count() == 4
    items = [offscreen_window.proto_status_combo.itemText(i) for i in range(4)]
    assert items == ["All", "auto", "corrected", "reviewed"]

    # Select a non-default index
    offscreen_window.proto_status_combo.setCurrentIndex(2)  # 'corrected'
    assert offscreen_window.proto_status_combo.currentText() == "corrected"

    # Clear results
    offscreen_window._clear_all_results()

    # Must still be enabled, have 4 items, and be reset to index 0 ("All")
    assert offscreen_window.proto_status_combo.isEnabled() is True
    assert offscreen_window.proto_status_combo.count() == 4
    assert offscreen_window.proto_status_combo.currentIndex() == 0
    assert offscreen_window.proto_status_combo.currentText() == "All"


def test_status_combo_resets_with_signals_blocked(offscreen_window):
    """
    T-007.1: _clear_all_results resets proto_status_combo with signals blocked.
    """
    offscreen_window.proto_status_combo.setCurrentIndex(2)

    signal_emissions = []
    offscreen_window.proto_status_combo.currentIndexChanged.connect(
        lambda idx: signal_emissions.append(idx)
    )

    offscreen_window._clear_all_results()

    assert offscreen_window.proto_status_combo.currentIndex() == 0
    assert signal_emissions == []


def test_outcome_and_group_combos_stay_enabled_on_clear(offscreen_window):
    """
    T-007.1: Outcome and group combos reset to ["All"] (never []) so safe_combo_populate
    keeps them enabled.
    """
    offscreen_window._clear_all_results()

    assert offscreen_window.proto_outcome_combo.isEnabled() is True
    assert offscreen_window.proto_outcome_combo.count() == 1
    assert offscreen_window.proto_outcome_combo.itemText(0) == "All"

    assert offscreen_window.proto_group_combo.isEnabled() is True
    assert offscreen_window.proto_group_combo.count() == 1
    assert offscreen_window.proto_group_combo.itemText(0) == "All"


def test_filter_combos_reset_on_new_image(offscreen_window):
    """
    T-007.1: Filter selection resets to 'All' when a new image loads.
    """
    # 1. Provide first analysis result
    rows_1 = [
        {"outcome": "Outcome 1", "group": "G1", "review_status": "auto", "value": 1.0},
        {"outcome": "Outcome 2", "group": "G2", "review_status": "corrected", "value": 2.0},
    ]
    offscreen_window.current_analysis_result = {
        "chart_type": "bar",
        "protocol_rows": rows_1,
    }
    offscreen_window._populate_protocol_tab()

    # Change combo selections
    idx_corr = offscreen_window.proto_status_combo.findText("corrected")
    assert idx_corr > 0
    offscreen_window.proto_status_combo.setCurrentIndex(idx_corr)
    assert offscreen_window.session.filters["status"] == "corrected"

    # 2. Load second analysis result (simulating new image)
    rows_2 = [
        {"outcome": "Outcome 3", "group": "G3", "review_status": "auto", "value": 3.0},
    ]
    offscreen_window.current_analysis_result = {
        "chart_type": "bar",
        "protocol_rows": rows_2,
    }
    offscreen_window._update_ui_with_results()
    offscreen_window._populate_protocol_tab()

    # Combos and session filters must be reset to "All"
    assert offscreen_window.proto_status_combo.currentText() == "All"
    assert offscreen_window.proto_status_combo.currentIndex() == 0
    assert offscreen_window.session.filters["status"] == "All"
    assert offscreen_window.session.filters["outcome"] == "All"
    assert offscreen_window.session.filters["group"] == "All"


def test_combo_forwarding_to_session_set_filter(offscreen_window):
    """
    T-007.2: Combos become views: currentIndexChanged forwards to session.set_filter,
    and session.filters_changed triggers _apply_protocol_filters.
    """
    rows = [
        {"outcome": "Alpha", "group": "Grp1", "review_status": "auto", "value": 10.0},
        {"outcome": "Beta", "group": "Grp2", "review_status": "corrected", "value": 20.0},
    ]
    offscreen_window.current_analysis_result = {
        "chart_type": "bar",
        "protocol_rows": rows,
    }
    offscreen_window._populate_protocol_tab()

    # Changing status combo updates session filter
    offscreen_window.proto_status_combo.setCurrentText("corrected")
    assert offscreen_window.session.filters["status"] == "corrected"

    # Protocol table should reflect filtered rows
    assert offscreen_window.protocol_table.rowCount() == 1
    val_item = offscreen_window.protocol_table.item(0, 4)  # 'value' column in bar chart
    assert val_item is not None and "20" in val_item.text()
    status_item = offscreen_window.protocol_table.item(0, 9)  # 'review_status' column
    assert status_item is not None and status_item.text() == "corrected"

    # Changing outcome combo updates session filter
    offscreen_window.proto_outcome_combo.setCurrentText("Alpha")
    assert offscreen_window.session.filters["outcome"] == "Alpha"
    # Status is 'corrected' and outcome is 'Alpha' -> 0 matching rows
    assert offscreen_window.protocol_table.rowCount() == 0


def test_macos_async_combo_race_guard(offscreen_window, monkeypatch):
    """
    T-007.2: macOS async combo race guard.
    safe_combo_populate defers combo updates via QTimer.singleShot(0, ...) on Darwin.
    _refresh_filter_combos resets session.filters synchronously BEFORE safe_combo_populate,
    and _apply_protocol_filters reads session.filtered_rows() directly.
    """
    # Force Darwin mode for safe_combo_populate
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    rows = [
        {"outcome": "X1", "group": "Y1", "review_status": "auto", "value": 100.0},
        {"outcome": "X2", "group": "Y2", "review_status": "corrected", "value": 200.0},
    ]
    offscreen_window.current_analysis_result = {
        "chart_type": "bar",
        "protocol_rows": rows,
    }

    # Set active filter before refresh
    offscreen_window.session.set_filter("status", "corrected")
    assert offscreen_window.session.filters["status"] == "corrected"

    # Refresh combos (in Darwin mode, safe_combo_populate defers execution)
    offscreen_window._refresh_filter_combos()

    # Verify session filters were reset synchronously BEFORE singleShot fired
    assert offscreen_window.session.filters == {"status": "All", "outcome": "All", "group": "All"}

    # _apply_protocol_filters called immediately should return all rows
    offscreen_window._apply_protocol_filters()
    assert offscreen_window.protocol_table.rowCount() == 2

    # Process pending events for deferred singleShot
    QApplication.processEvents()

    # Combos and table are consistent
    assert offscreen_window.proto_status_combo.currentIndex() == 0
    assert offscreen_window.protocol_table.rowCount() == 2
