# -*- coding: utf-8 -*-
"""
Unit and Contract Tests for ReviewSession (ADR-007 T-007.2, ADR-008 T-008.1).
"""

from __future__ import annotations

import ast
from pathlib import Path
import pytest

from visual.review_session import ProcessingState, ReviewSession, VALID_DIRTY_SOURCES


def test_review_session_no_qtwidgets_or_qtgui():
    """Enforce that src/visual/review_session.py has ZERO imports from QtWidgets or QtGui."""
    session_file = Path(__file__).resolve().parents[2] / "src" / "visual" / "review_session.py"
    assert session_file.exists(), f"File not found: {session_file}"

    tree = ast.parse(session_file.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "QtWidgets" in alias.name or "QtGui" in alias.name:
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if "QtWidgets" in mod or "QtGui" in mod:
                violations.append(mod)
            for alias in node.names:
                if "QtWidgets" in alias.name or "QtGui" in alias.name:
                    violations.append(f"{mod}.{alias.name}")

    assert violations == [], f"Prohibited imports detected in review_session.py: {violations}"


def test_session_initial_state():
    session = ReviewSession()
    assert session.result is None
    assert session.current_analysis_result is None
    assert session.is_processing is False
    assert session.processing == ProcessingState.IDLE
    assert session.is_dirty is False
    assert session.dirty_sources == set()
    assert session.filters == {"status": "All", "outcome": "All", "group": "All"}
    assert session.filtered_rows() == []
    assert session.counts() == (0, 0)


def test_processing_state_property():
    session = ReviewSession()
    emitted_states = []
    session.processing_changed.connect(lambda s: emitted_states.append(s))

    # Test boolean setter
    session.is_processing = True
    assert session.is_processing is True
    assert session.processing == ProcessingState.ANALYSING

    session.is_processing = False
    assert session.is_processing is False
    assert session.processing == ProcessingState.IDLE

    # Test enum setter
    session.processing = ProcessingState.BATCH
    assert session.is_processing is True
    assert session.processing == ProcessingState.BATCH

    # Setting same state should not emit signal
    count_before = len(emitted_states)
    session.processing = ProcessingState.BATCH
    assert len(emitted_states) == count_before


def test_set_result_and_on_demand_protocol_rows():
    session = ReviewSession()
    result_emitted = []
    rows_emitted = []
    session.result_changed.connect(lambda r: result_emitted.append(r))
    session.protocol_rows_changed.connect(lambda rows: rows_emitted.append(rows))

    # Result without protocol_rows triggers on-demand build
    raw_result = {
        "chart_type": "bar",
        "elements": [
            {
                "type": "bar",
                "label": "Control",
                "value": 10.5,
                "series": "Series A",
                "confidence": 0.95,
                "bbox": [10, 20, 30, 40],
            }
        ],
    }
    session.set_result(raw_result)

    assert session.result is not None
    assert "protocol_rows" in session.result
    assert len(session.result["protocol_rows"]) >= 1
    assert len(result_emitted) == 1
    assert len(rows_emitted) == 1
    assert session.is_dirty is False


def test_filter_and_counts():
    session = ReviewSession()
    rows = [
        {"outcome": "Outcome A", "group": "Group 1", "review_status": "auto", "value": 1.0},
        {"outcome": "Outcome A", "group": "Group 2", "review_status": "corrected", "value": 2.0},
        {"outcome": "Outcome B", "group": "Group 1", "review_status": "reviewed", "value": 3.0},
    ]
    session.set_result({"chart_type": "bar", "protocol_rows": rows})
    assert session.counts() == (3, 3)

    filter_events = []
    session.filters_changed.connect(lambda: filter_events.append(True))

    session.set_filter("outcome", "Outcome A")
    assert len(filter_events) == 1
    assert len(session.filtered_rows()) == 2
    assert session.counts() == (2, 3)

    session.set_filter("status", "corrected")
    assert len(filter_events) == 2
    filtered = session.filtered_rows()
    assert len(filtered) == 1
    assert filtered[0]["group"] == "Group 2"
    assert session.counts() == (1, 3)

    # Setting same filter does not re-emit
    session.set_filter("status", "corrected")
    assert len(filter_events) == 2

    # Reset filters
    session.reset_filters()
    assert len(filter_events) == 3
    assert session.counts() == (3, 3)
    assert session.filters == {"status": "All", "outcome": "All", "group": "All"}


def test_set_protocol_rows_and_update_row_field():
    session = ReviewSession()
    rows = [
        {"_uid": "u1", "outcome": "A", "group": "G1", "value": 10.0},
        {"_uid": "u2", "outcome": "B", "group": "G2", "value": 20.0},
    ]
    session.set_result({"protocol_rows": rows})

    rows_emitted = []
    session.protocol_rows_changed.connect(lambda r: rows_emitted.append(r))

    # Update by index
    session.update_row_field(0, "value", 15.5)
    assert session.result["protocol_rows"][0]["value"] == 15.5
    assert len(rows_emitted) == 1

    # Update by _uid
    session.update_row_field("u2", "value", 25.5)
    assert session.result["protocol_rows"][1]["value"] == 25.5
    assert len(rows_emitted) == 2


def test_dirty_sources_reversibility_and_transition_only_signals():
    session = ReviewSession()
    dirty_events = []
    session.dirty_changed.connect(lambda d: dirty_events.append(d))

    # 1. Unknown source raises ValueError
    with pytest.raises(ValueError, match="Unknown dirty source"):
        session.set_source_dirty("invalid_source", True)

    # 2. Transition False -> True emits dirty_changed(True) once
    session.set_source_dirty("boxes", True)
    assert session.is_dirty is True
    assert session.dirty_sources == {"boxes"}
    assert dirty_events == [True]

    # 3. Adding second dirty source does NOT re-emit dirty_changed (already dirty)
    session.set_source_dirty("ocr_text", True)
    assert session.is_dirty is True
    assert session.dirty_sources == {"boxes", "ocr_text"}
    assert dirty_events == [True]

    # 4. Removing one dirty source does NOT re-emit (still dirty via "boxes")
    session.set_source_dirty("ocr_text", False)
    assert session.is_dirty is True
    assert session.dirty_sources == {"boxes"}
    assert dirty_events == [True]

    # 5. Removing last dirty source emits dirty_changed(False)
    session.set_source_dirty("boxes", False)
    assert session.is_dirty is False
    assert session.dirty_sources == set()
    assert dirty_events == [True, False]

    # 6. mark_clean() on clean session does NOT emit
    session.mark_clean()
    assert dirty_events == [True, False]

    # 7. mark_clean() on dirty session clears all sources and emits dirty_changed(False)
    session.set_source_dirty("data_table", True)
    session.set_source_dirty("results_table", True)
    assert dirty_events == [True, False, True]
    session.mark_clean()
    assert session.is_dirty is False
    assert session.dirty_sources == set()
    assert dirty_events == [True, False, True, False]


def test_request_navigation_blocked_during_processing():
    session = ReviewSession()
    assert session.request_navigation(1) is True

    session.processing = ProcessingState.ANALYSING
    assert session.request_navigation(1) is False

    session.processing = ProcessingState.BATCH
    assert session.request_navigation(1) is False

    session.processing = ProcessingState.IDLE
    assert session.request_navigation(1) is True


def test_navigation_guard_callback():
    session = ReviewSession()
    guard_called = []

    def guard(target_idx: int) -> bool:
        guard_called.append(target_idx)
        return target_idx != 5

    session.register_navigation_guard(guard)

    assert session.request_navigation(2) is True
    assert guard_called == [2]

    assert session.request_navigation(5) is False
    assert guard_called == [2, 5]


def test_counts_lazy_evaluation_synchronization():
    """Defect 3 regression test: counts() reports total >= shown on unbuilt protocol_rows."""
    session = ReviewSession()
    session._result = {
        "chart_type": "bar",
        "elements": [{"type": "bar", "label": "A", "value": 10.0, "confidence": 0.9, "bbox": [0, 0, 10, 10]}],
    }
    shown, total = session.counts()
    assert total >= shown
    assert shown == 1
    assert total == 1


def test_filtered_rows_and_counts_handle_non_dict_elements():
    """Defect 6 regression test: non-dict and None elements in protocol_rows do not crash filters."""
    session = ReviewSession()
    session._result = {
        "protocol_rows": [
            {"outcome": "A", "group": "G1", "review_status": "auto"},
            None,
            "corrupted_string_row",
            123,
        ]
    }
    session.set_filter("status", "auto")
    rows = session.filtered_rows()
    assert len(rows) == 1
    assert rows[0]["outcome"] == "A"

    shown, total = session.counts()
    assert shown == 1
    assert total == 1


def test_navigation_guard_supports_zero_and_one_argument_signatures():
    """Defect 7 regression test: request_navigation supports 0-arg and 1-arg guard signatures."""
    session = ReviewSession()

    calls_1 = []
    session.register_navigation_guard(lambda idx: calls_1.append(idx) or (idx != 99))
    assert session.request_navigation(1) is True
    assert session.request_navigation(99) is False
    assert calls_1 == [1, 99]

    calls_0 = []
    session.register_navigation_guard(lambda: calls_0.append(True) or False)
    assert session.request_navigation(5) is False
    assert len(calls_0) == 1
    assert session.request_navigation() is False
    assert len(calls_0) == 2

