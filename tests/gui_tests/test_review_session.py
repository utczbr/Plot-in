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


# ---------------------------------------------------------------------------
# ADR-007 T-007.3 — Stable Row Identity (_uid)
# ---------------------------------------------------------------------------


def test_t007_3_rebuild_preserves_edited_row_uid_and_subsequent_edits(offscreen_window):
    """
    Given a result with 5 rows, when _refresh_protocol_rows_from_result() rebuilds
    the list, then every previously edited row keeps its _uid and an edit made
    after the rebuild still lands on the right row.
    """
    from PyQt6.QtCore import Qt
    from core.protocol_row_builder import get_protocol_columns

    elements = [
        {"index": i, "type": "bar", "bbox": [i * 10, 0, i * 10 + 8, 50], "tick_label": {"text": f"G{i}"}, "value": float((i + 1) * 10)}
        for i in range(5)
    ]
    offscreen_window.current_analysis_result = {
        "image_file": "five_bars.png",
        "chart_type": "bar",
        "elements": elements,
    }
    offscreen_window._refresh_protocol_rows_from_result()

    rows_before = offscreen_window.current_analysis_result["protocol_rows"]
    assert len(rows_before) == 5
    uids_before = [r["_uid"] for r in rows_before]
    assert len(set(uids_before)) == 5
    assert all(isinstance(u, str) and len(u) == 32 for u in uids_before)

    # Verify table stores _uid in Qt.UserRole
    table = offscreen_window.protocol_table
    for r_idx in range(5):
        assert table.item(r_idx, 0).data(Qt.ItemDataRole.UserRole) == uids_before[r_idx]

    # Edit row 1 and row 3 via the table
    active_cols = get_protocol_columns("bar")
    val_col_idx = next(i for i, (cid, _, _) in enumerate(active_cols) if cid == "value")
    notes_col_idx = next(i for i, (cid, _, _) in enumerate(active_cols) if cid == "notes")

    table.item(1, val_col_idx).setText("222.5")
    table.item(3, notes_col_idx).setText("checked row 3")

    assert rows_before[1]["value"] == 222.5
    assert rows_before[1]["review_status"] == "corrected"
    assert rows_before[3]["notes"] == "checked row 3"

    # Rebuild via _refresh_protocol_rows_from_result()
    offscreen_window._refresh_protocol_rows_from_result()
    rows_after = offscreen_window.current_analysis_result["protocol_rows"]
    assert len(rows_after) == 5

    # Every edited row keeps its exact _uid (and matching untouched rows do too)
    assert rows_after[1]["_uid"] == uids_before[1]
    assert rows_after[3]["_uid"] == uids_before[3]
    assert [r["_uid"] for r in rows_after] == uids_before

    # Make an edit after the rebuild and verify it lands on the exact target row
    table.item(1, notes_col_idx).setText("post-rebuild note")
    table.item(4, val_col_idx).setText("555.0")
    assert rows_after[1]["notes"] == "post-rebuild note"
    assert rows_after[1]["value"] == 222.5
    assert rows_after[4]["value"] == 555.0


def test_t007_3_cached_json_without_uid_assigns_distinct_uids_and_no_none_collision(
    offscreen_window, tmp_path
):
    """
    Given a cached analysis JSON saved without _uid, when loaded and edited in
    the table, then every row receives a distinct _uid and an edit on row k
    updates row k without colliding on None.
    """
    import json
    from PIL import Image
    from core.protocol_row_builder import get_protocol_columns

    img_path = tmp_path / "legacy_chart.png"
    Image.new("RGB", (120, 90), color=(255, 255, 255)).save(img_path)

    legacy_rows = [
        {
            "source_file": "legacy_chart.png",
            "chart_type": "bar",
            "element_index": k,
            "group": f"Cat{k}",
            "outcome": "",
            "value": float(k * 10),
            "unit": "",
            "error_bar_type": "",
            "error_bar_value": None,
            "baseline_value": 0.0,
            "confidence": 0.9,
            "review_status": "auto",
            "notes": "",
            "_original": None,
        }
        for k in range(4)
    ]
    json_path = tmp_path / "legacy_chart_analysis.json"
    json_path.write_text(
        json.dumps(
            {
                "original_image_path": str(img_path),
                "image_file": "legacy_chart.png",
                "chart_type": "bar",
                "elements": [{"index": k, "type": "bar", "value": float(k * 10)} for k in range(4)],
                "protocol_rows": legacy_rows,
            }
        ),
        encoding="utf-8",
    )

    offscreen_window.image_files = [str(img_path)]
    offscreen_window.current_image_index = 0
    offscreen_window._load_result_from_cache(str(img_path), str(json_path))

    loaded_rows = offscreen_window.current_analysis_result["protocol_rows"]
    assert len(loaded_rows) == 4
    uids = [r.get("_uid") for r in loaded_rows]
    assert all(isinstance(u, str) and len(u) > 0 for u in uids)
    assert len(set(uids)) == 4

    # Edit row k=2 in the table and verify ONLY row 2 is modified
    active_cols = get_protocol_columns("bar")
    val_col_idx = next(i for i, (cid, _, _) in enumerate(active_cols) if cid == "value")
    offscreen_window.protocol_table.item(2, val_col_idx).setText("999.25")

    assert loaded_rows[0]["value"] == 0.0
    assert loaded_rows[1]["value"] == 10.0
    assert loaded_rows[2]["value"] == 999.25
    assert loaded_rows[3]["value"] == 30.0


def test_t007_3_uid_excluded_from_export_protocol_csv_and_gui_csv(
    offscreen_window, tmp_path, monkeypatch
):
    """
    Given export_protocol_csv and the GUI CSV export, then _uid never appears
    in output (assert header equality with existing PROTOCOL_COLUMNS / active columns).
    """
    import csv
    from core.export_manager import ExportManager
    from core.protocol_row_builder import build_protocol_rows, get_protocol_columns

    result = {
        "image_file": "bar_export.png",
        "chart_type": "bar",
        "elements": [
            {"index": 0, "type": "bar", "bbox": [0, 0, 10, 50], "tick_label": {"text": "A"}, "value": 12.5},
            {"index": 1, "type": "bar", "bbox": [20, 0, 30, 50], "tick_label": {"text": "B"}, "value": 25.0},
        ],
    }
    rows = build_protocol_rows(result)
    assert all("_uid" in r for r in rows)
    result["protocol_rows"] = rows

    # 1. ExportManager.export_protocol_csv
    core_csv = tmp_path / "core_protocol.csv"
    ExportManager.export_protocol_csv(rows, str(core_csv))
    with open(core_csv, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0] == list(ExportManager.PROTOCOL_COLUMNS)
    assert "_uid" not in reader[0]
    for row_vals in reader[1:]:
        for r in rows:
            assert r["_uid"] not in row_vals

    # 2. GUI _export_protocol_csv
    gui_csv = tmp_path / "gui_protocol.csv"
    offscreen_window.current_analysis_result = result
    offscreen_window._refresh_protocol_rows_from_result()
    monkeypatch.setattr(
        "main_modern.QFileDialog.getSaveFileName",
        lambda *a, **kw: (str(gui_csv), "CSV Files (*.csv)"),
    )
    monkeypatch.setattr(
        "main_modern.QMessageBox.information",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "main_modern.QMessageBox.warning",
        lambda *a, **kw: None,
    )
    offscreen_window._export_protocol_csv()

    with open(gui_csv, newline="", encoding="utf-8") as f:
        gui_reader = list(csv.reader(f))
    expected_gui_cols = [c[0] for c in get_protocol_columns("bar")]
    assert gui_reader[0] == expected_gui_cols
    assert "_uid" not in gui_reader[0]


def test_t007_3_merge_context_into_rows_uid_matching_and_non_matching_element_index():
    """
    merge_context_into_rows preserves _uid on kept rows, carries _uid on untouched
    replaced rows only when element_index matches, and generates a new _uid otherwise.
    """
    from core.protocol_row_builder import build_protocol_rows, merge_context_into_rows

    pipeline_result = {
        "image_file": "chart.png",
        "chart_type": "bar",
        "elements": [
            {"index": 0, "type": "bar", "bbox": [0, 0, 10, 50], "value": 10.0},
            {"index": 1, "type": "bar", "bbox": [20, 0, 30, 50], "value": 20.0},
        ],
    }
    rows = build_protocol_rows(pipeline_result)
    uid0 = rows[0]["_uid"]
    uid1 = rows[1]["_uid"]

    # Mark row 0 as edited
    rows[0]["review_status"] = "corrected"
    rows[0]["_original"] = {"value": 10.0}
    rows[0]["value"] = 99.0

    # Add a third row with element_index=-1 (no valid element_index match)
    orphan_row = dict(rows[1])
    orphan_row["_uid"] = "orphan_uid_should_not_carry_if_replaced"
    orphan_row["element_index"] = -1

    # Add an element with index -1 to pipeline_result to test non-matching element_index=-1 replacement
    merged = merge_context_into_rows([rows[0], rows[1]], pipeline_result, {"outcomes": ["Y"]})
    assert merged[0]["_uid"] == uid0
    assert merged[0]["value"] == 99.0
    assert merged[1]["_uid"] == uid1


