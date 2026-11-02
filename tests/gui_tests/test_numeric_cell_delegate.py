"""GUI tests for NumericCellDelegate and T-009.3b Results table wiring."""

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QStyleOptionViewItem, QTableWidget, QTableWidgetItem, QWidget

from main_modern import ModernChartAnalysisApp
from visual.numeric_delegate import NumericCellDelegate
from visual.review_session import ReviewSession
from visual.table_widget import MultiSelectTableWidget


class _DummyStatusBar:
    def __init__(self):
        self.last_message = ""
        self.last_timeout = 0

    def showMessage(self, msg: str, timeout: int = 0):
        self.last_message = msg
        self.last_timeout = timeout


class _DummyWin:
    current_analysis_result = ModernChartAnalysisApp.current_analysis_result


def _make_protocol_dummy_window(qapp):
    """Build a lightweight dummy window with real protocol table and ReviewSession."""
    win = _DummyWin()
    win.session = ReviewSession()
    win.advanced_settings = {"display_settings": {"significant_digits": 5}}
    win.protocol_row_overlay_map = {}
    win.protocol_table = MultiSelectTableWidget()
    win.protocol_table.setWordWrap(False)
    win._status_bar = _DummyStatusBar()
    win.statusBar = lambda: win._status_bar
    win.update_status = lambda msg: win._status_bar.showMessage(msg)
    win.current_image_path = "sample_chart.png"
    win.output_path_edit = SimpleNamespace(text=lambda: "")

    win._on_protocol_value_rejected = ModernChartAnalysisApp._on_protocol_value_rejected.__get__(win)
    win._on_protocol_paste_completed = ModernChartAnalysisApp._on_protocol_paste_completed.__get__(win)
    win._render_protocol_table = ModernChartAnalysisApp._render_protocol_table.__get__(win)
    win._on_protocol_cell_changed = ModernChartAnalysisApp._on_protocol_cell_changed.__get__(win)
    win._apply_protocol_filters = ModernChartAnalysisApp._apply_protocol_filters.__get__(win)

    win.protocol_table.cellChanged.connect(win._on_protocol_cell_changed)
    win.protocol_table.paste_completed.connect(win._on_protocol_paste_completed)
    return win


# ---------------------------------------------------------------------------
# T-009.2 — NumericCellDelegate unit tests
# ---------------------------------------------------------------------------

def test_delegate_display_text_fast_paths_float_and_formats_sig_digits(qapp):
    delegate = NumericCellDelegate(sig=5)
    assert delegate.displayText(0.004, None) == "0.004"
    assert delegate.displayText(12345.678, None) == "12346"
    assert delegate.displayText(123456.7, None) == "123457"
    assert delegate.displayText(1e-7, None) == "1e-07"
    assert delegate.displayText("", None) == ""
    assert delegate.displayText(None, None) == ""


def test_delegate_editor_prefills_repr_and_commits_valid_float(qapp):
    table = QTableWidget(1, 1)
    delegate = NumericCellDelegate(table, sig=5, allow_empty=True)
    table.setItemDelegateForColumn(0, delegate)

    item = QTableWidgetItem()
    item.setData(Qt.ItemDataRole.DisplayRole, 12.3456789)
    table.setItem(0, 0, item)

    index = table.model().index(0, 0)
    editor = delegate.createEditor(table.viewport(), QStyleOptionViewItem(), index)
    delegate.setEditorData(editor, index)
    assert editor.text() == repr(12.3456789)

    editor.setText("12,5")
    delegate.setModelData(editor, table.model(), index)

    stored = table.item(0, 0).data(Qt.ItemDataRole.DisplayRole)
    assert isinstance(stored, float)
    assert stored == 12.5


def test_delegate_rejects_invalid_input_without_firing_cell_changed(qapp):
    table = QTableWidget(1, 1)
    delegate = NumericCellDelegate(table, sig=5, allow_empty=True)
    table.setItemDelegateForColumn(0, delegate)

    item = QTableWidgetItem()
    item.setData(Qt.ItemDataRole.DisplayRole, 3.2)
    table.setItem(0, 0, item)

    cell_changed_calls = []
    rejected_calls = []
    table.cellChanged.connect(lambda r, c: cell_changed_calls.append((r, c)))
    delegate.value_rejected.connect(lambda r, c, txt, msg: rejected_calls.append((r, c, txt, msg)))

    index = table.model().index(0, 0)
    editor = delegate.createEditor(table.viewport(), QStyleOptionViewItem(), index)
    delegate.setEditorData(editor, index)
    editor.setText("1.2e")
    delegate.setModelData(editor, table.model(), index)

    assert table.item(0, 0).data(Qt.ItemDataRole.DisplayRole) == 3.2
    assert len(rejected_calls) == 1
    assert rejected_calls[0][0] == 0 and rejected_calls[0][1] == 0
    assert rejected_calls[0][2] == "1.2e" and rejected_calls[0][3] != ""
    assert cell_changed_calls == []


# ---------------------------------------------------------------------------
# T-009.3b — Results table wiring, DisplayRole float contract, & Undo sync
# ---------------------------------------------------------------------------

def test_render_protocol_table_stores_float_in_display_role_and_disables_word_wrap(qapp):
    win = _make_protocol_dummy_window(qapp)
    rows = [
        {
            "_uid": "row_1",
            "source_file": "img.png",
            "page_index": 0,
            "chart_type": "bar",
            "outcome": "Survival",
            "group": "Control",
            "value": 0.004123456,
            "unit": "mg/L",
            "error_bar_value": None,
            "baseline_value": 0.0,
            "confidence": 0.95,
            "review_status": "auto",
        }
    ]
    win.session.set_result({"chart_type": "bar", "protocol_rows": rows})
    win._render_protocol_table(rows)

    assert win.protocol_table.wordWrap() is False

    # Locate 'value' column (numeric) and 'error_bar_value' (None -> "")
    from core.protocol_row_builder import get_protocol_columns
    cols = get_protocol_columns("bar")
    val_col = next(i for i, c in enumerate(cols) if c[0] == "value")
    err_col = next(i for i, c in enumerate(cols) if c[0] == "error_bar_value")

    val_item = win.protocol_table.item(0, val_col)
    err_item = win.protocol_table.item(0, err_col)

    assert isinstance(val_item.data(Qt.ItemDataRole.DisplayRole), float)
    assert val_item.data(Qt.ItemDataRole.DisplayRole) == pytest.approx(0.004123456)
    assert err_item.data(Qt.ItemDataRole.DisplayRole) == ""


def test_comma_decimal_edit_updates_row_marks_corrected_and_exports_csv(qapp, tmp_path: Path):
    win = _make_protocol_dummy_window(qapp)
    rows = [
        {
            "_uid": "row_1",
            "source_file": "img.png",
            "page_index": 0,
            "chart_type": "bar",
            "outcome": "Yield",
            "group": "Treatment",
            "value": 10.0,
            "unit": "%",
            "error_bar_value": 0.5,
            "baseline_value": 0.0,
            "confidence": 0.9,
            "review_status": "auto",
            "_original": None,
        }
    ]
    win.current_analysis_result = {"chart_type": "bar", "protocol_rows": rows}
    win.session.set_result(win.current_analysis_result)
    win._render_protocol_table(rows)

    from core.protocol_row_builder import get_protocol_columns
    cols = get_protocol_columns("bar")
    val_col = next(i for i, c in enumerate(cols) if c[0] == "value")

    # Simulate user typing "12,5" into the value cell
    win.protocol_table.item(0, val_col).setText("12,5")

    assert rows[0]["value"] == 12.5
    assert rows[0]["review_status"] == "corrected"
    assert rows[0]["_original"]["value"] == 10.0
    assert "results_table" in win.session.dirty_sources
    assert isinstance(win.protocol_table.item(0, val_col).data(Qt.ItemDataRole.DisplayRole), float)

    # Verify exported CSV contains 12.5
    csv_path = tmp_path / "exported_protocol.csv"
    col_ids = [c[0] for c in cols]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=col_ids, extrasaction="ignore")
        writer.writeheader()
        for r in win.session.filtered_rows():
            writer.writerow({k: ("" if r.get(k) is None else r.get(k, "")) for k in col_ids})

    with open(csv_path, "r", encoding="utf-8") as f:
        exported_rows = list(csv.DictReader(f))
    assert len(exported_rows) == 1
    assert exported_rows[0]["value"] == "12.5"
    assert exported_rows[0]["review_status"] == "corrected"


def test_rejected_edit_cleans_undo_stack_and_does_not_mark_dirty_or_corrected(qapp):
    win = _make_protocol_dummy_window(qapp)
    rows = [
        {
            "_uid": "row_1",
            "source_file": "img.png",
            "page_index": 0,
            "chart_type": "bar",
            "outcome": "Yield",
            "group": "Control",
            "value": 5.0,
            "unit": "%",
            "error_bar_value": 0.2,
            "baseline_value": 0.0,
            "confidence": 0.9,
            "review_status": "auto",
            "_original": None,
        }
    ]
    win.current_analysis_result = {"chart_type": "bar", "protocol_rows": rows}
    win.session.set_result(win.current_analysis_result)
    win._render_protocol_table(rows)
    win.session.mark_clean()

    from core.protocol_row_builder import get_protocol_columns
    cols = get_protocol_columns("bar")
    val_col = next(i for i, c in enumerate(cols) if c[0] == "value")

    # 1. Perform a valid edit first: 5.0 -> 8.5
    win.protocol_table.item(0, val_col).setText("8.5")
    assert rows[0]["value"] == 8.5
    assert win.protocol_table._undo_stack.count() == 1

    # 2. Perform an invalid edit: "1,234" (ambiguous thousands/decimal comma)
    win.protocol_table.item(0, val_col).setText("1,234")

    # Value must revert to 8.5, error message shown, and undo stack must still have only 1 valid command
    assert rows[0]["value"] == 8.5
    assert float(win.protocol_table.item(0, val_col).text()) == 8.5
    assert "Invalid input" in win._status_bar.last_message
    assert win.protocol_table._undo_stack.count() == 1

    # 3. Pressing Ctrl+Z (undo) must undo 8.5 -> 5.0 cleanly without replaying "1,234"
    win.protocol_table._undo_stack.undo()
    assert rows[0]["value"] == 5.0
    assert float(win.protocol_table.item(0, val_col).text()) == 5.0


def test_switching_chart_types_clears_stale_numeric_delegates(qapp):
    win = _make_protocol_dummy_window(qapp)

    # 1. Render box plot (has 5 numeric quartile columns: median, q1, q3, whisker_low, whisker_high)
    box_rows = [
        {
            "_uid": "box_1",
            "source_file": "box.png",
            "page_index": 0,
            "chart_type": "box",
            "outcome": "Score",
            "group": "A",
            "median": 10.0,
            "q1": 5.0,
            "q3": 15.0,
            "whisker_low": 2.0,
            "whisker_high": 18.0,
            "unit": "pts",
            "confidence": 0.9,
            "review_status": "auto",
        }
    ]
    win.current_analysis_result = {"chart_type": "box", "protocol_rows": box_rows}
    win.session.set_result(win.current_analysis_result)
    win._render_protocol_table(box_rows)

    from core.protocol_row_builder import get_protocol_columns
    box_cols = get_protocol_columns("box")
    for idx, (_cid, _label, is_num) in enumerate(box_cols):
        delegate = win.protocol_table.itemDelegateForColumn(idx)
        if is_num:
            assert isinstance(delegate, NumericCellDelegate)
        else:
            assert delegate is None

    # 2. Switch to bar chart and verify string columns do not retain stale NumericCellDelegate
    bar_rows = [
        {
            "_uid": "bar_1",
            "source_file": "bar.png",
            "page_index": 0,
            "chart_type": "bar",
            "outcome": "Score",
            "group": "A",
            "value": 42.0,
            "unit": "pts",
            "error_bar_value": 1.5,
            "baseline_value": 0.0,
            "confidence": 0.9,
            "review_status": "auto",
        }
    ]
    win.current_analysis_result = {"chart_type": "bar", "protocol_rows": bar_rows}
    win.session.set_result(win.current_analysis_result)
    win._render_protocol_table(bar_rows)

    bar_cols = get_protocol_columns("bar")
    for idx, (cid, _label, is_num) in enumerate(bar_cols):
        delegate = win.protocol_table.itemDelegateForColumn(idx)
        if is_num:
            assert isinstance(delegate, NumericCellDelegate), f"Column {cid} should have NumericCellDelegate"
        else:
            assert delegate is None, f"String column {cid} (idx {idx}) retained stale NumericCellDelegate"