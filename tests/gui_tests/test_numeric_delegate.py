"""Tests for NumericCellDelegate and MultiSelectTableWidget numeric paste.

Specifications:
- ADR-009: Numeric Cell Input Validation and Display Precision
- Decision Point DP-03: Display Precision for Large Numbers (<1e7 integer digit protection)
- Plan Tasks:
  - T-009.2: NumericCellDelegate
  - T-009.4: MultiSelectTableWidget paste validation & paste_completed signal
  - T-009.3a: _on_protocol_cell_changed parser guard & cell text reversion
  - T-009.5: _safe_float European comma handling
  - T-009.6: Truncating format replacement in OCR tab and overlays
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLineEdit,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
)

from core.numeric_format import format_display, parse_user_float
from visual.data_tab_schema import _safe_float
from visual.numeric_delegate import NumericCellDelegate
from visual.table_widget import MultiSelectTableWidget


# ---------------------------------------------------------------------------
# NumericCellDelegate unit tests (T-009.2)
# ---------------------------------------------------------------------------

class TestNumericCellDelegate:
    """Test suite for NumericCellDelegate display, editing, and validation."""

    def test_init_defaults(self, qapp):
        delegate = NumericCellDelegate()
        assert delegate.sig == 5
        assert delegate.allow_empty is True

    def test_display_text_raw_float_dp03(self, qapp):
        delegate = NumericCellDelegate(sig=5)
        # Small float legibility (T-009.6 acceptance)
        assert delegate.displayText(0.004, None) == "0.004"
        assert delegate.displayText(0.00012345, None) == "0.00012345"
        # Integer digit protection below 1e7 per DP-03
        assert delegate.displayText(12345.678, None) == "12346"
        assert delegate.displayText(123456.7, None) == "123457"
        assert delegate.displayText(500000.0, None) == "500000"
        # Zero
        assert delegate.displayText(0.0, None) == "0"
        # Scientific notation outside [1e-4, 1e7)
        assert delegate.displayText(1e-7, None) == "1e-07"
        # None and empty
        assert delegate.displayText(None, None) == ""
        assert delegate.displayText("", None) == ""
        # NaN and Inf
        assert delegate.displayText(float("nan"), None) == ""
        assert delegate.displayText(float("inf"), None) == ""

    def test_display_text_string_normalization(self, qapp):
        delegate = NumericCellDelegate(sig=5)
        assert delegate.displayText("12,5", None) == "12.5"
        assert delegate.displayText(" 0.004 ", None) == "0.004"
        # Unparseable string returns raw string
        assert delegate.displayText("not_a_number", None) == "not_a_number"

    def test_create_editor(self, qapp):
        delegate = NumericCellDelegate()
        table = QTableWidget(1, 1)
        editor = delegate.createEditor(table, QStyleOptionViewItem(), table.model().index(0, 0))
        assert isinstance(editor, QLineEdit)

    def test_set_editor_data_lossless_repr(self, qapp):
        delegate = NumericCellDelegate()
        table = QTableWidget(1, 1)
        # Storing a float that differs between str() and repr() or exact float
        test_val = 0.30000000000000004
        item = QTableWidgetItem()
        item.setData(Qt.ItemDataRole.DisplayRole, test_val)
        table.setItem(0, 0, item)

        editor = QLineEdit()
        delegate.setEditorData(editor, table.model().index(0, 0))
        assert editor.text() == repr(test_val)

    def test_set_editor_data_empty_and_nan(self, qapp):
        delegate = NumericCellDelegate()
        table = QTableWidget(1, 2)
        item_none = QTableWidgetItem()
        item_none.setData(Qt.ItemDataRole.DisplayRole, None)
        table.setItem(0, 0, item_none)

        item_nan = QTableWidgetItem()
        item_nan.setData(Qt.ItemDataRole.DisplayRole, float("nan"))
        table.setItem(0, 1, item_nan)

        ed1 = QLineEdit()
        delegate.setEditorData(ed1, table.model().index(0, 0))
        assert ed1.text() == ""

        ed2 = QLineEdit()
        delegate.setEditorData(ed2, table.model().index(0, 1))
        assert ed2.text() == ""

    def test_set_model_data_valid_float_commit(self, qapp):
        delegate = NumericCellDelegate()
        table = QTableWidget(1, 1)
        table.setItem(0, 0, QTableWidgetItem("0.0"))

        editor = QLineEdit()
        editor.setText("12.5")
        delegate.setModelData(editor, table.model(), table.model().index(0, 0))

        # Value in DisplayRole is float
        val = table.model().index(0, 0).data(Qt.ItemDataRole.DisplayRole)
        assert isinstance(val, float)
        assert val == 12.5

    def test_set_model_data_european_comma_accepted(self, qapp):
        delegate = NumericCellDelegate()
        table = QTableWidget(1, 1)
        table.setItem(0, 0, QTableWidgetItem("0.0"))

        editor = QLineEdit()
        editor.setText("12,5")
        delegate.setModelData(editor, table.model(), table.model().index(0, 0))

        val = table.model().index(0, 0).data(Qt.ItemDataRole.DisplayRole)
        assert val == 12.5

    def test_set_model_data_rejection_emits_signal_no_commit(self, qapp):
        """ADR-009 Acceptance: Given a cell 3.2, when user commits 1.2e,
        stored text is still 3.2, value_rejected fired once, cellChanged did NOT fire."""
        delegate = NumericCellDelegate()
        table = QTableWidget(1, 1)
        table.setItem(0, 0, QTableWidgetItem("3.2"))

        rejected_events: List[Tuple[int, int, str, str]] = []
        cell_changed_events: List[Tuple[int, int]] = []

        delegate.value_rejected.connect(lambda r, c, t, re: rejected_events.append((r, c, t, re)))
        table.cellChanged.connect(lambda r, c: cell_changed_events.append((r, c)))

        editor = QLineEdit()
        editor.setText("1.2e")
        delegate.setModelData(editor, table.model(), table.model().index(0, 0))

        # Stored value is still 3.2
        assert table.item(0, 0).text() == "3.2"
        # value_rejected fired exactly once
        assert len(rejected_events) == 1
        row, col, text, reason = rejected_events[0]
        assert row == 0
        assert col == 0
        assert text == "1.2e"
        assert "Invalid numeric format" in reason
        # cellChanged did NOT fire
        assert len(cell_changed_events) == 0

    def test_set_model_data_ambiguous_comma_rejected(self, qapp):
        """1,234 is ambiguous (1.234 vs 1234) and must be rejected without committing."""
        delegate = NumericCellDelegate()
        table = QTableWidget(1, 1)
        table.setItem(0, 0, QTableWidgetItem("100.0"))

        rejected_events: List[Tuple[int, int, str, str]] = []
        delegate.value_rejected.connect(lambda r, c, t, re: rejected_events.append((r, c, t, re)))

        editor = QLineEdit()
        editor.setText("1,234")
        delegate.setModelData(editor, table.model(), table.model().index(0, 0))

        assert table.item(0, 0).text() == "100.0"
        assert len(rejected_events) == 1
        assert "Ambiguous" in rejected_events[0][3]

    def test_set_model_data_allow_empty(self, qapp):
        # When allow_empty=True: empty string commits None
        delegate_empty = NumericCellDelegate(allow_empty=True)
        table = QTableWidget(1, 1)
        table.setItem(0, 0, QTableWidgetItem("5.0"))
        editor = QLineEdit()
        editor.setText("")
        delegate_empty.setModelData(editor, table.model(), table.model().index(0, 0))
        assert table.model().index(0, 0).data(Qt.ItemDataRole.DisplayRole) is None

        # When allow_empty=False: empty string rejected
        delegate_no_empty = NumericCellDelegate(allow_empty=False)
        table.setItem(0, 0, QTableWidgetItem("5.0"))
        rejected_events = []
        delegate_no_empty.value_rejected.connect(lambda r, c, t, re: rejected_events.append((r, c, t, re)))
        delegate_no_empty.setModelData(editor, table.model(), table.model().index(0, 0))
        assert len(rejected_events) == 1
        assert table.item(0, 0).text() == "5.0"

    def test_paint_renders_with_format_display(self, qapp):
        delegate = NumericCellDelegate(sig=5)
        table = QTableWidget(1, 1)
        item = QTableWidgetItem()
        item.setData(Qt.ItemDataRole.DisplayRole, 0.004)
        table.setItem(0, 0, item)

        pixmap = QPixmap(100, 30)
        painter = QPainter(pixmap)
        opt = QStyleOptionViewItem()
        delegate.paint(painter, opt, table.model().index(0, 0))
        painter.end()


# ---------------------------------------------------------------------------
# MultiSelectTableWidget paste & undo tests (T-009.4)
# ---------------------------------------------------------------------------

class TestMultiSelectTableWidgetPaste:
    """Test paste validation against NumericCellDelegate and undo stack integrity."""

    def test_paste_over_numeric_delegate_columns(self, qapp):
        """ADR-009 Acceptance: Given clipboard 1,5\\t2.5\\tabc pasted over three
        numeric cells, cells 1-2 update (1.5, 2.5), cell 3 is unchanged,
        paste_completed reports 1 rejected, and one Ctrl+Z restores cells 1-2."""
        table = MultiSelectTableWidget(3, 1)
        delegate = NumericCellDelegate()
        table.setItemDelegateForColumn(0, delegate)

        table.setItem(0, 0, QTableWidgetItem("10.0"))
        table.setItem(1, 0, QTableWidgetItem("20.0"))
        table.setItem(2, 0, QTableWidgetItem("30.0"))

        table.setCurrentItem(table.item(0, 0))

        paste_results = []
        table.paste_completed.connect(lambda s, r, d: paste_results.append((s, r, d)))

        # Clipboard content across 3 rows: 1,5 \n 2.5 \n abc
        QApplication.clipboard().setText("1,5\n2.5\nabc")
        table._paste_selection()

        # Cells 1-2 updated, cell 3 unchanged
        assert table.item(0, 0).text() == "1.5"
        assert table.item(1, 0).text() == "2.5"
        assert table.item(2, 0).text() == "30.0"

        # paste_completed emitted once with (2, 1, [...])
        assert len(paste_results) == 1
        success, rejected, details = paste_results[0]
        assert success == 2
        assert rejected == 1
        assert len(details) == 1
        r, c, text, err = details[0]
        assert r == 2
        assert c == 0
        assert text == "abc"

        # Undo in single step restores cells 0 and 1
        assert table._undo_stack.canUndo()
        table._undo_stack.undo()
        assert table.item(0, 0).text() == "10.0"
        assert table.item(1, 0).text() == "20.0"
        assert table.item(2, 0).text() == "30.0"

    def test_paste_horizontal_tsv(self, qapp):
        table = MultiSelectTableWidget(1, 3)
        delegate = NumericCellDelegate()
        for col in range(3):
            table.setItemDelegateForColumn(col, delegate)
            table.setItem(0, col, QTableWidgetItem("0.0"))

        table.setCurrentItem(table.item(0, 0))

        paste_results = []
        table.paste_completed.connect(lambda s, r, d: paste_results.append((s, r, d)))

        QApplication.clipboard().setText("1,5\t2.5\tabc")
        table._paste_selection()

        assert table.item(0, 0).text() == "1.5"
        assert table.item(0, 1).text() == "2.5"
        assert table.item(0, 2).text() == "0.0"  # rejected cell unchanged
        assert len(paste_results) == 1
        assert paste_results[0][0] == 2
        assert paste_results[0][1] == 1

    def test_paste_all_valid(self, qapp):
        table = MultiSelectTableWidget(2, 1)
        delegate = NumericCellDelegate()
        table.setItemDelegateForColumn(0, delegate)
        table.setItem(0, 0, QTableWidgetItem("0.0"))
        table.setItem(1, 0, QTableWidgetItem("0.0"))
        table.setCurrentItem(table.item(0, 0))

        paste_results = []
        table.paste_completed.connect(lambda s, r, d: paste_results.append((s, r, d)))

        QApplication.clipboard().setText("1.23\n4.56")
        table._paste_selection()

        assert table.item(0, 0).text() == "1.23"
        assert table.item(1, 0).text() == "4.56"
        assert paste_results[0] == (2, 0, [])


# ---------------------------------------------------------------------------
# Interim numeric hotfixes tests (T-009.3a, T-009.5, T-009.6)
# ---------------------------------------------------------------------------

class TestNumericHotfixes:
    """Test interim fixes in data_tab_schema, visualization_service, and main_modern."""

    def test_safe_float_european_comma(self):
        """T-009.5: European decimal comma must parse as 12.5, NOT 125.0."""
        assert _safe_float("12,5") == 12.5
        assert _safe_float("12.5") == 12.5
        assert _safe_float("0,125") == 0.125
        assert _safe_float("0.004") == 0.004
        assert _safe_float("1,234") is None  # Ambiguous single comma rejected
        assert _safe_float("invalid") is None
        assert _safe_float(None) is None

    def test_format_display_replaces_truncating_format(self):
        """T-009.6: format_display preserves small values like 0.004."""
        assert format_display(0.004) == "0.004"
        assert format_display(0.00012345) == "0.00012345"
        assert format_display(12.34567) == "12.346"
        assert format_display(123456.7) == "123457"

    def test_on_protocol_cell_changed_reverts_on_invalid_and_accepts_comma(self, offscreen_window, sample_result_payload):
        """T-009.3a: _on_protocol_cell_changed accepts European comma and reverts on invalid input."""
        win = offscreen_window
        payload = sample_result_payload('bar')
        if hasattr(win, 'session') and win.session is not None:
            win.session.set_result(payload)
        else:
            win.current_analysis_result = payload
        win._populate_protocol_tab()

        table = win.protocol_table
        from core.protocol_row_builder import get_protocol_columns
        cols = get_protocol_columns('bar')
        val_col_idx = next(i for i, (cid, _, _) in enumerate(cols) if cid == 'value')

        # 1. Edit with valid European comma "15,5"
        table.item(0, val_col_idx).setText("15,5")
        rows = (win.current_analysis_result or {}).get('protocol_rows', [])
        assert rows[0]['value'] == 15.5
        assert rows[0]['review_status'] == 'corrected'

        # 2. Edit with invalid format "1.2e"
        table.item(0, val_col_idx).setText("1.2e")
        # Cell text must be reverted back to previous value
        assert table.item(0, val_col_idx).text() == "15.5"
        assert rows[0]['value'] == 15.5
