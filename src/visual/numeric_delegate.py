"""Numeric cell delegate providing precision-preserving display and validated input.

Specifications:
- ADR-009: Numeric Cell Input Validation and Display Precision
- Decision Point DP-03: Display Precision for Large Numbers (<1e7 integer digit protection)
- Plan Tasks: T-009.2 (NumericCellDelegate)
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QLineEdit,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from core.numeric_format import format_display, parse_user_float


class NumericCellDelegate(QStyledItemDelegate):
    """Delegate for numeric table cells.

    - Uses format_display() for significant-digit presentation according to DP-03.
    - Pre-fills editor with lossless repr(float).
    - Validates edits with parse_user_float() before committing.
    - Emits value_rejected(row, col, text, reason) without modifying the model on failure.
    """

    value_rejected = pyqtSignal(int, int, str, str)  # (row, col, text, reason)

    def __init__(self, parent=None, sig: int = 5, allow_empty: bool = True):
        super().__init__(parent)
        self.sig = sig
        self.allow_empty = allow_empty

    def displayText(self, value, locale) -> str:
        """Format value for display using format_display with DP-03 rules."""
        if value is None or value == "":
            return ""
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return ""
            return format_display(value, sig=self.sig)
        if isinstance(value, str):
            res = parse_user_float(value, allow_empty=self.allow_empty)
            if res.error is None and res.value is not None:
                return format_display(res.value, sig=self.sig)
            return value
        return str(value)

    def createEditor(self, parent, option, index):
        """Create a single-line text editor for numeric input."""
        editor = QLineEdit(parent)
        return editor

    def setEditorData(self, editor: QLineEdit, index):
        """Pre-fill editor with lossless repr(float) or raw text."""
        val = index.data(Qt.ItemDataRole.EditRole)
        if val is None or val == "":
            val = index.data(Qt.ItemDataRole.DisplayRole)

        if val is None or val == "":
            editor.setText("")
            return

        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                editor.setText("")
            else:
                editor.setText(repr(float(val)))
            return

        if isinstance(val, str):
            res = parse_user_float(val, allow_empty=True)
            if res.error is None and res.value is not None:
                editor.setText(repr(float(res.value)))
            else:
                editor.setText(val)
            return

        editor.setText(str(val))

    def setModelData(self, editor: QLineEdit, model, index):
        """Validate editor text and commit raw float or reject without modifying model."""
        text = editor.text() if hasattr(editor, "text") else str(editor)
        res = parse_user_float(text, allow_empty=self.allow_empty)

        if res.error is not None:
            # Rejection: do NOT commit to model, emit value_rejected
            self.value_rejected.emit(index.row(), index.column(), text, res.error)
            return

        # Commit parsed float or None to model
        if res.value is not None:
            model.setData(index, float(res.value), Qt.ItemDataRole.DisplayRole)
            model.setData(index, float(res.value), Qt.ItemDataRole.EditRole)
        else:
            model.setData(index, None, Qt.ItemDataRole.DisplayRole)
            model.setData(index, None, Qt.ItemDataRole.EditRole)

    def paint(self, painter, option, index):
        """Paint cell text using format_display."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        raw_val = index.data(Qt.ItemDataRole.DisplayRole)
        opt.text = self.displayText(raw_val, opt.locale)
        widget = option.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
