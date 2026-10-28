"""Table widget and undo/redo command extracted from main_modern.py."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QUndoCommand, QUndoStack
from PyQt6.QtWidgets import QApplication, QTableWidget


class TableEditCommand(QUndoCommand):
    """Command to handle undo/redo of MultiSelectTableWidget cell edits."""
    def __init__(self, table, changes, description):
        super().__init__(description)
        self.table = table
        self.changes = changes
        self._first_run = True

    def _apply(self, idx):
        self.table.blockSignals(True)
        try:
            for row, col, *vals in self.changes:
                item = self.table.item(row, col)
                if item:
                    item.setText(vals[idx])
                    self.table._current_data[(row, col)] = vals[idx]
        finally:
            self.table.blockSignals(False)
        self.table._is_editing_undo = True
        for row, col, *_ in self.changes:
            self.table.cellChanged.emit(row, col)
        self.table._is_editing_undo = False

    def undo(self):
        self._apply(0)

    def redo(self):
        if self._first_run:
            self._first_run = False
            return
        self._apply(1)


class MultiSelectTableWidget(QTableWidget):
    """QTableWidget subclass with Ctrl+C, Ctrl+V, Delete multi-cell support and Undo/Redo."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._undo_stack = QUndoStack(self)
        self._undo_stack.setUndoLimit(50)
        self._current_data = {}
        self._is_editing_undo = False
        self.cellChanged.connect(self._on_cell_changed)

    def setItem(self, row, col, item):
        super().setItem(row, col, item)
        self._current_data[(row, col)] = item.text()

    def setRowCount(self, rows):
        if rows == 0:
            self._undo_stack.clear()
            self._current_data.clear()
        super().setRowCount(rows)

    def _on_cell_changed(self, row, col):
        if self._is_editing_undo:
            return
        item = self.item(row, col)
        if not item:
            return
        new_text, old_text = item.text(), self._current_data.get((row, col), "")
        if new_text != old_text:
            self._is_editing_undo = True
            self._undo_stack.push(TableEditCommand(self, [(row, col, old_text, new_text)], "Edit Cell"))
            self._is_editing_undo = False
            self._current_data[(row, col)] = new_text

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            self._copy_selection()
        elif event.matches(QKeySequence.StandardKey.Paste):
            self._paste_selection()
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_selection()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
            if self._undo_stack.canUndo():
                self._undo_stack.undo()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Y:
            if self._undo_stack.canRedo():
                self._undo_stack.redo()
        else:
            super().keyPressEvent(event)

    def _apply_and_push(self, changes, desc):
        self.blockSignals(True)
        for r, c, _, new in changes:
            self.item(r, c).setText(new)
            self._current_data[(r, c)] = new
        self.blockSignals(False)
        self._is_editing_undo = True
        self._undo_stack.push(TableEditCommand(self, changes, desc))
        self._is_editing_undo = False
        for r, c, *_ in changes:
            self.cellChanged.emit(r, c)

    def _copy_selection(self):
        selected = self.selectedRanges()
        if not selected:
            return
        rows = sorted({r for rng in selected for r in range(rng.topRow(), rng.bottomRow() + 1)})
        cols = sorted({c for rng in selected for c in range(rng.leftColumn(), rng.rightColumn() + 1)})
        lines = ["\t".join(getattr(self.item(r, c), "text", lambda: "")() for c in cols) for r in rows]
        QApplication.clipboard().setText("\n".join(lines))

    def _paste_selection(self):
        text = QApplication.clipboard().text()
        current = self.currentItem()
        if not text or not current:
            return
        start_r, start_c = current.row(), current.column()
        changes = []
        for r_off, row_text in enumerate(text.split("\n")):
            for c_off, cell_text in enumerate(row_text.split("\t")):
                r, c = start_r + r_off, start_c + c_off
                if r < self.rowCount() and c < self.columnCount():
                    item = self.item(r, c)
                    if item and (item.flags() & Qt.ItemFlag.ItemIsEditable) and item.text() != cell_text:
                        changes.append((r, c, item.text(), cell_text))
        if changes:
            self._apply_and_push(changes, "Paste")

    def _delete_selection(self):
        changes = [
            (it.row(), it.column(), it.text(), "")
            for it in self.selectedItems()
            if (it.flags() & Qt.ItemFlag.ItemIsEditable) and it.text()
        ]
        if changes:
            self._apply_and_push(changes, "Delete")
