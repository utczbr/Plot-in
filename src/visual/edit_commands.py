# -*- coding: utf-8 -*-
"""
Logical QUndoCommand implementations for Results table edits (ADR-015 T-015.3 / T-015.5).

Records edits by immutable `row_uid` and column identifier (`col_id`), never by
visual `(row, col)` coordinates, so filtering or sorting the table never misdirects
undo or redo operations.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple
from PyQt6.QtGui import QUndoCommand

from visual.row_edit_service import EditRecord, RowEditService


class EditCellCommand(QUndoCommand):
    """Logical undo command for a single cell edit on a protocol row."""

    def __init__(
        self,
        result: dict,
        row_uid: str,
        col_id: str,
        old_val: Any,
        new_val: Any,
        record: Optional[EditRecord] = None,
        on_after_apply: Optional[Callable[[List[str]], None]] = None,
        description: str = "Edit Cell",
    ):
        super().__init__(description)
        self.result = result
        self.row_uid = row_uid
        self.col_id = col_id
        self.old_val = old_val
        self.new_val = new_val
        self.record = record
        self._on_after_apply = on_after_apply
        self._first_redo = record is not None

    def redo(self) -> None:
        if self._first_redo:
            self._first_redo = False
            return
        self.record = RowEditService.apply(
            self.result, self.row_uid, self.col_id, self.new_val
        )
        if self._on_after_apply is not None:
            self._on_after_apply([self.row_uid])

    def undo(self) -> None:
        if self.record is not None:
            RowEditService.revert(self.record)
        if self._on_after_apply is not None:
            self._on_after_apply([self.row_uid])


class BatchEditCommand(QUndoCommand):
    """Single-step macro command for multi-cell Paste, Delete, and Ctrl+D Fill-Down."""

    def __init__(
        self,
        result: dict,
        edits: List[Tuple[str, str, Any, Any]],
        records: Optional[List[EditRecord]] = None,
        on_after_apply: Optional[Callable[[List[str]], None]] = None,
        description: str = "Batch Edit",
    ):
        super().__init__(description)
        self.result = result
        self.edits = list(edits)  # [(row_uid, col_id, old_val, new_val), ...]
        self.records: List[EditRecord] = list(records) if records is not None else []
        self._on_after_apply = on_after_apply
        self._first_redo = records is not None

    def redo(self) -> None:
        if self._first_redo:
            self._first_redo = False
            return
        self.records = []
        affected_uids: List[str] = []
        for row_uid, col_id, _old_val, new_val in self.edits:
            rec = RowEditService.apply(self.result, row_uid, col_id, new_val)
            self.records.append(rec)
            if row_uid not in affected_uids:
                affected_uids.append(row_uid)
        if self._on_after_apply is not None:
            self._on_after_apply(affected_uids)

    def undo(self) -> None:
        affected_uids: List[str] = []
        for rec in reversed(self.records):
            RowEditService.revert(rec)
            if rec.row_uid not in affected_uids:
                affected_uids.append(rec.row_uid)
        if self._on_after_apply is not None:
            self._on_after_apply(affected_uids)