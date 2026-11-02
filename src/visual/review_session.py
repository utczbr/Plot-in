# -*- coding: utf-8 -*-
"""
ReviewSession View-Model (ADR-007 T-007.2, ADR-008 T-008.1).

Central owner of review state:
- Current analysis result and context
- Processing lifecycle state (ProcessingState)
- Protocol row filtering and on-demand generation
- Reversible dirty tracking across sources
- Navigation guard interception

Imports ONLY from PyQt6.QtCore. Absolutely NO QtWidgets or QtGui.
"""

from __future__ import annotations

from enum import Enum
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

VALID_DIRTY_SOURCES = frozenset({"boxes", "ocr_text", "data_table", "results_table"})


class SessionUndoStack(QObject):
    """QtCore-only undo stack with QUndoStack-compatible API (limit 200, T-015.5).

    Supports both duck-typed commands and ``PyQt6.QtGui.QUndoCommand`` objects
    while keeping ``review_session.py`` strictly free of ``QtGui``/``QtWidgets`` imports.
    """

    indexChanged = pyqtSignal(int)
    cleanChanged = pyqtSignal(bool)
    canUndoChanged = pyqtSignal(bool)
    canRedoChanged = pyqtSignal(bool)

    def __init__(self, parent: Optional[QObject] = None, undo_limit: int = 200):
        super().__init__(parent)
        self._commands: List[Any] = []
        self._index: int = 0
        self._clean_index: int = 0
        self._limit: int = undo_limit

    def setUndoLimit(self, limit: int) -> None:
        self._limit = max(1, int(limit))

    def undoLimit(self) -> int:
        return self._limit

    def clear(self) -> None:
        was_can_undo = self.canUndo()
        was_can_redo = self.canRedo()
        was_clean = self.isClean()
        self._commands.clear()
        self._index = 0
        self._clean_index = 0
        if was_can_undo:
            self.canUndoChanged.emit(False)
        if was_can_redo:
            self.canRedoChanged.emit(False)
        if not was_clean:
            self.cleanChanged.emit(True)
        self.indexChanged.emit(0)

    def push(self, cmd: Any) -> None:
        was_clean = self.isClean()
        cmd.redo()
        if self._index < len(self._commands):
            if self._clean_index > self._index:
                self._clean_index = -1
            del self._commands[self._index :]
        self._commands.append(cmd)
        if len(self._commands) > self._limit:
            overflow = len(self._commands) - self._limit
            del self._commands[:overflow]
            if self._clean_index >= 0:
                self._clean_index -= overflow
                if self._clean_index < 0:
                    self._clean_index = -1
        self._index = len(self._commands)
        self.canUndoChanged.emit(self.canUndo())
        self.canRedoChanged.emit(self.canRedo())
        if was_clean != self.isClean():
            self.cleanChanged.emit(self.isClean())
        self.indexChanged.emit(self._index)

    def canUndo(self) -> bool:
        return self._index > 0

    def canRedo(self) -> bool:
        return self._index < len(self._commands)

    def undo(self) -> None:
        if not self.canUndo():
            return
        was_clean = self.isClean()
        self._index -= 1
        cmd = self._commands[self._index]
        cmd.undo()
        self.canUndoChanged.emit(self.canUndo())
        self.canRedoChanged.emit(self.canRedo())
        if was_clean != self.isClean():
            self.cleanChanged.emit(self.isClean())
        self.indexChanged.emit(self._index)

    def redo(self) -> None:
        if not self.canRedo():
            return
        was_clean = self.isClean()
        cmd = self._commands[self._index]
        self._index += 1
        cmd.redo()
        self.canUndoChanged.emit(self.canUndo())
        self.canRedoChanged.emit(self.canRedo())
        if was_clean != self.isClean():
            self.cleanChanged.emit(self.isClean())
        self.indexChanged.emit(self._index)

    def count(self) -> int:
        return len(self._commands)

    def index(self) -> int:
        return self._index

    def isClean(self) -> bool:
        return self._index == self._clean_index

    def setClean(self) -> None:
        was_clean = self.isClean()
        self._clean_index = self._index
        if not was_clean:
            self.cleanChanged.emit(True)


class ProcessingState(str, Enum):
    IDLE = "idle"
    ANALYSING = "analysing"
    BATCH = "batch"
    CANCELLING = "cancelling"


class ReviewSession(QObject):
    """
    QtCore-only view-model managing review state, active filters,
    and reversible dirty sources.
    """

    dirty_changed = pyqtSignal(bool)
    filters_changed = pyqtSignal()
    result_changed = pyqtSignal(dict)
    protocol_rows_changed = pyqtSignal(list)
    processing_changed = pyqtSignal(object)

    VALID_DIRTY_SOURCES = VALID_DIRTY_SOURCES

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._image_path: Optional[str] = None
        self._image_index: int = -1
        self._result: Optional[Dict[str, Any]] = None
        self._context: Optional[Dict[str, Any]] = None
        self._processing: ProcessingState = ProcessingState.IDLE
        self._filters: Dict[str, str] = {
            "status": "All",
            "outcome": "All",
            "group": "All",
        }
        self._dirty_sources: Set[str] = set()
        self._navigation_guard: Optional[Callable[[int], bool]] = None
        self._undo_stack: SessionUndoStack = SessionUndoStack(self, undo_limit=200)
        self._undo_stack.cleanChanged.connect(
            lambda is_clean: self.set_source_dirty("results_table", not is_clean)
        )

    @property
    def undo_stack(self) -> SessionUndoStack:
        return self._undo_stack

    # --- Processing State ---

    @property
    def processing(self) -> ProcessingState:
        return self._processing

    @processing.setter
    def processing(self, state: ProcessingState | bool) -> None:
        if isinstance(state, bool):
            state = ProcessingState.ANALYSING if state else ProcessingState.IDLE
        elif not isinstance(state, ProcessingState):
            try:
                state = ProcessingState(str(state))
            except ValueError:
                state = ProcessingState.ANALYSING if state else ProcessingState.IDLE

        if self._processing != state:
            self._processing = state
            self.processing_changed.emit(self._processing)

    @property
    def is_processing(self) -> bool:
        return self._processing != ProcessingState.IDLE

    @is_processing.setter
    def is_processing(self, value: bool) -> None:
        self.processing = value

    # --- Image Index / Path ---

    @property
    def image_path(self) -> Optional[str]:
        return self._image_path

    @image_path.setter
    def image_path(self, val: Optional[str]) -> None:
        self._image_path = val

    @property
    def image_index(self) -> int:
        return self._image_index

    @image_index.setter
    def image_index(self, val: int) -> None:
        self._image_index = val

    # --- Result Management ---

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        return self._result

    @property
    def current_analysis_result(self) -> Optional[Dict[str, Any]]:
        return self._result

    @current_analysis_result.setter
    def current_analysis_result(self, value: Optional[Dict[str, Any]]) -> None:
        self.set_result(value, self._context)

    def set_result(
        self,
        result: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._result = result
        self._context = context
        self._undo_stack.clear()
        if self._result is not None:
            from core.protocol_row_builder import build_protocol_rows, ensure_row_uids
            if "protocol_rows" not in self._result:
                self._result["protocol_rows"] = build_protocol_rows(self._result, self._context or {})
            elif isinstance(self._result.get("protocol_rows"), list):
                ensure_row_uids(self._result["protocol_rows"])
        rows = self._result.get("protocol_rows", []) if self._result else []
        self.reset_filters()
        self.mark_clean()
        self.result_changed.emit(self._result or {})
        self.protocol_rows_changed.emit(rows)

    # --- Protocol Rows & Filtering ---

    @property
    def filters(self) -> Dict[str, str]:
        return dict(self._filters)

    def set_filter(self, key: str, value: Optional[str]) -> None:
        val = value if value and value.strip() else "All"
        if self._filters.get(key) != val:
            self._filters[key] = val
            self.filters_changed.emit()

    def reset_filters(self) -> None:
        changed = any(v != "All" for v in self._filters.values())
        self._filters = {"status": "All", "outcome": "All", "group": "All"}
        if changed:
            self.filters_changed.emit()

    def filtered_rows(self) -> List[Dict[str, Any]]:
        if not self._result:
            return []

        if "protocol_rows" not in self._result:
            from core.protocol_row_builder import build_protocol_rows
            self._result["protocol_rows"] = build_protocol_rows(
                self._result, self._context or {}
            )

        rows = self._result.get("protocol_rows", [])
        if not isinstance(rows, list):
            return []
        from core.protocol_row_builder import ensure_row_uids
        ensure_row_uids(rows)

        status_filter = self._filters.get("status", "All")
        outcome_filter = self._filters.get("outcome", "All")
        group_filter = self._filters.get("group", "All")

        filtered = [r for r in rows if isinstance(r, dict)]
        if outcome_filter != "All":
            filtered = [r for r in filtered if r.get("outcome") == outcome_filter]
        if group_filter != "All":
            filtered = [r for r in filtered if r.get("group") == group_filter]
        if status_filter != "All":
            filtered = [r for r in filtered if r.get("review_status") == status_filter]
        return filtered

    def counts(self) -> Tuple[int, int]:
        shown_rows = self.filtered_rows()
        rows = (self._result or {}).get("protocol_rows", [])
        total = len([r for r in rows if isinstance(r, dict)]) if isinstance(rows, list) else 0
        return len(shown_rows), total

    def set_protocol_rows(self, rows: List[Dict[str, Any]]) -> None:
        if self._result is None:
            self._result = {}
        if isinstance(rows, list):
            from core.protocol_row_builder import ensure_row_uids
            ensure_row_uids(rows)
        self._result["protocol_rows"] = rows
        self.protocol_rows_changed.emit(rows)

    def update_row_field(
        self, row_index_or_uid: Any, field: str, value: Any
    ) -> None:
        if self._result is None or "protocol_rows" not in self._result:
            return
        rows = self._result["protocol_rows"]
        if not isinstance(rows, list):
            return
        from core.protocol_row_builder import ensure_row_uids
        ensure_row_uids(rows)

        if isinstance(row_index_or_uid, int):
            if 0 <= row_index_or_uid < len(rows):
                target_row = rows[row_index_or_uid]
                if isinstance(target_row, dict):
                    target_row[field] = value
                    self.protocol_rows_changed.emit(rows)
        elif row_index_or_uid is not None:
            for r in rows:
                if isinstance(r, dict) and (
                    r.get("_uid") == row_index_or_uid
                    or r.get("id") == row_index_or_uid
                ):
                    r[field] = value
                    self.protocol_rows_changed.emit(rows)
                    break

    # --- Dirty State Tracking (T-008.1) ---

    @property
    def is_dirty(self) -> bool:
        return len(self._dirty_sources) > 0

    @property
    def dirty_sources(self) -> Set[str]:
        return set(self._dirty_sources)

    def set_source_dirty(self, source: str, is_dirty: bool) -> None:
        if source not in VALID_DIRTY_SOURCES:
            raise ValueError(
                f"Unknown dirty source: {source}. Expected one of {sorted(VALID_DIRTY_SOURCES)}"
            )
        was_dirty = self.is_dirty
        if is_dirty:
            self._dirty_sources.add(source)
        else:
            self._dirty_sources.discard(source)
        now_dirty = self.is_dirty
        if was_dirty != now_dirty:
            self.dirty_changed.emit(now_dirty)

    def mark_clean(self) -> None:
        was_dirty = self.is_dirty
        self._dirty_sources.clear()
        self._undo_stack.setClean()
        if was_dirty:
            self.dirty_changed.emit(False)

    # --- Navigation Guard (T-008.2) ---

    def register_navigation_guard(
        self, guard: Optional[Callable[..., bool]]
    ) -> None:
        self._navigation_guard = guard

    def set_navigation_guard(
        self, guard: Optional[Callable[..., bool]]
    ) -> None:
        self._navigation_guard = guard

    def request_navigation(self, target_index: Optional[int] = None) -> bool:
        if self._processing in (ProcessingState.BATCH, ProcessingState.ANALYSING):
            return False
        if self._navigation_guard is not None:
            try:
                sig = inspect.signature(self._navigation_guard)
                if len(sig.parameters) == 0:
                    return bool(self._navigation_guard())
            except (ValueError, TypeError):
                pass
            return bool(self._navigation_guard(target_index))
        return True
