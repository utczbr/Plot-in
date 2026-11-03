# -*- coding: utf-8 -*-
"""
Model/View Virtualization and Async State Loading for File List (ADR-014 T-014.3/T-014.4 & ADR-017 T-017.5).

Provides:
- FileListRoles: custom ItemDataRole definitions for file list items.
- FileItem: dataclass representing a file entry.
- FileListModel: QAbstractListModel for virtualized, high-performance file listing (< 200 ms for 2,000 files).
- FileListFilterProxyModel: QSortFilterProxyModel for review queue filtering ("All", "Needs Review", "Reviewed").
- FileListDelegate: QStyledItemDelegate drawing state glyphs, filenames, and selection states with contrast >= 4.5:1.
- FileStateLoader: QRunnable for asynchronous, off-UI-thread state computation on QThreadPool.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PyQt6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    QRunnable,
    QSize,
    QSortFilterProxyModel,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from core.review_state import (
    ImageReviewState,
    derive_image_state,
)
from core.review_summary import (
    build_summary,
    write_summary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Data Roles
# ---------------------------------------------------------------------------

class FileListRoles:
    FilePathRole = Qt.ItemDataRole.UserRole + 1
    DisplayNameRole = Qt.ItemDataRole.UserRole + 2
    StateRole = Qt.ItemDataRole.UserRole + 3
    ConfidenceRole = Qt.ItemDataRole.UserRole + 4
    FailureReasonRole = Qt.ItemDataRole.UserRole + 5
    IsCachedRole = Qt.ItemDataRole.UserRole + 6


# ---------------------------------------------------------------------------
# State Glyph / Color Mappings
# ---------------------------------------------------------------------------

STATE_GLYPH_INFO = {
    ImageReviewState.PENDING.value: ("circle-regular-full.svg", "○", "#888888"),
    ImageReviewState.AUTO.value: ("circle-solid-full.svg", "●", "#4ec9b0"),
    ImageReviewState.NEEDS_REVIEW.value: ("triangle-exclamation-solid-full.svg", "▲", "#e5c07b"),
    ImageReviewState.REVIEWED.value: ("check-solid-full.svg", "✔", "#4CAF50"),
    ImageReviewState.FAILED.value: ("xmark-solid-full.svg", "✖", "#f44336"),
    ImageReviewState.UNKNOWN.value: ("circle-question-regular-full.svg", "?", "#9e9e9e"),
}


# ---------------------------------------------------------------------------
# Data Structure
# ---------------------------------------------------------------------------

@dataclass
class FileItem:
    path: str
    stem: str
    display_name: str
    state: str = ImageReviewState.PENDING.value
    confidence: Optional[float] = None
    failure_reason: Optional[str] = None
    is_cached: bool = False


# ---------------------------------------------------------------------------
# FileListModel
# ---------------------------------------------------------------------------

class FileListModel(QAbstractListModel):
    """Virtualized list model representing loaded images and their review states."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._items: List[FileItem] = []
        self._path_to_row: Dict[str, int] = {}

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._items)):
            return None

        item = self._items[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            if item.confidence is not None and item.state == ImageReviewState.NEEDS_REVIEW.value:
                pct = item.confidence * 100.0 if item.confidence <= 1.0 else item.confidence
                return f"[{pct:.0f}%] {item.display_name}"
            return item.display_name

        elif role == Qt.ItemDataRole.ToolTipRole:
            lines = [f"File: {item.display_name}", f"Path: {item.path}", f"State: {item.state}"]
            if item.confidence is not None:
                pct = item.confidence * 100.0 if item.confidence <= 1.0 else item.confidence
                lines.append(f"Confidence: {pct:.1f}%")
            if item.failure_reason:
                lines.append(f"Failure: {item.failure_reason}")
            if item.is_cached:
                lines.append("✓ Results cached")
            return "\n".join(lines)

        elif role == FileListRoles.FilePathRole:
            return item.path
        elif role == FileListRoles.DisplayNameRole:
            return item.display_name
        elif role == FileListRoles.StateRole:
            return item.state
        elif role == FileListRoles.ConfidenceRole:
            return item.confidence
        elif role == FileListRoles.FailureReasonRole:
            return item.failure_reason
        elif role == FileListRoles.IsCachedRole:
            return item.is_cached

        return None

    def set_files(
        self,
        files: List[str],
        unique_stems: Optional[Dict[str, str]] = None,
    ) -> None:
        """Populate the model with a list of file paths in a single model reset."""
        stems = unique_stems or {}
        self.beginResetModel()
        self._items = []
        self._path_to_row = {}
        for i, fpath in enumerate(files):
            stem = stems.get(fpath, Path(fpath).stem)
            display_name = os.path.basename(fpath)
            item = FileItem(path=fpath, stem=stem, display_name=display_name)
            self._items.append(item)
            self._path_to_row[fpath] = i
        self.endResetModel()

    def update_file_state(
        self,
        file_path: str,
        state: Union[str, ImageReviewState],
        confidence: Optional[float] = None,
        failure_reason: Optional[str] = None,
        is_cached: Optional[bool] = None,
    ) -> bool:
        """Update review state for a file by path. Returns True if row was found and updated."""
        row = self._path_to_row.get(file_path)
        if row is None or not (0 <= row < len(self._items)):
            return False

        if hasattr(state, "value"):
            state = state.value

        item = self._items[row]

        # Precedence guard: never downgrade an active cached state to PENDING
        if (item.is_cached or item.state != ImageReviewState.PENDING.value) and state == ImageReviewState.PENDING.value:
            return False

        changed = False

        if item.state != state:
            item.state = state
            changed = True
        if confidence is not None and item.confidence != confidence:
            item.confidence = confidence
            changed = True
        if failure_reason is not None and item.failure_reason != failure_reason:
            item.failure_reason = failure_reason
            changed = True
        if is_cached is not None and item.is_cached != is_cached:
            item.is_cached = is_cached
            changed = True

        if changed:
            idx = self.index(row, 0)
            self.dataChanged.emit(
                idx,
                idx,
                [
                    Qt.ItemDataRole.DisplayRole,
                    Qt.ItemDataRole.ToolTipRole,
                    FileListRoles.StateRole,
                    FileListRoles.ConfidenceRole,
                    FileListRoles.FailureReasonRole,
                    FileListRoles.IsCachedRole,
                ],
            )
        return True

    def reset_all_states(self) -> None:
        """Reset all items in the model back to PENDING and uncached (used when clearing cache)."""
        if not self._items:
            return
        for item in self._items:
            item.state = ImageReviewState.PENDING.value
            item.confidence = None
            item.failure_reason = None
            item.is_cached = False
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._items) - 1, 0)
        self.dataChanged.emit(
            top_left,
            bottom_right,
            [
                Qt.ItemDataRole.DisplayRole,
                Qt.ItemDataRole.ToolTipRole,
                FileListRoles.StateRole,
                FileListRoles.ConfidenceRole,
                FileListRoles.FailureReasonRole,
                FileListRoles.IsCachedRole,
            ],
        )

    def update_file_cached(self, file_path: str, cached: bool) -> bool:
        row = self._path_to_row.get(file_path)
        if row is None or not (0 <= row < len(self._items)):
            return False
        item = self._items[row]
        if item.is_cached != cached:
            item.is_cached = cached
            idx = self.index(row, 0)
            self.dataChanged.emit(idx, idx, [FileListRoles.IsCachedRole, Qt.ItemDataRole.ToolTipRole])
        return True

    def get_item(self, row: int) -> Optional[FileItem]:
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def index_for_path(self, file_path: str) -> QModelIndex:
        row = self._path_to_row.get(file_path)
        if row is not None and 0 <= row < len(self._items):
            return self.index(row, 0)
        return QModelIndex()


# ---------------------------------------------------------------------------
# FileListFilterProxyModel
# ---------------------------------------------------------------------------

class FileListFilterProxyModel(QSortFilterProxyModel):
    """Filter proxy for the review queue: [All | Needs Review | Reviewed]."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._filter_mode: str = "All"
        self._search_text: str = ""
        self.setDynamicSortFilter(True)

    @property
    def filter_mode(self) -> str:
        return self._filter_mode

    def set_filter_mode(self, mode: str) -> None:
        raw = mode.strip().lower()
        if raw in ("needs review", "needs_review"):
            normalized = "Needs Review"
        elif raw in ("reviewed",):
            normalized = "Reviewed"
        else:
            normalized = "All"
        if self._filter_mode != normalized:
            self._filter_mode = normalized
            self.invalidateFilter()

    def set_search_text(self, text: str) -> None:
        clean = text.strip().lower()
        if self._search_text != clean:
            self._search_text = clean
            self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return True

        idx = model.index(source_row, 0, source_parent)
        if not idx.isValid():
            return False

        # Search text match
        if self._search_text:
            display_name = str(idx.data(FileListRoles.DisplayNameRole) or "").lower()
            if self._search_text not in display_name:
                return False

        # Queue filter match
        if self._filter_mode == "All":
            return True

        state = idx.data(FileListRoles.StateRole)
        if self._filter_mode == "Needs Review":
            return state in (
                ImageReviewState.NEEDS_REVIEW.value,
                ImageReviewState.FAILED.value,
                ImageReviewState.UNKNOWN.value,
            )
        elif self._filter_mode == "Reviewed":
            return state == ImageReviewState.REVIEWED.value

        return True


# ---------------------------------------------------------------------------
# FileListDelegate
# ---------------------------------------------------------------------------

class FileListDelegate(QStyledItemDelegate):
    """Custom delegate painting state glyph, display name, and high-contrast selection."""

    def __init__(self, parent: Optional[QObject] = None, icon_getter=None) -> None:
        super().__init__(parent)
        self.icon_getter = icon_getter

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(option.rect.width(), 26)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        painter.save()

        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hover = bool(option.state & QStyle.StateFlag.State_MouseOver)

        rect = option.rect
        state = index.data(FileListRoles.StateRole) or ImageReviewState.PENDING.value
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""

        # Background rendering with WCAG >= 4.5:1 contrast
        if is_selected:
            bg_color = QColor("#264f78")  # High contrast selection
            border_color = QColor("#007acc")
            text_color = QColor("#ffffff")
        elif is_hover:
            bg_color = QColor("#2a2d2e")
            border_color = QColor("#3e3e42")
            text_color = QColor("#e0e0e0")
        else:
            bg_color = QColor("#1e1e1e")
            border_color = QColor("#252526")
            text_color = QColor("#cccccc")

        painter.fillRect(rect, bg_color)
        painter.setPen(border_color)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        # Glyph / Icon
        glyph_info = STATE_GLYPH_INFO.get(state, STATE_GLYPH_INFO[ImageReviewState.PENDING.value])
        icon_filename, fallback_char, default_color = glyph_info

        icon_rect = QRect(rect.x() + 6, rect.y() + (rect.height() - 14) // 2, 14, 14)
        painted_icon = False
        if self.icon_getter and callable(self.icon_getter):
            try:
                icon = self.icon_getter(icon_filename, color=default_color, size=14)
                if isinstance(icon, QIcon) and not icon.isNull():
                    icon.paint(painter, icon_rect)
                    painted_icon = True
            except Exception:
                painted_icon = False

        if not painted_icon:
            # Fallback text glyph
            painter.setPen(QColor(default_color))
            font = painter.font()
            font.setPixelSize(11)
            painter.setFont(font)
            painter.drawText(icon_rect, Qt.AlignmentFlag.AlignCenter, fallback_char)

        # Text
        text_rect = QRect(rect.x() + 26, rect.y(), rect.width() - 32, rect.height())
        painter.setPen(text_color)
        font = painter.font()
        font.setPixelSize(11)
        font.setBold(is_selected)
        painter.setFont(font)

        fm = QFontMetrics(font)
        elided_text = fm.elidedText(text, Qt.TextElideMode.ElideMiddle, text_rect.width() - 8)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided_text)

        painter.restore()


# ---------------------------------------------------------------------------
# FileStateLoader (Off-Thread Async State Computation)
# ---------------------------------------------------------------------------

class LoaderSignals(QObject):
    state_ready = pyqtSignal(str, str, object, object, object, int)  # (file_path, state, conf, reason, is_cached, gen)
    finished = pyqtSignal()


class FileStateLoader(QRunnable):
    """
    QRunnable background worker that computes review states across files.
    Reads lightweight <stem>_summary.json and <stem>_failed.json without blocking UI.
    Backfills missing summaries from legacy <stem>_analysis.json off-thread.
    """

    def __init__(
        self,
        output_dir: str,
        file_paths: List[str],
        unique_stems: Dict[str, str],
        threshold: float = 0.60,
        generation: int = 0,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.output_dir = Path(output_dir) if output_dir else None
        self.file_paths = list(file_paths)
        self.unique_stems = dict(unique_stems)
        self.threshold = threshold
        self.generation = generation
        self.signals = LoaderSignals()
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        if not self.output_dir or not self.output_dir.is_dir() or not self.file_paths:
            self.signals.finished.emit()
            return

        from visual.file_list_controller import (
            build_image_identity_metadata,
            compute_unique_image_stem,
            verify_identity_metadata,
        )

        for fpath in self.file_paths:
            if self._is_cancelled:
                break

            stem = self.unique_stems.get(fpath) or compute_unique_image_stem(fpath)
            failed_file = self.output_dir / f"{stem}_failed.json"
            summary_file = self.output_dir / f"{stem}_summary.json"
            analysis_file = self.output_dir / f"{stem}_analysis.json"

            # Fallback to raw stem only when unique_stems was not populated and legacy file exists
            if (
                fpath not in self.unique_stems
                and not failed_file.is_file()
                and not summary_file.is_file()
                and not analysis_file.is_file()
            ):
                raw_stem = Path(fpath).stem
                if (
                    (self.output_dir / f"{raw_stem}_failed.json").is_file()
                    or (self.output_dir / f"{raw_stem}_summary.json").is_file()
                    or (self.output_dir / f"{raw_stem}_analysis.json").is_file()
                ):
                    stem = raw_stem
                    failed_file = self.output_dir / f"{stem}_failed.json"
                    summary_file = self.output_dir / f"{stem}_summary.json"
                    analysis_file = self.output_dir / f"{stem}_analysis.json"

            # 1. Failed marker
            if failed_file.is_file():
                reason = "Analysis failed"
                try:
                    with open(failed_file, "r", encoding="utf-8") as f:
                        f_data = json.load(f)
                    if isinstance(f_data, dict):
                        reason = f_data.get("reason") or reason
                except Exception:
                    pass
                self.signals.state_ready.emit(
                    fpath,
                    ImageReviewState.FAILED.value,
                    None,
                    reason,
                    False,
                    self.generation,
                )
                continue

            # 2. Summary sidecar (lightweight + identity check)
            if summary_file.is_file():
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        summary_data = json.load(f)
                    if isinstance(summary_data, dict) and verify_identity_metadata(
                        fpath, summary_data.get("_image_identity")
                    ):
                        state = derive_image_state(summary_data, threshold=self.threshold)
                        conf = summary_data.get("min_confidence")
                        self.signals.state_ready.emit(
                            fpath,
                            state,
                            conf,
                            None,
                            True,
                            self.generation,
                        )
                        continue
                except Exception as ex:
                    logger.debug("Failed to read summary %s: %s", summary_file, ex)

            # 3. Analysis JSON exists
            if analysis_file.is_file():
                self.signals.state_ready.emit(
                    fpath,
                    ImageReviewState.UNKNOWN.value,
                    None,
                    None,
                    True,
                    self.generation,
                )
                try:
                    with open(analysis_file, "r", encoding="utf-8") as f:
                        res = json.load(f)
                    if isinstance(res, dict):
                        stored_id = (
                            res.get("_image_identity")
                            or (res.get("metadata") or {}).get("image_identity")
                        )
                        if not verify_identity_metadata(fpath, stored_id):
                            self.signals.state_ready.emit(
                                fpath,
                                ImageReviewState.PENDING.value,
                                None,
                                None,
                                False,
                                self.generation,
                            )
                            continue
                        summary = build_summary(res)
                        summary["_image_identity"] = stored_id or build_image_identity_metadata(fpath, unique_stem=stem)
                        try:
                            write_summary(summary_file, summary)
                        except TypeError:
                            write_summary(self.output_dir, stem, summary)
                        state = derive_image_state(summary, threshold=self.threshold)
                        conf = summary.get("min_confidence")
                        self.signals.state_ready.emit(
                            fpath,
                            state,
                            conf,
                            None,
                            True,
                            self.generation,
                        )
                        continue
                except Exception as ex:
                    logger.warning("Failed to backfill summary for %s: %s", analysis_file, ex)

            # 4. No valid cached files exist -> Pending
            self.signals.state_ready.emit(
                fpath,
                ImageReviewState.PENDING.value,
                None,
                None,
                False,
                self.generation,
            )

        self.signals.finished.emit()
