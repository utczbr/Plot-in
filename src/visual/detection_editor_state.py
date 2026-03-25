# -*- coding: utf-8 -*-
"""
Detection Editor State — QUndoStack-based editing commands for the detection scene.

Provides undo/redo support for all detection editing operations:
- Move (drag a bounding box)
- Resize (change bbox dimensions)
- Delete (remove a detection)
- Create (add a new detection)
- ChangeClass (reassign detection class)

Each detection stores ``original_xyxy`` on first edit to enable "Reset to Auto".
"""

from __future__ import annotations

import copy
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QRectF, pyqtSignal, QObject
from PyQt6.QtGui import QUndoCommand, QUndoStack

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Safety helper
# ──────────────────────────────────────────────────────────────────────

def _item_is_alive(item) -> bool:
    """Return False if the underlying C++ QGraphicsItem has been destroyed.

    Calling any method on a PyQt6 wrapper whose C++ peer was deleted raises
    RuntimeError.  We probe with ``scene()`` — a lightweight, side-effect-free
    call — to detect this before performing any real mutation.
    """
    try:
        _ = item.scene()
        return True
    except RuntimeError:
        return False


# ──────────────────────────────────────────────────────────────────────
# Undo Commands
# ──────────────────────────────────────────────────────────────────────

class MoveCommand(QUndoCommand):
    """Undo-able move of a detection bounding box."""

    def __init__(self, item, old_xyxy: List[float], new_xyxy: List[float],
                 description: str = "Move Box"):
        super().__init__(description)
        self._item = item
        self._old_xyxy = old_xyxy
        self._new_xyxy = new_xyxy
        self._ensure_original_snapshot()

    def _ensure_original_snapshot(self):
        if not _item_is_alive(self._item):
            return
        if 'original_xyxy' not in self._item.detection:
            self._item.detection['original_xyxy'] = list(self._old_xyxy)

    def redo(self):
        self._apply(self._new_xyxy)

    def undo(self):
        self._apply(self._old_xyxy)

    def _apply(self, xyxy: List[float]):
        if not _item_is_alive(self._item):
            logger.debug("MoveCommand._apply: item has been deleted — skipping")
            return
        x1, y1, x2, y2 = xyxy
        self._item.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        self._item.setPos(0, 0)


class ResizeCommand(QUndoCommand):
    """Undo-able resize of a detection bounding box."""

    def __init__(self, item, old_xyxy: List[float], new_xyxy: List[float],
                 description: str = "Resize Box"):
        super().__init__(description)
        self._item = item
        self._old_xyxy = old_xyxy
        self._new_xyxy = new_xyxy
        self._ensure_original_snapshot()

    def _ensure_original_snapshot(self):
        if not _item_is_alive(self._item):
            return
        if 'original_xyxy' not in self._item.detection:
            self._item.detection['original_xyxy'] = list(self._old_xyxy)

    def redo(self):
        self._apply(self._new_xyxy)

    def undo(self):
        self._apply(self._old_xyxy)

    def _apply(self, xyxy: List[float]):
        if not _item_is_alive(self._item):
            logger.debug("ResizeCommand._apply: item has been deleted — skipping")
            return
        x1, y1, x2, y2 = xyxy
        self._item.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        self._item.setPos(0, 0)


class DeleteCommand(QUndoCommand):
    """Undo-able deletion of a detection item from the scene."""

    def __init__(self, scene, item, description: str = "Delete Box"):
        super().__init__(description)
        self._scene = scene
        self._item = item
        self._was_visible = item.isVisible() if _item_is_alive(item) else True

    def redo(self):
        if not _item_is_alive(self._item):
            logger.debug("DeleteCommand.redo: item already deleted — skipping")
            return
        # Guard against double-removal (e.g. redo called twice)
        try:
            scene_of_item = self._item.scene()
        except RuntimeError:
            return
        if scene_of_item is not None:
            self._item.setVisible(False)
            self._scene.removeItem(self._item)
        if self._item in self._scene._rect_items:
            self._scene._rect_items.remove(self._item)

    def undo(self):
        if not _item_is_alive(self._item):
            logger.debug("DeleteCommand.undo: item has been deleted — cannot restore")
            return
        self._scene.addItem(self._item)
        self._scene._rect_items.append(self._item)
        self._item.setVisible(self._was_visible)


class CreateCommand(QUndoCommand):
    """Undo-able creation of a new detection item."""

    def __init__(self, scene, item, description: str = "Create Box"):
        super().__init__(description)
        self._scene = scene
        self._item = item

    def redo(self):
        if not _item_is_alive(self._item):
            logger.debug("CreateCommand.redo: item has been deleted — skipping")
            return
        # Avoid double-add
        if self._item.scene() is None:
            self._scene.addItem(self._item)
        if self._item not in self._scene._rect_items:
            self._scene._rect_items.append(self._item)

    def undo(self):
        if not _item_is_alive(self._item):
            logger.debug("CreateCommand.undo: item has been deleted — skipping")
            return
        if self._item in self._scene._rect_items:
            self._scene._rect_items.remove(self._item)
        if self._item.scene() is not None:
            self._scene.removeItem(self._item)


class ChangeClassCommand(QUndoCommand):
    """Undo-able class reassignment of a detection item."""

    def __init__(self, item, old_class: str, new_class: str,
                 colors_map: dict, description: str = "Change Class"):
        super().__init__(description)
        self._item = item
        self._old_class = old_class
        self._new_class = new_class
        self._colors_map = colors_map

    def redo(self):
        self._apply(self._new_class)

    def undo(self):
        self._apply(self._old_class)

    def _apply(self, class_name: str):
        if not _item_is_alive(self._item):
            logger.debug("ChangeClassCommand._apply: item has been deleted — skipping")
            return
        from visual.detection_scene import DEFAULT_COLORS
        self._item.class_name = class_name
        palette = self._colors_map.get(class_name,
                    DEFAULT_COLORS.get(class_name, DEFAULT_COLORS["other"]))
        from PyQt6.QtGui import QColor, QPen
        r, g, b = palette["normal"]
        self._item._normal_color = QColor(r, g, b)
        rh, gh, bh = palette["highlight"]
        self._item._highlight_color = QColor(rh, gh, bh)
        pen = self._item.pen()
        pen.setColor(self._item._normal_color)
        self._item.setPen(pen)


class MoveKeypointCommand(QUndoCommand):
    """Undo-able move of a single pie slice keypoint."""

    def __init__(self, point_item, old_pos, new_pos,
                 description: str = "Move Keypoint"):
        super().__init__(description)
        self._point_item = point_item
        self._old_pos = old_pos  # QPointF
        self._new_pos = new_pos

    def redo(self):
        if not _item_is_alive(self._point_item):
            logger.debug("MoveKeypointCommand.redo: item deleted — skipping")
            return
        self._point_item.setPos(self._new_pos)
        parent = self._point_item.parentItem()
        if parent is not None and hasattr(parent, '_update_lines'):
            parent._update_lines()

    def undo(self):
        if not _item_is_alive(self._point_item):
            logger.debug("MoveKeypointCommand.undo: item deleted — skipping")
            return
        self._point_item.setPos(self._old_pos)
        parent = self._point_item.parentItem()
        if parent is not None and hasattr(parent, '_update_lines'):
            parent._update_lines()


class CreateKeypointCommand(QUndoCommand):
    """Undo-able addition of a new pie slice keypoint."""

    def __init__(self, group, point_item, description: str = "Add Keypoint"):
        super().__init__(description)
        self._group = group
        self._point_item = point_item

    def redo(self):
        if not _item_is_alive(self._point_item):
            logger.debug("CreateKeypointCommand.redo: item deleted — skipping")
            return
        if self._point_item not in self._group.point_items:
            self._group.point_items.append(self._point_item)
        self._group._update_lines()

    def undo(self):
        if not _item_is_alive(self._point_item):
            logger.debug("CreateKeypointCommand.undo: item deleted — skipping")
            return
        if self._point_item in self._group.point_items:
            self._group.point_items.remove(self._point_item)
        if self._point_item.scene():
            self._point_item.scene().removeItem(self._point_item)
        self._group._update_lines()


# ──────────────────────────────────────────────────────────────────────
# Editor State Manager
# ──────────────────────────────────────────────────────────────────────

class EditorStateManager(QObject):
    """
    Manages the editing state for the detection editor.

    Wraps a ``QUndoStack`` and tracks edit counts for the status label.

    Signals
    -------
    edit_count_changed(int)
        Emitted when the number of edits changes (for status label update).
    """

    edit_count_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._undo_stack = QUndoStack(self)
        self._undo_stack.indexChanged.connect(self._on_index_changed)

    @property
    def undo_stack(self) -> QUndoStack:
        return self._undo_stack

    @property
    def edit_count(self) -> int:
        """Number of edits in the stack (may differ from index if undo was used)."""
        return self._undo_stack.index()

    @property
    def can_undo(self) -> bool:
        return self._undo_stack.canUndo()

    @property
    def can_redo(self) -> bool:
        return self._undo_stack.canRedo()

    @property
    def is_dirty(self) -> bool:
        """True if any edits have been made since last reset."""
        return not self._undo_stack.isClean()

    def push(self, command: QUndoCommand) -> None:
        """Push an undo command onto the stack."""
        self._undo_stack.push(command)

    def undo(self) -> None:
        self._undo_stack.undo()

    def redo(self) -> None:
        self._undo_stack.redo()

    def clear(self) -> None:
        """Clear all undo history."""
        self._undo_stack.clear()

    def mark_clean(self) -> None:
        """Mark the current state as clean (e.g., after Apply & Re-Extract)."""
        self._undo_stack.setClean()

    def _on_index_changed(self, idx: int):
        self.edit_count_changed.emit(idx)

    def get_status_text(self) -> str:
        """Generate a human-readable status string."""
        idx = self._undo_stack.index()
        if idx == 0:
            return ""
        parts = []
        # Count command types
        moves = resizes = deletes = creates = class_changes = 0
        for i in range(idx):
            cmd = self._undo_stack.command(i)
            if isinstance(cmd, MoveCommand):
                moves += 1
            elif isinstance(cmd, ResizeCommand):
                resizes += 1
            elif isinstance(cmd, DeleteCommand):
                deletes += 1
            elif isinstance(cmd, CreateCommand):
                creates += 1
            elif isinstance(cmd, ChangeClassCommand):
                class_changes += 1

        if moves:
            parts.append(f"{moves} moved")
        if resizes:
            parts.append(f"{resizes} resized")
        if deletes:
            parts.append(f"{deletes} deleted")
        if creates:
            parts.append(f"{creates} new")
        if class_changes:
            parts.append(f"{class_changes} reclassed")

        return " • ".join(parts) if parts else f"{idx} edits"
