# -*- coding: utf-8 -*-
"""
Interactive Detection Scene — QGraphicsView-based canvas for chart detection editing.

This module replaces the legacy QLabel + QPixmap image display with a
QGraphicsView + QGraphicsScene that lives at 1:1 original pixel coordinates.
All detection bounding boxes and keypoints are scene items at their original
``xyxy`` positions; zoom and pan are handled by the view transform.

Classes
-------
DetectionScene       — QGraphicsScene managing base image + detection items.
DetectionCanvasView  — QGraphicsView handling zoom, pan, and mode-dependent input.
EditableRectItem     — Draggable/resizable bounding-box item (Phase 3 adds handles).
BaselineItem         — Horizontal/vertical baseline line item.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF, QPoint, QRect, QSize
from PyQt6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
    QBrush,
    QCursor,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QGraphicsEllipseItem,
    QGraphicsItemGroup,
    QRubberBand,
)
from PyQt6.QtWidgets import QGraphicsSceneMouseEvent, QGraphicsSceneHoverEvent
from PyQt6.QtGui import QContextMenuEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication


logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class EditorMode(Enum):
    """Interaction mode for the detection canvas."""
    VIEW = auto()           # Pan / click-to-highlight only
    EDIT_BOXES = auto()     # Select, move, resize bounding boxes
    CREATE_BOX = auto()     # Click-drag to draw a new bbox
    EDIT_KEYPOINTS = auto() # Move keypoints (Phase 5)
    CREATE_KEYPOINT = auto()# Place new keypoint (Phase 5)


# ──────────────────────────────────────────────────────────────────────
# Default color palette  (matches main_modern.py self.colors)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_COLORS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "bar":         {"normal": (0, 120, 255),   "highlight": (30, 144, 255)},
    "slice":       {"normal": (255, 90, 90),   "highlight": (255, 140, 140)},
    "line":        {"normal": (255, 0, 0),     "highlight": (255, 99, 71)},
    "scatter":     {"normal": (0, 128, 0),     "highlight": (50, 205, 50)},
    "box":         {"normal": (128, 0, 128),   "highlight": (147, 112, 219)},
    "data_point":  {"normal": (255, 165, 60),  "highlight": (255, 195, 60)},
    "axis_title":  {"normal": (255, 165, 0),   "highlight": (255, 165, 0)},
    "chart_title": {"normal": (50, 50, 220),   "highlight": (100, 100, 255)},
    "legend":      {"normal": (210, 105, 30),  "highlight": (210, 180, 140)},
    "axis_labels": {"normal": (255, 0, 255),   "highlight": (255, 105, 180)},
    "scale_label": {"normal": (255, 117, 24),  "highlight": (255, 140, 0)},
    "tick_label":  {"normal": (0, 255, 255),   "highlight": (0, 206, 209)},
    "error_bar":   {"normal": (220, 20, 60),   "highlight": (255, 99, 71)},
    "range_indicator": {"normal": (0, 191, 255), "highlight": (135, 206, 250)},
    "median_line": {"normal": (255, 215, 0),   "highlight": (255, 223, 0)},
    "outlier":     {"normal": (255, 20, 147),  "highlight": (255, 105, 180)},
    "significance_marker": {"normal": (100, 200, 100), "highlight": (144, 238, 144)},
    "connector_line": {"normal": (205, 133, 63), "highlight": (222, 184, 135)},
    "other":       {"normal": (128, 128, 128), "highlight": (192, 192, 192)},
    "baseline":    {"normal": (240, 240, 240), "highlight": (240, 240, 240)},
}

_NORMAL_PEN_WIDTH = 2
_HIGHLIGHT_PEN_WIDTH = 3
_HIGHLIGHT_Z_BOOST = 100  # z-value boost for highlighted items


# ──────────────────────────────────────────────────────────────────────
# EditableRectItem
# ──────────────────────────────────────────────────────────────────────

class EditableRectItem(QGraphicsRectItem):
    """
    A bounding-box rectangle in original-pixel coordinates.

    In VIEW mode the item is not movable/selectable.
    In EDIT_BOXES mode it becomes movable+selectable (Phase 3 will add resize handles).

    The item stores a reference to its detection dict and class name so that
    ``export_detections()`` can reconstruct the full detections payload.
    """

    def __init__(
        self,
        xyxy: List[float],
        class_name: str,
        detection: Dict[str, Any],
        colors: Dict[str, Tuple[int, int, int]] | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        x1, y1, x2, y2 = xyxy
        w = x2 - x1
        h = y2 - y1
        super().__init__(QRectF(0, 0, w, h), parent)
        self.setPos(x1, y1)

        self.class_name = class_name
        self.detection = detection  # reference to the original dict
        self._base_z = 0.0

        # Colors
        palette = colors or DEFAULT_COLORS.get(class_name, DEFAULT_COLORS["other"])
        r, g, b = palette["normal"]
        self._normal_color = QColor(r, g, b)
        rh, gh, bh = palette["highlight"]
        self._highlight_color = QColor(rh, gh, bh)

        # Normal pen
        pen = QPen(self._normal_color)
        pen.setWidth(_NORMAL_PEN_WIDTH)
        pen.setCosmetic(True)  # width in screen pixels, not scene units
        self.setPen(pen)
        from PyQt6.QtGui import QBrush
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        # Tooltip
        conf = detection.get("conf", "")
        conf_str = f" ({conf:.0%})" if isinstance(conf, (int, float)) else ""
        text = detection.get("text", "")
        text_str = f": {text}" if text else ""
        self.setToolTip(f"{class_name}{conf_str}{text_str}")

        # Start non-interactive (VIEW mode)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        
        self._mode = EditorMode.VIEW
        self._is_dragging = False
        self._is_resizing = False
        self._resize_edges = ""
        self._drag_start_pos = QPointF()
        self._drag_start_rect = QRectF()
        self._drag_start_item_pos = QPointF()
        
    def set_editor_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        if mode != EditorMode.EDIT_BOXES:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    _MIN_HANDLE_SIZE = 2.0
    _MAX_HANDLE_SIZE = 12.0
    _EDGE_TOLERANCE_HANDLE_MULTIPLIER = 1.25
    _MAX_EDGE_TOLERANCE = 10.0

    def _handle_size(self) -> float:
        rect = self.rect()
        clamped_smallest_dimension = max(1.0, min(rect.width(), rect.height()))
        # Scale handles with box size so tiny boxes get tiny handles too.
        return max(self._MIN_HANDLE_SIZE, min(clamped_smallest_dimension * 0.06, self._MAX_HANDLE_SIZE))

    def _get_resize_edges(self, pos: QPointF) -> str:
        if self._mode != EditorMode.EDIT_BOXES:
            return ""
        edge_tolerance = max(
            self._MIN_HANDLE_SIZE,
            min(
                self._handle_size() * self._EDGE_TOLERANCE_HANDLE_MULTIPLIER,
                self._MAX_EDGE_TOLERANCE,
            ),
        )
        rect = self.rect()
        edges = ""
        if abs(pos.y() - rect.top()) <= edge_tolerance:
            edges += "top"
        elif abs(pos.y() - rect.bottom()) <= edge_tolerance:
            edges += "bottom"
        if abs(pos.x() - rect.left()) <= edge_tolerance:
            edges += "left"
        elif abs(pos.x() - rect.right()) <= edge_tolerance:
            edges += "right"
        return edges

    def paint(self, painter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        if self._mode == EditorMode.EDIT_BOXES and self.isSelected():
            painter.save()
            painter.setBrush(QBrush(QColor(0, 200, 0, 153)))  # 60% opacity green
            # Keep paint cosmetic
            pen = QPen(Qt.GlobalColor.black)
            pen.setWidth(1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            
            rect = self.rect()
            s = self._handle_size()
            handles = [
                QRectF(rect.left(), rect.top(), s, s),
                QRectF(rect.center().x() - s/2, rect.top(), s, s),
                QRectF(rect.right() - s, rect.top(), s, s),
                QRectF(rect.left(), rect.center().y() - s/2, s, s),
                QRectF(rect.right() - s, rect.center().y() - s/2, s, s),
                QRectF(rect.left(), rect.bottom() - s, s, s),
                QRectF(rect.center().x() - s/2, rect.bottom() - s, s, s),
                QRectF(rect.right() - s, rect.bottom() - s, s, s),
            ]
            for h in handles:
                painter.drawRect(h)
            painter.restore()

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        if self._mode != EditorMode.EDIT_BOXES:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            super().hoverMoveEvent(event)
            return
        edges = self._get_resize_edges(event.pos())
        if edges in ["topleft", "bottomright"]:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif edges in ["topright", "bottomleft"]:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif edges in ["left", "right"]:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif edges in ["top", "bottom"]:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._mode != EditorMode.EDIT_BOXES:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            edges = self._get_resize_edges(event.pos())
            if edges:
                self._is_resizing = True
                self._resize_edges = edges
            else:
                self._is_dragging = True
            
            self._drag_start_pos = event.scenePos()
            self._drag_start_rect = self.rect()
            self._drag_start_item_pos = self.pos()
            
            scene = self.scene()
            if scene:
                scene.clearSelection()
            
            self.setSelected(True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._mode != EditorMode.EDIT_BOXES:
            super().mouseMoveEvent(event)
            return

        if self._is_resizing:
            delta = event.scenePos() - self._drag_start_pos
            new_rect = QRectF(self._drag_start_rect)
            
            if "left" in self._resize_edges:
                new_rect.setLeft(min(new_rect.right() - 5, new_rect.left() + delta.x()))
            elif "right" in self._resize_edges:
                new_rect.setRight(max(new_rect.left() + 5, new_rect.right() + delta.x()))
                
            if "top" in self._resize_edges:
                new_rect.setTop(min(new_rect.bottom() - 5, new_rect.top() + delta.y()))
            elif "bottom" in self._resize_edges:
                new_rect.setBottom(max(new_rect.top() + 5, new_rect.bottom() + delta.y()))
                
            # Clamp to image bounds
            if self.scene():
                img = self.scene().sceneRect()
                if img.isValid():
                    local_img = self.mapRectFromScene(img)
                    new_rect = new_rect.intersected(local_img)
            self.setRect(new_rect)
            event.accept()
            return
            
        elif self._is_dragging:
            delta = event.scenePos() - self._drag_start_pos
            new_pos = self._drag_start_item_pos + delta
            # Clamp drag so box stays within image bounds
            if self.scene():
                img = self.scene().sceneRect()
                r = self.rect()
                new_pos.setX(max(img.left() - r.x(), min(new_pos.x(), img.right() - r.width() - r.x())))
                new_pos.setY(max(img.top() - r.y(), min(new_pos.y(), img.bottom() - r.height() - r.y())))
            self.setPos(new_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._mode != EditorMode.EDIT_BOXES:
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_resizing or self._is_dragging:
                old_x1 = self._drag_start_rect.x() + self._drag_start_item_pos.x()
                old_y1 = self._drag_start_rect.y() + self._drag_start_item_pos.y()
                old_x2 = old_x1 + self._drag_start_rect.width()
                old_y2 = old_y1 + self._drag_start_rect.height()
                old_xyxy = [old_x1, old_y1, old_x2, old_y2]
                
                curr = self.current_xyxy()
                if old_xyxy != curr:
                    scene = self.scene()
                    if hasattr(scene, "item_edited"):
                        scene.item_edited.emit(self, old_xyxy, self._is_resizing)
                        
                self._is_resizing = False
                self._is_dragging = False
                self._resize_edges = ""
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        if self._mode != EditorMode.EDIT_BOXES:
            super().contextMenuEvent(event)
            return

        menu = QMenu()
        delete_action = menu.addAction("Delete Box")
        
        class_menu = menu.addMenu("Change Class")
        for cname in DEFAULT_COLORS.keys():
            if cname not in ("baseline", "other"):
                action = class_menu.addAction(cname.replace("_", " ").title())
                action.setData(cname)
                
        action = menu.exec(event.screenPos())
        if not action:
            return
            
        if action == delete_action:
            scene = self.scene()
            if hasattr(scene, "item_deleted"):
                scene.item_deleted.emit(self)
        elif action.parentWidget() == class_menu:
            new_class = action.data()
            if new_class and new_class != self.class_name:
                scene = self.scene()
                if hasattr(scene, "item_class_changed"):
                    scene.item_class_changed.emit(self, new_class)
        event.accept()

    def keyPressEvent(self, event) -> None:
        if self._mode != EditorMode.EDIT_BOXES or not self.isSelected():
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()

        # Nudge
        step = 10.0 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1.0

        delta = QPointF(0, 0)
        if key == Qt.Key.Key_Left:
            delta = QPointF(-step, 0)
        elif key == Qt.Key.Key_Right:
            delta = QPointF(step, 0)
        elif key == Qt.Key.Key_Up:
            delta = QPointF(0, -step)
        elif key == Qt.Key.Key_Down:
            delta = QPointF(0, step)

        if not delta.isNull():
            old_xyxy = self.current_xyxy()
            new_pos = self.pos() + delta
            # Clamp so the box stays within image bounds (Fix E-3)
            scene = self.scene()
            if scene:
                img = scene.sceneRect()
                r = self.rect()
                new_pos.setX(max(img.left(), min(new_pos.x(), img.right() - r.width())))
                new_pos.setY(max(img.top(), min(new_pos.y(), img.bottom() - r.height())))
            self.setPos(new_pos)
            # Sync detection dict so highlight_item_by_bbox can find the new coords
            self.detection["xyxy"] = self.current_xyxy()
            if scene and hasattr(scene, "item_edited"):
                scene.item_edited.emit(self, old_xyxy, False)
            event.accept()
            return

        # Number-key class change (1-9)
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            idx = key - Qt.Key.Key_1
            # Standard classes
            classes = [k for k in DEFAULT_COLORS.keys() if k not in ("baseline", "other")]
            if idx < len(classes):
                new_class = classes[idx]
                if new_class != self.class_name:
                    scene = self.scene()
                    if hasattr(scene, "item_class_changed"):
                        scene.item_class_changed.emit(self, new_class)
                event.accept()
                return

        super().keyPressEvent(event)

    # ── Highlight (selection) ──

    def set_highlighted(self, highlighted: bool) -> None:
        """
        Toggle visual highlight state — pen width increase + z-value boost.
        Preserves the "pop" effect from the legacy create_image_with_highlight().
        """
        pen = self.pen()
        if highlighted:
            pen.setColor(self._highlight_color)
            pen.setWidth(_HIGHLIGHT_PEN_WIDTH)
            self.setZValue(self._base_z + _HIGHLIGHT_Z_BOOST)
        else:
            pen.setColor(self._normal_color)
            pen.setWidth(_NORMAL_PEN_WIDTH)
            self.setZValue(self._base_z)
        self.setPen(pen)

    # ── Coordinate export ──

    def current_xyxy(self) -> List[float]:
        """Return the current bounding box in scene (original-pixel) coordinates.

        Uses ``rect() + pos()`` rather than ``sceneBoundingRect()`` so the
        cosmetic pen width does not inflate the exported coordinates.
        """
        r = self.rect()
        p = self.pos()
        return [r.x() + p.x(), r.y() + p.y(),
                r.x() + r.width() + p.x(), r.y() + r.height() + p.y()]


# ──────────────────────────────────────────────────────────────────────
# PieSliceGroup & EditablePointItem (Phase 5)
# ──────────────────────────────────────────────────────────────────────

class EditablePointItem(QGraphicsEllipseItem):
    """A movable point item representing a pie chart keypoint."""
    def __init__(self, x: float, y: float, idx: int, parent: QGraphicsItem | None = None) -> None:
        r = 4.0
        super().__init__(-r, -r, r * 2, r * 2, parent)
        self.setPos(x, y)
        self.idx = idx
        self.setBrush(QBrush(QColor(255, 255, 0)))  # yellow
        pen = QPen(Qt.GlobalColor.black)
        pen.setCosmetic(True)
        self.setPen(pen)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self._mode = EditorMode.VIEW
        # Pre-drag position snapshot, updated at mouse-press before Qt commits the move.
        # This fixes the KP-2 bug where ItemPositionHasChanged fires *after* the position
        # is committed, making old_pos == new_pos in the undo command.
        self._drag_start_pos: QPointF = QPointF(x, y)

    def set_editor_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        can_move = (mode == EditorMode.EDIT_KEYPOINTS)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, can_move)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Snapshot position before Qt moves us — ensures undo command has a valid old_pos."""
        self._drag_start_pos = QPointF(self.pos())
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            p = self.parentItem()
            if hasattr(p, "point_moved"):
                # Pass the pre-drag snapshot as old_pos so MoveKeypointCommand
                # stores distinct old/new positions and undo actually works.
                p.point_moved(self.idx, self._drag_start_pos, self.pos())
        return super().itemChange(change, value)

class PieSliceGroup(QGraphicsItemGroup):
    """
    A group containing EditablePointItems and visual lines for a pie slice.
    Automatically updates its bounding box (`xyxy`) when keypoints move.
    """
    def __init__(
        self,
        class_name: str,
        detection: Dict[str, Any],
        colors: Dict[str, Tuple[int, int, int]] | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self.class_name = class_name
        self.detection = detection
        self._base_z = 0.0
        
        # Colors
        palette = colors or DEFAULT_COLORS.get(class_name, DEFAULT_COLORS["slice"])
        r, g, b = palette["normal"]
        self._normal_color = QColor(r, g, b)
        rh, gh, bh = palette["highlight"]
        self._highlight_color = QColor(rh, gh, bh)

        self._pen = QPen(self._normal_color)
        self._pen.setWidth(_NORMAL_PEN_WIDTH)
        self._pen.setCosmetic(True)

        self.point_items: List[EditablePointItem] = []
        self._lines: List[QGraphicsLineItem] = []
        self._mode = EditorMode.VIEW
        
        # Initialize
        self._init_keypoints()
        
    def set_editor_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        for p in self.point_items:
            p.set_editor_mode(mode)
            
    def _init_keypoints(self) -> None:
        kps = self.detection.get("keypoints", [])
        if not kps:
            return
            
        import numpy as np
        kps_arr = np.asarray(kps)
        if kps_arr.ndim == 1:
            if kps_arr.size >= 15:
                kps_arr = kps_arr.reshape(-1, 3)
            elif kps_arr.size >= 10:
                kps_arr = kps_arr.reshape(-1, 2)
            else:
                return
                
        # create points
        for i in range(min(5, len(kps_arr))):
            x, y = float(kps_arr[i, 0]), float(kps_arr[i, 1])
            if x <= 0 and y <= 0 and i > 0:
                continue
            pt = EditablePointItem(x, y, i, self)
            self.point_items.append(pt)
            
        # Draw lines from center to boundary points
        self._update_lines()
        
    def _update_lines(self) -> None:
        # Clear old lines
        for line in self._lines:
            if line.scene():
                line.scene().removeItem(line)
        self._lines.clear()
        
        if not self.point_items:
            return
            
        center = self.point_items[0].pos()
        for i in range(1, len(self.point_items)):
            p = self.point_items[i].pos()
            line = QGraphicsLineItem(center.x(), center.y(), p.x(), p.y(), self)
            line.setPen(self._pen)
            self._lines.append(line)
            
    def point_moved(self, idx: int, old_pos: QPointF, new_pos: QPointF) -> None:
        """Called by EditablePointItem when dragged.

        Parameters
        ----------
        idx:     Index of the moved point in self.point_items.
        old_pos: Position recorded at mouse-press (before Qt committed the move).
                 Supplied by EditablePointItem._drag_start_pos to fix KP-2 where
                 ItemPositionHasChanged fires after the position is committed, making
                 a naive snapshot inside itemChange return new_pos == old_pos.
        new_pos: Current (new) position after the move.
        """
        self._update_lines()
        if self._mode == EditorMode.EDIT_KEYPOINTS:
            pt = self.point_items[idx] if idx < len(self.point_items) else None
            if pt is not None:
                scene = self.scene()
                if hasattr(scene, 'keypoint_moved'):
                    scene.keypoint_moved.emit(pt, old_pos, new_pos)
        
    def set_highlighted(self, highlighted: bool) -> None:
        if highlighted:
            self._pen.setColor(self._highlight_color)
            self._pen.setWidth(_HIGHLIGHT_PEN_WIDTH)
            self.setZValue(self._base_z + _HIGHLIGHT_Z_BOOST)
        else:
            self._pen.setColor(self._normal_color)
            self._pen.setWidth(_NORMAL_PEN_WIDTH)
            self.setZValue(self._base_z)
            
        for line in self._lines:
            line.setPen(self._pen)
            
    def current_xyxy(self) -> List[float]:
        # Compute bounding box of all points
        if not self.point_items:
            return self.detection.get("xyxy", [0, 0, 0, 0])
        import numpy as np
        pts = np.array([[p.pos().x(), p.pos().y()] for p in self.point_items])
        return [float(pts[:, 0].min()), float(pts[:, 1].min()), 
                float(pts[:, 0].max()), float(pts[:, 1].max())]
                
    def export_keypoints(self) -> List[List[float]]:
        # Restore into original shape [x, y, conf(if original had it)]
        orig_kps = self.detection.get("keypoints", [])
        if not orig_kps:
            return []
            
        import numpy as np
        orig_arr = np.asarray(orig_kps)
        is_flat = (orig_arr.ndim == 1)
        if is_flat:
            if orig_arr.size >= 15:
                orig_arr = orig_arr.reshape(-1, 3)
            elif orig_arr.size >= 10:
                orig_arr = orig_arr.reshape(-1, 2)
                
        new_arr = np.copy(orig_arr)
        for i, pt in enumerate(self.point_items):
            if i < len(new_arr):
                new_arr[i, 0] = pt.pos().x()
                new_arr[i, 1] = pt.pos().y()
                
        if is_flat:
            return new_arr.flatten().tolist()
        return new_arr.tolist()

# ──────────────────────────────────────────────────────────────────────
# BaselineItem
# ──────────────────────────────────────────────────────────────────────

class BaselineItem(QGraphicsLineItem):
    """
    A horizontal or vertical baseline line in original-pixel coordinates.
    Gold-colored, cosmetic pen (constant screen width).
    """

    def __init__(
        self,
        coord: float,
        orientation: str,
        image_width: int,
        image_height: int,
        parent: QGraphicsItem | None = None,
    ) -> None:
        if orientation == "horizontal":
            super().__init__(0, coord, image_width, coord, parent)
        else:
            super().__init__(coord, 0, coord, image_height, parent)

        pen = QPen(QColor(255, 215, 0))  # gold
        pen.setWidth(2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(50)  # above rects, below highlight


# ──────────────────────────────────────────────────────────────────────
# DetectionScene
# ──────────────────────────────────────────────────────────────────────

class DetectionScene(QGraphicsScene):
    """
    QGraphicsScene managing the base chart image and detection overlay items.

    Coordinates are in **original image pixel space** (1:1). The view transform
    handles zoom/pan — items never need to be rescaled.

    Signals
    -------
    item_highlighted(class_name, xyxy)
        Emitted when an item receives a highlight (for table sync).
    """

    item_highlighted = pyqtSignal(str, list)
    
    @staticmethod
    def _resolve_cal_func(cal_obj) -> Optional[callable]:
        """
        Extract a pixel→value callable from a CalibrationResult dataclass or
        legacy dict.  Priority: .func attribute → .coeffs tuple → dict keys.
        """
        if cal_obj is None:
            return None
        # CalibrationResult dataclass (primary path)
        if hasattr(cal_obj, 'func') and callable(cal_obj.func):
            return cal_obj.func
        # Fallback: coeffs tuple (m, b) on dataclass
        if hasattr(cal_obj, 'coeffs') and cal_obj.coeffs:
            m, b = cal_obj.coeffs
            return lambda px, _m=m, _b=b: _m * px + _b
        # Legacy dict shapes
        if isinstance(cal_obj, dict):
            func = cal_obj.get('func') or cal_obj.get('model_func')
            if callable(func):
                return func
            coeffs = cal_obj.get('coeffs') or cal_obj.get('coefficients')
            if coeffs:
                m, b = coeffs
                return lambda px, _m=m, _b=b: _m * px + _b
        return None

    def get_calculated_coords(self, pixel_x: float, pixel_y: float) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to chart scale values using attached calibration.

        The calibration dict uses CalibrationResult objects (frozen dataclass
        with .func callable and .coeffs tuple) keyed by 'x' and 'y'.
        Falls back to 'primary' if neither axis key is populated.

        Returns None when no usable calibration is available.
        """
        if not hasattr(self, '_calibration') or not self._calibration:
            return None

        cal = self._calibration
        x_func = self._resolve_cal_func(cal.get('x') or cal.get('primary'))
        y_func = self._resolve_cal_func(cal.get('y') or cal.get('primary'))

        calc_x = float(x_func(pixel_x)) if x_func else None
        calc_y = float(y_func(pixel_y)) if y_func else None

        if calc_x is None and calc_y is None:
            return None

        return (calc_x if calc_x is not None else 0.0,
                calc_y if calc_y is not None else 0.0)

    def set_calibration(self, calibration_data: dict) -> None:
        """Store calibration mapping for real-time coordinate translation."""
        self._calibration = calibration_data or {}
    
    # Phase 3 Editor signals:
    item_edited = pyqtSignal(object, list, bool)  # item, old_xyxy, is_resize
    item_deleted = pyqtSignal(object)             # item
    item_class_changed = pyqtSignal(object, str)  # item, new_class_name
    item_created = pyqtSignal(str, list)          # class_name, xyxy
    
    # Phase 5 Keypoint signals:
    keypoint_moved = pyqtSignal(object, object, object)  # point_item, old_pos (QPointF), new_pos (QPointF)
    keypoint_created = pyqtSignal(object, object)        # group, point_item

    # Phase 7 Selection sync:
    box_selected = pyqtSignal(str)  # class_name of the selected box

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._rect_items: List[EditableRectItem] = []
        self._baseline_item: Optional['BaselineItem'] = None
        self._calibration: dict = {}
        self._visible_classes: Dict[str, bool] = {}
        self._colors: Dict[str, Dict[str, Tuple[int, int, int]]] = dict(DEFAULT_COLORS)
        self._image_width = 0
        self._image_height = 0
        self._highlighted_item: Optional[EditableRectItem] = None
        self._mode = EditorMode.VIEW

        # Emit box_selected when the user selects a single box in Edit mode
        self.selectionChanged.connect(self._on_selection_changed)
        
    def set_editor_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        for item in self._rect_items:
            item.set_editor_mode(mode)

    # ── Selection sync ──

    def _on_selection_changed(self) -> None:
        """Emit box_selected when a single EditableRectItem is selected in Edit mode."""
        if self._mode != EditorMode.EDIT_BOXES:
            return
        selected = self.selectedItems()
        print(f"_on_selection_changed: len(selected)={len(selected)}", flush=True)
        if len(selected) == 1 and isinstance(selected[0], EditableRectItem):
            print(f"_on_selection_changed emitting: {selected[0].class_name}", flush=True)
            self.box_selected.emit(selected[0].class_name)

    # ── Image ──

    def load_image(self, pixmap: QPixmap) -> None:
        """Set the base chart image. Clears any previous content."""
        self.clear()
        self._rect_items.clear()
        self._baseline_item = None
        self._highlighted_item = None
        self._visible_classes.clear()

        self._image_width = pixmap.width()
        self._image_height = pixmap.height()
        self.setSceneRect(0, 0, self._image_width, self._image_height)

        self._base_pixmap_item = QGraphicsPixmapItem(pixmap)
        self._base_pixmap_item.setZValue(-1)
        self.addItem(self._base_pixmap_item)

    def load_image_from_path(self, image_path: str) -> bool:
        """Load an image from a file path. Returns True on success."""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            logger.error("Failed to load image: %s", image_path)
            return False
        self.load_image(pixmap)
        return True

    # ── Detections ──

    def set_detections(
        self,
        detections: Dict[str, List[Dict[str, Any]]],
        colors: Optional[Dict[str, Dict[str, Tuple[int, int, int]]]] = None,
    ) -> None:
        """
        Populate the scene with detection bounding boxes.

        Parameters
        ----------
        detections : dict
            Mapping of class_name → list of detection dicts.
            Each detection dict must have an 'xyxy' key.
        colors : dict, optional
            Override color palette (class_name → {normal, highlight}).
        """
        # Remove old rect items (keep base image + baseline)
        for item in self._rect_items:
            self.removeItem(item)
        self._rect_items.clear()
        self._highlighted_item = None
        self._raw_detections = dict(detections) if detections else {}

        if colors:
            self._colors.update(colors)

        z = 1.0
        for class_name, items in detections.items():
            if not isinstance(items, list):
                continue

            for det in items:
                bbox = det.get("xyxy")
                if not bbox:
                    continue  # Skip non-bbox detections (e.g. layout_text_regions metadata)

                palette = self._colors.get(class_name, self._colors.get("other"))
                if class_name == "slice" and "keypoints" in det:
                    item = PieSliceGroup(
                        class_name=class_name,
                        detection=det,
                        colors=palette,
                    )
                else:
                    item = EditableRectItem(
                        xyxy=bbox,
                        class_name=class_name,
                        detection=det,
                        colors=palette,
                    )
                item._base_z = z
                item.setZValue(z)
                z += 0.01  # slight stacking order

                # Apply initial visibility
                visible = self._visible_classes.get(class_name, True)
                item.setVisible(visible)

                self.addItem(item)
                self._rect_items.append(item)

    def set_baseline(
        self, coord: float, orientation: str = "horizontal"
    ) -> None:
        """Add or update the baseline line."""
        if self._baseline_item:
            self.removeItem(self._baseline_item)

        self._baseline_item = BaselineItem(
            coord, orientation, self._image_width, self._image_height
        )
        visible = self._visible_classes.get("baseline", True)
        self._baseline_item.setVisible(visible)
        self.addItem(self._baseline_item)

    # ── Visibility ──

    def set_class_visible(self, class_name: str, visible: bool) -> None:
        """Show or hide all items of a given class."""
        self._visible_classes[class_name] = visible

        if class_name == "baseline":
            if self._baseline_item:
                self._baseline_item.setVisible(visible)
            return

        for item in self._rect_items:
            if item.class_name == class_name:
                item.setVisible(visible)

    # ── Highlight ──

    def highlight_item_by_bbox(
        self, bbox: Optional[List[float]], class_name: Optional[str] = None
    ) -> None:
        """
        Highlight a specific detection item by its xyxy coordinates.
        Clears any previous highlight first.
        """
        # Clear previous
        if self._highlighted_item is not None:
            self._highlighted_item.set_highlighted(False)
            self._highlighted_item = None

        if bbox is None:
            return

        # Find matching item
        from visual.detection_editor_state import _item_is_alive
        for item in list(self._rect_items):
            if not _item_is_alive(item):
                self._rect_items.remove(item)  # prune stale wrappers
                continue
            det_bbox = item.detection.get("xyxy")
            if det_bbox == bbox and (class_name is None or item.class_name == class_name):
                item.set_highlighted(True)
                self._highlighted_item = item
                self.item_highlighted.emit(item.class_name, list(det_bbox))
                break

    def clear_highlight(self) -> None:
        """Remove any active highlight."""
        self.highlight_item_by_bbox(None)

    # ── Export ──

    def export_detections(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Export current scene items back to the standard detections dict format.

        Returns original-pixel-space coordinates — no inverse scaling needed.
        Retains metadata properties without bounding boxes from original payload.
        """
        result: Dict[str, List[Dict[str, Any]]] = {}
        
        # Bring over metadata/non-bbox lists from the original input
        raw = getattr(self, '_raw_detections', {})
        import copy
        for k, v in raw.items():
            if not isinstance(v, list):
                result[k] = copy.deepcopy(v)
            else:
                # Keep elements that deliberately lack xyxy
                non_bbox = [i for i in v if isinstance(i, dict) and 'xyxy' not in i]
                if non_bbox:
                    result[k] = copy.deepcopy(non_bbox)
        
        # Export all active rect items
        for item in self._rect_items:
            det = dict(item.detection)  # shallow copy
            det["xyxy"] = item.current_xyxy()
            if isinstance(item, PieSliceGroup):
                det["keypoints"] = item.export_keypoints()
            result.setdefault(item.class_name, []).append(det)
        return result


# ──────────────────────────────────────────────────────────────────────
# DetectionCanvasView
# ──────────────────────────────────────────────────────────────────────

class DetectionCanvasView(QGraphicsView):
    """
    QGraphicsView with zoom (Shift+wheel), pan, and mode-dependent input.

    Input mapping
    -------------
    VIEW mode:
      - Left-click       → click-to-highlight
      - Middle-click      → pan
      - Shift+wheel       → zoom
    EDIT_BOXES mode:
      - Left-click        → select / drag item
      - Middle-click      → pan
      - Shift+Left        → rubber-band multi-select
      - Shift+wheel       → zoom
    CREATE_BOX mode:
      - Left-click+drag   → draw new box
      - Middle-click      → pan
      - Shift+wheel       → zoom
    """

    ZOOM_MIN = 0.1
    ZOOM_MAX = 10.0
    ZOOM_STEP = 1.15

    # Signal emitted when user clicks on an item in VIEW mode
    bbox_clicked = pyqtSignal(str, list)  # class_name, xyxy
    
    # Signal emitted continuously when hovering over the canvas
    coordinates_hovered = pyqtSignal(QPointF)

    def __init__(self, scene: DetectionScene, parent=None):
        self._rubber_band = None
        self._rubber_band_origin = QPoint()
        self._create_class_name = "box"  # Default
        super().__init__(scene, parent)
        self._det_scene = scene
        self._mode = EditorMode.VIEW
        self._zoom_level = 1.0

        # Rendering
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

        # Background
        self.setBackgroundBrush(QColor(30, 30, 30))

        # Start in scroll-hand-drag for VIEW mode pan
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # Track mouse to emit hover coords
        self.setMouseTracking(True)

    # ── Mode ──

    @property
    def mode(self) -> EditorMode:
        return self._mode

    def set_create_class(self, class_name: str) -> None:
        self._create_class_name = class_name

    def set_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        self._det_scene.set_editor_mode(mode)
        if mode == EditorMode.VIEW:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        elif mode == EditorMode.EDIT_BOXES:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        elif mode == EditorMode.CREATE_BOX:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    # ── Zoom ──

    @property
    def zoom_level(self) -> float:
        return self._zoom_level

    def set_zoom(self, level: float) -> None:
        """Set absolute zoom level."""
        level = max(self.ZOOM_MIN, min(self.ZOOM_MAX, level))
        factor = level / self._zoom_level
        self._zoom_level = level
        self.scale(factor, factor)

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom_level * self.ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom_level / self.ZOOM_STEP)

    def fit_to_view(self) -> None:
        """Fit the entire scene into the viewport."""
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # Compute the resulting zoom level from the transform matrix
        self._zoom_level = self.transform().m11()

    def reset_zoom(self) -> None:
        """Reset to 100% (1:1 pixel mapping)."""
        self.set_zoom(1.0)

    # ── Events ──

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Shift+wheel → zoom; plain wheel → scroll."""
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:
        """Delete key removes selected items in EDIT_BOXES mode.
           Ctrl+Z / Ctrl+Y trigger undo/redo via the scene signal.
        """
        from PyQt6.QtCore import Qt
        if self._mode == EditorMode.EDIT_BOXES:
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                for item in list(self._det_scene._rect_items):
                    if item.isSelected() and item.isVisible():
                        self._det_scene.item_deleted.emit(item)
                event.accept()
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        """Mode-dependent left-click handling."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            fake = type(event)(
                event.type(), event.position(), Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton, event.modifiers()
            )
            super().mousePressEvent(fake)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._mode == EditorMode.VIEW:
                scene_pos = self.mapToScene(event.position().toPoint())
                for item in self.scene().items(scene_pos):
                    if isinstance(item, EditableRectItem) and item.isVisible():
                        bbox = item.detection.get("xyxy")
                        if bbox:
                            self._det_scene.highlight_item_by_bbox(bbox, item.class_name)
                            self.bbox_clicked.emit(item.class_name, list(bbox))
                            event.accept()
                            return
                self._det_scene.clear_highlight()
            
            elif self._mode == EditorMode.CREATE_BOX:
                self._rubber_band_origin = event.position().toPoint()
                if not self._rubber_band:
                    self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
                self._rubber_band.setGeometry(QRect(self._rubber_band_origin, QSize(0, 0)))
                self._rubber_band.show()
                event.accept()
                return
            elif self._mode == EditorMode.CREATE_KEYPOINT:
                scene_pos = self.mapToScene(event.position().toPoint())
                clicked_groups = [item for item in self.scene().items(scene_pos) if isinstance(item, PieSliceGroup) and item.isVisible()]
                
                target_group = clicked_groups[0] if clicked_groups else None
                if target_group is not None:
                    if len(target_group.point_items) < 5:
                        pt = EditablePointItem(scene_pos.x(), scene_pos.y(), len(target_group.point_items), target_group)
                        pt.set_editor_mode(self._mode)
                        target_group.point_items.append(pt)
                        target_group._update_lines()
                        
                        scene = self.scene()
                        if hasattr(scene, 'keypoint_created'):
                            scene.keypoint_created.emit(target_group, pt)
                event.accept()
                return
            elif self._mode == EditorMode.EDIT_BOXES:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
                else:
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._mode == EditorMode.CREATE_BOX and self._rubber_band and self._rubber_band.isVisible():
            rect = QRect(self._rubber_band_origin, event.position().toPoint()).normalized()
            self._rubber_band.setGeometry(rect)
            event.accept()
            return
            
        super().mouseMoveEvent(event)
        
        # Emit hover coordinates
        scene_pos = self.mapToScene(event.position().toPoint())
        self.coordinates_hovered.emit(scene_pos)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            if self._mode == EditorMode.VIEW:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            else:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                
        elif event.button() == Qt.MouseButton.LeftButton and self._mode == EditorMode.CREATE_BOX:
            if self._rubber_band and self._rubber_band.isVisible():
                self._rubber_band.hide()
                
                # Convert screen rect to scene coordinates
                rect = self._rubber_band.geometry()
                if rect.width() > 5 and rect.height() > 5:
                    top_left = self.mapToScene(rect.topLeft())
                    bottom_right = self.mapToScene(rect.bottomRight())
                    
                    xyxy = [top_left.x(), top_left.y(), bottom_right.x(), bottom_right.y()]
                    
                    # Emit to scene
                    if hasattr(self._det_scene, "item_created"):
                        self._det_scene.item_created.emit(self._create_class_name, xyxy)
                
                event.accept()
                return
        elif event.button() == Qt.MouseButton.LeftButton and self._mode == EditorMode.EDIT_BOXES:
            # Always reset drag mode after rubber-band or normal release
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

        # In EDIT_BOXES mode always ensure drag mode is NoDrag after any release
        if self._mode == EditorMode.EDIT_BOXES:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """
        Right-click anywhere on the canvas → shows pixel + calculated coordinates.
        Delegates to item specific context menus if they handle it first.
        """
        # 1. Let the scene (and items) handle it first.
        super().contextMenuEvent(event)
        if event.isAccepted():
            return
            
        # 2. It wasn't handled by an item. Build the view-level context info menu.
        event.accept()
        
        scene_pos = self.mapToScene(event.pos())
        scene_rect = self.scene().sceneRect()

        # Ignore clicks outside the image bounds
        if not scene_rect.contains(scene_pos):
            return

        pixel_x = round(scene_pos.x(), 2)
        pixel_y = round(scene_pos.y(), 2)

        # Check if we are over a detection box
        det_info = None
        class_name = None
        for item in self.scene().items(scene_pos):
            if isinstance(item, EditableRectItem) and item.isVisible():
                det_info = item.detection
                class_name = item.class_name
                break

        menu = QMenu(self)

        # ── Basic info (shown as disabled labels) ──
        menu.addAction(f"Pixel: ({pixel_x}, {pixel_y})").setEnabled(False)

        # Calculated coordinates placeholder
        calc_coords = self._det_scene.get_calculated_coords(pixel_x, pixel_y)
        if calc_coords:
            calc_x_str = f"{calc_coords[0]:.3f}"
            calc_y_str = f"{calc_coords[1]:.3f}"
            calc_str = f"({calc_x_str}, {calc_y_str})"
        else:
            calc_x_str = "N/A"
            calc_y_str = "N/A"
            calc_str = "N/A – scaling not mapped"

        menu.addAction(f"Calculated: {calc_str}").setEnabled(False)

        # ── Copy actions ──
        menu.addSeparator()
        copy_calc_y = menu.addAction("Copy Calculated Y")
        copy_calc_x = menu.addAction("Copy Calculated X")
        copy_pixel = menu.addAction("Copy Pixel Coordinates")

        # ── Detection info (at the end) ──
        if det_info:
            menu.addSeparator()
            conf = det_info.get("conf")
            conf_str = f"{conf:.1%}" if isinstance(conf, (int, float)) else "N/A"
            menu.addAction(f"Class: {class_name or 'unknown'}").setEnabled(False)
            menu.addAction(f"Confidence: {conf_str}").setEnabled(False)
            text = det_info.get("text", "")
            if text:
                menu.addAction(f"OCR/Text: {text}").setEnabled(False)

        chosen = menu.exec(event.globalPos())

        if chosen == copy_calc_y:
            QApplication.clipboard().setText(calc_y_str)
        elif chosen == copy_calc_x:
            QApplication.clipboard().setText(calc_x_str)
        elif chosen == copy_pixel:
            QApplication.clipboard().setText(f"({pixel_x}, {pixel_y})")
