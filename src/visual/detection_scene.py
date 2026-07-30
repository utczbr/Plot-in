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
    CREATE_SLICE = auto()   # Create a new pie slice (Phase 6)


# ──────────────────────────────────────────────────────────────────────
# Default color palette  (matches main_modern.py self.colors)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_COLORS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "bar":         {"normal": (0, 120, 255),   "highlight": (30, 144, 255)},
    "slice":       {"normal": (255, 90, 90),   "highlight": (46, 204, 113)},
    "line":        {"normal": (255, 0, 0),     "highlight": (255, 99, 71)},
    "scatter":     {"normal": (0, 128, 0),     "highlight": (50, 205, 50)},
    "box":         {"normal": (128, 0, 128),   "highlight": (147, 112, 219)},
    "data_point":  {"normal": (255, 165, 60),  "highlight": (255, 195, 60)},
    "axis_title":  {"normal": (255, 165, 0),   "highlight": (255, 165, 0)},
    "chart_title": {"normal": (50, 50, 220),   "highlight": (100, 100, 255)},
    "legend":      {"normal": (210, 105, 30),  "highlight": (210, 180, 140)},
    "color_bar":   {"normal": (0, 160, 160),   "highlight": (0, 220, 220)},
    "color_bar_label": {"normal": (0, 200, 200), "highlight": (0, 255, 255)},
    "color_bar_title": {"normal": (200, 50, 200), "highlight": (255, 100, 255)},
    "cell":        {"normal": (100, 149, 237), "highlight": (135, 206, 250)},
    "axis_labels": {"normal": (255, 0, 255),   "highlight": (255, 105, 180)},
    "data_label":       {"normal": (200, 200, 80),  "highlight": (230, 230, 100)},
    "color_bar_region": {"normal": (0, 180, 180),  "highlight": (0, 240, 240)},
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
        self._update_tooltip()

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

    def _update_tooltip(self) -> None:
        conf = self.detection.get("conf", self.detection.get("ocr_confidence", ""))
        conf_str = f" ({conf:.0%})" if isinstance(conf, (int, float)) else ""
        text = self.detection.get("text", "")
        text_str = f": {text}" if text else ""
        self.setToolTip(f"{self.class_name}{conf_str}{text_str}")

    def update_text_and_tooltip(self, text: str, conf: Optional[float] = None) -> None:
        """Update detection text in memory and refresh item tooltip on canvas."""
        self.detection["text"] = text
        if conf is not None:
            self.detection["ocr_confidence"] = float(conf)
            self.detection["conf"] = float(conf)
        self._update_tooltip()
        
    def set_editor_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        if mode != EditorMode.EDIT_BOXES:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    _MIN_HANDLE_SIZE = 2.0
    _MAX_HANDLE_SIZE = 12.0
    _HANDLE_SIZE_SCALE_FACTOR = 0.06
    _EDGE_TOLERANCE_HANDLE_MULTIPLIER = 1.25
    _MAX_EDGE_TOLERANCE = 10.0

    def _handle_size(self) -> float:
        rect = self.rect()
        clamped_smallest_dimension = max(1.0, min(rect.width(), rect.height()))
        # Scale handles with box size so tiny boxes get tiny handles too.
        return max(
            self._MIN_HANDLE_SIZE,
            min(clamped_smallest_dimension * self._HANDLE_SIZE_SCALE_FACTOR, self._MAX_HANDLE_SIZE),
        )

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
        self.setZValue(2000)  # stay above slice lines for hit-testing

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self._mode = EditorMode.VIEW
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        # Pre-drag position snapshot, updated at mouse-press.
        # Uses scene coordinates to decouple from parent rebase shifts.
        self._drag_start_scene_pos: QPointF = QPointF()

    def set_editor_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        can_move = (mode == EditorMode.EDIT_KEYPOINTS)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, can_move)
        logger.debug("EditablePointItem mode=%s idx=%s movable=%s", mode, self.idx, can_move)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Snapshot position before Qt moves us — ensures undo command has a valid old_pos."""
        logger.debug("EditablePointItem press idx=%s mode=%s pos=%s", self.idx, self._mode, self.pos())
        
        # Bug 2 Fix: Clear selection to avoid unintended simultaneous multi-drag
        if event.modifiers() == Qt.KeyboardModifier.NoModifier:
            scene = self.scene()
            if scene:
                scene.clearSelection()
                
        self.setSelected(True)
        self._drag_start_scene_pos = QPointF(self.scenePos())
        parent = self.parentItem()
        if parent is not None and hasattr(parent, "begin_keypoint_drag"):
            parent.begin_keypoint_drag(self)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        scene_pos = self.scenePos()
        if scene_pos != self._drag_start_scene_pos:
            p = self.parentItem()
            if hasattr(p, "point_moved_commit"):
                p.point_moved_commit(self, self._drag_start_scene_pos, scene_pos, source="user")
                
        parent = self.parentItem()
        if parent is not None and hasattr(parent, "end_keypoint_drag"):
            parent.end_keypoint_drag(self)
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            p = self.parentItem()
            if hasattr(p, "_in_keypoint_update") and getattr(p, "_in_keypoint_update"):
                return super().itemChange(change, value)
            if hasattr(p, "_syncing_bbox") and getattr(p, "_syncing_bbox"):
                return super().itemChange(change, value)
            if hasattr(p, "_updating_lines") and getattr(p, "_updating_lines"):
                return super().itemChange(change, value)
            if hasattr(p, "point_moving"):
                # Update visual lines only; do not push undo commands during drag
                p.point_moving(self)
        return super().itemChange(change, value)

class PieSliceGroup(QGraphicsRectItem):
    """
    A group containing EditablePointItems and visual lines for a pie slice.
    Automatically updates its bounding box (`xyxy`) when keypoints move.
    """
    KEYPOINT_COUNT = 5
    def __init__(
        self,
        class_name: str,
        detection: Dict[str, Any],
        colors: Dict[str, Tuple[int, int, int]] | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        x1, y1, x2, y2 = detection.get("xyxy", [0, 0, 0, 0])
        w = max(1.0, float(x2) - float(x1))
        h = max(1.0, float(y2) - float(y1))
        super().__init__(QRectF(0.0, 0.0, w, h), parent)
        self.setPos(float(x1), float(y1))
        self.class_name = class_name
        self.detection = detection
        self._base_z = 0.0
        
        self.setFiltersChildEvents(False)
        # Let child keypoints handle mouse events without the container intercepting.
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        
        # Colors
        palette = colors or DEFAULT_COLORS.get(class_name, DEFAULT_COLORS["slice"])
        r, g, b = palette["normal"]
        self._normal_color = QColor(r, g, b)
        rh, gh, bh = palette["highlight"]
        self._highlight_color = QColor(rh, gh, bh)

        self._pen = QPen(self._normal_color)
        self._pen.setWidth(_NORMAL_PEN_WIDTH)
        self._pen.setCosmetic(True)

        self._is_hovered = False
        self._is_selected = False
        self._keypoint_color_normal = QColor(255, 255, 0)
        self._keypoint_color_hover = QColor(255, 255, 255)
        self._keypoint_color_selected = QColor(59, 130, 246)

        self.point_items: List[EditablePointItem] = []
        self._lines: List[QGraphicsLineItem] = []
        self._mode = EditorMode.VIEW
        self._syncing_bbox = False
        self._updating_lines = False
        self._in_keypoint_update = False
        self._active_drag_point: Optional[EditablePointItem] = None
        
        # Initialize
        self._init_keypoints()
        self._apply_visual_state()
        
    def set_editor_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        for p in self.point_items:
            p.set_editor_mode(mode)
        logger.debug("PieSliceGroup mode=%s points=%d", mode, len(self.point_items))

    def _apply_visual_state(self) -> None:
        highlighted = self._is_hovered or self._is_selected
        pen_color = self._highlight_color if highlighted else self._normal_color
        pen_width = _HIGHLIGHT_PEN_WIDTH if highlighted else _NORMAL_PEN_WIDTH
        self._pen.setColor(pen_color)
        self._pen.setWidth(pen_width)
        if highlighted:
            self.setZValue(self._base_z + _HIGHLIGHT_Z_BOOST)
        else:
            self.setZValue(self._base_z)

        for line in self._lines:
            line.setPen(self._pen)

        if self._is_selected:
            keypoint_color = self._keypoint_color_selected
        elif self._is_hovered:
            keypoint_color = self._keypoint_color_hover
        else:
            keypoint_color = self._keypoint_color_normal

        for pt in self.point_items:
            pt.setBrush(QBrush(keypoint_color))

    def apply_visual_state(self) -> None:
        self._apply_visual_state()

    def set_selected_state(self, selected: bool) -> None:
        self._is_selected = selected
        self._apply_visual_state()
            
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
        for i in range(min(self.KEYPOINT_COUNT, len(kps_arr))):
            x, y = float(kps_arr[i, 0]), float(kps_arr[i, 1])
            if x <= 0 and y <= 0 and i > 0:
                continue
            local = self.mapFromScene(QPointF(x, y))
            pt = EditablePointItem(local.x(), local.y(), i, self)
            self.point_items.append(pt)
            
        # Draw lines from center to boundary points
        self._update_lines()

    def _center_point(self) -> Optional[EditablePointItem]:
        for pt in self.point_items:
            if getattr(pt, "idx", None) == 0:
                return pt
        return self.point_items[0] if self.point_items else None

    def next_keypoint_index(self) -> Optional[int]:
        used = {pt.idx for pt in self.point_items if hasattr(pt, "idx")}
        for idx in range(self.KEYPOINT_COUNT):
            if idx not in used:
                return idx
        return None

    def missing_keypoint_indices(self) -> List[int]:
        used = {pt.idx for pt in self.point_items if hasattr(pt, "idx")}
        return [idx for idx in range(self.KEYPOINT_COUNT) if idx not in used]

    def is_complete(self) -> bool:
        return len(self.missing_keypoint_indices()) == 0

    def begin_keypoint_drag(self, point_item: EditablePointItem) -> None:
        self._active_drag_point = point_item

    def end_keypoint_drag(self, point_item: EditablePointItem) -> None:
        if self._active_drag_point is point_item:
            self._active_drag_point = None
            # Rebase after the drag ends to keep local coords stable.
            self.refresh_keypoint_geometry(rebase=True)

    def _begin_keypoint_update(self) -> bool:
        if self._in_keypoint_update:
            return False
        self._in_keypoint_update = True
        return True

    def _end_keypoint_update(self) -> None:
        self._in_keypoint_update = False

    def _refresh_keypoint_geometry(self, rebase: bool = True) -> None:
        self._update_lines()
        self.sync_bbox_to_points(rebase=rebase)

    def refresh_keypoint_geometry(self, rebase: bool = True) -> None:
        if not self._begin_keypoint_update():
            return
        try:
            self._refresh_keypoint_geometry(rebase=rebase)
        finally:
            self._end_keypoint_update()

    def _emit_keypoint_moved(
        self,
        point_item: EditablePointItem,
        old_pos: QPointF,
        new_pos: QPointF,
        source: str,
    ) -> None:
        scene = self.scene()
        if scene is not None and hasattr(scene, "keypoint_moved"):
            scene.keypoint_moved.emit(point_item, old_pos, new_pos, source)

    def apply_keypoint_move(
        self,
        point_item: EditablePointItem,
        old_scene_pos: QPointF,
        new_scene_pos: QPointF,
        source: str = "command",
    ) -> None:
        if point_item is None:
            return
        if not self._begin_keypoint_update():
            return
        try:
            new_local = self.mapFromScene(new_scene_pos)
            point_item.setPos(new_local)
            self._refresh_keypoint_geometry(rebase=True)
        finally:
            self._end_keypoint_update()
        self._emit_keypoint_moved(point_item, QPointF(old_scene_pos), QPointF(new_scene_pos), source)

    def sync_bbox_to_points(self, rebase: bool = True) -> None:
        if self._syncing_bbox:
            return
        if not self.point_items:
            return
        self._syncing_bbox = True
        try:
            x1, y1, x2, y2 = self.current_xyxy()
            if rebase:
                new_pos = QPointF(x1, y1)
                if new_pos != self.pos():
                    for pt in self.point_items:
                        scene_pos = pt.scenePos()
                        new_local = scene_pos - new_pos
                        if new_local != pt.pos():
                            pt.setPos(new_local)
                    self.setPos(new_pos)
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                self.setRect(QRectF(0.0, 0.0, w, h))
            else:
                local_pts = [pt.pos() for pt in self.point_items]
                min_x = min(p.x() for p in local_pts)
                min_y = min(p.y() for p in local_pts)
                max_x = max(p.x() for p in local_pts)
                max_y = max(p.y() for p in local_pts)
                w = max(1.0, max_x - min_x)
                h = max(1.0, max_y - min_y)
                self.setRect(QRectF(min_x, min_y, w, h))
            self.detection["xyxy"] = [x1, y1, x2, y2]
        finally:
            self._syncing_bbox = False
        
    def _update_lines(self) -> None:
        if self._updating_lines:
            return
        self._updating_lines = True
        try:
            # Clear old lines
            for line in self._lines:
                if line.scene():
                    line.scene().removeItem(line)
            self._lines.clear()
            
            center_item = self._center_point()
            if center_item is None:
                return

            # Order points by keypoint index so the path follows
            # center -> edge1 -> edge2 -> edge3 -> edge4 -> center.
            by_idx = {getattr(pt, "idx", None): pt for pt in self.point_items}
            center = by_idx.get(0, center_item)
            boundary_pts = [by_idx.get(i) for i in range(1, self.KEYPOINT_COUNT) if by_idx.get(i)]
            if not boundary_pts:
                return

            sequence = [center] + boundary_pts
            for a, b in zip(sequence, sequence[1:]):
                p1 = a.pos()
                p2 = b.pos()
                line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y(), self)
                line.setPen(self._pen)
                line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                line.setAcceptHoverEvents(False)
                self._lines.append(line)

            # Close the path back to center
            last = boundary_pts[-1]
            p_last = last.pos()
            p_center = center.pos()
            line = QGraphicsLineItem(p_last.x(), p_last.y(), p_center.x(), p_center.y(), self)
            line.setPen(self._pen)
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            line.setAcceptHoverEvents(False)
            self._lines.append(line)
        finally:
            self._updating_lines = False
            
    def point_moving(self, point_item: EditablePointItem) -> None:
        """Called continuously during drag to update visual lines."""
        if self._in_keypoint_update:
            return
        if not self._begin_keypoint_update():
            return
        try:
            self._refresh_keypoint_geometry(rebase=False)
        finally:
            self._end_keypoint_update()

    def point_moved_commit(
        self,
        point_item: EditablePointItem,
        old_scene_pos: QPointF,
        new_scene_pos: QPointF,
        source: str = "user",
    ) -> None:
        """Called on mouse release to commit the final position to undo stack."""
        if self._in_keypoint_update:
            return
        if not self._begin_keypoint_update():
            return
        try:
            self._refresh_keypoint_geometry(rebase=True)
        finally:
            self._end_keypoint_update()
        self._emit_keypoint_moved(point_item, old_scene_pos, new_scene_pos, source)
        
    def set_highlighted(self, highlighted: bool) -> None:
        self._is_hovered = highlighted
        self._apply_visual_state()
            
    def current_xyxy(self) -> List[float]:
        # Compute bounding box of all points
        if not self.point_items:
            return self.detection.get("xyxy", [0, 0, 0, 0])
        import numpy as np
        pts = np.array([[p.scenePos().x(), p.scenePos().y()] for p in self.point_items])
        return [float(pts[:, 0].min()), float(pts[:, 1].min()), 
                float(pts[:, 0].max()), float(pts[:, 1].max())]
                
    def export_keypoints(self) -> List[List[float]]:
        # Restore into original shape [x, y, conf(if original had it)]
        orig_kps = self.detection.get("keypoints")
        if orig_kps is None:
            orig_kps = []
            
        import numpy as np
        orig_arr = np.asarray(orig_kps) if len(orig_kps) else np.empty((0, 0))
        is_flat = (orig_arr.ndim == 1 and orig_arr.size > 0)
        if is_flat:
            if orig_arr.size % 3 == 0 and orig_arr.size >= 3:
                orig_arr = orig_arr.reshape(-1, 3)
            elif orig_arr.size % 2 == 0 and orig_arr.size >= 2:
                orig_arr = orig_arr.reshape(-1, 2)
            else:
                orig_arr = np.empty((0, 0))

        cols = orig_arr.shape[1] if orig_arr.ndim == 2 else 0
        if cols not in (2, 3):
            cols = 2

        new_list = orig_arr.tolist() if orig_arr.size else []
        for pt in self.point_items:
            idx = getattr(pt, "idx", None)
            if idx is None or idx < 0:
                continue
            while len(new_list) <= idx:
                if cols == 3:
                    new_list.append([0.0, 0.0, 1.0])
                else:
                    new_list.append([0.0, 0.0])
            scene_pos = pt.scenePos()
            new_list[idx][0] = float(scene_pos.x())
            new_list[idx][1] = float(scene_pos.y())

        if is_flat:
            return [value for row in new_list for value in row]
        return new_list

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
    keypoint_moved = pyqtSignal(object, object, object, str)  # point_item, old_pos, new_pos, source
    keypoint_created = pyqtSignal(object, object)        # group, point_item
    keypoint_deleted = pyqtSignal(object, object)        # group, point_item

    # Phase 6 Slice creation:
    slice_created = pyqtSignal(object)                   # group

    # Phase 7 Selection sync:
    box_selected = pyqtSignal(str)  # class_name of the selected box

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._rect_items: List[EditableRectItem] = []
        self._baseline_item: Optional['BaselineItem'] = None
        self._grid_items: List[QGraphicsLineItem] = []
        self._calibration: dict = {}
        self._visible_classes: Dict[str, bool] = {}
        self._colors: Dict[str, Dict[str, Tuple[int, int, int]]] = dict(DEFAULT_COLORS)
        self._image_width = 0
        self._image_height = 0
        self._highlighted_item: Optional[EditableRectItem] = None
        self._selected_slice_group: Optional[PieSliceGroup] = None
        self._mode = EditorMode.VIEW

        # Emit box_selected when the user selects a single box in Edit mode
        self.selectionChanged.connect(self._on_selection_changed)

    def _next_z_value(self) -> float:
        if not self._rect_items:
            return 1.0
        return max(getattr(item, "_base_z", 1.0) for item in self._rect_items) + 0.01

    def create_slice_group(self, xyxy: Optional[List[float]] = None) -> PieSliceGroup:
        if not xyxy:
            xyxy = [0.0, 0.0, 1.0, 1.0]
        det = {"xyxy": list(xyxy), "conf": 1.0, "text": "", "keypoints": []}
        palette = self._colors.get("slice", self._colors.get("other"))
        group = PieSliceGroup(class_name="slice", detection=det, colors=palette)
        group._base_z = self._next_z_value()
        group.setZValue(group._base_z)
        visible = self._visible_classes.get("slice", True)
        group.setVisible(visible)
        self.addItem(group)
        self._rect_items.append(group)
        return group

    def get_incomplete_slices(self) -> List[PieSliceGroup]:
        incomplete: List[PieSliceGroup] = []
        for item in self._rect_items:
            if isinstance(item, PieSliceGroup) and not item.is_complete():
                incomplete.append(item)
        return incomplete
        
    def set_editor_mode(self, mode: EditorMode) -> None:
        prev_mode = self._mode
        self._mode = mode
        for item in self._rect_items:
            item.set_editor_mode(mode)
        keypoint_modes = {EditorMode.EDIT_KEYPOINTS, EditorMode.CREATE_KEYPOINT, EditorMode.CREATE_SLICE}
        if prev_mode in keypoint_modes and mode not in keypoint_modes:
            self.clear_selected_slice_group()
        pie_groups = sum(1 for item in self._rect_items if isinstance(item, PieSliceGroup))
        logger.debug(
            "DetectionScene mode=%s items=%d pie_groups=%d",
            mode,
            len(self._rect_items),
            pie_groups,
        )

    # ── Selection sync ──

    def set_selected_slice_group(self, group: Optional[PieSliceGroup]) -> None:
        if group is self._selected_slice_group:
            return
        if self._selected_slice_group is not None:
            self._selected_slice_group.set_selected_state(False)
        self._selected_slice_group = group
        if self._selected_slice_group is not None:
            self._selected_slice_group.set_selected_state(True)

    def clear_selected_slice_group(self) -> None:
        self.set_selected_slice_group(None)

    def _on_selection_changed(self) -> None:
        """Sync selection state for boxes and pie keypoints."""
        selected = self.selectedItems()
        if self._mode == EditorMode.EDIT_BOXES:
            if len(selected) == 1 and isinstance(selected[0], EditableRectItem):
                self.box_selected.emit(selected[0].class_name)
            return

        if self._mode in (EditorMode.EDIT_KEYPOINTS, EditorMode.CREATE_KEYPOINT, EditorMode.CREATE_SLICE):
            points = [item for item in selected if isinstance(item, EditablePointItem)]
            if len(points) == 1:
                group = points[0].parentItem()
                if isinstance(group, PieSliceGroup):
                    self.set_selected_slice_group(group)
                    return
            self.clear_selected_slice_group()

    # ── Image ──

    def load_image(self, pixmap: QPixmap) -> None:
        """Set the base chart image. Clears any previous content."""
        self.clear()
        self._rect_items.clear()
        self._baseline_item = None
        self._grid_items.clear()
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
        if self._selected_slice_group is not None:
            self._selected_slice_group.set_selected_state(False)
        self._selected_slice_group = None
        self._raw_detections = dict(detections) if detections else {}

        if colors:
            self._colors.update(colors)

        z = 1.0
        for class_name, items in detections.items():
            if not isinstance(items, list) or class_name == "layout_text_regions":
                continue

            for det in items:
                bbox = det.get("xyxy")
                if not bbox:
                    continue  # Skip non-bbox detections

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

    def set_grid_lines(
        self,
        row_centers: Optional[List[float]],
        col_centers: Optional[List[float]],
        cell_detections: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Add horizontal and vertical grid lines representing the reconstructed heatmap grid."""
        # 1. Clear old grid items
        for item in self._grid_items:
            self.removeItem(item)
        self._grid_items.clear()

        if not row_centers or not col_centers or not cell_detections:
            return

        # 2. Determine the bounding box of the grid by taking the union of all cell bboxes
        x1s, y1s, x2s, y2s = [], [], [], []
        for det in cell_detections:
            bbox = det.get("xyxy")
            if bbox and len(bbox) == 4:
                x1s.append(bbox[0])
                y1s.append(bbox[1])
                x2s.append(bbox[2])
                y2s.append(bbox[3])

        if not x1s:
            return

        grid_x1 = min(x1s)
        grid_y1 = min(y1s)
        grid_x2 = max(x2s)
        grid_y2 = max(y2s)

        # 3. Create dashed line items
        grid_color = QColor(74, 144, 226, 180)  # Nice soft semi-transparent blue
        pen = QPen(grid_color)
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)

        # Draw vertical lines at col_centers
        for cx in col_centers:
            if grid_x1 <= cx <= grid_x2:
                line_item = QGraphicsLineItem(cx, grid_y1, cx, grid_y2)
                line_item.setPen(pen)
                line_item.setZValue(5)  # below texts/classes, above background image
                self.addItem(line_item)
                self._grid_items.append(line_item)

        # Draw horizontal lines at row_centers
        for cy in row_centers:
            if grid_y1 <= cy <= grid_y2:
                line_item = QGraphicsLineItem(grid_x1, cy, grid_x2, cy)
                line_item.setPen(pen)
                line_item.setZValue(5)
                self.addItem(line_item)
                self._grid_items.append(line_item)

        # Apply visibility
        visible = self._visible_classes.get("grid_lines", True)
        self.set_grid_visible(visible)

    def set_grid_visible(self, visible: bool) -> None:
        """Toggle visibility of grid line items."""
        self._visible_classes["grid_lines"] = visible
        for item in self._grid_items:
            item.setVisible(visible)

    # ── Visibility ──

    def set_class_visible(self, class_name: str, visible: bool) -> None:
        """Show or hide all items of a given class."""
        self._visible_classes[class_name] = visible

        if class_name == "baseline":
            if self._baseline_item:
                self._baseline_item.setVisible(visible)
            return

        if class_name == "grid_lines":
            self.set_grid_visible(visible)
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

    # Status messages for the host window
    status_message = pyqtSignal(str, int)

    # Request to sync toolbar mode when canvas exits a mode (e.g., Esc)
    mode_exit_requested = pyqtSignal(object)

    def __init__(self, scene: DetectionScene, parent=None):
        self._rubber_band = None
        self._rubber_band_origin = QPoint()
        self._create_class_name = "box"  # Default
        self._create_slice_group: Optional[PieSliceGroup] = None
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
        logger.debug("DetectionCanvasView mode=%s", mode)
        if mode != EditorMode.CREATE_SLICE:
            self._reset_create_slice_state()
        if mode == EditorMode.VIEW:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        elif mode == EditorMode.EDIT_BOXES:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        elif mode == EditorMode.CREATE_BOX:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

        if mode == EditorMode.CREATE_SLICE:
            self.status_message.emit(
                "Create Slice: click 5 points in order (center, arc start, arc mid 1, arc mid 2, arc end).",
                6000,
            )

    def _reset_create_slice_state(self) -> None:
        self._create_slice_group = None

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

    def _handle_view_press(self, event) -> bool:
        scene_pos = self.mapToScene(event.position().toPoint())
        for item in self.scene().items(scene_pos):
            if isinstance(item, EditableRectItem) and item.isVisible():
                bbox = item.detection.get("xyxy")
                if bbox:
                    self._det_scene.highlight_item_by_bbox(bbox, item.class_name)
                    self.bbox_clicked.emit(item.class_name, list(bbox))
                    event.accept()
                    return True
        self._det_scene.clear_highlight()
        return False

    def _handle_create_box_press(self, event) -> bool:
        self._rubber_band_origin = event.position().toPoint()
        if not self._rubber_band:
            self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._rubber_band.setGeometry(QRect(self._rubber_band_origin, QSize(0, 0)))
        self._rubber_band.show()
        event.accept()
        return True

    def _handle_edit_keypoints_press(self, event) -> bool:
        scene_pos = self.mapToScene(event.position().toPoint())
        hit_keypoint = any(
            isinstance(item, EditablePointItem)
            for item in self.scene().items(scene_pos)
        )
        if not hit_keypoint:
            self._det_scene.clearSelection()
            if hasattr(self._det_scene, "clear_selected_slice_group"):
                self._det_scene.clear_selected_slice_group()
            event.accept()
            return True
        return False

    def _handle_create_keypoint_press(self, event) -> bool:
        scene_pos = self.mapToScene(event.position().toPoint())
        logger.debug("CREATE_KEYPOINT click at=%s", scene_pos)

        target_group = None
        for item in self._det_scene._rect_items:
            if not isinstance(item, PieSliceGroup) or not item.isVisible():
                continue
            xyxy = item.detection.get("xyxy")
            if xyxy and len(xyxy) == 4:
                x1, y1, x2, y2 = xyxy
                hit_rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            else:
                hit_rect = item.sceneBoundingRect()
            if hit_rect.contains(scene_pos):
                target_group = item
                break

        if target_group is not None:
            if target_group.is_complete():
                self.status_message.emit("Slice already has 5 keypoints.", 4000)
                event.accept()
                return True
            next_idx = target_group.next_keypoint_index()
            if next_idx is not None:
                local_pos = target_group.mapFromScene(scene_pos)
                pt = EditablePointItem(local_pos.x(), local_pos.y(), next_idx, target_group)
                pt.set_editor_mode(EditorMode.EDIT_KEYPOINTS)
                target_group.point_items.append(pt)
                target_group.refresh_keypoint_geometry()
                target_group.apply_visual_state()

                scene = self.scene()
                if hasattr(scene, 'keypoint_created'):
                    scene.keypoint_created.emit(target_group, pt)
                event.accept()
                return True
        return False

    def keyPressEvent(self, event) -> None:
        """Delete key removes selected items in EDIT_BOXES mode.
           Ctrl+Z / Ctrl+Y trigger undo/redo via the scene signal.
        """
        from PyQt6.QtCore import Qt
        if self._mode == EditorMode.CREATE_SLICE and event.key() == Qt.Key.Key_Escape:
            self._reset_create_slice_state()
            self.set_mode(EditorMode.VIEW)
            self.mode_exit_requested.emit(EditorMode.VIEW)
            self.status_message.emit("Exited Create Slice mode.", 3000)
            event.accept()
            return
        if self._mode == EditorMode.EDIT_BOXES:
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                for item in list(self._det_scene._rect_items):
                    if item.isSelected() and item.isVisible():
                        self._det_scene.item_deleted.emit(item)
                event.accept()
                return
        if self._mode == EditorMode.EDIT_KEYPOINTS:
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                removed = 0
                for item in list(self._det_scene.selectedItems()):
                    if isinstance(item, EditablePointItem):
                        group = item.parentItem()
                        if isinstance(group, PieSliceGroup):
                            self._det_scene.keypoint_deleted.emit(group, item)
                            removed += 1
                if removed:
                    self.status_message.emit(
                        "Keypoint deleted. Complete the slice before re-extract.",
                        5000,
                    )
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
                if self._handle_view_press(event):
                    return
            elif self._mode == EditorMode.CREATE_BOX:
                if self._handle_create_box_press(event):
                    return
            elif self._mode == EditorMode.EDIT_KEYPOINTS:
                if self._handle_edit_keypoints_press(event):
                    return
            elif self._mode == EditorMode.CREATE_KEYPOINT:
                if self._handle_create_keypoint_press(event):
                    return
            elif self._mode == EditorMode.CREATE_SLICE:
                scene_pos = self.mapToScene(event.position().toPoint())
                logger.debug("CREATE_SLICE click at=%s", scene_pos)

                if self._create_slice_group is None or self._create_slice_group.is_complete():
                    seed = [scene_pos.x(), scene_pos.y(), scene_pos.x() + 1.0, scene_pos.y() + 1.0]
                    self._create_slice_group = self._det_scene.create_slice_group(seed)
                    if hasattr(self._det_scene, "slice_created"):
                        self._det_scene.slice_created.emit(self._create_slice_group)

                target_group = self._create_slice_group
                next_idx = target_group.next_keypoint_index() if target_group else None
                if target_group is not None and next_idx is not None:
                    local_pos = target_group.mapFromScene(scene_pos)
                    pt = EditablePointItem(local_pos.x(), local_pos.y(), next_idx, target_group)
                    pt.set_editor_mode(EditorMode.CREATE_SLICE)
                    target_group.point_items.append(pt)
                    target_group.refresh_keypoint_geometry()
                    target_group.apply_visual_state()

                    if hasattr(self._det_scene, "keypoint_created"):
                        self._det_scene.keypoint_created.emit(target_group, pt)

                    if target_group.is_complete():
                        self.status_message.emit(
                            "Slice complete. Click to start another slice or press Esc to exit.",
                            6000,
                        )
                        self._create_slice_group = None
                else:
                    self.status_message.emit("Slice already has 5 keypoints.", 4000)
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
