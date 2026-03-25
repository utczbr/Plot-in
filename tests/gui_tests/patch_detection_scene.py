import re
from pathlib import Path

file_path = Path("/home/stuart/Documentos/OCR/LYAA-fine-tuning/src/visual/detection_scene.py")
content = file_path.read_text()

# 1. Imports
imports_search = """from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)"""

imports_replace = """from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF, QPoint
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
    QRubberBand,
)
from PyQt6.QtWidgets import QGraphicsSceneMouseEvent, QGraphicsSceneHoverEvent
"""

content = content.replace(imports_search, imports_replace)

# 2. EditableRectItem
rect_item_search = """        # Start non-interactive (VIEW mode)
        self.setAcceptHoverEvents(True)"""

rect_item_replace = """        # Start non-interactive (VIEW mode)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        
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

    _EDGE_TOLERANCE = 6.0

    def _get_resize_edges(self, pos: QPointF) -> str:
        if self._mode != EditorMode.EDIT_BOXES:
            return ""
        rect = self.rect()
        edges = ""
        if abs(pos.y() - rect.top()) <= self._EDGE_TOLERANCE:
            edges += "top"
        elif abs(pos.y() - rect.bottom()) <= self._EDGE_TOLERANCE:
            edges += "bottom"
        if abs(pos.x() - rect.left()) <= self._EDGE_TOLERANCE:
            edges += "left"
        elif abs(pos.x() - rect.right()) <= self._EDGE_TOLERANCE:
            edges += "right"
        return edges

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
                
            self.setRect(new_rect)
            event.accept()
            return
            
        elif self._is_dragging:
            delta = event.scenePos() - self._drag_start_pos
            self.setPos(self._drag_start_item_pos + delta)
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
        event.accept()"""

content = content.replace(rect_item_search, rect_item_replace)

# 3. DetectionScene
scene_search = """    item_highlighted = pyqtSignal(str, list)"""

scene_replace = """    item_highlighted = pyqtSignal(str, list)
    
    # Phase 3 Editor signals:
    item_edited = pyqtSignal(object, list, bool)  # item, old_xyxy, is_resize
    item_deleted = pyqtSignal(object)             # item
    item_class_changed = pyqtSignal(object, str)  # item, new_class_name
    item_created = pyqtSignal(str, list)          # class_name, xyxy"""

content = content.replace(scene_search, scene_replace)

scene_mode_search = """        self._highlighted_item: Optional[EditableRectItem] = None"""
scene_mode_replace = """        self._highlighted_item: Optional[EditableRectItem] = None
        self._mode = EditorMode.VIEW
        
    def set_editor_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        for item in self._rect_items:
            item.set_editor_mode(mode)"""
            
content = content.replace(scene_mode_search, scene_mode_replace)

# 4. DetectionCanvasView
view_search = """    bbox_clicked = pyqtSignal(str, list)  # class_name, xyxy

    def __init__(self, scene: DetectionScene, parent=None):"""

view_replace = """    bbox_clicked = pyqtSignal(str, list)  # class_name, xyxy

    def __init__(self, scene: DetectionScene, parent=None):
        self._rubber_band = None
        self._rubber_band_origin = QPoint()
        self._create_class_name = "box"  # Default"""

content = content.replace(view_search, view_replace)

view_mode_search = """    def set_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        if mode == EditorMode.VIEW:"""

view_mode_replace = """    def set_create_class(self, class_name: str) -> None:
        self._create_class_name = class_name

    def set_mode(self, mode: EditorMode) -> None:
        self._mode = mode
        self._det_scene.set_editor_mode(mode)
        if mode == EditorMode.VIEW:"""

content = content.replace(view_mode_search, view_mode_replace)


view_events_search = """    def mousePressEvent(self, event) -> None:
        \"\"\"Mode-dependent left-click handling.\"\"\"
        if event.button() == Qt.MouseButton.MiddleButton:
            # Middle-click always pans
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Simulate left press for the drag mode
            fake = type(event)(
                event.type(),
                event.position(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            super().mousePressEvent(fake)
            return

        if event.button() == Qt.MouseButton.LeftButton and self._mode == EditorMode.VIEW:
            # Click-to-highlight: find item under mouse
            scene_pos = self.mapToScene(event.position().toPoint())
            items = self.scene().items(scene_pos)
            for item in items:
                if isinstance(item, EditableRectItem) and item.isVisible():
                    bbox = item.detection.get("xyxy")
                    if bbox:
                        self._det_scene.highlight_item_by_bbox(bbox, item.class_name)
                        self.bbox_clicked.emit(item.class_name, list(bbox))
                        event.accept()
                        return
            # Click on empty space — clear highlight
            self._det_scene.clear_highlight()

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            # Restore drag mode after middle-click pan
            if self._mode == EditorMode.VIEW:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            else:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)"""


view_events_replace = """    def mousePressEvent(self, event) -> None:
        \"\"\"Mode-dependent left-click handling.\"\"\"
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
                self._rubber_band.setGeometry(QRectF(self._rubber_band_origin, dict(width=0, height=0)).toRect())
                self._rubber_band.show()
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._mode == EditorMode.CREATE_BOX and self._rubber_band and self._rubber_band.isVisible():
            rect = QRectF(self._rubber_band_origin, event.position().toPoint()).normalized().toRect()
            self._rubber_band.setGeometry(rect)
            event.accept()
            return
            
        super().mouseMoveEvent(event)

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

        super().mouseReleaseEvent(event)"""

content = content.replace(view_events_search, view_events_replace)

file_path.write_text(content)
print("detection_scene.py patched successfully")
