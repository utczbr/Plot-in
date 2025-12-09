"""
Enhanced Image Viewer Component - Complete Implementation
Replaces the current QLabel-based image display with professional QGraphicsView
"""

from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                           QGraphicsRectItem, QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QPushButton, QSlider, QComboBox, QCheckBox, QFrame)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer
from PyQt6.QtGui import QWheelEvent, QMouseEvent, QPainter, QPen, QBrush, QColor, QPixmap
import json

class DraggableBoundingBox(QGraphicsRectItem):
    """Interactive draggable bounding box for detected elements"""
    
    def __init__(self, rect, class_name, detection_data, parent=None):
        super().__init__(rect)
        
        self.class_name = class_name
        self.detection_data = detection_data
        self.original_rect = rect
        self.parent_viewer = parent  # Store reference to parent viewer
        
        # Setup appearance and behavior
        self.setup_appearance()
        self.setup_interactivity()
        
    def setup_appearance(self):
        """Style the bounding box based on detection class"""
        color_map = {
            'bar': QColor(0, 120, 255, 120),
            'line': QColor(255, 0, 0, 120), 
            'scatter': QColor(0, 128, 0, 120),
            'box': QColor(128, 0, 128, 120),
            'scale_label': QColor(255, 165, 0, 120),
            'tick_label': QColor(0, 255, 255, 120),
            'chart_title': QColor(50, 50, 220, 120),
            'legend': QColor(210, 105, 30, 120),
            'axis_title': QColor(255, 100, 255, 120),
            'data_label': QColor(200, 200, 100, 120)
        }
        
        color = color_map.get(self.class_name, QColor(128, 128, 128, 120))
        highlight_color = QColor(color.red(), color.green(), color.blue(), 200)
        
        # Normal state
        self.normal_pen = QPen(color, 2)
        self.normal_brush = QBrush(color)
        
        # Hover state  
        self.hover_pen = QPen(highlight_color, 3)
        self.hover_brush = QBrush(highlight_color)
        
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        
    def setup_interactivity(self):
        """Configure interactive behavior"""
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        
        # Tooltip with detection info
        tooltip_text = f"Class: {self.class_name}\n"
        if 'text' in self.detection_data:
            tooltip_text += f"Text: {self.detection_data['text']}\n"
        if 'conf' in self.detection_data:
            tooltip_text += f"Confidence: {self.detection_data['conf']:.3f}"
        self.setToolTip(tooltip_text)
        
    def hoverEnterEvent(self, event):
        """Handle hover enter - highlight the box"""
        self.setPen(self.hover_pen)
        self.setBrush(self.hover_brush)
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        """Handle hover leave - restore normal appearance"""
        self.setPen(self.normal_pen) 
        self.setBrush(self.normal_brush)
        super().hoverLeaveEvent(event)
        
    def itemChange(self, change, value):
        """Handle item changes - update coordinates when moved"""
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionChange:
            # Calculate new bounding box coordinates
            new_pos = value
            rect = self.rect()
            new_xyxy = [
                new_pos.x() + rect.x(),
                new_pos.y() + rect.y(), 
                new_pos.x() + rect.x() + rect.width(),
                new_pos.y() + rect.y() + rect.height()
            ]
            
            # Update detection data
            self.detection_data['xyxy'] = new_xyxy
            
            # Call the parent viewer's method if available
            if self.parent_viewer:
                self.parent_viewer.on_bbox_moved(self.class_name, new_xyxy)
            
        return super().itemChange(change, value)


class ImageViewerWidget(QWidget):
    """Container widget for the advanced image viewer with zoom controls"""
    
    bbox_moved = pyqtSignal(str, list)  # class_name, new_xyxy
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the viewer
        self.viewer = AdvancedImageViewer()
        self.viewer.bbox_moved.connect(self.bbox_moved)
        
        # Create zoom controls
        self.setup_zoom_controls()
        
        # Add to layout
        layout.addWidget(self.zoom_controls_frame)
        layout.addWidget(self.viewer)
        
        # Set initial zoom level
        self.current_zoom = 1.0
        
    def setup_zoom_controls(self):
        """Create zoom controls"""
        self.zoom_controls_frame = QFrame()
        self.zoom_controls_frame.setFixedHeight(40)
        zoom_layout = QHBoxLayout(self.zoom_controls_frame)
        zoom_layout.setContentsMargins(4, 4, 4, 4)
        
        # Zoom in button
        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setFixedWidth(40)
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)
        
        # Current zoom level
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(60)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_layout.addWidget(self.zoom_label)
        
        # Zoom out button
        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setFixedWidth(40)
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)
        
        # Reset zoom
        reset_zoom_btn = QPushButton("🔄")
        reset_zoom_btn.setFixedWidth(40)
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(reset_zoom_btn)
        
        # Fill remaining space
        zoom_layout.addStretch()
        
    def zoom_in(self):
        """Zoom in the viewer"""
        self.viewer.zoom_in()
        self.current_zoom = self.viewer.scale_factor
        self.zoom_label.setText(f"{int(self.current_zoom * 100)}%")
        
    def zoom_out(self):
        """Zoom out the viewer"""
        self.viewer.zoom_out()
        self.current_zoom = self.viewer.scale_factor
        self.zoom_label.setText(f"{int(self.current_zoom * 100)}%")
        
    def reset_zoom(self):
        """Reset zoom to fit view"""
        self.viewer.fitInView(self.viewer.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.current_zoom = self.viewer.scale_factor
        self.zoom_label.setText(f"{int(self.current_zoom * 100)}%")
        
    def load_image(self, image_path_or_pixmap):
        """Load image into the viewer"""
        return self.viewer.load_image(image_path_or_pixmap)
        
    def add_detection_overlays(self, detections_dict):
        """Add detection overlays to the viewer"""
        self.viewer.add_detection_overlays(detections_dict)
        
    def clear_overlays(self):
        """Clear all detection overlays"""
        self.viewer.clear_overlays()


class AdvancedImageViewer(QGraphicsView):
    """Professional image viewer with pan, zoom, and interactive overlays"""
    
    coordinate_changed = pyqtSignal(float, float)  # x, y in image coordinates
    bbox_moved = pyqtSignal(str, list)  # class_name, new_xyxy
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Configure rendering for performance
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # State tracking
        self.pixmap_item = None
        self.bounding_boxes = {}
        self.original_image_size = None
        self.scale_factor = 1.0
        self.min_scale = 0.05
        self.max_scale = 20.0
        
        # Pan state
        self.last_pan_point = QPointF()
        self.is_panning = False
        
        # Coordinate tracking
        self.setMouseTracking(True)
        
    def load_image(self, image_path_or_pixmap):
        """Load image and setup coordinate system"""
        self.clear_scene()
        
        if isinstance(image_path_or_pixmap, str):
            pixmap = QPixmap(image_path_or_pixmap)
        else:
            pixmap = image_path_or_pixmap
            
        if pixmap.isNull():
            return False
            
        self.original_image_size = pixmap.size()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        
        # Set scene rect and fit in view
        rect = pixmap.rect()
        self.scene.setSceneRect(rect.x(), rect.y(), rect.width(), rect.height())
        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.scale_factor = min(self.viewport().width() / pixmap.width(), 
                               self.viewport().height() / pixmap.height())
        
        return True
        
    def clear_scene(self):
        """Clear all items from scene"""
        self.scene.clear()
        self.pixmap_item = None
        self.bounding_boxes.clear()
        
    def add_detection_overlays(self, detections_dict):
        """Add interactive bounding boxes for all detections"""
        self.clear_overlays()
        
        for class_name, detection_list in detections_dict.items():
            if not detection_list:
                continue
                
            class_boxes = []
            for i, detection in enumerate(detection_list):
                if 'xyxy' not in detection:
                    continue
                    
                bbox = detection['xyxy']
                rect = QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                
                # Create draggable bounding box with reference to this viewer
                bbox_item = DraggableBoundingBox(rect, class_name, detection, parent=self)
                
                # Then add to scene
                self.scene.addItem(bbox_item)
                class_boxes.append(bbox_item)
                
            self.bounding_boxes[class_name] = class_boxes
            
    def clear_overlays(self):
        """Remove all bounding box overlays"""
        for class_boxes in self.bounding_boxes.values():
            for bbox in class_boxes:
                self.scene.removeItem(bbox)
        self.bounding_boxes.clear()
        
    def on_bbox_moved(self, class_name, new_xyxy):
        """Called by DraggableBoundingBox when it moves"""
        self.bbox_moved.emit(class_name, new_xyxy)
        
    def wheelEvent(self, event):
        """Handle zoom with mouse wheel"""
        modifiers = event.modifiers()
        
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            # Zoom in/out
            zoom_factor = 1.25 if event.angleDelta().y() > 0 else 1/1.25
            
            # Calculate new scale
            new_scale = self.scale_factor * zoom_factor
            if self.min_scale <= new_scale <= self.max_scale:
                self.scale(zoom_factor, zoom_factor)
                self.scale_factor = new_scale
                
                # Calculate position to zoom towards mouse cursor
                mouse_pos = event.position()
                self.centerOn(self.mapToScene(mouse_pos.toPoint()))
        else:
            # Default wheel behavior (scroll)
            super().wheelEvent(event)
            
    def mousePressEvent(self, event):
        """Handle mouse press for panning or item interaction"""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self.is_panning = True
            self.last_pan_point = event.position()
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            # Let QGraphicsView handle selection, etc.
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """Handle mouse movement during panning"""
        if self.is_panning:
            # Continue panning
            offset = self.last_pan_point - event.position()
            self.last_pan_point = event.position()
            
            # Scroll the view
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() + int(offset.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() + int(offset.y())
            )
        else:
            # Pass to parent for normal behavior
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop panning"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            super().mouseReleaseEvent(event)
            
    def zoom_in(self):
        """Zoom in by a fixed factor"""
        zoom_factor = 1.25
        if self.scale_factor * zoom_factor <= self.max_scale:
            self.scale(zoom_factor, zoom_factor)
            self.scale_factor *= zoom_factor
            
    def zoom_out(self):
        """Zoom out by a fixed factor"""
        zoom_factor = 1/1.25
        if self.scale_factor * zoom_factor >= self.min_scale:
            self.scale(zoom_factor, zoom_factor)
            self.scale_factor *= zoom_factor