"""
Rendering Utilities - Helper Functions for LayeredRenderer

This module provides helper functions for creating overlay layers,
drawing bounding boxes, annotations, etc.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QRectF


def create_detection_overlay(
    width: int,
    height: int,
    detections: List[Dict],
    class_colors: Optional[Dict[str, QColor]] = None,
    line_width: int = 2
) -> QPixmap:
    """
    Create detection overlay with bounding boxes and labels.
    
    Args:
        width: Image width
        height: Image height
        detections: List of detection dicts with 'bbox', 'class', 'confidence'
        class_colors: Optional color mapping for classes
        line_width: Bounding box line width
    
    Returns:
        Transparent pixmap with drawn detections
    
    Example:
        >>> detections = [
        ...     {'bbox': [10, 20, 100, 50], 'class': 'bar', 'confidence': 0.95}
        ... ]
        >>> overlay = create_detection_overlay(640, 480, detections)
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    
    if not detections:
        return pixmap
    
    # Default colors
    if class_colors is None:
        class_colors = {
            'bar': QColor(0, 255, 0, 200),      # Green
            'line': QColor(255, 0, 0, 200),     # Red
            'pie': QColor(0, 0, 255, 200),      # Blue
            'scatter': QColor(255, 255, 0, 200), # Yellow
            'axis': QColor(255, 165, 0, 200),   # Orange
            'legend': QColor(255, 0, 255, 200), # Magenta
            'title': QColor(0, 255, 255, 200),  # Cyan
        }
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    for detection in detections:
        bbox = detection.get('bbox', [])
        if len(bbox) != 4:
            continue
        
        x, y, w, h = bbox
        class_name = detection.get('class', 'unknown')
        confidence = detection.get('confidence', 0.0)
        
        # Get color
        color = class_colors.get(class_name, QColor(255, 255, 255, 200))
        
        # Draw bounding box
        pen = QPen(color, line_width)
        painter.setPen(pen)
        painter.drawRect(int(x), int(y), int(w), int(h))
        
        # Draw label background
        label = f"{class_name} {confidence:.2f}"
        font = QFont("Arial", 10)
        painter.setFont(font)
        
        label_rect = painter.boundingRect(int(x), int(y - 20), 0, 0, 0, label)
        label_rect.adjust(-2, -2, 2, 2)
        
        painter.fillRect(label_rect, color)
        
        # Draw label text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)
    
    painter.end()
    return pixmap


def create_grid_overlay(
    width: int,
    height: int,
    grid_size: int = 50,
    color: QColor = QColor(128, 128, 128, 100)
) -> QPixmap:
    """
    Create grid overlay.
    
    Args:
        width: Image width
        height: Image height
        grid_size: Grid spacing in pixels
        color: Grid line color
    
    Returns:
        Transparent pixmap with grid
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    
    painter = QPainter(pixmap)
    painter.setPen(QPen(color, 1))
    
    # Vertical lines
    for x in range(0, width, grid_size):
        painter.drawLine(x, 0, x, height)
    
    # Horizontal lines
    for y in range(0, height, grid_size):
        painter.drawLine(0, y, width, y)
    
    painter.end()
    return pixmap


def create_baseline_overlay(
    width: int,
    height: int,
    baseline_points: List[Tuple[float, float]],
    color: QColor = QColor(0, 255, 0, 150),
    line_width: int = 2
) -> QPixmap:
    """
    Create baseline overlay from points.
    
    Args:
        width: Image width
        height: Image height
        baseline_points: List of (x, y) tuples
        color: Line color
        line_width: Line width
    
    Returns:
        Transparent pixmap with baseline
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    
    if not baseline_points or len(baseline_points) < 2:
        return pixmap
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(color, line_width))
    
    # Draw line segments
    for i in range(len(baseline_points) - 1):
        x1, y1 = baseline_points[i]
        x2, y2 = baseline_points[i + 1]
        painter.drawLine(int(x1), int(y1), int(x2), int(y2))
    
    painter.end()
    return pixmap


def create_highlight_overlay(
    width: int,
    height: int,
    bbox: List[float],
    color: QColor = QColor(255, 255, 0, 100),
    line_width: int = 3
) -> QPixmap:
    """
    Create highlight overlay for single bounding box.
    
    Args:
        width: Image width
        height: Image height
        bbox: [x, y, w, h] bounding box
        color: Highlight color
        line_width: Line width
    
    Returns:
        Transparent pixmap with highlight
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    
    if not bbox or len(bbox) != 4:
        return pixmap
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    x, y, w, h = bbox
    
    # Draw thick border
    pen = QPen(color, line_width)
    painter.setPen(pen)
    painter.drawRect(int(x), int(y), int(w), int(h))
    
    # Fill with semi-transparent color
    fill_color = QColor(color)
    fill_color.setAlpha(50)
    painter.fillRect(int(x), int(y), int(w), int(h), fill_color)
    
    painter.end()
    return pixmap


def numpy_array_to_pixmap(array: np.ndarray) -> QPixmap:
    """
    Convert numpy array directly to QPixmap (fast path).
    
    This is a convenience wrapper around the LayeredRenderer method.
    
    Args:
        array: Numpy array (H, W, 3) RGB or (H, W) grayscale
    
    Returns:
        QPixmap
    """
    from ui.rendering.layered_renderer import LayeredRenderer
    
    renderer = LayeredRenderer()
    qimage = renderer._numpy_to_qimage(array)
    return QPixmap.fromImage(qimage)
