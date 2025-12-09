"""
Hover Event Filter - Reusable Event Filter Replacing Lambda Overwrites

This module provides event filters to replace lambda event handlers that create
memory leaks (e.g., enterEvent/leaveEvent lambdas in main_modern.py).

Author: Chart Analysis Tool Team
Date: November 24, 2025
"""

from PyQt6.QtCore import QObject, QEvent, pyqtSignal
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HoverEventFilter(QObject):
    """
    Reusable event filter for hover events, replacing lambda overwrites.
    
    Replaces patterns like:
        label.enterEvent = lambda e: self.on_hover_enter(bbox)  # MEMORY LEAK!
        label.leaveEvent = lambda e: self.on_hover_leave()
    
    With:
        filter = HoverEventFilter(bbox, parent=label)
        label.installEventFilter(filter)
        filter.hover_enter.connect(self.on_hover_enter)
    
    Benefits:
    - No circular references (parent set correctly)
    - Automatic cleanup when parent widget deleted
    - Type-safe signal parameters
    - Reusable across widgets
    
    Signals:
        hover_enter(dict, str): Emitted on mouse enter (bbox, class_name)
        hover_leave(): Emitted on mouse leave
    """
    
    hover_enter = pyqtSignal(dict, str)  # bbox, class_name
    hover_leave = pyqtSignal()
    
    def __init__(self, bbox: Dict[str, Any], class_name: str, parent: Optional[QObject] = None):
        """
        Initialize hover event filter.
        
        Args:
            bbox: Bounding box dictionary
            class_name: Detection class name
            parent: Parent QObject (typically the widget being monitored)
                   CRITICAL: Must be set for automatic cleanup!
        
        Example:
            >>> label = QLabel()
            >>> filter = HoverEventFilter(bbox={'x': 10, 'y': 20}, class_name='bar', parent=label)
            >>> label.installEventFilter(filter)
            >>> filter.hover_enter.connect(lambda bbox, cls: print(f"Hover: {cls}"))
        """
        super().__init__(parent)  # CRITICAL: Set parent for auto-cleanup
        self.bbox = bbox
        self.class_name = class_name
        logger.debug(f"HoverEventFilter created for class '{class_name}'")
    
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Filter events for hover detection.
        
        Args:
            watched: Object being watched
            event: Event to filter
        
        Returns:
            False to allow event propagation (don't block events)
        """
        if event.type() == QEvent.Type.Enter:
            self.hover_enter.emit(self.bbox, self.class_name)
            return False  # Don't block event
        
        elif event.type() == QEvent.Type.Leave:
            self.hover_leave.emit()
            return False
        
        return False  # Don't block any events


class ClickEventFilter(QObject):
    """
    Event filter for click events with custom data.
    
    Signals:
        clicked(dict): Emitted on mouse click with custom data
    """
    
    clicked = pyqtSignal(dict)  # data
    
    def __init__(self, data: Dict[str, Any], parent: Optional[QObject] = None):
        """
        Initialize click event filter.
        
        Args:
            data: Custom data to emit on click
            parent: Parent QObject for auto-cleanup
        """
        super().__init__(parent)
        self.data = data
    
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Filter mouse press events."""
        if event.type() == QEvent.Type.MouseButtonPress:
            self.clicked.emit(self.data)
            return False
        return False


class HoverHighlightFilter(QObject):
    """
    Advanced hover filter with highlight state tracking.
    
    Tracks whether the widget is currently highlighted to prevent
    redundant updates.
    
    Signals:
        request_highlight(str): Emitted when highlight should be shown
        request_unhighlight(): Emitted when highlight should be removed
    """
    
    request_highlight = pyqtSignal(str)  # detection_id
    request_unhighlight = pyqtSignal()
    
    def __init__(self, detection_id: str, parent: Optional[QObject] = None):
        """
        Initialize hover highlight filter.
        
        Args:
            detection_id: Unique detection identifier
            parent: Parent QObject
        """
        super().__init__(parent)
        self.detection_id = detection_id
        self._is_highlighted = False
    
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Filter hover events with state tracking."""
        if event.type() == QEvent.Type.Enter and not self._is_highlighted:
            self._is_highlighted = True
            self.request_highlight.emit(self.detection_id)
            return False
        
        elif event.type() == QEvent.Type.Leave and self._is_highlighted:
            self._is_highlighted = False
            self.request_unhighlight.emit()
            return False
        
        return False
    
    def reset(self) -> None:
        """Reset highlight state."""
        self._is_highlighted = False


def install_hover_filter(widget: QObject, bbox: Dict, class_name: str, 
                        enter_handler: Optional[callable] = None,
                        leave_handler: Optional[callable] = None) -> HoverEventFilter:
    """
    Helper function to install hover filter with automatic connection.
    
    Args:
        widget: Widget to install filter on
        bbox: Bounding box data
        class_name: Detection class name
        enter_handler: Optional callback for hover enter (receives bbox, class_name)
        leave_handler: Optional callback for hover leave
    
    Returns:
        Installed HoverEventFilter instance
    
    Example:
        >>> label = QLabel()
        >>> filter = install_hover_filter(
        ...     label,
        ...     bbox={'x': 10, 'y': 20},
        ...     class_name='bar',
        ...     enter_handler=lambda bbox, cls: print(f"Enter {cls}"),
        ...     leave_handler=lambda: print("Leave")
        ... )
    """
    event_filter = HoverEventFilter(bbox, class_name, parent=widget)
    widget.installEventFilter(event_filter)
    
    if enter_handler:
        event_filter.hover_enter.connect(enter_handler)
    
    if leave_handler:
        event_filter.hover_leave.connect(leave_handler)
    
    return event_filter
