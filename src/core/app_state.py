"""
Immutable Application State Management

This module provides immutable state management with automatic change tracking,
validation, and undo/redo support. Replaces the 30+ mutable instance variables
in ModernChartAnalysisApp (lines 202-249 of main_modern.py).

Author: Chart Analysis Tool Team
Date: November 24, 2025
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Tuple, FrozenSet, List, Any
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, QRect
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualizationState:
    """
    Immutable visualization settings.
    
    Replaces mutable visibility_checks dict and related flags.
    """
    visible_classes: FrozenSet[str] = field(default_factory=frozenset)
    show_baseline: bool = True
    show_grid: bool = False
    show_annotations: bool = True
    opacity_map: Dict[str, float] = field(default_factory=lambda: {
        'detections': 1.0,
        'baseline': 1.0,
        'annotations': 0.8
    })
    
    def with_class_visibility(self, class_name: str, visible: bool) -> 'VisualizationState':
        """
        Return new state with updated class visibility.
        
        Args:
            class_name: Detection class name (e.g., 'bar', 'line')
            visible: Whether class should be visible
        
        Returns:
            New VisualizationState with updated visibility
        
        Example:
            >>> state = viz_state.with_class_visibility('bar', True)
            >>> assert 'bar' in state.visible_classes
        """
        new_visible = set(self.visible_classes)
        if visible:
            new_visible.add(class_name)
        else:
            new_visible.discard(class_name)
        return replace(self, visible_classes=frozenset(new_visible))
    
    def with_opacity(self, layer: str, opacity: float) -> 'VisualizationState':
        """
        Return new state with updated layer opacity.
        
        Args:
            layer: Layer name (e.g., 'detections', 'baseline')
            opacity: Opacity value (0.0 to 1.0)
        
        Returns:
            New VisualizationState with updated opacity
        """
        opacity = max(0.0, min(1.0, opacity))  # Clamp to valid range
        new_opacity_map = self.opacity_map.copy()
        new_opacity_map[layer] = opacity
        return replace(self, opacity_map=new_opacity_map)


@dataclass(frozen=True)
class CanvasState:
    """
    Immutable canvas view state.
    
    Replaces mutable zoom_level, pan_offset, highlighted_bbox variables.
    """
    zoom_level: float = 1.0
    pan_offset: Tuple[float, float] = (0.0, 0.0)
    highlighted_bbox: Optional[str] = None  # detection ID
    hovered_bbox: Optional[str] = None  # detection ID
    
    def zoom_by(self, factor: float) -> 'CanvasState':
        """
        Return new state with zoom applied.
        
        Args:
            factor: Zoom factor (e.g., 1.2 for 20% zoom in)
        
        Returns:
            New CanvasState with updated zoom, clamped to valid range
        
        Example:
            >>> state = canvas_state.zoom_by(1.2)
            >>> assert 0.1 <= state.zoom_level <= 10.0
        """
        new_zoom = max(0.1, min(10.0, self.zoom_level * factor))
        return replace(self, zoom_level=new_zoom)
    
    def zoom_to(self, level: float) -> 'CanvasState':
        """
        Return new state with absolute zoom level.
        
        Args:
            level: Absolute zoom level (0.1 to 10.0)
        
        Returns:
            New CanvasState with updated zoom
        """
        new_zoom = max(0.1, min(10.0, level))
        return replace(self, zoom_level=new_zoom)
    
    def pan(self, dx: float, dy: float) -> 'CanvasState':
        """
        Return new state with pan applied.
        
        Args:
            dx: Delta X in pixels
            dy: Delta Y in pixels
        
        Returns:
            New CanvasState with updated pan offset
        """
        new_offset = (self.pan_offset[0] + dx, self.pan_offset[1] + dy)
        return replace(self, pan_offset=new_offset)
    
    def reset_pan(self) -> 'CanvasState':
        """Return new state with pan reset to origin."""
        return replace(self, pan_offset=(0.0, 0.0))


@dataclass(frozen=True)
class AppState:
    """
    Complete immutable application state.
    
    Replaces 30+ mutable instance variables (lines 202-249 of main_modern.py):
    - current_image_path, original_pil_image, current_analysis_result
    - zoom_level, highlighted_bbox, hover_widgets
    - visibility_checks, etc.
    
    Benefits:
    - Thread-safe (immutable)
    - Automatic undo/redo (state history)
    - Traceable changes (all updates explicit)
    - Testable (pure functions)
    """
    
    # Image state
    current_image_path: Optional[Path] = None
    current_image_index: int = -1
    image_files: Tuple[str, ...] = field(default_factory=tuple)
    
    # Analysis state
    current_analysis: Optional[Dict[str, Any]] = None
    is_processing: bool = False
    
    # Visual state
    visualization: VisualizationState = field(default_factory=VisualizationState)
    canvas: CanvasState = field(default_factory=CanvasState)
    
    # Window state
    window_geometry: Optional[QRect] = None
    left_panel_width: int = 320
    bottom_panel_height: int = 300
    
    # Computed properties
    @property
    def has_image(self) -> bool:
        """Check if an image is currently loaded."""
        return self.current_image_path is not None
    
    @property
    def has_analysis(self) -> bool:
        """Check if analysis results are available."""
        return self.current_analysis is not None
    
    @property
    def can_navigate_prev(self) -> bool:
        """Check if previous image navigation is possible."""
        return self.current_image_index > 0
    
    @property
    def can_navigate_next(self) -> bool:
        """Check if next image navigation is possible."""
        return self.current_image_index < len(self.image_files) - 1
    
    def with_update(self, **kwargs) -> 'AppState':
        """
        Immutable update pattern - return new state with changes.
        
        Args:
            **kwargs: Fields to update
        
        Returns:
            New AppState with updates applied
        
        Example:
            >>> new_state = state.with_update(zoom_level=1.5, is_processing=True)
        """
        return replace(self, **kwargs)
    
    def to_dict(self) -> dict:
        """
        Serialize state to dictionary for persistence.
        
        Returns:
            Dictionary representation of state
        """
        return {
            'current_image_path': str(self.current_image_path) if self.current_image_path else None,
            'current_image_index': self.current_image_index,
            'zoom_level': self.canvas.zoom_level,
            'pan_offset': self.canvas.pan_offset,
            'visible_classes': list(self.visualization.visible_classes),
            'show_baseline': self.visualization.show_baseline,
            'show_grid': self.visualization.show_grid,
            'window_geometry': self.window_geometry.getRect() if self.window_geometry else None,
            'left_panel_width': self.left_panel_width,
            'bottom_panel_height': self.bottom_panel_height,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AppState':
        """
        Deserialize state from dictionary.
        
        Args:
            data: Dictionary representation
        
        Returns:
            AppState instance
        """
        canvas = CanvasState(
            zoom_level=data.get('zoom_level', 1.0),
            pan_offset=tuple(data.get('pan_offset', (0.0, 0.0)))
        )
        
        visualization = VisualizationState(
            visible_classes=frozenset(data.get('visible_classes', [])),
            show_baseline=data.get('show_baseline', True),
            show_grid=data.get('show_grid', False)
        )
        
        return cls(
            current_image_path=Path(data['current_image_path']) if data.get('current_image_path') else None,
            current_image_index=data.get('current_image_index', -1),
            canvas=canvas,
            visualization=visualization,
            left_panel_width=data.get('left_panel_width', 320),
            bottom_panel_height=data.get('bottom_panel_height', 300)
        )


class StateManager(QObject):
    """
    Manages application state transitions with undo/redo support.
    
    Provides:
    - Immutable state updates with change notifications
    - Undo/redo history (up to 50 states)
    - State persistence (save/load)
    
    Signals:
        state_changed: Emitted when state changes (passes new AppState)
    """
    
    state_changed = pyqtSignal(object)  # AppState
    
    def __init__(self, initial_state: Optional[AppState] = None):
        """
        Initialize state manager.
        
        Args:
            initial_state: Initial application state (default: empty AppState)
        """
        super().__init__()
        self._current = initial_state or AppState()
        self._history: List[AppState] = [self._current]
        self._history_index = 0
        self._max_history = 50
        
        logger.info("StateManager initialized")
    
    def get_state(self) -> AppState:
        """
        Get current immutable state.
        
        Returns:
            Current AppState (immutable snapshot)
        
        Example:
            >>> state = manager.get_state()
            >>> if state.has_analysis:
            ...     display_results(state.current_analysis)
        """
        return self._current
    
    def update(self, **kwargs) -> None:
        """
        Update state and emit change signal.
        
        Creates new state with updates, adds to history, and notifies observers.
        
        Args:
            **kwargs: Fields to update
        
        Example:
            >>> manager.update(zoom_level=1.5, is_processing=True)
            # state_changed signal emitted automatically
        """
        new_state = self._current.with_update(**kwargs)
        if new_state != self._current:
            self._push_history(new_state)
            self._current = new_state
            self.state_changed.emit(self._current)
            logger.debug(f"State updated: {kwargs.keys()}")
    
    def replace_state(self, new_state: AppState) -> None:
        """
        Replace entire state (for complex updates).
        
        Args:
            new_state: New AppState to replace current
        
        Example:
            >>> new_viz = state.visualization.with_class_visibility('bar', True)
            >>> new_state = state.with_update(visualization=new_viz)
            >>> manager.replace_state(new_state)
        """
        if new_state != self._current:
            self._push_history(new_state)
            self._current = new_state
            self.state_changed.emit(self._current)
            logger.debug("State replaced")
    
    def _push_history(self, state: AppState) -> None:
        """
        Add state to undo history.
        
        Args:
            state: State to add to history
        """
        # Truncate forward history if we're not at the end
        self._history = self._history[:self._history_index + 1]
        
        # Add new state
        self._history.append(state)
        self._history_index += 1
        
        # Limit history size (keep most recent states)
        if len(self._history) > self._max_history:
            self._history.pop(0)
            self._history_index -= 1
    
    def undo(self) -> bool:
        """
        Undo last state change.
        
        Returns:
            True if undo succeeded, False if at beginning of history
        
        Example:
            >>> manager.update(zoom_level=2.0)
            >>> manager.undo()  # Restores previous zoom_level
        """
        if self._history_index > 0:
            self._history_index -= 1
            self._current = self._history[self._history_index]
            self.state_changed.emit(self._current)
            logger.debug(f"Undo to history index {self._history_index}")
            return True
        logger.debug("Cannot undo: at beginning of history")
        return False
    
    def redo(self) -> bool:
        """
        Redo last undone state change.
        
        Returns:
            True if redo succeeded, False if at end of history
        
        Example:
            >>> manager.undo()
            >>> manager.redo()  # Restores state after undo
        """
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._current = self._history[self._history_index]
            self.state_changed.emit(self._current)
            logger.debug(f"Redo to history index {self._history_index}")
            return True
        logger.debug("Cannot redo: at end of history")
        return False
    
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return self._history_index > 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return self._history_index < len(self._history) - 1
    
    def clear_history(self) -> None:
        """Clear undo/redo history, keeping current state."""
        self._history = [self._current]
        self._history_index = 0
        logger.debug("History cleared")
    
    def get_history_info(self) -> dict:
        """
        Get undo/redo history information.
        
        Returns:
            Dictionary with history stats
        """
        return {
            'total_states': len(self._history),
            'current_index': self._history_index,
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo()
        }
