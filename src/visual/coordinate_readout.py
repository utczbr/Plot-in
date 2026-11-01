# -*- coding: utf-8 -*-
"""
Coordinate Readout Widget — Live canvas pixel and data coordinate display.

Implements ADR-013 (T-013.1):
- Listens to coordinates_hovered(QPointF) and coordinates_left() from DetectionCanvasView.
- Displays formatted coordinates: "X: <val>  Y1: <val>" or "X: <val>  Y1: <val>  Y2: <val>"
  when dual-axis calibration is present.
- Fallback: if uncalibrated, displays "X: <px> px  Y: <py> px" (or "X: --  Y: --" outside canvas).
- Fixed axis fallback: on vertical charts without horizontal calibration, never falls back
  to the vertical calibration formula for X! Categorical/uncalibrated X is displayed as "--".
- Fixed log scale reconstruction: resolves log scale calibrations via exp(m * px + b).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QLabel, QWidget

try:
    from core.numeric_format import format_display
except ImportError:
    from src.core.numeric_format import format_display


class CoordinateReadout(QLabel):
    """
    Status-bar widget displaying live canvas pixel and data coordinates.

    Inherits from QLabel (which is a QWidget), providing permanent status bar
    compatibility with fixed-width formatting (< 1 ms update time).
    """

    DEFAULT_UNHOVERED_TEXT = "X: --  Y: --"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setMinimumWidth(240)
        self.setText(self.DEFAULT_UNHOVERED_TEXT)

        # Internal state
        self._canvas_view: Optional[Any] = None
        self._calibration: Optional[Dict[str, Any]] = None
        self._chart_type: Optional[str] = None
        self._current_pos: Optional[QPointF] = None
        self._current_pixel: Optional[Tuple[float, float]] = None
        self._current_calculated: Optional[Dict[str, Optional[float]]] = None

        # Expose self as label property for backwards compatibility
        self.label = self

    # ── Context Configuration ──

    def set_context(
        self,
        calibration: Optional[Dict[str, Any]] = None,
        chart_type: Optional[str] = None,
    ) -> None:
        """
        Set or update the active calibration dictionary and chart type.

        Args:
            calibration: Mapping containing 'primary', 'secondary', 'x', and/or 'y'
                         calibration objects (CalibrationResult or legacy dict).
            chart_type: Optional chart type identifier (e.g. 'bar', 'line', 'scatter',
                        'horizontal_bar', 'pie').
        """
        self._calibration = calibration or {}
        if chart_type is not None:
            self._chart_type = chart_type
        # Re-render if cursor is currently hovering
        if self._current_pos is not None:
            self.on_coordinates_hovered(self._current_pos)

    def attach_canvas(self, canvas_view: Any) -> None:
        """
        Connect to a DetectionCanvasView instance's hover and leave signals.

        Args:
            canvas_view: A DetectionCanvasView instance emitting coordinates_hovered(QPointF)
                         and coordinates_left().
        """
        self._canvas_view = canvas_view
        if hasattr(canvas_view, "coordinates_hovered"):
            canvas_view.coordinates_hovered.connect(self.on_coordinates_hovered)
        if hasattr(canvas_view, "coordinates_left"):
            canvas_view.coordinates_left.connect(self.clear_coordinates)

    def connect_canvas(self, canvas_view: Any) -> None:
        """Alias for attach_canvas."""
        self.attach_canvas(canvas_view)

    # ── Signal Handlers ──

    def on_coordinates_hovered(self, scene_pos: QPointF) -> None:
        """
        Slot invoked when mouse hovers over the canvas scene.

        Args:
            scene_pos: Scene position in image pixels.
        """
        if scene_pos is None:
            self.clear_coordinates()
            return

        self._current_pos = scene_pos
        px = float(scene_pos.x())
        py = float(scene_pos.y())
        self._current_pixel = (px, py)

        # Check bounds if canvas view / scene is attached
        if self._canvas_view is not None:
            scene = getattr(self._canvas_view, "scene", lambda: None)()
            if scene is not None and hasattr(scene, "sceneRect"):
                if not scene.sceneRect().contains(scene_pos):
                    self.setText(self.DEFAULT_UNHOVERED_TEXT)
                    self._current_calculated = None
                    return

        # Retrieve calibration & chart type (local context takes precedence over scene)
        cal = self._calibration
        chart_type = self._chart_type
        if cal is None and self._canvas_view is not None:
            scene = getattr(self._canvas_view, "scene", lambda: None)()
            if scene is not None:
                cal = getattr(scene, "_calibration", None)
                if chart_type is None:
                    chart_type = getattr(scene, "_chart_type", None)

        # Fallback when uncalibrated: format pixel coordinates
        if not cal:
            px_str = format_display(round(px, 1))
            py_str = format_display(round(py, 1))
            self.setText(f"X: {px_str} px  Y: {py_str} px")
            self._current_calculated = None
            return

        # Resolve axis calibrations without invalid fallback
        from visual.detection_scene import DetectionScene

        is_horizontal = str(chart_type).lower() == "horizontal_bar"
        if is_horizontal:
            cal_x = cal.get("x") or cal.get("primary")
            cal_y1 = cal.get("y")
        else:
            # On vertical charts: primary is Y. Never fall back to primary for X!
            cal_x = cal.get("x")
            cal_y1 = cal.get("y") or cal.get("primary")
        cal_y2 = cal.get("secondary")

        func_x = DetectionScene._resolve_cal_func(cal_x) if cal_x else None
        func_y1 = DetectionScene._resolve_cal_func(cal_y1) if cal_y1 else None
        func_y2 = DetectionScene._resolve_cal_func(cal_y2) if cal_y2 else None

        # If no axis functions could be resolved, fall back to pixel readout
        if func_x is None and func_y1 is None and func_y2 is None:
            px_str = format_display(round(px, 1))
            py_str = format_display(round(py, 1))
            self.setText(f"X: {px_str} px  Y: {py_str} px")
            self._current_calculated = None
            return

        calc_x = float(func_x(px)) if func_x else None
        calc_y1 = float(func_y1(py)) if func_y1 else None
        calc_y2 = float(func_y2(py)) if func_y2 else None

        self._current_calculated = {
            "x": calc_x,
            "y1": calc_y1,
            "y2": calc_y2,
        }

        # Format values
        str_x = format_display(calc_x) if calc_x is not None else "--"
        str_y1 = format_display(calc_y1) if calc_y1 is not None else "--"

        if cal_y2 is not None:
            str_y2 = format_display(calc_y2) if calc_y2 is not None else "--"
            self.setText(f"X: {str_x}  Y1: {str_y1}  Y2: {str_y2}")
        else:
            self.setText(f"X: {str_x}  Y1: {str_y1}")

    def update_coordinates(
        self,
        scene_pos: QPointF,
        calculated_coords: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> None:
        """
        Explicit update method accepting scene_pos and optional pre-calculated coordinates.
        """
        if calculated_coords is not None:
            self._current_pos = scene_pos
            if scene_pos is not None:
                self._current_pixel = (float(scene_pos.x()), float(scene_pos.y()))
            cx, cy = calculated_coords
            str_x = format_display(cx) if cx is not None else "--"
            str_y = format_display(cy) if cy is not None else "--"
            self.setText(f"X: {str_x}  Y1: {str_y}")
            self._current_calculated = {"x": cx, "y1": cy, "y2": None}
            return
        self.on_coordinates_hovered(scene_pos)

    def clear_coordinates(self) -> None:
        """Clear coordinate display (called when cursor leaves canvas)."""
        self.setText(self.DEFAULT_UNHOVERED_TEXT)
        self._current_pos = None
        self._current_pixel = None
        self._current_calculated = None

    def on_coordinates_left(self) -> None:
        """Slot for mouse leave event."""
        self.clear_coordinates()

    # ── Inspection Properties ──

    @property
    def current_pixel(self) -> Optional[Tuple[float, float]]:
        """Active pixel coordinates (x, y) or None if outside canvas."""
        return self._current_pixel

    @property
    def current_calculated(self) -> Optional[Dict[str, Optional[float]]]:
        """Active calculated coordinates dict with 'x', 'y1', and 'y2'."""
        return self._current_calculated
