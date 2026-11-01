# -*- coding: utf-8 -*-
"""
Tests for CoordinateReadout widget and axis coordinate resolution (T-013.1, ADR-013).
"""

import math
import time
import pytest
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import QWidget

from calibration.calibration_base import CalibrationResult
from visual.coordinate_readout import CoordinateReadout
from visual.detection_scene import DetectionCanvasView, DetectionScene


@pytest.fixture
def scene_and_view(qapp):
    scene = DetectionScene()
    scene.setSceneRect(QRectF(0, 0, 800, 600))
    view = DetectionCanvasView(scene)
    return scene, view


class TestCoordinateReadoutWidget:
    """Test CoordinateReadout widget display formats and edge cases."""

    def test_widget_inheritance_and_defaults(self, qapp):
        readout = CoordinateReadout()
        assert isinstance(readout, QWidget)
        assert readout.text() == "X: --  Y: --"
        assert readout.current_pixel is None
        assert readout.current_calculated is None

    def test_uncalibrated_pixel_readout(self, qapp, scene_and_view):
        scene, view = scene_and_view
        readout = CoordinateReadout()
        readout.attach_canvas(view)

        # Hover inside canvas bounds (120.5, 340.0)
        pos = QPointF(120.5, 340.0)
        readout.on_coordinates_hovered(pos)

        text = readout.text()
        assert "X: 120.5 px" in text
        assert "Y: 340 px" in text
        assert readout.current_pixel == (120.5, 340.0)
        assert readout.current_calculated is None

    def test_outside_canvas_clears_readout(self, qapp, scene_and_view):
        scene, view = scene_and_view
        readout = CoordinateReadout()
        readout.attach_canvas(view)

        # Hover outside scene bounds (-10, 700)
        readout.on_coordinates_hovered(QPointF(-10.0, 700.0))
        assert readout.text() == "X: --  Y: --"
        assert readout.current_calculated is None

    def test_mouse_leave_signal_clears_readout(self, qapp, scene_and_view):
        scene, view = scene_and_view
        readout = CoordinateReadout()
        readout.attach_canvas(view)

        # First hover inside
        readout.on_coordinates_hovered(QPointF(200.0, 200.0))
        assert "px" in readout.text()

        # Emit coordinates_left from view
        view.coordinates_left.emit()
        assert readout.text() == "X: --  Y: --"
        assert readout.current_pixel is None

    def test_vertical_bar_chart_axis_fallback_fix(self, qapp, scene_and_view):
        """
        T-013.1 bugfix: On vertical charts without horizontal calibration,
        X must NOT fall back to the vertical calibration formula!
        """
        scene, view = scene_and_view
        readout = CoordinateReadout()
        readout.attach_canvas(view)

        # Vertical calibration: y = -0.1 * py + 50.0 (maps py=100 -> 40.0, py=300 -> 20.0)
        cal_y = CalibrationResult(
            func=lambda py: -0.1 * py + 50.0,
            r2=0.999,
            coeffs=(-0.1, 50.0),
            scale_type="linear",
        )
        calibration = {
            "primary": cal_y,
            "y": cal_y,
            "x": None,  # Categorical bar chart, no X calibration
        }
        scene.set_calibration(calibration, chart_type="bar")
        readout.set_context(calibration, chart_type="bar")

        # Hover at px=200, py=100
        readout.on_coordinates_hovered(QPointF(200.0, 100.0))

        text = readout.text()
        # X must be uncalibrated placeholder "--" and NOT 30.0 (which would be -0.1 * 200 + 50)
        assert "X: --" in text
        assert "Y1: 40" in text
        assert "-0.1" not in text
        assert "30" not in text  # No bogus horizontal numbers calculated from vertical formula

        calc = readout.current_calculated
        assert calc is not None
        assert calc["x"] is None
        assert math.isclose(calc["y1"], 40.0, rel_tol=1e-5)

    def test_dual_axis_calibration_readout(self, qapp, scene_and_view):
        """
        T-013.1 / DP-08: Displays X, Y1, and Y2 when secondary axis calibration is present.
        """
        scene, view = scene_and_view
        readout = CoordinateReadout()
        readout.attach_canvas(view)

        cal_y1 = CalibrationResult(
            func=lambda py: 0.1 * py,
            r2=0.998,
            coeffs=(0.1, 0.0),
            scale_type="linear",
        )
        cal_y2 = CalibrationResult(
            func=lambda py: 0.01 * py,
            r2=0.995,
            coeffs=(0.01, 0.0),
            scale_type="linear",
        )
        calibration = {
            "primary": cal_y1,
            "y": cal_y1,
            "secondary": cal_y2,
            "x": None,
        }
        scene.set_calibration(calibration, chart_type="bar")
        readout.set_context(calibration, chart_type="bar")

        # py = 200.0 -> Y1 = 20, Y2 = 2
        readout.on_coordinates_hovered(QPointF(150.0, 200.0))
        text = readout.text()
        assert "X: --" in text
        assert "Y1: 20" in text
        assert "Y2: 2" in text

        calc = readout.current_calculated
        assert calc is not None
        assert calc["x"] is None
        assert math.isclose(calc["y1"], 20.0, rel_tol=1e-5)
        assert math.isclose(calc["y2"], 2.0, rel_tol=1e-5)

    def test_dual_axis_scatter_chart_readout(self, qapp, scene_and_view):
        """Scatter chart with both X and Y calibrated."""
        scene, view = scene_and_view
        readout = CoordinateReadout()
        readout.attach_canvas(view)

        cal_x = CalibrationResult(
            func=lambda px: 0.5 * px,
            r2=0.999,
            coeffs=(0.5, 0.0),
            scale_type="linear",
        )
        cal_y = CalibrationResult(
            func=lambda py: 2.0 * py,
            r2=0.999,
            coeffs=(2.0, 0.0),
            scale_type="linear",
        )
        calibration = {"x": cal_x, "y": cal_y, "primary": cal_y}
        scene.set_calibration(calibration, chart_type="scatter")
        readout.set_context(calibration, chart_type="scatter")

        # px=10 -> X=5, py=20 -> Y=40
        readout.on_coordinates_hovered(QPointF(10.0, 20.0))
        text = readout.text()
        assert "X: 5" in text
        assert "Y1: 40" in text

    def test_log_scale_reconstruction_bugfix(self, qapp, scene_and_view):
        """
        T-013.1 bugfix: On log scale, _resolve_cal_func must reconstruct exp(m * px + b)
        rather than returning the natural logarithm.
        """
        scene, view = scene_and_view
        scene.setSceneRect(QRectF(0, 0, 1000, 1000))
        readout = CoordinateReadout()
        readout.attach_canvas(view)

        # Suppose log10 or ln fit: m = 0.01, b = 0.0 on log scale
        # At px = 690.7755, m*px = 6.907755 -> exp(6.907755) = 1000.0
        cal_log_dict = {
            "scale_type": "log",
            "coeffs": (0.01, 0.0),
            "r2": 0.99,
        }
        calibration = {"primary": cal_log_dict, "y": cal_log_dict, "x": None}
        scene.set_calibration(calibration, chart_type="line")
        readout.set_context(calibration, chart_type="line")

        # Test scene resolve directly
        func = DetectionScene._resolve_cal_func(cal_log_dict)
        assert func is not None
        val = func(690.7755278982137)
        assert math.isclose(val, 1000.0, rel_tol=1e-4)

        # Test through readout
        readout.on_coordinates_hovered(QPointF(100.0, 690.7755278982137))
        text = readout.text()
        assert "Y1: 1000" in text

    def test_formatting_latency_under_1ms(self, qapp, scene_and_view):
        """ADR-013 acceptance: formatting time < 1 ms."""
        scene, view = scene_and_view
        readout = CoordinateReadout()
        readout.attach_canvas(view)

        cal_y = CalibrationResult(
            func=lambda py: 0.1 * py,
            r2=0.999,
            coeffs=(0.1, 0.0),
            scale_type="linear",
        )
        calibration = {"primary": cal_y, "y": cal_y, "x": None}
        readout.set_context(calibration, chart_type="bar")

        start = time.perf_counter()
        for i in range(100):
            readout.on_coordinates_hovered(QPointF(100.0 + i, 200.0 + i))
        elapsed_total = time.perf_counter() - start
        avg_ms = (elapsed_total / 100.0) * 1000.0
        assert avg_ms < 1.0, f"Average formatting time was {avg_ms:.3f} ms (must be < 1 ms)"
