"""
Unit tests for ManualAxisCalibration model and CalibrationResult provenance fields (ADR-013, T-013.3).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from calibration.calibration_base import CalibrationResult
from calibration.calibration_manual import ManualAxisCalibration


class TestManualAxisCalibrationLinear:
    """Tests for linear scale manual calibration."""

    def test_linear_2_point_interpolation(self):
        """Linear 2-point: pixels (100->0, 300->10) => f(200) == 5.0."""
        cal = ManualAxisCalibration(
            axis="x",
            points=[(100.0, 0.0), (300.0, 10.0)],
            scale="linear",
        )
        assert cal.axis == "x"
        assert cal.scale == "linear"
        assert cal.r_squared == 1.0
        assert cal.r2 == 1.0
        assert not cal.is_inverted
        assert math.isclose(cal.slope, 0.05, abs_tol=1e-9)
        assert math.isclose(cal.intercept, -5.0, abs_tol=1e-9)

        # Forward mapping: f(200) == 5.0
        val = cal.pixel_to_value(200.0)
        assert math.isclose(val, 5.0, abs_tol=1e-9)

        # Reverse mapping: value_to_pixel(5.0) == 200.0
        px = cal.value_to_pixel(5.0)
        assert math.isclose(px, 200.0, abs_tol=1e-9)

        # Edge points
        assert math.isclose(cal.pixel_to_value(100.0), 0.0, abs_tol=1e-9)
        assert math.isclose(cal.pixel_to_value(300.0), 10.0, abs_tol=1e-9)

    def test_inverted_y_axis(self):
        """Inverted y: pixels (400->0, 100->30) => is_inverted is True, f(250) == 15.0."""
        cal = ManualAxisCalibration(
            axis="y",
            points=[(400.0, 0.0), (100.0, 30.0)],
            scale="linear",
        )
        assert cal.axis == "y"
        assert cal.is_inverted is True
        assert cal.r_squared == 1.0
        assert math.isclose(cal.slope, -0.1, abs_tol=1e-9)
        assert math.isclose(cal.intercept, 40.0, abs_tol=1e-9)

        # Forward mapping: f(250) == 15.0
        val = cal.pixel_to_value(250.0)
        assert math.isclose(val, 15.0, abs_tol=1e-9)

        # Reverse mapping: value_to_pixel(15.0) == 250.0
        px = cal.value_to_pixel(15.0)
        assert math.isclose(px, 250.0, abs_tol=1e-9)

        # Value decreases as pixel increases:
        assert cal.pixel_to_value(100.0) > cal.pixel_to_value(400.0)

    def test_inverted_x_axis(self):
        """Inverted x axis when slope < 0 (e.g. astronomical coordinates)."""
        cal = ManualAxisCalibration(
            axis="x",
            points=[(100.0, 100.0), (500.0, 20.0)],
            scale="linear",
        )
        assert cal.is_inverted is True
        assert cal.pixel_to_value(100.0) > cal.pixel_to_value(500.0)


class TestManualAxisCalibrationLog:
    """Tests for log scale manual calibration."""

    def test_log_2_point_interpolation(self):
        """Log 2-point: pixels (100->1, 300->100) => f(200) == 10.0 (+-1e-9)."""
        cal = ManualAxisCalibration(
            axis="y",
            points=[(100.0, 1.0), (300.0, 100.0)],
            scale="log",
        )
        assert cal.scale == "log"
        assert cal.r_squared == 1.0
        assert not cal.is_inverted

        # Midpoint on log scale: ln(100)/200 * 200 - ln(10) = ln(10) => exp(ln(10)) = 10.0
        val = cal.pixel_to_value(200.0)
        assert math.isclose(val, 10.0, rel_tol=1e-9, abs_tol=1e-9)

        # Endpoints
        assert math.isclose(cal.pixel_to_value(100.0), 1.0, rel_tol=1e-9)
        assert math.isclose(cal.pixel_to_value(300.0), 100.0, rel_tol=1e-9)

        # Reverse mapping: value_to_pixel(10.0) == 200.0
        px = cal.value_to_pixel(10.0)
        assert math.isclose(px, 200.0, rel_tol=1e-9, abs_tol=1e-9)

    def test_log_inverted_axis(self):
        """Inverted log axis (value decreases as pixel increases)."""
        cal = ManualAxisCalibration(
            axis="y",
            points=[(100.0, 1000.0), (400.0, 1.0)],
            scale="log",
        )
        assert cal.is_inverted is True
        assert cal.slope < 0.0
        # f(250) is geometric mean of 1000 and 1 = sqrt(1000) ~ 31.6227766
        val = cal.pixel_to_value(250.0)
        assert math.isclose(val, math.sqrt(1000.0), rel_tol=1e-7)

    def test_log_rejects_zero_and_negative_values(self):
        """Log scale must reject non-positive values (value <= 0)."""
        with pytest.raises(ValueError, match="Log scale requires all values to be strictly positive"):
            ManualAxisCalibration(
                axis="y",
                points=[(100.0, 0.0), (300.0, 10.0)],
                scale="log",
            )

        with pytest.raises(ValueError, match="Log scale requires all values to be strictly positive"):
            ManualAxisCalibration(
                axis="x",
                points=[(100.0, -5.0), (300.0, 10.0)],
                scale="log",
            )

    def test_log_value_to_pixel_rejects_non_positive(self):
        """value_to_pixel on log scale must reject non-positive values."""
        cal = ManualAxisCalibration(
            axis="y",
            points=[(100.0, 1.0), (300.0, 100.0)],
            scale="log",
        )
        with pytest.raises(ValueError, match="Cannot map non-positive value"):
            cal.value_to_pixel(0.0)

        with pytest.raises(ValueError, match="Cannot map non-positive value"):
            cal.value_to_pixel(-1.0)


class TestManualAxisCalibrationValidation:
    """Tests for input validation and rejections."""

    def test_rejection_fewer_than_2_points(self):
        """Reject fewer than 2 points."""
        with pytest.raises(ValueError, match="At least 2 points are required"):
            ManualAxisCalibration(axis="x", points=[])

        with pytest.raises(ValueError, match="At least 2 points are required"):
            ManualAxisCalibration(axis="x", points=[(100.0, 10.0)])

    def test_rejection_identical_pixels(self):
        """Reject identical or near-identical pixel coordinates (|delta_px| < 1.0)."""
        # Exactly identical
        with pytest.raises(ValueError, match=r"Pixel coordinates must be distinct \(\|delta_px\| >= 1.0 px\)"):
            ManualAxisCalibration(axis="x", points=[(100.0, 0.0), (100.0, 10.0)])

        # Closer than 1.0 px
        with pytest.raises(ValueError, match=r"Pixel coordinates must be distinct \(\|delta_px\| >= 1.0 px\)"):
            ManualAxisCalibration(axis="x", points=[(100.0, 0.0), (100.6, 10.0)])

    def test_rejection_identical_values(self):
        """Reject identical data values."""
        with pytest.raises(ValueError, match="Data values must be distinct"):
            ManualAxisCalibration(axis="x", points=[(100.0, 5.0), (300.0, 5.0)])

        with pytest.raises(ValueError, match="Data values must be distinct"):
            ManualAxisCalibration(axis="y", points=[(100.0, 0.0), (200.0, 10.0), (300.0, 0.0)])

    def test_rejection_invalid_axis(self):
        """Reject axis names other than 'x' and 'y'."""
        with pytest.raises(ValueError, match="Invalid axis 'z'"):
            ManualAxisCalibration(axis="z", points=[(100.0, 0.0), (300.0, 10.0)])

    def test_rejection_invalid_scale(self):
        """Reject scale types other than 'linear' and 'log'."""
        with pytest.raises(ValueError, match="Invalid scale 'exp'"):
            ManualAxisCalibration(axis="x", points=[(100.0, 0.0), (300.0, 10.0)], scale="exp")

    def test_rejection_non_finite_coordinates(self):
        """Reject inf and nan in points."""
        with pytest.raises(ValueError, match="pixel coordinate must be finite"):
            ManualAxisCalibration(axis="x", points=[(float("nan"), 0.0), (300.0, 10.0)])

        with pytest.raises(ValueError, match="value must be finite"):
            ManualAxisCalibration(axis="x", points=[(100.0, 0.0), (300.0, float("inf"))])

    def test_rejection_malformed_point_shape(self):
        """Reject points that are not length 2."""
        with pytest.raises(ValueError, match="must be a \\(pixel, value\\) pair"):
            ManualAxisCalibration(axis="x", points=[(100.0,), (300.0, 10.0)])  # type: ignore

    def test_case_insensitivity_and_normalization(self):
        """Axis and scale should be normalized to lower case."""
        cal = ManualAxisCalibration(
            axis="X",
            points=[[100, 0], [300, 10]],  # list of lists converts cleanly
            scale="LINEAR",
        )
        assert cal.axis == "x"
        assert cal.scale == "linear"
        assert cal.points == [(100.0, 0.0), (300.0, 10.0)]


class TestManualAxisCalibrationMultiPoint:
    """Tests for multi-point least-squares calibration and residuals."""

    def test_3_points_collinear(self):
        """3 collinear points should yield R^2 = 1.0."""
        cal = ManualAxisCalibration(
            axis="x",
            points=[(100.0, 0.0), (200.0, 5.0), (300.0, 10.0)],
        )
        assert math.isclose(cal.r_squared, 1.0, abs_tol=1e-9)
        assert math.isclose(cal.pixel_to_value(250.0), 7.5, abs_tol=1e-9)

        # Residuals on calibration points should be virtually 0
        res = cal.residuals(cal.points)
        assert len(res) == 3
        for r in res:
            assert math.isclose(r, 0.0, abs_tol=1e-9)

    def test_3_points_with_noise_yields_r2_less_than_1(self):
        """3 points with noise => r2 < 1 and n_points == 3."""
        # Expected linear points: (100, 0), (200, 10), (300, 20).
        # Perturb middle point to (200, 11)
        cal = ManualAxisCalibration(
            axis="y",
            points=[(100.0, 0.0), (200.0, 11.0), (300.0, 20.0)],
        )
        assert cal.r_squared < 1.0
        assert cal.r_squared > 0.95
        res_cal = cal.to_calibration_result()
        assert res_cal.n_points == 3
        assert res_cal.r2 < 1.0
        assert res_cal.provenance == "manual"

    def test_residuals_against_reference_points(self):
        """Residuals calculation against external reference tick points."""
        cal = ManualAxisCalibration(
            axis="x",
            points=[(100.0, 0.0), (300.0, 10.0)],
        )
        # Reference ticks:
        # px 200: predicted 5.0, actual 5.0 => residual 0.0
        # px 250: predicted 7.5, actual 7.0 => residual 0.5
        # px 150: predicted 2.5, actual 3.0 => residual -0.5
        ref_points = [(200.0, 5.0), (250.0, 7.0), (150.0, 3.0)]
        res = cal.residuals(ref_points)
        assert len(res) == 3
        assert math.isclose(res[0], 0.0, abs_tol=1e-9)
        assert math.isclose(res[1], 0.5, abs_tol=1e-9)
        assert math.isclose(res[2], -0.5, abs_tol=1e-9)


class TestManualAxisCalibrationSerialization:
    """Tests for to_dict / from_dict roundtrip."""

    def test_roundtrip_equality_with_created_at(self):
        """round-trip from_dict(to_dict(x)) equals x with created_at timestamp."""
        orig = ManualAxisCalibration(
            axis="y",
            points=[(400.0, 0.0), (100.0, 30.0)],
            scale="linear",
            created_at="2026-11-01T15:00:00Z",
        )
        d = orig.to_dict()
        assert d == {
            "axis": "y",
            "points": [(400.0, 0.0), (100.0, 30.0)],
            "scale": "linear",
            "created_at": "2026-11-01T15:00:00Z",
        }

        reconstructed = ManualAxisCalibration.from_dict(d)
        assert reconstructed == orig
        assert reconstructed.is_inverted == orig.is_inverted
        assert math.isclose(reconstructed.slope, orig.slope, abs_tol=1e-12)
        assert math.isclose(reconstructed.intercept, orig.intercept, abs_tol=1e-12)
        assert math.isclose(reconstructed.r_squared, orig.r_squared, abs_tol=1e-12)

    def test_roundtrip_equality_without_created_at(self):
        """round-trip from_dict(to_dict(x)) equals x when created_at is None."""
        orig = ManualAxisCalibration(
            axis="x",
            points=[(100.0, 1.0), (200.0, 10.0), (300.0, 100.0)],
            scale="log",
        )
        d = orig.to_dict()
        reconstructed = ManualAxisCalibration.from_dict(d)
        assert reconstructed == orig
        assert reconstructed.created_at is None


class TestCalibrationResultConversion:
    """Tests for to_calibration_result conversion and CalibrationResult provenance fields."""

    def test_to_calibration_result_fields_linear(self):
        """Convert linear ManualAxisCalibration to CalibrationResult."""
        cal = ManualAxisCalibration(
            axis="y",
            points=[(400.0, 0.0), (100.0, 30.0)],
            scale="linear",
        )
        res = cal.to_calibration_result()
        assert isinstance(res, CalibrationResult)
        assert res.provenance == "manual"
        assert res.n_points == 2
        assert res.r2 == 1.0
        assert res.scale_type == "linear"
        assert res.is_inverted is True
        assert res.coordinate_system == "image"
        assert res.coeffs == (cal.slope, cal.intercept)
        assert res.inliers is not None
        assert len(res.inliers) == 2
        assert np.all(res.inliers)

        # Test func with scalar
        assert math.isclose(res.func(250.0), 15.0, abs_tol=1e-9)

        # Test func with numpy array
        arr = np.array([400.0, 250.0, 100.0])
        val_arr = res.func(arr)
        assert isinstance(val_arr, np.ndarray)
        np.testing.assert_allclose(val_arr, [0.0, 15.0, 30.0])

        # Test func with list
        val_list = res.func([400.0, 100.0])
        assert isinstance(val_list, list)
        assert math.isclose(val_list[0], 0.0, abs_tol=1e-9)
        assert math.isclose(val_list[1], 30.0, abs_tol=1e-9)

    def test_to_calibration_result_fields_log(self):
        """Convert log ManualAxisCalibration to CalibrationResult."""
        cal = ManualAxisCalibration(
            axis="x",
            points=[(100.0, 1.0), (300.0, 100.0)],
            scale="log",
        )
        res = cal.to_calibration_result()
        assert isinstance(res, CalibrationResult)
        assert res.provenance == "manual"
        assert res.n_points == 2
        assert res.r2 == 1.0
        assert res.scale_type == "log"
        assert not res.is_inverted

        # Test func with scalar
        assert math.isclose(res.func(200.0), 10.0, rel_tol=1e-9)

        # Test func with numpy array
        arr = np.array([100.0, 200.0, 300.0])
        val_arr = res.func(arr)
        assert isinstance(val_arr, np.ndarray)
        np.testing.assert_allclose(val_arr, [1.0, 10.0, 100.0], rtol=1e-9)

    def test_calibration_result_default_provenance(self):
        """CalibrationResult should have safe defaults for backward compatibility."""
        res = CalibrationResult(
            func=lambda px: px * 2.0,
            r2=0.999,
            coeffs=(2.0, 0.0),
        )
        assert res.provenance == "auto"
        assert res.n_points is None
        assert res.scale_type == "linear"
        assert not res.is_inverted
