"""
Manual axis calibration model and utilities (pure Python, zero Qt dependencies).

Implements ManualAxisCalibration according to ADR-013 (T-013.3) for user-defined
calibration overrides on cartesian chart axes. Supports linear and log scales,
multi-point least-squares fitting, and conversion to CalibrationResult.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from .calibration_base import CalibrationFunc, CalibrationResult
except ImportError:
    from calibration.calibration_base import CalibrationFunc, CalibrationResult


@dataclass
class ManualAxisCalibration:
    """
    User-defined manual axis calibration model.

    Attributes:
        axis: Axis identifier ('x' or 'y').
        points: List of (pixel, value) calibration points (>= 2 points).
        scale: Scale type ('linear' or 'log'). Default is 'linear'.
        created_at: Optional ISO-8601 timestamp string when calibration was created.
    """
    axis: str
    points: List[Tuple[float, float]]
    scale: str = "linear"
    created_at: Optional[str] = None

    _slope: float = field(init=False, repr=False, compare=False)
    _intercept: float = field(init=False, repr=False, compare=False)
    _r_squared: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate and normalize axis
        if not isinstance(self.axis, str) or self.axis.lower() not in ("x", "y"):
            raise ValueError(f"Invalid axis '{self.axis}'. Must be 'x' or 'y'.")
        self.axis = self.axis.lower()

        # Validate and normalize scale
        if not isinstance(self.scale, str) or self.scale.lower() not in ("linear", "log"):
            raise ValueError(f"Invalid scale '{self.scale}'. Must be 'linear' or 'log'.")
        self.scale = self.scale.lower()

        # Validate points count
        if not hasattr(self.points, "__len__") or len(self.points) < 2:
            count = len(self.points) if hasattr(self.points, "__len__") else 0
            raise ValueError(f"At least 2 points are required for calibration, got {count}.")

        # Convert and validate each point as (float, float)
        normalized_points: List[Tuple[float, float]] = []
        for idx, pt in enumerate(self.points):
            if not hasattr(pt, "__len__") or len(pt) != 2:
                raise ValueError(f"Point {idx} must be a (pixel, value) pair, got {pt}.")
            px, val = float(pt[0]), float(pt[1])
            if not math.isfinite(px):
                raise ValueError(f"Point {idx} pixel coordinate must be finite, got {px}.")
            if not math.isfinite(val):
                raise ValueError(f"Point {idx} value must be finite, got {val}.")
            normalized_points.append((px, val))
        self.points = normalized_points

        # Validate log scale positivity
        if self.scale == "log":
            for idx, (px, val) in enumerate(self.points):
                if val <= 0.0:
                    raise ValueError(
                        f"Log scale requires all values to be strictly positive (> 0), "
                        f"got {val} at point {idx}."
                    )

        # Validate distinct pixel coordinates and distinct data values across all pairs
        n = len(self.points)
        for i in range(n):
            px_i, val_i = self.points[i]
            for j in range(i + 1, n):
                px_j, val_j = self.points[j]
                if abs(px_i - px_j) < 1.0:
                    raise ValueError(
                        f"Pixel coordinates must be distinct (|delta_px| >= 1.0 px): "
                        f"point {i} ({px_i}) and point {j} ({px_j})."
                    )
                if math.isclose(val_i, val_j, abs_tol=1e-12):
                    raise ValueError(
                        f"Data values must be distinct: "
                        f"point {i} ({val_i}) and point {j} ({val_j})."
                    )

        # Compute fit parameters
        self._compute_fit()

    def _compute_fit(self) -> None:
        """Calculate least-squares linear or log-linear model parameters."""
        x = [p[0] for p in self.points]
        y = [p[1] for p in self.points]

        if self.scale == "log":
            y_fit = [math.log(v) for v in y]
        else:
            y_fit = [v for v in y]

        n = len(self.points)
        if n == 2:
            delta_px = x[1] - x[0]
            delta_y = y_fit[1] - y_fit[0]
            slope = delta_y / delta_px
            intercept = y_fit[0] - slope * x[0]
            r2 = 1.0
        else:
            x_arr = np.array(x, dtype=float)
            y_arr = np.array(y_fit, dtype=float)
            poly = np.polyfit(x_arr, y_arr, 1)
            slope = float(poly[0])
            intercept = float(poly[1])

            # R^2 calculation
            y_pred = slope * x_arr + intercept
            ss_res = float(np.sum((y_arr - y_pred) ** 2))
            y_mean = float(np.mean(y_arr))
            ss_tot = float(np.sum((y_arr - y_mean) ** 2))
            if ss_tot > 1e-15:
                r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))
            else:
                r2 = 0.0

        self._slope = float(slope)
        self._intercept = float(intercept)
        self._r_squared = float(r2)

    @property
    def slope(self) -> float:
        """Model slope (in linear or log space)."""
        return self._slope

    @property
    def intercept(self) -> float:
        """Model intercept (in linear or log space)."""
        return self._intercept

    @property
    def r_squared(self) -> float:
        """Coefficient of determination R^2."""
        return self._r_squared

    @property
    def r2(self) -> float:
        """Alias for r_squared."""
        return self._r_squared

    @property
    def is_inverted(self) -> bool:
        """
        True if value decreases as pixel coordinate increases.

        For both linear and log scales, this corresponds to slope < 0.
        """
        return bool(self._slope < 0.0)

    def pixel_to_value(self, px: float) -> float:
        """
        Map a pixel coordinate to a data value.

        Args:
            px: Pixel coordinate along the calibrated axis.

        Returns:
            Calculated data value.
        """
        px_f = float(px)
        if self.scale == "log":
            exponent = self._slope * px_f + self._intercept
            clamped_exp = max(-700.0, min(700.0, exponent))
            return float(math.exp(clamped_exp))
        return float(self._slope * px_f + self._intercept)

    def value_to_pixel(self, val: float) -> float:
        """
        Map a data value back to a pixel coordinate.

        Args:
            val: Data value along the calibrated axis.

        Returns:
            Calculated pixel coordinate.

        Raises:
            ValueError: If val <= 0 on a log scale.
            ZeroDivisionError: If slope is zero.
        """
        val_f = float(val)
        if math.isclose(self._slope, 0.0, abs_tol=1e-15):
            raise ZeroDivisionError("Slope is zero; cannot invert calibration.")

        if self.scale == "log":
            if val_f <= 0.0:
                raise ValueError(
                    f"Cannot map non-positive value {val} to pixel coordinate on log scale."
                )
            log_val = math.log(val_f)
            return float((log_val - self._intercept) / self._slope)
        return float((val_f - self._intercept) / self._slope)

    def residuals(self, ref_points: List[Tuple[float, float]]) -> List[float]:
        """
        Compute residuals for a set of reference (pixel, value) points.

        Residual is defined as (predicted_value - actual_value).

        Args:
            ref_points: List of (pixel, reference_value) tuples.

        Returns:
            List of float residuals corresponding to each reference point.
        """
        return [float(self.pixel_to_value(pt[0]) - float(pt[1])) for pt in ref_points]

    def to_calibration_result(self) -> CalibrationResult:
        """
        Convert to CalibrationResult container for pipeline/handler consumption.

        Returns:
            CalibrationResult with provenance='manual', n_points=len(points),
            r2, coeffs, and a callable supporting both scalar and array inputs.
        """
        slope = self._slope
        intercept = self._intercept
        scale = self.scale

        if scale == "log":
            def _func(px: Union[np.ndarray, float, List[float]]) -> Union[np.ndarray, float]:
                if isinstance(px, (np.ndarray, list, tuple)):
                    arr = np.asarray(px, dtype=float)
                    exp_arr = np.clip(slope * arr + intercept, -700.0, 700.0)
                    res = np.exp(exp_arr)
                    return res if isinstance(px, np.ndarray) else res.tolist()
                exp_val = max(-700.0, min(700.0, slope * float(px) + intercept))
                return float(math.exp(exp_val))
        else:
            def _func(px: Union[np.ndarray, float, List[float]]) -> Union[np.ndarray, float]:
                if isinstance(px, (np.ndarray, list, tuple)):
                    arr = np.asarray(px, dtype=float)
                    res = slope * arr + intercept
                    return res if isinstance(px, np.ndarray) else res.tolist()
                return float(slope * float(px) + intercept)

        return CalibrationResult(
            func=_func,
            r2=float(self._r_squared),
            coeffs=(float(self._slope), float(self._intercept)),
            inliers=np.ones(len(self.points), dtype=bool),
            is_inverted=self.is_inverted,
            coordinate_system="image",
            scale_type=self.scale,
            n_points=len(self.points),
            provenance="manual",
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary representation.

        Returns:
            Dict containing axis, points, scale, and created_at.
        """
        return {
            "axis": self.axis,
            "points": [(float(p[0]), float(p[1])) for p in self.points],
            "scale": self.scale,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ManualAxisCalibration:
        """
        Construct a ManualAxisCalibration instance from a dictionary.

        Args:
            d: Dictionary containing axis, points, and optional scale/created_at.

        Returns:
            ManualAxisCalibration instance.
        """
        raw_points = d["points"]
        points = [(float(p[0]), float(p[1])) for p in raw_points]
        return cls(
            axis=str(d["axis"]),
            points=points,
            scale=str(d.get("scale", "linear")),
            created_at=d.get("created_at"),
        )
