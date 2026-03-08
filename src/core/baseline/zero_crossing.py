"""Calibration zero-crossing baseline resolvers."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .geometry import validate_xyxy

try:
    from calibration.calibration_factory import CalibrationFactory
    from calibration.calibration_base import CalibrationResult, BaseCalibration
except Exception:  # pragma: no cover - optional dependency wiring
    CalibrationFactory = None
    CalibrationResult = None
    BaseCalibration = None


def _is_numeric(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def baseline_fallback_interpolation(
    usable: List[Dict],
    is_vertical: bool,
    logger: Optional[logging.Logger] = None,
) -> Optional[float]:
    """Fallback interpolation/extrapolation when robust calibration is unavailable."""
    log = logger or logging.getLogger(__name__)
    try:
        if is_vertical:
            coords = np.array([(l["xyxy"][1] + l["xyxy"][3]) / 2.0 for l in usable], dtype=np.float32)
        else:
            coords = np.array([(l["xyxy"][0] + l["xyxy"][2]) / 2.0 for l in usable], dtype=np.float32)

        values = np.array([float(l["cleanedvalue"]) for l in usable], dtype=np.float32)

        order = np.argsort(values)
        v = values[order]
        c = coords[order]

        signs = np.sign(v)
        flips = np.where(np.diff(signs) != 0)[0]

        if len(flips) > 0:
            i = int(flips[0])
            v1, v2, c1, c2 = float(v[i]), float(v[i + 1]), float(c[i]), float(c[i + 1])
            if abs(v2 - v1) > 1e-6:
                frac = abs(v1) / abs(v2 - v1)
                zero = float(c1 + frac * (c2 - c1))
                log.info("Fallback interpolation: baseline at %.1fpx (between %.3f and %.3f)", zero, v1, v2)
                return zero

        if v[0] > 0 and len(v) >= 2:
            denom = c[1] - c[0]
            if abs(denom) > 1e-6:
                slope = (v[1] - v[0]) / denom
                if abs(slope) > 1e-6:
                    zero = float(c[0] - v[0] / slope)
                    log.info("Fallback extrapolation (all positive): baseline at %.1fpx", zero)
                    return zero

        if v[-1] < 0 and len(v) >= 2:
            denom = c[-1] - c[-2]
            if abs(denom) > 1e-6:
                slope = (v[-1] - v[-2]) / denom
                if abs(slope) > 1e-6:
                    zero = float(c[-1] - v[-1] / slope)
                    log.info("Fallback extrapolation (all negative): baseline at %.1fpx", zero)
                    return zero

        log.warning("Fallback interpolation: no valid method found for baseline")
        return None
    except Exception as exc:  # pragma: no cover - defensive parity path
        log.error("Fallback interpolation failed: %s", exc, exc_info=True)
        return None


def baseline_from_scale_zero(
    axis_labels: List[Dict],
    is_vertical: bool,
    calibration_mode: str,
    logger: Optional[logging.Logger] = None,
    use_as_fallback_only: bool = False,
) -> Optional[float]:
    """Infer baseline from scale labels using robust calibration with fallback interpolation."""
    log = logger or logging.getLogger(__name__)

    if use_as_fallback_only:
        log.warning("Using baseline_from_scale_zero as FALLBACK (primary calibration unavailable)")

    if len(axis_labels) == 0:
        return None

    usable = [
        lbl
        for lbl in axis_labels
        if "cleanedvalue" in lbl
        and lbl["cleanedvalue"] is not None
        and validate_xyxy(lbl.get("xyxy"))
        and _is_numeric(lbl["cleanedvalue"])
    ]

    if not is_vertical and len(usable) > 3:
        high_conf_labels = [
            lbl
            for lbl in usable
            if lbl.get("ocr_confidence", 0.5) > 0.7 and lbl.get("cleanedvalue") is not None
        ]
        if len(high_conf_labels) >= 3:
            log.info(
                "Using %d/%d high-confidence X-axis labels for calibration",
                len(high_conf_labels),
                len(usable),
            )
            usable = high_conf_labels

    if not usable:
        return None

    zeros = [lbl for lbl in usable if abs(float(lbl["cleanedvalue"])) < 1e-6]
    if zeros:
        if is_vertical:
            coords = [(z["xyxy"][1] + z["xyxy"][3]) / 2.0 for z in zeros]
        else:
            coords = [(z["xyxy"][0] + z["xyxy"][2]) / 2.0 for z in zeros]
        return float(np.nanmean(coords))

    if len(usable) < 2:
        return None

    if CalibrationFactory is None:
        log.warning("CalibrationFactory not available, falling back to interpolation")
        return baseline_fallback_interpolation(usable, is_vertical, logger=log)

    try:
        axis_type = "y" if is_vertical else "x"

        mode = (calibration_mode or "prosac").lower()
        if mode == "fast":
            engine = CalibrationFactory.create("fast", use_weights=True)
        elif mode == "ransac":
            engine = CalibrationFactory.create("ransac", max_trials=300, residual_threshold=3.0)
        elif mode in ("prosac", "precise"):
            engine = CalibrationFactory.create("prosac", max_trials=500, lo_iters=2)
        elif mode == "optimized":
            engine = CalibrationFactory.create("prosac", max_trials=400)
        else:
            log.warning("Unknown calibration mode '%s', using prosac", calibration_mode)
            engine = CalibrationFactory.create("prosac")

        result = engine.calibrate(usable, axis_type)

        if result is None or result.coeffs is None:
            log.debug("Calibration returned None, falling back to interpolation")
            return baseline_fallback_interpolation(usable, is_vertical, logger=log)

        m, b = result.coeffs

        if abs(m) < 1e-6:
            log.debug("Calibration slope near zero, cannot solve for baseline")
            return baseline_fallback_interpolation(usable, is_vertical, logger=log)

        expected_negative = is_vertical
        if (m < 0) != expected_negative:
            log.warning(
                "Unexpected slope sign: m=%.4f for %s axis. Expected %s.",
                m,
                "vertical" if is_vertical else "horizontal",
                "negative" if expected_negative else "positive",
            )

        zero_coord = -b / m
        inliers = result.inliers.sum() if result.inliers is not None else len(usable)
        log.info(
            "Calibration-derived baseline: %.1fpx (R²=%.3f, slope=%.4f, intercept=%.2f, inliers=%s/%d)",
            zero_coord,
            result.r2,
            m,
            b,
            inliers,
            len(usable),
        )
        return float(zero_coord)

    except Exception as exc:  # pragma: no cover - defensive parity path
        log.warning("Calibration failed: %s, falling back to interpolation", exc)
        return baseline_fallback_interpolation(usable, is_vertical, logger=log)


__all__ = ["baseline_from_scale_zero", "baseline_fallback_interpolation"]
