"""Baseline detector orchestration built from decomposed modules."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.enums import ChartType
from services.orientation_service import Orientation, OrientationService
from .clustering import cluster_bars_by_axis

from .geometry import aggregate_stack_near_ends, validate_xyxy
from .policy import axis_id_single, chart_type_constraints
from .scatter import scatter_axis_baseline
from .types import BaselineLine, BaselineResult, DetectorConfig
from .zero_crossing import baseline_fallback_interpolation, baseline_from_scale_zero

try:
    from calibration.calibration_base import CalibrationResult, BaseCalibration
except Exception:  # pragma: no cover - optional import in some environments
    CalibrationResult = Any  # type: ignore
    BaseCalibration = None


class ModularBaselineDetector:
    """Modular, multi-strategy baseline detection with diagnostics."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.info("ModularBaselineDetector initialized with config: %s", self.cfg)

    def _chart_type_constraints(self, chart_type: ChartType) -> Dict[str, Any]:
        """Get constraints for chart type."""
        return chart_type_constraints(chart_type)

    def _detect_single_axis(
        self,
        img: np.ndarray,
        chart_elements: List[Dict],
        axis_labels: Optional[List[Dict]],
        orientation: Orientation,
        chart_type: ChartType,
        image_size: Optional[Tuple[int, int]] = None,
        primary_calibration_zero: Optional[float] = None,
        primary_calibration_result: Optional[CalibrationResult] = None,
    ) -> float:
        """Compute one baseline value for one axis partition."""
        result = self.detect(
            img=img,
            chart_elements=chart_elements,
            axis_labels=axis_labels,
            orientation=orientation,
            chart_type=chart_type,
            image_size=image_size,
            primary_calibration_zero=primary_calibration_zero,
            primary_calibration_result=primary_calibration_result,
            secondary_axis_labels=None,
            dual_axis_info=None,
        )
        if result.baselines:
            return float(result.baselines[0].value)

        self.logger.warning("Single-axis baseline detection returned no baselines; using 0.0 fallback")
        return 0.0

    def detect(
        self,
        img: np.ndarray,
        chart_elements: List[Dict],
        axis_labels: Optional[List[Dict]] = None,
        orientation: Orientation = Orientation.VERTICAL,
        chart_type: ChartType = ChartType.BAR,
        image_size: Optional[Tuple[int, int]] = None,
        primary_calibration_zero: Optional[float] = None,
        primary_calibration_result: Optional[CalibrationResult] = None,
        secondary_axis_labels: Optional[List[Dict]] = None,
        dual_axis_info: Optional[Dict] = None,
    ) -> BaselineResult:
        try:
            if image_size is None:
                h, w = img.shape[:2]
            else:
                w, h = image_size

            orientation = OrientationService.from_any(orientation)
            if not isinstance(chart_type, ChartType):
                chart_type = ChartType(str(chart_type).lower())

            is_vertical = orientation == Orientation.VERTICAL
            _ = self._chart_type_constraints(chart_type)

            if (
                dual_axis_info
                and dual_axis_info.get("has_dual_axis", False)
                and secondary_axis_labels is not None
                and len(secondary_axis_labels) >= 2
            ):
                self.logger.info("Dual-axis chart detected: computing two baselines...")
                bar_clusters = cluster_bars_by_axis(chart_elements, w, dual_axis_info)

                primary_baseline = self._detect_single_axis(
                    img,
                    bar_clusters["primary_bars"],
                    axis_labels,
                    orientation,
                    chart_type,
                    image_size,
                    primary_calibration_zero=primary_calibration_zero,
                    primary_calibration_result=primary_calibration_result,
                )

                secondary_baseline = self._detect_single_axis(
                    img,
                    bar_clusters["secondary_bars"],
                    secondary_axis_labels,
                    orientation,
                    chart_type,
                    image_size,
                )

                if orientation == Orientation.VERTICAL:
                    baseline_lines = [
                        BaselineLine(axis_id="y1", value=primary_baseline, confidence=0.8, orientation=orientation),
                        BaselineLine(axis_id="y2", value=secondary_baseline, confidence=0.8, orientation=orientation),
                    ]
                else:
                    baseline_lines = [
                        BaselineLine(axis_id="x1", value=primary_baseline, confidence=0.8, orientation=orientation),
                        BaselineLine(axis_id="x2", value=secondary_baseline, confidence=0.8, orientation=orientation),
                    ]

                return BaselineResult(
                    baselines=baseline_lines,
                    method="dual_axis_clustering",
                    diagnostics={
                        "dual_axis": True,
                        "threshold_x": bar_clusters["threshold_x"],
                        "primary_elements_count": len(bar_clusters["primary_bars"]),
                        "secondary_elements_count": len(bar_clusters["secondary_bars"]),
                    },
                )

            if (
                orientation == Orientation.HORIZONTAL
                and primary_calibration_zero is not None
                and primary_calibration_result is not None
            ):
                baseline_value = primary_calibration_zero
                self.logger.info(
                    "✓ Using primary calibration zero for horizontal baseline: %.2fpx (skipping local recalibration)",
                    baseline_value,
                )

                inverted_axis = False
                coeffs = getattr(primary_calibration_result, "coeffs", None)
                if coeffs:
                    m_primary = coeffs[0]
                    inverted_axis = m_primary < 0.0
                    if inverted_axis:
                        self.logger.warning("⚠️ INVERTED X-axis: primary slope=%.6f < 0", m_primary)

                return BaselineResult(
                    baselines=[
                        BaselineLine(
                            axis_id="x",
                            orientation=orientation,
                            value=baseline_value,
                            confidence=0.9,
                        )
                    ],
                    method="primary_calibration_zero",
                    diagnostics={
                        "primary_zero_used": True,
                        "inverted_axis": inverted_axis,
                        "source": "analysis.py primary calibration",
                    },
                )

            if chart_type == ChartType.SCATTER:
                y_b = self._scatter_axis_baseline(img, axis_labels, axis="y")
                x_b = self._scatter_axis_baseline(img, axis_labels, axis="x")
                bl: List[BaselineLine] = []
                if y_b is not None:
                    bl.append(BaselineLine(axis_id="y", orientation=orientation, value=float(y_b), confidence=0.7))
                if x_b is not None:
                    bl.append(BaselineLine(axis_id="x", orientation=orientation, value=float(x_b), confidence=0.7))
                return BaselineResult(baselines=bl, method="scatter_dual", diagnostics={"chart_type": chart_type.value})

            if not chart_elements:
                self.logger.warning("No chart elements provided")
                return BaselineResult(baselines=[], method="no_elements", diagnostics={"reason": "no_chart_elements"})

            inverted_axis = False
            if primary_calibration_result is not None and hasattr(primary_calibration_result, "coeffs"):
                coeffs = getattr(primary_calibration_result, "coeffs", None)
                if coeffs:
                    slope = coeffs[0]
                    if orientation == Orientation.HORIZONTAL:
                        inverted_axis = slope < 0.0
                    else:
                        inverted_axis = slope > 0.0
                    if inverted_axis:
                        self.logger.info("Detected INVERTED axis from primary calibration: slope=%.4f", slope)
            elif self.cfg.calibration_mode != "none" and axis_labels and BaseCalibration is not None:
                try:
                    coords, vals, _ = BaseCalibration._extract_points(
                        axis_labels,
                        "x" if orientation == Orientation.HORIZONTAL else "y",
                        prefer_cleaned=True,
                    )
                    if len(coords) >= 2:
                        A = np.vstack([coords, np.ones_like(coords)]).T
                        sol = np.linalg.lstsq(A, vals, rcond=None)[0]
                        slope = float(sol[0])

                        if orientation == Orientation.HORIZONTAL:
                            inverted_axis = slope < 0.0
                        else:
                            inverted_axis = slope > 0.0

                        if inverted_axis:
                            self.logger.info("Detected INVERTED axis: slope=%.4f for %s", slope, orientation.value)
                except Exception as exc:  # pragma: no cover - defensive parity path
                    self.logger.debug("Could not detect axis inversion: %s", exc)
                    inverted_axis = False

            if primary_calibration_result is not None and hasattr(primary_calibration_result, "func"):
                zero_baseline = primary_calibration_zero
                self.logger.info(
                    "✅ Using PRIMARY calibration zero: %.1fpx (R²=%.4f)",
                    zero_baseline,
                    getattr(primary_calibration_result, "r2", 0.0),
                )
            else:
                zero_baseline = (
                    self._baseline_from_scale_zero(
                        axis_labels or [],
                        is_vertical,
                        use_as_fallback_only=True,
                    )
                    if self.cfg.calibration_mode != "none"
                    else None
                )

            agg_near = aggregate_stack_near_ends(
                chart_elements,
                orientation,
                img_h=h,
                band_frac=self.cfg.stack_band_frac,
                inverted_axis=inverted_axis,
            )

            if agg_near.size >= 2:
                if orientation == Orientation.HORIZONTAL:
                    pct = 5.0 if not inverted_axis else 95.0
                    single_est = float(np.nanpercentile(agg_near, pct))
                else:
                    pct = 95.0 if not inverted_axis else 5.0
                    single_est = float(np.nanpercentile(agg_near, pct))

                self.logger.info(
                    "Stack-aware baseline: %.1fpx (pct=%.0f, inverted=%s, n_bands=%d)",
                    single_est,
                    pct,
                    inverted_axis,
                    len(agg_near),
                )
            else:
                self.logger.warning("Insufficient elements for stack aggregation, using simple median")
                if is_vertical:
                    coords_raw = [
                        max(el["xyxy"][1], el["xyxy"][3])
                        for el in chart_elements
                        if validate_xyxy(el.get("xyxy"))
                    ]
                else:
                    coords_raw = [
                        min(el["xyxy"][0], el["xyxy"][2])
                        for el in chart_elements
                        if validate_xyxy(el.get("xyxy"))
                    ]
                    if len(chart_elements) == 1 and axis_labels and not is_vertical:
                        label_x_coords = [
                            (lbl["xyxy"][0] + lbl["xyxy"][2]) / 2.0
                            for lbl in axis_labels
                            if "xyxy" in lbl and validate_xyxy(lbl.get("xyxy"))
                        ]
                        if label_x_coords:
                            baseline_hint = min(label_x_coords) - 20
                            self.logger.info(
                                "Single horizontal bar: using label-based baseline hint %.1fpx",
                                baseline_hint,
                            )
                            coords_raw.append(baseline_hint)
                pct = None
                single_est = float(np.nanmedian(coords_raw)) if coords_raw else 0.0

            baseline_value = single_est
            if zero_baseline is not None and not np.isnan(zero_baseline):
                if orientation == Orientation.HORIZONTAL:
                    self.logger.info(
                        "Using calibrated X-axis baseline for horizontal chart: %.1fpx (overriding statistical estimate %.1fpx)",
                        zero_baseline,
                        single_est,
                    )
                    baseline_value = float(zero_baseline)
                else:
                    snap_tol = 10.0
                    if abs(baseline_value - float(zero_baseline)) <= snap_tol:
                        self.logger.info("Snapping baseline %.1f -> %.1f (calibration zero)", baseline_value, zero_baseline)
                        baseline_value = float(zero_baseline)
                    else:
                        self.logger.warning(
                            "Baseline %.1f differs from calibration zero %.1f by %.1fpx (> %.1fpx tolerance)",
                            baseline_value,
                            zero_baseline,
                            abs(baseline_value - zero_baseline),
                            snap_tol,
                        )
                        self.logger.warning(
                            "Vertical chart has significant baseline discrepancy. Consider checking axis labels or calibration quality."
                        )

            return BaselineResult(
                baselines=[
                    BaselineLine(
                        axis_id=axis_id_single(orientation),
                        orientation=orientation,
                        value=baseline_value,
                        confidence=0.8,
                    )
                ],
                method=f"single_stackaware_{self.cfg.cluster_backend}",
                diagnostics={
                    "inverted_axis": inverted_axis,
                    "n_bands": int(len(agg_near)),
                    "percentile_used": pct,
                    "calibration_zero": float(zero_baseline) if zero_baseline is not None else None,
                },
            )

        except Exception as exc:  # pragma: no cover - defensive parity path
            self.logger.error("Baseline detection failed: %s", exc, exc_info=True)
            return BaselineResult(baselines=[], method="error", diagnostics={"error": str(exc)})

    def _scatter_axis_baseline(
        self,
        img: np.ndarray,
        axis_labels: Optional[List[Dict]],
        axis: str,
    ) -> Optional[float]:
        """Backward-compatible wrapper for scatter baseline estimation."""
        return scatter_axis_baseline(img, axis_labels, axis)

    def _baseline_from_scale_zero(
        self,
        axis_labels: List[Dict],
        is_vertical: bool,
        use_as_fallback_only: bool = False,
    ) -> Optional[float]:
        """Backward-compatible wrapper for zero-crossing baseline estimation."""
        return baseline_from_scale_zero(
            axis_labels=axis_labels,
            is_vertical=is_vertical,
            calibration_mode=self.cfg.calibration_mode,
            logger=self.logger,
            use_as_fallback_only=use_as_fallback_only,
        )

    def _baseline_fallback_interpolation(
        self,
        usable: List[Dict],
        is_vertical: bool,
    ) -> Optional[float]:
        """Backward-compatible wrapper for interpolation fallback."""
        return baseline_fallback_interpolation(usable, is_vertical, logger=self.logger)


def detect_baselines(
    img: np.ndarray,
    chart_elements: List[Dict],
    axis_labels: Optional[List[Dict]] = None,
    orientation: Orientation = Orientation.VERTICAL,
    chart_type: ChartType = ChartType.BAR,
    image_size: Optional[Tuple[int, int]] = None,
    config: Optional[DetectorConfig] = None,
) -> BaselineResult:
    """Convenience function for baseline detection."""
    detector = ModularBaselineDetector(config=config)
    return detector.detect(
        img=img,
        chart_elements=chart_elements,
        axis_labels=axis_labels,
        orientation=orientation,
        chart_type=chart_type,
        image_size=image_size,
    )


__all__ = ["ModularBaselineDetector", "detect_baselines"]
