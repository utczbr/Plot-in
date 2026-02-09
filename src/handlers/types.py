"""
Common types, enums and data classes for chart handlers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, TYPE_CHECKING

from services.orientation_service import Orientation

if TYPE_CHECKING:
    import numpy as np


class ChartCoordinateSystem(Enum):
    """Defines the coordinate system used by a chart type."""
    CARTESIAN = "cartesian"      # Standard X/Y axes (bar, line, scatter, box, histogram)
    POLAR = "polar"              # Radial/angular (pie, radar)
    GRID = "grid"                # Row/column matrix (heatmap)
    HIERARCHICAL = "hierarchical" # Tree-based (treemap, sunburst)


@dataclass
class ExtractionResult:
    """
    Standardized output from any chart handler (New Format).

    Attributes:
        chart_type: Type of chart processed.
        coordinate_system: The coordinate system of the chart.
        elements: List of extracted data points with their values.
        calibration: Calibration metadata (empty dict for non-Cartesian charts).
        baselines: Detected baseline information (empty for non-Cartesian).
        diagnostics: Processing metadata and quality metrics.
        errors: List of error messages encountered during processing.
        warnings: List of non-fatal warnings.
        orientation: Chart orientation (VERTICAL/HORIZONTAL/NOT_APPLICABLE).
    """
    chart_type: str
    coordinate_system: ChartCoordinateSystem
    elements: List[Dict[str, Any]] = field(default_factory=list)
    calibration: Dict[str, Any] = field(default_factory=dict)
    baselines: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    orientation: Orientation = Orientation.VERTICAL

    @classmethod
    def from_error(cls, chart_type: str, error: Exception) -> ExtractionResult:
        """Creates an ExtractionResult representing a failure."""
        return cls(
            chart_type=chart_type,
            coordinate_system=ChartCoordinateSystem.CARTESIAN,  # Default fallback
            errors=[f"{type(error).__name__}: {str(error)}"]
        )


@dataclass(frozen=True)
class HandlerContext:
    """
    Canonical handler input contract enforced at orchestrator boundaries.

    Attributes:
        image: Source image in BGR format.
        chart_type: Normalized chart type string.
        detections: Raw detector outputs grouped by class name.
        axis_labels: Axis label detections (with OCR text when available).
        chart_elements: Primary chart elements for the specific chart type.
        orientation: Normalized orientation enum.
    """
    image: "np.ndarray"
    chart_type: str
    detections: Dict[str, Any]
    axis_labels: List[Dict[str, Any]]
    chart_elements: List[Dict[str, Any]]
    orientation: Orientation


@dataclass
class OldExtractionResult:
    """Structured result from chart extraction (old format)."""
    chart_type: str
    orientation: str
    elements: List[Dict]
    baselines: Dict[str, Any]
    calibration: Dict[str, Any]
    diagnostics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
