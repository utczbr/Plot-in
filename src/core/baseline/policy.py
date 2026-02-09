"""Policy helpers for baseline detection."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from core.enums import ChartType
from services.orientation_service import Orientation


def chart_type_constraints(chart_type: ChartType) -> Dict[str, Any]:
    """Get baseline-policy constraints by chart type."""
    if chart_type == ChartType.SCATTER:
        return {"force_dual": True, "allow_dual": True}
    if chart_type in (ChartType.LINE, ChartType.AREA, ChartType.HISTOGRAM):
        return {"allow_dual": False}
    if chart_type in (ChartType.BAR, ChartType.BOX):
        return {"allow_dual": True}
    return {"allow_dual": False}


def axis_id_map_for_dual(orientation: Orientation) -> Tuple[str, str]:
    """Map ordered cluster IDs to axis identifiers."""
    if orientation == Orientation.VERTICAL:
        return "y1", "y2"
    return "x1", "x2"


def axis_id_single(orientation: Orientation) -> str:
    """Get axis identifier for single-axis case."""
    return "y" if orientation == Orientation.VERTICAL else "x"


__all__ = [
    "chart_type_constraints",
    "axis_id_map_for_dual",
    "axis_id_single",
]
