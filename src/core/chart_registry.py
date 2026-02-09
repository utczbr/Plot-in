"""
Shared chart-type registry and canonical element-key mapping.
"""

from __future__ import annotations

from typing import Any

from core.enums import ChartType


CHART_ELEMENT_KEY_MAP = {
    ChartType.BAR.value: "bar",
    ChartType.BOX.value: "box",
    ChartType.LINE.value: "data_point",
    ChartType.SCATTER.value: "data_point",
    ChartType.HISTOGRAM.value: "bar",
    ChartType.HEATMAP.value: "cell",
    ChartType.PIE.value: "slice",
}


def normalize_chart_type(value: Any, default: str = ChartType.BAR.value) -> str:
    """
    Normalize raw chart type values (string/enum) to the canonical lower-case value.

    Notes:
    - The classification model can emit a generic 'chart' class; this is normalized
      to the configured default.
    - Unknown values safely fallback to `default`.
    """
    if isinstance(value, ChartType):
        return value.value

    if value is None:
        return default

    normalized = str(value).strip().lower()
    if normalized in ("", "chart", "unknown"):
        return default

    supported = {ct.value for ct in ChartType}
    return normalized if normalized in supported else default


def get_chart_element_key(chart_type: Any, default: str = "bar") -> str:
    """Resolve the primary element key expected for a chart type."""
    normalized = normalize_chart_type(chart_type)
    return CHART_ELEMENT_KEY_MAP.get(normalized, default)
