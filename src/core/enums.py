from enum import Enum


class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    BOX = "box"
    PIE = "pie"
    AREA = "area"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    UNKNOWN = "unknown"


class Orientation(str, Enum):
    """Type-safe orientation enum."""
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    NOT_APPLICABLE = "not_applicable"

