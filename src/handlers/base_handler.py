"""
Base handler pass-through module re-exporting symbols from handlers.types and handlers.base.
"""

from __future__ import annotations

# Re-export types
from handlers.types import (
    ChartCoordinateSystem,
    ExtractionResult,
    OldExtractionResult,
    HandlerContext,
)

# Re-export base classes
from handlers.base import (
    BaseHandler,
    CartesianChartHandler,
    CartesianExtractionHandler,
    GridChartHandler,
    PolarChartHandler,
)

__all__ = [
    'ChartCoordinateSystem',
    'ExtractionResult',
    'OldExtractionResult',
    'HandlerContext',
    'BaseHandler',
    'CartesianChartHandler',
    'CartesianExtractionHandler',
    'GridChartHandler',
    'PolarChartHandler',
]
