"""
Base handler with backward compatibility for existing handlers and new hierarchical architecture.

This module now serves as a compatibility adapter, re-exporting symbols from:
- handlers.types
- handlers.base
- handlers.legacy
"""

from __future__ import annotations

# Re-export types
from handlers.types import (
    ChartCoordinateSystem,
    ExtractionResult,
    OldExtractionResult
)

# Re-export New Architecture
from handlers.base import (
    BaseHandler,
    CartesianChartHandler,
    GridChartHandler,
    PolarChartHandler
)

# Re-export Legacy Architecture
from handlers.legacy import BaseChartHandler

__all__ = [
    'ChartCoordinateSystem',
    'ExtractionResult',
    'OldExtractionResult',
    'BaseHandler',
    'CartesianChartHandler',
    'GridChartHandler',
    'PolarChartHandler',
    'BaseChartHandler'
]