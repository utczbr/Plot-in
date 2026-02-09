"""
Deprecated compatibility wrapper for Cartesian handlers.

`BaseChartHandler` is preserved for import compatibility but now delegates all
shared Cartesian orchestration logic to `CartesianExtractionHandler`.
"""

from __future__ import annotations

from handlers.base import CartesianExtractionHandler


class BaseChartHandler(CartesianExtractionHandler):
    """Compatibility shim; use CartesianExtractionHandler for new code."""

    pass


__all__ = ["BaseChartHandler"]
