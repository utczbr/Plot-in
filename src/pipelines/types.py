"""
Typed pipeline output contracts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ProtocolRow(TypedDict, total=False):
    """A single protocol-level data row derived from a chart element."""
    source_file: str
    page_index: Optional[int]
    chart_type: str
    element_index: int
    series_id: str
    group: str
    outcome: str
    value: Optional[float]
    unit: str
    error_bar_type: str
    error_bar_value: Optional[float]
    baseline_value: Optional[float]
    confidence: Optional[float]
    review_status: str
    notes: str
    _original: Optional[Dict[str, Any]]


class PipelineResult(TypedDict, total=False):
    image_file: str
    chart_type: str
    orientation: str
    elements: List[Dict[str, Any]]
    calibration: Dict[str, Any]
    baselines: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    detections: Dict[str, Any]
    _provenance: Optional[Dict[str, Any]]
    protocol_rows: List[Dict[str, Any]]
