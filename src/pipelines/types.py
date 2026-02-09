"""
Typed pipeline output contracts.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class PipelineResult(TypedDict):
    image_file: str
    chart_type: str
    orientation: str
    elements: List[Dict[str, Any]]
    calibration: Dict[str, Any]
    baselines: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    detections: Dict[str, Any]
