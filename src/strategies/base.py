"""
§1.3: PipelineStrategy ABC and StrategyServices dataclass.

All strategies receive the same inputs and return ExtractionResult.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np

from services.orientation_service import Orientation


@dataclass(frozen=True)
class StrategyServices:
    """
    Frozen bundle of service references currently owned by ChartAnalysisOrchestrator.

    Passed to strategies so they can access calibration, spatial classification,
    dual-axis, meta-clustering, color mapping, and legend matching services
    without taking ownership of the orchestrator itself.
    """
    calibration_service: Any = None
    spatial_classifier: Any = None
    dual_axis_service: Any = None
    meta_clustering_service: Any = None
    color_mapping_service: Any = None
    legend_matching_service: Any = None


class PipelineStrategy(ABC):
    """
    §1.3: Abstract base for pipeline execution strategies.

    Strategies bypass different pipeline stages depending on their approach:
    - StandardStrategy: uses all stages (detection, OCR, calibration, handler)
    - VLMStrategy: bypasses detection, OCR, handler (end-to-end VLM)
    - ChartToTableStrategy: bypasses detection, OCR, handler (Pix2Struct)
    - HybridStrategy: runs Standard + VLM, merges results
    """

    STRATEGY_ID: str = "base"

    @abstractmethod
    def execute(
        self,
        image: np.ndarray,
        chart_type: str,
        detections: Dict[str, Any],
        axis_labels: List[Dict[str, Any]],
        chart_elements: List[Dict[str, Any]],
        orientation: Orientation,
        services: StrategyServices,
        **kwargs,
    ) -> Any:  # Returns ExtractionResult (avoid circular import)
        """
        Execute the strategy and return an ExtractionResult.

        Args:
            image: Chart image (BGR numpy array).
            chart_type: Classified chart type string.
            detections: Detection results from YOLO.
            axis_labels: OCR-recognized axis labels.
            chart_elements: Detected chart elements.
            orientation: Detected chart orientation.
            services: Bundle of shared services.

        Returns:
            ExtractionResult with extracted data and diagnostics.
        """
        ...
