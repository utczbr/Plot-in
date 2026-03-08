"""
§1.6: VLMStrategy — delegates to abstract VLMBackend for end-to-end chart understanding.

Bypasses detection, OCR, and handler stages entirely. Uses models like
UniChart, ChartVLM, or TinyChart to go directly from image to extracted data.

Behind feature flag: pipeline_mode = 'vlm'
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List

import numpy as np

from strategies.base import PipelineStrategy, StrategyServices
from services.orientation_service import Orientation

logger = logging.getLogger(__name__)


class VLMBackend(ABC):
    """Abstract backend for VLM-based chart understanding."""

    @abstractmethod
    def predict(self, image: np.ndarray, chart_type: str, prompt: str) -> Dict[str, Any]:
        """Run VLM inference and return structured extraction."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...


class VLMStrategy(PipelineStrategy):
    """
    End-to-end VLM strategy bypassing detection/OCR/handler stages.

    Requires a VLMBackend implementation to be injected. Without a backend,
    raises NotImplementedError at execution time.
    """

    STRATEGY_ID = "vlm"

    def __init__(self, backend: VLMBackend = None):
        self.backend = backend

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
    ) -> Any:
        if self.backend is None:
            raise NotImplementedError(
                "VLMStrategy requires a VLMBackend implementation. "
                "Install a VLM model (UniChart, ChartVLM, TinyChart) "
                "and configure it in advanced_settings."
            )

        from handlers.types import ExtractionResult

        prompt = f"Extract all data values from this {chart_type} chart as a structured table."

        try:
            raw_result = self.backend.predict(image, chart_type, prompt)
            elements = raw_result.get('elements', [])

            return ExtractionResult(
                chart_type=chart_type,
                elements=elements,
                diagnostics={
                    'strategy_id': self.STRATEGY_ID,
                    'vlm_model': self.backend.model_name(),
                    'value_source': 'vlm',
                },
                orientation=orientation,
            )
        except Exception as e:
            logger.error(f"VLMStrategy failed: {e}")
            return ExtractionResult.from_error(chart_type, e)
