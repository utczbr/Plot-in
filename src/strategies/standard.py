"""
§1.4: StandardStrategy — wraps existing ChartAnalysisOrchestrator pipeline.

Zero behavioral change from current pipeline. Only adds diagnostics keys:
- diagnostics['strategy_id'] = 'standard'
- diagnostics['value_source'] = 'calibrated_geometry'
"""
import logging
from typing import Dict, Any, List

import numpy as np

from strategies.base import PipelineStrategy, StrategyServices
from services.orientation_service import Orientation

logger = logging.getLogger(__name__)


class StandardStrategy(PipelineStrategy):
    """Wraps the existing orchestration path through ChartAnalysisOrchestrator."""

    STRATEGY_ID = "standard"

    def __init__(self, orchestrator=None):
        """
        Args:
            orchestrator: ChartAnalysisOrchestrator instance (injected at pipeline init).
        """
        self.orchestrator = orchestrator

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
        """
        Delegate to the existing ChartAnalysisOrchestrator.process_chart().

        The orchestrator handles handler lookup, calibration, baseline detection,
        value extraction, and result formatting — exactly as the current pipeline does.
        """
        if self.orchestrator is None:
            raise RuntimeError("StandardStrategy requires an orchestrator instance")

        result = self.orchestrator.process_chart(
            image=image,
            chart_type=chart_type,
            detections=detections,
            axis_labels=axis_labels,
            chart_elements=chart_elements,
            orientation=orientation,
            **kwargs,
        )

        # Add strategy diagnostics (additive only)
        if hasattr(result, 'diagnostics') and isinstance(result.diagnostics, dict):
            result.diagnostics['strategy_id'] = self.STRATEGY_ID
            result.diagnostics['value_source'] = 'calibrated_geometry'

        return result
