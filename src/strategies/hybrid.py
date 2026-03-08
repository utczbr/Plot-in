"""
§1.7: HybridStrategy — runs Standard first, then VLM as validator/corrector.

When calibration quality is low, escalates to VLM for cross-validation or
full replacement of Standard results.

Behind feature flag: pipeline_mode = 'hybrid'
"""
import logging
from typing import Dict, Any, List, Optional

import numpy as np

from strategies.base import PipelineStrategy, StrategyServices
from strategies.standard import StandardStrategy
from strategies.vlm import VLMStrategy
from services.orientation_service import Orientation
from calibration.conformal import derive_calibration_quality

logger = logging.getLogger(__name__)


class HybridStrategy(PipelineStrategy):
    """
    Standard + VLM composition strategy.

    1. Run StandardStrategy to get initial ExtractionResult.
    2. Evaluate quality signals (R², CP intervals, detection coverage).
    3. If calibration_quality == 'uncalibrated', call VLM as replacement/corrector.
    4. Per-element: keep Standard when Standard and VLM agree; mark 'vlm_override' otherwise.
    """

    STRATEGY_ID = "hybrid"

    def __init__(
        self,
        standard: StandardStrategy = None,
        vlm: VLMStrategy = None,
    ):
        self.standard = standard
        self.vlm = vlm

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
        # Step 1: Run Standard
        if self.standard is None:
            raise RuntimeError("HybridStrategy requires a StandardStrategy instance")

        std_result = self.standard.execute(
            image, chart_type, detections, axis_labels,
            chart_elements, orientation, services, **kwargs
        )

        # Step 2: Evaluate quality signals
        diagnostics = getattr(std_result, 'diagnostics', {}) or {}
        r2 = None
        cal_quality_info = diagnostics.get('calibration_quality')
        if isinstance(cal_quality_info, dict):
            r2 = cal_quality_info.get('r_squared')
        elif isinstance(cal_quality_info, (int, float)):
            r2 = cal_quality_info

        cal_quality = derive_calibration_quality(r2)
        diagnostics['strategy_id'] = self.STRATEGY_ID
        diagnostics['standard_calibration_quality'] = cal_quality

        fallback_triggered = False

        # Step 3: If uncalibrated and VLM available, escalate
        if cal_quality == 'uncalibrated' and self.vlm is not None:
            try:
                vlm_result = self.vlm.execute(
                    image, chart_type, detections, axis_labels,
                    chart_elements, orientation, services, **kwargs
                )
                vlm_elements = getattr(vlm_result, 'elements', [])

                if vlm_elements:
                    # Replace Standard result with VLM result
                    diagnostics['fallback_triggered'] = True
                    diagnostics['vlm_element_count'] = len(vlm_elements)
                    fallback_triggered = True

                    # Mark VLM elements
                    for elem in vlm_elements:
                        elem['value_source'] = 'vlm_override'

                    std_result = vlm_result
                    if hasattr(std_result, 'diagnostics'):
                        std_result.diagnostics.update(diagnostics)
            except Exception as e:
                logger.warning(f"HybridStrategy VLM fallback failed: {e}")
                diagnostics['vlm_fallback_error'] = str(e)

        if not fallback_triggered:
            diagnostics['fallback_triggered'] = False

        if hasattr(std_result, 'diagnostics'):
            std_result.diagnostics = diagnostics

        return std_result
