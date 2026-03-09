"""
§1.8: StrategyRouter — selects PipelineStrategy based on pipeline_mode and quality signals.

Default pipeline_mode='standard' ensures zero behavioral change until explicitly opted in.
"""
import logging
from typing import Optional

from strategies.base import PipelineStrategy, StrategyServices
from strategies.standard import StandardStrategy

logger = logging.getLogger(__name__)


class StrategyRouter:
    """
    §1.8: Route to the appropriate PipelineStrategy based on configuration
    and runtime quality signals.

    Routing policy:
    - 'standard' (default) → StandardStrategy
    - 'vlm' → VLMStrategy
    - 'chart_to_table' → ChartToTableStrategy
    - 'hybrid' → HybridStrategy
    - 'auto' → quality-signal-based selection
    """

    def __init__(
        self,
        standard: StandardStrategy = None,
        vlm=None,
        chart_to_table=None,
        hybrid=None,
    ):
        self._strategies = {
            'standard': standard,
            'vlm': vlm,
            'chart_to_table': chart_to_table,
            'hybrid': hybrid,
        }

    def select(
        self,
        chart_type: str,
        classification_confidence: float = 1.0,
        detection_coverage: float = 1.0,
        calibration_quality: Optional[str] = None,
        pipeline_mode: str = 'standard',
    ) -> PipelineStrategy:
        """
        Select strategy based on mode and quality signals.

        Args:
            chart_type: Classified chart type.
            classification_confidence: Confidence from chart classifier (0-1).
            detection_coverage: Fraction of expected elements detected (0-1).
            calibration_quality: 'high', 'approximate', 'uncalibrated', or None.
            pipeline_mode: From advanced_settings (default 'standard').

        Returns:
            Selected PipelineStrategy instance.

        Raises:
            ValueError: If requested mode is unavailable.
        """
        # Explicit mode selection
        if pipeline_mode != 'auto':
            strategy = self._strategies.get(pipeline_mode)
            if strategy is None:
                if pipeline_mode == 'standard':
                    raise ValueError(
                        "StandardStrategy not configured. "
                        "Pass orchestrator to StrategyRouter."
                    )
                # §1.8 contract: explicit non-standard modes MUST raise when
                # the backend is unavailable.  The pipeline caller is
                # responsible for catching this and returning a clear error.
                raise ValueError(
                    f"Requested pipeline_mode='{pipeline_mode}' is not available. "
                    f"The '{pipeline_mode}' backend was not loaded or is missing "
                    f"required weights/dependencies."
                )
            return strategy

        # Auto mode: quality-signal-based selection
        # Low classification confidence + sparse detections → VLM or ChartToTable
        if (classification_confidence < 0.4 and detection_coverage < 0.3):
            if self._strategies.get('vlm') is not None:
                logger.info(
                    f"Auto-routing to VLM: low conf={classification_confidence:.2f}, "
                    f"sparse detections={detection_coverage:.2f}"
                )
                return self._strategies['vlm']
            if self._strategies.get('chart_to_table') is not None:
                logger.info("Auto-routing to chart_to_table: VLM unavailable")
                return self._strategies['chart_to_table']

        # High detection but low calibration → Hybrid
        if (calibration_quality == 'uncalibrated'
                and detection_coverage >= 0.5
                and self._strategies.get('hybrid') is not None):
            logger.info(
                f"Auto-routing to hybrid: uncalibrated, "
                f"detection_coverage={detection_coverage:.2f}"
            )
            return self._strategies['hybrid']

        # Default: Standard
        strategy = self._strategies.get('standard')
        if strategy is None:
            raise ValueError("StandardStrategy not configured")

        logger.debug("Auto-routing to standard: all signals acceptable")
        return strategy
