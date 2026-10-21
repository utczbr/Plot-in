"""
Calibration factory with parameter validation and deterministic estimator routing.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Set
import warnings

from .calibration_base import BaseCalibration
from .calibration_fast import FastCalibration
from .calibration_siegel import RobustSiegelCalibration
from .visual_tick_detector import VisualTickCalibration

try:
    from .calibration_neural import NeuralCalibration, LogCalibration
    _NEURAL_AVAILABLE = True
except (ImportError, OSError, Exception):  # noqa: BLE001
    # PyTorch not installed or its DLLs failed to load (common on Windows
    # without the right MSVC Redistributable or CUDA toolkit).
    NeuralCalibration = None  # type: ignore[assignment,misc]
    LogCalibration = None     # type: ignore[assignment,misc]
    _NEURAL_AVAILABLE = False

logger = logging.getLogger(__name__)


class CalibrationFactory:
    """Factory for creating calibration engines with strict validation."""

    _LEGACY_STOCHASTIC_PARAMS: Set[str] = {
        'max_trials',
        'random_state',
        'early_termination_ratio',
        'lo_iters',
        'prosac_growth',
        'convergence_threshold',
    }

    _SIEGEL_PARAMS: Set[str] = {
        'residual_threshold',
        'min_inliers',
        'detect_log',
    }

    _VALID_PARAMS: Dict[str, Set[str]] = {
        'fast': {'use_weights'},
        'siegel': _SIEGEL_PARAMS | _LEGACY_STOCHASTIC_PARAMS,
        'visual': {'edge_threshold', 'min_tick_spacing', 'max_tick_spacing', 'min_ticks'},
        'neural': {'hidden_dim', 'max_epochs', 'learning_rate', 'early_stop_patience'},
        'log': set(),  # No parameters
    }

    @staticmethod
    def create(engine_type: str, **kwargs: Any) -> BaseCalibration:
        """
        Create calibration engine with strict parameter validation.

        Args:
            engine_type: One of 'linear', 'fast', 'ransac', 'adaptive', 'prosac', 'precise',
                         'siegel', 'robust', 'visual', 'neural', 'log'
            **kwargs: Engine-specific parameters

        Returns:
            Calibration engine instance

        Raises:
            ValueError: For unknown engine type or invalid parameters
        """
        engine_type_lower = engine_type.lower()

        # Normalize aliases
        if engine_type_lower in ('linear', 'fast'):
            normalized = 'fast'
        elif engine_type_lower in ('ransac', 'adaptive', 'prosac', 'precise', 'siegel', 'robust', 'siegel (robust)'):
            normalized = 'siegel'
        elif engine_type_lower in ('visual', 'tick', 'visual_tick'):
            normalized = 'visual'
        elif engine_type_lower in ('neural', 'adaptive_neural', 'mlp'):
            normalized = 'neural'
        elif engine_type_lower in ('log', 'logarithmic'):
            normalized = 'log'
        else:
            normalized = engine_type_lower

        if normalized not in CalibrationFactory._VALID_PARAMS:
            known = ['linear', 'fast', 'siegel', 'robust', 'prosac', 'ransac', 'visual', 'neural', 'log']
            raise ValueError(f"Unknown calibration engine: {engine_type}. Valid types: {known}")

        # Validate parameters
        valid_params = CalibrationFactory._VALID_PARAMS[normalized]
        invalid_params = set(kwargs.keys()) - valid_params

        if invalid_params:
            raise ValueError(
                f"Invalid parameters for {engine_type_lower}: {invalid_params}. "
                f"Valid parameters: {valid_params}"
            )

        # Handle legacy stochastic parameters for Siegel-routed engines
        if normalized == 'siegel':
            active_kwargs = dict(kwargs)
            for legacy_param in CalibrationFactory._LEGACY_STOCHASTIC_PARAMS:
                if legacy_param in active_kwargs:
                    active_kwargs.pop(legacy_param)
                    warnings.warn(
                        f"stochastic-sampling parameter '{legacy_param}' was ignored; "
                        "using deterministic RobustSiegelCalibration",
                        DeprecationWarning,
                        stacklevel=2,
                    )
            return RobustSiegelCalibration(**active_kwargs)

        if normalized == 'fast':
            return FastCalibration(**kwargs)

        if normalized == 'visual':
            return VisualTickCalibration(**kwargs)

        if normalized == 'neural':
            if not _NEURAL_AVAILABLE or NeuralCalibration is None:
                logger.warning(
                    "NeuralCalibration requested but PyTorch is not available "
                    "(missing or failed to load). Falling back to FastCalibration."
                )
                return FastCalibration()
            return NeuralCalibration(**kwargs)

        if normalized == 'log':
            if not _NEURAL_AVAILABLE or LogCalibration is None:
                logger.warning(
                    "LogCalibration requested but PyTorch is not available. "
                    "Falling back to FastCalibration."
                )
                return FastCalibration()
            return LogCalibration(**kwargs)

        raise ValueError(f"Engine creation not implemented for: {normalized}")