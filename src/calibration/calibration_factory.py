"""
Enhanced factory with strict parameter validation.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from .calibration_base import BaseCalibration
from .calibration_fast import FastCalibration
from .calibration_adaptive import RANSACCalibration
from .calibration_precise import PROSACCalibration
from .visual_tick_detector import VisualTickCalibration
from .calibration_neural import NeuralCalibration, LogCalibration

logger = logging.getLogger(__name__)


class CalibrationFactory:
    """Factory for creating calibration engines with strict validation."""
    
    _VALID_PARAMS = {
        'linear': {'use_weights'},
        'fast': {'use_weights'},
        'ransac': {'max_trials', 'residual_threshold', 'min_inliers', 'random_state', 'early_termination_ratio'},
        'adaptive': {'max_trials', 'residual_threshold', 'min_inliers', 'random_state', 'early_termination_ratio'},
        'prosac': {
            'max_trials', 'residual_threshold', 'min_inliers', 'random_state',
            'lo_iters', 'prosac_growth', 'early_termination_ratio', 'convergence_threshold'
        },
        'precise': {
            'max_trials', 'residual_threshold', 'min_inliers', 'random_state',
            'lo_iters', 'prosac_growth', 'early_termination_ratio', 'convergence_threshold'
        },
        'visual': {'edge_threshold', 'min_tick_spacing', 'max_tick_spacing', 'min_ticks'},
        'neural': {'hidden_dim', 'max_epochs', 'learning_rate', 'early_stop_patience'},
        'log': set(),  # No parameters
    }
    
    @staticmethod
    def create(engine_type: str, **kwargs: Any) -> BaseCalibration:
        """
        Create calibration engine with strict parameter validation.
        
        Args:
            engine_type: One of 'linear', 'fast', 'ransac', 'adaptive', 'prosac', 'precise'
            **kwargs: Engine-specific parameters
            
        Returns:
            Calibration engine instance
            
        Raises:
            ValueError: For unknown engine type or invalid parameters
        """
        engine_type_lower = engine_type.lower()
        
        # Normalize aliases
        if engine_type_lower in ('linear', 'fast'):
            engine_type_lower = 'fast'
        elif engine_type_lower in ('ransac', 'adaptive'):
            engine_type_lower = 'ransac'
        elif engine_type_lower in ('prosac', 'precise'):
            engine_type_lower = 'prosac'
        elif engine_type_lower in ('visual', 'tick', 'visual_tick'):
            engine_type_lower = 'visual'
        elif engine_type_lower in ('neural', 'adaptive_neural', 'mlp'):
            engine_type_lower = 'neural'
        elif engine_type_lower in ('log', 'logarithmic'):
            engine_type_lower = 'log'
        
        if engine_type_lower not in CalibrationFactory._VALID_PARAMS:
            raise ValueError(f"Unknown calibration engine: {engine_type}. Valid types: {list(CalibrationFactory._VALID_PARAMS.keys())}")
        
        # Validate parameters
        valid_params = CalibrationFactory._VALID_PARAMS[engine_type_lower]
        invalid_params = set(kwargs.keys()) - valid_params
        
        if invalid_params:
            raise ValueError(
                f"Invalid parameters for {engine_type_lower}: {invalid_params}. "
                f"Valid parameters: {valid_params}"
            )
        
        # Create engine with validated parameters
        if engine_type_lower == 'fast':
            return FastCalibration(**kwargs)
        elif engine_type_lower == 'ransac':
            return RANSACCalibration(**kwargs)
        elif engine_type_lower == 'prosac':
            return PROSACCalibration(**kwargs)
        elif engine_type_lower == 'visual':
            return VisualTickCalibration(**kwargs)
        elif engine_type_lower == 'neural':
            return NeuralCalibration(**kwargs)
        elif engine_type_lower == 'log':
            return LogCalibration(**kwargs)
        
        raise ValueError(f"Engine creation not implemented for: {engine_type_lower}")