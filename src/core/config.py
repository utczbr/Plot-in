"""
Configuration system for different processing modes (fast, optimized, precise).
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModeConfig:
    name: str
    ocr_config: Dict[str, Any]
    calibration_config: Dict[str, Any]
    preprocessing_config: Dict[str, Any]
    parallel_config: Dict[str, Any]

FAST_MODE = ModeConfig(
    name='fast',
    ocr_config={
        'max_retries': 0,
        'use_variations': False,
        'use_validation': False,
        'scale_factor': 2.0,
        'batch_size': 32,
        'use_cache': True,
        'deduplicate': True
    },
    calibration_config={
        'method': 'linear',
        'use_prosac': False,
        'use_ensemble': False
    },
    preprocessing_config={
        'clahe_enabled': False,
        'bilateral_filter_enabled': False
    },
    parallel_config={
        'max_workers': 4,
        'batch_size': 32
    }
)

OPTIMIZED_MODE = ModeConfig(
    name='optimized',
    ocr_config={
        'max_retries': 1,
        'use_variations': True,  # Only for suspicious
        'use_validation': True,
        'scale_factor': 3.0,
        'batch_size': 16,
        'use_cache': True,
        'deduplicate': True
    },
    calibration_config={
        'method': 'adaptive',
        'quality_threshold': 0.85,
        'max_iterations': 500
    },
    preprocessing_config={
        'clahe_enabled': True,
        'bilateral_filter_enabled': True
    },
    parallel_config={
        'max_workers': 4,
        'batch_size': 16
    }
)

PRECISE_MODE = ModeConfig(
    name='precise',
    ocr_config={
        'max_retries': 3,
        'use_variations': True,
        'use_validation': True,
        'scale_factor': 3.0,
        'batch_size': 8,
        'use_cache': True
    },
    calibration_config={
        'method': 'prosac',
        'use_prosac': True,
        'use_ensemble': True,
        'max_iterations': 1000
    },
    preprocessing_config={
        'clahe_enabled': True,
        'bilateral_filter_enabled': True,
        'aggressive_threshold': True
    },
    parallel_config={
        'max_workers': 2,
        'batch_size': 8
    }
)

MODE_CONFIGS = {
    'fast': FAST_MODE,
    'optimized': OPTIMIZED_MODE,
    'precise': PRECISE_MODE
}