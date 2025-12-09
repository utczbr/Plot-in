"""
DEPRECATED: This module has been refactored into the utils package.

All functionality has been moved to:
- utils.errors: ErrorSeverity, AnalysisError, ModelLoadError, OCRError, CalibrationError
- utils.inference: preprocess_with_letterbox, run_inference, run_inference_on_image
- utils.analysis_helpers: safe_execute, sanitize_for_json, is_valid_bar_chart, etc.

This file exists for backward compatibility. New code should import from utils directly.
"""
import warnings
warnings.warn(
    "core.utils is deprecated. Import from 'utils' package instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility
from utils.errors import (
    ErrorSeverity,
    AnalysisError,
    ModelLoadError,
    OCRError,
    CalibrationError,
)

from utils.inference import (
    preprocess_with_letterbox,
    _postprocess_onnx_output,
    run_inference,
    run_inference_on_image,
)

from utils.analysis_helpers import (
    safe_execute,
    is_valid_bar_chart,
    _bars_have_good_distribution,
    sanitize_for_json,
    apply_whitelist_settings,
    map_bar_coordinates_to_values,
)

__all__ = [
    'ErrorSeverity',
    'AnalysisError',
    'ModelLoadError',
    'OCRError',
    'CalibrationError',
    'safe_execute',
    'preprocess_with_letterbox',
    '_postprocess_onnx_output',
    'run_inference',
    'run_inference_on_image',
    'is_valid_bar_chart',
    '_bars_have_good_distribution',
    'sanitize_for_json',
    'apply_whitelist_settings',
    'map_bar_coordinates_to_values',
]