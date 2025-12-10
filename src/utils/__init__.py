"""
Centralized utilities package for chart analysis.

This package consolidates all utility functions used across the project.
"""

# Error classes
from .errors import (
    ErrorSeverity,
    AnalysisError,
    ModelLoadError,
    OCRError,
    CalibrationError,
)

# Inference utilities
from .inference import (
    preprocess_with_letterbox,
    run_inference,
    run_inference_on_image,
)

# Analysis helpers
from .analysis_helpers import (
    safe_execute,
    sanitize_for_json,
    is_valid_bar_chart,
    apply_whitelist_settings,
    map_bar_coordinates_to_values,
)

# Geometry utilities
from .geometry_utils import (
    calculate_pixel_distance,
    compute_aabb_intersection,
    get_center,
    find_closest_element,
)

# Validation utilities
from .validation_utils import (
    is_numeric,
    clean_numeric_text,
    is_continuous_scale,
)

__all__ = [
    # Errors
    'ErrorSeverity',
    'AnalysisError',
    'ModelLoadError',
    'OCRError',
    'CalibrationError',
    # Inference
    'preprocess_with_letterbox',
    'run_inference',
    'run_inference_on_image',
    # Analysis helpers
    'safe_execute',
    'sanitize_for_json',
    'is_valid_bar_chart',
    'apply_whitelist_settings',
    'map_bar_coordinates_to_values',
    # Geometry
    'calculate_pixel_distance',
    'compute_aabb_intersection',
    'get_center',
    'find_closest_element',
    # Validation
    'is_numeric',
    'clean_numeric_text',
    'is_continuous_scale',
]
