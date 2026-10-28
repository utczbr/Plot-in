"""
Centralized utilities package for chart analysis.

This package consolidates all utility functions used across the project.
"""

# Inference utilities
from .inference import (
    preprocess_with_letterbox,
    run_inference_on_image,
)

# Analysis helpers
from .analysis_helpers import (
    sanitize_for_json,
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
    # Inference
    'preprocess_with_letterbox',
    'run_inference_on_image',
    # Analysis helpers
    'sanitize_for_json',
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
