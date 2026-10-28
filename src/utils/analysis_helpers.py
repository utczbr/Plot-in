"""
Analysis helper functions for chart validation and processing.
"""
from pathlib import Path
import numpy as np



def sanitize_for_json(obj):
    """Recursively sanitize all data structures for JSON serialization."""
    # Import here to avoid circular dependency
    from core.classifiers.base_classifier import ClassificationResult
    
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, ClassificationResult):
        return {
            "scale_labels": sanitize_for_json(obj.scale_labels),
            "tick_labels": sanitize_for_json(obj.tick_labels),
            "axis_titles": sanitize_for_json(obj.axis_titles),
            "confidence": sanitize_for_json(obj.confidence),
            "metadata": sanitize_for_json(obj.metadata),
        }
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif callable(obj):
        return str(obj)
    else:
        return obj

