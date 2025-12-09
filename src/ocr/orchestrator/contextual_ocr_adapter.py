"""
Contextual OCR adapter that removes duplicate preprocessing decisions 
and directly leverages the orchestrator's validated, mode-aware pipeline
"""
import numpy as np
from typing import Dict, Tuple, List
from .unified_ocr_system_v2 import UnifiedOCRSystemV2, QualityMode, OCRResult


MODE_MAP = {
    "unified_fast": QualityMode.FAST, 
    "unified_optimized": QualityMode.BALANCED, 
    "unified_precise": QualityMode.ACCURATE
}


def run_contextual_unified(crop: np.ndarray, unified: UnifiedOCRSystemV2, class_name: str, spatial_context: Dict, mode_name: str) -> Tuple[str, float]:
    """
    Run contextual OCR using the unified system with proper context mapping.
    
    Args:
        crop: Image crop to process
        unified: UnifiedOCRSystemV2 instance
        class_name: Name of the object class (e.g., "scale_label", "axis_title")
        spatial_context: Dictionary with spatial information
        mode_name: Name of the quality mode to use
        
    Returns:
        Tuple of (recognized_text, confidence)
    """
    # Map class to context string the orchestrator expects
    context = map_class_to_context(class_name)
    
    # Process the single crop via the batch interface
    results = unified.process_batch([(crop, context)])
    
    if results and len(results) > 0:
        result = results[0]
        return result.text, result.confidence
    else:
        return "", 0.0


def run_contextual_batch_unified(crops_with_info: List[Tuple[np.ndarray, str, Dict]], 
                                unified: UnifiedOCRSystemV2, 
                                mode_name: str) -> List[Tuple[str, float]]:
    """
    Run contextual OCR on a batch of crops with their class information.
    
    Args:
        crops_with_info: List of (image_crop, class_name, spatial_context) tuples
        unified: UnifiedOCRSystemV2 instance
        mode_name: Name of the quality mode to use
        
    Returns:
        List of (recognized_text, confidence) tuples
    """
    # Map class names to context strings
    crops_with_context = [(crop, map_class_to_context(class_name)) 
                         for crop, class_name, _ in crops_with_info]
    
    # Process the batch
    results = unified.process_batch(crops_with_context)
    
    # Extract text and confidence
    return [(result.text, result.confidence) for result in results]


def map_class_to_context(name: str) -> str:
    """
    Map a class name to the appropriate context string for the OCR system.
    
    Args:
        name: Class name (e.g., "scale_label", "axis_title")
        
    Returns:
        Context string for OCR processing
    """
    s = name.lower()
    if any(k in s for k in ("scale", "tick", "number", "value")):
        return "scale_label"
    if any(k in s for k in ("title", "label", "name")):
        return "axis_title"
    if any(k in s for k in ("legend", "cat", "series")):
        return "tick_label"
    if any(k in s for k in ("data", "point", "plot")):
        return "data_label"
    return "default"


def get_context_specific_config(context: str, base_config) -> 'OCRConfig':
    """
    Get OCR config adjusted for specific context.
    This could be used to apply context-specific parameters like character allowlists.
    
    Args:
        context: Context string (e.g., "scale_label", "axis_title")
        base_config: Base OCRConfig to modify
        
    Returns:
        Modified OCRConfig with context-appropriate settings
    """
    # For now, return the base config; in the future this could modify settings
    # based on context (e.g., numeric allowlist for scale_label context)
    return base_config