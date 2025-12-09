"""
Context-aware OCR orchestrator for the unified OCR system.
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
import re
from .unified_ocr_system_v2 import UnifiedOCRSystemV2, QualityMode, OCRConfig, OCREngine
# Import UnifiedOCRSystemBridge locally to avoid circular import issues
# from ..compat.unified_ocr_bridge import UnifiedOCRSystemBridge
from .contextual_ocr_adapter import run_contextual_unified, map_class_to_context


def ocr_orchestrator_contextual_with_mode(
    crop: np.ndarray,
    ocr_engine,
    class_name: str,
    spatial_context: Dict,
    advanced_settings: Dict = None,
    mode: str = 'precise'  # NEW: 'fast', 'optimized', 'precise'
) -> Tuple[str, float]:
    """
    Context-aware OCR with adaptive preprocessing pipelines and mode support.
    
    Modes:
    - 'fast': Minimal preprocessing, fastest inference
    - 'optimized': Balanced preprocessing with SIMD optimization  
    - 'precise': Maximum accuracy with multi-variant preprocessing
    - 'unified_*': New unified architecture modes
    """
    if advanced_settings is None:
        advanced_settings = {}
    
    # Check if we're using the new unified system
    if mode.startswith('unified_'):
        # Map unified mode names to the new system
        if mode == 'unified_fast':
            quality_mode = QualityMode.FAST
        elif mode == 'unified_optimized':
            quality_mode = QualityMode.BALANCED
        elif mode == 'unified_precise':
            quality_mode = QualityMode.ACCURATE
        else:
            quality_mode = QualityMode.BALANCED  # default
        
        # Check if ocr_engine is already a unified system bridge
        try:
            # Import locally to avoid circular import
            from ..compat.unified_ocr_bridge import UnifiedOCRSystemBridge
            if isinstance(ocr_engine, UnifiedOCRSystemBridge):
                # Run directly using the contextual adapter
                try:
                    # Use the contextual adapter with the unified system
                    text, confidence = run_contextual_unified(
                        crop, 
                        ocr_engine.unified_system, 
                        class_name, 
                        spatial_context, 
                        mode
                    )
                    return text, confidence
                except Exception as e:
                    logging.error(f"Unified OCR failed: {e}")
                    return "", 0.0
            else:
                # For backward compatibility with legacy engine, fall back to original logic
                pass
        except ImportError:
            # UnifiedOCRSystemBridge not available, continue with legacy logic
            pass
    
    # For legacy engines, continue with original logic
    try:
        # Apply context-aware rules based on class_name
        context_type = _map_class_to_context(class_name)
        
        # Apply spatial context if available
        if spatial_context:
            # Adjust processing based on spatial location, size, etc.
            pass
        
        # Apply advanced settings if provided
        if advanced_settings:
            # Use settings for preprocessing adjustments
            pass
        
        # Preprocess based on context
        processed_crop = _contextual_preprocess(crop, context_type, advanced_settings)
        
        # Perform OCR with engine using legacy interface
        if hasattr(ocr_engine, 'process_batch'):
            # Legacy engine interface
            result = ocr_engine.process_batch([(processed_crop, context_type)])
            if result:
                # Handle new format: [{'text': str, 'confidence': float}, ...]
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    first_result = result[0]
                    text = first_result.get('text', '')
                    confidence = first_result.get('confidence', 0.8)
                else:
                    # Handle old format: [str, ...]
                    text = result[0] if isinstance(result, list) else str(result)
                    # Simple confidence estimation
                    confidence = 0.8 if text and len(text.strip()) > 0 else 0.2
                return text, confidence
        else:
            # New engine interface
            if hasattr(ocr_engine, 'recognize'):
                text, confidence = ocr_engine.recognize(processed_crop, context_type)
                return text, confidence
    
    except Exception as e:
        logging.error(f"Contextual OCR failed: {e}")
        return "", 0.0
    
    return "", 0.0


def _map_class_to_context(class_name: str) -> str:
    """Map class names to appropriate context types for OCR."""
    # This function is now handled by the contextual adapter
    # but keeping for backward compatibility
    return map_class_to_context(class_name)


def _contextual_preprocess(crop: np.ndarray, context_type: str, advanced_settings: Dict) -> np.ndarray:
    """Apply context-aware preprocessing."""
    # Apply preprocessing based on context
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    # Context-specific scaling
    if context_type in ['scale_label', 'tick_label']:
        # Small text needs upscaling
        h, w = gray.shape
        if min(h, w) < 20:  # If text is very small
            gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    
    # Apply any advanced settings
    if advanced_settings.get('binarize', False):
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    return gray