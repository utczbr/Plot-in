"""
Compatibility module for the unified OCR system.
This module provides backward compatibility bridges.
"""
from .unified_ocr_bridge import UnifiedOCRSystemBridge
from .create_unified_engine import (
    create_easyocr_unified,
    create_paddle_unified,
    create_unified_system,
    create_unified_system_with_bridge
)

__all__ = [
    'UnifiedOCRSystemBridge',
    'create_easyocr_unified',
    'create_paddle_unified', 
    'create_unified_system',
    'create_unified_system_with_bridge'
]