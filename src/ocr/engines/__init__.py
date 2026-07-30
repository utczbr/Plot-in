"""
Engines module for OCR systems.
"""

from .ocr_engine_base import (
    EasyOCREngine,
    PaddleOCRBaseEngine
)
try:
    from .ocr_paddle_onnx import PaddleOCREngine
except ImportError:
    PaddleOCREngine = PaddleOCRBaseEngine  # type: ignore

# BaseOCREngine is in the OCR base module, not in this file
from ..ocr_base import BaseOCRLegacy as BaseOCREngine

__all__ = [
    'BaseOCREngine',
    'EasyOCREngine',
    'PaddleOCRBaseEngine',
    'PaddleOCREngine'
]