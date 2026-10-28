"""
Engines module for OCR systems.
"""

from ..ocr_base import BaseOCREngine
try:
    from .ocr_paddle_onnx import PaddleOCREngine
except ImportError:
    PaddleOCREngine = None  # type: ignore

__all__ = [
    'BaseOCREngine',
    'PaddleOCREngine',
]