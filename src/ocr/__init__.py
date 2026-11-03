"""
OCR module for the Chart Analysis Tool.
Contains various OCR engine implementations.
"""

from .preprocessing.ocr_validator import OCRValidator
from .ocr_factory import OCREngineFactory
from .ocr_base import BaseOCREngine

__all__ = [
    'OCRValidator',
    'OCREngineFactory',
    'BaseOCREngine',
]
