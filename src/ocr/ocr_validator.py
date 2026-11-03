"""Compatibility wrapper for OCRValidator."""
from .preprocessing.ocr_validator import (
    OCRValidator,
    _UNICODE_MINUS_TRANS,
)

__all__ = [
    'OCRValidator',
    '_UNICODE_MINUS_TRANS',
]
