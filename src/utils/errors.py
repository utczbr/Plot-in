"""
Error classes for chart analysis.
"""
from enum import Enum


class ErrorSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AnalysisError(Exception):
    """Base exception with context"""
    def __init__(self, message, severity=ErrorSeverity.ERROR, context=None):
        self.message = message
        self.severity = severity
        self.context = context or {}
        super().__init__(self.message)


class ModelLoadError(AnalysisError):
    pass


class OCRError(AnalysisError):
    pass


class CalibrationError(AnalysisError):
    pass
