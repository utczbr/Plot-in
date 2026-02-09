"""Canonical baseline detection package."""

from .types import BaselineLine, BaselineResult, DetectorConfig
from .detector import ModularBaselineDetector, detect_baselines

__all__ = [
    "BaselineLine",
    "BaselineResult",
    "DetectorConfig",
    "ModularBaselineDetector",
    "detect_baselines",
]
