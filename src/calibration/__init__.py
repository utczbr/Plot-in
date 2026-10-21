"""
Initialization for calibration module.
Exports CalibrationFactory and RobustSiegelCalibration for external use.
"""

from .calibration_factory import CalibrationFactory
from .calibration_siegel import RobustSiegelCalibration

__all__ = ["CalibrationFactory", "RobustSiegelCalibration"]