"""
Calibration adapter to bridge CalibrationService output and ModularBaselineDetector input.

This adapter addresses the critical gap where ModularBaselineDetector needs
(slope, intercept, zero_crossing) from calibration but CalibrationService
provides a different structure.
"""
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

@dataclass
class DetectorCalibrationInput:
    """
    Adapter between CalibrationService output and ModularBaselineDetector input.
    
    Bridges:
    - CalibrationService.CalibrationInfo (from calibration_service.py)
    - CalibrationResult (from calibration_base.py)
    - ModularBaselineDetector's expected format (baseline_detection.py)
    """
    slope: float  # m in y = mx + b
    intercept: float  # b in y = mx + b
    zero_crossing: Optional[float]  # Pixel where value = 0
    r_squared: float  # Fit quality
    axis_type: str  # 'x' or 'y'
    model_func: Optional[Callable[[float], float]]  # pixel -> value function

class CalibrationAdapter:
    """Converts calibration results to detector-compatible format."""
    
    @staticmethod
    def from_service_result(cal_info) -> Optional[DetectorCalibrationInput]:
        """
        Convert CalibrationService.CalibrationInfo to detector input.
        
        Args:
            cal_info: CalibrationInfo from calibration_service
        
        Returns:
            DetectorCalibrationInput or None if invalid
        """
        if cal_info is None:
            return None
        
        # Extract coefficients
        if hasattr(cal_info, 'coefficients') and cal_info.coefficients:
            m, b = cal_info.coefficients
        else:
            return None
        
        # Extract or compute zero crossing
        if hasattr(cal_info, 'zero_crossing') and cal_info.zero_crossing is not None:
            zero = cal_info.zero_crossing
        else:
            # Compute: 0 = mx + b → x = -b/m
            if abs(m) > 1e-6:
                zero = -b / m
            else:
                zero = None
        
        return DetectorCalibrationInput(
            slope=float(m),
            intercept=float(b),
            zero_crossing=float(zero) if zero is not None else None,
            r_squared=float(cal_info.r_squared if hasattr(cal_info, 'r_squared') else 0.0),
            axis_type=str(cal_info.axis_type if hasattr(cal_info, 'axis_type') else 'y'),
            model_func=cal_info.model if hasattr(cal_info, 'model') else None
        )
    
    @staticmethod
    def from_engine_result(cal_result, axis_type: str) -> Optional[DetectorCalibrationInput]:
        """
        Convert CalibrationResult (from calibration engines) to detector input.
        
        Args:
            cal_result: CalibrationResult from PROSAC/Adaptive/Fast engine
            axis_type: 'x' or 'y'
        
        Returns:
            DetectorCalibrationInput or None
        """
        if cal_result is None or not hasattr(cal_result, 'coeffs'):
            return None
        
        m, b = cal_result.coeffs
        
        # Compute zero crossing
        zero = -b / m if abs(m) > 1e-6 else None
        
        return DetectorCalibrationInput(
            slope=float(m),
            intercept=float(b),
            zero_crossing=float(zero) if zero is not None else None,
            r_squared=float(cal_result.r2 if hasattr(cal_result, 'r2') else 0.0),
            axis_type=axis_type,
            model_func=cal_result.func if hasattr(cal_result, 'func') else None
        )