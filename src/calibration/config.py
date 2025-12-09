"""
Configuration management for calibration parameters.
Provides default configurations for different use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class CalibrationConfig:
    """
    Configuration class for calibration parameters.
    """
    # Fast calibration parameters
    fast_max_trials: int = 500
    fast_residual_threshold: float = 3.0
    fast_min_inliers: int = 3
    
    # RANSAC calibration parameters
    ransac_max_trials: int = 500
    ransac_residual_threshold: float = 3.0
    ransac_min_inliers: int = 3
    ransac_random_state: Optional[int] = None
    
    # PROSAC calibration parameters
    prosac_max_trials: int = 800
    prosac_residual_threshold: float = 3.0
    prosac_min_inliers: int = 3
    prosac_random_state: Optional[int] = None
    prosac_lo_iters: int = 2
    prosac_prosac_growth: int = 25
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> CalibrationConfig:
        """
        Create a CalibrationConfig from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            CalibrationConfig instance
        """
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary with configuration parameters
        """
        return {
            'fast_max_trials': self.fast_max_trials,
            'fast_residual_threshold': self.fast_residual_threshold,
            'fast_min_inliers': self.fast_min_inliers,
            'ransac_max_trials': self.ransac_max_trials,
            'ransac_residual_threshold': self.ransac_residual_threshold,
            'ransac_min_inliers': self.ransac_min_inliers,
            'ransac_random_state': self.ransac_random_state,
            'prosac_max_trials': self.prosac_max_trials,
            'prosac_residual_threshold': self.prosac_residual_threshold,
            'prosac_min_inliers': self.prosac_min_inliers,
            'prosac_random_state': self.prosac_random_state,
            'prosac_lo_iters': self.prosac_lo_iters,
            'prosac_prosac_growth': self.prosac_prosac_growth,
        }


class CalibrationConfigFactory:
    """
    Factory for creating different calibration configurations based on use case.
    """
    
    @staticmethod
    def get_default_config() -> CalibrationConfig:
        """
        Get the default configuration.
        
        Returns:
            Default CalibrationConfig
        """
        return CalibrationConfig()
    
    @staticmethod
    def get_fast_config() -> CalibrationConfig:
        """
        Get configuration optimized for speed (minimal parameters).
        
        Returns:
            Fast CalibrationConfig
        """
        return CalibrationConfig(
            fast_max_trials=100,
            ransac_max_trials=200,
            prosac_max_trials=300,
            ransac_residual_threshold=5.0,
            prosac_residual_threshold=5.0,
        )
    
    @staticmethod
    def get_precise_config() -> CalibrationConfig:
        """
        Get configuration optimized for precision (more trials, stricter thresholds).
        
        Returns:
            Precise CalibrationConfig
        """
        return CalibrationConfig(
            fast_max_trials=1000,
            ransac_max_trials=1000,
            prosac_max_trials=1500,
            ransac_residual_threshold=1.5,
            prosac_residual_threshold=1.5,
            ransac_min_inliers=5,
            prosac_min_inliers=5,
            prosac_lo_iters=3,
        )
    
    @staticmethod
    def get_robust_config() -> CalibrationConfig:
        """
        Get configuration optimized for handling noisy data (more robust settings).
        
        Returns:
            Robust CalibrationConfig
        """
        return CalibrationConfig(
            fast_max_trials=500,
            ransac_max_trials=800,
            prosac_max_trials=1200,
            ransac_residual_threshold=4.0,
            prosac_residual_threshold=4.0,
            ransac_min_inliers=3,
            prosac_min_inliers=3,
            prosac_lo_iters=2,
        )
    
    @staticmethod
    def create_custom_config(**kwargs) -> CalibrationConfig:
        """
        Create a custom configuration with specified parameters.
        
        Args:
            **kwargs: Configuration parameters to override defaults
            
        Returns:
            Custom CalibrationConfig
        """
        config = CalibrationConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return config