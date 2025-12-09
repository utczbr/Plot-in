"""
Type-safe orientation service to prevent string/boolean confusion bugs.

This service addresses the critical issue where string/boolean confusion
caused TypeErrors in ModularBaselineDetector.detect() method.
"""
from enum import Enum
from typing import Union
import logging

class Orientation(str, Enum):
    """Type-safe orientation enum matching baseline_detection.py."""
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    NOT_APPLICABLE = "not_applicable"

class OrientationService:
    """Validates and converts orientation values to enum."""
    
    @staticmethod
    def from_any(value: Union[str, bool, Orientation]) -> Orientation:
        """
        Convert any orientation representation to enum.
        
        Args:
            value: Can be:
                - Orientation.VERTICAL/HORIZONTAL (pass-through)
                - "vertical"/"horizontal" (string)
                - True/False (boolean: True=vertical)
                - "v"/"h" (shorthand)
        
        Returns:
            Orientation enum
        
        Raises:
            ValueError: If value is invalid
        
        Examples:
            >>> OrientationService.from_any(True)
            Orientation.VERTICAL
            >>> OrientationService.from_any("horizontal")
            Orientation.HORIZONTAL
            >>> OrientationService.from_any("v")
            Orientation.VERTICAL
        """
        # Pass-through if already enum
        if isinstance(value, Orientation):
            return value
        
        # Boolean: True=vertical, False=horizontal
        if isinstance(value, bool):
            logging.debug(f"Converting boolean {value} to Orientation")
            return Orientation.VERTICAL if value else Orientation.HORIZONTAL
        
        # String normalization
        if isinstance(value, str):
            normalized = value.lower().strip()
            
            # Full names
            if normalized in ("vertical", "vert", "v"):
                return Orientation.VERTICAL
            elif normalized in ("horizontal", "horiz", "h"):
                return Orientation.HORIZONTAL
            elif normalized in ("not_applicable", "n/a", "na", "none"):
                return Orientation.NOT_APPLICABLE
            
            # Handle common mistakes
            elif normalized in ("true", "1", "yes"):
                logging.warning(f"Converting string '{value}' to VERTICAL (interpret as boolean)")
                return Orientation.VERTICAL
            elif normalized in ("false", "0", "no"):
                logging.warning(f"Converting string '{value}' to HORIZONTAL")
                return Orientation.HORIZONTAL
        
        # Invalid input
        raise ValueError(
            f"Cannot convert {value!r} (type {type(value).__name__}) to Orientation. "
            f"Valid inputs: Orientation.VERTICAL/HORIZONTAL, 'vertical'/'horizontal', "
            f"True/False, 'v'/'h'"
        )
    
    @staticmethod
    def validate_early(orientation: Union[str, bool, Orientation]) -> None:
        """Validate orientation before pipeline processing."""
        try:
            OrientationService.from_any(orientation)
        except ValueError as e:
            raise TypeError(f"Invalid orientation in orchestrator: {e}")