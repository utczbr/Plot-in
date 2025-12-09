"""
OCR Post-Processing
Handles numeric validation, unit conversion, and special formatting
"""

import re
from typing import Tuple, Optional


class OCRPostProcessor:
    """Post-process OCR results for numeric labels."""
    
    # Unit conversion patterns
    UNIT_PATTERNS = {
        'k': 1e3, 'K': 1e3,
        'm': 1e6, 'M': 1e6,
        'b': 1e9, 'B': 1e9,
        't': 1e12, 'T': 1e12
    }
    
    # Currency symbols
    CURRENCY_SYMBOLS = ['$', '€', '£', '¥', '₹']
    
    def __init__(self):
        self.numeric_pattern = re.compile(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?')
    
    def parse_numeric_label(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse OCR text to extract numeric value and unit.
        
        Args:
            text: Raw OCR text (e.g., "$1.2K", "~10", "3.14×10²")
        
        Returns:
            (numeric_value, unit) or (None, None) if not numeric
        """
        if not text:
            return None, None
        
        original_text = text.strip()
        text = original_text
        
        # Handle approximation symbols
        text = text.replace('~', '').replace('≈', '').replace('±', '')
        
        # Remove currency symbols (but track them)
        currency = None
        for symbol in self.CURRENCY_SYMBOLS:
            if symbol in text:
                currency = symbol
                text = text.replace(symbol, '')
        
        # Handle percentage
        is_percent = '%' in text
        text = text.replace('%', '')
        
        # Handle scientific notation (Unicode)
        text = text.replace('×10', 'e')
        text = text.replace('x10', 'e')
        
        # Handle superscript numbers (common in scientific notation)
        superscripts = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
        text = text.translate(superscripts)
        
        # Remove commas and spaces
        text = text.replace(',', '').replace(' ', '')
        
        # Extract numeric value
        match = self.numeric_pattern.search(text)
        if not match:
            return None, None
        
        try:
            value = float(match.group())
        except ValueError:
            return None, None
        
        # Check for unit multipliers (k, M, B, etc.)
        remaining = text[match.end():]
        unit_multiplier = 1.0
        
        for unit_char, multiplier in self.UNIT_PATTERNS.items():
            if unit_char in remaining:
                unit_multiplier = multiplier
                break
        
        value *= unit_multiplier
        
        # Apply percentage
        if is_percent:
            value /= 100
        
        # Determine unit
        unit = currency if currency else None
        if is_percent:
            unit = '%' if not unit else f"{unit}%"
        
        return value, unit
    
    def validate_axis_label(self, text: str, expected_range: Tuple[float, float]) -> bool:
        """
        Validate that OCR text is a plausible axis label.
        
        Args:
            text: OCR text
            expected_range: (min, max) expected values on axis
        
        Returns:
            True if label is plausible
        """
        value, _ = self.parse_numeric_label(text)
        
        if value is None:
            return False
        
        # Check if value is within expanded range (allow 20% margin)
        margin = 0.2 * (expected_range[1] - expected_range[0])
        return (expected_range[0] - margin) <= value <= (expected_range[1] + margin)
    
    def clean_text(self, text: str) -> str:
        """Remove common OCR artifacts."""
        # Remove zero-width characters
        text = text.replace('\u200b', '').replace('\ufeff', '')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR mistakes
        replacements = {
            'O': '0',  # Letter O → zero (context-dependent)
            'l': '1',  # Letter l → one (context-dependent)
            'I': '1',  # Letter I → one (context-dependent)
        }
        
        # Only apply if text looks like a number
        if any(char.isdigit() for char in text):
            for old, new in replacements.items():
                text = text.replace(old, new)
        
        return text


# Integration example
def integrate_with_ocr(ocr_result: str, axis_range: Tuple[float, float]) -> Optional[float]:
    """
    Integrate post-processor with OCR pipeline.
    
    Args:
        ocr_result: Raw OCR text
        axis_range: Expected axis range for validation
    
    Returns:
        Parsed numeric value or None
    """
    processor = OCRPostProcessor()
    
    # Clean text
    cleaned = processor.clean_text(ocr_result)
    
    # Parse numeric value
    value, unit = processor.parse_numeric_label(cleaned)
    
    if value is None:
        return None
    
    # Validate against expected range
    if not processor.validate_axis_label(cleaned, axis_range):
        print(f"Warning: OCR value {value} outside expected range {axis_range}")
        return None
    
    return value