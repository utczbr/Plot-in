"""
Centralized validation utilities for chart analysis.
"""
from typing import Optional, Union

def is_numeric(text: Optional[str]) -> bool:
    """
    Check if text represents a numeric value.
    Handles common chart formatting like commas, percentages, and currency symbols.
    """
    if not text:
        return False
    try:
        # Remove common non-numeric characters found in chart labels
        text_clean = text.replace(',', '').replace('%', '').replace('$', '').replace(' ', '').strip()
        float(text_clean)
        return True
    except (ValueError, TypeError):
        return False

def clean_numeric_text(text: Optional[str]) -> Optional[float]:
    """
    Convert text to float, handling common formatting.
    Returns None if conversion fails.
    """
    if not text:
        return None
    try:
        text_clean = text.replace(',', '').replace('%', '').replace('$', '').replace(' ', '').strip()
        return float(text_clean)
    except (ValueError, TypeError):
        return None

def is_continuous_scale(text: Optional[str]) -> bool:
    """
    Check if text represents a continuous scale value (e.g., decimal, range, scientific).
    """
    if not text:
        return False
    
    # Check for decimal points, ranges, or scientific notation
    has_decimal = '.' in text
    has_range = '-' in text and text.count('-') == 1 and text.index('-') > 0
    has_scientific = 'e' in text.lower() or 'E' in text
    
    return (has_decimal or has_range or has_scientific) and is_numeric(text.split('-')[0] if has_range else text)
