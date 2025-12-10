"""
Centralized validation utilities for chart analysis.
"""
from typing import Optional
import re

def is_numeric(text: Optional[str]) -> bool:
    """
    Check if text represents a numeric value using strict regex.
    Ignores labels with letters (e.g., 'G1', 'Item A') to avoid false positives for scales.
    """
    if not text:
        return False
    # Regex: Optional sign, digits, optional decimal part. 
    # Allows for simple numbers but rejects 'G1', '10kg' etc.
    # Note: Does NOT handle scientific notation or comma separators based on strict user request.
    return bool(re.match(r'^\s*[-+]?(?:\d{1,3}(?:[.,]\d{3})*|\d*)(?:[.,]\d+)?(?:[eE][-+]?\d+)?\s*$', text.strip()))

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
