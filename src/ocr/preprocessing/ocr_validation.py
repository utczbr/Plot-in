"""
Validate and clean OCR results with context-aware rules - optimized version.
"""

import re
import numpy as np
from typing import Optional, Tuple


class OCRValidator:
    """Context-aware OCR validation with optimized performance"""
    __slots__ = ('char_corrections', 'numeric_pattern', '_correction_trans')
    
    def __init__(self):
        # Common OCR errors for numbers
        self.char_corrections = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', '|': '1',
            'Z': '2',
            'S': '5', 's': '5',
            'G': '6',
            'B': '8',
            'g': '9', 'q': '9',
        }
        
        # Pre-compile translation table for faster character replacement
        self._correction_trans = str.maketrans(self.char_corrections)
        
        # Pre-compiled regex patterns
        self.numeric_pattern = re.compile(r'^-?\\d+\\.?\\d*[eE]?[-+]?\\d*%?$')

    def validate_numeric(
        self,
        text: str,
        expected_range: Optional[Tuple[float, float]] = None,
        context: str = 'scale'
    ) -> Tuple[Optional[float], float]:
        """Validate and clean numeric OCR result with optimized processing"""
        if not text:
            return None, 0.0
        
        text = text.strip()
        if not text:
            return None, 0.0
        
        original_text = text
        
        # Remove common junk - single pass
        text = text.replace(',', '').replace(' ', '').replace('\\n', '')
        
        # Fast character correction using translation table
        corrected = text.translate(self._correction_trans)
        
        # Try to parse
        value = self._parse_number(corrected)
        
        if value is None:
            # Fallback: fuzzy parse
            value = self._fuzzy_parse(original_text)
            if value is None:
                return None, 0.0
        
        # Compute confidence score
        confidence = self._compute_confidence(
            original_text, corrected, value, expected_range, context
        )
        
        return value, confidence

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse cleaned text to number - fast path"""
        if not text:
            return None
        
        # Handle percentage
        is_percentage = text.endswith('%')
        if is_percentage:
            text = text[:-1]
        
        try:
            value = float(text)
            return value / 100.0 if is_percentage else value
        except ValueError:
            return None

    def _fuzzy_parse(self, text: str) -> Optional[float]:
        """Fuzzy parsing by extracting digits - optimized regex"""
        # Extract digit sequences
        digit_parts = re.findall(r'\\d+', text)
        if not digit_parts:
            return None
        
        # Reconstruct and validate length
        reconstructed = ''.join(digit_parts)
        if 0 < len(reconstructed) <= 10:
            try:
                return float(reconstructed)
            except ValueError:
                pass
        
        return None

    def _compute_confidence(
        self,
        original: str,
        corrected: str,
        value: float,
        expected_range: Optional[Tuple[float, float]],
        context: str
    ) -> float:
        """Compute confidence score with optimized calculations"""
        score = 1.0
        
        # Factor 1: String similarity (Jaccard-like, fast)
        if original != corrected:
            set_orig = set(original)
            set_corr = set(corrected)
            intersection = len(set_orig & set_corr)
            union = max(len(original), len(corrected))
            similarity = intersection / union if union > 0 else 0
            score *= (0.7 + 0.3 * similarity)
        
        # Factor 2: Pattern match
        if not self.numeric_pattern.match(corrected):
            score *= 0.8
        
        # Factor 3: Range validation (if provided)
        if expected_range is not None:
            min_val, max_val = expected_range
            if value < min_val or value > max_val:
                # Exponential decay based on distance
                range_span = max_val - min_val + 1e-6
                distance_ratio = min(
                    abs(value - min_val) / range_span,
                    abs(value - max_val) / range_span
                )
                score *= np.exp(-distance_ratio)
        
        # Factor 4: Context-specific checks
        if context == 'scale':
            abs_val = abs(value)
            if abs_val > 1e6:
                score *= 0.7
            elif abs_val < 1e-3 and value != 0:
                score *= 0.8
        
        # Factor 5: Length check
        orig_len = len(original)
        if orig_len < 1 or orig_len > 12:
            score *= 0.7
        
        return float(np.clip(score, 0.0, 1.0))