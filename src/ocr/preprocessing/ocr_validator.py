"""
Validate and clean OCR results with context-aware rules - optimized version.
"""

import re
import numpy as np
from typing import Optional, Tuple


_UNICODE_MINUS_TRANS = str.maketrans({
    '\u2212': '-',  # Mathematical minus '−'
    '\u2013': '-',  # En-dash '–'
    '\u2014': '-',  # Em-dash '—'
    '\u2015': '-',  # Horizontal bar '―'
    '\ufe63': '-',  # Small hyphen-minus
    '\uff0d': '-',  # Fullwidth hyphen-minus
})


class OCRValidator:
    """Context-aware OCR validation with optimized performance"""
    __slots__ = ('char_corrections', 'numeric_pattern', 'alpha_pattern', '_correction_trans', '_compiled_patterns')

    def __init__(self):
        # Common OCR errors for numbers
        self.char_corrections = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', '|': '1',
            'Z': '2',
            'S': '5', 's': '5',
            'B': '8',
        }
        
        # Pre-compile regex patterns for efficiency
        self.numeric_pattern = re.compile(r'^[0-9.,+*/=%<>{}[\]()\-]+$')
        self.alpha_pattern = re.compile(r'^[a-zA-Z\s\-\_\.]+$')
        
        # Translation table for common corrections (faster than replace)
        self._correction_trans = str.maketrans(self.char_corrections)
        
        # Additional attributes that may be needed based on error log
        self._compiled_patterns = {
            'numeric': self.numeric_pattern,
            'alpha': self.alpha_pattern
        }

    def validate_and_clean(self, text: str, context_type: str = 'default') -> Tuple[str, float]:
        """
        Validate and clean OCR text with context-aware rules
        
        Args:
            text: Raw OCR text
            context_type: Context for validation ('scale', 'title', 'tick', etc.)
            
        Returns:
            Tuple of (cleaned_text, confidence_score)
        """
        if not text or not isinstance(text, str):
            return "", 0.0

        original_text = text
        text = text.strip()
        if not text:
            return "", 0.0

        # Apply context-independent corrections first
        cleaned = self._basic_cleaning(text, context_type)

        # Apply context-specific validation
        final_text, confidence = self._validate_by_context(cleaned, context_type, original_text)

        # Final validation
        if not final_text or len(final_text.strip()) == 0:
            return "", 0.0

        return final_text.strip(), min(1.0, max(0.0, confidence))

    def _should_apply_digit_corrections(self, text: str) -> bool:
        """Return True only when lookalike digit substitution won't corrupt words or category IDs."""
        stripped = re.sub(r'[\s\$\%\€\£\+\-\.,]', '', text)
        if not stripped:
            return False
        allowed_alpha = set(self.char_corrections.keys()) | {'e', 'E'}
        if any(c.isalpha() and c not in allowed_alpha for c in stripped):
            return False
        if all(c.isalpha() or c == '|' for c in stripped):
            return len(stripped) == 1 and stripped in ('O', 'o', 'I', 'l', '|')
        if re.match(r'^[BSZbsz]\d+$', stripped):
            return False
        return True

    def _basic_cleaning(self, text: str, context_type: str = 'default') -> str:
        """Apply general OCR corrections."""
        normalized = text.translate(_UNICODE_MINUS_TRANS)
        # Apply digit substitutions unconditionally for 'scale', and selectively for mixed contexts
        if context_type == 'scale':
            corrected = normalized.translate(self._correction_trans)
        elif (
            context_type not in ('title', 'chart_title', 'axis_title', 'legend')
            and self._should_apply_digit_corrections(normalized)
        ):
            corrected = normalized.translate(self._correction_trans)
        else:
            corrected = normalized
        
        # Remove excessive whitespace
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        # Remove invalid characters at start/end that are common OCR errors (preserving currency and percent)
        before_strip = corrected
        corrected = re.sub(r'^[^\w\s\-\_\.\$\%\€\£]+|[^\w\s\-\_\.\$\%\€\£]+$', '', corrected).strip()

        # Restore a closing bracket that balances an unmatched opener (e.g. "Time (s)")
        for opener, closer in (('(', ')'), ('[', ']')):
            if (
                before_strip.endswith(closer)
                and corrected.count(opener) > corrected.count(closer)
            ):
                corrected += closer
        
        return corrected

    def _validate_by_context(self, text: str, context_type: str, original: str) -> Tuple[str, float]:
        """Apply context-specific validation and confidence scoring."""
        if context_type == 'scale':
            return self._validate_scale(text, original)
        elif context_type in ('title', 'chart_title', 'axis_title', 'legend'):
            return self._validate_title(text)
        elif context_type in ('tick', 'tick_label', 'axis_labels', 'axis_label', 'data_label'):
            return self._validate_tick(text, original)
        else:
            return self._validate_default(text)

    def _validate_scale(self, text: str, original: str) -> Tuple[str, float]:
        """Validate scale/numeric context with higher numeric bias."""
        # Extract numeric content
        numeric_chars = sum(c.isdigit() for c in text)
        total_chars = len(text)
        
        if total_chars == 0:
            return "", 0.0

        # Check for common numeric patterns
        numeric_ratio = numeric_chars / total_chars
        is_numeric = self.numeric_pattern.search(text) is not None
        
        # If majority of characters are numeric, keep it even with some non-numeric chars
        if numeric_ratio > 0.6 or is_numeric:
            # Additional validation: try to parse as number
            clean_for_parsing = re.sub(r'[^\d.,+\-*/=%<>{}[\]()\s]', '', text)
            try:
                # Simple parsing check - not full validation
                if clean_for_parsing and not clean_for_parsing.isspace():
                    # Keep the cleaned version
                    confidence = min(0.8, 0.3 + 0.5 * numeric_ratio)
                    return text, confidence
            except:
                pass  # Fall through to return with lower confidence
        
        # If not numeric enough, return with low confidence
        confidence = 0.2 * numeric_ratio  # Low confidence for non-numeric scale
        return text, confidence

    def _validate_title(self, text: str) -> Tuple[str, float]:
        """Validate title context with higher alpha bias."""
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text)
        
        if total_chars == 0:
            return "", 0.0

        alpha_ratio = alpha_chars / total_chars
        
        # Titles should have high alphabetic content
        if alpha_ratio > 0.4:  # More than 40% alphabetic
            confidence = 0.6 + 0.4 * alpha_ratio  # Base confidence 0.6, up to 1.0
            return text, min(1.0, confidence)
        else:
            # Lower confidence for non-alpha-heavy titles
            return text, 0.3 * alpha_ratio

    def _validate_tick(self, text: str, original: str) -> Tuple[str, float]:
        """Validate tick label context - could be numeric or alpha."""
        # Ticks can be either numeric or alphabetic
        numeric_chars = sum(c.isdigit() for c in text)
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text)
        
        if total_chars == 0:
            return "", 0.0

        numeric_ratio = numeric_chars / total_chars
        alpha_ratio = alpha_chars / total_chars

        # Mixed context - either numeric or alphabetic is acceptable
        if numeric_ratio > 0.5 or alpha_ratio > 0.5:
            base_confidence = 0.7
            quality_factor = (numeric_ratio + alpha_ratio) / 2
            confidence = base_confidence * (0.5 + 0.5 * quality_factor)
            return text, min(1.0, confidence)
        else:
            # Lower confidence for non-numeric/non-alpha content
            return text, 0.3

    def _validate_default(self, text: str) -> Tuple[str, float]:
        """Default validation for unknown context types."""
        # Just basic length and content validation
        if len(text.strip()) < 1:
            return "", 0.0
        elif len(text.strip()) > 100:
            # Truncate very long results which are likely errors
            return text[:50] + "...", 0.5
        else:
            # Base confidence based on length and character variety
            unique_chars = len(set(text.lower()))
            length_factor = min(1.0, len(text) / 20)  # Up to length 20 is good
            variety_factor = min(1.0, unique_chars / len(text) if text else 1)
            
            confidence = 0.5 + 0.3 * length_factor + 0.2 * variety_factor
            return text, min(1.0, confidence)

    def validate_confidence(self, text: str, original_confidence: float, context_type: str = 'default') -> float:
        """Apply post-processing validation to adjust confidence scores."""
        if not text or len(text.strip()) == 0:
            return 0.0

        # Get validation result and score
        _, validation_score = self.validate_and_clean(text, context_type)
        
        # Combine original confidence with validation score
        # Use geometric mean to prevent overly optimistic scores
        combined_confidence = (original_confidence * validation_score) ** 0.5
        
        return combined_confidence

    def validate_numeric(
        self,
        text: str,
        expected_range: Optional[Tuple[float, float]] = None,
        context: str = 'scale',
        min_confidence: float = 0.5,
    ) -> Tuple[Optional[float], float]:
        """Validate and clean numeric OCR result with optimized processing."""
        if not text:
            return None, 0.0

        text = text.strip()
        if not text:
            return None, 0.0

        original_text = text
        text = text.translate(_UNICODE_MINUS_TRANS).replace(' ', '').replace('\n', '')
        if ',' in text and '.' not in text:
            if re.match(r'^[-+]?\d{1,3}(?:,\d{3})+%?$', text):
                text = text.replace(',', '')
            elif text.count(',') == 1:
                text = text.replace(',', '.')
            else:
                text = text.replace(',', '')
        else:
            text = text.replace(',', '')
        corrected = text.translate(self._correction_trans)

        value = self._parse_number(corrected)
        if value is None:
            value = self._fuzzy_parse(original_text)
            if value is None:
                return None, 0.0

        confidence = self._compute_confidence(
            original_text, corrected, value, expected_range, context
        )
        return value, confidence

    def _parse_number(self, text: str) -> Optional[float]:
        """Parse cleaned text to number - fast path."""
        if not text:
            return None

        is_percentage = text.endswith('%')
        if is_percentage:
            text = text[:-1]

        try:
            value = float(text)
            return value / 100.0 if is_percentage else value
        except ValueError:
            return None

    def _fuzzy_parse(self, text: str) -> Optional[float]:
        """Fuzzy parsing by extracting digits - optimized regex."""
        digit_parts = re.findall(r'\d+', text)
        if not digit_parts:
            return None

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
        context: str,
    ) -> float:
        """Compute confidence score with optimized calculations."""
        score = 1.0

        if original != corrected:
            set_orig = set(original)
            set_corr = set(corrected)
            intersection = len(set_orig & set_corr)
            union = max(len(original), len(corrected))
            similarity = intersection / union if union > 0 else 0
            score *= (0.7 + 0.3 * similarity)

        if not self.numeric_pattern.match(corrected):
            score *= 0.8

        if expected_range is not None:
            min_val, max_val = expected_range
            if value < min_val or value > max_val:
                range_span = max_val - min_val + 1e-6
                distance_ratio = min(
                    abs(value - min_val) / range_span,
                    abs(value - max_val) / range_span,
                )
                score *= np.exp(-distance_ratio)

        if context == 'scale':
            abs_val = abs(value)
            if abs_val > 1e6:
                score *= 0.7
            elif abs_val < 1e-3 and value != 0:
                score *= 0.8

        orig_len = len(original)
        if orig_len < 1 or orig_len > 12:
            score *= 0.7

        return float(np.clip(score, 0.0, 1.0))