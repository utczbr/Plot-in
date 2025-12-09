"""
Validate and clean OCR results with context-aware rules - optimized version.
"""

import re
import numpy as np
from typing import Optional, Tuple


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
            '$': '5',
            '%': '9',
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
        cleaned = self._basic_cleaning(text)

        # Apply context-specific validation
        final_text, confidence = self._validate_by_context(cleaned, context_type, original_text)

        # Final validation
        if not final_text or len(final_text.strip()) == 0:
            return "", 0.0

        return final_text.strip(), min(1.0, max(0.0, confidence))

    def _basic_cleaning(self, text: str) -> str:
        """Apply general OCR corrections."""
        # Apply character substitutions using translation table (fastest method)
        corrected = text.translate(self._correction_trans)
        
        # Remove excessive whitespace
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        # Remove invalid characters at start/end that are common OCR errors
        corrected = re.sub(r'^[^\w\s\-\_\.]+|[^\w\s\-\_\.]+$', '', corrected).strip()
        
        return corrected

    def _validate_by_context(self, text: str, context_type: str, original: str) -> Tuple[str, float]:
        """Apply context-specific validation and confidence scoring."""
        if context_type == 'scale':
            return self._validate_scale(text, original)
        elif context_type == 'title':
            return self._validate_title(text)
        elif context_type == 'tick':
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

    def validate_numeric(self, text: str, context: str = None, min_confidence: float = 0.5) -> tuple:
        """Validate if text is numeric based on content."""
        if not text:
            return "", 0.0
        
        # Clean the text first
        cleaned = self._basic_cleaning(text)
        
        # Check if it's mostly numeric
        numeric_chars = sum(c.isdigit() or c in '.,+-eE%' for c in cleaned)
        total_chars = len(cleaned)
        
        if total_chars == 0:
            return "", 0.0
            
        numeric_ratio = numeric_chars / total_chars
        confidence = numeric_ratio if numeric_ratio >= min_confidence else 0.0
        return cleaned, confidence

    def validate_alphanumeric(self, text: str, context: str = None, min_confidence: float = 0.5) -> tuple:
        """Validate if text is alphanumeric based on content."""
        if not text:
            return "", 0.0
        
        clean_text = text.strip()
        if len(clean_text) == 0:
            return "", 0.0
            
        alphanumeric_chars = sum(c.isalnum() or c.isspace() for c in clean_text)
        total_chars = len(clean_text)
        
        if total_chars == 0:
            return "", 0.0
            
        alphanumeric_ratio = alphanumeric_chars / total_chars
        confidence = alphanumeric_ratio if alphanumeric_ratio >= min_confidence else 0.0
        return clean_text, confidence

    def clean_numeric(self, text: str, context: str = None) -> tuple:
        """Clean text for numeric validation."""
        if not text:
            return "", 0.0
        
        # Remove common non-numeric characters that might be around numbers
        cleaned = re.sub(r'[^\d.,+\-*/=%<>{}[\]()\s]', '', text)
        cleaned = cleaned.strip()
        
        # Calculate confidence based on how much of the original text remains after cleaning
        if len(text) > 0:
            confidence = len(cleaned) / len(text)  # How much of original text survived cleaning
        else:
            confidence = 0.0
            
        return cleaned, confidence

    def validate_range(self, text: str, context: str = None, min_val: float = None, max_val: float = None) -> tuple:
        """Validate if numeric text is within a specified range."""
        if not text:
            return text, 0.0
            
        try:
            # Clean the text to extract numeric value
            clean_num, _ = self.clean_numeric(text)
            if not clean_num:
                return text, 0.0
                
            # Extract first number from the text
            numbers = re.findall(r'-?\d+\.?\d*', clean_num)
            if not numbers:
                return text, 0.0
                
            num = float(numbers[0])
            
            in_range = True
            if min_val is not None and num < min_val:
                in_range = False
            if max_val is not None and num > max_val:
                in_range = False
                
            confidence = 1.0 if in_range else 0.0
            return text, confidence
        except (ValueError, IndexError):
            return text, 0.0