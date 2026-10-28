"""
Centralized validation utilities for chart analysis.
"""
from typing import Optional
import re

_UNICODE_MINUS_TRANS = str.maketrans({
    '\u2212': '-',  # Mathematical minus '−'
    '\u2013': '-',  # En-dash '–'
    '\u2014': '-',  # Em-dash '—'
    '\u2015': '-',  # Horizontal bar '―'
    '\ufe63': '-',  # Small hyphen-minus
    '\uff0d': '-',  # Fullwidth hyphen-minus
})


def _normalize_minus(text: str) -> str:
    return text.translate(_UNICODE_MINUS_TRANS)


def _parse_latex_math(text: str) -> Optional[float]:
    """Parse LaTeX math numeric formats such as $\mathdefault{10^{2}}$, $10^2$, $\mathdefault{0}$."""
    t = _normalize_minus(text).strip()
    # Match $\mathdefault{-10^{2}}$, $10^{2}$, $10^2$, $-10^{1}$, $\mathdefault{10^{-2}}$
    m_exp = re.search(r'^\$?(?:\\mathdefault\{)?(-?)10\^\{?(-?\d+)\}?\}?\$?$', t)
    if m_exp:
        sign = -1.0 if m_exp.group(1) == '-' else 1.0
        power = float(m_exp.group(2))
        return sign * (10.0 ** power)
    # Match $\mathdefault{0}$, $\mathdefault{-5.5}$
    m_num = re.search(r'^\$?(?:\\mathdefault\{)?(-?\d+(?:\.\d+)?)\}?\$?$', t)
    if m_num:
        try:
            return float(m_num.group(1))
        except (ValueError, TypeError):
            pass
    return None


def is_numeric(text: Optional[str]) -> bool:
    """
    Check if text represents a numeric value using strict regex.
    Ignores labels with letters (e.g., 'G1', 'Item A') to avoid false positives for scales.
    Supports LaTeX math notation (e.g., '$\\mathdefault{10^{2}}$', '$\\mathdefault{0}$').
    """
    if not text or not isinstance(text, str):
        return False
    if _parse_latex_math(text) is not None:
        return True
    s = _normalize_minus(text).strip()
    if not s:
        return False
    # Require at least one digit while supporting decimals, thousands separators, and scientific notation.
    return bool(
        re.match(
            r'^[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|(?:\d{1,3}(?:\.\d{3})+)(?:,\d+)?|\d+,\d+|\.\d+)(?:[eE][-+]?\d+)?$',
            s,
        )
    )


def clean_numeric_text(text: Optional[str]) -> Optional[float]:
    """
    Convert text to float, handling common formatting (Unicode minus, thousands vs decimal commas,
    and LaTeX math notation).
    Returns None if conversion fails.
    """
    if not text or not isinstance(text, str):
        return None
    val_latex = _parse_latex_math(text)
    if val_latex is not None:
        return val_latex
    try:
        t = (
            _normalize_minus(text)
            .replace('%', '')
            .replace('$', '')
            .replace('€', '')
            .replace('£', '')
            .replace(' ', '')
            .strip()
        )
        if not t:
            return None
        if ',' in t and '.' not in t:
            if re.match(r'^[-+]?\d{1,3}(?:,\d{3})+(?:[eE][-+]?\d+)?$', t):
                t = t.replace(',', '')
            elif t.count(',') == 1:
                t = t.replace(',', '.')
            else:
                t = t.replace(',', '')
        elif ',' in t and '.' in t:
            if t.rfind(',') > t.rfind('.'):
                t = t.replace('.', '').replace(',', '.')
            else:
                t = t.replace(',', '')
        return float(t)
    except (ValueError, TypeError):
        return None


def is_continuous_scale(text: Optional[str]) -> bool:
    """
    Check if text represents a continuous scale value (e.g., decimal, range, scientific, LaTeX math).
    """
    if not text or not isinstance(text, str):
        return False

    if _parse_latex_math(text) is not None:
        return True

    s = _normalize_minus(text).strip()
    if not s:
        return False

    # Scientific notation must be checked before hyphenated range splitting (e.g. '1e-3')
    if re.match(r'^[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+$', s):
        return True

    if '.' in s and is_numeric(s):
        return True

    # Positive range like '10-20' or '0.5-1.5' (exclude leading negative sign)
    if '-' in s[1:]:
        idx = s.find('-', 1)
        left, right = s[:idx].strip(), s[idx + 1:].strip()
        if is_numeric(left) and is_numeric(right):
            return True

    return False
