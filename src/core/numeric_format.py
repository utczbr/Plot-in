"""Numeric parsing and display formatting module for table cells and overlays.

Provides pure Python (zero Qt) functions for locale-tolerant float parsing
with explicit ambiguity rejection, and significant-digit display formatting
respecting maintainer decision DP-03 (<1e7 integer digit protection).

Specifications:
- ADR-009: Numeric Cell Input Validation and Display Precision
- ADR-007-to-018 Decision Points: DP-03 (Display Precision for Large Numbers)
"""

from decimal import Decimal
import math
import re
from typing import NamedTuple, Optional, Union

__all__ = ["ParseResult", "parse_user_float", "format_display"]


class ParseResult(NamedTuple):
    """Result of parsing user numeric input."""

    value: Optional[float]
    error: Optional[str]


# Ambiguous single-comma pattern: e.g. "1,234", "+12,345", "-123,456".
# Cannot distinguish whether comma is European decimal comma or US thousands separator.
# Note: "0,125" does not match because leading digit is 0 (not 1-9), so 0,125 parses as 0.125.
_AMBIGUOUS_COMMA_RE = re.compile(r"^[+-]?[1-9]\d{0,2},\d{3}$")

# Normative numeric grammar per ADR-009:
# ^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$
_NUMERIC_RE = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def parse_user_float(
    text: Optional[Union[str, int, float]] = None,
    allow_empty: bool = False,
) -> ParseResult:
    """Parse user-provided text into a float or return a clear rejection error.

    Rules:
    1. Normalization: strip whitespace & non-breaking spaces (\u00a0);
       replace Unicode minus \u2212 with ASCII -.
    2. Empty or None:
       - If allow_empty is True: ParseResult(None, None)
       - If allow_empty is False: ParseResult(None, "Empty value not allowed")
    3. Rejections:
       - Multiple commas (e.g. "1,234,567"): "Thousands separators not allowed: write without commas"
       - Ambiguous comma group (^[+-]?[1-9]\\d{0,2},\\d{3}$): "Ambiguous: write 1.234 or 1234"
       - Mixed separators (e.g. "1.234,5" or "1,234.5"): "Thousands separators not allowed: write without commas"
       - Non-numeric or malformed syntax: "Invalid numeric format: '<text>'"
       - Non-finite numbers (NaN, Inf, overflow to inf)
    4. Valid numeric: single comma replaced with dot, converted to float.
    """
    if text is None:
        if allow_empty:
            return ParseResult(None, None)
        return ParseResult(None, "Empty value not allowed")

    if not isinstance(text, str):
        if isinstance(text, (int, float)):
            val = float(text)
            if math.isnan(val) or math.isinf(val):
                return ParseResult(None, "Non-finite values (NaN, Inf) not allowed")
            return ParseResult(val, None)
        text = str(text)

    # Normalize whitespace, NBSP, and Unicode minus
    normalized = text.replace("\u00a0", " ").replace("\u2212", "-").strip()

    if not normalized:
        if allow_empty:
            return ParseResult(None, None)
        return ParseResult(None, "Empty value not allowed")

    # Reject multiple commas
    if normalized.count(",") > 1:
        return ParseResult(None, "Thousands separators not allowed: write without commas")

    # Reject ambiguous single comma group like 1,234 or 12,345
    if _AMBIGUOUS_COMMA_RE.match(normalized):
        return ParseResult(None, "Ambiguous: write 1.234 or 1234")

    # Reject mixed separators (e.g. 1.234,5 or 1,234.5)
    if "," in normalized and "." in normalized:
        return ParseResult(None, "Thousands separators not allowed: write without commas")

    # Match normative grammar
    if not _NUMERIC_RE.match(normalized):
        return ParseResult(None, f"Invalid numeric format: '{text}'")

    # Convert single comma to dot
    dot_str = normalized.replace(",", ".")
    try:
        val = float(dot_str)
    except (ValueError, OverflowError):
        return ParseResult(None, f"Invalid numeric format: '{text}'")

    if math.isnan(val) or math.isinf(val):
        return ParseResult(None, "Non-finite values (NaN, Inf) not allowed")

    return ParseResult(val, None)


def format_display(
    value: Optional[Union[float, int, str]],
    sig: int = 5,
    empty: str = "",
) -> str:
    """Format a numeric value for display according to ADR-009 and DP-03.

    Formatting rules:
    - If value is None, empty string, NaN, or Inf: return empty string (or empty param).
    - If float(value) == 0.0: return "0".
    - DP-03 rule: If 1e-4 <= |d| < 1e7:
        * If |value| >= 1.0: int_digits = len(str(int(abs(value)))),
          effective_sig = max(sig, int_digits) so integer digits below 1e7
          are never rounded to 0 (e.g. 123456.7 -> "123457", 12.34567 -> "12.346").
        * If |value| < 1.0: effective_sig = sig.
        * Round using Decimal(f"{val:.{effective_sig}g}").
        * Format as plain fixed-point format(d, "f"), stripping trailing fractional
          zeros and any trailing decimal point. If "-0", return "0".
    - Outside [1e-4, 1e7) (|d| < 1e-4 or |d| >= 1e7):
        * Scientific notation with sig significant digits:
          d = Decimal(f"{val:.{sig}g}")
          exp = d.adjusted()
          mant = format(d.scaleb(-exp), "f") (strip trailing zeros)
          Emit f"{mant}e{exp:+03d}" (e.g. "1e-07", "1.2346e-05", "1e+07").
    """
    if value is None or value == "":
        return empty

    try:
        val = float(value)
    except (ValueError, TypeError):
        return empty

    if math.isnan(val) or math.isinf(val):
        return empty

    if val == 0.0:
        return "0"

    sig = max(1, int(sig))
    abs_val = abs(val)

    if 1.0 <= abs_val < 1e7:
        int_digits = len(str(int(abs_val)))
        effective_sig = max(sig, int_digits)
    else:
        effective_sig = sig

    d = Decimal(f"{val:.{effective_sig}g}")
    if d == 0:
        return "0"

    abs_d = abs(d)
    if Decimal("1e-4") <= abs_d < Decimal("1e7"):
        s = format(d, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        if s == "-0":
            return "0"
        return s
    else:
        if effective_sig != sig and abs_d >= Decimal("1e7"):
            d = Decimal(f"{val:.{sig}g}")
        exp = d.adjusted()
        mant = format(d.scaleb(-exp), "f")
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")
        if mant == "-0":
            mant = "0"
        return f"{mant}e{exp:+03d}"
