"""Tests for src.core.numeric_format.

Covers:
1. Pure Python invariant: zero Qt imports.
2. parse_user_float table-driven acceptance and edge cases from ADR-009 & DP-03.
3. format_display table-driven acceptance and DP-03 integer preservation (<1e7).
4. Property-based roundtrip testing:
   - Hypothesis property test (when hypothesis is installed)
   - Extensive deterministic property test over IEEE 754 edge-cases and 2000+ floats.
"""

from decimal import Decimal
import math
import random
import struct
import sys
from typing import Optional

import pytest

from src.core.numeric_format import ParseResult, format_display, parse_user_float

try:
    import hypothesis
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAVE_HYPOTHESIS = True
except ImportError:
    HAVE_HYPOTHESIS = False


# ============================================================================
# 1. Zero Qt Imports Invariant
# ============================================================================


def test_zero_qt_imports():
    """Verify that src.core.numeric_format has zero Qt dependencies."""
    qt_modules = [mod for mod in sys.modules if "PyQt" in mod or "PySide" in mod]
    # Re-importing or inspecting numeric_format should not import Qt
    import src.core.numeric_format

    new_qt = [mod for mod in sys.modules if ("PyQt" in mod or "PySide" in mod) and mod not in qt_modules]
    assert len(new_qt) == 0, f"numeric_format must not import Qt modules: {new_qt}"


# ============================================================================
# 2. Table-Driven Tests for parse_user_float
# ============================================================================


@pytest.mark.parametrize(
    "input_text,expected_val",
    [
        ("12.5", 12.5),
        ("12,5", 12.5),
        (" 12,5 ", 12.5),
        ("\u00a012,5\u00a0", 12.5),  # Non-breaking spaces
        (" \t12,5 \n", 12.5),
        ("−3.4", -3.4),  # Unicode minus \u2212
        ("+12.5", 12.5),
        ("1e-3", 0.001),
        ("1,5E2", 150.0),
        ("1.5e+2", 150.0),
        ("0,125", 0.125),  # Leading zero before comma is not thousands separator
        ("0,001", 0.001),
        ("1,5", 1.5),
        ("12,345678", 12.345678),
        (".5", 0.5),
        (",5", 0.5),
        ("12.", 12.0),
        ("12,", 12.0),
        ("0", 0.0),
        ("0.0", 0.0),
        ("-0.0", -0.0),
        ("1234,567", 1234.567),  # 4 digits before comma: not a 1-3 digit thousands group
        ("12345.678", 12345.678),
    ],
)
def test_parse_user_float_accepted_cases(input_text: str, expected_val: float):
    """Test valid numeric formats specified in ADR-009."""
    res = parse_user_float(input_text)
    assert res.error is None
    assert res.value is not None
    assert math.isclose(res.value, expected_val, rel_tol=1e-12, abs_tol=1e-12)
    if expected_val == 0.0 and str(expected_val).startswith("-"):
        assert str(res.value) == "-0.0" or math.copysign(1.0, res.value) < 0


@pytest.mark.parametrize(
    "empty_input",
    ["", None, "   ", "\t\n", "\u00a0", "  \u00a0  "],
)
def test_parse_user_float_empty_inputs(empty_input: Optional[str]):
    """Test allow_empty=True vs allow_empty=False on empty/whitespace inputs."""
    # When allow_empty is True: ParseResult(None, None)
    res_allow = parse_user_float(empty_input, allow_empty=True)
    assert res_allow == ParseResult(None, None)

    # When allow_empty is False: ParseResult(None, 'Empty value not allowed')
    res_disallow = parse_user_float(empty_input, allow_empty=False)
    assert res_disallow.value is None
    assert res_disallow.error == "Empty value not allowed"


@pytest.mark.parametrize(
    "ambiguous_input",
    [
        "1,234",
        "-1,234",
        "+1,234",
        "12,345",
        "-12,345",
        "123,456",
        "-123,456",
    ],
)
def test_parse_user_float_ambiguous_comma_rejected(ambiguous_input: str):
    """Test rejection of ambiguous single comma group ^[+-]?[1-9]\\d{0,2},\\d{3}$."""
    res = parse_user_float(ambiguous_input)
    assert res.value is None
    assert res.error == "Ambiguous: write 1.234 or 1234"


@pytest.mark.parametrize(
    "multiple_commas_input",
    [
        "1,234,567",
        "12,5,3",
        "1,000,000",
        "-1,234,567",
        "1,2,3",
    ],
)
def test_parse_user_float_multiple_commas_rejected(multiple_commas_input: str):
    """Test rejection of multiple comma groups."""
    res = parse_user_float(multiple_commas_input)
    assert res.value is None
    assert res.error == "Thousands separators not allowed: write without commas"


@pytest.mark.parametrize(
    "mixed_separators_input",
    [
        "1.234,5",
        "1,234.5",
        "1.000,00",
        "1,000.00",
    ],
)
def test_parse_user_float_mixed_separators_rejected(mixed_separators_input: str):
    """Test rejection of mixed periods and commas."""
    res = parse_user_float(mixed_separators_input)
    assert res.value is None
    assert res.error == "Thousands separators not allowed: write without commas"


@pytest.mark.parametrize(
    "invalid_input",
    [
        "1.2e",
        "1.2e+",
        "1.2e-",
        "e5",
        "nan",
        "NaN",
        "NAN",
        "inf",
        "Inf",
        "-inf",
        "+infinity",
        "0x10",
        "0b101",
        "0o77",
        "1_000",
        "1_000.5",
        "abc",
        "12a34",
        ".",
        ",",
        "+",
        "-",
        "+.",
        "-,",
        "1.2.3",
        "1e309",  # Overflow to infinity
        "-1e309",
    ],
)
def test_parse_user_float_invalid_numeric_rejected(invalid_input: str):
    """Test rejection of non-numeric, malformed, or non-finite inputs."""
    res = parse_user_float(invalid_input)
    assert res.value is None
    assert res.error is not None


def test_parse_user_float_numeric_types():
    """Verify parse_user_float handles int and float inputs gracefully."""
    res_float = parse_user_float(12.5)
    assert res_float == ParseResult(12.5, None)

    res_int = parse_user_float(42)
    assert res_int == ParseResult(42.0, None)

    res_nan = parse_user_float(float("nan"))
    assert res_nan.value is None
    assert res_nan.error is not None

    res_inf = parse_user_float(float("inf"))
    assert res_inf.value is None
    assert res_inf.error is not None


# ============================================================================
# 3. Table-Driven Tests for format_display (ADR-009 & DP-03)
# ============================================================================


@pytest.mark.parametrize(
    "val,expected_str",
    [
        # Standard ADR-009 acceptance examples
        (0.004, "0.004"),
        (0.00012345, "0.00012345"),
        (12345.678, "12346"),
        (500000.0, "500000"),
        (123456.7, "123457"),  # DP-03 integer preservation
        (12.34567, "12.346"),
        (1e-7, "1e-07"),
        # Zero and signed zero
        (0.0, "0"),
        (-0.0, "0"),
        (0, "0"),
        # Boundary numbers for [1e-4, 1e7)
        (1e-4, "0.0001"),
        (0.0001, "0.0001"),
        (9999999.0, "9999999"),
        (1e7, "1e+07"),
        (10000000.0, "1e+07"),
        (1.2345678e7, "1.2346e+07"),
        # Negative values
        (-0.004, "-0.004"),
        (-0.00012345, "-0.00012345"),
        (-12345.678, "-12346"),
        (-500000.0, "-500000"),
        (-123456.7, "-123457"),
        (-12.34567, "-12.346"),
        (-1e-7, "-1e-07"),
        (-1.2345678e7, "-1.2346e+07"),
        # Small values in scientific notation
        (1e-5, "1e-05"),
        (1.2346e-5, "1.2346e-05"),
        (1.23456e-5, "1.2346e-05"),
        (3.14e-6, "3.14e-06"),
    ],
)
def test_format_display_acceptance_table(val: float, expected_str: str):
    """Test format_display against all authoritative ADR-009 & DP-03 cases."""
    assert format_display(val) == expected_str


@pytest.mark.parametrize(
    "empty_val",
    [None, "", float("nan"), float("inf"), float("-inf"), "invalid"],
)
def test_format_display_empty_values(empty_val):
    """Test format_display returns empty string (or fallback) on non-values."""
    assert format_display(empty_val) == ""
    assert format_display(empty_val, empty="—") == "—"
    assert format_display(empty_val, empty="N/A") == "N/A"


def test_format_display_custom_significant_digits():
    """Test format_display with custom sig parameter."""
    # sig=3 on small decimals
    assert format_display(0.004567, sig=3) == "0.00457"
    assert format_display(12.34567, sig=3) == "12.3"
    assert format_display(12.34567, sig=7) == "12.34567"

    # DP-03 check: even with sig=2, integer digits inside [1, 1e7) are preserved
    assert format_display(123456.7, sig=2) == "123457"
    assert format_display(12345.6, sig=2) == "12346"

    # Outside [1e-4, 1e7), sig is strictly respected
    assert format_display(1.23456e7, sig=3) == "1.23e+07"
    assert format_display(1.23456e-7, sig=3) == "1.23e-07"


# ============================================================================
# 4. Property-Based Roundtrip Testing
# ============================================================================


def test_deterministic_roundtrip_property_finite_floats():
    """Verify roundtrip property parse_user_float(repr(x)).value == x across 2000+ floats."""
    special_floats = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1e-4,
        1e7,
        1e-7,
        math.pi,
        math.e,
        1.2345678901234567,
        # IEEE 754 subnormals and extrema
        5e-324,
        2.2250738585072014e-308,  # min positive normal
        1.7976931348623157e308,   # max finite float
        -1.7976931348623157e308,
    ]

    # Powers of 10
    powers_of_10 = [10.0**k for k in range(-307, 308)]
    # Powers of 2
    powers_of_2 = [2.0**k for k in range(-1022, 1024)]

    # 2000 pseudo-random IEEE-754 64-bit floats
    random_floats = []
    rng = random.Random(20261101)
    for _ in range(2000):
        bits = rng.getrandbits(64)
        val = struct.unpack(">d", struct.pack(">Q", bits))[0]
        if not math.isnan(val) and not math.isinf(val):
            random_floats.append(val)

    all_test_floats = special_floats + powers_of_10 + powers_of_2 + random_floats

    for x in all_test_floats:
        r = repr(x)
        res = parse_user_float(r)
        assert res.error is None, f"Failed parsing repr({x!r}) = {r!r}: error={res.error}"
        assert res.value is not None
        # Bit-level IEEE-754 verification
        assert struct.pack(">d", res.value) == struct.pack(">d", x), (
            f"Bit mismatch for {x!r}: repr={r!r}, got={res.value!r}"
        )


if HAVE_HYPOTHESIS:

    @settings(max_examples=500)
    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_hypothesis_parse_user_float_roundtrip(x: float):
        """Hypothesis property test: for any finite float x, parse_user_float(repr(x)).value == x."""
        r = repr(x)
        res = parse_user_float(r)
        assert res.error is None
        assert res.value is not None
        assert struct.pack(">d", res.value) == struct.pack(">d", x)
