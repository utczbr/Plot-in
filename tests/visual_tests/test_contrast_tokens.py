"""Unit tests for WCAG 2.1 contrast calculations and theme token requirements.

Verifies:
- Standard WCAG 2.1 relative luminance calculation against reference values (black=0.0, white=1.0).
- Standard WCAG 2.1 contrast ratio calculations (black/white=21.0, identical=1.0, symmetry, bounds).
- Hex color syntax validation and error handling (raises ValueError on invalid syntax).
- Table-driven verification that every pair in CONTRAST_REQUIREMENTS passes its threshold for Dark theme.
- Selection surface verification against row foreground tokens.
- Mutation tests: mutating any token below its required threshold fails with the pair named.
- High-contrast theme tokens verification.
- Zero Qt imports across both the module and the test suite.

References:
- ADR-018: Theme Tokens, Contrast Targets and Accessibility Baseline (T-018.2)
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from typing import Dict

import pytest

from visual.contrast import (
    CONTRAST_REQUIREMENTS,
    DARK_TOKENS,
    HIGH_CONTRAST_TOKENS,
    ContrastRequirement,
    check_contrast,
    contrast_ratio,
    relative_luminance,
    verify_contrast_requirements,
)


# ---------------------------------------------------------------------------
# Zero Qt Imports Enforcement
# ---------------------------------------------------------------------------

def test_zero_qt_imports_in_contrast_and_tests():
    """Ensure neither visual.contrast nor this test module imports any Qt dependencies."""
    repo_root = Path(__file__).resolve().parents[2]
    contrast_source_file = repo_root / "src" / "visual" / "contrast.py"
    test_source_file = Path(__file__).resolve()
    forbidden_prefixes = ("PyQt", "PySide", "qt", "Qt")

    for file_path in [contrast_source_file, test_source_file]:
        assert file_path.is_file(), f"File not found: {file_path}"
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden_prefixes:
                        assert not alias.name.startswith(prefix), (
                            f"Forbidden Qt import '{alias.name}' found in {file_path}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for prefix in forbidden_prefixes:
                        assert not node.module.startswith(prefix), (
                            f"Forbidden Qt import from '{node.module}' found in {file_path}"
                        )


def test_zero_qt_loaded_at_runtime():
    """Verify importing visual.contrast in a clean Python process loads zero Qt modules."""
    import os
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "-c",
        "import sys, visual.contrast; "
        "qt = [m for m in sys.modules if m.startswith(('PyQt', 'PySide', 'Qt'))]; "
        "assert not qt, f'Qt modules loaded: {qt}'",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"Clean-process Qt check failed: {result.stderr}"


# ---------------------------------------------------------------------------
# Relative Luminance Tests
# ---------------------------------------------------------------------------

def test_relative_luminance_reference_extremes():
    """Test relative luminance against WCAG 2.1 reference values (black=0.0, white=1.0)."""
    assert relative_luminance("#000000") == pytest.approx(0.0, abs=1e-6)
    assert relative_luminance("#ffffff") == pytest.approx(1.0, abs=1e-6)
    assert relative_luminance("#000") == pytest.approx(0.0, abs=1e-6)
    assert relative_luminance("#fff") == pytest.approx(1.0, abs=1e-6)
    assert relative_luminance("000000") == pytest.approx(0.0, abs=1e-6)
    assert relative_luminance("ffffff") == pytest.approx(1.0, abs=1e-6)


def test_relative_luminance_primary_channels():
    """Test standard sRGB coefficients for linearized pure red, green, and blue channels."""
    # WCAG weights: R=0.2126, G=0.7152, B=0.0722
    assert relative_luminance("#ff0000") == pytest.approx(0.2126, abs=1e-4)
    assert relative_luminance("#00ff00") == pytest.approx(0.7152, abs=1e-4)
    assert relative_luminance("#0000ff") == pytest.approx(0.0722, abs=1e-4)


def test_relative_luminance_rgb_tuple_input():
    """Test that relative_luminance accepts (R, G, B) integer sequences in [0, 255]."""
    assert relative_luminance((0, 0, 0)) == pytest.approx(0.0, abs=1e-6)
    assert relative_luminance((255, 255, 255)) == pytest.approx(1.0, abs=1e-6)
    assert relative_luminance([255, 0, 0]) == pytest.approx(0.2126, abs=1e-4)


@pytest.mark.parametrize(
    "invalid_color",
    [
        "",
        "#",
        "#12",
        "#1234",
        "#12345",
        "#1234567",
        "#gggggg",
        "xyz",
        "#ff-ff-ff",
        None,
        123456,
        (256, 0, 0),
        (-1, 0, 0),
        (255, 255),
        (255, 255, 255, 255),
    ],
)
def test_relative_luminance_invalid_syntax_raises_value_error(invalid_color):
    """Test that relative_luminance raises ValueError on invalid syntax or out-of-range inputs."""
    with pytest.raises(ValueError):
        relative_luminance(invalid_color)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Contrast Ratio Tests
# ---------------------------------------------------------------------------

def test_contrast_ratio_reference_extremes():
    """Test contrast ratio against WCAG reference values (black/white=21.0, identical=1.0)."""
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=1e-6)
    assert contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0, abs=1e-6)
    assert contrast_ratio("#1e1e1e", "#1e1e1e") == pytest.approx(1.0, abs=1e-6)
    assert contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0, abs=1e-6)
    assert contrast_ratio("#000000", "#000000") == pytest.approx(1.0, abs=1e-6)


def test_contrast_ratio_symmetry_and_bounds():
    """Test that contrast_ratio is symmetric and bounded strictly within [1.0, 21.0]."""
    test_colors = ["#1e1e1e", "#252526", "#2d2d2d", "#3c3c3c", "#d4d4d4", "#0b6fb8", "#FF9800", "#ffffff"]
    for c1 in test_colors:
        for c2 in test_colors:
            ratio1 = contrast_ratio(c1, c2)
            ratio2 = contrast_ratio(c2, c1)
            assert ratio1 == pytest.approx(ratio2, abs=1e-6)
            assert 1.0 <= ratio1 <= 21.0


def test_contrast_ratio_adr018_key_findings():
    """Test exact contrast ratios documented in ADR-018 NF-18.2 for key UI elements."""
    # 1. Recalibrate button text (dark text #1e1e1e on #FF9800): 7.73:1 (passes target >= 4.5)
    ratio_recal_fixed = contrast_ratio("#1e1e1e", "#FF9800")
    assert ratio_recal_fixed >= 7.7
    assert ratio_recal_fixed == pytest.approx(7.73, abs=0.05)

    # Old failing recalibrate text (#d4d4d4 on #FF9800): 1.45:1
    ratio_recal_old = contrast_ratio("#d4d4d4", "#FF9800")
    assert ratio_recal_old < 2.0
    assert ratio_recal_old == pytest.approx(1.45, abs=0.05)

    # 2. Selected file row text (#ffffff on #264f78): 8.50:1 (passes target >= 4.5)
    ratio_sel_fixed = contrast_ratio("#ffffff", "#264f78")
    assert ratio_sel_fixed >= 8.4
    assert ratio_sel_fixed == pytest.approx(8.50, abs=0.05)

    # Old failing row selection (#ffffff on #4a90e2): 3.29:1
    ratio_sel_old = contrast_ratio("#ffffff", "#4a90e2")
    assert ratio_sel_old < 4.5
    assert ratio_sel_old == pytest.approx(3.29, abs=0.05)

    # 3. Accent button base (#ffffff on #0b6fb8): 5.28:1
    assert contrast_ratio("#ffffff", "#0b6fb8") == pytest.approx(5.28, abs=0.05)

    # 4. Accent button hover (#ffffff on #0a62a3): 6.39:1
    assert contrast_ratio("#ffffff", "#0a62a3") == pytest.approx(6.39, abs=0.05)

    # 5. Control border on input (#8a8a8a on #3c3c3c): 3.20:1 (passes target >= 3.0)
    assert contrast_ratio("#8a8a8a", "#3c3c3c") == pytest.approx(3.20, abs=0.05)

    # 6. Focus ring on toolbar (#4da3ff on #333333): 4.81:1 (passes target >= 3.0)
    assert contrast_ratio("#4da3ff", "#333333") == pytest.approx(4.81, abs=0.05)


# ---------------------------------------------------------------------------
# check_contrast Function Tests
# ---------------------------------------------------------------------------

def test_check_contrast():
    """Test check_contrast helper function boolean and ratio returns."""
    passed, ratio = check_contrast("#ffffff", "#000000", min_ratio=4.5)
    assert passed is True
    assert ratio == pytest.approx(21.0, abs=1e-6)

    # Identical colors fail 4.5:1 requirement
    passed, ratio = check_contrast("#1e1e1e", "#1e1e1e", min_ratio=4.5)
    assert passed is False
    assert ratio == pytest.approx(1.0, abs=1e-6)

    # Boundary check: #8a8a8a on #3c3c3c is 3.20: passes 3.0, fails 4.5
    assert check_contrast("#8a8a8a", "#3c3c3c", min_ratio=3.0)[0] is True
    assert check_contrast("#8a8a8a", "#3c3c3c", min_ratio=4.5)[0] is False


# ---------------------------------------------------------------------------
# Table-Driven Requirements Verification (Parametrized over CONTRAST_REQUIREMENTS)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("req_name,req", list(CONTRAST_REQUIREMENTS.items()))
def test_all_contrast_requirements_pass_for_dark_theme(req_name: str, req: ContrastRequirement):
    """Test that every normative requirement pair meets its specified minimum ratio in Dark theme."""
    fg_token = req["fg"]
    bg_token = req["bg"]
    min_ratio = req["min_ratio"]

    assert fg_token in DARK_TOKENS, f"Foreground token '{fg_token}' missing from DARK_TOKENS"
    assert bg_token in DARK_TOKENS, f"Background token '{bg_token}' missing from DARK_TOKENS"

    fg_hex = DARK_TOKENS[fg_token]
    bg_hex = DARK_TOKENS[bg_token]

    passed, measured_ratio = check_contrast(fg_hex, bg_hex, min_ratio=min_ratio)
    assert passed, (
        f"Requirement '{req_name}' failed: measured ratio {measured_ratio:.2f}:1 "
        f"is below required minimum {min_ratio:.1f}:1 ({fg_token}={fg_hex} on {bg_token}={bg_hex}). "
        f"Usage: {req.get('usage', '')}"
    )


def test_verify_contrast_requirements_clean_pass():
    """Test verify_contrast_requirements helper returns an empty list of failures for standard DARK_TOKENS."""
    failures = verify_contrast_requirements()
    assert failures == []


# ---------------------------------------------------------------------------
# Selection Surface Contract Tests (ADR-018 § T-018.2)
# ---------------------------------------------------------------------------

def test_selection_surface_against_all_row_foreground_tokens():
    """Test surface.selection meets contrast requirements against all row foreground tokens."""
    selection_bg = DARK_TOKENS["surface.selection"]

    # White text on selection row (8.50:1 >= 4.5:1)
    passed, ratio = check_contrast(DARK_TOKENS["text.white"], selection_bg, min_ratio=4.5)
    assert passed
    assert ratio >= 8.4

    # Secondary text on selection row (text.selection_secondary #dcdcdc: 6.17:1 >= 4.5:1)
    passed, ratio = check_contrast(DARK_TOKENS["text.selection_secondary"], selection_bg, min_ratio=4.5)
    assert passed
    assert ratio >= 6.0

    # Focus ring on selection row (#4da3ff: 3.24:1 >= 3.0:1)
    passed, ratio = check_contrast(DARK_TOKENS["focus.ring"], selection_bg, min_ratio=3.0)
    assert passed
    assert ratio >= 3.2


# ---------------------------------------------------------------------------
# Acceptance Criterion: Mutation Testing (ADR-018 § T-018.2 Acceptance)
# "the test passes for Dark; mutating one token below its minimum makes it fail with the pair named."
# ---------------------------------------------------------------------------

def test_mutating_token_below_minimum_fails_with_pair_named():
    """Acceptance: mutating a token below its minimum makes verify_contrast_requirements fail with the pair named."""
    # 1. Mutate text.primary to a low-contrast dark gray (#303030)
    mutated_tokens = dict(DARK_TOKENS)
    mutated_tokens["text.primary"] = "#303030"

    failures = verify_contrast_requirements(tokens=mutated_tokens)
    assert len(failures) >= 1
    # Check that failure explicitly names the mutated pair
    primary_failures = [f for f in failures if "text.primary on surface.window" in f]
    assert len(primary_failures) == 1
    assert "ratio" in primary_failures[0]
    assert "text.primary" in primary_failures[0]
    assert "surface.window" in primary_failures[0]


def test_mutating_warn_button_text_to_legacy_fails():
    """Mutating warn.button_text to the legacy value (#d4d4d4) fails the warning button requirement."""
    mutated_tokens = dict(DARK_TOKENS)
    mutated_tokens["warn.button_text"] = "#d4d4d4"

    failures = verify_contrast_requirements(tokens=mutated_tokens)
    base_failures = [f for f in failures if "'warn.button_text on warn.button'" in f]
    hover_failures = [f for f in failures if "'warn.button_text on warn.button_hover'" in f]
    assert len(base_failures) == 1
    assert len(hover_failures) == 1
    assert "1.45:1" in base_failures[0]
    assert "1.17:1" in hover_failures[0]


def test_mutating_selection_surface_to_legacy_fails():
    """Mutating surface.selection to legacy (#4a90e2) fails the selected row requirement."""
    mutated_tokens = dict(DARK_TOKENS)
    mutated_tokens["surface.selection"] = "#4a90e2"

    failures = verify_contrast_requirements(tokens=mutated_tokens)
    sel_failures = [f for f in failures if "text.white on surface.selection" in f]
    assert len(sel_failures) == 1
    assert "3.29:1" in sel_failures[0] or "3.3" in sel_failures[0]


def test_mutating_accent_base_fails():
    """Mutating accent.base to a pale blue fails the accent button white text requirement."""
    mutated_tokens = dict(DARK_TOKENS)
    mutated_tokens["accent.base"] = "#80c0ff"  # high luminance blue

    failures = verify_contrast_requirements(tokens=mutated_tokens)
    accent_failures = [f for f in failures if "text.white on accent.base" in f]
    assert len(accent_failures) == 1


# ---------------------------------------------------------------------------
# High-Contrast Theme Verification (ADR-018 line 91)
# ---------------------------------------------------------------------------

def test_high_contrast_theme_tokens():
    """Verify High-contrast theme tokens meet WCAG AAA / High-Contrast policy requirements (text >= 7:1)."""
    # Text on window (black background): 21.0:1
    assert contrast_ratio(HIGH_CONTRAST_TOKENS["text.primary"], HIGH_CONTRAST_TOKENS["surface.window"]) == 21.0

    # Text on panel (#0a0a0a): >= 19.0:1
    assert contrast_ratio(HIGH_CONTRAST_TOKENS["text.primary"], HIGH_CONTRAST_TOKENS["surface.panel"]) >= 19.0

    # Selection surface with white text: >= 12.0:1
    assert contrast_ratio(HIGH_CONTRAST_TOKENS["text.white"], HIGH_CONTRAST_TOKENS["surface.selection"]) >= 12.0

    # Focus ring (#ffd23f) on window: >= 14.0:1
    assert contrast_ratio(HIGH_CONTRAST_TOKENS["focus.ring"], HIGH_CONTRAST_TOKENS["surface.window"]) >= 14.0

    # Accent button text (black on #1aebff): >= 14.0:1
    assert contrast_ratio("#000000", HIGH_CONTRAST_TOKENS["accent.base"]) >= 14.0

    # Status colors on window: all >= 7.0:1
    assert contrast_ratio(HIGH_CONTRAST_TOKENS["status.ok"], HIGH_CONTRAST_TOKENS["surface.window"]) >= 7.0
    assert contrast_ratio(HIGH_CONTRAST_TOKENS["status.warn"], HIGH_CONTRAST_TOKENS["surface.window"]) >= 7.0
    assert contrast_ratio(HIGH_CONTRAST_TOKENS["status.error"], HIGH_CONTRAST_TOKENS["surface.window"]) >= 7.0
