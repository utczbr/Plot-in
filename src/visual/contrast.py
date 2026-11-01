"""WCAG 2.1 Contrast Tooling and Requirement Tables.

Provides pure-Python calculations for WCAG 2.1 relative luminance and contrast ratio,
along with normative token definitions and contrast requirements for Plot-in themes.
Zero Qt imports.

References:
- ADR-018: Theme Tokens, Contrast Targets and Accessibility Baseline
- WCAG 2.1 Success Criterion 1.4.3 (Contrast Minimum, Level AA)
- WCAG 2.1 Success Criterion 1.4.11 (Non-text Contrast, Level AA)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# Regex matching valid 3-digit or 6-digit RGB hex colors (with optional #)
_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


# ---------------------------------------------------------------------------
# Normative Theme Tokens (ADR-018)
# ---------------------------------------------------------------------------

DARK_TOKENS: Dict[str, str] = {
    # Surfaces
    "surface.window": "#1e1e1e",
    "surface.panel": "#252526",
    "surface.control": "#2d2d2d",
    "surface.input": "#3c3c3c",
    "surface.toolbar": "#333333",
    "surface.selection": "#264f78",
    # Text
    "text.primary": "#d4d4d4",
    "text.secondary": "#b4b4b4",
    "text.selection_secondary": "#dcdcdc",
    "text.placeholder": "#a8a8a8",
    "text.disabled": "#8c8c8c",
    "text.white": "#ffffff",
    # Accent
    "accent.base": "#0b6fb8",
    "accent.hover": "#0a62a3",
    "accent.pressed": "#005a9e",
    "accent.text": "#4da3ff",
    # Action buttons & Warnings
    "warn.button": "#FF9800",
    "warn.button_hover": "#FFB74D",
    "warn.button_text": "#1e1e1e",
    "success.button": "#2d5a27",
    "success.button_hover": "#3a7a32",
    "danger.button": "#5a3a27",
    "danger.button_hover": "#7a4a32",
    # Status & Review
    "status.ok": "#4CAF50",
    "status.warn": "#FF9800",
    "status.error": "#FF6B5E",
    "review.low": "#e5c07b",
    # Borders & Focus
    "border.control": "#8a8a8a",
    "border.subtle": "#454545",
    "focus.ring": "#4da3ff",
}

HIGH_CONTRAST_TOKENS: Dict[str, str] = {
    "surface.window": "#000000",
    "surface.panel": "#0a0a0a",
    "surface.selection": "#1a365d",
    "text.primary": "#ffffff",
    "text.secondary": "#ffffff",
    "text.disabled": "#a0a0a0",
    "text.white": "#ffffff",
    "border.control": "#ffffff",
    "focus.ring": "#ffd23f",
    "accent.base": "#1aebff",
    "accent.text": "#6cb6ff",
    "status.ok": "#6fdc6f",
    "status.warn": "#ffb000",
    "status.error": "#ff6b6b",
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class ContrastRequirement(dict):
    """Specification of a contrast requirement between two tokens or hex colors.

    Supports both dictionary-style key access (`req['min_ratio']`) and attribute
    access (`req.min_ratio`).
    """

    def __init__(self, fg: str, bg: str, min_ratio: float, usage: str):
        super().__init__(
            fg=fg,
            bg=bg,
            min_ratio=min_ratio,
            usage=usage,
            fg_token=fg,
            bg_token=bg,
            minimum=min_ratio,
            usage_note=usage,
        )
        self.fg = fg
        self.bg = bg
        self.min_ratio = min_ratio
        self.usage = usage
        self.fg_token = fg
        self.bg_token = bg
        self.minimum = min_ratio
        self.usage_note = usage

    def __repr__(self) -> str:
        return f"ContrastRequirement(fg={self.fg!r}, bg={self.bg!r}, min_ratio={self.min_ratio}, usage={self.usage!r})"


class ContrastRequirementDict(dict):
    """Dictionary of contrast requirements supporting lookup by name or (fg, bg) tuple."""

    def __getitem__(self, key: Any) -> ContrastRequirement:
        if isinstance(key, (tuple, list)) and len(key) == 2:
            pair_name = f"{key[0]} on {key[1]}"
            if pair_name in self:
                return super().__getitem__(pair_name)
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, (tuple, list)) and len(key) == 2:
            pair_name = f"{key[0]} on {key[1]}"
            if super().__contains__(pair_name):
                return True
        return super().__contains__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, (tuple, list)) and len(key) == 2:
            pair_name = f"{key[0]} on {key[1]}"
            if super().__contains__(pair_name):
                return super().__getitem__(pair_name)
        return super().get(key, default)


# ---------------------------------------------------------------------------
# Normative Requirements Table (ADR-018)
# ---------------------------------------------------------------------------

def _build_contrast_requirements() -> ContrastRequirementDict:
    reqs = ContrastRequirementDict()

    def add(fg: str, bg: str, min_ratio: float, usage: str) -> None:
        name = f"{fg} on {bg}"
        reqs[name] = ContrastRequirement(fg, bg, min_ratio, usage)

    # 1. Primary body text on standard surfaces (WCAG AA Normal Text: min 4.5:1)
    add("text.primary", "surface.window", 4.5, "Primary body/label text on main window surface")
    add("text.primary", "surface.panel", 4.5, "Primary body/label text on panel/sidebar surface")
    add("text.primary", "surface.control", 4.5, "Primary text on control/button surface")
    add("text.primary", "surface.input", 4.5, "Primary text inside text input fields")
    add("text.primary", "surface.toolbar", 4.5, "Primary text on canvas editor toolbar")

    # 2. Secondary text on standard surfaces (WCAG AA Normal Text: min 4.5:1)
    add("text.secondary", "surface.window", 4.5, "Secondary/dimmed text on main window surface")
    add("text.secondary", "surface.panel", 4.5, "Secondary/dimmed text on panel surface")
    add("text.secondary", "surface.control", 4.5, "Secondary text on control surface")
    add("text.secondary", "surface.input", 4.5, "Secondary text on input surface")

    # 3. Selection surface contrast (selected rows and items: min 4.5:1)
    add("text.white", "surface.selection", 4.5, "White text on table row / item selection surface")
    add("text.selection_secondary", "surface.selection", 4.5, "Secondary text on selection surface")

    # 4. Input placeholders (WCAG AA: min 4.5:1)
    add("text.placeholder", "surface.input", 4.5, "Placeholder text in input fields")

    # 5. Disabled text (ADR-018 policy: target min 3.0:1 for legibility)
    add("text.disabled", "surface.panel", 3.0, "Disabled action/menu text on panel surface")
    add("text.disabled", "surface.window", 3.0, "Disabled text on window surface")
    add("text.disabled", "surface.input", 3.0, "Disabled text inside input fields")

    # 6. Accent action buttons (white text on primary blue accents: min 4.5:1)
    add("text.white", "accent.base", 4.5, "White text on accent button base state")
    add("text.white", "accent.hover", 4.5, "White text on accent button hover state")
    add("text.white", "accent.pressed", 4.5, "White text on accent button pressed state")

    # 7. Accent link/info text (WCAG AA: min 4.5:1)
    add("accent.text", "surface.window", 4.5, "Accent link/info text on window surface")
    add("accent.text", "surface.panel", 4.5, "Accent link/info text on panel surface")
    add("accent.text", "surface.control", 4.5, "Accent link/info text on control surface")
    add("accent.text", "surface.toolbar", 4.5, "Accent link/info text on toolbar surface")

    # 8. Warning / Recalibrate button (dark text on amber/orange: min 4.5:1)
    add("warn.button_text", "warn.button", 4.5, "Dark text on warning/recalibrate button base")
    add("warn.button_text", "warn.button_hover", 4.5, "Dark text on warning/recalibrate button hover")

    # 9. Status indicators and badges (WCAG AA: min 4.5:1)
    add("status.ok", "surface.window", 4.5, "Status OK (green) indicator on window surface")
    add("status.ok", "surface.panel", 4.5, "Status OK (green) indicator on panel surface")
    add("status.warn", "surface.window", 4.5, "Status warning indicator on window surface")
    add("status.warn", "surface.panel", 4.5, "Status warning indicator on panel surface")
    add("status.error", "surface.window", 4.5, "Status error indicator on window surface")
    add("status.error", "surface.panel", 4.5, "Status error indicator on panel surface")

    # 10. Low-confidence review badge/text (WCAG AA: min 4.5:1)
    add("review.low", "surface.panel", 4.5, "Low-confidence amber text on panel surface")
    add("review.low", "surface.window", 4.5, "Low-confidence amber text on window surface")

    # 11. UI Component boundaries (WCAG 1.4.11 UI Components: min 3.0:1)
    add("border.control", "surface.input", 3.0, "Control boundary border against input surface")
    add("border.control", "surface.window", 3.0, "Control boundary border against window surface")

    # 12. Focus indicators (WCAG 1.4.11 Focus Rings: min 3.0:1)
    add("focus.ring", "surface.selection", 3.0, "Focus ring against selection surface")
    add("focus.ring", "surface.input", 3.0, "Focus ring against input surface")
    add("focus.ring", "surface.toolbar", 3.0, "Focus ring against toolbar surface")
    add("focus.ring", "surface.window", 3.0, "Focus ring against window surface")

    # 13. Success and Danger buttons (white text: min 4.5:1)
    add("text.white", "success.button", 4.5, "White text on success (Apply) button base")
    add("text.white", "success.button_hover", 4.5, "White text on success button hover")
    add("text.white", "danger.button", 4.5, "White text on danger (Reset) button base")
    add("text.white", "danger.button_hover", 4.5, "White text on danger button hover")

    return reqs


CONTRAST_REQUIREMENTS: ContrastRequirementDict = _build_contrast_requirements()


# ---------------------------------------------------------------------------
# Core Calculation Functions
# ---------------------------------------------------------------------------

def _resolve_color(
    color: Union[str, Sequence[Union[int, float]]],
    tokens: Optional[Mapping[str, str]] = None,
) -> Tuple[float, float, float]:
    """Resolve color input to normalized RGB floats in [0.0, 1.0]."""
    if isinstance(color, (tuple, list)):
        if len(color) != 3 or not all(isinstance(c, (int, float)) and 0.0 <= float(c) <= 255.0 for c in color):
            raise ValueError(f"Invalid RGB sequence: {color!r}. Expected 3 values in [0, 255].")
        return float(color[0]) / 255.0, float(color[1]) / 255.0, float(color[2]) / 255.0

    if not isinstance(color, str):
        raise ValueError(f"Expected color as hex string or RGB sequence, got {type(color).__name__}")

    raw = color.strip()
    if not raw:
        raise ValueError("Hex color string cannot be empty.")

    # Check if color is a token name
    if tokens and raw in tokens:
        raw = tokens[raw].strip()
    elif raw in DARK_TOKENS:
        raw = DARK_TOKENS[raw].strip()

    match = _HEX_COLOR_RE.match(raw)
    if not match:
        raise ValueError(f"Invalid hex color syntax: {color!r}. Expected '#RGB' or '#RRGGBB'.")

    hex_digits = match.group(1)
    if len(hex_digits) == 3:
        hex_digits = "".join(c * 2 for c in hex_digits)

    try:
        r = int(hex_digits[0:2], 16) / 255.0
        g = int(hex_digits[2:4], 16) / 255.0
        b = int(hex_digits[4:6], 16) / 255.0
    except ValueError as err:
        raise ValueError(f"Invalid hex color syntax: {color!r}") from err

    return r, g, b


def relative_luminance(
    hex_str: Union[str, Sequence[Union[int, float]]],
    tokens: Optional[Mapping[str, str]] = None,
) -> float:
    """Calculate WCAG 2.1 relative luminance for an sRGB color.

    Linearization formula:
        if c <= 0.04045: c_lin = c / 12.92
        else:            c_lin = ((c + 0.055) / 1.055) ** 2.4
        L = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin

    Args:
        hex_str: Hex color string (e.g. '#ffffff', '#fff', '#FF9800') or token name,
                 or an (R, G, B) sequence in [0, 255].
        tokens: Optional token dictionary for resolving token names.

    Returns:
        Relative luminance in [0.0, 1.0].

    Raises:
        ValueError: If hex_str syntax is invalid.
    """
    r, g, b = _resolve_color(hex_str, tokens=tokens)

    def linearize(c: float) -> float:
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)
    return max(0.0, min(1.0, float(lum)))


def contrast_ratio(
    fg_hex: Union[str, Sequence[Union[int, float]]],
    bg_hex: Union[str, Sequence[Union[int, float]]],
    tokens: Optional[Mapping[str, str]] = None,
) -> float:
    """Calculate WCAG 2.1 contrast ratio between foreground and background colors.

    Formula:
        L1 = max(L_fg, L_bg), L2 = min(L_fg, L_bg)
        ratio = (L1 + 0.05) / (L2 + 0.05)

    Args:
        fg_hex: Foreground hex color string, token name, or RGB sequence.
        bg_hex: Background hex color string, token name, or RGB sequence.
        tokens: Optional token dictionary for resolving token names.

    Returns:
        Contrast ratio guaranteed to be in [1.0, 21.0].

    Raises:
        ValueError: If either color has invalid syntax.
    """
    l_fg = relative_luminance(fg_hex, tokens=tokens)
    l_bg = relative_luminance(bg_hex, tokens=tokens)

    l1 = max(l_fg, l_bg)
    l2 = min(l_fg, l_bg)
    ratio = (l1 + 0.05) / (l2 + 0.05)
    return max(1.0, min(21.0, float(ratio)))


def check_contrast(
    fg_hex: Union[str, Sequence[Union[int, float]]],
    bg_hex: Union[str, Sequence[Union[int, float]]],
    min_ratio: float = 4.5,
    tokens: Optional[Mapping[str, str]] = None,
) -> Tuple[bool, float]:
    """Check whether contrast ratio meets or exceeds the required minimum.

    Args:
        fg_hex: Foreground hex color string, token name, or RGB sequence.
        bg_hex: Background hex color string, token name, or RGB sequence.
        min_ratio: Required minimum ratio (default: 4.5 for WCAG AA normal text).
        tokens: Optional token dictionary for resolving token names.

    Returns:
        Tuple of (passed: bool, measured_ratio: float).

    Raises:
        ValueError: If either color has invalid syntax.
    """
    ratio = contrast_ratio(fg_hex, bg_hex, tokens=tokens)
    return ratio >= min_ratio, ratio


def verify_contrast_requirements(
    tokens: Optional[Mapping[str, str]] = None,
    requirements: Optional[Mapping[str, ContrastRequirement]] = None,
) -> List[str]:
    """Verify all requirements against a token mapping, returning a list of failure descriptions.

    Args:
        tokens: Token mapping (defaults to DARK_TOKENS).
        requirements: Contrast requirements mapping (defaults to CONTRAST_REQUIREMENTS).

    Returns:
        List of failure strings. An empty list indicates all requirements passed.
    """
    effective_tokens = tokens if tokens is not None else DARK_TOKENS
    effective_reqs = requirements if requirements is not None else CONTRAST_REQUIREMENTS
    failures: List[str] = []

    for name, req in effective_reqs.items():
        fg_token = req["fg"]
        bg_token = req["bg"]
        min_ratio = req["min_ratio"]

        # Resolve tokens to hex
        fg_hex = effective_tokens.get(fg_token, fg_token)
        bg_hex = effective_tokens.get(bg_token, bg_token)

        passed, ratio = check_contrast(fg_hex, bg_hex, min_ratio=min_ratio)
        if not passed:
            failures.append(
                f"Contrast failure for '{name}' ({req.get('usage', '')}): "
                f"ratio {ratio:.2f}:1 does not meet required minimum {min_ratio:.1f}:1 "
                f"({fg_token}={fg_hex} on {bg_token}={bg_hex})"
            )

    return failures
