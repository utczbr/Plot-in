"""Tests for src/strategies/router.py — StrategyRouter dispatch logic.

Uses mocks exclusively; no real models or weights are loaded.
"""
from unittest.mock import MagicMock

import pytest

from strategies.router import StrategyRouter
from strategies.standard import StandardStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router(**extra):
    """Build a StrategyRouter with a mocked StandardStrategy and optional extras."""
    std = MagicMock(spec=StandardStrategy)
    std.STRATEGY_ID = "standard"
    return StrategyRouter(standard=std, **extra)


# ---------------------------------------------------------------------------
# Explicit mode tests
# ---------------------------------------------------------------------------

def test_standard_mode_always_returns_standard():
    router = _make_router()
    strategy = router.select(
        chart_type="bar",
        classification_confidence=0.95,
        detection_coverage=0.9,
        pipeline_mode="standard",
    )
    assert strategy.STRATEGY_ID == "standard"


def test_explicit_vlm_mode_raises_when_unavailable():
    """When pipeline_mode='vlm' but VLM is not available, router should
    raise ValueError (not silently fall back)."""
    router = _make_router(vlm=None)
    with pytest.raises(ValueError, match="vlm"):
        router.select(
            chart_type="bar",
            classification_confidence=1.0,
            detection_coverage=1.0,
            pipeline_mode="vlm",
        )


def test_explicit_chart_to_table_mode_raises_when_unavailable():
    """When pipeline_mode='chart_to_table' but backend is None, raise ValueError."""
    router = _make_router(chart_to_table=None)
    with pytest.raises(ValueError, match="chart_to_table"):
        router.select(
            chart_type="bar",
            classification_confidence=1.0,
            detection_coverage=1.0,
            pipeline_mode="chart_to_table",
        )


# ---------------------------------------------------------------------------
# Auto mode tests
# ---------------------------------------------------------------------------

def test_auto_low_confidence_routes_to_vlm_when_available():
    vlm = MagicMock()
    vlm.STRATEGY_ID = "vlm"
    router = _make_router(vlm=vlm)
    strategy = router.select(
        chart_type="bar",
        classification_confidence=0.3,
        detection_coverage=0.2,
        pipeline_mode="auto",
    )
    assert strategy.STRATEGY_ID == "vlm"


def test_auto_low_confidence_falls_back_to_standard_when_vlm_none():
    router = _make_router(vlm=None)
    strategy = router.select(
        chart_type="bar",
        classification_confidence=0.3,
        detection_coverage=0.2,
        pipeline_mode="auto",
    )
    assert strategy.STRATEGY_ID == "standard"


def test_auto_uncalibrated_routes_to_hybrid_when_available():
    hybrid = MagicMock()
    hybrid.STRATEGY_ID = "hybrid"
    router = _make_router(hybrid=hybrid)
    strategy = router.select(
        chart_type="bar",
        classification_confidence=0.8,
        detection_coverage=0.7,
        calibration_quality="uncalibrated",
        pipeline_mode="auto",
    )
    assert strategy.STRATEGY_ID == "hybrid"
