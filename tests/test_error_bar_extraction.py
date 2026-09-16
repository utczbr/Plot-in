# -*- coding: utf-8 -*-
"""
Tests for bar chart error bar extraction, dict unpacking, and validator fallback.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from extractors.bar_extractor import BarExtractor
from extractors.error_bar_validator import ErrorBarValidator
from handlers.bar_handler import BarHandler


def test_bar_extractor_dict_unpacking_and_margin():
    """Verify BarExtractor handles associated_error_bar as a dict and calculates margin."""
    extractor = BarExtractor()
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # A single bar from y=50 to y=150, x=40 to x=60
    detections = {
        'bar': [{'xyxy': [40, 50, 60, 150], 'conf': 0.9}],
        'error_bar': [{'xyxy': [48, 30, 52, 70], 'conf': 0.85}],  # Span = 40px
    }

    # Calibration model: pixel -> value (e.g. y=150 -> 0.0, y=50 -> 10.0, slope = -0.1)
    # y=30 -> 12.0, y=70 -> 8.0 -> difference = 4.0, half-span = 2.0
    scale_model = lambda y: (150 - y) * 0.1

    result = extractor.extract(
        img=img,
        detections=detections,
        scale_model=scale_model,
        baseline_coord=150.0,
        img_dimensions={'width': 200, 'height': 200},
    )

    bars = result.get('bars', [])
    assert len(bars) == 1, "Expected 1 extracted bar"
    bar = bars[0]
    assert 'error_bar' in bar, "Bar should have associated error_bar"
    assert bar['error_bar']['bbox'] == [48, 30, 52, 70]
    assert pytest.approx(bar['error_bar']['margin'], 0.01) == 2.0


def test_error_bar_validator_relaxed_thresholds():
    """Verify ErrorBarValidator associates user-drawn error bars with aspect_ratio < 2.0."""
    validator = ErrorBarValidator()
    bar = {'xyxy': [100, 80, 140, 200]}  # width=40, height=120, cx=120
    # Error bar with wide caps: width=18, height=24 -> aspect_ratio = 24/18 = 1.33 < 2.0
    eb = {'xyxy': [111, 60, 129, 84]}

    enriched = validator.associate_and_validate([bar], [eb], 'vertical')
    assert len(enriched) == 1
    assert 'error_bar_validated' in enriched[0], "Should associate error bar despite aspect ratio < 2.0"
    eb_val = enriched[0]['error_bar_validated']
    assert eb_val['is_valid'] is True
    assert eb_val['bbox'] == [111, 60, 129, 84]


def test_bar_handler_end_to_end_error_bar():
    """Verify BarHandler.extract_values enriches bars with calibrated error bar margin."""
    handler = BarHandler(MagicMock(), MagicMock())
    img = np.zeros((300, 300, 3), dtype=np.uint8)

    detections = {
        'bar': [
            {'xyxy': [50, 100, 80, 250], 'conf': 0.95},
            {'xyxy': [150, 150, 180, 250], 'conf': 0.95},
        ],
        'error_bar': [
            {'xyxy': [63, 80, 67, 120], 'conf': 1.0},  # for bar 0, span=40px, half=20px
        ],
    }

    # Linear calibration: val = (250 - y) * 0.5 -> 20px = 10.0 units
    cal_func = lambda y: (250.0 - y) * 0.5
    calib = {'primary': MagicMock(func=cal_func)}
    baselines = MagicMock(baselines=[MagicMock(axis_id='y', value=250.0)])

    bars = handler.extract_values(img, detections, calib, baselines, 'vertical')
    assert len(bars) == 2
    # Bar 0 should have error_bar with margin=10.0
    assert 'error_bar' in bars[0]
    assert bars[0]['error_bar']['bbox'] == [63, 80, 67, 120]
    assert pytest.approx(bars[0]['error_bar']['margin'], 0.01) == 10.0
    # Bar 1 has no error bar
    assert 'error_bar' not in bars[1] or bars[1]['error_bar'] is None
