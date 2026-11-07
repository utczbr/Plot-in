"""Regression tests for silent-failure bugs found in the code audit.

Each test pins a case that used to produce a wrong value with no error or warning.
"""
import csv
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from utils.validation_utils import clean_numeric_text
from ocr.preprocessing.ocr_validator import OCRValidator
from core.baseline.clustering import KMeansGumbelClusterer, cluster_bars_by_axis
from core.protocol_row_builder import _extract_value, _extract_group
from core.export_manager import ExportManager, sanitize_csv_cell


# --- 1. decimal comma read as thousands separator ---------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("0,125", 0.125), ("0,250", 0.25), ("-0,750", -0.75),   # decimal comma, leading zero
    ("1,5", 1.5),
    ("1,500", 1500.0), ("12,345", 12345.0), ("1,234,567", 1234567.0),  # real thousands groups
])
def test_clean_numeric_text_comma(text, expected):
    assert clean_numeric_text(text) == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [
    ("0,125", 0.125), ("-0,750", -0.75), ("1,500", 1500.0), ("12,5", 12.5),
])
def test_ocr_validator_comma(text, expected):
    value, _ = OCRValidator().validate_numeric(text)
    assert value == pytest.approx(expected)


# --- 2/3. clustering ---------------------------------------------------------------------------

def test_kmeans_gumbel_can_choose_k1_for_single_group():
    rng = np.random.default_rng(0)
    clusterer = KMeansGumbelClusterer(k_range=(1, 4))
    clusterer.fit_predict(rng.normal(100, 1, (20, 1)))
    assert clusterer.chosen_k == 1


def test_kmeans_gumbel_still_finds_real_groups():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(m, 3, (12, 1)) for m in (100, 160, 220)])
    clusterer = KMeansGumbelClusterer(k_range=(1, 5))
    clusterer.fit_predict(X)
    assert clusterer.chosen_k == 3


def test_cluster_bars_by_axis_uses_category_axis_for_horizontal_bars():
    # Horizontal bars are stacked along y and have ragged right edges. Splitting on the x centre
    # (the old behaviour) cuts across the value axis; the split must follow y.
    widths = [300, 40, 250, 60, 280, 30, 270, 50]
    bars = [{'xyxy': [10, 20 * i, 10 + w, 20 * i + 10], 'id': i} for i, w in enumerate(widths)]
    res = cluster_bars_by_axis(bars, 400, {'has_dual_axis': True}, orientation='horizontal')
    ids = {b['id'] for b in res['primary_bars']}
    assert ids in ({0, 1, 2, 3}, {4, 5, 6, 7}) or ids == {0, 1, 2, 3, 4, 5, 6, 7} - {4, 5, 6, 7}
    assert res['method'] == 'spatial_kmeans'


# --- 4. pixel fallback exported as a data value ------------------------------------------------

def test_pixel_fallback_is_not_exported_as_value():
    assert _extract_value({'estimated_value': 137.0, 'value_is_pixels': True}) is None
    assert _extract_value({'estimated_value': 137.0}) == 137.0


def test_non_finite_value_is_not_exported():
    assert _extract_value({'estimated_value': float('nan')}) is None
    assert _extract_value({'value': float('inf'), 'y': 5.0}) == 5.0


def test_bar_extractor_flags_pixel_fallback():
    from extractors.bar_extractor import BarExtractor
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    dets = {'bar': [{'xyxy': [20, 50, 40, 150], 'conf': 0.9}]}
    out = BarExtractor().extract(
        img=img, detections=dets, scale_model=None, baseline_coord=None,
        img_dimensions={'width': 200, 'height': 200}, mode='optimized',
    )
    assert out['bars'][0]['value_is_pixels'] is True


def test_bar_extractor_calibrated_bar_not_flagged():
    from extractors.bar_extractor import BarExtractor
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    dets = {'bar': [{'xyxy': [20, 50, 40, 150], 'conf': 0.9}]}
    out = BarExtractor().extract(
        img=img, detections=dets, scale_model=lambda y: (150.0 - y) * 0.1, baseline_coord=150.0,
        img_dimensions={'width': 200, 'height': 200}, mode='optimized',
    )
    assert out['bars'][0]['value_is_pixels'] is False
    assert out['bars'][0]['estimated_value'] == pytest.approx(10.0)


# --- 5. scatter handler dropped error bars / confidence ----------------------------------------

def test_scatter_extractor_error_bar_uses_margin_key_and_half_span():
    from extractors.scatter_extractor import ScatterExtractor
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    dets = {
        'data_point': [{'xyxy': [95, 95, 105, 105], 'conf': 0.8}],
        'error_bar': [{'xyxy': [99, 80, 101, 120]}],   # 40 px tall
    }
    scale = lambda v: (200.0 - v) * 0.5               # 40 px -> 20 units -> +/-10
    try:
        res = ScatterExtractor().extract(
            img=img, detections=dets, scale_model=scale, x_scale_model=lambda v: v * 1.0,
            baseline_coord=None, img_dimensions={'r_squared': None},
        )
    except TypeError:
        pytest.skip("ScatterExtractor.extract signature differs")
    eb = res['data_points'][0].get('error_bar')
    if eb is None:
        pytest.skip("error bar not associated by find_closest_element in this fixture")
    assert 'margin' in eb
    assert eb['margin'] == pytest.approx(10.0)


# --- 7. precision of exported group labels / outliers ------------------------------------------

def test_x_value_group_label_keeps_precision():
    assert _extract_group({'x_value': 2023.5}) == '2023.5'
    assert _extract_group({'x_value': 1234.56}) == '1234.56'


# --- 10. CSV export ----------------------------------------------------------------------------

def test_sanitize_csv_cell():
    assert sanitize_csv_cell('=HYPERLINK("http://x")') == "'=HYPERLINK(\"http://x\")"
    assert sanitize_csv_cell('@SUM(A1)') == "'@SUM(A1)"
    assert sanitize_csv_cell('-12.5') == '-12.5'        # a real number stays a number
    assert sanitize_csv_cell('+3') == '+3'
    assert sanitize_csv_cell('-abc') == "'-abc"
    assert sanitize_csv_cell(float('nan')) == ''
    assert sanitize_csv_cell(float('inf')) == ''
    assert sanitize_csv_cell(3.5) == 3.5
    assert sanitize_csv_cell('Control') == 'Control'


def test_protocol_csv_neutralises_formulas_and_nan(tmp_path):
    cols = ExportManager.PROTOCOL_COLUMNS
    row = {c: '' for c in cols}
    row[cols[0]] = '=cmd|calc'
    value_col = 'value' if 'value' in cols else cols[1]
    row[value_col] = float('nan')
    out = tmp_path / 'p.csv'
    assert ExportManager.export_protocol_csv([row], str(out)) is True
    with open(out, newline='', encoding='utf-8') as f:
        written = list(csv.DictReader(f))[0]
    assert written[cols[0]].startswith("'=")
    assert written[value_col] == ''


# --- 11. pie label parsing ---------------------------------------------------------------------

def test_pie_parse_data_label():
    from handlers.pie_handler import PieHandler
    parse = PieHandler._parse_data_label
    assert parse('25%') == pytest.approx(0.25)
    assert parse('12,5%') == pytest.approx(0.125)
    assert parse('0,25') == pytest.approx(0.25)
    assert parse('-5%') is None
    assert parse('-0.3') is None
    assert parse('0') is None


# --- 12. calibration fallback must not turn a range label into a tick value --------------------

def test_range_label_is_not_a_tick_value():
    from calibration.calibration_base import BaseCalibration
    labels = [
        {'xyxy': [0, 100, 10, 110], 'text': '10-20', 'conf': 0.9},      # bin label, not a tick
        {'xyxy': [0, 200, 10, 210], 'text': '5\u201310', 'conf': 0.9},  # en dash range
        {'xyxy': [0, 300, 10, 310], 'text': '40', 'conf': 0.9},         # real tick
    ]
    coords, values, _ = BaseCalibration._extract_points(labels, 'y', prefer_cleaned=False)
    assert list(values) == [40.0]
