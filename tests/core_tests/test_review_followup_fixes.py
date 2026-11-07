"""Regression tests for the second review round (numeric parsing, value_is_pixels chain,
dual-axis error-bar calibration, K selection robustness, row-edit/Data-tab edit paths)."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.label_overrides import parse_label_number
from ocr.preprocessing.ocr_validator import OCRValidator
from utils.validation_utils import clean_numeric_text, normalize_decimal_separators
from core.baseline.clustering import KMeansGumbelClusterer


# --- 5. one numeric parser behaviour across the three entry points ------------------------------

@pytest.mark.parametrize("text,expected", [
    ("$10^{2}$", 100.0), (r"$\mathdefault{10^{2}}$", 100.0), ("10^2", 100.0),   # was 102.0
    ("1.234,56", 1234.56), ("1,234.56", 1234.56), ("€1.234,56", 1234.56),       # was 1.23456
    ("-¥100", -100.0), ("¥100", 100.0),                                         # was None
    ("0,125", 0.125), ("1,500", 1500.0), ("$12.5", 12.5), ("1O0", 100.0),
])
def test_parse_label_number(text, expected):
    assert parse_label_number(text) == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [
    ("1.234,56", 1234.56), ("1,234.56", 1234.56), ("-¥100", -100.0), ("0,125", 0.125), ("12,5%", 12.5),
])
def test_validator_and_clean_numeric_text_agree(text, expected):
    assert OCRValidator().validate_numeric(text)[0] == pytest.approx(expected)
    assert clean_numeric_text(text) == pytest.approx(expected)


def test_validator_does_not_concatenate_exponent_digits():
    assert OCRValidator().validate_numeric("10^{2}")[0] is None  # was 102.0


def test_normalize_decimal_separators():
    assert normalize_decimal_separators("1.234.567,8") == "1234567.8"
    assert normalize_decimal_separators("0,5") == "0.5"
    assert normalize_decimal_separators("12,345") == "12345"


# --- 6. K selection -----------------------------------------------------------------------------

JITTER = [100, 101, 102, 103, 104, 100.5, 101.5, 102.5, 103.5, 104,
          300, 301, 302, 303, 304, 300.5, 301.5, 302.5, 325.0]   # last bar is 21 px off its group


@pytest.mark.parametrize("k_range", [(1, 2), (1, 3), (1, 4)])
def test_one_jittered_bar_neither_blocks_nor_adds_a_baseline(k_range):
    c = KMeansGumbelClusterer(k_range=k_range)
    c.fit_predict(np.array(JITTER, dtype=float).reshape(-1, 1))
    assert c.chosen_k == 2


def test_kmeans_gumbel_does_not_refit_chosen_k(monkeypatch):
    import core.baseline.clustering as mod
    calls = []
    real = mod.KMeans

    class Spy(real):
        def fit_predict(self, *a, **k):
            calls.append(self.n_clusters)
            return super().fit_predict(*a, **k)

    monkeypatch.setattr(mod, "KMeans", Spy)
    c = KMeansGumbelClusterer(k_range=(1, 3))
    c.fit_predict(np.array(JITTER, dtype=float).reshape(-1, 1))
    assert calls.count(2) == 1


def test_small_n_still_works():
    c = KMeansGumbelClusterer(k_range=(1, 3))
    assert list(c.fit_predict(np.array([[1.0], [2.0]]))) == [0, 0]
    assert c.chosen_k == 1


# --- 3. value_is_pixels chain ------------------------------------------------------------------

def _img():
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_bar_data_label_clears_pixel_flag():
    from extractors.bar_extractor import BarExtractor
    bar = {'xyxy': [20, 50, 40, 150], 'conf': 0.9,
           'associated_data_label': {'label': {'text': '7.5', 'cleanedvalue': 7.5, 'xyxy': [20, 30, 40, 48]}}}
    out = BarExtractor().extract(img=_img(), detections={'bar': [bar]}, scale_model=None,
                                 baseline_coord=None, img_dimensions={'width': 200, 'height': 200},
                                 mode='optimized')
    b = out['bars'][0]
    assert b['estimated_value'] == 7.5 and b['value_is_pixels'] is False


def test_line_handler_forwards_value_is_pixels():
    from handlers.line_handler import LineHandler

    def failing_scale(_y):
        raise ValueError("calibration broke")   # extractor falls back to a pixel distance

    dets = {'data_point': [{'xyxy': [20, 60, 30, 70], 'conf': 0.9}]}
    calib = {'primary': MagicMock(func=failing_scale)}
    base = MagicMock(baselines=[MagicMock(axis_id='y', value=150.0)])
    out = LineHandler(MagicMock(), MagicMock()).extract_values(_img(), dets, calib, base, 'vertical')
    rows = [e for e in (out or []) if isinstance(e, dict) and e.get('type') == 'line_segment']
    assert rows, "LineHandler produced no entries"
    assert all(e['value_is_pixels'] is True for e in rows)   # flag must reach result['elements']


def test_area_extractor_flags_pixels():
    from extractors.area_extractor import AreaExtractor
    dets = {'data_point': [{'xyxy': [20, 60, 30, 70], 'conf': 0.9}]}
    res = AreaExtractor().extract(img=_img(), detections=dets, scale_model=None, baseline_coord=150.0,
                                  img_dimensions={'width': 200, 'height': 200})
    pts = res.get('data_points') or []
    assert pts and pts[0]['value_is_pixels'] is True


def test_manual_value_edit_clears_pixel_flag_and_data_tab_hides_pixels():
    from visual import data_tab_schema as dts
    from core.protocol_row_builder import _extract_value
    el = {'estimated_value': 120.0, 'value_is_pixels': True}
    assert dts._value_or_estimated(el) is None          # no pixel length shown as a value
    dts._apply_bar_field(el, "value", "4.2")
    assert el['value_is_pixels'] is False
    assert dts._value_or_estimated(el) == pytest.approx(4.2)
    assert _extract_value(el) == pytest.approx(4.2)


def test_line_edit_clears_pixel_flag():
    from visual import data_tab_schema as dts
    el = {'value': 120.0, 'value_is_pixels': True}
    dts._apply_line_area_field(el, "value", "3")
    assert el['value_is_pixels'] is False


def test_row_edit_service_value_clears_pixel_flag():
    from visual.row_edit_service import RowEditService
    el = {'value': 120.0, 'estimated_value': 120.0, 'value_is_pixels': True}
    RowEditService._write_field_to_element(el, {'value': 4.0}, 'bar', 'value', 4.0)
    assert el['value_is_pixels'] is False and el['value'] == 4.0


# --- 9. editing X keeps group (tick_label) and x_value in step ---------------------------------

def test_x_edit_updates_tick_label():
    from visual import data_tab_schema as dts
    el = {'x_value': 2020.0, 'tick_label': {'text': '2020'}}
    dts._apply_line_area_field(el, "x", "2025")
    assert el['x_value'] == 2025.0 and el['tick_label']['text'] == '2025'


def test_non_numeric_x_edit_is_not_dropped():
    from visual import data_tab_schema as dts
    el = {'x_value': 2020.0, 'tick_label': {'text': '2020'}}
    dts._apply_line_area_field(el, "x", "Control")
    assert el['tick_label']['text'] == 'Control' and el['x_value'] == 2020.0


# --- 4. dual-axis error bars use the secondary calibration --------------------------------------

def test_secondary_axis_error_bar_margin_uses_secondary_model():
    from handlers.bar_handler import BarHandler
    handler = BarHandler(MagicMock(), MagicMock())
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    dets = {
        'bar': [{'xyxy': [50, 100, 80, 250], 'conf': 0.95},     # left  -> primary
                {'xyxy': [300, 100, 330, 250], 'conf': 0.95}],  # right -> secondary
        'error_bar': [{'xyxy': [63, 80, 67, 120], 'conf': 1.0},
                      {'xyxy': [313, 80, 317, 120], 'conf': 1.0}],   # both 40 px tall
    }
    calib = {'primary': MagicMock(func=lambda y: (250.0 - y) * 0.5),         # 20 px -> 10.0
             'secondary': MagicMock(func=lambda y: (250.0 - y) * 0.005)}     # 20 px -> 0.1
    base = MagicMock(baselines=[MagicMock(axis_id='y', value=250.0), MagicMock(axis_id='y2', value=250.0)])
    bars = handler.extract_values(img, dets, calib, base, 'vertical')
    margins = {b['axis']: b['error_bar']['margin'] for b in bars if b.get('error_bar')}
    assert margins['primary'] == pytest.approx(10.0)
    assert margins['secondary'] == pytest.approx(0.1)     # was 10.0 (primary model)
    assert sorted(b['index'] for b in bars) == [0, 1]
