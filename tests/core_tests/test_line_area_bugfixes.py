"""Regression tests for line/area extraction, label classification and pipeline fixes."""
from __future__ import annotations

import numpy as np

from core.chart_registry import resolve_chart_elements
from core.classifiers.line_chart_classifier import LineChartClassifier
from core.protocol_row_builder import build_protocol_rows
from extractors.area_seg_extractor import AreaSegExtractor
from extractors.series_linker import compute_cielab_color_distance, fit_anchor_guided_curve
from extractors.tick_label_associator import TickLabelAssociator


def test_cielab_gray_has_zero_chroma_distance():
    gray = np.array([128, 128, 128], dtype=np.uint8)
    assert compute_cielab_color_distance(gray, gray) < 1e-3
    # Different greys differ only in lightness
    d = compute_cielab_color_distance(np.array([50, 50, 50]), np.array([200, 200, 200]))
    assert 50 < d < 100


def test_fit_curve_sorted_and_unique_x():
    xs = np.repeat(np.arange(0, 40), 2).astype(float)  # vertical steps: 2 pts per column
    ys = xs * 0.5 + np.tile([0.0, 1.0], 40)
    curve = fit_anchor_guided_curve(np.column_stack([xs, ys]))
    assert np.all(np.diff(curve[:, 0]) >= -1e-9)


def test_area_non_zero_axis_minimum_and_negative_values():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:90, 10:190] = 1  # flat top at y=40, bottom y=89
    det = {'area_series': [{'xyxy': [0, 0, 200, 100], 'conf': 0.9, 'mask': mask}]}
    # Axis starts at 100: value(y) = 100 + (90 - y); baseline pixel y=90
    scale = lambda y: 100.0 + (90.0 - y)
    res = AreaSegExtractor().extract(
        img=np.zeros((120, 220, 3), np.uint8), detections=det,
        scale_model=scale, baseline_coord=90.0, img_dimensions={},
    )
    vals = [p['estimated_value'] for p in res['data_points']]
    assert all(abs(v - 150.0) < 1e-6 for v in vals)  # NOT shifted by -100

    # Below the axis-zero baseline → negative values preserved
    scale_neg = lambda y: 30.0 - y
    res = AreaSegExtractor().extract(
        img=np.zeros((120, 220, 3), np.uint8), detections=det,
        scale_model=scale_neg, baseline_coord=90.0, img_dimensions={},
    )
    assert all(p['estimated_value'] < 0 for p in res['data_points'])
    assert res['auc'][0]['auc_units'] == 'value*pixel_x'


def test_area_overlapping_layers_not_treated_as_stacked():
    a = np.zeros((100, 200), np.uint8); a[30:90, 10:190] = 1
    b = np.zeros((100, 200), np.uint8); b[60:90, 10:190] = 1  # sits on the baseline
    det = {'area_series': [
        {'xyxy': [0, 0, 200, 100], 'conf': 0.9, 'mask': a},
        {'xyxy': [0, 0, 200, 100], 'conf': 0.9, 'mask': b},
    ]}
    res = AreaSegExtractor().extract(
        img=np.zeros((120, 220, 3), np.uint8), detections=det,
        scale_model=lambda y: 90.0 - y, baseline_coord=90.0, img_dimensions={},
    )
    assert [r['stacked'] for r in res['auc']] == [False, False]


def test_resolve_chart_elements_fallbacks():
    assert resolve_chart_elements('line', {'data_point': [], 'line_series': [{'a': 1}]}) == [{'a': 1}]
    assert resolve_chart_elements('line', {'data_marker': [{'m': 1}]}) == [{'m': 1}]
    assert resolve_chart_elements('area', {'area_series': [{'s': 1}]}) == [{'s': 1}]
    assert resolve_chart_elements('line', {'data_point': [{'p': 1}], 'line_series': [{'a': 1}]}) == [{'p': 1}]


def test_line_classifier_axis_split_and_cleanedvalue():
    w, h = 1000, 800
    def lab(text, cx, cy):
        return {'text': text, 'xyxy': [cx - 15, cy - 8, cx + 15, cy + 8], 'conf': 0.9}
    labels = [
        lab('0', 120, 620),     # bottom-most Y label (nx=.12, ny=.78) must stay on Y
        lab('50', 120, 300),
        lab('100', 120, 100),
        lab('2020', 300, 700),  # X numeric labels
        lab('2021', 500, 700),
        lab('2022', 700, 700),
    ]
    points = [{'xyxy': [290 + 40 * i, 200 + 30 * i, 300 + 40 * i, 210 + 30 * i]} for i in range(5)]
    res = LineChartClassifier().classify(
        axis_labels=labels, chart_elements=points, img_width=w, img_height=h,
        orientation='vertical',
    )
    y_txt = {l['text'] for l in res.y_scale_labels}
    x_txt = {l['text'] for l in res.x_scale_labels}
    assert '0' in y_txt
    assert {'2020', '2021', '2022'} <= x_txt
    # X labels become tick labels, not scale labels
    assert {l['text'] for l in res.tick_labels} >= {'2020', '2021', '2022'}
    assert not ({'2020', '2021', '2022'} & {l['text'] for l in res.scale_labels})
    assert all('cleanedvalue' in l for l in res.y_scale_labels)


def test_tick_associator_accepts_x_axis_scale_labels():
    elems = [{'bbox': [100, 100, 110, 110], 'type': 'line_segment'}]
    lbl = {'text': '2020', 'xyxy': [90, 300, 120, 316], 'label_type': 'scale_label', 'axis': 'x'}
    out = TickLabelAssociator.associate(elems, {}, 'vertical', axis_labels=[lbl])
    assert out[0].get('tick_label')
    ylbl = {'text': '50', 'xyxy': [10, 100, 40, 116], 'label_type': 'scale_label', 'axis': 'y'}
    out = TickLabelAssociator.associate(
        [{'bbox': [100, 100, 110, 110]}], {}, 'vertical', axis_labels=[ylbl])
    assert not out[0].get('tick_label')


def test_protocol_rows_line_value_x_and_legend():
    result = {
        'chart_type': 'line',
        'elements': [{
            'type': 'line_segment', 'bbox': [0, 0, 4, 4], 'value': 3.0,
            'x_value': 2021.0, 'series_id': 1, 'index': 0,
        }],
    }
    row = build_protocol_rows(result)[0]
    assert row['value_x'] == 2021.0
    assert row['outcome'] == 'Series 2'
