"""Unit tests for src/core/protocol_row_builder.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from core.protocol_row_builder import (
    build_protocol_rows,
    merge_context_into_rows,
    _extract_value,
    _is_synthetic,
    _match_outcome,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _bar_element(index=0, estimated_value=12.5, confidence=0.9, tick_text="Group A"):
    return {
        'index': index,
        'xyxy': [10, 20, 30, 100],
        'confidence': confidence,
        'estimated_value': estimated_value,
        'tick_label': {'text': tick_text, 'bbox': [5, 110, 35, 120]},
        'text_label': '',
        'error_bar': None,
    }


def _line_element(index=0, value=7.3, confidence=0.85):
    return {
        'index': index,
        'type': 'line_segment',
        'bbox': [50, 60, 55, 65],
        'position': 62.5,
        'value': value,
        'orientation': 'vertical',
        'confidence': confidence,
    }


def _scatter_element(index=0, x=3.14, y=9.81):
    return {
        'index': index,
        'type': 'point',
        'bbox': [100, 200, 110, 210],
        'x': x,
        'y': y,
        'center': [105, 205],
    }


def _area_element(index=0, value=5.5, confidence=0.88):
    return {
        'index': index,
        'type': 'data_point',
        'bbox': [20, 30, 25, 35],
        'value': value,
        'confidence': confidence,
    }


def _pie_element(index=0, label="Slice A", value=0.42, confidence=0.93):
    return {
        'index': index,
        'type': 'pie_slice',
        'xyxy': [60, 60, 120, 120],
        'label': label,
        'value': value,
        'confidence': confidence,
    }


def _pipeline_result(elements, chart_type='bar', provenance=None, baselines=None, metadata=None):
    return {
        'image_file': 'chart.png',
        'chart_type': chart_type,
        'elements': elements,
        '_provenance': provenance,
        'baselines': baselines or [],
        'metadata': metadata or {},
        'detections': {},
    }


def _context(outcomes=None, groups=None, units=None, error_bar_type=''):
    return {
        'outcomes': outcomes or [],
        'groups': groups or [],
        'units': units or {},
        'error_bar_type': error_bar_type,
    }


# ---------------------------------------------------------------------------
# _extract_value
# ---------------------------------------------------------------------------

class TestExtractValue:
    def test_value_key_first(self):
        assert _extract_value({'value': 1.0, 'estimated_value': 2.0}) == 1.0

    def test_estimated_value_fallback(self):
        assert _extract_value({'estimated_value': 2.0}) == 2.0

    def test_y_fallback(self):
        assert _extract_value({'y': 3.0}) == 3.0

    def test_none_when_missing(self):
        assert _extract_value({}) is None

    def test_skips_non_numeric(self):
        assert _extract_value({'value': 'not_a_number', 'y': 4.0}) == 4.0


# ---------------------------------------------------------------------------
# _is_synthetic
# ---------------------------------------------------------------------------

class TestIsSynthetic:
    def test_summary_type(self):
        assert _is_synthetic({'type': 'summary', 'bbox': [0, 0, 1, 1]}) is True

    def test_no_bbox_or_xyxy(self):
        assert _is_synthetic({'type': 'data', 'value': 5}) is True

    def test_has_bbox(self):
        assert _is_synthetic({'bbox': [0, 0, 1, 1]}) is False

    def test_has_xyxy(self):
        assert _is_synthetic({'xyxy': [0, 0, 1, 1]}) is False


# ---------------------------------------------------------------------------
# _match_outcome
# ---------------------------------------------------------------------------

class TestMatchOutcome:
    def test_exact_match(self):
        assert _match_outcome(['Body weight (g)'], ['Body weight (g)']) == 'Body weight (g)'

    def test_substring_match(self):
        assert _match_outcome(['Y-axis: Body weight (g)'], ['Body weight (g)']) == 'Body weight (g)'

    def test_case_insensitive(self):
        assert _match_outcome(['body weight (g)'], ['Body Weight (g)']) == 'Body Weight (g)'

    def test_no_match_returns_empty(self):
        assert _match_outcome(['Temperature'], ['Weight', 'Volume']) == ''

    def test_single_outcome_default(self):
        assert _match_outcome(['Something'], ['Body weight (g)']) == 'Body weight (g)'

    def test_empty_returns_empty(self):
        assert _match_outcome([], ['Weight']) == ''
        assert _match_outcome(['Title'], []) == ''


# ---------------------------------------------------------------------------
# build_protocol_rows — bar elements
# ---------------------------------------------------------------------------

class TestBuildFromBarElements:
    def test_basic_mapping(self):
        result = _pipeline_result([_bar_element()])
        rows = build_protocol_rows(result)
        assert len(rows) == 1
        row = rows[0]
        assert row['chart_type'] == 'bar'
        assert row['value'] == 12.5
        assert row['confidence'] == 0.9
        assert row['group'] == 'Group A'
        assert row['review_status'] == 'auto'
        assert row['_original'] is None
        assert row['source_file'] == 'chart.png'

    def test_multiple_bars(self):
        elements = [_bar_element(i, estimated_value=float(i)) for i in range(5)]
        rows = build_protocol_rows(_pipeline_result(elements))
        assert len(rows) == 5
        assert [r['value'] for r in rows] == [0.0, 1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# build_protocol_rows — line elements
# ---------------------------------------------------------------------------

class TestBuildFromLineElements:
    def test_value_key_used(self):
        result = _pipeline_result([_line_element(value=7.3)], chart_type='line')
        rows = build_protocol_rows(result)
        assert len(rows) == 1
        assert rows[0]['value'] == 7.3


# ---------------------------------------------------------------------------
# build_protocol_rows — scatter elements
# ---------------------------------------------------------------------------

class TestBuildFromScatterElements:
    def test_y_as_value_x_as_series(self):
        result = _pipeline_result([_scatter_element(x=3.14, y=9.81)], chart_type='scatter')
        rows = build_protocol_rows(result)
        assert len(rows) == 1
        assert rows[0]['value'] == 9.81
        assert rows[0]['series_id'] == '3.14'


# ---------------------------------------------------------------------------
# build_protocol_rows — area elements
# ---------------------------------------------------------------------------

class TestBuildFromAreaElements:
    def test_area_value_mapping(self):
        result = _pipeline_result([_area_element(value=5.5)], chart_type='area')
        rows = build_protocol_rows(result)
        assert len(rows) == 1
        assert rows[0]['value'] == 5.5


# ---------------------------------------------------------------------------
# build_protocol_rows — pie elements
# ---------------------------------------------------------------------------

class TestBuildFromPieElements:
    def test_pie_label_maps_to_group(self):
        result = _pipeline_result([_pie_element(label="Item 3")], chart_type='pie')
        rows = build_protocol_rows(result)
        assert len(rows) == 1
        assert rows[0]['group'] == 'Item 3'
        assert rows[0]['value'] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Synthetic elements skipped
# ---------------------------------------------------------------------------

class TestSyntheticSkipped:
    def test_summary_skipped(self):
        elements = [
            _bar_element(0),
            {'type': 'summary', 'total': 100},  # no bbox
        ]
        rows = build_protocol_rows(_pipeline_result(elements))
        assert len(rows) == 1

    def test_no_bbox_skipped(self):
        elements = [
            _bar_element(0),
            {'index': 99, 'value': 42},  # no bbox/xyxy
        ]
        rows = build_protocol_rows(_pipeline_result(elements))
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_provenance_populated(self):
        prov = {'source_document': '/data/report.pdf', 'page_index': 3, 'figure_id': 'report_p003_f01'}
        result = _pipeline_result([_bar_element()], provenance=prov)
        rows = build_protocol_rows(result)
        assert rows[0]['source_file'] == '/data/report.pdf'
        assert rows[0]['page_index'] == 3

    def test_provenance_absent(self):
        result = _pipeline_result([_bar_element()])
        rows = build_protocol_rows(result)
        assert rows[0]['source_file'] == 'chart.png'
        assert rows[0]['page_index'] is None


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

class TestContext:
    def test_context_populates_outcome_unit(self):
        metadata = {'label_classification': {'axis_titles': [{'text': 'Body weight (g)'}]}}
        result = _pipeline_result([_bar_element()], metadata=metadata)
        ctx = _context(outcomes=['Body weight (g)'], units={'Body weight (g)': 'g'}, error_bar_type='SEM')
        rows = build_protocol_rows(result, ctx)
        assert rows[0]['outcome'] == 'Body weight (g)'
        assert rows[0]['unit'] == 'g'
        assert rows[0]['error_bar_type'] == 'SEM'

    def test_context_none(self):
        rows = build_protocol_rows(_pipeline_result([_bar_element()]), None)
        assert rows[0]['outcome'] == ''
        assert rows[0]['unit'] == ''
        assert rows[0]['error_bar_type'] == ''


# ---------------------------------------------------------------------------
# Error bar / baseline
# ---------------------------------------------------------------------------

class TestErrorBarBaseline:
    def test_error_bar_mapping(self):
        elem = _bar_element()
        elem['error_bar'] = {'margin': 1.5, 'lower_bound': 11.0, 'upper_bound': 14.0}
        rows = build_protocol_rows(_pipeline_result([elem]))
        assert rows[0]['error_bar_value'] == 1.5

    def test_baseline_mapping(self):
        baselines = [{'axis_id': 'y', 'value': 50.0, 'type': 'zero'}]
        result = _pipeline_result([_bar_element()], baselines=baselines)
        rows = build_protocol_rows(result)
        assert rows[0]['baseline_value'] == 50.0

    def test_no_baseline(self):
        rows = build_protocol_rows(_pipeline_result([_bar_element()]))
        assert rows[0]['baseline_value'] is None


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_elements(self):
        assert build_protocol_rows(_pipeline_result([])) == []

    def test_non_dict_elements_skipped(self):
        rows = build_protocol_rows(_pipeline_result([_bar_element(), "garbage", 42]))
        assert len(rows) == 1

    def test_element_index_preserved(self):
        elem = _bar_element(index=7)
        rows = build_protocol_rows(_pipeline_result([elem]))
        assert rows[0]['element_index'] == 7

    def test_review_status_default(self):
        rows = build_protocol_rows(_pipeline_result([_bar_element()]))
        assert rows[0]['review_status'] == 'auto'

    def test_original_is_none(self):
        rows = build_protocol_rows(_pipeline_result([_bar_element()]))
        assert rows[0]['_original'] is None


# ---------------------------------------------------------------------------
# merge_context_into_rows
# ---------------------------------------------------------------------------

class TestMergeContext:
    def test_untouched_rows_regenerated(self):
        # Use metadata with axis title so outcome matching works
        metadata = {'label_classification': {'axis_titles': [{'text': 'Weight'}]}}
        result = _pipeline_result([_bar_element()], metadata=metadata)
        old_rows = build_protocol_rows(result, None)
        assert old_rows[0]['outcome'] == ''

        ctx = _context(outcomes=['Weight'])
        merged = merge_context_into_rows(old_rows, result, ctx)
        assert len(merged) == 1
        assert merged[0]['outcome'] == 'Weight'

    def test_edited_rows_preserve_values(self):
        metadata = {'label_classification': {'axis_titles': [{'text': 'Weight'}]}}
        result = _pipeline_result([_bar_element()], metadata=metadata)
        old_rows = build_protocol_rows(result, None)
        # Simulate user edit
        old_rows[0]['value'] = 99.9
        old_rows[0]['review_status'] = 'corrected'
        old_rows[0]['_original'] = {'value': 12.5, 'review_status': 'auto'}

        ctx = _context(outcomes=['Weight'], units={'Weight': 'kg'})
        merged = merge_context_into_rows(old_rows, result, ctx)
        assert len(merged) == 1
        assert merged[0]['value'] == 99.9  # user edit preserved
        assert merged[0]['outcome'] == 'Weight'  # context filled empty field
        assert merged[0]['unit'] == 'kg'  # context filled empty field
        assert merged[0]['review_status'] == 'corrected'  # preserved

    def test_edited_rows_dont_overwrite_filled_fields(self):
        metadata = {'label_classification': {'axis_titles': [{'text': 'OldOutcome'}]}}
        result = _pipeline_result([_bar_element()], metadata=metadata)
        old_rows = build_protocol_rows(result, _context(outcomes=['OldOutcome']))
        assert old_rows[0]['outcome'] == 'OldOutcome'
        # Simulate user edit
        old_rows[0]['_original'] = {'outcome': 'OldOutcome'}
        old_rows[0]['review_status'] = 'corrected'

        ctx = _context(outcomes=['NewOutcome'])
        merged = merge_context_into_rows(old_rows, result, ctx)
        # outcome was already filled ('OldOutcome'), should NOT be overwritten
        assert merged[0]['outcome'] == 'OldOutcome'
