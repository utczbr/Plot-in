"""Golden snapshot tests for protocol columns and merge_context_into_rows backfill (T-012.0)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from core.protocol_row_builder import (
    _CONTEXT_FIELDS,
    build_protocol_rows,
    get_protocol_columns,
    merge_context_into_rows,
)


# ---------------------------------------------------------------------------
# Golden column lists by chart type
# ---------------------------------------------------------------------------

GOLDEN_COMMON_HEAD = [
    ('source_file', 'Source', False),
    ('chart_type', 'Type', False),
    ('group', 'Tick label', False),
    ('outcome', 'Legend', False),
]

GOLDEN_COMMON_TAIL = [
    ('unit', 'Unit', False),
    ('confidence', 'Conf.', True),
    ('review_status', 'Status', False),
    ('notes', 'Notes', False),
]

GOLDEN_COLUMNS_BY_TYPE = {
    'bar': GOLDEN_COMMON_HEAD + [
        ('value', 'Value', True),
        ('error_bar_value', 'Error Bar', True),
        ('baseline_value', 'Baseline', True),
    ] + GOLDEN_COMMON_TAIL,

    'line': GOLDEN_COMMON_HEAD + [
        ('value', 'Value', True),
        ('error_bar_value', 'Error Bar', True),
        ('baseline_value', 'Baseline', True),
    ] + GOLDEN_COMMON_TAIL,

    'scatter': GOLDEN_COMMON_HEAD + [
        ('value_x', 'X', True),
        ('value', 'Y', True),
    ] + GOLDEN_COMMON_TAIL,

    'box': GOLDEN_COMMON_HEAD + [
        ('whisker_low', 'Min (Whisker)', True),
        ('q1', 'Q1', True),
        ('median', 'Median', True),
        ('q3', 'Q3', True),
        ('whisker_high', 'Max (Whisker)', True),
        ('outliers_str', 'Outliers', False),
    ] + GOLDEN_COMMON_TAIL,

    'pie': GOLDEN_COMMON_HEAD + [
        ('value', 'Proportion', True),
        ('percent', '%', True),
    ] + GOLDEN_COMMON_TAIL,

    'heatmap': GOLDEN_COMMON_HEAD + [
        ('row_label', 'Row', False),
        ('col_label', 'Col', False),
        ('value', 'Value', True),
    ] + GOLDEN_COMMON_TAIL,

    'area': GOLDEN_COMMON_HEAD + [
        ('value', 'Value', True),
        ('error_bar_value', 'Error Bar', True),
        ('baseline_value', 'Baseline', True),
    ] + GOLDEN_COMMON_TAIL,

    'histogram': GOLDEN_COMMON_HEAD + [
        ('value', 'Value', True),
        ('error_bar_value', 'Error Bar', True),
        ('baseline_value', 'Baseline', True),
    ] + GOLDEN_COMMON_TAIL,
}


class TestProtocolColumnsGolden:
    """Assert canonical protocol column schema across all supported chart types."""

    @pytest.mark.parametrize('chart_type', ['bar', 'line', 'scatter', 'box', 'pie', 'heatmap'])
    def test_get_protocol_columns_exact_sequence(self, chart_type: str):
        actual = get_protocol_columns(chart_type)
        expected = GOLDEN_COLUMNS_BY_TYPE[chart_type]
        assert actual == expected, f"Column sequence mismatch for chart type: {chart_type}"

    @pytest.mark.parametrize('chart_type', ['area', 'histogram', 'unknown_type'])
    def test_get_protocol_columns_default_fallback(self, chart_type: str):
        actual = get_protocol_columns(chart_type)
        expected = GOLDEN_COLUMNS_BY_TYPE.get(chart_type, GOLDEN_COLUMNS_BY_TYPE['bar'])
        assert actual == expected

    def test_column_lengths(self):
        assert len(get_protocol_columns('bar')) == 11
        assert len(get_protocol_columns('line')) == 11
        assert len(get_protocol_columns('scatter')) == 10
        assert len(get_protocol_columns('box')) == 14
        assert len(get_protocol_columns('pie')) == 10
        assert len(get_protocol_columns('heatmap')) == 11


# ---------------------------------------------------------------------------
# Golden group value extraction for line and scatter fixtures
# ---------------------------------------------------------------------------

class TestLineScatterGroupGolden:
    """Assert current row `group` values for line and scatter fixtures per ADR-012 T-012.0."""

    def test_line_group_from_x_value(self):
        element = {
            'index': 0,
            'type': 'line_segment',
            'bbox': [10, 20, 30, 40],
            'x_value': 2.5,
            'value': 10.0,
            'confidence': 0.9,
        }
        res = {'image_file': 'line.png', 'chart_type': 'line', 'elements': [element]}
        rows = build_protocol_rows(res)
        assert len(rows) == 1
        # In current builder, x_value formats as :.4g into group
        assert rows[0]['group'] == '2.5'

    def test_line_group_without_x_value_empty(self):
        element = {
            'index': 0,
            'type': 'line_segment',
            'bbox': [10, 20, 30, 40],
            'position': 62.5,
            'value': 7.3,
            'confidence': 0.85,
        }
        res = {'image_file': 'line.png', 'chart_type': 'line', 'elements': [element]}
        rows = build_protocol_rows(res)
        assert len(rows) == 1
        # In current builder, without x_value or tick_label, group is empty string
        assert rows[0]['group'] == ''

    def test_line_group_from_tick_label(self):
        element = {
            'index': 0,
            'type': 'line_segment',
            'bbox': [10, 20, 30, 40],
            'tick_label': {'text': 'Day 14', 'bbox': [10, 50, 30, 60]},
            'value': 8.0,
            'confidence': 0.85,
        }
        res = {'image_file': 'line.png', 'chart_type': 'line', 'elements': [element]}
        rows = build_protocol_rows(res)
        assert len(rows) == 1
        assert rows[0]['group'] == 'Day 14'

    def test_scatter_group_unlabelled_empty(self):
        element = {
            'index': 0,
            'type': 'point',
            'bbox': [100, 200, 110, 210],
            'x': 3.14,
            'y': 9.81,
        }
        res = {'image_file': 'scatter.png', 'chart_type': 'scatter', 'elements': [element]}
        rows = build_protocol_rows(res)
        assert len(rows) == 1
        # Unlabelled scatter point has empty group
        assert rows[0]['group'] == ''
        # value_x is populated for scatter
        assert rows[0]['value_x'] == 3.14
        # After T-012.1 (DP-07): unlabelled scatter point has empty series_id
        assert rows[0]['series_id'] == ''

    def test_scatter_group_with_tick_label(self):
        element = {
            'index': 0,
            'type': 'point',
            'bbox': [100, 200, 110, 210],
            'x': 3.14,
            'y': 9.81,
            'tick_label': {'text': 'Cluster 1'},
        }
        res = {'image_file': 'scatter.png', 'chart_type': 'scatter', 'elements': [element]}
        rows = build_protocol_rows(res)
        assert len(rows) == 1
        assert rows[0]['group'] == 'Cluster 1'


# ---------------------------------------------------------------------------
# Context fields constant & non-destructive backfill
# ---------------------------------------------------------------------------

class TestContextFieldsAndBackfill:
    """Verify _CONTEXT_FIELDS set and merge_context_into_rows backfill behavior."""

    def test_context_fields_contains_series_id_and_value_x(self):
        expected = frozenset({'group', 'outcome', 'unit', 'error_bar_type', 'series_id', 'value_x'})
        assert _CONTEXT_FIELDS == expected

    def test_edited_row_backfills_series_id_and_value_x(self):
        """User edited value/group, but row lacks series_id and value_x; fresh extraction fills them."""
        fresh_elem = {
            'index': 0,
            'type': 'point',
            'bbox': [10, 20, 30, 40],
            'x': 5.5,
            'y': 12.0,
            'series_id': 'Series Alpha',
        }
        fresh_result = {'image_file': 'test.png', 'chart_type': 'scatter', 'elements': [fresh_elem]}
        fresh_rows = build_protocol_rows(fresh_result)
        assert fresh_rows[0]['series_id'] == 'Series Alpha'
        assert fresh_rows[0]['value_x'] == 5.5

        # Simulate existing edited row (e.g. from prior session or sidecar) missing series_id and value_x
        edited_row: Dict[str, Any] = {
            'element_index': 0,
            'source_file': 'test.png',
            'chart_type': 'scatter',
            'group': 'User Category',  # user edited
            'outcome': '',
            'value': 99.0,             # user edited
            'unit': '',
            'error_bar_type': '',
            'series_id': '',           # missing
            'value_x': None,           # missing
            'review_status': 'corrected',
            '_original': {'value': 12.0, 'group': '', 'review_status': 'auto'},
        }

        merged = merge_context_into_rows([edited_row], fresh_result, {})
        assert len(merged) == 1
        m = merged[0]

        # User edits must be strictly preserved
        assert m['value'] == 99.0
        assert m['group'] == 'User Category'
        assert m['review_status'] == 'corrected'
        assert m['_original'] == {'value': 12.0, 'group': '', 'review_status': 'auto'}

        # Missing series_id and value_x must be backfilled from fresh
        assert m['series_id'] == 'Series Alpha'
        assert m['value_x'] == 5.5

    def test_edited_row_retains_existing_series_id_and_value_x(self):
        """If user already entered series_id or value_x, do NOT overwrite with fresh."""
        fresh_elem = {
            'index': 0,
            'type': 'point',
            'bbox': [10, 20, 30, 40],
            'x': 100.0,
            'y': 200.0,
            'series_id': 'Auto Series',
        }
        fresh_result = {'image_file': 'test.png', 'chart_type': 'scatter', 'elements': [fresh_elem]}

        edited_row: Dict[str, Any] = {
            'element_index': 0,
            'source_file': 'test.png',
            'chart_type': 'scatter',
            'group': 'Tick 1',
            'outcome': '',
            'value': 200.0,
            'unit': '',
            'error_bar_type': '',
            'series_id': 'User Series Custom',
            'value_x': 0.0,  # 0.0 is a valid float, must not be treated as None!
            'review_status': 'corrected',
            '_original': {'series_id': 'Auto Series', 'value_x': 100.0},
        }

        merged = merge_context_into_rows([edited_row], fresh_result, {})
        assert len(merged) == 1
        m = merged[0]

        # Existing series_id and value_x must be retained
        assert m['series_id'] == 'User Series Custom'
        assert m['value_x'] == 0.0

    def test_untouched_row_is_fully_regenerated(self):
        """Untouched row (review_status='auto' and _original is None) is fully replaced."""
        fresh_elem = {
            'index': 0,
            'type': 'point',
            'bbox': [10, 20, 30, 40],
            'x': 7.7,
            'y': 8.8,
            'series_id': 'New Series',
        }
        fresh_result = {'image_file': 'test.png', 'chart_type': 'scatter', 'elements': [fresh_elem]}

        old_untouched_row: Dict[str, Any] = {
            'element_index': 0,
            'source_file': 'test.png',
            'chart_type': 'scatter',
            'group': 'Old Group',
            'outcome': 'Old Outcome',
            'value': 1.1,
            'unit': 'mg',
            'error_bar_type': 'SD',
            'series_id': 'Old Series',
            'value_x': 1.1,
            'review_status': 'auto',
            '_original': None,
        }

        merged = merge_context_into_rows([old_untouched_row], fresh_result, {})
        assert len(merged) == 1
        m = merged[0]

        assert m['value'] == 8.8
        assert m['value_x'] == 7.7
        assert m['series_id'] == 'New Series'

    def test_edited_row_missing_keys_entirely(self):
        """If series_id and value_x are not present in row dict at all, backfill populates them."""
        fresh_elem = {
            'index': 0,
            'type': 'point',
            'bbox': [10, 20, 30, 40],
            'x': 3.3,
            'y': 4.4,
            'series_id': 'Series Delta',
        }
        fresh_result = {'image_file': 'test.png', 'chart_type': 'scatter', 'elements': [fresh_elem]}

        edited_row: Dict[str, Any] = {
            'element_index': 0,
            'source_file': 'test.png',
            'chart_type': 'scatter',
            'value': 50.0,
            'review_status': 'corrected',
            '_original': {'value': 4.4},
        }
        # Explicitly ensure keys are not present
        assert 'series_id' not in edited_row
        assert 'value_x' not in edited_row

        merged = merge_context_into_rows([edited_row], fresh_result, {})
        assert len(merged) == 1
        assert merged[0]['value'] == 50.0  # preserved
        assert merged[0]['series_id'] == 'Series Delta'
        assert merged[0]['value_x'] == 3.3

    def test_mixed_rows_and_new_elements(self):
        """Mixed scenario: row 0 is edited, row 1 is untouched, row 2 is newly discovered."""
        fresh_elems = [
            {'index': 0, 'bbox': [0, 0, 10, 10], 'x': 1.0, 'y': 10.0, 'series_id': 'S1'},
            {'index': 1, 'bbox': [10, 10, 20, 20], 'x': 2.0, 'y': 20.0, 'series_id': 'S2'},
            {'index': 2, 'bbox': [20, 20, 30, 30], 'x': 3.0, 'y': 30.0, 'series_id': 'S3'},
        ]
        fresh_result = {'image_file': 'test.png', 'chart_type': 'scatter', 'elements': fresh_elems}

        existing_rows = [
            # Row 0: edited
            {
                'element_index': 0,
                'source_file': 'test.png',
                'chart_type': 'scatter',
                'value': 99.0,
                'series_id': '',
                'value_x': None,
                'review_status': 'corrected',
                '_original': {'value': 10.0},
            },
            # Row 1: untouched
            {
                'element_index': 1,
                'source_file': 'test.png',
                'chart_type': 'scatter',
                'value': 15.0,
                'series_id': 'OldS2',
                'value_x': 1.5,
                'review_status': 'auto',
                '_original': None,
            },
        ]

        merged = merge_context_into_rows(existing_rows, fresh_result, {})
        assert len(merged) == 3

        # Row 0: edited, value preserved, series_id and value_x backfilled
        assert merged[0]['element_index'] == 0
        assert merged[0]['value'] == 99.0
        assert merged[0]['series_id'] == 'S1'
        assert merged[0]['value_x'] == 1.0

        # Row 1: untouched, fully regenerated from fresh
        assert merged[1]['element_index'] == 1
        assert merged[1]['value'] == 20.0
        assert merged[1]['series_id'] == 'S2'
        assert merged[1]['value_x'] == 2.0

        # Row 2: new row appended from fresh
        assert merged[2]['element_index'] == 2
        assert merged[2]['value'] == 30.0
        assert merged[2]['series_id'] == 'S3'
        assert merged[2]['value_x'] == 3.0

