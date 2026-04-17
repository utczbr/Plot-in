"""
Protocol Row Builder — converts pipeline elements to protocol-level rows.

Pure functions, no GUI dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart-type-aware column definitions
# ---------------------------------------------------------------------------

# Each entry: (column_id, display_label, is_numeric)
_COMMON_COLUMNS = [
    ('source_file',    'Source',       False),
    ('chart_type',     'Type',         False),
    ('group',          'Tick label',   False),
    ('outcome',        'Legend',       False),
]
_COMMON_TAIL = [
    ('unit',           'Unit',         False),
    ('confidence',     'Conf.',        True),
    ('review_status',  'Status',       False),
    ('notes',          'Notes',        False),
]

# Readonly columns (not user-editable)
PROTOCOL_READONLY = frozenset({'source_file', 'chart_type', 'confidence'})

# Per-chart-type value columns inserted between common-head and common-tail
_VALUE_COLUMNS: Dict[str, List] = {
    'box': [
        ('whisker_low',  'Min (Whisker)', True),
        ('q1',           'Q1',            True),
        ('median',       'Median',        True),
        ('q3',           'Q3',            True),
        ('whisker_high', 'Max (Whisker)', True),
        ('outliers_str', 'Outliers',      False),
    ],
    'scatter': [
        ('value_x',      'X',             True),
        ('value',        'Y',             True),
    ],
    'pie': [
        ('value',        'Proportion',    True),
        ('percent',      '%',             True),
    ],
    'heatmap': [
        ('row_label',    'Row',           False),
        ('col_label',    'Col',           False),
        ('value',        'Value',         True),
    ],
}
_DEFAULT_VALUE_COLUMNS = [
    ('value', 'Value', True),
    ('error_bar_value', 'Error Bar', True),
    ('baseline_value',  'Baseline',  True),
]


def get_protocol_columns(chart_type: str) -> List[tuple]:
    """
    Return ordered list of (column_id, label, is_numeric) for a given chart type.
    Used by the GUI to set dynamic protocol table headers.
    """
    value_cols = _VALUE_COLUMNS.get(chart_type, _DEFAULT_VALUE_COLUMNS)
    return _COMMON_COLUMNS + value_cols + _COMMON_TAIL



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_value(element: Dict[str, Any]) -> Optional[float]:
    """Extract numeric value with fallback: value → estimated_value → y → median (box) → q1/q3."""
    for key in ('value', 'estimated_value', 'y', 'median', 'q1', 'q3'):
        v = element.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _is_synthetic(element: Dict[str, Any]) -> bool:
    """True for summary/synthetic entries that should not become protocol rows."""
    if element.get('type') == 'summary':
        return True
    if 'bbox' not in element and 'xyxy' not in element:
        return True
    return False


def _match_outcome(
    axis_titles: List[str],
    context_outcomes: List[str],
) -> str:
    """Case-insensitive substring match of axis titles against outcome list."""
    if not context_outcomes or not axis_titles:
        return ''
    for outcome in context_outcomes:
        outcome_lower = outcome.lower()
        for title in axis_titles:
            if outcome_lower in title.lower() or title.lower() in outcome_lower:
                return outcome
    # Default to first outcome if only one defined
    if len(context_outcomes) == 1:
        return context_outcomes[0]
    return ''


def _extract_group(element: Dict[str, Any]) -> str:
    """Extract group label from tick_label or bar_label."""
    tick = element.get('tick_label')
    if isinstance(tick, dict):
        text = tick.get('text', '')
        if isinstance(text, str) and text.strip():
            return text.strip()
    # Fallback to bar_label for bar charts
    bar_label = element.get('bar_label', '')
    if isinstance(bar_label, str) and bar_label.strip():
        return bar_label.strip()
    # Polar charts (pie) commonly expose semantic category in `label`.
    pie_label = element.get('label', '')
    if isinstance(pie_label, str) and pie_label.strip():
        return pie_label.strip()
    return ''


def _extract_series_id(element: Dict[str, Any], chart_type: str) -> str:
    """Extract series identifier from element."""
    if chart_type == 'scatter':
        x = element.get('x')
        if x is not None:
            return str(x)
    text_label = element.get('text_label', '')
    if isinstance(text_label, str) and text_label.strip():
        return text_label.strip()
    return ''


def _get_baseline_value(baselines: Any) -> Optional[float]:
    """Extract baseline value from pipeline baselines."""
    if not baselines:
        return None
    if isinstance(baselines, list):
        for b in baselines:
            if isinstance(b, dict):
                val = b.get('value')
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
    return None


def _get_axis_titles(pipeline_result: Dict[str, Any]) -> List[str]:
    """Extract axis title strings from pipeline result metadata."""
    titles = []
    metadata = pipeline_result.get('metadata', {})
    if isinstance(metadata, dict):
        label_cls = metadata.get('label_classification', {})
        if isinstance(label_cls, dict):
            for at in label_cls.get('axis_titles', []):
                if isinstance(at, dict):
                    t = at.get('text', '')
                elif isinstance(at, str):
                    t = at
                else:
                    continue
                if t.strip():
                    titles.append(t.strip())
    # Also check detections
    detections = pipeline_result.get('detections', {})
    if isinstance(detections, dict):
        for det in detections.get('axis_title', []):
            if isinstance(det, dict):
                t = det.get('text', '')
                if isinstance(t, str) and t.strip() and t.strip() not in titles:
                    titles.append(t.strip())
    return titles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_protocol_rows(
    pipeline_result: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert pipeline elements to protocol-level rows.

    Args:
        pipeline_result: PipelineResult dict with elements, chart_type, etc.
        context: Optional context-of-interest dict.

    Returns:
        List of ProtocolRow dicts.
    """
    elements = pipeline_result.get('elements', [])
    if not isinstance(elements, list):
        return []

    chart_type = str(pipeline_result.get('chart_type', ''))
    provenance = pipeline_result.get('_provenance') or {}
    source_file = provenance.get('source_document') or pipeline_result.get('image_file', '')
    page_index = provenance.get('page_index')
    baselines = pipeline_result.get('baselines', [])
    baseline_value = _get_baseline_value(baselines)

    # Context fields
    ctx = context or {}
    context_outcomes = ctx.get('outcomes', [])
    context_units = ctx.get('units', {})
    context_error_bar_type = ctx.get('error_bar_type', '')
    axis_titles = _get_axis_titles(pipeline_result)
    outcome = _match_outcome(axis_titles, context_outcomes)
    unit = context_units.get(outcome, '') if isinstance(context_units, dict) else ''

    rows: List[Dict[str, Any]] = []
    for idx, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        if _is_synthetic(element):
            continue

        error_bar = element.get('error_bar')
        error_bar_value = None
        if isinstance(error_bar, dict):
            margin = error_bar.get('margin')
            if margin is not None:
                try:
                    error_bar_value = float(margin)
                except (TypeError, ValueError):
                    pass

        # Box-specific stats
        box_fields: Dict[str, Any] = {}
        if chart_type == 'box':
            for k in ('whisker_low', 'q1', 'median', 'q3', 'whisker_high'):
                v = element.get(k)
                try:
                    box_fields[k] = float(v) if v is not None else None
                except (TypeError, ValueError):
                    box_fields[k] = None
            raw_outliers = element.get('outliers', [])
            if isinstance(raw_outliers, list):
                box_fields['outliers_str'] = ', '.join(
                    f'{o:.4g}' for o in raw_outliers if o is not None
                )
            else:
                box_fields['outliers_str'] = ''

        # Scatter x-coordinate
        scatter_x = None
        if chart_type == 'scatter':
            try:
                scatter_x = float(element.get('x')) if element.get('x') is not None else None
            except (TypeError, ValueError):
                pass

        # Pie percent
        pie_pct = None
        if chart_type == 'pie':
            v = _extract_value(element)
            if v is not None:
                pie_pct = round(v * 100.0, 4)

        row: Dict[str, Any] = {
            'source_file': str(source_file),
            'page_index': page_index,
            'chart_type': chart_type,
            'element_index': element.get('index', idx),
            'series_id': _extract_series_id(element, chart_type),
            'group': _extract_group(element),
            'outcome': outcome,
            'value': _extract_value(element),
            'unit': unit,
            'error_bar_type': context_error_bar_type,
            'error_bar_value': error_bar_value,
            'baseline_value': baseline_value,
            'confidence': element.get('confidence'),
            'review_status': 'auto',
            'notes': '',
            '_original': None,
            # Scatter
            'value_x': scatter_x,
            # Pie
            'percent': pie_pct,
            # Heatmap
            'row_label': element.get('row_label', ''),
            'col_label': element.get('col_label', ''),
            # Box
            **box_fields,
        }
        rows.append(row)

    return rows


# Context fields that can be filled non-destructively on edited rows
_CONTEXT_FIELDS = frozenset({'group', 'outcome', 'unit', 'error_bar_type'})


def merge_context_into_rows(
    existing_rows: List[Dict[str, Any]],
    pipeline_result: Dict[str, Any],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Non-destructive context merge into existing protocol rows.

    - Untouched rows (review_status='auto' and _original is None): fully regenerated.
    - Edited rows (_original is not None): only empty context-derived fields are filled.
    """
    # Build fresh rows for reference
    fresh_rows = build_protocol_rows(pipeline_result, context)

    # Build lookup by element_index for matching
    fresh_by_idx: Dict[int, Dict[str, Any]] = {}
    for row in fresh_rows:
        fresh_by_idx[row.get('element_index', -1)] = row

    merged: List[Dict[str, Any]] = []
    seen_indices = set()

    for row in existing_rows:
        elem_idx = row.get('element_index', -1)
        is_untouched = (
            row.get('review_status') == 'auto'
            and row.get('_original') is None
        )

        if is_untouched and elem_idx in fresh_by_idx:
            # Fully replace with fresh row
            merged.append(fresh_by_idx[elem_idx])
        else:
            # Fill only empty context fields
            fresh = fresh_by_idx.get(elem_idx, {})
            for field in _CONTEXT_FIELDS:
                current = row.get(field, '')
                if not current and fresh.get(field):
                    row[field] = fresh[field]
            merged.append(row)
        seen_indices.add(elem_idx)

    # Add any new rows from fresh that weren't in existing
    for row in fresh_rows:
        if row.get('element_index', -1) not in seen_indices:
            merged.append(row)

    return merged
