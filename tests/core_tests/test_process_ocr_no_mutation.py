# -*- coding: utf-8 -*-
"""
Tests for the _process_ocr mutation fix in ChartAnalysisPipeline.

Verifies that _process_ocr does NOT mutate detections['axis_labels']
by appending chart_title/legend/axis_title/data_label dicts into it,
while still running OCR on all elements.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Ensure src is on sys.path
_src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))


def _make_pipeline():
    """Create a ChartAnalysisPipeline with mocked dependencies."""
    from pipelines.chart_pipeline import ChartAnalysisPipeline

    models = MagicMock()
    ocr = MagicMock()
    cal = MagicMock()

    # OCR engine returns text for each crop
    ocr.process_batch = MagicMock(
        side_effect=lambda crops: [{'text': f'text_{i}', 'confidence': 0.95} for i in range(len(crops))]
    )

    pipeline = ChartAnalysisPipeline(models, ocr, cal)
    return pipeline, ocr


def _make_detections():
    """Build a detections dict with separate class lists that must NOT be merged."""
    return {
        'axis_labels': [
            {'xyxy': [10, 10, 50, 30], 'cls': 0, 'conf': 0.9},
            {'xyxy': [60, 10, 100, 30], 'cls': 0, 'conf': 0.8},
        ],
        'chart_title': [
            {'xyxy': [100, 0, 300, 40], 'cls': 1, 'conf': 0.95},
        ],
        'legend': [
            {'xyxy': [400, 200, 500, 250], 'cls': 2, 'conf': 0.85},
        ],
        'axis_title': [
            {'xyxy': [0, 200, 50, 220], 'cls': 3, 'conf': 0.9},
        ],
        'data_label': [
            {'xyxy': [150, 100, 200, 120], 'cls': 4, 'conf': 0.7},
        ],
        'bar': [
            {'xyxy': [100, 50, 200, 200], 'cls': 5, 'conf': 0.99},
        ],
    }


class TestProcessOcrNoMutation:
    """_process_ocr must not pollute detections['axis_labels']."""

    @patch('pipelines.chart_pipeline.TextLayoutService')
    def test_axis_labels_length_unchanged(self, mock_tls):
        """axis_labels list must have the same length before and after _process_ocr."""
        mock_tls.merge_with_axis_labels.return_value = []

        pipeline, _ = _make_pipeline()
        detections = _make_detections()
        img = np.zeros((300, 500, 3), dtype=np.uint8)

        original_count = len(detections['axis_labels'])
        pipeline._process_ocr(img, detections)

        assert len(detections['axis_labels']) == original_count, (
            f"axis_labels grew from {original_count} to {len(detections['axis_labels'])}. "
            "Titles/legends/etc. should NOT be appended."
        )

    @patch('pipelines.chart_pipeline.TextLayoutService')
    def test_axis_labels_contains_no_foreign_dicts(self, mock_tls):
        """axis_labels must not contain dicts that belong to chart_title/legend/etc."""
        mock_tls.merge_with_axis_labels.return_value = []

        pipeline, _ = _make_pipeline()
        detections = _make_detections()
        img = np.zeros((300, 500, 3), dtype=np.uint8)

        title_dict = detections['chart_title'][0]
        legend_dict = detections['legend'][0]

        pipeline._process_ocr(img, detections)

        assert title_dict not in detections['axis_labels'], \
            "chart_title dict was appended into axis_labels"
        assert legend_dict not in detections['axis_labels'], \
            "legend dict was appended into axis_labels"

    @patch('pipelines.chart_pipeline.TextLayoutService')
    def test_ocr_still_runs_on_all_text_elements(self, mock_tls):
        """All text-bearing detections must still receive 'text' from OCR."""
        mock_tls.merge_with_axis_labels.return_value = []

        pipeline, ocr = _make_pipeline()
        detections = _make_detections()
        img = np.zeros((300, 500, 3), dtype=np.uint8)

        pipeline._process_ocr(img, detections)

        # axis_labels should have text
        for det in detections['axis_labels']:
            assert 'text' in det, "axis_label missing OCR text"

        # Titles/legends must also have text (set on the original dict objects)
        for det in detections['chart_title']:
            assert 'text' in det, "chart_title missing OCR text"
        for det in detections['legend']:
            assert 'text' in det, "legend missing OCR text"
        for det in detections['axis_title']:
            assert 'text' in det, "axis_title missing OCR text"
        for det in detections['data_label']:
            assert 'text' in det, "data_label missing OCR text"

    @patch('pipelines.chart_pipeline.TextLayoutService')
    def test_repeated_calls_do_not_bloat(self, mock_tls):
        """Calling _process_ocr twice must not double the axis_labels entries."""
        mock_tls.merge_with_axis_labels.return_value = []

        pipeline, _ = _make_pipeline()
        detections = _make_detections()
        img = np.zeros((300, 500, 3), dtype=np.uint8)

        original_count = len(detections['axis_labels'])

        pipeline._process_ocr(img, detections)
        pipeline._process_ocr(img, detections)

        assert len(detections['axis_labels']) == original_count, (
            f"axis_labels grew to {len(detections['axis_labels'])} after two calls. "
            "Re-extract would cause unbounded growth."
        )

    @patch('pipelines.chart_pipeline.TextLayoutService')
    def test_other_class_lists_unchanged(self, mock_tls):
        """bar, chart_title etc. lists must not be mutated in length."""
        mock_tls.merge_with_axis_labels.return_value = []

        pipeline, _ = _make_pipeline()
        detections = _make_detections()
        img = np.zeros((300, 500, 3), dtype=np.uint8)

        orig_bar = len(detections['bar'])
        orig_title = len(detections['chart_title'])
        orig_legend = len(detections['legend'])

        pipeline._process_ocr(img, detections)

        assert len(detections['bar']) == orig_bar
        assert len(detections['chart_title']) == orig_title
        assert len(detections['legend']) == orig_legend
