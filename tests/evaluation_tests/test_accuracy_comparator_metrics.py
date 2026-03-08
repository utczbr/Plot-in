"""Tests for CCC, Kappa, and extended metrics in accuracy_comparator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from evaluation.accuracy_comparator import (
    _lin_ccc,
    _safe_cohens_kappa,
    AccuracyComparator,
    BatchEvaluator,
)


# ---------------------------------------------------------------------------
# _lin_ccc
# ---------------------------------------------------------------------------

class TestLinCCC:
    def test_perfect_agreement(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _lin_ccc(y, y) == pytest.approx(1.0)

    def test_negatively_correlated(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ccc = _lin_ccc(y_true, y_pred)
        assert ccc is not None
        assert ccc < 0

    def test_constant_vector(self):
        y = np.array([3.0, 3.0, 3.0, 3.0])
        assert _lin_ccc(y, y) is None

    def test_single_element(self):
        assert _lin_ccc(np.array([1.0]), np.array([2.0])) is None

    def test_empty(self):
        assert _lin_ccc(np.array([]), np.array([])) is None

    def test_with_nan(self):
        y_true = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        ccc = _lin_ccc(y_true, y_pred)
        assert ccc is not None
        assert ccc == pytest.approx(1.0)

    def test_with_offset(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true + 10.0  # Same pattern, shifted
        ccc = _lin_ccc(y_true, y_pred)
        assert ccc is not None
        assert 0 < ccc < 1.0  # High correlation but poor agreement


# ---------------------------------------------------------------------------
# _safe_cohens_kappa
# ---------------------------------------------------------------------------

class TestSafeCohensKappa:
    def test_perfect_agreement(self):
        labels = ['a', 'b', 'c', 'a', 'b']
        assert _safe_cohens_kappa(labels, labels) == pytest.approx(1.0)

    def test_random_agreement(self):
        y_true = ['a', 'b', 'a', 'b', 'a', 'b']
        y_pred = ['b', 'a', 'b', 'a', 'b', 'a']
        kappa = _safe_cohens_kappa(y_true, y_pred)
        assert kappa is not None
        assert kappa < 0.5

    def test_single_class(self):
        assert _safe_cohens_kappa(['a', 'a', 'a'], ['a', 'a', 'a']) is None

    def test_empty(self):
        assert _safe_cohens_kappa([], []) is None

    def test_single_element(self):
        assert _safe_cohens_kappa(['a'], ['b']) is None


# ---------------------------------------------------------------------------
# AccuracyComparator — categorical_metrics
# ---------------------------------------------------------------------------

class TestComparatorCategoricalMetrics:
    def test_compare_chart_has_categorical_metrics(self, tmp_path):
        gt = {'charts': [{'chart_type': 'bar', 'bar_values': []}], 'annotations': []}
        pred = {'chart_type': 'bar', 'elements': [], 'calibrated_values': []}
        gt_file = tmp_path / 'gt.json'
        pred_file = tmp_path / 'pred.json'
        gt_file.write_text(json.dumps(gt))
        pred_file.write_text(json.dumps(pred))

        comp = AccuracyComparator()
        metrics = comp.compare_chart(gt_file, pred_file)
        assert 'categorical_metrics' in metrics
        assert metrics['categorical_metrics']['gt_chart_type'] == 'bar'
        assert metrics['categorical_metrics']['pred_chart_type'] == 'bar'
        assert metrics['categorical_metrics']['chart_type_match'] is True

    def test_chart_type_mismatch(self, tmp_path):
        gt = {'charts': [{'chart_type': 'bar', 'bar_values': []}], 'annotations': []}
        pred = {'chart_type': 'line', 'elements': [], 'calibrated_values': []}
        gt_file = tmp_path / 'gt.json'
        pred_file = tmp_path / 'pred.json'
        gt_file.write_text(json.dumps(gt))
        pred_file.write_text(json.dumps(pred))

        comp = AccuracyComparator()
        metrics = comp.compare_chart(gt_file, pred_file)
        assert metrics['categorical_metrics']['chart_type_match'] is False


# ---------------------------------------------------------------------------
# BatchEvaluator summary — CCC and Kappa
# ---------------------------------------------------------------------------

class TestSummaryExtensions:
    def _make_metrics(self, chart_type='bar', ccc=None, mae=1.0):
        val = {'mae': mae, 'relative_error_pct': 5.0, 'relaxed_accuracy': 0.8}
        if ccc is not None:
            val['ccc'] = ccc
        return {
            'chart_type': chart_type,
            'detection_metrics': {'f1': 0.9, 'precision': 0.9, 'recall': 0.9, 'avg_iou': 0.7},
            'value_metrics': val,
            'categorical_metrics': {
                'gt_chart_type': chart_type,
                'pred_chart_type': chart_type,
                'chart_type_match': True,
            },
        }

    def test_summary_includes_ccc(self):
        evaluator = BatchEvaluator()
        all_metrics = [self._make_metrics(ccc=0.95), self._make_metrics(ccc=0.85)]
        summary = evaluator._compute_summary(all_metrics)
        assert 'avg_value_ccc' in summary
        assert summary['avg_value_ccc'] == pytest.approx(0.90)

    def test_summary_includes_kappa(self):
        evaluator = BatchEvaluator()
        all_metrics = [
            self._make_metrics(chart_type='bar'),
            self._make_metrics(chart_type='line'),
        ]
        summary = evaluator._compute_summary(all_metrics)
        assert 'cohens_kappa' in summary
        assert summary['cohens_kappa'] == pytest.approx(1.0)

    def test_summary_preserves_legacy_keys(self):
        evaluator = BatchEvaluator()
        all_metrics = [self._make_metrics()]
        summary = evaluator._compute_summary(all_metrics)
        assert 'avg_detection_f1' in summary
        assert 'avg_value_mae' in summary
        assert 'total_charts' in summary
