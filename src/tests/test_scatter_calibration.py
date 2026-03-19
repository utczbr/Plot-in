"""
Tests for scatter plot axis label grouping and calibration bug fixes.

Covers:
  - Bug A: is_numeric stored as bool (not function reference)
  - Bug B: x_scale_labels / y_scale_labels propagated from ScatterChartClassifier
  - Bug C/D: ScatterHandler._calibrate_axes uses axis_type='x' for x-labels
             and axis_type='y' for y-labels
"""
import pytest
from unittest.mock import MagicMock, call
from typing import List, Dict

from core.classifiers.scatter_chart_classifier import ScatterChartClassifier
from handlers.scatter_handler import ScatterHandler
from services.orientation_service import Orientation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label(x1, y1, x2, y2, text="1.0", conf=0.9):
    """Create a minimal axis-label dict."""
    return {
        "xyxy": [x1, y1, x2, y2],
        "text": text,
        "cleanedvalue": None,
        "ocr_confidence": conf,
    }


def _make_scatter_point(cx, cy, size=10):
    """Create a minimal scatter data-point dict."""
    return {"xyxy": [cx - size, cy - size, cx + size, cy + size]}


# ---------------------------------------------------------------------------
# Bug A — is_numeric must be a bool, not the function object
# ---------------------------------------------------------------------------

class TestBugA_IsNumericBool:

    def test_is_numeric_is_bool_for_numeric_text(self):
        """
        _extract_scatter_features must store the *result* of is_numeric(text),
        not the function reference itself.

        Before the fix, 'is_numeric' was always truthy (function object).
        After the fix, it should be True for '1.5' and False for 'Weight'.
        """
        classifier = ScatterChartClassifier()

        img_w, img_h = 800, 600
        labels = [
            _make_label(10, 290, 40, 310, text="1.5"),   # numeric
            _make_label(10, 190, 40, 210, text="Weight"), # non-numeric
        ]
        points = [_make_scatter_point(400, 300)]

        features = classifier._extract_scatter_features(labels, img_w, img_h)

        assert len(features) == 2, "Should produce one feature per label"

        numeric_feat = features[0]
        text_feat = features[1]

        assert isinstance(numeric_feat['is_numeric'], bool), (
            "is_numeric must be a bool, got: {}".format(type(numeric_feat['is_numeric']))
        )
        assert numeric_feat['is_numeric'] is True,  "Label '1.5' should be numeric"
        assert isinstance(text_feat['is_numeric'], bool), (
            "is_numeric must be a bool, got: {}".format(type(text_feat['is_numeric']))
        )
        assert text_feat['is_numeric'] is False, "Label 'Weight' should not be numeric"


# ---------------------------------------------------------------------------
# Bug B — x_scale_labels / y_scale_labels populated in ClassificationResult
# ---------------------------------------------------------------------------

class TestBugB_XYSplitPropagated:

    def _make_vertical_scatter_labels(self, img_w=800, img_h=600):
        """
        Create labels that cleanly separate into x-axis (bottom) and y-axis (left).

        Y-axis: small X-center (~50px), spread vertically
        X-axis: large Y-center (~550px), spread horizontally
        """
        y_axis_labels = [
            _make_label(10, 100, 90, 120, text="10"),
            _make_label(10, 200, 90, 220, text="20"),
            _make_label(10, 300, 90, 320, text="30"),
            _make_label(10, 400, 90, 420, text="40"),
        ]
        x_axis_labels = [
            _make_label(150, 540, 230, 560, text="1"),
            _make_label(350, 540, 430, 560, text="2"),
            _make_label(550, 540, 630, 560, text="3"),
            _make_label(650, 540, 730, 560, text="4"),
        ]
        return y_axis_labels, x_axis_labels

    def test_x_scale_labels_and_y_scale_labels_populated(self):
        """
        After the fix, ClassificationResult.x_scale_labels and .y_scale_labels
        must both be non-empty when labels are spatially separated.
        """
        classifier = ScatterChartClassifier()
        img_w, img_h = 800, 600

        y_labels, x_labels = self._make_vertical_scatter_labels(img_w, img_h)
        all_labels = y_labels + x_labels

        points = [
            _make_scatter_point(200, 200),
            _make_scatter_point(400, 300),
            _make_scatter_point(600, 400),
        ]

        result = classifier.classify(
            axis_labels=all_labels,
            chart_elements=points,
            img_width=img_w,
            img_height=img_h,
            orientation=Orientation.VERTICAL,
        )

        assert len(result.x_scale_labels) > 0, (
            "x_scale_labels must be populated — bottom-edge labels should be classified as x-axis"
        )
        assert len(result.y_scale_labels) > 0, (
            "y_scale_labels must be populated — left-edge labels should be classified as y-axis"
        )

    def test_no_label_in_both_x_and_y(self):
        """No label should appear in both x_scale_labels and y_scale_labels."""
        classifier = ScatterChartClassifier()
        img_w, img_h = 800, 600

        y_labels, x_labels = self._make_vertical_scatter_labels(img_w, img_h)
        all_labels = y_labels + x_labels
        points = [_make_scatter_point(400, 300)]

        result = classifier.classify(
            axis_labels=all_labels,
            chart_elements=points,
            img_width=img_w,
            img_height=img_h,
            orientation=Orientation.VERTICAL,
        )

        x_ids = {id(lbl) for lbl in result.x_scale_labels}
        y_ids = {id(lbl) for lbl in result.y_scale_labels}
        overlap = x_ids & y_ids
        assert len(overlap) == 0, (
            f"{len(overlap)} label(s) appear in both x_scale_labels and y_scale_labels"
        )


# ---------------------------------------------------------------------------
# Bug C/D — ScatterHandler._calibrate_axes uses correct axis_type per axis
# ---------------------------------------------------------------------------

class TestBugCD_AxisTypeRouting:

    def _make_classified_result(self, x_labels, y_labels):
        """Create a minimal ClassificationResult-like namespace."""
        from dataclasses import dataclass, field
        from core.classifiers.base_classifier import ClassificationResult
        return ClassificationResult(
            scale_labels=x_labels + y_labels,
            tick_labels=[],
            axis_titles=[],
            confidence=0.9,
            metadata={},
            x_scale_labels=x_labels,
            y_scale_labels=y_labels,
        )

    def test_calibrate_called_with_x_and_y_axis_types(self):
        """
        ScatterHandler._calibrate_axes must call calibration_service.calibrate()
        twice: once with axis_type='x' for x-labels, once with axis_type='y' for y-labels.
        """
        mock_calibration = MagicMock()
        mock_calibration.calibrate.side_effect = [
            MagicMock(name="cal_x"),  # first call → x-axis
            MagicMock(name="cal_y"),  # second call → y-axis
        ]

        mock_spatial = MagicMock()
        mock_dual = MagicMock()
        mock_meta = MagicMock()

        handler = ScatterHandler(
            calibration_service=mock_calibration,
            spatial_classifier=mock_spatial,
            dual_axis_service=mock_dual,
            meta_clustering_service=mock_meta,
        )

        x_labels = [_make_label(150, 540, 230, 560, text="1")]
        y_labels = [_make_label(10, 200, 90, 220, text="10")]
        classified = self._make_classified_result(x_labels, y_labels)

        dual_decision = MagicMock()  # not used in the override
        result = handler._calibrate_axes(classified, dual_decision, Orientation.VERTICAL)

        calls = mock_calibration.calibrate.call_args_list
        axis_types_used = [c.kwargs.get('axis_type') or c.args[1] for c in calls]

        assert 'x' in axis_types_used, (
            "calibrate() must be called with axis_type='x' for x_scale_labels"
        )
        assert 'y' in axis_types_used, (
            "calibrate() must be called with axis_type='y' for y_scale_labels"
        )

    def test_calibrations_dict_has_x_and_y_keys(self):
        """
        The returned dict must have non-None 'x' and 'y' keys when both
        label pools are provided.
        """
        mock_cal_x = MagicMock(name="cal_x")
        mock_cal_y = MagicMock(name="cal_y")

        mock_calibration = MagicMock()
        mock_calibration.calibrate.side_effect = [mock_cal_x, mock_cal_y]

        handler = ScatterHandler(
            calibration_service=mock_calibration,
            spatial_classifier=MagicMock(),
            dual_axis_service=MagicMock(),
            meta_clustering_service=MagicMock(),
        )

        x_labels = [_make_label(150, 540, 230, 560, text="1")]
        y_labels = [_make_label(10, 200, 90, 220, text="10")]
        classified = self._make_classified_result(x_labels, y_labels)

        result = handler._calibrate_axes(classified, MagicMock(), Orientation.VERTICAL)

        assert result.get('x') is not None, "calibrations['x'] must not be None"
        assert result.get('y') is not None, "calibrations['y'] must not be None"
        assert result['x'] is not result['y'], (
            "calibrations['x'] and ['y'] must be distinct calibration objects"
        )

    def test_fallback_when_no_split(self):
        """
        When x_scale_labels and y_scale_labels are both empty, the handler
        falls back to using scale_labels as y-axis only and logs a warning.
        """
        mock_cal_y = MagicMock(name="cal_y_fallback")
        mock_calibration = MagicMock()
        mock_calibration.calibrate.return_value = mock_cal_y

        handler = ScatterHandler(
            calibration_service=mock_calibration,
            spatial_classifier=MagicMock(),
            dual_axis_service=MagicMock(),
            meta_clustering_service=MagicMock(),
        )

        from core.classifiers.base_classifier import ClassificationResult
        classified = ClassificationResult(
            scale_labels=[_make_label(10, 200, 90, 220, text="10")],
            tick_labels=[],
            axis_titles=[],
            confidence=0.5,
            metadata={},
            x_scale_labels=[],   # no split provided
            y_scale_labels=[],
        )

        result = handler._calibrate_axes(classified, MagicMock(), Orientation.VERTICAL)

        # Should have calibrated using the full pool as y-axis
        assert mock_calibration.calibrate.called, "calibrate() should still be called for fallback"
        calls = mock_calibration.calibrate.call_args_list
        axis_types = [c.kwargs.get('axis_type') or c.args[1] for c in calls]
        assert 'y' in axis_types, "Fallback must use axis_type='y'"
        assert result.get('x') is None, "x calibration should be None in fallback mode"
