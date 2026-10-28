"""Tests for ScatterHandler axis calibration routing."""

from unittest.mock import MagicMock

from core.classifiers.base_classifier import ClassificationResult
from handlers.scatter_handler import ScatterHandler
from services.orientation_service import Orientation


def _make_label(x1, y1, x2, y2, text="1.0", conf=0.9):
    return {
        "xyxy": [x1, y1, x2, y2],
        "text": text,
        "cleanedvalue": None,
        "ocr_confidence": conf,
    }


class TestScatterHandlerAxisTypeRouting:
    def _make_classified_result(self, x_labels, y_labels):
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
        mock_calibration = MagicMock()
        mock_calibration.calibrate.side_effect = [
            MagicMock(name="cal_x"),
            MagicMock(name="cal_y"),
        ]
        handler = ScatterHandler(
            calibration_service=mock_calibration,
            spatial_classifier=MagicMock(),
            dual_axis_service=MagicMock(),
            meta_clustering_service=MagicMock(),
        )
        x_labels = [_make_label(150, 540, 230, 560, text="1")]
        y_labels = [_make_label(10, 200, 90, 220, text="10")]
        classified = self._make_classified_result(x_labels, y_labels)

        handler._calibrate_axes(classified, MagicMock(), Orientation.VERTICAL)

        calls = mock_calibration.calibrate.call_args_list
        axis_types_used = [c.kwargs.get('axis_type') or c.args[1] for c in calls]
        assert 'x' in axis_types_used
        assert 'y' in axis_types_used

    def test_calibrations_dict_has_x_and_y_keys(self):
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
        assert result.get('x') is not None
        assert result.get('y') is not None
        assert result['x'] is not result['y']

    def test_fallback_when_no_split(self):
        mock_cal_y = MagicMock(name="cal_y_fallback")
        mock_calibration = MagicMock()
        mock_calibration.calibrate.return_value = mock_cal_y

        handler = ScatterHandler(
            calibration_service=mock_calibration,
            spatial_classifier=MagicMock(),
            dual_axis_service=MagicMock(),
            meta_clustering_service=MagicMock(),
        )
        classified = ClassificationResult(
            scale_labels=[_make_label(10, 200, 90, 220, text="10")],
            tick_labels=[],
            axis_titles=[],
            confidence=0.5,
            metadata={},
            x_scale_labels=[],
            y_scale_labels=[],
        )

        result = handler._calibrate_axes(classified, MagicMock(), Orientation.VERTICAL)
        assert mock_calibration.calibrate.called
        calls = mock_calibration.calibrate.call_args_list
        axis_types = [c.kwargs.get('axis_type') or c.args[1] for c in calls]
        assert 'y' in axis_types
        assert result.get('x') is None
