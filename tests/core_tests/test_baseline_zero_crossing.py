import sys
import unittest
from unittest.mock import patch

sys.path.append('src')

from core.baseline.zero_crossing import baseline_from_scale_zero


class TestBaselineZeroCrossing(unittest.TestCase):
    def test_horizontal_ocr_confidence_filtering_ignores_low_conf_outlier_zero(self):
        labels = [
            {'xyxy': [8, 10, 12, 20], 'cleanedvalue': -10, 'ocr_confidence': 0.95},
            {'xyxy': [28, 10, 32, 20], 'cleanedvalue': 10, 'ocr_confidence': 0.96},
            {'xyxy': [38, 10, 42, 20], 'cleanedvalue': 20, 'ocr_confidence': 0.97},
            {'xyxy': [78, 10, 82, 20], 'cleanedvalue': 0, 'ocr_confidence': 0.20},
        ]

        with patch('core.baseline.zero_crossing.CalibrationFactory', None):
            baseline = baseline_from_scale_zero(labels, is_vertical=False, calibration_mode='prosac')

        self.assertIsNotNone(baseline)
        # If low-confidence zero were used, baseline would be ~80.
        self.assertAlmostEqual(baseline, 20.0, delta=1.0)

    def test_explicit_zero_label_returns_zero_crossing_directly(self):
        labels = [
            {'xyxy': [8, 10, 12, 20], 'cleanedvalue': -10, 'ocr_confidence': 0.95},
            {'xyxy': [38, 10, 42, 20], 'cleanedvalue': 0, 'ocr_confidence': 0.95},
            {'xyxy': [68, 10, 72, 20], 'cleanedvalue': 10, 'ocr_confidence': 0.95},
        ]

        with patch('core.baseline.zero_crossing.CalibrationFactory', None):
            baseline = baseline_from_scale_zero(labels, is_vertical=False, calibration_mode='prosac')

        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(baseline, 40.0, delta=0.5)

    def test_fallback_sign_flip_interpolation(self):
        labels = [
            {'xyxy': [8, 10, 12, 20], 'cleanedvalue': -10, 'ocr_confidence': 0.9},
            {'xyxy': [28, 10, 32, 20], 'cleanedvalue': 10, 'ocr_confidence': 0.9},
        ]
        with patch('core.baseline.zero_crossing.CalibrationFactory', None):
            baseline = baseline_from_scale_zero(labels, is_vertical=False, calibration_mode='prosac')
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(baseline, 20.0, delta=1.0)

    def test_fallback_all_positive_extrapolation(self):
        labels = [
            {'xyxy': [10, 10, 14, 20], 'cleanedvalue': 10, 'ocr_confidence': 0.9},
            {'xyxy': [20, 10, 24, 20], 'cleanedvalue': 20, 'ocr_confidence': 0.9},
        ]
        with patch('core.baseline.zero_crossing.CalibrationFactory', None):
            baseline = baseline_from_scale_zero(labels, is_vertical=False, calibration_mode='prosac')
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(baseline, 2.0, delta=1.0)

    def test_fallback_all_negative_extrapolation(self):
        labels = [
            {'xyxy': [10, 10, 14, 20], 'cleanedvalue': -20, 'ocr_confidence': 0.9},
            {'xyxy': [20, 10, 24, 20], 'cleanedvalue': -10, 'ocr_confidence': 0.9},
        ]
        with patch('core.baseline.zero_crossing.CalibrationFactory', None):
            baseline = baseline_from_scale_zero(labels, is_vertical=False, calibration_mode='prosac')
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(baseline, 32.0, delta=1.0)

    def test_invalid_or_empty_labels_return_none(self):
        with patch('core.baseline.zero_crossing.CalibrationFactory', None):
            self.assertIsNone(baseline_from_scale_zero([], is_vertical=True, calibration_mode='prosac'))
            self.assertIsNone(
                baseline_from_scale_zero(
                    [{'xyxy': [0, 0, 1, 1], 'cleanedvalue': 'abc'}],
                    is_vertical=True,
                    calibration_mode='prosac',
                )
            )


if __name__ == '__main__':
    unittest.main()
