import sys
import unittest
from unittest.mock import patch

sys.path.append('src')

from core.baseline.zero_crossing import baseline_from_scale_zero


class TestBaselineZeroCrossingCharacterization(unittest.TestCase):
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
