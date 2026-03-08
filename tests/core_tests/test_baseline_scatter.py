import sys
import unittest

import numpy as np

sys.path.append('src')

from core.baseline.scatter import scatter_axis_baseline


class TestBaselineScatter(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((120, 160, 3), dtype=np.uint8)

    def test_x_axis_baseline_prefers_bottom_labels(self):
        axis_labels = [
            {'xyxy': [20, 100, 30, 110]},
            {'xyxy': [50, 98, 60, 108]},
            {'xyxy': [20, 10, 30, 20]},
        ]
        baseline = scatter_axis_baseline(self.img, axis_labels, axis='x')
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(baseline, 104.0, delta=1.0)

    def test_y_axis_baseline_prefers_left_labels(self):
        axis_labels = [
            {'xyxy': [5, 30, 15, 40]},
            {'xyxy': [8, 50, 18, 60]},
            {'xyxy': [130, 30, 140, 40]},
        ]
        baseline = scatter_axis_baseline(self.img, axis_labels, axis='y')
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(baseline, 11.5, delta=1.0)

    def test_returns_none_without_valid_labels(self):
        self.assertIsNone(scatter_axis_baseline(self.img, [], axis='x'))
        self.assertIsNone(scatter_axis_baseline(self.img, [{'xyxy': [1, 2, 3]}], axis='y'))


if __name__ == '__main__':
    unittest.main()
