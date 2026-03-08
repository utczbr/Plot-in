import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.append('src')

from core.baseline_detection import DetectorConfig, ModularBaselineDetector
from core.enums import ChartType
from services.orientation_service import Orientation


class TestBaselineDetectorCharacterization(unittest.TestCase):
    def setUp(self):
        self.detector = ModularBaselineDetector(config=DetectorConfig(cluster_backend='dbscan'))
        self.img = np.zeros((120, 160, 3), dtype=np.uint8)

    def test_horizontal_primary_zero_short_circuit(self):
        chart_elements = [
            {'xyxy': [20, 20, 60, 40]},
            {'xyxy': [20, 60, 80, 80]},
        ]
        cal = SimpleNamespace(coeffs=(1.0, -30.0), r2=0.99)

        result = self.detector.detect(
            img=self.img,
            chart_elements=chart_elements,
            axis_labels=[],
            orientation=Orientation.HORIZONTAL,
            chart_type=ChartType.BAR,
            primary_calibration_zero=30.0,
            primary_calibration_result=cal,
        )

        self.assertEqual(result.method, 'primary_calibration_zero')
        self.assertEqual(len(result.baselines), 1)
        self.assertEqual(result.baselines[0].axis_id, 'x')
        self.assertAlmostEqual(result.baselines[0].value, 30.0)

    def test_vertical_bar_returns_single_baseline_and_diagnostics(self):
        chart_elements = [
            {'xyxy': [10, 20, 20, 100]},
            {'xyxy': [50, 30, 60, 95]},
            {'xyxy': [90, 25, 100, 92]},
        ]

        result = self.detector.detect(
            img=self.img,
            chart_elements=chart_elements,
            axis_labels=[],
            orientation=Orientation.VERTICAL,
            chart_type=ChartType.BAR,
        )

        self.assertTrue(result.method.startswith('single_stackaware_'))
        self.assertEqual(len(result.baselines), 1)
        self.assertEqual(result.baselines[0].axis_id, 'y')
        self.assertIn('inverted_axis', result.diagnostics)
        self.assertIn('n_bands', result.diagnostics)
        self.assertIn('percentile_used', result.diagnostics)
        self.assertIn('calibration_zero', result.diagnostics)

    def test_no_elements_returns_no_elements_method(self):
        result = self.detector.detect(
            img=self.img,
            chart_elements=[],
            axis_labels=[],
            orientation=Orientation.VERTICAL,
            chart_type=ChartType.HISTOGRAM,
        )

        self.assertEqual(result.method, 'no_elements')
        self.assertEqual(result.baselines, [])

    def test_scatter_returns_x_and_y_baselines(self):
        axis_labels = [
            {'xyxy': [8, 100, 20, 112], 'cleanedvalue': '-10'},
            {'xyxy': [24, 100, 36, 112], 'cleanedvalue': '0'},
            {'xyxy': [40, 100, 52, 112], 'cleanedvalue': '10'},
            {'xyxy': [8, 20, 20, 32], 'cleanedvalue': '-10'},
        ]

        result = self.detector.detect(
            img=self.img,
            chart_elements=[{'xyxy': [30, 30, 35, 35]}],
            axis_labels=axis_labels,
            orientation=Orientation.VERTICAL,
            chart_type=ChartType.SCATTER,
        )

        axis_ids = {line.axis_id for line in result.baselines}
        self.assertEqual(result.method, 'scatter_dual')
        self.assertIn('x', axis_ids)
        self.assertIn('y', axis_ids)


if __name__ == '__main__':
    unittest.main()
