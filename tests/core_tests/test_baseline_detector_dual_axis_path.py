import sys
import unittest

import numpy as np

sys.path.append('src')

from core.baseline_detection import DetectorConfig, ModularBaselineDetector
from core.enums import ChartType
from services.orientation_service import Orientation


class TestBaselineDetectorDualAxisPath(unittest.TestCase):
    def test_dual_axis_branch_no_undefined_method_failure(self):
        detector = ModularBaselineDetector(config=DetectorConfig(cluster_backend='dbscan'))
        img = np.zeros((120, 200, 3), dtype=np.uint8)

        chart_elements = [
            {'xyxy': [20, 40, 40, 110]},
            {'xyxy': [45, 45, 65, 108]},
            {'xyxy': [130, 35, 150, 109]},
            {'xyxy': [155, 30, 175, 100]},
        ]

        primary_labels = [
            {'xyxy': [8, 20, 18, 30], 'cleanedvalue': '0'},
            {'xyxy': [8, 80, 18, 90], 'cleanedvalue': '10'},
        ]
        secondary_labels = [
            {'xyxy': [180, 20, 190, 30], 'cleanedvalue': '0'},
            {'xyxy': [180, 80, 190, 90], 'cleanedvalue': '100'},
        ]

        result = detector.detect(
            img=img,
            chart_elements=chart_elements,
            axis_labels=primary_labels,
            secondary_axis_labels=secondary_labels,
            dual_axis_info={'has_dual_axis': True},
            orientation=Orientation.VERTICAL,
            chart_type=ChartType.BAR,
        )

        self.assertEqual(result.method, 'dual_axis_clustering')
        self.assertEqual(len(result.baselines), 2)
        self.assertEqual(result.baselines[0].axis_id, 'y1')
        self.assertEqual(result.baselines[1].axis_id, 'y2')


if __name__ == '__main__':
    unittest.main()
