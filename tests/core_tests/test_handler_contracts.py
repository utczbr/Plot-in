import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

sys.path.append('src')
sys.modules.setdefault('sklearn', MagicMock())
sys.modules.setdefault('sklearn.cluster', MagicMock())
sys.modules.setdefault('sklearn.metrics', MagicMock())
sys.modules.setdefault('sklearn.ensemble', MagicMock())
sys.modules.setdefault('sklearn.preprocessing', MagicMock())

from handlers.line_handler import LineHandler


class TestHandlerContracts(unittest.TestCase):
    def test_line_handler_accepts_object_based_baseline_and_calibration(self):
        handler = LineHandler(
            calibration_service=MagicMock(),
            spatial_classifier=MagicMock(),
            dual_axis_service=MagicMock(),
            meta_clustering_service=MagicMock(),
            logger=MagicMock(),
        )

        detections = {
            "data_point": [
                {"xyxy": [10, 20, 20, 30], "conf": 0.95},
            ]
        }

        calibration_result = SimpleNamespace(func=lambda px: float(px), r2=0.99)
        calibration = {"primary": calibration_result, "y": calibration_result}
        baselines = SimpleNamespace(
            baselines=[
                SimpleNamespace(axis_id="y", value=40.0),
            ]
        )

        extracted = handler.extract_values(
            img=np.zeros((64, 64, 3), dtype=np.uint8),
            detections=detections,
            calibration=calibration,
            baselines=baselines,
            orientation="vertical",
        )

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0]["orientation"], "vertical")
        self.assertIn("value", extracted[0])


if __name__ == "__main__":
    unittest.main()
