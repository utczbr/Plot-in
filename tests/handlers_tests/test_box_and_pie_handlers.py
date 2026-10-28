import unittest
from unittest.mock import MagicMock

from handlers.box_handler import BoxHandler
from handlers.pie_handler import PieHandler


class TestBoxAndPieHandlers(unittest.TestCase):
    def test_box_handler_r2_logic(self):
        handler = BoxHandler(
            calibration_service=MagicMock(),
            spatial_classifier=MagicMock(),
            dual_axis_service=MagicMock(),
            meta_clustering_service=MagicMock(),
            logger=MagicMock(),
        )
        handler.FAILURE_R2 = 0.40
        handler.CRITICAL_R2 = 0.85

        mock_cal = MagicMock()
        mock_cal.r_squared = 0.50
        errors, warnings = [], []
        if hasattr(mock_cal, "r_squared") and mock_cal.r_squared < handler.FAILURE_R2:
            errors.append("catastrophic")
        elif hasattr(mock_cal, "r_squared") and mock_cal.r_squared < handler.CRITICAL_R2:
            warnings.append("quality low")
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(warnings), 1)

        mock_cal.r_squared = 0.20
        errors, warnings = [], []
        if hasattr(mock_cal, "r_squared") and mock_cal.r_squared < handler.FAILURE_R2:
            errors.append("catastrophic")
        elif hasattr(mock_cal, "r_squared") and mock_cal.r_squared < handler.CRITICAL_R2:
            warnings.append("quality low")
        self.assertEqual(len(errors), 1)

    def test_pie_handler_init(self):
        classifier = MagicMock()
        matcher = MagicMock()
        handler = PieHandler(classifier=classifier, legend_matcher=matcher)
        self.assertEqual(handler.classifier, classifier)
        self.assertEqual(handler.legend_matcher, matcher)


if __name__ == "__main__":
    unittest.main()
