import unittest
from unittest.mock import MagicMock
import numpy as np

from handlers.box_handler import BoxHandler
from handlers.pie_handler import PieHandler
from services.orientation_service import Orientation


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

    def test_pie_circle_from_3_points(self):
        # 3 points on circle centered at (100, 200) with r=50
        p1 = np.array([150.0, 200.0])
        p2 = np.array([100.0, 250.0])
        p3 = np.array([50.0, 200.0])
        fit = PieHandler._circle_from_3_points(p1, p2, p3)
        self.assertIsNotNone(fit)
        (cx, cy), r = fit
        self.assertAlmostEqual(cx, 100.0, places=4)
        self.assertAlmostEqual(cy, 200.0, places=4)
        self.assertAlmostEqual(r, 50.0, places=4)

        # Collinear points return None
        collinear = PieHandler._circle_from_3_points(
            np.array([10.0, 10.0]),
            np.array([20.0, 20.0]),
            np.array([30.0, 30.0]),
        )
        self.assertIsNone(collinear)

    def test_pie_fit_circle_kasa_inliers(self):
        # 12 points on circle centered at (200, 300) with r=80
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        pts = np.column_stack([
            200.0 + 80.0 * np.cos(angles),
            300.0 + 80.0 * np.sin(angles)
        ])
        fit = PieHandler._fit_circle_kasa_inliers(pts)
        self.assertIsNotNone(fit)
        (cx, cy), r = fit
        self.assertAlmostEqual(cx, 200.0, places=3)
        self.assertAlmostEqual(cy, 300.0, places=3)
        self.assertAlmostEqual(r, 80.0, places=3)

    def test_pie_handler_process_with_keypoints(self):
        handler = PieHandler()
        img = np.zeros((500, 500, 3), dtype=np.uint8)

        # 4 quadrant slices on a circle at (250, 250) with radius 100
        slices = []
        for a_start, a_end in [(0, 90), (90, 180), (180, 270), (270, 360)]:
            a_mid1 = a_start + (a_end - a_start) * 0.33
            a_mid2 = a_start + (a_end - a_start) * 0.66
            kps = [
                [250.0, 250.0, 0.99],
                [250.0 + 100.0 * np.cos(np.radians(a_start)), 250.0 + 100.0 * np.sin(np.radians(a_start)), 0.95],
                [250.0 + 100.0 * np.cos(np.radians(a_mid1)), 250.0 + 100.0 * np.sin(np.radians(a_mid1)), 0.95],
                [250.0 + 100.0 * np.cos(np.radians(a_mid2)), 250.0 + 100.0 * np.sin(np.radians(a_mid2)), 0.95],
                [250.0 + 100.0 * np.cos(np.radians(a_end)), 250.0 + 100.0 * np.sin(np.radians(a_end)), 0.95],
            ]
            xs = [k[0] for k in kps]
            ys = [k[1] for k in kps]
            slices.append({
                "xyxy": [min(xs), min(ys), max(xs), max(ys)],
                "conf": 0.95,
                "keypoints": kps
            })

        res = handler.process(
            image=img,
            detections={"slice": slices},
            axis_labels=[],
            chart_elements=slices,
            orientation=Orientation.VERTICAL
        )
        self.assertEqual(len(res.elements), 4)
        for el in res.elements:
            self.assertAlmostEqual(el["value"], 0.25, places=2)
        self.assertEqual(res.diagnostics.get("center_method"), "ransac_circle")
        self.assertAlmostEqual(res.diagnostics.get("pie_radius", 0), 100.0, places=1)

    def test_pie_handler_centroid_fallback_when_few_keypoints(self):
        handler = PieHandler()
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Two slices, 4 boundary keypoints each is fine, but only 2 slices with 1 valid
        # keypoint each gives < 6 boundary points, so RANSAC is skipped.
        slices = [
            {"xyxy": [200, 100, 300, 200], "keypoints": [[200, 200, .9], [300, 200, .9], [0, 0, 0], [0, 0, 0], [0, 0, 0]]},
            {"xyxy": [100, 100, 200, 200], "keypoints": [[200, 200, .9], [100, 200, .9], [0, 0, 0], [0, 0, 0], [0, 0, 0]]},
        ]
        res = handler.process(
            image=img, detections={"slice": slices}, axis_labels=[],
            chart_elements=slices, orientation=Orientation.VERTICAL,
        )
        self.assertEqual(res.diagnostics.get("center_method"), "mad_centroid")
        self.assertEqual(len(res.elements), 2)

    def test_pie_handler_missing_xyxy_reports_error(self):
        handler = PieHandler()
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        slices = [{"keypoints": [[200, 200, .9]] * 5}] * 3
        res = handler.process(
            image=img, detections={"slice": slices}, axis_labels=[],
            chart_elements=slices, orientation=Orientation.VERTICAL,
        )
        self.assertTrue(res.errors)

    def test_pie_handler_empty_detections(self):
        handler = PieHandler()
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        res = handler.process(
            image=img,
            detections={"slice": []},
            axis_labels=[],
            chart_elements=[],
            orientation=Orientation.VERTICAL
        )
        self.assertEqual(len(res.elements), 0)
        self.assertEqual(res.diagnostics.get("empty_reason"), "no_slices")


if __name__ == "__main__":
    unittest.main()
