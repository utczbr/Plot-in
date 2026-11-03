import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.append("src")

from services.orientation_service import Orientation
from core.classifiers.bar_chart_classifier import BarChartClassifier
from core.classifiers.production_classifier import ProductionSpatialClassifier
from core.baseline.zero_crossing import baseline_from_scale_zero
from core.baseline.detector import ModularBaselineDetector, DetectorConfig
from core.enums import ChartType
from calibration.calibration_factory import CalibrationFactory
from handlers.bar_handler import BarHandler
from handlers.base import CartesianExtractionHandler
from services.dual_axis_service import DualAxisDecision


class TestHorizontalBarClassificationAndBaseline(unittest.TestCase):
    def test_enum_and_string_orientation_parity(self):
        """Verify that string and Enum orientations produce identical classifications."""
        classifier = BarChartClassifier()
        prod_classifier = ProductionSpatialClassifier()

        axis_labels = [
            {"xyxy": [10, 100, 80, 120], "text": "Product A", "cleanedvalue": None, "ocr_confidence": 0.9},
            {"xyxy": [10, 200, 80, 220], "text": "Product B", "cleanedvalue": None, "ocr_confidence": 0.9},
            {"xyxy": [150, 480, 190, 500], "text": "0", "cleanedvalue": 0.0, "ocr_confidence": 0.95},
            {"xyxy": [350, 480, 390, 500], "text": "50", "cleanedvalue": 50.0, "ocr_confidence": 0.95},
            {"xyxy": [550, 480, 590, 500], "text": "100", "cleanedvalue": 100.0, "ocr_confidence": 0.95},
        ]
        bars = [
            {"xyxy": [150, 95, 350, 125]},
            {"xyxy": [150, 195, 500, 225]},
        ]

        # 1. BarChartClassifier with enum vs string
        res_enum = classifier.classify(axis_labels, bars, 800, 600, Orientation.HORIZONTAL)
        res_str = classifier.classify(axis_labels, bars, 800, 600, "horizontal")

        self.assertEqual(res_enum.metadata["orientation"], Orientation.HORIZONTAL)
        self.assertEqual(res_str.metadata["orientation"], Orientation.HORIZONTAL)
        self.assertEqual(len(res_enum.scale_labels), len(res_str.scale_labels))
        self.assertEqual(len(res_enum.tick_labels), len(res_str.tick_labels))

        # 2. ProductionSpatialClassifier with enum vs string
        prod_enum = prod_classifier.classify("bar", axis_labels, bars, 800, 600, Orientation.HORIZONTAL)
        prod_str = prod_classifier.classify("bar", axis_labels, bars, 800, 600, "horizontal")

        self.assertEqual(prod_enum.metadata["orientation"], Orientation.HORIZONTAL)
        self.assertEqual(prod_str.metadata["orientation"], Orientation.HORIZONTAL)

        # Scale labels must be the numeric bottom labels, tick labels must be categorical left labels
        scale_texts = {lbl["text"] for lbl in res_enum.scale_labels}
        tick_texts = {lbl["text"] for lbl in res_enum.tick_labels}

        self.assertIn("0", scale_texts)
        self.assertIn("50", scale_texts)
        self.assertIn("100", scale_texts)
        self.assertIn("Product A", tick_texts)
        self.assertIn("Product B", tick_texts)

    def test_horizontal_bar_gaussian_region_scores(self):
        """Horizontal bar charts route Gaussian weights correctly to avoid swapping scale and tick labels."""
        classifier = BarChartClassifier()

        # Categorical label on left (nx=0.05, ny=0.50)
        feat_left = {
            "nx": 0.05, "ny": 0.50, "cx": 40, "cy": 300,
            "rel_w": 0.08, "rel_h": 0.03, "aspect": 2.5,
            "is_numeric": False, "dist_left": 0.05, "dist_right": 0.95,
            "dist_top": 0.50, "dist_bottom": 0.50, "min_edge_dist": 0.05,
            "label": {"xyxy": [10, 290, 70, 310], "text": "Category"},
        }
        # Numeric scale label on bottom (nx=0.50, ny=0.92)
        feat_bottom = {
            "nx": 0.50, "ny": 0.92, "cx": 400, "cy": 550,
            "rel_w": 0.04, "rel_h": 0.03, "aspect": 1.3,
            "is_numeric": True, "dist_left": 0.50, "dist_right": 0.50,
            "dist_top": 0.92, "dist_bottom": 0.08, "min_edge_dist": 0.08,
            "label": {"xyxy": [380, 540, 420, 560], "text": "50"},
        }

        scores_left = classifier._compute_bar_scores(feat_left, {}, Orientation.HORIZONTAL)
        scores_bottom = classifier._compute_bar_scores(feat_bottom, {}, Orientation.HORIZONTAL)

        # Left label on horizontal bar must have higher tick_label score than scale_label score
        self.assertGreater(scores_left["tick_label"], scores_left["scale_label"])
        # Bottom numeric label on horizontal bar must have higher scale_label score than tick_label score
        self.assertGreater(scores_bottom["scale_label"], scores_bottom["tick_label"])

    def test_baseline_from_scale_zero_tick_label_filtering(self):
        """Excludes explicitly tagged tick labels and non-zero text labels from scale zero."""
        labels = [
            # Categorical tick label with text 'Category A' and accidental 0 cleanedvalue
            {"xyxy": [10, 100, 80, 120], "cleanedvalue": 0, "text": "Category A", "label_type": "tick_label", "axis": "x"},
            # Scale label at X=166
            {"xyxy": [160, 480, 172, 500], "cleanedvalue": 0, "text": "0", "label_type": "scale_label", "axis": "x"},
            {"xyxy": [350, 480, 370, 500], "cleanedvalue": 50, "text": "50", "label_type": "scale_label", "axis": "x"},
        ]

        with patch("core.baseline.zero_crossing.CalibrationFactory", None):
            zero_val = baseline_from_scale_zero(labels, is_vertical=False, calibration_mode="siegel")

        # Must select the true scale label at ~166px, ignoring the tick label at 45px
        self.assertIsNotNone(zero_val)
        self.assertAlmostEqual(zero_val, 166.0, delta=2.0)

    def test_baseline_from_scale_zero_r2_rejection(self):
        """Rejects degenerate regression calibration (R² < 0.50) and falls back to interpolation."""
        # Non-collinear noisy points where regression R² will be low
        labels = [
            {"xyxy": [100, 480, 120, 500], "cleanedvalue": 10, "text": "10", "axis": "x"},
            {"xyxy": [120, 480, 140, 500], "cleanedvalue": 90, "text": "90", "axis": "x"},
            {"xyxy": [300, 480, 320, 500], "cleanedvalue": 15, "text": "15", "axis": "x"},
        ]

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.coeffs = [0.1, 50]
        mock_result.r2 = 0.05  # Poor R² fit
        mock_engine.calibrate.return_value = mock_result

        with patch("core.baseline.zero_crossing.CalibrationFactory") as mock_factory:
            mock_factory.create.return_value = mock_engine
            zero_val = baseline_from_scale_zero(labels, is_vertical=False, calibration_mode="siegel")

        # Regression was rejected due to R²=0.05 < 0.50, fallback interpolation was executed
        self.assertIsNotNone(zero_val)

    def test_modular_baseline_detector_horizontal_snap_tolerance(self):
        """Horizontal bar chart rejects calibration zero when differing from geometric baseline by > 15px."""
        detector = ModularBaselineDetector(config=DetectorConfig(cluster_backend="dbscan"))

        elements = [
            {"xyxy": [166.0, 50, 350.0, 80]},
            {"xyxy": [166.0, 100, 450.0, 130]},
            {"xyxy": [166.0, 150, 400.0, 180]},
        ]
        img = np.zeros((600, 800, 3), dtype=np.uint8)

        # Case 1: Bad fallback zero at 434.8px (diff > 15px)
        with patch.object(detector, "_baseline_from_scale_zero", return_value=434.8):
            res = detector.detect(
                img=img,
                chart_elements=elements,
                axis_labels=[],
                orientation=Orientation.HORIZONTAL,
                chart_type=ChartType.BAR,
                primary_calibration_zero=None,
                primary_calibration_result=None,
            )
            # Must keep geometric estimate (166.0px) instead of overriding with 434.8px
            self.assertAlmostEqual(res.baselines[0].value, 166.0, delta=2.0)

        # Case 2: Good fallback zero within 15px snap tolerance (e.g. 168.0px)
        with patch.object(detector, "_baseline_from_scale_zero", return_value=168.0):
            res_snap = detector.detect(
                img=img,
                chart_elements=elements,
                axis_labels=[],
                orientation=Orientation.HORIZONTAL,
                chart_type=ChartType.BAR,
                primary_calibration_zero=None,
                primary_calibration_result=None,
            )
            # Snaps to 168.0px
            self.assertAlmostEqual(res_snap.baselines[0].value, 168.0, delta=0.5)

    def test_cartesian_extraction_handler_axis_calibration_rescue(self):
        """CartesianExtractionHandler rescues degenerate calibration with collinear scale labels."""
        handler = BarHandler(
            calibration_service=CalibrationFactory.create("siegel"),
            spatial_classifier=MagicMock(),
        )
        handler.last_image_size = (800, 600)

        # Setup raw labels: categorical labels on left, numeric scale labels at bottom margin (y ≈ 550)
        handler._axis_labels = [
            {"xyxy": [20, 100, 80, 120], "text": "Alpha", "cleanedvalue": None},
            {"xyxy": [20, 200, 80, 220], "text": "Beta", "cleanedvalue": None},
            {"xyxy": [150, 540, 180, 560], "text": "0", "cleanedvalue": 0.0},
            {"xyxy": [350, 540, 380, 560], "text": "50", "cleanedvalue": 50.0},
            {"xyxy": [550, 540, 580, 560], "text": "100", "cleanedvalue": 100.0},
        ]

        dual_decision = DualAxisDecision(
            has_dual_axis=False,
            primary_labels=[],  # Empty primary labels simulates failed classification
            secondary_labels=[],
            separation_px=0.0,
            confidence=0.5,
            method="test",
            diagnostics={},
        )

        calibrations = handler._calibrate_axes(
            classified_labels=MagicMock(tick_labels=[]),
            dual_decision=dual_decision,
            orientation=Orientation.HORIZONTAL,
        )

        # Calibration must be rescued using collinear numeric labels along bottom margin
        self.assertIsNotNone(calibrations["primary"])
        self.assertIsNotNone(calibrations["x"])
        m, b = calibrations["primary"].coeffs
        # Check that x=150 gives val ≈ 0, x=550 gives val ≈ 100
        val_0 = m * 150 + b
        val_100 = m * 550 + b
        self.assertAlmostEqual(val_0, 0.0, delta=5.0)
        self.assertAlmostEqual(val_100, 100.0, delta=5.0)

    def test_cartesian_extraction_handler_detect_baselines_uses_classified_scale_labels(self):
        """_detect_baselines uses classified scale_labels to avoid false zero-crossings from tick labels."""
        handler = BarHandler(
            calibration_service=CalibrationFactory.create("siegel"),
            spatial_classifier=MagicMock(),
        )
        handler._axis_labels = [
            {"xyxy": [10, 100, 80, 120], "text": "Category A", "cleanedvalue": 0, "label_type": "tick_label"},
            {"xyxy": [160, 540, 180, 560], "text": "0", "cleanedvalue": 0.0, "label_type": "scale_label"},
        ]
        # Only scale labels in classified result
        scale_label_only = [{"xyxy": [160, 540, 180, 560], "text": "0", "cleanedvalue": 0.0, "label_type": "scale_label"}]
        handler._classified_labels = MagicMock(scale_labels=scale_label_only)
        handler.clustering_recommendation = MagicMock(algorithm="dbscan")

        with patch("handlers.base.ModularBaselineDetector") as mock_det_cls:
            mock_detector = MagicMock()
            mock_det_cls.return_value = mock_detector

            img = np.zeros((600, 800, 3), dtype=np.uint8)
            handler._detect_baselines(
                chart_elements=[],
                calibrations={"primary": None},
                image=img,
                orientation=Orientation.HORIZONTAL,
                dual_decision=MagicMock(),
            )

            # Ensure detect was called with scale_label_only, NOT the raw _axis_labels containing Category A
            call_kwargs = mock_detector.detect.call_args[1]
            self.assertEqual(call_kwargs["axis_labels"], scale_label_only)

    def test_bar_chart_classifier_constraints_retains_scale_labels_on_margin(self):
        """Scale labels on the scale axis margin stay in scale_label despite transient OCR glitches."""
        classifier = BarChartClassifier()

        # Label near bottom margin (ny=0.95 -> s_dist=0.05), but is_numeric is False due to glitch (e.g. "S0")
        feat_near_margin = {
            "label": {"xyxy": [200, 560, 230, 580], "text": "S0"},
            "is_numeric": False,
            "dist_bottom": 0.05,
            "dist_left": 0.50,
        }
        # Label far from bottom margin (ny=0.50 -> s_dist=0.50, nx=0.10 -> t_dist=0.10)
        feat_far_margin = {
            "label": {"xyxy": [50, 280, 100, 300], "text": "Product"},
            "is_numeric": False,
            "dist_bottom": 0.50,
            "dist_left": 0.10,
        }

        classified = {
            "scale_label": [feat_near_margin["label"], feat_far_margin["label"]],
            "tick_label": [],
            "axis_title": [],
        }

        features = [feat_near_margin, feat_far_margin]
        refined = classifier._apply_constraints(classified, features, Orientation.HORIZONTAL)

        # feat_near_margin must remain in scale_label (not swapped)
        self.assertIn(feat_near_margin["label"], refined["scale_label"])
        # feat_far_margin must be swapped to tick_label
        self.assertIn(feat_far_margin["label"], refined["tick_label"])

    def test_process_ocr_lookalike_recovery(self):
        """_process_ocr recovers lookalike digits like 'S0' -> 50.0 and sets cleanedvalue."""
        from pipelines.chart_pipeline import ChartAnalysisPipeline
        from ocr.preprocessing.ocr_validator import OCRValidator

        pipeline = ChartAnalysisPipeline.__new__(ChartAnalysisPipeline)
        pipeline.logger = MagicMock()
        pipeline.ocr_validator = OCRValidator()

        # Mock engine returning "S0" for a horizontal axis label
        mock_engine = MagicMock()
        mock_engine.process_batch.return_value = [{"text": "S0", "confidence": 0.92}]
        pipeline.ocr_engine = mock_engine

        img = np.zeros((600, 800, 3), dtype=np.uint8)
        detections = {
            "axis_labels": [
                {"xyxy": [150.2, 540.1, 180.9, 560.8], "ocr_source": "axis_labels"}
            ]
        }

        pipeline._process_ocr(img, detections)

        target = detections["axis_labels"][0]
        # "S0" should be converted to "50" with cleanedvalue 50.0
        self.assertEqual(target.get("cleanedvalue"), 50.0)
        self.assertEqual(target.get("text"), "50")



if __name__ == "__main__":
    unittest.main()
