import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.baseline.clustering import cluster_1d_dbscan
from core.class_maps import (
    CLASS_MAP_HEATMAP_COLORBAR,
    CLASS_MAP_HEATMAP_LATTICE,
    CLASS_MAP_HEATMAP_MACRO,
    CLASS_MAP_HEATMAP_TEXT,
)
from core.enums import Orientation
from handlers.heatmap_handler import HeatmapHandler
from pipelines.chart_pipeline import ChartAnalysisPipeline


class TestHeatmapExpertPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_models = MagicMock()
        self.mock_ocr = MagicMock()
        self.mock_cal = MagicMock()
        self.pipeline = ChartAnalysisPipeline(
            models_manager=self.mock_models,
            ocr_engine=self.mock_ocr,
            calibration_engine=self.mock_cal,
        )
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)

    @patch("pipelines.chart_pipeline.run_inference_on_image")
    def test_detect_heatmap_experts_respects_override(self, mock_run_inference):
        def mock_run(model, img, conf, class_map, nms_threshold=None, **kwargs):
            if class_map == CLASS_MAP_HEATMAP_MACRO:
                return [{"xyxy": [10, 10, 80, 80], "cls": 0, "conf": 0.9}]
            elif class_map == CLASS_MAP_HEATMAP_LATTICE:
                return [{"xyxy": [5, 5, 15, 15], "cls": 0, "conf": 0.85}]
            elif class_map == CLASS_MAP_HEATMAP_COLORBAR:
                return [{"xyxy": [2, 2, 8, 8], "cls": 0, "conf": 0.88}]
            elif class_map == CLASS_MAP_HEATMAP_TEXT:
                return [{"xyxy": [0, 0, 5, 5], "cls": 0, "conf": 0.82}]
            return []

        mock_run_inference.side_effect = mock_run

        self.pipeline._detect_heatmap_experts(self.img, advanced_settings={})
        first_call_args = mock_run_inference.call_args_list[0]
        self.assertAlmostEqual(first_call_args[0][2], 0.4)

        mock_run_inference.reset_mock()
        advanced_settings = {"detection_confidence_overrides": {"heatmap": 0.65}}
        self.pipeline._detect_heatmap_experts(self.img, advanced_settings=advanced_settings)
        first_call_args = mock_run_inference.call_args_list[0]
        self.assertAlmostEqual(first_call_args[0][2], 0.65)

    @patch("pipelines.chart_pipeline.run_inference_on_image")
    def test_heatmap_rescue_respects_override(self, mock_run_inference):
        self.mock_models.get_model.return_value = MagicMock()
        mock_run_inference.return_value = [
            {"xyxy": [10, 10, 20, 20], "cls": 0, "conf": 0.9}
        ] * 15

        self.pipeline._heatmap_rescue(self.img, current_types=[], advanced_settings={})
        first_call_args = mock_run_inference.call_args_list[0]
        self.assertAlmostEqual(first_call_args[0][2], 0.4)

        mock_run_inference.reset_mock()
        advanced_settings = {"detection_confidence_overrides": {"heatmap": 0.72}}
        self.pipeline._heatmap_rescue(self.img, current_types=[], advanced_settings=advanced_settings)
        first_call_args = mock_run_inference.call_args_list[0]
        self.assertAlmostEqual(first_call_args[0][2], 0.72)

    def test_cluster_1d_dbscan(self):
        values = [10, 11, 12, 100, 101, 102]
        centers = cluster_1d_dbscan(values, tolerance=5)
        self.assertEqual(len(centers), 2)
        self.assertAlmostEqual(centers[0], 11.0, delta=0.5)
        self.assertAlmostEqual(centers[1], 101.0, delta=0.5)

        values = [10, 12, 14]
        centers = cluster_1d_dbscan(values, tolerance=10)
        self.assertEqual(len(centers), 1)
        self.assertAlmostEqual(centers[0], 12.0)
        self.assertEqual(cluster_1d_dbscan([], 1.0), [])

    def test_heatmap_process_flow(self):
        handler = HeatmapHandler(classifier=MagicMock(), color_mapper=MagicMock())
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = {
            "heatmap_cell": [
                {"xyxy": [10, 10, 40, 40], "conf": 0.9},
                {"xyxy": [60, 10, 90, 40], "conf": 0.9},
                {"xyxy": [10, 60, 40, 90], "conf": 0.9},
                {"xyxy": [60, 60, 90, 90], "conf": 0.9},
            ],
            "color_bar": [],
        }
        result = handler.process(
            image=fake_image,
            detections=detections,
            axis_labels=[],
            chart_elements=[],
            orientation=Orientation.VERTICAL,
        )
        self.assertEqual(result.chart_type, "heatmap")
        self.assertEqual(result.diagnostics["grid_rows"], 2)
        self.assertEqual(result.diagnostics["grid_cols"], 2)
        self.assertEqual(len(result.elements), 4)


if __name__ == "__main__":
    unittest.main()
