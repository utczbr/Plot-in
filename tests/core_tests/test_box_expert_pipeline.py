"""
Unit and integration tests for box plot modular specialist models (box_global_detect and box_element_detect).
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from pathlib import Path

from pipelines.chart_pipeline import ChartAnalysisPipeline
from core.class_maps import (
    CLASS_MAP_BOX_GLOBAL,
    CLASS_MAP_BOX_ELEMENT,
    CLASS_MAP_BOX,
    get_class_map,
)
from models.config import MODELS_CONFIG
from core.model_manager import ModelManager


class TestBoxExpertPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_models = MagicMock()
        self.mock_ocr = MagicMock()
        self.mock_cal = MagicMock()
        self.pipeline = ChartAnalysisPipeline(
            models_manager=self.mock_models,
            ocr_engine=self.mock_ocr,
            calibration_engine=self.mock_cal
        )
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_class_maps_and_config(self):
        """Verify class map definitions and model config."""
        self.assertEqual(get_class_map('box_global'), CLASS_MAP_BOX_GLOBAL)
        self.assertEqual(get_class_map('box_element'), CLASS_MAP_BOX_ELEMENT)
        
        # Verify box_global classes
        self.assertIn('chart', CLASS_MAP_BOX_GLOBAL.values())
        self.assertIn('axis_title', CLASS_MAP_BOX_GLOBAL.values())
        self.assertIn('legend', CLASS_MAP_BOX_GLOBAL.values())
        self.assertIn('chart_title', CLASS_MAP_BOX_GLOBAL.values())
        self.assertIn('axis_labels', CLASS_MAP_BOX_GLOBAL.values())
        
        # Verify box_element classes
        self.assertIn('box', CLASS_MAP_BOX_ELEMENT.values())
        self.assertIn('range_indicator', CLASS_MAP_BOX_ELEMENT.values())
        self.assertIn('median_line', CLASS_MAP_BOX_ELEMENT.values())
        self.assertIn('outlier', CLASS_MAP_BOX_ELEMENT.values())
        self.assertIn('significance_marker', CLASS_MAP_BOX_ELEMENT.values())

        # Verify MODELS_CONFIG detection registration
        self.assertEqual(MODELS_CONFIG.detection.get('box_global'), 'box_global_detect.onnx')
        self.assertEqual(MODELS_CONFIG.detection.get('box_element'), 'box_element_detect.onnx')
        self.assertEqual(MODELS_CONFIG.detection_output_type.get('box_global'), 'yolo_nms')
        self.assertEqual(MODELS_CONFIG.detection_output_type.get('box_element'), 'yolo_nms')

    @patch('pipelines.chart_pipeline.run_inference_on_image')
    def test_detect_box_experts_combines_detections(self, mock_run_inference):
        """Verify _detect_box_experts runs both models and merges into organized dict."""
        self.mock_models.get_model.side_effect = lambda name: MagicMock() if name in ('box_global', 'box_element') else None

        def mock_run(model, img, conf, class_map, **kwargs):
            if class_map == CLASS_MAP_BOX_GLOBAL:
                return [
                    {'xyxy': [10, 10, 90, 90], 'cls': 0, 'conf': 0.95},  # chart
                    {'xyxy': [5, 45, 10, 55], 'cls': 1, 'conf': 0.88},   # axis_title
                    {'xyxy': [12, 92, 20, 98], 'cls': 4, 'conf': 0.85},  # axis_labels
                ]
            elif class_map == CLASS_MAP_BOX_ELEMENT:
                return [
                    {'xyxy': [20, 30, 40, 70], 'cls': 0, 'conf': 0.98},  # box
                    {'xyxy': [28, 15, 32, 85], 'cls': 1, 'conf': 0.92},  # range_indicator
                    {'xyxy': [20, 50, 40, 52], 'cls': 2, 'conf': 0.87},  # median_line
                    {'xyxy': [29, 10, 31, 12], 'cls': 3, 'conf': 0.76},  # outlier
                ]
            return []

        mock_run_inference.side_effect = mock_run

        detections = self.pipeline._detect_box_experts(self.img, advanced_settings={})

        self.assertIn('chart', detections)
        self.assertIn('axis_title', detections)
        self.assertIn('axis_labels', detections)
        self.assertIn('box', detections)
        self.assertIn('range_indicator', detections)
        self.assertIn('median_line', detections)
        self.assertIn('outlier', detections)

        self.assertEqual(len(detections['chart']), 1)
        self.assertEqual(len(detections['axis_title']), 1)
        self.assertEqual(len(detections['axis_labels']), 1)
        self.assertEqual(len(detections['box']), 1)
        self.assertEqual(len(detections['range_indicator']), 1)
        self.assertEqual(len(detections['median_line']), 1)
        self.assertEqual(len(detections['outlier']), 1)

    @patch('pipelines.chart_pipeline.run_inference_on_image')
    def test_detect_box_experts_respects_overrides(self, mock_run_inference):
        """Verify _detect_box_experts respects custom confidence & NMS overrides."""
        self.mock_models.get_model.side_effect = lambda name: MagicMock() if name in ('box_global', 'box_element') else None
        mock_run_inference.return_value = []

        advanced_settings = {
            'detection_confidence_overrides': {'box': 0.62},
            'detection_nms_overrides': {'box': 0.85},
        }

        self.pipeline._detect_box_experts(self.img, advanced_settings=advanced_settings)

        # Check call arguments to run_inference_on_image
        self.assertEqual(mock_run_inference.call_count, 2)
        for call_args in mock_run_inference.call_args_list:
            args, kwargs = call_args
            self.assertAlmostEqual(args[2], 0.62)  # conf
            self.assertAlmostEqual(kwargs.get('nms_threshold'), 0.85)

    def test_integration_real_box_models(self):
        """Integration test with actual ONNX models on a sample image."""
        models_dir = Path("src/models")
        global_path = models_dir / "box_global_detect.onnx"
        element_path = models_dir / "box_element_detect.onnx"

        if not global_path.exists() or not element_path.exists():
            self.skipTest("ONNX model files not found in src/models")

        test_img_path = Path("src/images/box/chart_00001.png")
        if not test_img_path.exists():
            self.skipTest("Test image chart_00001.png not found")

        img = cv2.imread(str(test_img_path))
        self.assertIsNotNone(img)

        # Initialize real ModelManager
        mm = ModelManager()
        mm.reset_models()
        mm.load_models(str(models_dir), force_reload=True)

        pipeline = ChartAnalysisPipeline(
            models_manager=mm,
            ocr_engine=MagicMock(),
            calibration_engine=MagicMock(),
        )
        detections = pipeline._detect_box_experts(img)

        # Assert detections are found
        self.assertGreater(len(detections.get('chart', [])), 0)
        self.assertGreater(len(detections.get('box', [])), 0)
        self.assertGreater(len(detections.get('range_indicator', [])), 0)
        self.assertGreater(len(detections.get('median_line', [])), 0)
        self.assertGreater(len(detections.get('axis_labels', [])), 0)


if __name__ == '__main__':
    unittest.main()
