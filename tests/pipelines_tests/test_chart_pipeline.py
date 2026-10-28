
import unittest
import sys
import numpy as np
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

sys.path.append('src')

# Mock CV2 before import only if not installed
sys.modules.setdefault('cv2', MagicMock())
sys.modules.setdefault('onnxruntime', MagicMock())
sys.modules.setdefault('sklearn', MagicMock())
sys.modules.setdefault('sklearn.cluster', MagicMock())
sys.modules.setdefault('sklearn.metrics', MagicMock())
sys.modules.setdefault('sklearn.ensemble', MagicMock())
sys.modules.setdefault('sklearn.preprocessing', MagicMock())

from pipelines.chart_pipeline import ChartAnalysisPipeline
from services.orientation_service import Orientation
from handlers.types import ExtractionResult, ChartCoordinateSystem

class TestChartPipeline(unittest.TestCase):
    
    def setUp(self):
        self.mock_models = MagicMock()
        self.mock_ocr = MagicMock()
        self.mock_calibration = MagicMock()
        self.pipeline = ChartAnalysisPipeline(
            self.mock_models, self.mock_ocr, self.mock_calibration
        )
        
    def test_instantiation(self):
        self.assertIsInstance(self.pipeline, ChartAnalysisPipeline)
        
    @patch('pipelines.chart_pipeline.cv2.imread')
    @patch('pipelines.chart_pipeline.run_inference_on_image')
    @patch('pipelines.chart_pipeline.ChartAnalysisOrchestrator')
    def test_run_basic_flow(self, mock_orch, mock_inference, mock_imread):
        # Setup Mocks
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock Classification (bar at 0.89 conf skips heatmap rescue — above 0.70 threshold)
        mock_inference.side_effect = [
            [
                {'cls': 1, 'conf': 0.89}, # specific "bar"
            ],
            [{'cls': 0, 'conf': 0.9, 'xyxy': [10, 10, 50, 50]}] # Detection: Bar
        ]
        
        # Mock Orchestrator
        mock_orch_instance = mock_orch.return_value
        mock_orch_instance.process_chart.return_value = ExtractionResult(
            chart_type='bar',
            coordinate_system=ChartCoordinateSystem.CARTESIAN,
            elements=[],
            calibration={},
            baselines=[],
            diagnostics={},
            errors=[],
            warnings=[],
            orientation=Orientation.VERTICAL
        )
        
        # Run
        result = self.pipeline.run("test.png")
        
        # Verify
        self.assertIsNotNone(result)
        self.assertEqual(result['chart_type'], 'bar')
        self.assertEqual(result['orientation'], 'vertical')
        mock_inference.assert_called()
        mock_orch_instance.process_chart.assert_called()

    @patch('pipelines.chart_pipeline.run_inference_on_image')
    def test_classification_prefers_specific_over_generic_chart(self, mock_inference):
        mock_model = MagicMock()
        self.mock_models.get_model.return_value = mock_model
        mock_inference.return_value = [
            {'cls': 4, 'conf': 0.97},  # histogram
        ]

        types, conf = self.pipeline._classify_chart_types(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(types[0], 'histogram')

    @patch('pipelines.chart_pipeline.MODELS_CONFIG', new=SimpleNamespace(
        detection_output_type={'pie': 'pose'},
        detection_keypoints={'pie': 5},
    ))
    @patch('pipelines.chart_pipeline.run_inference_on_image')
    def test_detect_elements_uses_pose_mode_for_pie(self, mock_inference):
        mock_model = MagicMock()
        self.mock_models.get_model.return_value = mock_model
        mock_inference.return_value = [{'cls': 0, 'conf': 0.9, 'xyxy': [1, 1, 10, 10]}]

        result = self.pipeline._detect_elements(np.zeros((10, 10, 3), dtype=np.uint8), 'pie')

        self.assertIn('slice', result)
        self.assertEqual(len(result['slice']), 1)
        _args, kwargs = mock_inference.call_args
        self.assertEqual(kwargs.get('model_output_type'), 'pose')
        self.assertEqual(kwargs.get('expected_keypoints'), 5)

    @patch('pipelines.chart_pipeline.MODELS_CONFIG', new=SimpleNamespace(
        detection_output_type={'pie': 'pose'},
        detection_keypoints={'pie': 5},
    ))
    @patch('pipelines.chart_pipeline.run_inference_on_image')
    def test_detect_elements_uses_bbox_mode_for_non_pie(self, mock_inference):
        mock_model = MagicMock()
        self.mock_models.get_model.return_value = mock_model
        mock_inference.return_value = [{'cls': 1, 'conf': 0.9, 'xyxy': [1, 1, 10, 10]}]

        result = self.pipeline._detect_elements(np.zeros((10, 10, 3), dtype=np.uint8), 'bar')

        self.assertIn('bar', result)
        self.assertEqual(len(result['bar']), 1)
        _args, kwargs = mock_inference.call_args
        self.assertEqual(kwargs.get('model_output_type'), 'bbox')
        self.assertIsNone(kwargs.get('expected_keypoints'))

    def test_issue_c_process_ocr_reads_chart_title(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = {'chart_title': [{'xyxy': [0, 0, 10, 10]}]}
        self.pipeline.ocr_engine.process_batch.return_value = [{"text": "MockTitle", "confidence": 0.99}]
        self.pipeline._process_ocr(img, detections)
        self.assertEqual(len(detections.get('chart_title', [])), 1)
        self.assertEqual(detections['chart_title'][0]['text'], "MockTitle")
        self.assertEqual(detections['chart_title'][0]['ocr_source'], "chart_title")

    def test_issue_d_process_ocr_reads_legend(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = {'legend': [{'xyxy': [0, 0, 10, 10]}]}
        self.pipeline.ocr_engine.process_batch.return_value = [{"text": "LegendA", "confidence": 0.99}]
        self.pipeline._process_ocr(img, detections)
        self.assertEqual(len(detections.get('legend', [])), 1)
        self.assertEqual(detections['legend'][0]['text'], "LegendA")
        self.assertEqual(detections['legend'][0]['ocr_source'], "legend")

    def test_issue_e_legend_misclassified_as_title(self):
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        self.pipeline.models_manager.get_model.return_value = MagicMock()
        with patch('pipelines.chart_pipeline.run_inference_on_image') as mock_run:
            mock_run.return_value = [
                {'xyxy': [50, 10, 950, 40], 'cls': 0},
                {'xyxy': [800, 100, 900, 300], 'cls': 1},
            ]
            with patch('pipelines.chart_pipeline.get_class_map') as mock_map:
                mock_map.return_value = {0: 'chart_title', 1: 'legend'}
                organized = self.pipeline._detect_elements(img, 'bar')
                self.assertEqual(len(organized['chart_title']), 1)
                self.assertEqual(organized['chart_title'][0]['xyxy'], [50, 10, 950, 40])
                self.assertEqual(len(organized['legend']), 1)
                self.assertEqual(organized['legend'][0]['xyxy'], [800, 100, 900, 300])

    def test_classify_chart_types_error_fallback_to_unknown(self):
        self.mock_models.get_model.side_effect = RuntimeError("Models not loaded")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        types, top_conf = self.pipeline._classify_chart_types(img)
        self.assertEqual(types, ['unknown'])
        self.assertEqual(top_conf, 0.0)


if __name__ == '__main__':
    unittest.main()
