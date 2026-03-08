
import unittest
import sys
import numpy as np
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

sys.path.append('src')

# Mock CV2 before import
sys.modules['cv2'] = MagicMock()
sys.modules['onnxruntime'] = MagicMock()
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
        
        # Mock Classification
        mock_inference.side_effect = [
            [
                {'cls': 0, 'conf': 0.9},  # generic "chart"
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
            {'cls': 0, 'conf': 0.98},  # generic "chart"
            {'cls': 7, 'conf': 0.97},  # histogram
        ]

        chart_type = self.pipeline._classify_chart_type(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(chart_type, 'histogram')

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

if __name__ == '__main__':
    unittest.main()
