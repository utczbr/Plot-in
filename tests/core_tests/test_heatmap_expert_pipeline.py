import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pipelines.chart_pipeline import ChartAnalysisPipeline
from core.class_maps import (
    CLASS_MAP_HEATMAP_MACRO,
    CLASS_MAP_HEATMAP_LATTICE,
    CLASS_MAP_HEATMAP_COLORBAR,
    CLASS_MAP_HEATMAP_TEXT,
)

class TestHeatmapExpertPipeline(unittest.TestCase):
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

    @patch('pipelines.chart_pipeline.run_inference_on_image')
    def test_detect_heatmap_experts_respects_override(self, mock_run_inference):
        # Setup mock detections
        # Macro returns chart ROI
        # Lattice returns cells
        # Colorbar returns colorbars
        # Text returns axis labels
        def mock_run(model, img, conf, class_map, nms_threshold=None):
            if class_map == CLASS_MAP_HEATMAP_MACRO:
                return [{'xyxy': [10, 10, 80, 80], 'cls': 0, 'conf': 0.9}] # chart
            elif class_map == CLASS_MAP_HEATMAP_LATTICE:
                return [{'xyxy': [5, 5, 15, 15], 'cls': 0, 'conf': 0.85}] # cell
            elif class_map == CLASS_MAP_HEATMAP_COLORBAR:
                return [{'xyxy': [2, 2, 8, 8], 'cls': 0, 'conf': 0.88}] # colorbar
            elif class_map == CLASS_MAP_HEATMAP_TEXT:
                return [{'xyxy': [0, 0, 5, 5], 'cls': 0, 'conf': 0.82}] # axis label
            return []

        mock_run_inference.side_effect = mock_run

        # Scenario 1: No override (default to 0.4)
        self.pipeline._detect_heatmap_experts(self.img, advanced_settings={})
        
        # Verify first call (macro model) was called with conf=0.4
        first_call_args = mock_run_inference.call_args_list[0]
        self.assertAlmostEqual(first_call_args[0][2], 0.4)

        # Scenario 2: With override (e.g. 0.65)
        mock_run_inference.reset_mock()
        advanced_settings = {
            'detection_confidence_overrides': {
                'heatmap': 0.65
            }
        }
        self.pipeline._detect_heatmap_experts(self.img, advanced_settings=advanced_settings)
        
        first_call_args = mock_run_inference.call_args_list[0]
        self.assertAlmostEqual(first_call_args[0][2], 0.65)

    @patch('pipelines.chart_pipeline.run_inference_on_image')
    def test_heatmap_rescue_respects_override(self, mock_run_inference):
        # Mock models to return truthy
        self.mock_models.get_model.return_value = MagicMock()
        
        # Setup mock run_inference
        # Returns cells so count is > rescue threshold
        mock_run_inference.return_value = [
            {'xyxy': [10, 10, 20, 20], 'cls': 0, 'conf': 0.9}
        ] * 15 # cell class = 0

        # Scenario 1: No override (default to 0.4)
        self.pipeline._heatmap_rescue(self.img, current_types=[], advanced_settings={})
        
        # Macro model called first with conf=0.4
        first_call_args = mock_run_inference.call_args_list[0]
        self.assertAlmostEqual(first_call_args[0][2], 0.4)

        # Scenario 2: With override (e.g. 0.72)
        mock_run_inference.reset_mock()
        advanced_settings = {
            'detection_confidence_overrides': {
                'heatmap': 0.72
            }
        }
        self.pipeline._heatmap_rescue(self.img, current_types=[], advanced_settings=advanced_settings)
        
        first_call_args = mock_run_inference.call_args_list[0]
        self.assertAlmostEqual(first_call_args[0][2], 0.72)

if __name__ == '__main__':
    unittest.main()
