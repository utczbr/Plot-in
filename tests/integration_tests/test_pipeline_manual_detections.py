import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pipelines.chart_pipeline import ChartAnalysisPipeline

@pytest.fixture
def dummy_pipeline():
    mock_mm = MagicMock()
    mock_ocr = MagicMock()
    mock_calib = MagicMock()
    return ChartAnalysisPipeline(
        models_manager=mock_mm,
        ocr_engine=mock_ocr,
        calibration_engine=mock_calib
    )

def test_pipeline_manual_detections_bypass(dummy_pipeline, tmp_path):
    # Prepare fake manual detections
    manual_dets = {
        "chart_title": [{"xyxy": [10, 10, 100, 30], "conf": 1.0}],
        "legend": [{"xyxy": [200, 200, 300, 300], "conf": 0.9}]
    }

    # Create dummy image
    image_path = tmp_path / "test.jpg"
    image_path.touch()

    # Mock the internal detection and extraction methods
    with patch('pipelines.chart_pipeline.cv2.imread', return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
         patch.object(dummy_pipeline, '_classify_chart_types', return_value=(['bar'], 1.0)), \
         patch.object(dummy_pipeline, '_detect_elements') as mock_detect_elements, \
         patch.object(dummy_pipeline, '_detect_text_layout', return_value={}) as mock_layout, \
         patch.object(dummy_pipeline, '_detect_orientation', return_value=MagicMock(value="UP")) as mock_orientation, \
         patch.object(dummy_pipeline, '_format_result', return_value={}) as mock_format, \
         patch.object(dummy_pipeline, '_save_results', return_value={}) as mock_save, \
         patch.object(dummy_pipeline, '_strategy_router') as mock_router:
         
         # Mock strategy execution to not crash
         mock_strategy = MagicMock()
         mock_strategy_result = MagicMock()
         mock_strategy_result.errors = []
         mock_strategy_result.elements = []
         mock_strategy.execute.return_value = mock_strategy_result
         mock_router.select.return_value = mock_strategy

         # Run pipeline with manual_detections
         result = dummy_pipeline.run(
             image_input=image_path,
             output_dir=str(tmp_path),
             manual_detections=manual_dets
         )

         # Verify _detect_elements was NOT called
         mock_detect_elements.assert_not_called()

         # Verify that the strategy selection executed
         assert mock_router.select.called

         # Verify the provenance and review status
         assert result, "Pipeline result should not be None"
         assert result.get('review_status') == 'reviewed'
         assert result.get('correction_source') == 'manual_edit'
