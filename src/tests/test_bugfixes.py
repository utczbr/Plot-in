import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from pipelines.chart_pipeline import ChartAnalysisPipeline
from ChartAnalysisOrchestrator import ChartAnalysisOrchestrator
from visual.data_tab_schema import _build_rows, build_data_tab_model
from handlers.bar_handler import BarHandler
from visual.data_tab_schema import _safe_float

def test_issue_a_bar_schema_reads_bars():
    # Verify build_data_tab_model doesn't fail when elements are empty but bars exist
    result = {
        "chart_type": "bar",
        "bars": [{"confidence": 0.9, "value": 10.0}],
        "elements": []
    }
    # It should read from "bars"
    rows = _build_rows(result, "bar")
    assert len(rows) == 1
    assert rows[0].values['value'] == 10.0

def test_issue_c_process_ocr_reads_chart_title():
    pipeline = ChartAnalysisPipeline(MagicMock(), MagicMock(), MagicMock())
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = {
        'chart_title': [{'xyxy': [0, 0, 10, 10]}]
    }
    # Mock ocr_engine to return "MockTitle"
    pipeline.ocr_engine.process_batch.return_value = [{"text": "MockTitle", "confidence": 0.99}]
    
    pipeline._process_ocr(img, detections)
    # The title should be added to axis_labels and carry text
    assert len(detections.get('axis_labels', [])) == 1
    assert detections['axis_labels'][0]['text'] == "MockTitle"
    assert detections['axis_labels'][0]['ocr_source'] == "chart_title"

def test_issue_d_process_ocr_reads_legend():
    pipeline = ChartAnalysisPipeline(MagicMock(), MagicMock(), MagicMock())
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = {
        'legend': [{'xyxy': [0, 0, 10, 10]}]
    }
    pipeline.ocr_engine.process_batch.return_value = [{"text": "LegendA", "confidence": 0.99}]
    
    pipeline._process_ocr(img, detections)
    assert len(detections.get('axis_labels', [])) == 1
    assert detections['axis_labels'][0]['text'] == "LegendA"
    assert detections['axis_labels'][0]['ocr_source'] == "legend"

def test_issue_e_legend_misclassified_as_title():
    pipeline = ChartAnalysisPipeline(MagicMock(), MagicMock(), MagicMock())
    # Mock img of width 1000
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    
    # A wide box (width 900) should be title
    # A tall box should be legend
    detections = {
        'chart_title': [
            {'xyxy': [50, 10, 950, 40], 'cls': 0} # Width 900 -> > 0.85*1000 -> title
        ],
        'legend': [
            {'xyxy': [800, 100, 900, 300], 'cls': 1} # Width 100, height 200 -> legend
        ]
    }
    
    # Mock models_manager to return a dummy model
    pipeline.models_manager.get_model.return_value = MagicMock()
    with patch('pipelines.chart_pipeline.run_inference_on_image') as mock_run:
        # Just return the same boxes as raw_dets with mapped classes
        mock_run.return_value = [
            {'xyxy': [50, 10, 950, 40], 'cls': 0}, # mapped to chart_title
            {'xyxy': [800, 100, 900, 300], 'cls': 1} # mapped to legend
        ]
        with patch('pipelines.chart_pipeline.get_class_map') as mock_map:
            mock_map.return_value = {0: 'chart_title', 1: 'legend'}
            organized = pipeline._detect_elements(img, 'bar')
            
            assert len(organized['chart_title']) == 1
            assert organized['chart_title'][0]['xyxy'] == [50, 10, 950, 40]
            assert len(organized['legend']) == 1
            assert organized['legend'][0]['xyxy'] == [800, 100, 900, 300]
            
def test_issue_g_line_chart_error_bar_association():
    from handlers.line_handler import LineHandler
    handler = LineHandler()
    img = np.zeros((100, 100, 3))
    
    # Needs to mock the extractor
    with patch('extractors.line_extractor.LineExtractor.extract') as mock_ext:
        mock_ext.return_value = {
            'data_points': [
                {'xyxy': [10, 10, 20, 20], 'estimated_value': 5.0, 'x_calibrated': 1, 'y_calibrated': 5, 'x_pixel': 15, 'y_pixel': 15}
            ]
        }
        
        detections = {
            'error_bar': [{'xyxy': [14, 5, 16, 25]}] # Center X is 15
        }
        
        calib = {'primary': MagicMock()}
        calib['primary'].func = lambda x: x
        
        base = MagicMock()
        base.baselines = [MagicMock(axis_id='y', value=90)]
        
        result = handler.extract_values(img, detections, calib, base, 'vertical')
        assert len(result) == 1
        assert 'error_bar' in result[0]
        assert result[0]['error_bar']['bbox'] == [14, 5, 16, 25]

def test_issue_h_x_order_tabulation():
    result = {
        "chart_type": "bar",
        "bars": [
            {'xyxy': [50, 10, 60, 20], 'value': 2}, # second
            {'xyxy': [10, 10, 20, 20], 'value': 1}, # first
        ]
    }
    rows = _build_rows(result, "bar")
    assert rows[0].values['value'] == 1.0
    assert rows[1].values['value'] == 2.0
    
def test_issue_b_safe_float_non_ascii():
    assert _safe_float("१२३") is not None # Hindi 123
    assert _safe_float("123.45") == 123.45
    assert _safe_float(None) is None
