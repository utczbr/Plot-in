
import sys
import os
import types

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock onnxruntime to avoid import error
ort_mock = types.ModuleType('onnxruntime')
ort_mock.InferenceSession = type('InferenceSession', (), {})
sys.modules['onnxruntime'] = ort_mock
sys.modules['cv2'] = types.ModuleType('cv2')

# Mock pipelines to avoid circular import in core
pipelines_mock = types.ModuleType('pipelines')
chart_pipeline_mock = types.ModuleType('chart_pipeline')
chart_pipeline_mock.ChartAnalysisPipeline = type('ChartAnalysisPipeline', (), {})
sys.modules['pipelines'] = pipelines_mock
sys.modules['pipelines.chart_pipeline'] = chart_pipeline_mock

import logging
from src.core.classifiers.histogram_chart_classifier import HistogramChartClassifier
from src.services.orientation_service import Orientation

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_classifier_defaults():
    logger.info("Verifying default parameters...")
    classifier = HistogramChartClassifier()
    defaults = classifier.params
    
    assert defaults['scale_size_max_width'] == 0.09, "scale_size_max_width should be 0.09"
    assert defaults['classification_threshold'] == 2.0, "classification_threshold should be 2.0"
    assert defaults.get('gaussian_sigma') == 0.11, "gaussian_sigma should be 0.11"
    assert defaults['edge_threshold'] == 0.22, "edge_threshold should be 0.22"
    
    logger.info("✅ All default parameters match expected new values.")

def verify_classification_logic():
    logger.info("Verifying classification logic with mock data...")
    classifier = HistogramChartClassifier()
    
    img_width = 1000
    img_height = 1000
    
    # Mock label at 21% from left (x=210)
    # With new threshold (0.22), it should be scored as 'scale_label'.
    axis_labels = [{
        'xyxy': [200, 500, 220, 520],
        'text': '10',
        'confidence': 0.99
    }]
    
    # Mock bins
    chart_elements = [{
        'xyxy': [250, 500, 300, 900], 
    }]
    
    result = classifier.classify(
        axis_labels=axis_labels,
        chart_elements=chart_elements,
        img_width=img_width,
        img_height=img_height,
        orientation=Orientation.VERTICAL
    )
    
    logger.info(f"Result for numeric label at 21% width: {result.scale_labels}")
    
    # Check if result is empty (it shouldn't be)
    found = False
    for label in result.scale_labels:
        if label.get('text') == '10':
            found = True
            break
            
    if found:
         logger.info("✅ Label correctly classified as scale_label.")
    else:
         logger.warning(f"⚠️ Label NOT classified as scale_label. Result: {result}")
         # Raise error to fail verification if strictly needed, but warning is info enough for now.
         raise AssertionError("Label missed by classifier")

if __name__ == "__main__":
    try:
        verify_classifier_defaults()
        verify_classification_logic()
        print("\nVerification Passed!")
    except AssertionError as e:
        print(f"\nVerification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
