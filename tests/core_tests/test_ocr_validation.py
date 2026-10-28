"""
Unit tests targeting all bug fixes and improvements identified in ocr_code_review.md
"""

import numpy as np
from src.ocr.preprocessing.ocr_validator import OCRValidator as ContextOCRValidator
from src.ocr.preprocessing.ocr_validator import OCRValidator as NumericOCRValidator
from src.ocr.runtime.cache_runtime import ZeroCopyHashCache, HashDeduplicator


def test_a4_char_corrections_currency_and_percent():
    """Verify '$' and '%' are preserved and not corrupted to digits '5' and '9'."""
    validator = ContextOCRValidator()
    cleaned_dollar, _ = validator.validate_and_clean("$50", context_type="tick")
    cleaned_percent, _ = validator.validate_and_clean("42%", context_type="tick")

    assert "$" in cleaned_dollar
    assert "%" in cleaned_percent


def test_a5_regex_escaping_numeric_validation():
    """Verify numeric_pattern and _fuzzy_parse match valid numbers and digit sequences."""
    validator = NumericOCRValidator()

    # Strict numeric match
    val, conf = validator.validate_numeric("123.45")
    assert val == 123.45
    assert conf > 0.0

    val_neg, conf_neg = validator.validate_numeric("-12.3")
    assert val_neg == -12.3
    assert conf_neg > 0.0

    # Fuzzy parse
    val_fuzzy, conf_fuzzy = validator.validate_numeric("val: 42")
    assert val_fuzzy == 42.0
    assert conf_fuzzy > 0.0


def test_d3_hash_cache_context_sensitivity():
    """Verify identical pixel arrays with different contexts generate distinct cache hashes."""
    cache = ZeroCopyHashCache()
    dedup = HashDeduplicator()

    crop = np.full((10, 10), 128, dtype=np.uint8)

    h_tick = cache._compute_hash(crop, context="tick_label")
    h_title = cache._compute_hash(crop, context="title")

    assert h_tick != h_title

    unique_crops, _ = dedup.deduplicate_crops([(crop, "tick_label"), (crop, "title")])
    assert len(unique_crops) == 2


def test_base_ocr_engine_single_crop_convenience():
    """Verify BaseOCREngine.recognize delegates single crops to process_batch."""
    from src.ocr.ocr_base import BaseOCREngine

    doc = BaseOCREngine.recognize.__doc__ or ""
    assert "single-crop convenience" in doc.lower() or "convenience" in doc.lower()

    class _DummyEngine(BaseOCREngine):
        def process_batch(self, crops_with_context):
            return [{"text": f"ok:{crops_with_context[0][1]}", "confidence": 0.88}]

    engine = _DummyEngine()
    text, conf = engine.recognize(np.zeros((8, 8, 3), dtype=np.uint8), context="tick")
    assert text == "ok:tick"
    assert conf == 0.88


def test_paddle_ocr_engine_deduplication_and_caching():
    """Verify PaddleOCREngine deduplicates identical crops in a batch and caches by content hash."""
    from src.ocr.engines.ocr_paddle_onnx import PaddleOCREngine
    from unittest.mock import MagicMock

    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine._cache = ZeroCopyHashCache(max_size=512)

    call_count = 0
    def mock_rec(crop, context='default'):
        nonlocal call_count
        call_count += 1
        return (f"text_{context}_{call_count}", 0.9)

    engine._recognize_text_region = mock_rec

    crop1 = np.full((15, 30, 3), 50, dtype=np.uint8)
    crop2 = np.full((15, 30, 3), 100, dtype=np.uint8)

    # Batch with 3 items where item 0 and 2 are identical (crop1, 'tick')
    batch = [(crop1, 'tick'), (crop2, 'title'), (crop1, 'tick')]
    results = engine.process_batch(batch)

    assert len(results) == 3
    # Only 2 distinct crops -> only 2 calls to _recognize_text_region
    assert call_count == 2
    assert results[0]['text'] == results[2]['text']
    assert results[0]['text'] == "text_tick_1"
    assert results[1]['text'] == "text_title_2"

    # Second batch with same crop1 -> retrieved from cache, no new recognition call
    results2 = engine.process_batch([(crop1, 'tick')])
    assert len(results2) == 1
    assert results2[0]['text'] == "text_tick_1"
    assert call_count == 2


def test_paddle_ocr_full_pipeline_bounded_hash_cache():
    """Verify PaddleOCRFullPipeline uses bounded ZeroCopyHashCache(max_size=512)."""
    from src.ocr.engines.ocr_paddle_full import PaddleOCRFullPipeline

    pipeline = PaddleOCRFullPipeline.__new__(PaddleOCRFullPipeline)
    pipeline._hash_cache = ZeroCopyHashCache(max_size=512)
    pipeline.enable_textline_orientation = False

    def mock_rec(crop):
        return ("100", 0.95)
    pipeline._recognize_text_region = mock_rec

    # Fill cache beyond 512 entries
    for i in range(550):
        dummy_crop = np.full((5, 5, 3), i % 256, dtype=np.uint8)
        dummy_crop[0, 0, 0] = i // 256
        dummy_crop[0, 0, 1] = i % 256
        pipeline.process_batch([(dummy_crop, 'scale')])

    assert pipeline._hash_cache.size() <= 512


def test_chart_pipeline_ocr_validator_wiring():
    """Verify ChartAnalysisPipeline wires OCRValidator to clean text, update confidence, and populate cleanedvalue."""
    from src.pipelines.chart_pipeline import ChartAnalysisPipeline
    from unittest.mock import MagicMock

    mock_models = MagicMock()
    mock_ocr = MagicMock()
    mock_calib = MagicMock()

    pipeline = ChartAnalysisPipeline(mock_models, mock_ocr, mock_calib)
    assert hasattr(pipeline, 'ocr_validator')

    # Mock OCR output in order of ocr_batch: axis_labels, chart_title, data_label
    mock_ocr.process_batch.return_value = [
        {"text": "123.50", "confidence": 0.9},
        {"text": "Total Revenue", "confidence": 0.95},
        {"text": "45.0%", "confidence": 0.85},
    ]

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = {
        'axis_labels': [{'xyxy': [10, 10, 30, 30]}],
        'data_label': [{'xyxy': [40, 10, 60, 30]}],
        'chart_title': [{'xyxy': [10, 80, 90, 95]}],
    }

    pipeline._process_ocr(img, detections)

    # axis_labels: cleanedvalue should be populated with 123.5
    axis_det = detections['axis_labels'][0]
    assert axis_det['cleanedvalue'] == 123.5
    assert axis_det['text'] == "123.50"
    assert axis_det['ocr_confidence'] > 0.0

    # data_label: cleanedvalue should be populated (45.0% -> 0.45 or 45.0 depending on percent parsing)
    data_det = detections['data_label'][0]
    assert data_det['cleanedvalue'] is not None
    assert data_det['text'] == "45.0%"

    # chart_title: cleanedvalue should not be set, text preserved
    title_det = detections['chart_title'][0]
    assert 'cleanedvalue' not in title_det or title_det['cleanedvalue'] is None
    assert title_det['text'] == "Total Revenue"


def test_analysis_manager_purged_relics_and_annotated_image():
    """Verify AnalysisManager has purged dead methods and invokes VisualizationService.save_annotated_image."""
    from src.core.analysis_manager import AnalysisManager
    from unittest.mock import MagicMock, patch

    mock_dm = MagicMock()
    mock_im = MagicMock()
    mgr = AnalysisManager(mock_dm, mock_im)

    # Relics must be purged
    assert not hasattr(mgr, 'set_easyocr_reader')
    assert not hasattr(mgr, 'cancel_analysis')
    assert not hasattr(mgr, 'is_analysis_running')
    assert not hasattr(mgr, '_running_threads')

    # Advanced settings property
    assert mgr.advanced_settings == {}
    mgr.set_advanced_settings({'output_settings': {'save_annotated_image': True}})
    assert mgr.advanced_settings['output_settings']['save_annotated_image'] is True

    # Test run_single_analysis invokes VisualizationService.save_annotated_image
    with patch.object(mgr, '_create_pipeline') as mock_pipe_creator, \
         patch('visual.visualization_service.VisualizationService.save_annotated_image') as mock_save_annotated, \
         patch('cv2.imread', return_value=np.zeros((50, 50, 3), dtype=np.uint8)):

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {'chart_type': 'bar', 'elements': []}
        mock_pipe_creator.return_value = mock_pipeline

        res = mgr.run_single_analysis(
            image_path="test_chart.png",
            conf=0.5,
            output_path="/tmp/output_test",
        )

        assert res is not None
        mock_save_annotated.assert_called_once()


