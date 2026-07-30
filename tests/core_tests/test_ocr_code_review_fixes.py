"""
Unit tests targeting all bug fixes and improvements identified in ocr_code_review.md
"""

import numpy as np
import pytest
from src.ocr.orchestrator.unified_ocr_system_v2 import UnifiedOCRSystemV2, OCRConfig, OCREngine, QualityMode
from src.ocr.engines.parallel_ocr_engine import ParallelOCREngine, OCRTask
from src.ocr.preprocessing.ocr_validator import OCRValidator as ContextOCRValidator
from src.ocr.preprocessing.ocr_validation import OCRValidator as NumericOCRValidator
from src.ocr.runtime.cache_runtime import ZeroCopyHashCache, HashDeduplicator
from src.ocr.preprocessing.preprocessing_base import PaddleOCRPreprocessing
from src.ocr.engines.ocr_engine_base import PaddleOCRBaseEngine


class DummyEasyOCREngine:
    def recognize(self, image: np.ndarray, context: str = "default"):
        return ("123.45", 0.95)


def test_a2_accurate_mode_unpacking():
    """Verify QualityMode.ACCURATE does not raise ValueError due to 4-into-3 unpacking."""
    cfg = OCRConfig(engine=OCREngine.EASYOCR, quality_mode=QualityMode.ACCURATE, max_workers=1)
    system = UnifiedOCRSystemV2(cfg, engine_instance=DummyEasyOCREngine())
    # Mock ocr_engine to return text, conf
    system.ocr_engine = DummyEasyOCREngine()

    crop = np.zeros((30, 80, 3), dtype=np.uint8)
    results = system.process_batch([(crop, "scale")])

    assert len(results) == 1
    assert results[0].text == "123.45"
    assert results[0].confidence == 0.95


def test_a3_parallel_ocr_engine_batch_ordering():
    """Verify ParallelOCREngine maintains correct crop-result ordering when batch size is non-multiple."""
    class DummyParallelEngine(ParallelOCREngine):
        def _process_single_vectorized(self, task: OCRTask):
            # Return text equal to task's detection_id
            return OCRTask(task.detection_id, task.crop, task.bbox, task.context, task.whitelist)

    engine = DummyParallelEngine(reader=None, batch_size=32, max_workers=2)
    crops = [np.full((20, 20), i, dtype=np.uint8) for i in range(50)]
    bboxes = [(0, 0, 20, 20)] * 50
    contexts = ["default"] * 50

    results = engine.process_batch(crops, bboxes, contexts)

    assert len(results) == 50
    # Every task's result at index i must have detection_id == i
    for i, res in enumerate(results):
        assert res.detection_id == i


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


def test_d1_aspect_ratio_preservation():
    """Verify narrow tick crop aspect ratio is preserved in PaddleOCRPreprocessing."""
    preproc = PaddleOCRPreprocessing()
    narrow_crop = np.zeros((18, 8), dtype=np.uint8)

    processed = preproc.preprocess_for_speed(narrow_crop)
    h, w = processed.shape[:2]

    # Aspect ratio h/w should remain close to 18/8 = 2.25, not 36/100 = 0.36
    ratio_orig = 18.0 / 8.0
    ratio_new = float(h) / float(w)
    assert abs(ratio_orig - ratio_new) < 0.5


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


def test_a1_paddle_base_engine_stub_error():
    """Verify PaddleOCRBaseEngine raises explicit NotImplementedError when unconfigured instead of returning fake text."""
    engine = PaddleOCRBaseEngine(det_session=None, rec_session=None, character_dict=[])
    with pytest.raises(NotImplementedError):
        engine.recognize(np.zeros((30, 80, 3), dtype=np.uint8))
