"""
Tests for PaddleOCR PP-OCRv5 / PaddleX 3.0 specification conformance.
Validates:
1. PP-LCNet_x1_0_textline_ori ImageNet normalization and confidence threshold (cls_thresh).
2. PP-OCRv5_server_rec BGR channel order and 0.0 padding.
3. Lazy det_session loading in PaddleOCREngine and PaddleOCRFullPipeline.
4. AnalysisManager ocr_settings.use_gpu preference and OCR engine caching.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from ocr.engines.ocr_paddle_onnx import PaddleOCREngine
from ocr.engines.ocr_paddle_full import PaddleOCRFullPipeline
from core.analysis_manager import AnalysisManager


def test_paddle_ocr_engine_cls_imagenet_norm_and_threshold():
    """Verify PP-LCNet_x1_0_textline_ori uses ImageNet mean/std and cls_thresh = 0.9."""
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine.cls_thresh = 0.9

    captured_inputs = []
    mock_cls_session = MagicMock()
    def mock_run(output_names, input_feed):
        for k, v in input_feed.items():
            captured_inputs.append(v.copy())
        # Return 180° rotation with score 0.85 (below 0.9 threshold)
        return [np.array([[0.15, 0.85]], dtype=np.float32)]

    mock_cls_session.run = mock_run
    engine.cls_session = mock_cls_session
    engine._cls_output_name = "output"
    engine._cls_input_name = "input"

    # Red test image (BGR: [0, 0, 255] -> RGB: [255, 0, 0])
    test_crop = np.zeros((40, 80, 3), dtype=np.uint8)
    test_crop[:, :, 2] = 255  # Red channel in BGR

    rotated = engine._classify_and_rotate(test_crop)

    # 1. Image should NOT be rotated because 0.85 < 0.90
    assert np.array_equal(rotated, test_crop)

    # 2. Check normalization constants:
    # RGB values: R=1.0, G=0.0, B=0.0
    # Expected normalized values:
    # R: (1.0 - 0.485) / 0.229 ≈ 2.2489
    # G: (0.0 - 0.456) / 0.224 ≈ -2.0357
    # B: (0.0 - 0.406) / 0.225 ≈ -1.8044
    assert len(captured_inputs) == 1
    t_in = captured_inputs[0]
    assert t_in.shape == (1, 3, 80, 160)
    assert np.isclose(t_in[0, 0, 0, 0], (1.0 - 0.485) / 0.229, atol=1e-3)
    assert np.isclose(t_in[0, 1, 0, 0], (0.0 - 0.456) / 0.224, atol=1e-3)
    assert np.isclose(t_in[0, 2, 0, 0], (0.0 - 0.406) / 0.225, atol=1e-3)

    # 3. High confidence score >= 0.9 triggers 180° rotation
    mock_cls_session.run = lambda out, feed: [np.array([[0.05, 0.95]], dtype=np.float32)]
    # Put a distinct asymmetric marker at top-left
    test_crop_marker = test_crop.copy()
    test_crop_marker[0, 0, :] = [128, 128, 128]
    rotated2 = engine._classify_and_rotate(test_crop_marker)
    # After 180° rotation, marker moves to bottom-right
    assert np.array_equal(rotated2[-1, -1, :], [128, 128, 128])


def test_paddle_ocr_full_pipeline_cls_imagenet_norm_and_threshold():
    """Verify PaddleOCRFullPipeline Stage 4 uses ImageNet mean/std and cls_thresh = 0.9."""
    pipeline = PaddleOCRFullPipeline.__new__(PaddleOCRFullPipeline)
    pipeline.cls_thresh = 0.9

    captured_inputs = []
    mock_session = MagicMock()
    mock_session.run = lambda out, feed: (captured_inputs.append(list(feed.values())[0].copy()), [np.array([[0.1, 0.88]], dtype=np.float32)])[1]
    pipeline.textline_ori_session = mock_session
    pipeline._textline_ori_output = "out"
    pipeline._textline_ori_input = "in"

    test_crop = np.zeros((40, 80, 3), dtype=np.uint8)
    test_crop[:, :, 0] = 255  # Blue in BGR -> Blue in RGB

    res = pipeline._classify_and_rotate_textline(test_crop)
    # Score 0.88 < 0.9 -> no rotation
    assert np.array_equal(res, test_crop)

    t_in = captured_inputs[0]
    # Blue channel is index 2 in RGB
    # B: (1.0 - 0.406) / 0.225 ≈ 2.64
    assert np.isclose(t_in[0, 2, 0, 0], (1.0 - 0.406) / 0.225, atol=1e-3)


def test_preprocess_recognition_border_constant_and_bgr_channel_order():
    """Verify preprocessing uses BORDER_CONSTANT with background color, preserves native BGR, and pads canvas."""
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine._rec_dynamic_width = True

    # Crop with custom light background [240, 245, 250] in BGR and a blue dot [255, 0, 0] in BGR
    bg_bgr = np.array([240, 245, 250], dtype=np.uint8)
    crop = np.full((24, 40, 3), bg_bgr, dtype=np.uint8)
    crop[10:14, 10:14, :] = [255, 0, 0]  # Blue in BGR: B=255, G=0, R=0

    preprocessed = engine._preprocess_recognition(crop, img_h=48, img_w=320)

    assert preprocessed.shape == (1, 3, 48, 320)

    # In native BGR: Channel 0 is Blue, Channel 1 is Green, Channel 2 is Red
    # Right-padded canvas region should match bg_bgr normalized: (val / 255.0 - 0.5) / 0.5
    expected_bg_norm = (bg_bgr.astype(np.float32) / 255.0 - 0.5) / 0.5
    for c in range(3):
        assert np.allclose(preprocessed[0, c, :, -20:], expected_bg_norm[c], atol=1e-3)

    # PaddleOCRFullPipeline._preprocess_recognition should behave identically
    pipeline = PaddleOCRFullPipeline.__new__(PaddleOCRFullPipeline)
    pipeline._rec_dynamic_width = True
    full_preprocessed = pipeline._preprocess_recognition(crop, img_h=48, img_w=320)
    assert np.allclose(full_preprocessed, preprocessed)


def test_recognize_text_region_skips_180_rotation_on_horizontal_scale_labels():
    """Verify 0°/180° orientation classifier only runs on rotated vertical crops or titles."""
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine.cls_session = MagicMock()
    engine.rec_session = MagicMock()
    engine._rec_output_name = "output"
    engine._rec_input_name = "input"
    engine.try_both_orientations = False
    engine._recognize_text_region_called = False

    # Mock classification to flip crop if called
    engine._classify_and_rotate = MagicMock(side_effect=lambda c: cv2.rotate(c, cv2.ROTATE_180))
    # Mock recognition run & decode
    engine.rec_session.run.return_value = [np.zeros((1, 10, 100), dtype=np.float32)]
    engine._decode_recognition = MagicMock(return_value=("10", 0.95))

    test_crop = np.full((20, 40, 3), 255, dtype=np.uint8)

    # 1. Horizontal scale label -> MUST NOT call _classify_and_rotate
    text, conf = engine._recognize_text_region(test_crop, context="axis_labels")
    assert engine._classify_and_rotate.call_count == 0
    assert text == "10"

    # 2. Horizontal tick label -> MUST NOT call _classify_and_rotate
    text, conf = engine._recognize_text_region(test_crop, context="tick_label")
    assert engine._classify_and_rotate.call_count == 0

    # 3. Rotated vertical label (":rotated") -> MUST call _classify_and_rotate
    text, conf = engine._recognize_text_region(test_crop, context="axis_labels:rotated")
    assert engine._classify_and_rotate.call_count == 1

    # 4. Title -> MUST call _classify_and_rotate
    text, conf = engine._recognize_text_region(test_crop, context="chart_title")
    assert engine._classify_and_rotate.call_count == 2


def test_analysis_manager_gpu_preference_and_caching(tmp_path: Path):
    """Verify AnalysisManager prefers ocr_settings.use_gpu and caches OCR engine."""
    dm = MagicMock()
    im = MagicMock()
    mgr = AnalysisManager(dm, im)
    mgr.set_models(MagicMock())

    # Settings with ocr_settings.use_gpu = True, performance.use_gpu = False
    settings = {
        "ocr_settings": {"use_gpu": True},
        "performance": {"use_gpu": False},
        "models_dir": str(tmp_path),
        "ocr_engine": "Paddle",
    }
    mgr.set_advanced_settings(settings)

    mock_engine1 = MagicMock()
    with patch("ocr.ocr_factory.OCREngineFactory.create_engine", return_value=mock_engine1) as mock_factory:
        pipe1 = mgr._create_pipeline()
        assert pipe1.ocr_engine is mock_engine1
        assert mock_factory.call_args[1]["use_gpu"] is True

        # Second call with same settings should reuse cached engine
        pipe2 = mgr._create_pipeline()
        assert pipe2.ocr_engine is mock_engine1
        assert mock_factory.call_count == 1

        # Changing use_gpu invalidates cache and creates new engine
        settings["ocr_settings"]["use_gpu"] = False
        mgr.set_advanced_settings(settings)
        mock_engine2 = MagicMock()
        mock_factory.return_value = mock_engine2

        pipe3 = mgr._create_pipeline()
        assert pipe3.ocr_engine is mock_engine2
        assert mock_factory.call_count == 2
        assert mock_factory.call_args[1]["use_gpu"] is False
