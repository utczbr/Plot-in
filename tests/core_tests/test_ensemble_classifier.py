# -*- coding: utf-8 -*-
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from core.ensemble_classifier import WeightedChartClassifier, compute_box_iou


def test_compute_box_iou():
    box1 = (0, 0, 100, 100)
    box2 = (0, 0, 100, 100)
    assert compute_box_iou(box1, box2) == 1.0

    box3 = (50, 50, 150, 150)
    iou = compute_box_iou(box1, box3)
    assert 0.14 < iou < 0.15

    box4 = (200, 200, 300, 300)
    assert compute_box_iou(box1, box4) == 0.0


def test_ensemble_classifier_single_model():
    mock_models_manager = MagicMock()
    mock_cls_model = MagicMock()
    mock_models_manager.get_model.side_effect = lambda name: mock_cls_model if name == 'classification' else None

    with patch("utils.inference.run_inference_on_image") as mock_infer:
        mock_infer.return_value = [
            {'cls': 1, 'conf': 0.85},  # Class 1 -> bar
        ]

        ensemble = WeightedChartClassifier(mock_models_manager, infer_func=mock_infer)
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        types = ensemble.classify_image(img)
        assert 'bar' in types


def test_ensemble_classifier_low_confidence_unknown():
    mock_models_manager = MagicMock()
    mock_cls_model = MagicMock()
    mock_models_manager.get_model.side_effect = lambda name: mock_cls_model if name == 'classification' else None

    with patch("utils.inference.run_inference_on_image") as mock_infer:
        mock_infer.return_value = [
            {'cls': 1, 'conf': 0.20},  # Low confidence
        ]

        ensemble = WeightedChartClassifier(mock_models_manager, infer_func=mock_infer)
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        types = ensemble.classify_image(img, advanced_settings={'fusion_threshold': 0.50})
        assert types == ['unknown']


def test_ensemble_classifier_weight_renormalization():
    mock_models_manager = MagicMock()
    mock_cls_model = MagicMock()
    mock_det_model = MagicMock()
    mock_models_manager.get_model.side_effect = lambda name: mock_cls_model if name == 'classification' else (mock_det_model if name == 'chart_detector' else None)

    with patch("utils.inference.run_inference_on_image") as mock_infer:
        # First call is for classifier, second call for chart_detector
        mock_infer.side_effect = [
            [{'cls': 5, 'conf': 0.90}],  # Class 5 -> line
            [{'cls': 5, 'conf': 0.80}],  # Class 5 -> line
        ]

        ensemble = WeightedChartClassifier(mock_models_manager, infer_func=mock_infer)
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        types = ensemble.classify_image(img)
        assert types[0] == 'line'


def test_ensemble_classifier_uses_only_primary_crop_when_available():
    WeightedChartClassifier.clear_cache()
    mock_models_manager = MagicMock()
    mock_cls_model = MagicMock()
    mock_det_model = MagicMock()
    mock_models_manager._models = {'chart_detector': mock_det_model, 'classification': mock_cls_model}
    mock_models_manager.get_model.side_effect = lambda name: mock_cls_model if name == 'classification' else (mock_det_model if name == 'chart_detector' else None)

    with patch("utils.inference.run_inference_on_image") as mock_infer:
        # 1st call: chart detector returns a valid primary crop box [20, 20, 100, 100] (80x80)
        # 2nd call: classifier should run ONLY on that 80x80 crop (no full-image pass!)
        mock_infer.side_effect = [
            [{'cls': 5, 'conf': 0.90, 'xyxy': [20, 20, 100, 100]}],
            [{'cls': 5, 'conf': 0.85}],
        ]

        ensemble = WeightedChartClassifier(mock_models_manager, infer_func=mock_infer)
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        types, conf = ensemble.classify_image_with_conf(img)
        assert types[0] == 'line'
        # Exactly 2 inferences should occur: 1 for detector, 1 for crop classification
        assert mock_infer.call_count == 2

        # Verify the 2nd call was passed the primary crop (80x80), not the full image (200x200)
        crop_arg = mock_infer.call_args_list[1][0][1]
        assert crop_arg.shape == (80, 80, 3)


def test_ensemble_classifier_cache_hits_on_repeat(tmp_path):
    WeightedChartClassifier.clear_cache()
    mock_models_manager = MagicMock()
    mock_cls_model = MagicMock()
    mock_models_manager.get_model.side_effect = lambda name: mock_cls_model if name == 'classification' else None

    img_file = tmp_path / "chart.png"
    img_file.touch()

    with patch("utils.inference.run_inference_on_image") as mock_infer:
        mock_infer.return_value = [{'cls': 1, 'conf': 0.95}]

        ensemble = WeightedChartClassifier(mock_models_manager, infer_func=mock_infer)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # First call: runs inference
        types1, conf1 = ensemble.classify_image_with_conf(img, image_path=img_file)
        assert mock_infer.call_count == 1

        # Second call with same image_path: must return cached result without running inference
        types2, conf2 = ensemble.classify_image_with_conf(img, image_path=img_file)
        assert mock_infer.call_count == 1  # Unchanged!
        assert types1 == types2
        assert conf1 == conf2


def test_ensemble_classifier_early_exit_bypasses_doclayout():
    WeightedChartClassifier.clear_cache()
    mock_models_manager = MagicMock()
    mock_cls_model = MagicMock()
    mock_doc_model = MagicMock()
    mock_models_manager._models = {'classification': mock_cls_model, 'doclayout': mock_doc_model}
    mock_models_manager.get_model.side_effect = lambda name: mock_cls_model if name == 'classification' else (mock_doc_model if name == 'doclayout' else None)

    # 1. High confidence case (0.95 >= 0.80): doclayout must be bypassed
    with patch("utils.inference.run_inference_on_image") as mock_infer:
        mock_infer.return_value = [{'cls': 1, 'conf': 0.95}]  # Class 1 -> bar
        ensemble = WeightedChartClassifier(mock_models_manager, infer_func=mock_infer)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        types, conf = ensemble.classify_image_with_conf(img)
        assert types == ['bar']
        assert mock_infer.call_count == 1  # doclayout was bypassed!

    # 2. Ambiguous confidence case (0.50 < 0.80): doclayout must be evaluated
    with patch("utils.inference.run_inference_on_image") as mock_infer:
        mock_infer.side_effect = [
            [{'cls': 1, 'conf': 0.50}],  # Classifier returns ambiguous score
            [{'cls': 3, 'bbox': (0, 0, 100, 100), 'conf': 0.90}],  # doclayout returns figure region
        ]
        ensemble = WeightedChartClassifier(mock_models_manager, infer_func=mock_infer)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        types, conf = ensemble.classify_image_with_conf(img)
        assert mock_infer.call_count == 2  # doclayout was evaluated!


