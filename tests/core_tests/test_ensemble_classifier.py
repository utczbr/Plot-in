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
