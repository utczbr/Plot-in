import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.inference import run_inference_on_image


class _DummyInput:
    name = "images"


class _DummySession:
    def __init__(self, output):
        self._output = output

    def get_inputs(self):
        return [_DummyInput()]

    def run(self, *_args, **_kwargs):
        return [self._output]


def _to_raw_output(detections):
    """Build raw ONNX output tensor with shape (1, F, N)."""
    arr = np.array(detections, dtype=np.float32)
    return arr.T[None, :, :]


def _identity_preprocess(img, new_shape=None, color=None):
    return img, 1.0, (0.0, 0.0)


def test_pose_output_uses_conf_not_keypoint_values(monkeypatch):
    monkeypatch.setattr("utils.inference.preprocess_with_letterbox", _identity_preprocess)
    monkeypatch.setattr("utils.inference.cv2.dnn.NMSBoxes", lambda *args, **kwargs: np.array([[0]]))

    high_conf = [
        100, 100, 20, 20, 0.90,
        10, 10, 0.9, 12, 12, 0.9, 14, 14, 0.9, 16, 16, 0.9, 18, 18, 0.9,
    ]
    low_conf_high_kpts = [
        200, 200, 20, 20, 0.10,
        999, 999, 0.9, 999, 999, 0.9, 999, 999, 0.9, 999, 999, 0.9, 999, 999, 0.9,
    ]

    session = _DummySession(_to_raw_output([high_conf, low_conf_high_kpts]))
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = run_inference_on_image(
        session,
        img,
        conf_threshold=0.5,
        class_map={0: "slice"},
        model_output_type="pose",
        expected_keypoints=5,
    )

    assert len(detections) == 1
    assert detections[0]["conf"] == pytest.approx(0.90)


def test_pose_output_parses_keypoints_and_returns_cls0(monkeypatch):
    monkeypatch.setattr("utils.inference.preprocess_with_letterbox", _identity_preprocess)
    monkeypatch.setattr("utils.inference.cv2.dnn.NMSBoxes", lambda *args, **kwargs: np.array([[0]]))

    det = [
        40, 60, 20, 10, 0.85,
        30, 55, 0.7, 35, 56, 0.8, 40, 57, 0.9, 45, 58, 1.0, 50, 59, 0.95,
    ]
    session = _DummySession(_to_raw_output([det]))
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    detections = run_inference_on_image(
        session,
        img,
        conf_threshold=0.5,
        class_map={0: "slice"},
        model_output_type="pose",
        expected_keypoints=5,
    )

    assert len(detections) == 1
    parsed = detections[0]
    assert parsed["cls"] == 0
    assert "keypoints" in parsed
    assert len(parsed["keypoints"]) == 5
    assert parsed["keypoints"][0][0] == pytest.approx(30.0)
    assert parsed["keypoints"][4][1] == pytest.approx(59.0)


def test_pose_output_applies_nms(monkeypatch):
    monkeypatch.setattr("utils.inference.preprocess_with_letterbox", _identity_preprocess)
    monkeypatch.setattr("utils.inference.cv2.dnn.NMSBoxes", lambda *args, **kwargs: np.array([[1]]))

    det_a = [
        20, 20, 12, 12, 0.95,
        18, 18, 0.9, 19, 17, 0.9, 21, 16, 0.9, 22, 17, 0.9, 23, 18, 0.9,
    ]
    det_b = [
        50, 60, 20, 10, 0.80,
        45, 55, 0.9, 47, 56, 0.9, 50, 57, 0.9, 53, 58, 0.9, 55, 59, 0.9,
    ]
    session = _DummySession(_to_raw_output([det_a, det_b]))
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    detections = run_inference_on_image(
        session,
        img,
        conf_threshold=0.5,
        class_map={0: "slice"},
        model_output_type="pose",
        expected_keypoints=5,
    )

    assert len(detections) == 1
    assert detections[0]["conf"] == pytest.approx(0.80)
    assert detections[0]["xyxy"] == [40, 55, 60, 65]


def test_bbox_path_unchanged_for_standard_models(monkeypatch):
    monkeypatch.setattr("utils.inference.preprocess_with_letterbox", _identity_preprocess)
    monkeypatch.setattr("utils.inference.cv2.dnn.NMSBoxes", lambda *args, **kwargs: np.array([[0]]))

    det = [10, 10, 4, 4, 0.1, 0.8, 0.2]  # argmax class=1 with score=0.8
    session = _DummySession(_to_raw_output([det]))
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    detections = run_inference_on_image(
        session,
        img,
        conf_threshold=0.5,
        class_map={0: "chart", 1: "bar", 2: "axis_title"},
        model_output_type="bbox",
    )

    assert len(detections) == 1
    assert detections[0]["cls"] == 1
    assert detections[0]["conf"] == pytest.approx(0.8)
    assert "keypoints" not in detections[0]
