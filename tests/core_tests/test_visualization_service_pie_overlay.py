import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from visual.visualization_service import VisualizationService


def _empty_analysis_with_detection(class_name, item):
    return {
        "chart_type": "pie" if class_name == "slice" else "bar",
        "detections": {class_name: [item]},
    }


def test_slice_with_valid_keypoints_draws_wedge_lines_not_bbox_corners(monkeypatch):
    calls = {"rectangle": 0, "line": 0, "polylines": 0}

    monkeypatch.setattr(
        "visual.visualization_service.cv2.rectangle",
        lambda *args, **kwargs: calls.__setitem__("rectangle", calls["rectangle"] + 1),
    )
    monkeypatch.setattr(
        "visual.visualization_service.cv2.line",
        lambda *args, **kwargs: calls.__setitem__("line", calls["line"] + 1),
    )
    monkeypatch.setattr(
        "visual.visualization_service.cv2.polylines",
        lambda *args, **kwargs: calls.__setitem__("polylines", calls["polylines"] + 1),
    )
    monkeypatch.setattr("visual.visualization_service.cv2.putText", lambda *args, **kwargs: None)

    det = {
        "xyxy": [10, 10, 30, 30],
        "keypoints": [
            [20.0, 20.0, 0.9],
            [30.0, 20.0, 0.9],
            [28.0, 24.0, 0.9],
            [24.0, 28.0, 0.9],
            [20.0, 30.0, 0.9],
        ],
    }
    analysis_data = _empty_analysis_with_detection("slice", det)

    image = np.zeros((60, 60, 3), dtype=np.uint8)
    VisualizationService.draw_results_on_image(image, analysis_data)

    assert calls["rectangle"] == 0
    assert calls["line"] == 2
    assert calls["polylines"] == 1


def test_slice_with_invalid_keypoints_falls_back_to_bbox(monkeypatch):
    calls = {"rectangle": 0, "line": 0, "polylines": 0}

    monkeypatch.setattr(
        "visual.visualization_service.cv2.rectangle",
        lambda *args, **kwargs: calls.__setitem__("rectangle", calls["rectangle"] + 1),
    )
    monkeypatch.setattr(
        "visual.visualization_service.cv2.line",
        lambda *args, **kwargs: calls.__setitem__("line", calls["line"] + 1),
    )
    monkeypatch.setattr(
        "visual.visualization_service.cv2.polylines",
        lambda *args, **kwargs: calls.__setitem__("polylines", calls["polylines"] + 1),
    )
    monkeypatch.setattr("visual.visualization_service.cv2.putText", lambda *args, **kwargs: None)

    det = {
        "xyxy": [10, 10, 30, 30],
        "keypoints": [[20.0, 20.0, 0.9], [30.0, 20.0, 0.9]],  # malformed contract
    }
    analysis_data = _empty_analysis_with_detection("slice", det)

    image = np.zeros((60, 60, 3), dtype=np.uint8)
    VisualizationService.draw_results_on_image(image, analysis_data)

    assert calls["rectangle"] == 1
    assert calls["line"] == 0
    assert calls["polylines"] == 0


def test_non_slice_classes_still_draw_bbox(monkeypatch):
    calls = {"rectangle": 0}

    monkeypatch.setattr(
        "visual.visualization_service.cv2.rectangle",
        lambda *args, **kwargs: calls.__setitem__("rectangle", calls["rectangle"] + 1),
    )
    monkeypatch.setattr("visual.visualization_service.cv2.putText", lambda *args, **kwargs: None)

    det = {"xyxy": [5, 6, 25, 26]}
    analysis_data = _empty_analysis_with_detection("bar", det)

    image = np.zeros((60, 60, 3), dtype=np.uint8)
    VisualizationService.draw_results_on_image(image, analysis_data)

    assert calls["rectangle"] == 1


def test_original_dims_scaling_applies_to_slice_overlay(monkeypatch):
    line_calls = []

    monkeypatch.setattr("visual.visualization_service.cv2.rectangle", lambda *args, **kwargs: None)
    monkeypatch.setattr("visual.visualization_service.cv2.polylines", lambda *args, **kwargs: None)
    monkeypatch.setattr("visual.visualization_service.cv2.putText", lambda *args, **kwargs: None)

    def _record_line(_img, pt1, pt2, _color, _thickness, *_args):
        line_calls.append((tuple(pt1), tuple(pt2)))

    monkeypatch.setattr("visual.visualization_service.cv2.line", _record_line)

    det = {
        "xyxy": [10, 10, 30, 30],
        "keypoints": [
            [10.0, 10.0, 0.9],  # center
            [20.0, 10.0, 0.9],  # start
            [20.0, 20.0, 0.9],
            [10.0, 20.0, 0.9],
            [5.0, 15.0, 0.9],   # end
        ],
    }
    analysis_data = _empty_analysis_with_detection("slice", det)

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    VisualizationService.draw_results_on_image(image, analysis_data, original_dims=(100, 100))

    # 2x scale from (50,50) to (100,100)
    assert ((20, 20), (40, 20)) in line_calls
    assert ((20, 20), (10, 30)) in line_calls
