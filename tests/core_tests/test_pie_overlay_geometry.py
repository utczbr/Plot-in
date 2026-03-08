import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from visual.pie_geometry import extract_slice_overlay_points


def test_extract_slice_overlay_points_valid_contract():
    det = {
        "keypoints": [
            [10.0, 20.0, 0.9],  # center
            [30.0, 20.0, 0.9],  # arc_start
            [28.0, 24.0, 0.9],  # arc_inter_1
            [24.0, 28.0, 0.9],  # arc_inter_2
            [20.0, 30.0, 0.9],  # arc_end
        ]
    }

    overlay = extract_slice_overlay_points(det)
    assert overlay is not None
    center, arc_points = overlay
    assert center == (10, 20)
    assert arc_points == [(30, 20), (28, 24), (24, 28), (20, 30)]


def test_extract_slice_overlay_points_applies_scaling():
    det = {
        "keypoints": [
            [10.0, 20.0, 0.9],
            [30.0, 20.0, 0.9],
            [28.0, 24.0, 0.9],
            [24.0, 28.0, 0.9],
            [20.0, 30.0, 0.9],
        ]
    }

    center, arc_points = extract_slice_overlay_points(det, scale_x=2.0, scale_y=0.5)
    assert center == (20, 10)
    assert arc_points == [(60, 10), (56, 12), (48, 14), (40, 15)]


def test_extract_slice_overlay_points_rejects_missing_keypoints():
    det = {"xyxy": [1, 2, 3, 4]}
    assert extract_slice_overlay_points(det) is None


def test_extract_slice_overlay_points_rejects_wrong_keypoint_count():
    det = {
        "keypoints": [
            [10.0, 20.0, 0.9],
            [30.0, 20.0, 0.9],
        ]
    }
    assert extract_slice_overlay_points(det) is None


def test_extract_slice_overlay_points_rejects_non_finite_values():
    det = {
        "keypoints": [
            [10.0, 20.0, 0.9],
            [30.0, 20.0, 0.9],
            [float("nan"), 24.0, 0.9],
            [24.0, float("inf"), 0.9],
            [20.0, 30.0, 0.9],
        ]
    }
    assert extract_slice_overlay_points(det) is None
