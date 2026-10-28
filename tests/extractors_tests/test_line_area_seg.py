"""
Unit and integration tests for segmentation-based line and area chart extraction.
Tests:
- Series linker (skeletonization, 2D PCA tangents, anchor-guided splprep)
- LineSegExtractor (anchor-guided extraction, X-tick snapping)
- AreaSegExtractor (topological sort, dynamic envelope subtraction, _trapz AUC)
- Segmentation postprocessing in inference.py
"""
import pytest
import numpy as np
import cv2

from extractors.series_linker import (
    skeletonize_mask,
    find_skeleton_junctions,
    estimate_endpoint_tangent_pca,
    fit_anchor_guided_curve,
    compute_cielab_color_distance,
)
from extractors.line_seg_extractor import LineSegExtractor
from extractors.area_seg_extractor import AreaSegExtractor, _trapz
from utils.inference import _postprocess_segmentation_output, _infer_model_output_type


def test_skeletonize_and_junctions():
    """Test skeletonization and junction node detection on synthetic crossing lines."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Draw two crossing lines (thick)
    cv2.line(mask, (10, 10), (90, 90), 1, thickness=5)
    cv2.line(mask, (10, 90), (90, 10), 1, thickness=5)

    skel = skeletonize_mask(mask)
    assert np.any(skel > 0)
    assert skel.shape == (100, 100)

    junctions, endpoints = find_skeleton_junctions(skel)
    # The crossing at center (~50, 50) should produce at least one junction point
    assert len(junctions) >= 1
    # 4 line ends should produce endpoints
    assert len(endpoints) >= 2


def test_pca_tangent_estimation():
    """Test Weighted 2D PCA tangent vector estimation."""
    # Horizontal line points
    points_h = np.column_stack([np.linspace(10, 50, 20), np.full(20, 30.0)])
    tangent_h = estimate_endpoint_tangent_pca(points_h, np.array([50.0, 30.0]))
    assert np.isclose(np.linalg.norm(tangent_h), 1.0)
    assert np.isclose(tangent_h[1], 0.0, atol=1e-2)

    # Vertical line points (which would cause Cartesian slope m -> inf)
    points_v = np.column_stack([np.full(20, 30.0), np.linspace(10, 50, 20)])
    tangent_v = estimate_endpoint_tangent_pca(points_v, np.array([30.0, 50.0]))
    assert np.isclose(np.linalg.norm(tangent_v), 1.0)
    assert np.isclose(tangent_v[0], 0.0, atol=1e-2)


def test_cielab_color_distance():
    """Test CIELAB Delta E94 color distance."""
    red_bgr = np.array([0, 0, 255], dtype=np.uint8)
    dark_red_bgr = np.array([0, 0, 200], dtype=np.uint8)
    blue_bgr = np.array([255, 0, 0], dtype=np.uint8)

    d_same = compute_cielab_color_distance(red_bgr, red_bgr)
    assert np.isclose(d_same, 0.0)

    d_close = compute_cielab_color_distance(red_bgr, dark_red_bgr)
    d_far = compute_cielab_color_distance(red_bgr, blue_bgr)
    assert d_close < d_far


def test_fit_anchor_guided_curve():
    """Test that anchor markers pull the spline through true peak coordinates."""
    # Sine wave skeleton
    x = np.linspace(10, 200, 50)
    y = 50 + 20 * np.sin(x / 20.0)
    skeleton_pts = np.column_stack([x, y])

    # True marker peak at (x=41.4, y=70.0)
    marker_pts = np.array([[41.4, 70.0], [104.2, 30.0]])

    fitted_curve = fit_anchor_guided_curve(
        skeleton_pts=skeleton_pts,
        marker_pts=marker_pts,
        max_marker_dist=15.0,
        smoothing=0.1,
        num_samples=100,
    )

    assert fitted_curve.shape == (100, 2)
    # Check that the fitted curve passes within sub-pixel distance of marker peaks
    min_dist_to_peak = np.min(np.linalg.norm(fitted_curve - marker_pts[0], axis=1))
    assert min_dist_to_peak < 1.0


def test_line_seg_extractor():
    """Test LineSegExtractor on synthetic line mask."""
    extractor = LineSegExtractor()

    # Create dummy synthetic mask
    mask = np.zeros((100, 200), dtype=np.uint8)
    for px in range(10, 190):
        py = int(50 + 20 * np.sin(px / 30.0))
        mask[max(0, py-2):min(100, py+3), px] = 1

    detections = {
        'line_series': [{
            'xyxy': [100, 100, 300, 200],
            'conf': 0.95,
            'cls': 0,
            'mask': mask,
        }],
        'data_marker': [{
            'xyxy': [140, 165, 150, 175],  # center (145, 170)
            'conf': 0.9,
            'cls': 0,
        }],
        'data_label': [],
        'axis_labels': [],
    }

    # Linear calibration: y_real = (y_pixel - 100) * 0.5
    scale_model = lambda y: (y - 100) * 0.5
    baseline_coord = 200.0

    result = extractor.extract(
        img=np.zeros((300, 400, 3), dtype=np.uint8),
        detections=detections,
        scale_model=scale_model,
        baseline_coord=baseline_coord,
        img_dimensions={'r_squared': 0.99},
    )

    assert result['chart_type'] == 'line'
    assert len(result['data_points']) > 0
    assert 'series_curves' in result
    assert len(result['series_curves']) == 1


def test_area_seg_extractor_stacked():
    """Test AreaSegExtractor for stacked layers and AUC."""
    extractor = AreaSegExtractor()

    # Create 2 stacked area masks
    mask_bottom = np.zeros((100, 200), dtype=np.uint8)
    mask_top = np.zeros((100, 200), dtype=np.uint8)

    # Bottom layer: y between 50 and 90
    mask_bottom[50:90, 20:180] = 1
    # Top layer: y between 20 and 50
    mask_top[20:50, 20:180] = 1

    detections = {
        'area_series': [
            {'xyxy': [100, 100, 300, 200], 'conf': 0.9, 'cls': 0, 'mask': mask_top},
            {'xyxy': [100, 100, 300, 200], 'conf': 0.95, 'cls': 0, 'mask': mask_bottom},
        ],
        'data_label': [],
    }

    scale_model = lambda y: (200 - y) * 1.0
    baseline_coord = 190.0

    result = extractor.extract(
        img=np.zeros((300, 400, 3), dtype=np.uint8),
        detections=detections,
        scale_model=scale_model,
        baseline_coord=baseline_coord,
        img_dimensions={'r_squared': 0.98},
    )

    assert result['chart_type'] == 'area'
    assert len(result['data_points']) > 0
    assert len(result['auc']) == 2
    # Verify both layers have positive AUC
    assert result['auc'][0]['auc'] > 0
    assert result['auc'][1]['auc'] > 0


def test_segmentation_inference_postprocessing():
    """Test _postprocess_segmentation_output with synthetic Top-300 tensors."""
    # output0: (1, 300, 38)
    output0 = np.zeros((1, 300, 38), dtype=np.float32)
    # Detection 0: box [100, 100, 500, 500], score 0.85, cls 0
    output0[0, 0, :4] = [100.0, 100.0, 500.0, 500.0]
    output0[0, 0, 4] = 0.85
    output0[0, 0, 5] = 0.0
    output0[0, 0, 6:] = 1.0  # Coeffs

    # output1: (1, 32, 256, 256)
    output1 = np.ones((1, 32, 256, 256), dtype=np.float32)

    dets = _postprocess_segmentation_output(
        output0=output0,
        output1=output1,
        conf_threshold=0.5,
        ratio=1.0,
        pad=(0, 0),
        mask_threshold=0.5,
    )

    assert len(dets) == 1
    d = dets[0]
    assert d['xyxy'] == [100, 100, 500, 500]
    assert d['conf'] == pytest.approx(0.85)
    assert d['mask'].shape == (400, 400)
    assert np.all(d['mask'] == 1)

    # Test allowlist
    assert _infer_model_output_type(output0, {0: 'line'}, 'segmentation') == 'segmentation'


def test_line_handler_segmentation_integration():
    """Test LineHandler executes LineSegExtractor and formats outputs correctly."""
    from handlers.line_handler import LineHandler

    handler = LineHandler(
        calibration_service=object(),
        spatial_classifier=object(),
    )

    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:60, 20:180] = 1

    detections = {
        'line_series': [{
            'xyxy': [50, 50, 250, 150],
            'conf': 0.9,
            'cls': 0,
            'mask': mask,
        }],
        'data_marker': [],
        'data_label': [],
    }

    calibration = {
        'y': {'func': lambda y: 200.0 - y, 'r2': 0.99},
        'x': {'func': lambda x: x * 2.0, 'r2': 0.99},
    }

    class MockBaseline:
        baselines = []

    extracted = handler.extract_values(
        img=np.zeros((300, 300, 3), dtype=np.uint8),
        detections=detections,
        calibration=calibration,
        baselines=MockBaseline(),
        orientation='vertical',
    )

    assert len(extracted) > 0
    first_pt = extracted[0]
    assert first_pt['type'] == 'line_segment'
    assert 'value' in first_pt
    assert 'x_value' in first_pt
    assert first_pt['x_value'] is not None


def test_area_handler_segmentation_integration():
    """Test AreaHandler executes AreaSegExtractor and appends AUC summary."""
    from handlers.area_handler import AreaHandler

    handler = AreaHandler(
        calibration_service=object(),
        spatial_classifier=object(),
    )

    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[30:70, 20:180] = 1

    detections = {
        'area_series': [{
            'xyxy': [50, 50, 250, 150],
            'conf': 0.95,
            'cls': 0,
            'mask': mask,
        }],
        'data_label': [],
    }

    calibration = {
        'y': {'func': lambda y: 300.0 - y, 'r2': 0.99},
    }

    class MockBaseline:
        baselines = []

    extracted = handler.extract_values(
        img=np.zeros((300, 300, 3), dtype=np.uint8),
        detections=detections,
        calibration=calibration,
        baselines=MockBaseline(),
        orientation='vertical',
    )

    assert len(extracted) > 0
    # Check that area_point and area_series_summary entries were generated
    point_entries = [e for e in extracted if e['type'] == 'area_point']
    summary_entries = [e for e in extracted if e['type'] == 'area_series_summary']

    assert len(point_entries) > 0
    assert len(summary_entries) == 1
    assert 'auc' in summary_entries[0]
    assert summary_entries[0]['auc'] > 0


def test_accuracy_comparator_curve_evaluation():
    """Test AccuracyComparator._compare_curve_values on dense predictions."""
    from evaluation.accuracy_comparator import AccuracyComparator

    comparator = AccuracyComparator()

    # Ground truth with 5 sparse points
    gt = {
        "charts": [{
            "chart_type": "line",
            "data_points": [
                {"x": 10.0, "y": 20.0},
                {"x": 20.0, "y": 40.0},
                {"x": 30.0, "y": 60.0},
                {"x": 40.0, "y": 80.0},
                {"x": 50.0, "y": 100.0},
            ]
        }]
    }

    # Dense prediction sampled every 1.0 units (perfect match y = 2x)
    pred_x = np.linspace(10.0, 50.0, 41)
    pred_y = 2.0 * pred_x

    pred = {
        "chart_type": "line",
        "metadata": {
            "data_points": [
                {"x_value": float(x), "estimated_value": float(y)}
                for x, y in zip(pred_x, pred_y)
            ]
        }
    }

    metrics = comparator._compute_value_metrics(gt, pred, "line")
    assert "mae" in metrics
    assert "rmse" in metrics
    assert metrics["mae"] < 1e-4
    assert metrics["rmse"] < 1e-4
    assert "ccc" in metrics
    assert np.isclose(metrics["ccc"], 1.0)


