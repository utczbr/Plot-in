"""Tests for area chart handler and extractor."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Ensure src/ is on sys.path
_src_dir = Path(__file__).resolve().parents[2] / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


# ---------------------------------------------------------------------------
# Config / registry wiring tests
# ---------------------------------------------------------------------------

class TestAreaChartWiring:
    """Verify area is wired into config, class maps, chart registry, and orchestrator."""

    def test_area_in_detection_config(self):
        from core.config import MODELS_CONFIG
        assert 'area' in MODELS_CONFIG.detection
        assert MODELS_CONFIG.detection['area'] == 'detect_line.onnx'

    def test_area_class_map_exists(self):
        from core.class_maps import CLASS_MAP_AREA, get_class_map
        assert CLASS_MAP_AREA is not None
        assert CLASS_MAP_AREA[1] == 'data_point'
        # get_class_map should return CLASS_MAP_AREA for 'area', not bar fallback
        result = get_class_map('area')
        assert result is CLASS_MAP_AREA

    def test_area_in_chart_element_key_map(self):
        from core.chart_registry import CHART_ELEMENT_KEY_MAP
        from core.enums import ChartType
        assert ChartType.AREA.value in CHART_ELEMENT_KEY_MAP
        assert CHART_ELEMENT_KEY_MAP[ChartType.AREA.value] == 'data_point'

    def test_area_in_orchestrator_registry(self):
        from ChartAnalysisOrchestrator import ChartAnalysisOrchestrator
        assert 'area' in ChartAnalysisOrchestrator._HANDLER_REGISTRY

    def test_area_handler_is_cartesian(self):
        from handlers.area_handler import AreaHandler
        from handlers.base_handler import CartesianExtractionHandler
        assert issubclass(AreaHandler, CartesianExtractionHandler)


# ---------------------------------------------------------------------------
# AreaExtractor unit tests
# ---------------------------------------------------------------------------

class TestAreaExtractor:
    """Unit tests for the AreaExtractor."""

    def _make_point(self, x_center: float, y_center: float, conf: float = 0.9):
        half = 5
        return {
            'xyxy': [x_center - half, y_center - half, x_center + half, y_center + half],
            'conf': conf,
        }

    def test_extract_with_data_points(self):
        from extractors.area_extractor import AreaExtractor

        extractor = AreaExtractor()
        detections = {
            'data_point': [
                self._make_point(100, 200),
                self._make_point(200, 150),
                self._make_point(300, 180),
            ],
            'chart_title': [],
            'axis_title': [],
        }

        # Linear scale model: pixel y -> value (inverted — higher pixel = lower value)
        def scale_model(y):
            return 400 - y

        result = extractor.extract(
            img=np.zeros((400, 400, 3), dtype=np.uint8),
            detections=detections,
            scale_model=scale_model,
            baseline_coord=350.0,
            img_dimensions={'r_squared': 0.99},
        )

        assert len(result['data_points']) == 3
        for pt in result['data_points']:
            assert pt['estimated_value'] is not None
            assert pt['estimated_value'] > 0

    def test_auc_computation_simple(self):
        from extractors.area_extractor import AreaExtractor

        extractor = AreaExtractor()

        # 3 points at x=100,200,300 all with value=10 → AUC = 10 * (300-100) = 2000
        detections = {
            'data_point': [
                self._make_point(100, 200),
                self._make_point(200, 200),
                self._make_point(300, 200),
            ],
            'chart_title': [],
            'axis_title': [],
        }

        # Scale model where value at y=200 is 10, value at baseline y=300 is 0
        def scale_model(y):
            return (300 - y) / 10.0

        result = extractor.extract(
            img=np.zeros((400, 400, 3), dtype=np.uint8),
            detections=detections,
            scale_model=scale_model,
            baseline_coord=300.0,
            img_dimensions={},
        )

        auc = result['auc']
        assert auc['total_auc'] is not None
        assert auc['num_points'] == 3
        assert auc['baseline_value'] is not None
        # All points have estimated_value = |scale(300) - scale(200)| = |0 - 10| = 10
        # Trapezoidal AUC over pixel x-coords [100, 200, 300] with y=[10, 10, 10]
        # = 10 * (300 - 100) = 2000
        assert abs(auc['total_auc'] - 2000.0) < 1.0

    def test_auc_with_single_point(self):
        from extractors.area_extractor import AreaExtractor

        extractor = AreaExtractor()
        detections = {
            'data_point': [self._make_point(100, 200)],
            'chart_title': [],
            'axis_title': [],
        }

        def scale_model(y):
            return 300 - y

        result = extractor.extract(
            img=np.zeros((400, 400, 3), dtype=np.uint8),
            detections=detections,
            scale_model=scale_model,
            baseline_coord=300.0,
            img_dimensions={},
        )

        # Single point can't form a trapezoid
        assert result['auc']['total_auc'] is None

    def test_empty_detections(self):
        from extractors.area_extractor import AreaExtractor

        extractor = AreaExtractor()
        detections = {
            'data_point': [],
            'chart_title': [],
            'axis_title': [],
        }

        result = extractor.extract(
            img=np.zeros((400, 400, 3), dtype=np.uint8),
            detections=detections,
            scale_model=lambda y: y,
            baseline_coord=300.0,
            img_dimensions={},
        )

        assert len(result['data_points']) == 0
        assert result['auc']['total_auc'] is None

    def test_no_scale_model(self):
        from extractors.area_extractor import AreaExtractor

        extractor = AreaExtractor()
        detections = {
            'data_point': [self._make_point(100, 200)],
            'chart_title': [],
            'axis_title': [],
        }

        result = extractor.extract(
            img=np.zeros((400, 400, 3), dtype=np.uint8),
            detections=detections,
            scale_model=None,
            baseline_coord=None,
            img_dimensions={},
        )

        # Without scale model, values should still be computed (pixel distance)
        assert len(result['data_points']) == 1


# ---------------------------------------------------------------------------
# AreaHandler unit tests
# ---------------------------------------------------------------------------

class TestAreaHandler:
    """Tests for the AreaHandler class."""

    @staticmethod
    def _make_handler():
        from handlers.area_handler import AreaHandler
        return AreaHandler(
            calibration_service=mock.MagicMock(),
            spatial_classifier=mock.MagicMock(),
        )

    def test_get_chart_type(self):
        handler = self._make_handler()
        assert handler.get_chart_type() == "area"

    def test_extract_values_returns_list(self):
        handler = self._make_handler()

        # Minimal mock calibration
        class MockCalResult:
            func = lambda self, y: 300 - y
            r2 = 0.99

        calibration = {'y': MockCalResult()}

        # Minimal mock baselines
        class MockBaseline:
            axis_id = 'y'
            value = 300.0

        class MockBaselineResult:
            baselines = [MockBaseline()]

        detections = {
            'data_point': [
                {'xyxy': [95, 195, 105, 205], 'conf': 0.9},
                {'xyxy': [195, 145, 205, 155], 'conf': 0.9},
            ],
        }

        result = handler.extract_values(
            img=np.zeros((400, 400, 3), dtype=np.uint8),
            detections=detections,
            calibration=calibration,
            baselines=MockBaselineResult(),
            orientation='vertical',
        )

        assert isinstance(result, list)
        assert len(result) > 0

        # Check for area_point entries
        area_points = [r for r in result if r.get('type') == 'area_point']
        assert len(area_points) == 2
        for pt in area_points:
            assert 'value' in pt
            assert 'bbox' in pt

    def test_extract_values_includes_auc_summary(self):
        handler = self._make_handler()

        class MockCalResult:
            func = lambda self, y: (300 - y) / 10.0
            r2 = 0.99

        calibration = {'y': MockCalResult()}

        class MockBaseline:
            axis_id = 'y'
            value = 300.0

        class MockBaselineResult:
            baselines = [MockBaseline()]

        detections = {
            'data_point': [
                {'xyxy': [95, 195, 105, 205], 'conf': 0.9},
                {'xyxy': [195, 195, 205, 205], 'conf': 0.9},
                {'xyxy': [295, 195, 305, 205], 'conf': 0.9},
            ],
        }

        result = handler.extract_values(
            img=np.zeros((400, 400, 3), dtype=np.uint8),
            detections=detections,
            calibration=calibration,
            baselines=MockBaselineResult(),
            orientation='vertical',
        )

        summaries = [r for r in result if r.get('type') == 'area_series_summary']
        assert len(summaries) == 1
        assert summaries[0]['auc'] is not None
        assert summaries[0]['baseline_value'] is not None

    def test_extract_values_no_calibration(self):
        handler = self._make_handler()

        class MockBaselineResult:
            baselines = []

        result = handler.extract_values(
            img=np.zeros((400, 400, 3), dtype=np.uint8),
            detections={'data_point': [{'xyxy': [95, 195, 105, 205], 'conf': 0.9}]},
            calibration={},
            baselines=MockBaselineResult(),
            orientation='vertical',
        )

        # Should return empty list when calibration is missing
        assert result == []
