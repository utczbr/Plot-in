import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.append('src')
sys.modules.setdefault('cv2', MagicMock())
sys.modules.setdefault('onnxruntime', MagicMock())
sys.modules.setdefault('sklearn', MagicMock())
sys.modules.setdefault('sklearn.cluster', MagicMock())
sys.modules.setdefault('sklearn.metrics', MagicMock())
sys.modules.setdefault('sklearn.ensemble', MagicMock())
sys.modules.setdefault('sklearn.preprocessing', MagicMock())

from ChartAnalysisOrchestrator import ChartAnalysisOrchestrator
from handlers.base_handler import CartesianChartHandler, GridChartHandler, PolarChartHandler
from handlers.types import ExtractionResult, ChartCoordinateSystem
from services.orientation_service import Orientation


class DummyCartesian(CartesianChartHandler):
    def process(self, image, detections, axis_labels, chart_elements, orientation, **kwargs):
        return ExtractionResult(chart_type='dummy', coordinate_system=ChartCoordinateSystem.CARTESIAN, orientation=orientation)


class DummyGrid(GridChartHandler):
    def process(self, image, detections, axis_labels, chart_elements, orientation, **kwargs):
        return ExtractionResult(chart_type='dummy', coordinate_system=ChartCoordinateSystem.GRID, orientation=orientation)


class DummyPolar(PolarChartHandler):
    def process(self, image, detections, axis_labels, chart_elements, orientation, **kwargs):
        return ExtractionResult(chart_type='dummy', coordinate_system=ChartCoordinateSystem.POLAR, orientation=orientation)


class TestOrchestratorRegistry(unittest.TestCase):
    def test_registry_contains_expected_chart_types(self):
        expected = {'bar', 'scatter', 'line', 'box', 'histogram', 'heatmap', 'pie', 'area'}
        self.assertEqual(set(ChartAnalysisOrchestrator._HANDLER_REGISTRY.keys()), expected)

    @patch('ChartAnalysisOrchestrator.ProductionSpatialClassifier', return_value=MagicMock())
    @patch('ChartAnalysisOrchestrator.HeatmapChartClassifier', return_value=MagicMock())
    @patch('ChartAnalysisOrchestrator.PieChartClassifier', return_value=MagicMock())
    def test_initialize_handlers_from_registry(self, *_):
        orchestrator = ChartAnalysisOrchestrator(
            calibration_service=MagicMock(),
            logger=MagicMock(),
        )

        expected = {'bar', 'scatter', 'line', 'box', 'histogram', 'heatmap', 'pie', 'area'}
        self.assertEqual(set(orchestrator.get_supported_chart_types()), expected)

    def test_build_handler_injects_dependencies_by_hierarchy(self):
        orchestrator = ChartAnalysisOrchestrator.__new__(ChartAnalysisOrchestrator)
        orchestrator.calibration_service = MagicMock(name='calibration')
        orchestrator.spatial_classifier = MagicMock(name='spatial')
        orchestrator.dual_axis_service = MagicMock(name='dual')
        orchestrator.meta_clustering_service = MagicMock(name='meta')
        orchestrator.color_mapping_service = MagicMock(name='color')
        orchestrator.legend_matching_service = MagicMock(name='legend')
        orchestrator.logger = MagicMock(name='logger')

        cartesian = orchestrator._build_handler('dummy_cartesian', DummyCartesian)
        grid = orchestrator._build_handler('dummy_grid', DummyGrid)
        polar = orchestrator._build_handler('dummy_polar', DummyPolar)

        self.assertIs(cartesian.calibration_service, orchestrator.calibration_service)
        self.assertIs(cartesian.spatial_classifier, orchestrator.spatial_classifier)
        self.assertIs(cartesian.dual_axis_service, orchestrator.dual_axis_service)
        self.assertIs(cartesian.meta_clustering_service, orchestrator.meta_clustering_service)

        self.assertIs(grid.color_mapper, orchestrator.color_mapping_service)
        self.assertIs(polar.legend_matcher, orchestrator.legend_matching_service)

    @patch('ChartAnalysisOrchestrator.ProductionSpatialClassifier', return_value=MagicMock())
    @patch('ChartAnalysisOrchestrator.HeatmapChartClassifier', return_value=MagicMock())
    @patch('ChartAnalysisOrchestrator.PieChartClassifier', return_value=MagicMock())
    def test_process_chart_backwards_compatible_arguments(self, *_):
        orchestrator = ChartAnalysisOrchestrator(
            calibration_service=MagicMock(),
            logger=MagicMock(),
        )

        result = orchestrator.process_chart(
            image=np.zeros((32, 32, 3), dtype=np.uint8),
            chart_type='bar',
            detections={'bar': []},
            axis_labels=[],
            chart_elements=[],
            orientation='vertical',
        )

        self.assertIsInstance(result, ExtractionResult)


if __name__ == '__main__':
    unittest.main()
