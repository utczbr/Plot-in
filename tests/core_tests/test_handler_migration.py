import sys
import unittest
from types import SimpleNamespace
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

from handlers.base_handler import CartesianExtractionHandler
from handlers.legacy import BaseChartHandler
from handlers.bar_handler import BarHandler
from handlers.line_handler import LineHandler
from handlers.scatter_handler import ScatterHandler
from handlers.box_handler import BoxHandler
from handlers.histogram_handler import HistogramHandler
from services.orientation_service import Orientation


class TestHandlerMigration(unittest.TestCase):
    def _make_cartesian_handler(self, handler_cls):
        return handler_cls(
            calibration_service=MagicMock(),
            spatial_classifier=MagicMock(),
            dual_axis_service=MagicMock(),
            meta_clustering_service=MagicMock(),
            logger=MagicMock(),
        )

    def test_cartesian_handlers_inherit_canonical_runtime_base(self):
        handlers = [BarHandler, LineHandler, ScatterHandler, BoxHandler, HistogramHandler]
        for handler_cls in handlers:
            self.assertTrue(issubclass(handler_cls, CartesianExtractionHandler))
            self.assertNotIn(BaseChartHandler, handler_cls.__mro__)

    def test_legacy_base_is_compatibility_shim(self):
        self.assertTrue(issubclass(BaseChartHandler, CartesianExtractionHandler))

    @patch('extractors.bar_extractor.BarExtractor.extract', return_value={'bars': [{'value': 1}]})
    def test_bar_extract_values_accepts_orientation_enum(self, mock_extract):
        handler = self._make_cartesian_handler(BarHandler)
        baselines = SimpleNamespace(baselines=[SimpleNamespace(axis_id='y', value=50.0)])
        calibration = {'primary': SimpleNamespace(func=lambda x: x)}
        detections = {'bar': [{'xyxy': [10, 10, 20, 30]}], 'axis_labels': []}

        result = handler.extract_values(
            img=np.zeros((64, 64, 3), dtype=np.uint8),
            detections=detections,
            calibration=calibration,
            baselines=baselines,
            orientation=Orientation.VERTICAL,
        )

        mock_extract.assert_called_once()
        self.assertEqual(len(result), 1)

    @patch('extractors.histogram_extractor.HistogramExtractor.extract', return_value={'bars': [{'value': 1}]})
    def test_histogram_extract_values_accepts_orientation_enum(self, mock_extract):
        handler = self._make_cartesian_handler(HistogramHandler)
        baselines = SimpleNamespace(baselines=[SimpleNamespace(axis_id='y', value=40.0)])
        calibration = {'primary': SimpleNamespace(func=lambda x: x)}
        detections = {'bar': [{'xyxy': [10, 10, 20, 30]}], 'axis_labels': []}

        result = handler.extract_values(
            img=np.zeros((64, 64, 3), dtype=np.uint8),
            detections=detections,
            calibration=calibration,
            baselines=baselines,
            orientation=Orientation.VERTICAL,
        )

        mock_extract.assert_called_once()
        self.assertEqual(len(result), 1)

    @patch('extractors.scatter_extractor.ScatterExtractor.extract')
    def test_scatter_extract_values_uses_enum_safe_axis_mapping(self, mock_extract):
        primary_func = lambda x: x + 1.0
        secondary_func = lambda x: x + 2.0

        captured = {}

        def _fake_extract(**kwargs):
            captured['scale_model'] = kwargs.get('scale_model')
            captured['x_scale_model'] = kwargs.get('x_scale_model')
            return {
                'data_points': [
                    {
                        'xyxy': [10, 10, 20, 20],
                        'x_calibrated': 1.0,
                        'y_calibrated': 2.0,
                        'x_pixel': 15.0,
                        'y_pixel': 15.0,
                    }
                ]
            }

        mock_extract.side_effect = _fake_extract

        handler = self._make_cartesian_handler(ScatterHandler)
        calibration = {
            'primary': SimpleNamespace(func=primary_func),
            'secondary': SimpleNamespace(func=secondary_func),
        }

        result = handler.extract_values(
            img=np.zeros((64, 64, 3), dtype=np.uint8),
            detections={'data_point': [{'xyxy': [10, 10, 20, 20]}]},
            calibration=calibration,
            baselines=SimpleNamespace(baselines=[]),
            orientation=Orientation.VERTICAL,
        )

        self.assertIs(captured['scale_model'], primary_func)
        self.assertIs(captured['x_scale_model'], secondary_func)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
