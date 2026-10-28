
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock

# Ensure src is in path
sys.path.append('src')

from services.orientation_service import Orientation
from core.classifiers.bar_chart_classifier import BarChartClassifier
from core.classifiers.line_chart_classifier import LineChartClassifier
from core.classifiers.scatter_chart_classifier import ScatterChartClassifier
from core.classifiers.box_chart_classifier import BoxChartClassifier
from core.classifiers.histogram_chart_classifier import HistogramChartClassifier
from core.classifiers.production_classifier import ProductionSpatialClassifier

class TestClassifiers(unittest.TestCase):
    
    def test_orientation_enum_usage(self):
        """Verify that classifiers accept Orientation enum"""
        classifier = BarChartClassifier()
        # Mock data
        axis_labels = [{'xyxy': [0, 0, 10, 10], 'text': '10', 'confidence': 0.9, 'is_numeric': True}]
        elements = [{'xyxy': [20, 20, 30, 100], 'text': 'bar'}]
        
        # Should not raise type error
        try:
            result = classifier.classify(
                axis_labels=axis_labels,
                chart_elements=elements,
                img_width=800,
                img_height=600,
                orientation=Orientation.VERTICAL
            )
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Classify raised exception with Orientation enum: {e}")

    def test_get_default_params_consistency(self):
        """Verify that get_default_params returns expected keys"""
        classifiers = [
            BarChartClassifier,
            LineChartClassifier,
            ScatterChartClassifier,
            BoxChartClassifier,
            HistogramChartClassifier
        ]
        
        for cls in classifiers:
            defaults = cls.get_default_params()
            self.assertIsInstance(defaults, dict)
            self.assertTrue(len(defaults) > 0, f"{cls.__name__} returned empty defaults")

    def test_production_classifier_defaults(self):
        """Verify that ProductionSpatialClassifier loads defaults correctly"""
        prod_classifier = ProductionSpatialClassifier()
        defaults = prod_classifier._get_default_parameters()
        
        self.assertIn('bar', defaults)
        self.assertIn('line', defaults)
        
        # Verify specific key from BarChartClassifier exists
        self.assertIn('scale_size_max_width', defaults['bar'])
        self.assertEqual(defaults['bar']['scale_size_max_width'], 0.08)

    def test_production_classifier_integration(self):
        """Verify ProductionSpatialClassifier integration"""
        prod = ProductionSpatialClassifier()
        
        # Mocking
        axis_labels = [{'xyxy': [0, 550, 50, 560], 'text': '10', 'confidence': 0.9, 'is_numeric': True}]
        elements = [{'xyxy': [100, 100, 120, 500]}]
        
        result = prod.classify(
            chart_type='bar',
            axis_labels=axis_labels,
            chart_elements=elements,
            img_width=800,
            img_height=600,
            orientation=Orientation.VERTICAL
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.metadata['chart_type'], 'bar')
        self.assertEqual(result.metadata['orientation'], Orientation.VERTICAL)


def _make_label(x1, y1, x2, y2, text="1.0", conf=0.9):
    return {
        "xyxy": [x1, y1, x2, y2],
        "text": text,
        "cleanedvalue": None,
        "ocr_confidence": conf,
    }


def _make_scatter_point(cx, cy, size=10):
    return {"xyxy": [cx - size, cy - size, cx + size, cy + size]}


class TestScatterClassifierFeaturesAndSplit:
    def test_is_numeric_is_bool_for_numeric_text(self):
        classifier = ScatterChartClassifier()
        img_w, img_h = 800, 600
        labels = [
            _make_label(10, 290, 40, 310, text="1.5"),
            _make_label(10, 190, 40, 210, text="Weight"),
        ]
        features = classifier._extract_scatter_features(labels, img_w, img_h)
        assert len(features) == 2
        assert isinstance(features[0]['is_numeric'], bool)
        assert features[0]['is_numeric'] is True
        assert isinstance(features[1]['is_numeric'], bool)
        assert features[1]['is_numeric'] is False

    def _make_vertical_scatter_labels(self):
        y_axis_labels = [
            _make_label(10, 100, 90, 120, text="10"),
            _make_label(10, 200, 90, 220, text="20"),
            _make_label(10, 300, 90, 320, text="30"),
            _make_label(10, 400, 90, 420, text="40"),
        ]
        x_axis_labels = [
            _make_label(150, 540, 230, 560, text="1"),
            _make_label(350, 540, 430, 560, text="2"),
            _make_label(550, 540, 630, 560, text="3"),
            _make_label(650, 540, 730, 560, text="4"),
        ]
        return y_axis_labels, x_axis_labels

    def test_x_scale_labels_and_y_scale_labels_populated(self):
        classifier = ScatterChartClassifier()
        y_labels, x_labels = self._make_vertical_scatter_labels()
        all_labels = y_labels + x_labels
        points = [
            _make_scatter_point(200, 200),
            _make_scatter_point(400, 300),
            _make_scatter_point(600, 400),
        ]
        result = classifier.classify(
            axis_labels=all_labels,
            chart_elements=points,
            img_width=800,
            img_height=600,
            orientation=Orientation.VERTICAL,
        )
        assert len(result.x_scale_labels) > 0
        assert len(result.y_scale_labels) > 0

    def test_no_label_in_both_x_and_y(self):
        classifier = ScatterChartClassifier()
        y_labels, x_labels = self._make_vertical_scatter_labels()
        all_labels = y_labels + x_labels
        points = [_make_scatter_point(400, 300)]
        result = classifier.classify(
            axis_labels=all_labels,
            chart_elements=points,
            img_width=800,
            img_height=600,
            orientation=Orientation.VERTICAL,
        )
        x_ids = {id(lbl) for lbl in result.x_scale_labels}
        y_ids = {id(lbl) for lbl in result.y_scale_labels}
        assert len(x_ids & y_ids) == 0

