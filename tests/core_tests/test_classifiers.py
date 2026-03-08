
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
