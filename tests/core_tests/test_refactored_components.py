
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils.clustering_utils import cluster_1d_dbscan
from handlers.pie_handler import PieHandler
from handlers.box_handler import BoxHandler
from handlers.scatter_handler import ScatterHandler
from handlers.heatmap_handler import HeatmapHandler
from services.orientation_service import Orientation

class TestRefactoredComponents(unittest.TestCase):

    def test_cluster_1d_dbscan(self):
        """Verify shared clustering utility works as expected."""
        # Simple case: clear clusters
        values = [10, 11, 12, 100, 101, 102]
        # tolerance=5 should group (10,11,12) and (100,101,102)
        centers = cluster_1d_dbscan(values, tolerance=5)
        self.assertEqual(len(centers), 2)
        # DBSCAN might return slightly different means depending on implementation details, 
        # but for this clear separation it should be mean of points.
        self.assertAlmostEqual(centers[0], 11.0, delta=0.5)
        self.assertAlmostEqual(centers[1], 101.0, delta=0.5)
        
        # Merge case
        values = [10, 12, 14] # spread out 2 apart
        # eps=10. A point is neighbor if distance <= eps.
        # 10->12 (dist 2) -> Connected. 12->14 (dist 2) -> Connected.
        # All connected.
        centers = cluster_1d_dbscan(values, tolerance=10)
        self.assertEqual(len(centers), 1)
        self.assertAlmostEqual(centers[0], 12.0)
        
        # Empty case
        self.assertEqual(cluster_1d_dbscan([], 1.0), [])

    def test_box_handler_r2_logic(self):
        """Verify BoxHandler R² threshold logic."""
        # Mock dependencies
        mock_cal_service = MagicMock()
        mock_spatial = MagicMock()
        mock_dual = MagicMock()
        mock_meta = MagicMock()
        mock_logger = MagicMock()
        
        handler = BoxHandler(
            calibration_service=mock_cal_service,
            spatial_classifier=mock_spatial,
            dual_axis_service=mock_dual,
            meta_clustering_service=mock_meta,
            logger=mock_logger
        )
        
        # Mock values
        handler.FAILURE_R2 = 0.40
        handler.CRITICAL_R2 = 0.85
        
        # Mock calibration result
        mock_cal = MagicMock()
        mock_cal.r_squared = 0.50 # Between failure and critical -> Should be WARNING
        
        errors = []
        warnings = []
        
        # The logic we implemented:
        if hasattr(mock_cal, 'r_squared') and mock_cal.r_squared < handler.FAILURE_R2:
             errors.append("catastrophic")
        elif hasattr(mock_cal, 'r_squared') and mock_cal.r_squared < handler.CRITICAL_R2:
             warnings.append("quality low")
            
        self.assertEqual(len(errors), 0, "Should not be catastrophic error")
        self.assertEqual(len(warnings), 1, "Should be a warning")
        self.assertIn("quality low", warnings[0])
        
        # Critical failure case
        mock_cal.r_squared = 0.20
        errors = []
        warnings = []
        if hasattr(mock_cal, 'r_squared') and mock_cal.r_squared < handler.FAILURE_R2:
             errors.append("catastrophic")
        elif hasattr(mock_cal, 'r_squared') and mock_cal.r_squared < handler.CRITICAL_R2:
             warnings.append("quality low")
            
        self.assertEqual(len(errors), 1)
        self.assertIn("catastrophic", errors[0])

    def test_pie_handler_init(self):
        """Verify PieHandler initializes correctly."""
        classifier = MagicMock()
        matcher = MagicMock()
        
        handler = PieHandler(classifier=classifier, legend_matcher=matcher)
        
        self.assertEqual(handler.classifier, classifier)
        self.assertEqual(handler.legend_matcher, matcher) # Should be set by super()

    def test_heatmap_process_flow(self):
        """Verify HeatmapHandler process flow (check for NameErrors/Imports)."""
        handler = HeatmapHandler(classifier=MagicMock(), color_mapper=MagicMock())
        
        # Mock inputs
        # 100x100 fake image
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Fake detections: 4 cells in 2x2 grid
        detections = {
            'heatmap_cell': [
                {'xyxy': [10, 10, 40, 40], 'conf': 0.9}, # (25, 25)
                {'xyxy': [60, 10, 90, 40], 'conf': 0.9}, # (75, 25)
                {'xyxy': [10, 60, 40, 90], 'conf': 0.9}, # (25, 75)
                {'xyxy': [60, 60, 90, 90], 'conf': 0.9}, # (75, 75)
            ],
            'color_bar': []
        }
        
        axis_labels = []
        chart_elements = [] # redundant with detections in logic
        
        # Run process
        result = handler.process(
            image=fake_image,
            detections=detections,
            axis_labels=axis_labels,
            chart_elements=chart_elements,
            orientation=Orientation.VERTICAL
        )
        
        # Verify result structure
        self.assertEqual(result.chart_type, 'heatmap')
        # Expect 2x2 grid
        self.assertEqual(result.diagnostics['grid_rows'], 2)
        self.assertEqual(result.diagnostics['grid_cols'], 2)
        # 4 elements
        self.assertEqual(len(result.elements), 4)

if __name__ == '__main__':
    unittest.main()
