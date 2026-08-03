# -*- coding: utf-8 -*-
import unittest
import numpy as np
import cv2
from core.pdf_processor import _find_chart_regions_cv2


class TestPDFMultiChartSlicer(unittest.TestCase):
    def test_find_chart_regions_cv2_multi_chart_canvas(self):
        # Create a synthetic 1000x800 page canvas with two separated chart boxes
        canvas = np.full((1000, 800, 3), 255, dtype=np.uint8)

        # Chart 1: top box at y=50..400, x=100..700 with grid lines & text-like elements
        cv2.rectangle(canvas, (100, 50), (700, 400), (0, 0, 0), 2)
        cv2.line(canvas, (100, 200), (700, 200), (200, 200, 200), 1)
        # Add text-like boxes so _is_likely_chart_image passes
        for x in range(150, 650, 80):
            cv2.rectangle(canvas, (x, 370), (x + 40, 390), (0, 0, 0), -1)

        # Chart 2: bottom box at y=500..850, x=100..700 with grid lines & text-like elements
        cv2.rectangle(canvas, (100, 500), (700, 850), (0, 0, 0), 2)
        cv2.line(canvas, (100, 650), (700, 650), (200, 200, 200), 1)
        for x in range(150, 650, 80):
            cv2.rectangle(canvas, (x, 820), (x + 40, 840), (0, 0, 0), -1)

        crops = _find_chart_regions_cv2(canvas, min_width=300, min_height=200)
        self.assertGreaterEqual(len(crops), 1)

    def test_find_chart_regions_cv2_empty_canvas(self):
        blank = np.full((400, 400, 3), 255, dtype=np.uint8)
        crops = _find_chart_regions_cv2(blank)
        self.assertEqual(len(crops), 0)


if __name__ == '__main__':
    unittest.main()
