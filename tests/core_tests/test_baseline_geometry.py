import sys
import unittest

import numpy as np

sys.path.append('src')

from core.baseline.geometry import (
    aggregate_stack_near_ends,
    extract_near_far_ends,
    validate_xyxy,
)
from services.orientation_service import Orientation


class TestBaselineGeometry(unittest.TestCase):
    def test_validate_xyxy(self):
        self.assertTrue(validate_xyxy([0, 1, 2, 3]))
        self.assertFalse(validate_xyxy([0, 1, 2]))
        self.assertFalse(validate_xyxy(['x', 1, 2, 3]))

    def test_aggregate_stack_near_ends_vertical_normal_and_inverted(self):
        elements = [
            {'xyxy': [10, 20, 20, 90]},
            {'xyxy': [12, 30, 22, 80]},
            {'xyxy': [80, 25, 90, 85]},
        ]

        normal = aggregate_stack_near_ends(
            elements,
            orientation=Orientation.VERTICAL,
            img_h=100,
            band_frac=0.5,
            inverted_axis=False,
        )
        inverted = aggregate_stack_near_ends(
            elements,
            orientation=Orientation.VERTICAL,
            img_h=100,
            band_frac=0.5,
            inverted_axis=True,
        )

        self.assertEqual(len(normal), 2)
        self.assertAlmostEqual(float(normal[0]), 90.0, delta=0.1)
        self.assertAlmostEqual(float(normal[1]), 85.0, delta=0.1)
        self.assertAlmostEqual(float(inverted[0]), 20.0, delta=0.1)
        self.assertAlmostEqual(float(inverted[1]), 25.0, delta=0.1)

    def test_aggregate_stack_near_ends_horizontal_normal_and_inverted(self):
        elements = [
            {'xyxy': [15, 10, 80, 20]},
            {'xyxy': [20, 12, 70, 22]},
            {'xyxy': [25, 80, 90, 90]},
        ]

        normal = aggregate_stack_near_ends(
            elements,
            orientation=Orientation.HORIZONTAL,
            img_h=100,
            band_frac=0.5,
            inverted_axis=False,
        )
        inverted = aggregate_stack_near_ends(
            elements,
            orientation=Orientation.HORIZONTAL,
            img_h=100,
            band_frac=0.5,
            inverted_axis=True,
        )

        self.assertEqual(len(normal), 2)
        self.assertAlmostEqual(float(normal[0]), 15.0, delta=0.1)
        self.assertAlmostEqual(float(normal[1]), 25.0, delta=0.1)
        self.assertAlmostEqual(float(inverted[0]), 80.0, delta=0.1)
        self.assertAlmostEqual(float(inverted[1]), 90.0, delta=0.1)

    def test_extract_near_far_ends(self):
        elements = [
            {'xyxy': [10, 20, 20, 90]},
            {'xyxy': [40, 30, 50, 80]},
        ]
        v = extract_near_far_ends(elements, is_vertical=True)
        h = extract_near_far_ends(elements, is_vertical=False)

        self.assertTrue(np.allclose(v['near_ends'], np.array([90, 80], dtype=np.float32)))
        self.assertTrue(np.allclose(v['far_ends'], np.array([20, 30], dtype=np.float32)))
        self.assertTrue(np.allclose(h['near_ends'], np.array([10, 40], dtype=np.float32)))
        self.assertTrue(np.allclose(h['far_ends'], np.array([20, 50], dtype=np.float32)))


if __name__ == '__main__':
    unittest.main()
