from __future__ import annotations

import unittest

from boot_breaker.collision_model import CollisionModel


class CollisionModelTests(unittest.TestCase):
    def test_center_and_edges(self):
        model = CollisionModel()
        self.assertEqual(model.outgoing_angle(0.0), 0.0)
        self.assertEqual(model.outgoing_angle(-1.0), -52.0)
        self.assertEqual(model.outgoing_angle(1.0), 52.0)

    def test_inverse(self):
        model = CollisionModel()
        self.assertAlmostEqual(model.impact_offset_for_angle(26.0), 0.5)
        self.assertEqual(model.impact_offset_for_angle(80.0), 1.0)


if __name__ == "__main__":
    unittest.main()
