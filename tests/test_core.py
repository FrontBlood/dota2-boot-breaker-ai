from __future__ import annotations

import unittest

from lockpick_ai.config import ControlConfig, TrackingConfig, load_config
from lockpick_ai.controller import PredictiveController
from lockpick_ai.geometry import AngularZone, forward_distance, signed_delta
from lockpick_ai.simulator import render_frame
from lockpick_ai.tracker import AngleTracker, MotionState
from lockpick_ai.vision import DialVision


class GeometryTests(unittest.TestCase):
    def test_wraparound(self):
        self.assertEqual(signed_delta(359.0, 1.0), 2.0)
        self.assertEqual(forward_distance(350.0, 10.0, 1), 20.0)
        self.assertTrue(AngularZone(350.0, 15.0, "yellow").contains(2.0, 2.0))
        self.assertFalse(AngularZone(350.0, 15.0, "yellow").contains(14.0, 2.0))


class TrackerTests(unittest.TestCase):
    def test_clockwise_speed_across_zero(self):
        tracker = AngleTracker(TrackingConfig(history_ms=500, min_samples=4))
        state = None
        for i, angle in enumerate((350.0, 355.0, 0.0, 5.0, 10.0)):
            state = tracker.update(i * 0.05, angle)
        assert state is not None
        self.assertTrue(state.stable)
        self.assertAlmostEqual(state.speed_dps, 100.0, delta=1.0)


class ControllerTests(unittest.TestCase):
    def test_predictive_click(self):
        config = ControlConfig(input_latency_ms=50, latency_jitter_ms=0, base_margin_deg=2, speed_error_fraction=0)
        controller = PredictiveController(config)
        state = MotionState(angle=12.0, speed_dps=100.0, direction=1, stable=True, sample_count=5)
        decision = controller.decide(1.0, state, (AngularZone(14.0, 24.0, "yellow"),))
        self.assertAlmostEqual(decision.predicted_angle or 0, 17.0)
        self.assertTrue(decision.click)


class VisionTests(unittest.TestCase):
    def test_synthetic_detection(self):
        config = load_config("config.json")
        zones = (AngularZone(25.0, 42.0, "yellow"), AngularZone(205.0, 226.0, "blue"))
        frame = render_frame(config, 315.0, zones)
        result = DialVision(config.dial, config.vision).detect(frame)
        self.assertIsNotNone(result.pointer_angle)
        self.assertAlmostEqual(result.pointer_angle or 0, 315.0, delta=2.0)
        self.assertEqual({zone.kind for zone in result.zones}, {"yellow", "blue"})

    def test_pointer_glow_is_not_a_blue_zone(self):
        config = load_config("config.json")
        frame = render_frame(config, 315.0, (AngularZone(25.0, 42.0, "yellow"),))
        result = DialVision(config.dial, config.vision).detect(frame)
        self.assertNotIn("blue", {zone.kind for zone in result.zones})


if __name__ == "__main__":
    unittest.main()
