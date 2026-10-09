from __future__ import annotations

import unittest

import numpy as np

from boot_breaker.config import ControlConfig, FieldConfig, TrackingConfig, VisionConfig
from boot_breaker.controller import BootBreakerController, Phase
from boot_breaker.tracker import BootTracker, MotionState, predict_paddle_intercept, reflected_x
from boot_breaker.vision import AimDetection, BootBreakerVision, BootDetection, PaddleDetection, VisionResult


class TrajectoryTests(unittest.TestCase):
    def test_reflection(self):
        self.assertAlmostEqual(reflected_x(90.0, 30.0, 1.0, 0.0, 100.0), 80.0)
        self.assertAlmostEqual(reflected_x(10.0, -30.0, 1.0, 0.0, 100.0), 20.0)

    def test_intercept(self):
        state = MotionState(100.0, 500.0, 200.0, 400.0, True, 5, 1.0)
        self.assertAlmostEqual(predict_paddle_intercept(state, 700.0, 0.0, 600.0) or 0, 200.0)

    def test_tracker_velocity(self):
        tracker = BootTracker(TrackingConfig(history_ms=500, min_samples=3))
        state = None
        for index in range(5):
            state = tracker.update(index * 0.02, BootDetection(100 + index * 4, 200 + index * 6, 1.0, "test"))
        assert state is not None
        self.assertTrue(state.stable)
        self.assertAlmostEqual(state.vx, 200.0, delta=2.0)
        self.assertAlmostEqual(state.vy, 300.0, delta=2.0)

    def test_default_tracker_recovers_on_second_sample(self):
        tracker = BootTracker(TrackingConfig())
        first = tracker.update(0.00, BootDetection(100.0, 200.0, 1.0, "test"))
        second = tracker.update(0.04, BootDetection(108.0, 212.0, 1.0, "test"))
        self.assertIsNotNone(first)
        self.assertFalse(first.stable)
        self.assertIsNotNone(second)
        self.assertTrue(second.stable)

    def test_track_is_discarded_when_extrapolated_below_paddle(self):
        tracker = BootTracker(TrackingConfig(max_tracking_y=730.0))
        tracker.update(0.00, BootDetection(100.0, 680.0, 1.0, "test"))
        tracker.update(0.04, BootDetection(104.0, 700.0, 1.0, "test"))
        state = tracker.update(0.12, None)
        self.assertIsNone(state)
        self.assertEqual(tracker.rejection_reason, "below_paddle")

    def test_stationary_false_track_is_rejected(self):
        tracker = BootTracker(
            TrackingConfig(
                history_ms=500,
                min_samples=3,
                stationary_timeout_ms=200,
                stationary_radius_px=4,
            )
        )
        state = None
        rejected = False
        for index in range(8):
            state = tracker.update(index * 0.05, BootDetection(300, 100, 1.0, "test"))
            if tracker.rejected_stationary:
                rejected = True
                break
        self.assertIsNone(state)
        self.assertTrue(rejected)

    def test_two_low_motion_frames_reject_immediately(self):
        tracker = BootTracker(
            TrackingConfig(
                history_ms=500,
                min_samples=3,
                min_frame_displacement_px=3,
                low_motion_reject_frames=2,
                stationary_timeout_ms=5000,
            )
        )
        tracker.update(0.00, BootDetection(300.0, 100.0, 1.0, "test"))
        tracker.update(0.04, BootDetection(301.0, 100.5, 1.0, "test"))
        state = tracker.update(0.08, BootDetection(301.5, 101.0, 1.0, "test"))
        self.assertIsNone(state)
        self.assertTrue(tracker.rejected_stationary)

    def test_direction_reversal_uses_path_speed_not_net_speed(self):
        tracker = BootTracker(
            TrackingConfig(
                history_ms=500,
                min_samples=3,
                min_valid_speed_px_s=120,
                stationary_timeout_ms=5000,
            )
        )
        state = None
        for index, y in enumerate((100.0, 110.0, 120.0, 110.0, 100.0)):
            state = tracker.update(index * 0.04, BootDetection(300.0, y, 1.0, "test"))
        self.assertIsNotNone(state)
        self.assertFalse(tracker.rejected_stationary)

    def test_low_motion_near_paddle_is_not_rejected(self):
        tracker = BootTracker(
            TrackingConfig(
                history_ms=500,
                min_samples=3,
                no_brick_zone_y=500,
                stationary_timeout_ms=100,
            )
        )
        state = None
        for index in range(5):
            state = tracker.update(index * 0.05, BootDetection(300.0, 690.0, 1.0, "test"))
        self.assertIsNotNone(state)
        self.assertFalse(tracker.rejected_stationary)

    def test_rejected_location_is_not_immediately_reacquired(self):
        vision = BootBreakerVision(VisionConfig())
        cyan = BootDetection(300, 100, 1.0, "cyan")
        moving = BootDetection(302, 101, 0.8, "motion")
        vision.reject_boot_location(cyan)
        self.assertIsNone(vision._select_boot([cyan], [moving]))

    def test_new_track_requires_arc_geometry(self):
        vision = BootBreakerVision(VisionConfig())
        moving = BootDetection(302, 101, 0.8, "motion")
        plain_cyan = BootDetection(300, 100, 1.0, "cyan", arc_score=0.0)
        arc_cyan = BootDetection(300, 100, 1.0, "cyan", arc_score=0.8)
        self.assertIsNone(vision._select_boot([plain_cyan], [moving]))
        self.assertEqual(vision._select_boot([arc_cyan], [moving]), arc_cyan)

    def test_low_safe_zone_reacquires_without_arc(self):
        vision = BootBreakerVision(VisionConfig())
        cyan = BootDetection(300, 600, 1.0, "cyan", arc_score=0.0)
        moving = BootDetection(304, 602, 0.8, "motion")
        self.assertEqual(vision._select_boot([cyan], [moving]), cyan)

    def test_low_rebound_uses_wider_continuity_gate(self):
        vision = BootBreakerVision(VisionConfig())
        vision.last_boot = BootDetection(300, 690, 1.0, "silhouette")
        cyan = BootDetection(300, 530, 1.0, "cyan", arc_score=0.0)
        moving = BootDetection(304, 532, 0.8, "motion")
        self.assertEqual(vision._select_boot([cyan], [moving]), cyan)

    def test_upper_candidate_without_arc_requires_three_moving_frames(self):
        vision = BootBreakerVision(VisionConfig())
        outputs = []
        for x in (100.0, 110.0, 120.0):
            cyan = BootDetection(x, 300, 1.0, "cyan", arc_score=0.0)
            moving = BootDetection(x + 2, 301, 0.8, "motion")
            outputs.append(vision._select_boot([cyan], [moving]))
        self.assertIsNone(outputs[0])
        self.assertIsNone(outputs[1])
        self.assertEqual(outputs[2], BootDetection(120.0, 300, 1.0, "cyan", arc_score=0.0))

    def test_prompt_requires_three_frames_and_releases_after_three(self):
        vision = BootBreakerVision(VisionConfig(prompt_confirm_frames=3, prompt_release_frames=3))
        self.assertFalse(vision._debounce_prompt(True, 1.0)[0])
        self.assertFalse(vision._debounce_prompt(True, 1.0)[0])
        self.assertTrue(vision._debounce_prompt(True, 1.0)[0])
        self.assertTrue(vision._debounce_prompt(False, 0.0)[0])
        self.assertTrue(vision._debounce_prompt(False, 0.0)[0])
        self.assertFalse(vision._debounce_prompt(False, 0.0)[0])

    def test_prompt_mask_must_remain_spatially_stable(self):
        vision = BootBreakerVision(VisionConfig(prompt_min_mask_iou=0.65))
        fixed = np.zeros((55, 205), dtype=bool)
        fixed[15:35, 60:140] = True
        shifted = np.zeros_like(fixed)
        shifted[15:35, 90:170] = True
        self.assertEqual(vision._prompt_mask_stability(fixed), 0.0)
        self.assertEqual(vision._prompt_mask_stability(fixed), 1.0)
        self.assertLess(vision._prompt_mask_stability(shifted), VisionConfig().prompt_min_mask_iou)

    def test_launch_acknowledgement_releases_prompt_immediately(self):
        vision = BootBreakerVision(VisionConfig(prompt_confirm_frames=3))
        for _ in range(3):
            visible, _ = vision._debounce_prompt(True, 1.0)
        self.assertTrue(visible)
        vision.acknowledge_launch()
        self.assertFalse(vision.prompt_latched)
        self.assertFalse(vision._debounce_prompt(True, 1.0)[0])
        self.assertFalse(vision._debounce_prompt(True, 1.0)[0])
        self.assertTrue(vision._debounce_prompt(True, 1.0)[0])

    def test_launch_line_is_used_as_initial_tracking_prior(self):
        vision = BootBreakerVision(VisionConfig())
        aim = AimDetection(((300, 700), (320, 650), (340, 600), (360, 550)), -0.4, 372.0, 1.0)
        vision.acknowledge_launch(aim)
        # At y=600 the launch line expects x=340.
        on_line = BootDetection(342.0, 600.0, 1.0, "cyan", arc_score=0.0)
        moving = BootDetection(344.0, 601.0, 0.8, "motion")
        self.assertEqual(vision._select_boot([on_line], [moving]), on_line)

    def test_launch_uses_most_recent_aim_when_space_frame_has_none(self):
        vision = BootBreakerVision(VisionConfig())
        aim = AimDetection(((300, 700), (320, 650), (340, 600), (360, 550)), -0.4, 372.0, 1.0)
        vision.last_aim = aim
        vision.acknowledge_launch(None)
        self.assertEqual(vision.launch_aim, aim)

    def test_circle_arc_scores_higher_than_rectangle(self):
        vision = BootBreakerVision(VisionConfig())
        arc = np.zeros((100, 100), dtype=np.uint8)
        angles = np.linspace(-1.1, 1.1, 80)
        for angle in angles:
            x = int(round(50 + 20 * np.cos(angle)))
            y = int(round(50 + 20 * np.sin(angle)))
            arc[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 1
        rectangle = np.zeros((100, 100), dtype=np.uint8)
        rectangle[42:58, 35:65] = 1
        self.assertGreater(vision._boot_arc_score(arc, 70, 50), VisionConfig().boot_arc_min_score)
        self.assertLess(vision._boot_arc_score(rectangle, 50, 50), VisionConfig().boot_arc_min_score)

    def test_rejection_preserves_motion_baseline(self):
        vision = BootBreakerVision(VisionConfig())
        vision.previous_gray = object()
        vision.reject_boot_location(BootDetection(300, 100, 1.0, "cyan"))
        self.assertIsNotNone(vision.previous_gray)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.field = FieldConfig(left=0, right=600, paddle_y=700, brick_bottom_y=450)
        self.controller = BootBreakerController(
            self.field,
            ControlConfig(state_confirm_ms=0, space_cooldown_ms=0, auto_adjust_aim=True),
        )

    def test_positioning(self):
        vision = VisionResult(PaddleDetection(100, 120, 1), None, None, 0, True, 1.0)
        decision = self.controller.decide(1.0, vision, None, False)
        self.assertEqual(decision.phase, Phase.POSITIONING)
        self.assertEqual(decision.horizontal, 0)
        self.assertTrue(decision.press_space)

    def test_position_locks_on_first_center_frame(self):
        vision = VisionResult(PaddleDetection(300, 120, 1), None, None, 0, True, 1.0)
        decision = self.controller.decide(1.0, vision, None, False)
        self.assertEqual(decision.horizontal, 0)
        self.assertTrue(decision.press_space)

    def test_aiming(self):
        aim = AimDetection(((200, 600), (220, 550), (240, 500), (260, 450)), -0.4, 260, 1)
        vision = VisionResult(PaddleDetection(300, 120, 1), None, aim, 0, True, 1.0)
        decision = self.controller.decide(1.0, vision, None, False)
        self.assertEqual(decision.phase, Phase.AIMING)
        self.assertEqual(decision.horizontal, 0)
        self.assertTrue(decision.press_space)

    def test_default_aim_does_not_block_launch(self):
        controller = BootBreakerController(
            self.field,
            ControlConfig(state_confirm_ms=0, space_cooldown_ms=0, auto_adjust_aim=False),
        )
        aim = AimDetection(((200, 600), (220, 550), (240, 500), (260, 450)), -0.4, 260, 1)
        vision = VisionResult(PaddleDetection(300, 120, 1), None, aim, 0, True, 1.0)
        first = controller.decide(1.0, vision, None, False)
        self.assertEqual(first.horizontal, 0)
        self.assertTrue(first.press_space)

    def test_flight_intercept(self):
        vision = VisionResult(PaddleDetection(100, 120, 1), BootDetection(300, 500, 1, "test"), None, 0)
        motion = MotionState(300, 500, 100, 400, True, 5, 1.0)
        decision = self.controller.decide(1.0, vision, motion, True)
        self.assertEqual(decision.phase, Phase.IN_FLIGHT)
        self.assertIsNotNone(decision.intercept_x)
        self.assertEqual(decision.horizontal, 1)

    def test_intercept_uses_physical_contact_line(self):
        field = FieldConfig(left=0, right=600, paddle_y=730, paddle_contact_y=695, brick_bottom_y=450)
        controller = BootBreakerController(field, ControlConfig(state_confirm_ms=0))
        vision = VisionResult(PaddleDetection(100, 120, 1), BootDetection(300, 500, 1, "test"), None, 0)
        motion = MotionState(300, 500, 100, 400, True, 5, 1.0)
        decision = controller.decide(1.0, vision, motion, True)
        self.assertAlmostEqual(decision.intercept_x or 0.0, 351.55, delta=1.0)

    def test_lost_track_continues_toward_last_intercept(self):
        vision = VisionResult(PaddleDetection(100, 120, 1), BootDetection(300, 500, 1, "test"), None, 0)
        motion = MotionState(300, 500, 100, 400, True, 5, 1.0)
        tracked = self.controller.decide(1.0, vision, motion, True)
        lost_vision = VisionResult(PaddleDetection(100, 120, 1), None, None, 0)
        held = self.controller.decide(1.1, lost_vision, None, False)
        self.assertEqual(held.phase, Phase.IN_FLIGHT)
        self.assertEqual(held.reason, "hold-intercept")
        self.assertEqual(held.target_x, tracked.intercept_x)
        self.assertEqual(held.horizontal, tracked.horizontal)

    def test_lost_intercept_is_held_without_timeout(self):
        vision = VisionResult(PaddleDetection(100, 120, 1), BootDetection(300, 500, 1, "test"), None, 0)
        motion = MotionState(300, 500, 100, 400, True, 5, 1.0)
        self.controller.decide(1.0, vision, motion, True)
        lost_vision = VisionResult(PaddleDetection(100, 120, 1), None, None, 0)
        held = self.controller.decide(12.3, lost_vision, None, False)
        self.assertEqual(held.reason, "hold-intercept")
        self.assertNotEqual(held.horizontal, 0)


if __name__ == "__main__":
    unittest.main()
