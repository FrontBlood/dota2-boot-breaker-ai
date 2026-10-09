from __future__ import annotations

import unittest

from boot_breaker.config import FieldConfig
from boot_breaker.controller import Phase
from boot_breaker.reward import RewardClassifier
from boot_breaker.score import ScoreResult
from boot_breaker.tracker import MotionState


def score(delta: int | None, changed: bool = True) -> ScoreResult:
    return ScoreResult(None, delta, changed, 1.0, "x")


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.classifier = RewardClassifier(FieldConfig(top_exit_y=55))

    def test_block_types(self):
        self.assertEqual(self.classifier.update(1.0, score(5), None, Phase.IN_FLIGHT).kind, "normal_block")
        self.assertEqual(self.classifier.update(2.0, score(50), None, Phase.IN_FLIGHT).kind, "reward_block")

    def test_top_exit_completion(self):
        motion = MotionState(300, 40, 20, -400, True, 5, 1.0)
        pending = self.classifier.update(1.0, score(None, False), motion, Phase.IN_FLIGHT)
        self.assertTrue(pending.top_exit_pending)
        completed = self.classifier.update(1.4, score(500), None, Phase.POSITIONING, True)
        self.assertEqual(completed.kind, "level_complete")
        self.assertEqual(completed.value, 500)

    def test_top_exit_candidate_does_not_confirm_completion(self):
        motion = MotionState(300, 40, 20, -400, True, 5, 1.0)
        pending = self.classifier.update(1.0, score(None, False), motion, Phase.IN_FLIGHT)
        self.assertTrue(pending.top_exit_pending)

    def test_unknown_score_change_during_flight_is_not_completion(self):
        motion = MotionState(300, 40, 20, -400, True, 5, 1.0)
        self.classifier.update(1.0, score(None, False), motion, Phase.IN_FLIGHT)
        event = self.classifier.update(1.2, score(None), None, Phase.IN_FLIGHT, False)
        self.assertNotEqual(event.kind, "level_complete")

    def test_next_launch_prompt_confirms_completion_without_ocr_delta(self):
        motion = MotionState(300, 40, 20, -400, True, 5, 1.0)
        self.classifier.update(1.0, score(None, False), motion, Phase.IN_FLIGHT)
        event = self.classifier.update(1.2, score(None, False), None, Phase.POSITIONING, True)
        self.assertEqual(event.kind, "level_complete")
        self.assertFalse(event.top_exit_pending)


if __name__ == "__main__":
    unittest.main()
