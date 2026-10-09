from __future__ import annotations

from dataclasses import dataclass

from .config import FieldConfig
from .controller import Phase
from .score import ScoreResult
from .tracker import MotionState


@dataclass(frozen=True)
class RewardEvent:
    kind: str | None
    value: int
    top_exit_pending: bool


class RewardClassifier:
    """Classify score changes using game rules and top-exit trajectory context."""

    def __init__(self, field: FieldConfig):
        self.field = field
        self.top_exit_until = float("-inf")

    def reset(self) -> None:
        self.top_exit_until = float("-inf")

    def update(
        self,
        timestamp: float,
        score: ScoreResult,
        motion: MotionState | None,
        phase: Phase,
        launch_prompt: bool = False,
    ) -> RewardEvent:
        if motion is not None and motion.stable and motion.vy < -40.0 and motion.y <= self.field.top_exit_y:
            self.top_exit_until = timestamp + 1.5
        pending = timestamp <= self.top_exit_until
        if pending and launch_prompt and phase in (Phase.POSITIONING, Phase.AIMING):
            self.top_exit_until = float("-inf")
            return RewardEvent("level_complete", score.delta if score.delta is not None else 500, False)
        if not score.changed:
            return RewardEvent(None, 0, pending)

        delta = score.delta
        if pending and delta is not None and delta >= 500:
            self.top_exit_until = float("-inf")
            return RewardEvent("level_complete", delta if delta is not None else 500, False)
        if delta == 5:
            return RewardEvent("normal_block", 5, pending)
        if delta == 50:
            return RewardEvent("reward_block", 50, pending)
        if delta is not None and delta > 0:
            return RewardEvent("composite_score", delta, pending)
        return RewardEvent("score_change_unknown", 1, pending)
