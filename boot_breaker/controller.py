from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections import deque
import statistics

from .config import ControlConfig, FieldConfig
from .tracker import MotionState, predict_paddle_intercept
from .vision import VisionResult


class Phase(str, Enum):
    SEARCHING = "searching"
    POSITIONING = "positioning"
    AIMING = "aiming"
    IN_FLIGHT = "in_flight"


@dataclass(frozen=True)
class Decision:
    phase: Phase
    horizontal: int
    press_space: bool
    target_x: float | None
    intercept_x: float | None
    reason: str


class BootBreakerController:
    def __init__(self, field: FieldConfig, config: ControlConfig):
        self.field = field
        self.config = config
        self.phase = Phase.SEARCHING
        self.phase_since = 0.0
        self.aligned_since: float | None = None
        self.last_space = float("-inf")
        self.space_sent_in_phase = False
        self.launch_direction = 1
        self.intercept_history: deque[float] = deque(maxlen=5)
        self.last_intercept_x: float | None = None
        self.last_intercept_at = float("-inf")

    def reset(self) -> None:
        self.phase = Phase.SEARCHING
        self.phase_since = 0.0
        self.aligned_since = None
        self.last_space = float("-inf")
        self.space_sent_in_phase = False
        self.intercept_history.clear()
        self.last_intercept_x = None
        self.last_intercept_at = float("-inf")

    def _set_phase(self, phase: Phase, timestamp: float) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_since = timestamp
            self.aligned_since = None
            self.space_sent_in_phase = False
            if phase != Phase.IN_FLIGHT:
                self.intercept_history.clear()
                self.last_intercept_x = None
                self.last_intercept_at = float("-inf")

    def _held_intercept(self, timestamp: float, vision: VisionResult) -> Decision | None:
        if vision.paddle is None or self.last_intercept_x is None:
            return None
        hold_seconds = self.config.lost_prediction_hold_ms / 1000.0
        if hold_seconds > 0.0 and timestamp - self.last_intercept_at > hold_seconds:
            return None
        horizontal = self._horizontal(vision.paddle.center_x, self.last_intercept_x)
        return Decision(
            self.phase,
            horizontal,
            False,
            self.last_intercept_x,
            self.last_intercept_x,
            "hold-intercept",
        )

    def _space_ready(self, timestamp: float) -> bool:
        cooldown = self.config.space_cooldown_ms / 1000.0
        retry = self.config.space_retry_ms / 1000.0
        if timestamp - self.last_space < cooldown:
            return False
        return not self.space_sent_in_phase or timestamp - self.last_space >= retry

    def _horizontal(self, current: float, target: float) -> int:
        error = target - current
        if abs(error) <= self.config.paddle_deadzone_px:
            return 0
        return 1 if error > 0.0 else -1

    def _space_when_aligned(self, timestamp: float, aligned: bool) -> bool:
        if not aligned:
            self.aligned_since = None
            return False
        if self.aligned_since is None:
            self.aligned_since = timestamp
            return False
        held = timestamp - self.aligned_since >= self.config.state_confirm_ms / 1000.0
        cooled = self._space_ready(timestamp)
        if held and cooled and self.config.auto_launch:
            self.last_space = timestamp
            self.space_sent_in_phase = True
            self.aligned_since = None
            return True
        return False

    def _space_immediately(self, timestamp: float) -> bool:
        cooled = self._space_ready(timestamp)
        if cooled and self.config.auto_launch:
            self.last_space = timestamp
            self.space_sent_in_phase = True
            self.aligned_since = None
            return True
        return False

    def _space_on_prompt(self, timestamp: float) -> bool:
        """Prompt-only launch policy: retry at the normal input cooldown."""
        cooled = timestamp - self.last_space >= self.config.space_cooldown_ms / 1000.0
        if cooled and self.config.auto_launch:
            self.last_space = timestamp
            self.space_sent_in_phase = True
            self.aligned_since = None
            return True
        return False

    def decide(self, timestamp: float, vision: VisionResult, motion: MotionState | None, boot_recent: bool) -> Decision:
        if vision.prompt_visible and vision.aim is not None:
            self._set_phase(Phase.AIMING, timestamp)
        elif vision.prompt_visible and vision.paddle is not None:
            self._set_phase(Phase.POSITIONING, timestamp)
        elif motion is not None or boot_recent or self.phase == Phase.IN_FLIGHT:
            self._set_phase(Phase.IN_FLIGHT, timestamp)
        else:
            self._set_phase(Phase.SEARCHING, timestamp)

        if vision.prompt_visible and vision.paddle is not None:
            space = self._space_on_prompt(timestamp)
            return Decision(self.phase, 0, space, None, None, "prompt-space")

        if self.phase == Phase.POSITIONING and vision.paddle is not None:
            target = self.field.left + (self.field.right - self.field.left) * self.config.launch_position_ratio
            horizontal = self._horizontal(vision.paddle.center_x, target)
            # The cart moves far enough per frame that requiring a long stable
            # dwell causes endless overshoot. Lock on the first center crossing.
            space = self._space_immediately(timestamp) if horizontal == 0 else False
            return Decision(self.phase, horizontal, space, target, None, "center-cart" if horizontal else "lock-cart")

        if self.phase == Phase.AIMING and vision.aim is not None and vision.paddle is not None:
            desired_top = vision.paddle.center_x + self.launch_direction * self.config.launch_target_dx
            if not self.config.auto_adjust_aim:
                space = self._space_when_aligned(timestamp, True)
                return Decision(self.phase, 0, space, vision.aim.top_x, None, "launch-default")
            error = desired_top - vision.aim.top_x
            horizontal = 0 if abs(error) <= self.config.launch_aim_tolerance_px else (1 if error > 0 else -1)
            space = self._space_when_aligned(timestamp, horizontal == 0)
            return Decision(self.phase, horizontal, space, desired_top, None, "adjust-aim" if horizontal else "launch")

        if self.phase == Phase.IN_FLIGHT:
            self.aligned_since = None
            if motion is None or vision.paddle is None:
                held = self._held_intercept(timestamp, vision)
                return held or Decision(self.phase, 0, False, None, None, "track-lost")
            if not motion.stable:
                held = self._held_intercept(timestamp, vision)
                return held or Decision(self.phase, 0, False, None, None, "track-warmup")
            intercept = None
            if motion.vy > 0.0 and motion.y >= self.field.brick_bottom_y:
                raw_intercept = predict_paddle_intercept(
                    motion,
                    self.field.paddle_contact_y,
                    self.field.left,
                    self.field.right,
                    self.config.prediction_lead_ms / 1000.0,
                )
                if raw_intercept is not None:
                    self.intercept_history.append(raw_intercept)
                    intercept = float(statistics.median(self.intercept_history))
                    self.last_intercept_x = intercept
                    self.last_intercept_at = timestamp
            else:
                self.intercept_history.clear()
            target = intercept if intercept is not None else motion.x
            horizontal = self._horizontal(vision.paddle.center_x, target)
            return Decision(self.phase, horizontal, False, target, intercept, "intercept" if intercept is not None else "follow")

        return Decision(self.phase, 0, False, None, None, "search")
