from __future__ import annotations

from dataclasses import dataclass

from .config import ControlConfig
from .geometry import AngularZone, forward_distance, norm_angle
from .tracker import MotionState


@dataclass(frozen=True)
class Decision:
    click: bool
    desired_rmb: bool
    predicted_angle: float | None
    target: AngularZone | None
    margin_deg: float
    reason: str


class PredictiveController:
    def __init__(self, config: ControlConfig):
        self.config = config
        self.last_click_time = float("-inf")
        self.awaiting_feedback = False
        self.feedback_direction = 0

    def reset(self) -> None:
        self.last_click_time = float("-inf")
        self.awaiting_feedback = False
        self.feedback_direction = 0

    def _margin(self, speed: float) -> float:
        jitter_s = self.config.latency_jitter_ms / 1000.0
        return self.config.base_margin_deg + abs(speed) * jitter_s + abs(speed) * self.config.speed_error_fraction * (self.config.input_latency_ms / 1000.0)

    @staticmethod
    def _value(zone: AngularZone, distance: float, blue_value: float) -> float:
        reward = blue_value if zone.kind == "blue" else 100.0
        return reward * zone.confidence - 0.08 * distance

    def decide(self, timestamp: float, state: MotionState | None, zones: tuple[AngularZone, ...]) -> Decision:
        if state is None or not state.stable:
            return Decision(False, False, None, None, 0.0, "tracking")

        if self.awaiting_feedback:
            direction_changed = state.direction != self.feedback_direction and abs(state.speed_dps) > 20.0
            timed_out = timestamp - self.last_click_time >= self.config.feedback_timeout_ms / 1000.0
            if direction_changed or timed_out:
                self.awaiting_feedback = False
            else:
                return Decision(False, False, state.angle, None, 0.0, "feedback")

        latency_s = self.config.input_latency_ms / 1000.0
        predicted = norm_angle(state.angle + state.speed_dps * latency_s)
        margin = self._margin(state.speed_dps)
        candidates: list[tuple[float, float, AngularZone]] = []
        for zone in zones:
            if zone.width <= 2.0 * margin:
                continue
            distance = forward_distance(predicted, zone.entry(state.direction, margin), state.direction)
            candidates.append((self._value(zone, distance, self.config.blue_time_value_points), distance, zone))
        if not candidates:
            return Decision(False, self.config.rmb_accel_enabled, predicted, None, margin, "no-zone")

        immediate = [item for item in candidates if item[2].contains(predicted, margin)]
        if immediate:
            _, distance, target = max(immediate, key=lambda item: item[0])
        else:
            _, distance, target = max(candidates, key=lambda item: item[0])
        safe_speed = abs(state.speed_dps) <= self.config.max_safe_speed_dps
        cooldown_done = timestamp - self.last_click_time >= self.config.click_cooldown_ms / 1000.0
        inside = target.contains(predicted, margin)
        click = safe_speed and cooldown_done and inside
        if click:
            self.last_click_time = timestamp
            self.awaiting_feedback = True
            self.feedback_direction = state.direction

        desired_rmb = bool(
            self.config.rmb_accel_enabled
            and distance > self.config.rmb_release_distance_deg
            and abs(state.speed_dps) < self.config.rmb_max_accel_speed_dps
            and not click
        )
        reason = "hit" if click else ("overspeed" if not safe_speed else "approach")
        return Decision(click, desired_rmb, predicted, target, margin, reason)
