from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import TrackingConfig
from .vision import BootDetection


@dataclass(frozen=True)
class MotionState:
    x: float
    y: float
    vx: float
    vy: float
    stable: bool
    sample_count: int
    last_seen: float


class BootTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.samples: deque[tuple[float, float, float]] = deque()
        self.last_seen = float("-inf")
        self.stationary_anchor: tuple[float, float] | None = None
        self.stationary_since = float("-inf")
        self.rejected_stationary = False
        self.low_motion_frames = 0
        self.rejection_reason: str | None = None

    def reset(self) -> None:
        self.samples.clear()
        self.last_seen = float("-inf")
        self.stationary_anchor = None
        self.stationary_since = float("-inf")
        self.rejected_stationary = False
        self.low_motion_frames = 0
        self.rejection_reason = None

    def is_recent(self, timestamp: float) -> bool:
        return timestamp - self.last_seen <= self.config.lost_timeout_ms / 1000.0

    def update(self, timestamp: float, detection: BootDetection | None) -> MotionState | None:
        self.rejected_stationary = False
        self.rejection_reason = None
        if detection is not None and detection.y >= self.config.max_tracking_y:
            detection = None
        if detection is not None:
            if self.samples:
                _, last_x, last_y = self.samples[-1]
                jump = float(np.hypot(detection.x - last_x, detection.y - last_y))
                elapsed = max(timestamp - self.samples[-1][0], 1e-3)
                allowed = max(self.config.max_jump_px, self.config.max_speed_px_s * elapsed * 1.35)
                if jump > allowed and self.is_recent(timestamp):
                    detection = None
            if detection is not None:
                if self.samples:
                    _, previous_x, previous_y = self.samples[-1]
                    frame_displacement = float(np.hypot(detection.x - previous_x, detection.y - previous_y))
                    in_no_brick_zone = detection.y >= self.config.no_brick_zone_y
                    if not in_no_brick_zone and frame_displacement < self.config.min_frame_displacement_px:
                        self.low_motion_frames += 1
                    else:
                        self.low_motion_frames = 0
                    if self.low_motion_frames >= self.config.low_motion_reject_frames:
                        self.reset()
                        self.rejected_stationary = True
                        self.rejection_reason = "consecutive_low_displacement"
                        return None
                if self.stationary_anchor is None:
                    self.stationary_anchor = (detection.x, detection.y)
                    self.stationary_since = timestamp
                else:
                    distance_from_anchor = float(
                        np.hypot(detection.x - self.stationary_anchor[0], detection.y - self.stationary_anchor[1])
                    )
                    if distance_from_anchor > self.config.stationary_radius_px:
                        self.stationary_anchor = (detection.x, detection.y)
                        self.stationary_since = timestamp
                    elif (
                        detection.y < self.config.no_brick_zone_y
                        and timestamp - self.stationary_since >= self.config.stationary_timeout_ms / 1000.0
                    ):
                        self.reset()
                        self.rejected_stationary = True
                        self.rejection_reason = "stationary_timeout"
                        return None
                self.samples.append((timestamp, detection.x, detection.y))
                self.last_seen = timestamp

        cutoff = timestamp - self.config.history_ms / 1000.0
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        if not self.samples or not self.is_recent(timestamp):
            return None

        vx = vy = 0.0
        filtered_x = self.samples[-1][1]
        filtered_y = self.samples[-1][2]
        stable = len(self.samples) >= self.config.min_samples
        if stable:
            absolute_times = np.array([sample[0] for sample in self.samples], dtype=np.float64)
            xs = np.array([sample[1] for sample in self.samples], dtype=np.float64)
            ys = np.array([sample[2] for sample in self.samples], dtype=np.float64)
            mean_time = float(absolute_times.mean())
            times = absolute_times - mean_time
            denominator = float(np.dot(times, times))
            if denominator > 1e-8:
                vx = float(np.dot(times, xs - xs.mean()) / denominator)
                vy = float(np.dot(times, ys - ys.mean()) / denominator)
                # Evaluate the fitted center at the current time. This rejects
                # periodic sprite-rotation wobble in the raw silhouette center.
                filtered_x = float(xs.mean() + vx * (timestamp - mean_time))
                filtered_y = float(ys.mean() + vy * (timestamp - mean_time))
            stable = float(np.hypot(vx, vy)) <= self.config.max_speed_px_s
            path_distance = sum(
                float(np.hypot(right[1] - left[1], right[2] - left[2]))
                for left, right in zip(self.samples, list(self.samples)[1:])
            )
            path_seconds = self.samples[-1][0] - self.samples[0][0]
            path_speed = path_distance / path_seconds if path_seconds > 1e-6 else 0.0
            if (
                stable
                and self.samples[-1][2] < self.config.no_brick_zone_y
                and path_speed < self.config.min_valid_speed_px_s
            ):
                self.reset()
                self.rejected_stationary = True
                self.rejection_reason = "low_path_speed"
                return None
        if filtered_y >= self.config.max_tracking_y:
            # A real boot cannot remain below the paddle. Regression can
            # otherwise extrapolate a briefly missing target far below it and
            # steer the cart toward unrelated animation there.
            self.reset()
            self.rejection_reason = "below_paddle"
            return None
        return MotionState(filtered_x, filtered_y, vx, vy, stable, len(self.samples), self.last_seen)


def reflected_x(x: float, vx: float, seconds: float, left: float, right: float) -> float:
    """Project x through any number of vertical-wall bounces."""
    width = right - left
    if width <= 0:
        return x
    unfolded = x - left + vx * seconds
    period = 2.0 * width
    folded = unfolded % period
    if folded > width:
        folded = period - folded
    return left + folded


def predict_paddle_intercept(state: MotionState, paddle_y: float, left: float, right: float, lead_seconds: float = 0.0) -> float | None:
    if state.vy <= 1e-3 or state.y >= paddle_y:
        return None
    seconds = (paddle_y - state.y) / state.vy + lead_seconds
    if seconds < 0.0 or seconds > 5.0:
        return None
    return reflected_x(state.x, state.vx, seconds, left, right)
