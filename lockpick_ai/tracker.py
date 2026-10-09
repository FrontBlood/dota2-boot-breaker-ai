from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import TrackingConfig
from .geometry import norm_angle, signed_delta


@dataclass(frozen=True)
class MotionState:
    angle: float
    speed_dps: float
    direction: int
    stable: bool
    sample_count: int


class AngleTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.samples: deque[tuple[float, float]] = deque()
        self._last_raw: float | None = None
        self._unwrapped = 0.0

    def reset(self) -> None:
        self.samples.clear()
        self._last_raw = None
        self._unwrapped = 0.0

    def update(self, timestamp: float, angle: float | None) -> MotionState | None:
        if angle is None:
            return None
        angle = norm_angle(angle)
        if self._last_raw is None:
            self._unwrapped = angle
        else:
            delta = signed_delta(self._last_raw, angle)
            if abs(delta) > self.config.max_jump_deg:
                self.reset()
                self._unwrapped = angle
            else:
                self._unwrapped += delta
        self._last_raw = angle
        self.samples.append((timestamp, self._unwrapped))
        cutoff = timestamp - self.config.history_ms / 1000.0
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

        speed = 0.0
        stable = len(self.samples) >= self.config.min_samples
        if stable:
            times = np.array([x[0] for x in self.samples], dtype=np.float64)
            values = np.array([x[1] for x in self.samples], dtype=np.float64)
            times -= times.mean()
            denominator = float(np.dot(times, times))
            if denominator > 1e-8:
                speed = float(np.dot(times, values - values.mean()) / denominator)
            stable = abs(speed) <= self.config.max_abs_speed_dps
        direction = 1 if speed >= 0.0 else -1
        return MotionState(angle, speed, direction, stable, len(self.samples))
