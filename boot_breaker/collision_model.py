from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollisionModel:
    intercept_deg: float = 0.0
    impact_gain_deg: float = 52.0
    max_abs_angle_deg: float = 52.0

    def outgoing_angle(self, impact_offset: float) -> float:
        angle = self.intercept_deg + self.impact_gain_deg * impact_offset
        return max(-self.max_abs_angle_deg, min(self.max_abs_angle_deg, angle))

    def impact_offset_for_angle(self, desired_angle_deg: float) -> float:
        if abs(self.impact_gain_deg) < 1e-9:
            return 0.0
        offset = (desired_angle_deg - self.intercept_deg) / self.impact_gain_deg
        return max(-1.0, min(1.0, offset))
