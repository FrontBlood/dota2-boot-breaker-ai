from __future__ import annotations

from dataclasses import dataclass


def norm_angle(angle: float) -> float:
    return angle % 360.0


def signed_delta(start: float, end: float) -> float:
    """Shortest signed clockwise-positive delta from start to end."""
    return (end - start + 180.0) % 360.0 - 180.0


def forward_distance(start: float, target: float, direction: int) -> float:
    if direction >= 0:
        return (target - start) % 360.0
    return (start - target) % 360.0


@dataclass(frozen=True)
class AngularZone:
    start: float
    end: float
    kind: str
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return (self.end - self.start) % 360.0

    @property
    def center(self) -> float:
        return norm_angle(self.start + self.width / 2.0)

    def contains(self, angle: float, margin: float = 0.0) -> bool:
        usable = self.width - 2.0 * margin
        if usable <= 0.0:
            return False
        return (norm_angle(angle) - norm_angle(self.start + margin)) % 360.0 <= usable

    def entry(self, direction: int, margin: float = 0.0) -> float:
        return norm_angle(self.start + margin if direction >= 0 else self.end - margin)
