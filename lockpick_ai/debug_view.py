from __future__ import annotations

import math

import cv2
import numpy as np

from .config import DialConfig
from .controller import Decision
from .geometry import AngularZone
from .tracker import MotionState
from .vision import VisionResult


COLORS = {"yellow": (0, 220, 255), "blue": (255, 150, 40)}


def _point(dial: DialConfig, angle: float, radius: float) -> tuple[int, int]:
    radians = math.radians(angle)
    return int(dial.center_x + radius * math.cos(radians)), int(dial.center_y + radius * math.sin(radians))


def _arc(image: np.ndarray, dial: DialConfig, zone: AngularZone, radius: int, color: tuple[int, int, int], thickness: int) -> None:
    start = zone.start
    end = zone.start + zone.width
    cv2.ellipse(image, (int(dial.center_x), int(dial.center_y)), (radius, radius), 0, start, end, color, thickness, cv2.LINE_AA)


def draw_debug(
    frame: np.ndarray,
    dial: DialConfig,
    vision: VisionResult,
    state: MotionState | None,
    decision: Decision,
    armed: bool,
    fps: float,
) -> np.ndarray:
    canvas = frame.copy()
    center = (int(dial.center_x), int(dial.center_y))
    cv2.circle(canvas, center, int(dial.radius), (90, 90, 90), 1, cv2.LINE_AA)
    for zone in vision.zones:
        color = COLORS.get(zone.kind, (255, 255, 255))
        _arc(canvas, dial, zone, int(dial.radius * 0.89), color, 8)
        if zone.width > 2.0 * decision.margin_deg:
            safe = AngularZone(zone.start + decision.margin_deg, zone.end - decision.margin_deg, zone.kind)
            _arc(canvas, dial, safe, int(dial.radius * 0.82), color, 2)

    if vision.pointer_angle is not None:
        cv2.line(canvas, center, _point(dial, vision.pointer_angle, dial.radius * 0.95), (255, 255, 255), 2, cv2.LINE_AA)
    if decision.predicted_angle is not None:
        cv2.line(canvas, center, _point(dial, decision.predicted_angle, dial.radius * 0.72), (70, 255, 70), 2, cv2.LINE_AA)

    speed = state.speed_dps if state else 0.0
    stable = state.stable if state else False
    status = "ARMED" if armed else "LOCKED"
    color = (40, 60, 255) if armed else (180, 180, 180)
    lines = [
        f"{status}  {decision.reason.upper()}  {fps:5.1f} FPS",
        f"angle={vision.pointer_angle if vision.pointer_angle is not None else -1:6.1f}  conf={vision.pointer_confidence:.2f}",
        f"speed={speed:+7.1f} deg/s  {'TRACK' if stable else 'WARMUP'}  zones={len(vision.zones)}",
        "F8 arm/disarm | F9 pause | ESC quit",
    ]
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (10, 24 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color if index == 0 else (230, 230, 230), 1, cv2.LINE_AA)
    return canvas
