from __future__ import annotations

import cv2
import numpy as np

from .config import FieldConfig
from .controller import Decision
from .tracker import MotionState, reflected_x
from .vision import VisionResult
from .score import ScoreResult


def choose_window_position(
    capture_region: dict[str, int],
    desktop_bounds: dict[str, int],
    window_width: int,
    window_height: int,
    margin: int = 10,
) -> tuple[int, int]:
    """Place debug beside the capture so it cannot contaminate screenshots."""
    desktop_left = desktop_bounds["left"]
    desktop_top = desktop_bounds["top"]
    desktop_right = desktop_left + desktop_bounds["width"]
    desktop_bottom = desktop_top + desktop_bounds["height"]
    capture_left = capture_region["left"]
    capture_top = capture_region["top"]
    capture_right = capture_left + capture_region["width"]
    capture_bottom = capture_top + capture_region["height"]
    candidates = [
        (capture_right + margin, capture_top),
        (capture_left - window_width - margin, capture_top),
        (capture_left, capture_bottom + margin),
        (capture_left, capture_top - window_height - margin),
    ]
    for x, y in candidates:
        if (
            desktop_left <= x
            and x + window_width <= desktop_right
            and desktop_top <= y
            and y + window_height <= desktop_bottom
        ):
            return x, y
    return (
        min(max(capture_right + margin, desktop_left), max(desktop_left, desktop_right - window_width)),
        min(max(capture_top, desktop_top), max(desktop_top, desktop_bottom - window_height)),
    )


def draw_debug(
    frame: np.ndarray,
    field: FieldConfig,
    vision: VisionResult,
    motion: MotionState | None,
    decision: Decision,
    score: ScoreResult,
    armed: bool,
    fps: float,
) -> np.ndarray:
    canvas = frame.copy()
    if vision.paddle is not None:
        half = vision.paddle.width / 2.0
        cv2.rectangle(
            canvas,
            (int(vision.paddle.center_x - half), int(field.paddle_y - 14)),
            (int(vision.paddle.center_x + half), int(field.paddle_y + 14)),
            (30, 255, 30),
            2,
        )
    if vision.aim is not None:
        points = np.array(vision.aim.points, dtype=np.int32)
        for x, y in points:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (int(vision.aim.top_x), int(vision.aim_band_top) if hasattr(vision, "aim_band_top") else 520), (int(points[-1][0]), int(points[-1][1])), (0, 255, 255), 1)
    if motion is not None:
        color = (255, 255, 255) if motion.stable else (150, 150, 150)
        cv2.circle(canvas, (int(motion.x), int(motion.y)), 16, color, 2, cv2.LINE_AA)
        cv2.line(canvas, (int(motion.x), int(motion.y)), (int(motion.x + motion.vx * 0.12), int(motion.y + motion.vy * 0.12)), color, 2, cv2.LINE_AA)
    if decision.target_x is not None:
        if decision.reason == "intercept":
            target_label = "PREDICTED LANDING"
            target_color = (255, 220, 0)
        elif decision.reason == "hold-intercept":
            target_label = "HELD LANDING"
            target_color = (0, 165, 255)
        else:
            target_label = "LIVE X"
            target_color = (255, 255, 0)
        target_x = int(decision.target_x)
        cv2.line(canvas, (target_x, int(field.paddle_y - 30)), (target_x, int(field.paddle_y + 30)), target_color, 2)
        cv2.putText(
            canvas,
            target_label,
            (max(4, min(target_x - 55, canvas.shape[1] - 160)), int(field.paddle_y - 36)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            target_color,
            1,
            cv2.LINE_AA,
        )
    if decision.reason == "intercept" and decision.intercept_x is not None and motion is not None and motion.vy > 0.0:
        seconds = (field.paddle_contact_y - motion.y) / motion.vy
        if 0.0 < seconds <= 5.0:
            points = []
            for step in range(25):
                t = seconds * step / 24.0
                x = reflected_x(motion.x, motion.vx, t, field.left, field.right)
                y = motion.y + motion.vy * t
                points.append((int(x), int(y)))
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, (255, 200, 0), 1, cv2.LINE_AA)

    velocity = (motion.vx, motion.vy) if motion else (0.0, 0.0)
    status = "ARMED" if armed else "LOCKED"
    direction = {-1: "LEFT", 0: "STOP", 1: "RIGHT"}[decision.horizontal]
    lines = [
        f"{status} | {decision.phase.value.upper()} | {decision.reason} | {fps:5.1f} FPS | score={score.value} delta={score.delta}",
        f"boot=({motion.x:6.1f},{motion.y:6.1f}) v=({velocity[0]:+6.0f},{velocity[1]:+6.0f})" if motion else "boot=LOST",
        f"paddle={vision.paddle.center_x:6.1f}" if vision.paddle else "paddle=LOST",
        f"action={direction} space={decision.press_space} motion_px={vision.motion_pixels}",
        f"launch_prompt={vision.prompt_visible} conf={vision.prompt_confidence:.2f} stable={vision.prompt_stability:.2f}",
        "F8 arm/disarm | F9 pause | ESC quit",
    ]
    for index, text in enumerate(lines):
        color = (40, 60, 255) if index == 0 and armed else (235, 235, 235)
        cv2.putText(canvas, text, (10, 22 + index * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return canvas
