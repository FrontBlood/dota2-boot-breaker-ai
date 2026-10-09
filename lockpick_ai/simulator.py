from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np

from .config import AppConfig, load_config
from .controller import PredictiveController
from .debug_view import draw_debug
from .geometry import AngularZone, norm_angle
from .tracker import AngleTracker
from .vision import DialVision


def _hsv_color(h: int, s: int, v: int) -> tuple[int, int, int]:
    pixel = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _annular_sector(center: tuple[float, float], inner: float, outer: float, start: float, end: float) -> np.ndarray:
    width = (end - start) % 360.0
    count = max(8, int(width * 2))
    angles = np.radians(np.linspace(start, start + width, count))
    cx, cy = center
    outer_points = np.c_[cx + outer * np.cos(angles), cy + outer * np.sin(angles)]
    inner_points = np.c_[cx + inner * np.cos(angles[::-1]), cy + inner * np.sin(angles[::-1])]
    return np.round(np.vstack((outer_points, inner_points))).astype(np.int32)


def render_frame(config: AppConfig, angle: float, zones: tuple[AngularZone, ...]) -> np.ndarray:
    height, width = config.roi.height, config.roi.width
    frame = np.full((height, width, 3), (16, 15, 20), dtype=np.uint8)
    dial = config.dial
    center = (dial.center_x, dial.center_y)
    cv2.circle(frame, (int(center[0]), int(center[1])), int(dial.radius), (42, 39, 45), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(center[0]), int(center[1])), int(dial.radius * 0.53), (52, 48, 50), -1, cv2.LINE_AA)
    colors = {"yellow": _hsv_color(20, 195, 105), "blue": _hsv_color(108, 135, 130)}
    for zone in zones:
        polygon = _annular_sector(center, dial.radius * 0.57, dial.radius * 0.93, zone.start, zone.end)
        cv2.fillPoly(frame, [polygon], colors[zone.kind], cv2.LINE_AA)
    radians = math.radians(angle)
    p1 = (int(center[0] + dial.radius * 0.54 * math.cos(radians)), int(center[1] + dial.radius * 0.54 * math.sin(radians)))
    p2 = (int(center[0] + dial.radius * 0.95 * math.cos(radians)), int(center[1] + dial.radius * 0.95 * math.sin(radians)))
    cv2.line(frame, p1, p2, _hsv_color(120, 160, 255), 7, cv2.LINE_AA)
    cv2.line(frame, p1, p2, (255, 245, 245), 2, cv2.LINE_AA)
    return frame


def run_simulation(config_path: str, seconds: float, headless: bool = False) -> dict[str, float]:
    config = load_config(config_path)
    detector = DialVision(config.dial, config.vision)
    tracker = AngleTracker(config.tracking)
    controller = PredictiveController(config.control)
    zones = (AngularZone(25.0, 42.0, "yellow"), AngularZone(205.0, 226.0, "blue"))
    angle = 320.0
    speed = 130.0
    start = previous = time.perf_counter()
    frames = detections = clicks = 0
    while time.perf_counter() - start < seconds:
        now = time.perf_counter()
        dt = min(now - previous, 0.05)
        previous = now
        angle = norm_angle(angle + speed * dt)
        frame = render_frame(config, angle, zones)
        result = detector.detect(frame)
        state = tracker.update(now, result.pointer_angle)
        decision = controller.decide(now, state, result.zones)
        if result.pointer_angle is not None:
            detections += 1
        if decision.click:
            clicks += 1
            speed = -speed
        frames += 1
        if not headless:
            debug = draw_debug(frame, config.dial, result, state, decision, False, frames / max(now - start, 1e-6))
            cv2.imshow("Lockpick AI - simulator", debug)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cv2.destroyAllWindows()
    return {"frames": float(frames), "detection_rate": detections / max(frames, 1), "clicks": float(clicks)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic lockpick detector/controller smoke test")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    print(run_simulation(args.config, args.seconds, args.headless))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
