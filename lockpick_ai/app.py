from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .capture import ScreenCapture
from .config import load_config
from .controller import PredictiveController
from .debug_view import draw_debug
from .input_windows import HotkeyEdges, MouseOutput, VK_ESCAPE, VK_F8, VK_F9
from .telemetry import TelemetryWriter
from .tracker import AngleTracker
from .vision import DialVision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dota 2 lockpick visual controller")
    default_config = Path(__file__).resolve().parent.parent / "config.json"
    parser.add_argument("--config", default=str(default_config), help="configuration JSON path")
    parser.add_argument("--control", action="store_true", help="allow F8 to arm real mouse output")
    parser.add_argument("--no-window", action="store_true", help="disable OpenCV debug window")
    parser.add_argument("--smoke-test", action="store_true", help="capture and analyze one frame, then exit")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    config = load_config(args.config)
    vision = DialVision(config.dial, config.vision)
    if args.smoke_test:
        with ScreenCapture(config.roi) as capture:
            result = vision.detect(capture.grab())
        print(
            "Smoke test OK:",
            f"pointer={result.pointer_angle}",
            f"confidence={result.pointer_confidence:.3f}",
            f"zones={len(result.zones)}",
        )
        return 0
    tracker = AngleTracker(config.tracking)
    controller = PredictiveController(config.control)
    hotkeys = HotkeyEdges()
    mouse = MouseOutput()
    armed = False
    paused = False
    last_frame_time = time.perf_counter()
    fps = 0.0

    print("观察模式已启动。F8 解锁/锁定，F9 暂停，Esc 退出。")
    if not args.control:
        print("未指定 --control：F8 只改变界面状态，不会发送鼠标输入。")

    try:
        with ScreenCapture(config.roi) as capture, TelemetryWriter(config.runtime.record_telemetry) as telemetry:
            while True:
                now = time.perf_counter()
                if hotkeys.pressed(VK_ESCAPE):
                    break
                if hotkeys.pressed(VK_F9):
                    paused = not paused
                    armed = False
                    mouse.release_all()
                    tracker.reset()
                    controller.reset()
                if hotkeys.pressed(VK_F8):
                    armed = not armed and not paused
                    if not armed:
                        mouse.release_all()

                frame = capture.grab()
                result = vision.detect(frame)
                state = tracker.update(now, result.pointer_angle) if not paused else None
                decision = controller.decide(now, state, result.zones)

                output_enabled = bool(args.control and armed and not paused)
                if output_enabled and state is not None and state.stable:
                    mouse.set_right(decision.desired_rmb)
                    if decision.click:
                        mouse.click_left()
                else:
                    mouse.release_all()

                elapsed = max(now - last_frame_time, 1e-6)
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else fps * 0.9 + instant_fps * 0.1
                last_frame_time = now
                telemetry.write({
                    "t": now,
                    "angle": result.pointer_angle,
                    "pointer_confidence": result.pointer_confidence,
                    "speed_dps": state.speed_dps if state else None,
                    "stable": state.stable if state else False,
                    "zones": [{"start": z.start, "end": z.end, "kind": z.kind, "confidence": z.confidence} for z in result.zones],
                    "predicted": decision.predicted_angle,
                    "reason": decision.reason,
                    "click": bool(output_enabled and decision.click),
                    "rmb": bool(output_enabled and decision.desired_rmb),
                    "armed": output_enabled,
                })

                if config.runtime.show_debug and not args.no_window:
                    debug = draw_debug(frame, config.dial, result, state, decision, armed and args.control, fps)
                    if config.runtime.debug_scale != 1.0:
                        debug = cv2.resize(debug, None, fx=config.runtime.debug_scale, fy=config.runtime.debug_scale)
                    cv2.imshow("Lockpick AI - debug", debug)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if config.runtime.idle_sleep_ms > 0:
                    time.sleep(config.runtime.idle_sleep_ms / 1000.0)
    finally:
        mouse.release_all()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
