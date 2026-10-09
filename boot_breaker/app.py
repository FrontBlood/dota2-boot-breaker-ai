from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

from lockpick_ai.capture import ScreenCapture
from lockpick_ai.input_windows import HotkeyEdges, KeyboardOutput, VK_ESCAPE, VK_F8, VK_F9
from lockpick_ai.telemetry import TelemetryWriter

from .config import load_config
from .controller import BootBreakerController
from .debug_view import choose_window_position, draw_debug
from .score import ScoreMonitor
from .reward import RewardClassifier
from .tracker import BootTracker
from .vision import BootBreakerVision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dota 2 Boot Breaker visual controller")
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).resolve().parent
        os.chdir(app_dir)
        external_config = app_dir / "boot_config.json"
        bundled_config = Path(getattr(sys, "_MEIPASS", app_dir)) / "boot_config.json"
        default_config = external_config if external_config.exists() else bundled_config
    else:
        default_config = Path(__file__).resolve().parent.parent / "boot_config.json"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--control", action="store_true", help="allow F8 to arm keyboard output")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    config = load_config(args.config)
    vision = BootBreakerVision(config.vision)
    score_monitor = ScoreMonitor(config.score)
    reward_classifier = RewardClassifier(config.field)
    capture_options = {}
    if config.display.auto_scale:
        capture_options = {
            "reference_size": (config.display.reference_width, config.display.reference_height),
            "aspect_tolerance": config.display.aspect_tolerance,
            "auto_content_align": config.display.auto_content_align,
            "window_executable": config.display.window_executable,
            "require_window": config.display.require_window,
            "window_refresh_seconds": config.display.window_refresh_seconds,
        }
    if args.smoke_test:
        with ScreenCapture(config.roi, **capture_options) as capture:
            full_frame = capture.grab()
            playfield = full_frame[config.vision.playfield_top :]
            result = vision.detect(playfield)
            score = score_monitor.update(full_frame)
        print("Smoke test OK:", f"paddle={result.paddle}", f"boot={result.boot}", f"aim={result.aim is not None}", f"score={score.value}")
        return 0

    tracker = BootTracker(config.tracking)
    controller = BootBreakerController(config.field, config.control)
    keyboard = KeyboardOutput(config.control.key_mode)
    hotkeys = HotkeyEdges()
    armed = False
    paused = False
    fps = 0.0
    previous_time = time.perf_counter()
    tracking_allowed = True
    prompt_cycle_seen = False
    prompt_visible_last = False
    flight_seen_since_prompt = False
    unprotected_since: float | None = None
    max_unprotected_ms_since_launch = 0.0
    level_completion_waiting_prompt = False
    debug_window_initialized = False
    debug_window_region: tuple[int, int, int, int] | None = None
    debug_window_name = "Boot Breaker AI - debug"
    score_samples = Path("boot_runs") / "score_samples"
    if config.runtime.record_score_samples:
        score_samples.mkdir(parents=True, exist_ok=True)
    print("破牢之靴观察模式已启动。F8 解锁/锁定，F9 暂停，Esc 退出。")
    if not args.control:
        print("未指定 --control：不会发送方向键或空格。")

    try:
        with ScreenCapture(config.roi, **capture_options) as capture, TelemetryWriter(config.runtime.record_telemetry, "boot_runs") as telemetry:
            print("Capture target:", capture.target_description)
            while True:
                now = time.perf_counter()
                keyboard.update(now)
                if hotkeys.pressed(VK_ESCAPE):
                    break
                if hotkeys.pressed(VK_F9):
                    paused = not paused
                    armed = False
                    keyboard.release_all()
                    tracker.reset()
                    controller.reset()
                    reward_classifier.reset()
                    level_completion_waiting_prompt = False
                if hotkeys.pressed(VK_F8):
                    armed = not armed and not paused
                    if not armed:
                        keyboard.release_all()

                capture_start = time.perf_counter()
                full_frame = capture.grab()
                capture_ms = (time.perf_counter() - capture_start) * 1000.0
                if not capture.target_available:
                    armed = False
                    keyboard.release_all()
                    tracker.reset()
                    controller.reset()
                frame = full_frame[config.vision.playfield_top :]
                vision_start = time.perf_counter()
                detected = vision.detect(frame, tracking_allowed)
                prompt_rising = detected.prompt_visible and not prompt_visible_last
                vision_ms = (time.perf_counter() - vision_start) * 1000.0
                score_start = time.perf_counter()
                score = score_monitor.update(full_frame)
                score_ms = (time.perf_counter() - score_start) * 1000.0
                if score.changed and config.runtime.record_score_samples:
                    sample = score_monitor.sample_image()
                    if sample is not None:
                        sample_name = f"{int(now * 1000)}_score-{score.value}_conf-{score.confidence:.2f}.png"
                        cv2.imwrite(str(score_samples / sample_name), sample)
                if detected.prompt_visible:
                    tracking_allowed = False
                    prompt_cycle_seen = True
                    tracker.reset()
                elif prompt_cycle_seen:
                    # The prompt detector already requires three absent frames.
                    # Reopen immediately and retain previous_gray so motion can
                    # be detected on the next frame without another blind gap.
                    tracking_allowed = True
                    prompt_cycle_seen = False
                    tracker.reset()

                motion = tracker.update(now, detected.boot) if not paused and tracking_allowed else None
                stationary_rejected = tracker.rejected_stationary
                rejection_reason = tracker.rejection_reason
                if stationary_rejected:
                    # Preserve previous_gray. Resetting it creates a guaranteed
                    # blind frame before motion-based reacquisition can begin.
                    vision.reject_boot_location(detected.boot)
                decision = controller.decide(now, detected, motion, tracker.is_recent(now))
                reward = reward_classifier.update(
                    now,
                    score,
                    motion,
                    decision.phase,
                    detected.prompt_visible,
                )
                if reward.kind == "level_complete":
                    level_completion_waiting_prompt = True
                if decision.phase.value == "in_flight" and not detected.prompt_visible:
                    flight_seen_since_prompt = True
                    if decision.target_x is None:
                        if unprotected_since is None:
                            unprotected_since = now
                        max_unprotected_ms_since_launch = max(
                            max_unprotected_ms_since_launch,
                            (now - unprotected_since) * 1000.0,
                        )
                    else:
                        unprotected_since = None

                launch_prompt_event = None
                relaunch_unprotected_ms = None
                if prompt_rising:
                    if reward.kind == "level_complete" or level_completion_waiting_prompt:
                        launch_prompt_event = "level_transition"
                    elif flight_seen_since_prompt:
                        launch_prompt_event = "unexpected_relaunch"
                    else:
                        launch_prompt_event = "initial_launch"
                    relaunch_unprotected_ms = max_unprotected_ms_since_launch
                    flight_seen_since_prompt = False
                    unprotected_since = None
                    max_unprotected_ms_since_launch = 0.0
                    level_completion_waiting_prompt = False
                output_enabled = bool(args.control and armed and not paused and capture.target_available)
                if output_enabled:
                    keyboard.set_horizontal(decision.horizontal)
                    if decision.press_space:
                        keyboard.press_space(config.control.space_hold_ms)
                        # Our launch input is the synchronization point: begin
                        # tracking on the very next frame, without waiting for
                        # the visual prompt release debounce.
                        vision.acknowledge_launch(detected.aim)
                        tracking_allowed = True
                        prompt_cycle_seen = False
                        tracker.reset()
                else:
                    keyboard.release_all()

                elapsed = max(now - previous_time, 1e-6)
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else fps * 0.9 + instant_fps * 0.1
                previous_time = now
                telemetry.write({
                    "t": now,
                    "phase": decision.phase.value,
                    "paddle_x": detected.paddle.center_x if detected.paddle else None,
                    "boot_x": motion.x if motion else None,
                    "boot_y": motion.y if motion else None,
                    "boot_detection_source": detected.boot.source if detected.boot else None,
                    "boot_box_width": detected.boot.width if detected.boot else None,
                    "boot_box_height": detected.boot.height if detected.boot else None,
                    "boot_arc_score": detected.boot.arc_score if detected.boot else None,
                    "vx": motion.vx if motion else None,
                    "vy": motion.vy if motion else None,
                    "aim_top_x": detected.aim.top_x if detected.aim else None,
                    "aim_slope_dx_dy": detected.aim.slope_dx_dy if detected.aim else None,
                    "target_x": decision.target_x,
                    "intercept_x": decision.intercept_x,
                    "horizontal": decision.horizontal if output_enabled else 0,
                    "space": bool(decision.press_space and output_enabled),
                    "reason": decision.reason,
                    "armed": output_enabled,
                    "score": score.value,
                    "score_delta": score.delta,
                    "score_changed": score.changed,
                    "score_confidence": score.confidence,
                    "score_fingerprint": score.fingerprint if score.changed else None,
                    "reward_type": reward.kind,
                    "reward_signal": reward.value,
                    "top_exit_pending": reward.top_exit_pending,
                    "launch_prompt": detected.prompt_visible,
                    "launch_prompt_confidence": detected.prompt_confidence,
                    "launch_prompt_stability": detected.prompt_stability,
                    "launch_prompt_event": launch_prompt_event,
                    "relaunch_unprotected_ms": relaunch_unprotected_ms,
                    "current_unprotected_ms": (
                        (now - unprotected_since) * 1000.0 if unprotected_since is not None else 0.0
                    ),
                    "stationary_track_rejected": stationary_rejected,
                    "track_rejection_reason": rejection_reason,
                    "capture_ms": capture_ms,
                    "vision_ms": vision_ms,
                    "score_ms": score_ms,
                    "tracking_allowed": tracking_allowed,
                    "screen_resolution": capture.screen_size,
                    "capture_region": capture.region,
                    "content_crop": capture.content_crop,
                    "dota_window": (
                        {
                            "left": capture.window_bounds.left,
                            "top": capture.window_bounds.top,
                            "width": capture.window_bounds.width,
                            "height": capture.window_bounds.height,
                            "title": capture.window_bounds.title,
                        }
                        if capture.window_bounds is not None
                        else None
                    ),
                    "dota_window_available": capture.target_available,
                })
                prompt_visible_last = detected.prompt_visible

                if config.runtime.show_debug and not args.no_window:
                    playfield_debug = draw_debug(frame, config.field, detected, motion, decision, score, output_enabled, fps)
                    debug = full_frame.copy()
                    playfield_top = config.vision.playfield_top
                    playfield_bottom = min(debug.shape[0], playfield_top + playfield_debug.shape[0])
                    debug[playfield_top:playfield_bottom] = playfield_debug[: playfield_bottom - playfield_top]
                    score_cfg = config.score
                    cv2.rectangle(
                        debug,
                        (score_cfg.left, score_cfg.top),
                        (score_cfg.right, score_cfg.bottom),
                        (0, 255, 255),
                        1,
                    )
                    cv2.putText(
                        debug,
                        f"SCORE ROI: {score.value}",
                        (max(0, score_cfg.left - 110), score_cfg.bottom - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    if config.runtime.debug_scale != 1.0:
                        debug = cv2.resize(debug, None, fx=config.runtime.debug_scale, fy=config.runtime.debug_scale)
                    if not debug_window_initialized:
                        cv2.namedWindow(debug_window_name, cv2.WINDOW_AUTOSIZE)
                        cv2.imshow(debug_window_name, debug)
                        if config.runtime.debug_topmost:
                            cv2.setWindowProperty(debug_window_name, cv2.WND_PROP_TOPMOST, 1)
                        debug_window_initialized = True
                    else:
                        cv2.imshow(debug_window_name, debug)
                    current_region = (
                        capture.region["left"],
                        capture.region["top"],
                        capture.region["width"],
                        capture.region["height"],
                    )
                    if config.runtime.debug_auto_position and current_region != debug_window_region:
                        window_x, window_y = choose_window_position(
                            capture.region,
                            capture.desktop_bounds,
                            debug.shape[1],
                            debug.shape[0],
                        )
                        cv2.moveWindow(debug_window_name, window_x, window_y)
                        debug_window_region = current_region
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if config.runtime.idle_sleep_ms > 0:
                    time.sleep(config.runtime.idle_sleep_ms / 1000.0)
    finally:
        keyboard.release_all()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
