from __future__ import annotations

import time

import cv2
import mss
import numpy as np

from .config import ROIConfig
from .window_locator import WindowBounds, find_process_window, refresh_window_bounds


class ScreenCapture:
    def __init__(
        self,
        roi: ROIConfig,
        reference_size: tuple[int, int] | None = None,
        aspect_tolerance: float = 0.03,
        auto_content_align: bool = False,
        window_executable: str | None = None,
        require_window: bool = False,
        window_refresh_seconds: float = 1.0,
    ):
        self._sct = mss.mss()
        self.output_size: tuple[int, int] | None = None
        self.content_crop: tuple[int, int, int, int] | None = None
        self._pending_crop: tuple[int, int, int, int] | None = None
        self._pending_crop_frames = 0
        monitor = self._sct.monitors[1]
        virtual = self._sct.monitors[0]
        self.screen_size = (monitor["width"], monitor["height"])
        self.desktop_bounds = {
            "left": virtual["left"],
            "top": virtual["top"],
            "width": virtual["width"],
            "height": virtual["height"],
        }
        self.auto_content_align = auto_content_align
        self.reference_size = reference_size
        self.aspect_tolerance = aspect_tolerance
        self.roi = roi
        self.window_executable = window_executable
        self.require_window = require_window
        self.window_refresh_seconds = window_refresh_seconds
        self.window_bounds: WindowBounds | None = None
        self.target_available = True
        self._next_window_refresh = 0.0
        if window_executable:
            self.window_bounds = find_process_window(window_executable)
            if self.window_bounds is None and require_window:
                raise RuntimeError(
                    f"Could not find a visible {window_executable} window. "
                    "Start Dota 2 and make sure it is not minimized."
                )
            self.target_available = self.window_bounds is not None
            if self.window_bounds is not None and reference_size is not None:
                monitor = {
                    "left": self.window_bounds.left,
                    "top": self.window_bounds.top,
                    "width": self.window_bounds.width,
                    "height": self.window_bounds.height,
                }
        if reference_size is None:
            self.region = {"left": roi.left, "top": roi.top, "width": roi.width, "height": roi.height}
        else:
            if self.window_bounds is not None:
                self.region = fitted_region(
                    roi,
                    monitor["left"],
                    monitor["top"],
                    monitor["width"],
                    monitor["height"],
                    reference_size[0],
                    reference_size[1],
                )
            else:
                self.region = scaled_region(
                    roi,
                    monitor["left"],
                    monitor["top"],
                    monitor["width"],
                    monitor["height"],
                    reference_size[0],
                    reference_size[1],
                    aspect_tolerance,
                )
            self.output_size = (roi.width, roi.height)

    @property
    def target_description(self) -> str:
        if self.window_bounds is not None:
            item = self.window_bounds
            return f"{item.executable} client=({item.left},{item.top},{item.width},{item.height}) roi={self.region}"
        return f"display={self.screen_size[0]}x{self.screen_size[1]} roi={self.region}"

    def _refresh_window_region(self) -> None:
        if not self.window_executable or self.reference_size is None:
            return
        now = time.monotonic()
        if now < self._next_window_refresh:
            return
        self._next_window_refresh = now + self.window_refresh_seconds
        bounds = refresh_window_bounds(self.window_bounds) if self.window_bounds is not None else None
        if bounds is None:
            bounds = find_process_window(self.window_executable)
        if bounds is None:
            self.target_available = False
            return
        self.target_available = True
        changed = self.window_bounds != bounds
        self.window_bounds = bounds
        new_region = fitted_region(
            self.roi,
            bounds.left,
            bounds.top,
            bounds.width,
            bounds.height,
            self.reference_size[0],
            self.reference_size[1],
        )
        if new_region != self.region:
            self.region = new_region
            changed = True
        if changed:
            self.content_crop = None
            self._pending_crop = None
            self._pending_crop_frames = 0

    def grab(self) -> np.ndarray:
        self._refresh_window_region()
        bgra = np.asarray(self._sct.grab(self.region))
        frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        if self.output_size is not None and (frame.shape[1], frame.shape[0]) != self.output_size:
            interpolation = cv2.INTER_AREA if frame.shape[1] > self.output_size[0] else cv2.INTER_LINEAR
            frame = cv2.resize(frame, self.output_size, interpolation=interpolation)
        if self.auto_content_align and self.output_size is not None:
            if self.content_crop is None:
                detected = detect_content_rect(frame, self.output_size[0], self.output_size[1])
                if detected is not None:
                    if self._pending_crop is not None and max(
                        abs(left - right) for left, right in zip(detected, self._pending_crop)
                    ) <= 6:
                        self._pending_crop_frames += 1
                    else:
                        self._pending_crop = detected
                        self._pending_crop_frames = 1
                    if self._pending_crop_frames >= 2:
                        self.content_crop = detected
                else:
                    self._pending_crop = None
                    self._pending_crop_frames = 0
            if self.content_crop is not None:
                x, y, width, height = self.content_crop
                content = frame[y : y + height, x : x + width]
                if content.size:
                    interpolation = cv2.INTER_AREA if width > self.output_size[0] else cv2.INTER_LINEAR
                    frame = cv2.resize(content, self.output_size, interpolation=interpolation)
        return frame

    def close(self) -> None:
        self._sct.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def scaled_region(
    roi: ROIConfig,
    screen_left: int,
    screen_top: int,
    screen_width: int,
    screen_height: int,
    reference_width: int,
    reference_height: int,
    aspect_tolerance: float = 0.03,
) -> dict[str, int]:
    """Scale an absolute reference ROI to a same-aspect-ratio display."""
    reference_aspect = reference_width / reference_height
    actual_aspect = screen_width / screen_height
    relative_error = abs(actual_aspect / reference_aspect - 1.0)
    if relative_error > aspect_tolerance:
        raise ValueError(
            f"Unsupported display aspect ratio {screen_width}x{screen_height}; "
            f"expected approximately {reference_width}:{reference_height}."
        )
    scale_x = screen_width / reference_width
    scale_y = screen_height / reference_height
    return {
        "left": screen_left + round(roi.left * scale_x),
        "top": screen_top + round(roi.top * scale_y),
        "width": max(1, round(roi.width * scale_x)),
        "height": max(1, round(roi.height * scale_y)),
    }


def fitted_region(
    roi: ROIConfig,
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
    reference_width: int,
    reference_height: int,
) -> dict[str, int]:
    """Fit a reference canvas inside any client aspect ratio and scale its ROI."""
    scale = min(window_width / reference_width, window_height / reference_height)
    canvas_width = reference_width * scale
    canvas_height = reference_height * scale
    origin_x = window_left + (window_width - canvas_width) / 2.0
    origin_y = window_top + (window_height - canvas_height) / 2.0
    return {
        "left": round(origin_x + roi.left * scale),
        "top": round(origin_y + roi.top * scale),
        "width": max(1, round(roi.width * scale)),
        "height": max(1, round(roi.height * scale)),
    }


def detect_content_rect(frame: np.ndarray, target_width: int, target_height: int) -> tuple[int, int, int, int] | None:
    """Find the minigame's fixed gold/red frame inside an oversized capture."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = ((hue <= 12) | (hue >= 170)) & (saturation >= 105) & (value >= 65)
    gold = (hue >= 5) & (hue <= 38) & (saturation >= 95) & (value >= 85)
    border = (red | gold).astype(np.uint8)
    height, width = border.shape
    if width < 100 or height < 160:
        return None

    raw_vertical = border.sum(axis=0)
    if raw_vertical[:3].max() >= height * 0.45 and raw_vertical[-3:].max() >= height * 0.45:
        return 0, 0, width, height

    vertical = raw_vertical.astype(np.float32)
    vertical = np.convolve(vertical, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    active_x = np.flatnonzero(vertical >= height * 0.16)
    if len(active_x) < 2:
        return None
    splits = np.flatnonzero(np.diff(active_x) > 8) + 1
    clusters = [cluster for cluster in np.split(active_x, splits) if len(cluster)]
    centers = [int(round(float(cluster.mean()))) for cluster in clusters]
    pairs = [
        (left, right)
        for left in centers
        for right in centers
        if width * 0.50 <= right - left <= width * 0.98
    ]
    if not pairs:
        return None
    for left, right in sorted(
        pairs,
        key=lambda pair: float(vertical[pair[0]] + vertical[pair[1]]),
        reverse=True,
    ):
        crop_width = right - left + 1
        expected_height = int(round(crop_width * target_height / target_width))
        if expected_height > height:
            continue
        horizontal = border[:, left : right + 1].sum(axis=1).astype(np.float32)
        horizontal = np.convolve(horizontal, np.ones(3, dtype=np.float32) / 3.0, mode="same")
        top_candidates = np.flatnonzero(horizontal[: max(1, height // 3)] >= crop_width * 0.30)
        for top_value in top_candidates:
            top = int(top_value)
            if top + expected_height <= height:
                return left, top, crop_width, expected_height
    return None
