from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import DialConfig, HSVRange, VisionConfig
from .geometry import AngularZone, signed_delta


@dataclass(frozen=True)
class VisionResult:
    pointer_angle: float | None
    pointer_confidence: float
    zones: tuple[AngularZone, ...]


class DialVision:
    def __init__(self, dial: DialConfig, config: VisionConfig):
        self.dial = dial
        self.config = config
        self.angles = np.linspace(0.0, 2.0 * np.pi, dial.angle_bins, endpoint=False, dtype=np.float32)
        self._pointer_maps = self._make_maps(dial.pointer_inner_ratio, dial.pointer_outer_ratio, 72)
        self._zone_maps = self._make_maps(dial.zone_inner_ratio, dial.zone_outer_ratio, 28)
        self._trusted_blue: list[tuple[AngularZone, int]] = []

    def _make_maps(self, inner_ratio: float, outer_ratio: float, radial_samples: int):
        radii = np.linspace(
            self.dial.radius * inner_ratio,
            self.dial.radius * outer_ratio,
            radial_samples,
            dtype=np.float32,
        )[:, None]
        x = self.dial.center_x + radii * np.cos(self.angles)[None, :]
        y = self.dial.center_y + radii * np.sin(self.angles)[None, :]
        return x.astype(np.float32), y.astype(np.float32)

    @staticmethod
    def _mask(hsv: np.ndarray, limits: HSVRange) -> np.ndarray:
        lower = np.array([limits.h_min, limits.s_min, limits.v_min], dtype=np.uint8)
        upper = np.array([limits.h_max, limits.s_max, limits.v_max], dtype=np.uint8)
        if limits.h_min <= limits.h_max:
            return cv2.inRange(hsv, lower, upper)
        left = cv2.inRange(hsv, np.array([0, limits.s_min, limits.v_min], np.uint8), upper)
        right = cv2.inRange(hsv, lower, np.array([179, limits.s_max, limits.v_max], np.uint8))
        return cv2.bitwise_or(left, right)

    @staticmethod
    def _hue_mask(hsv: np.ndarray, limits: HSVRange) -> np.ndarray:
        hue = hsv[:, :, 0]
        if limits.h_min <= limits.h_max:
            active = (hue >= limits.h_min) & (hue <= limits.h_max)
        else:
            active = (hue >= limits.h_min) | (hue <= limits.h_max)
        return active

    def _adaptive_zone_mask(self, bgr: np.ndarray, hsv: np.ndarray, limits: HSVRange, kind: str) -> np.ndarray:
        """Brightness-independent color mask using normalized opponent channels."""
        pixels = bgr.astype(np.float32)
        blue, green, red = cv2.split(pixels)
        total = blue + green + red + 1.0
        if kind == "yellow":
            opponent = ((red + green) * 0.5 - blue) / total
            color_active = opponent >= self.config.yellow_opponent_min
        else:
            opponent = (blue - (red + green) * 0.5) / total
            color_active = opponent >= self.config.blue_opponent_min
        hue_active = self._hue_mask(hsv, limits)
        visible = (hsv[:, :, 2] >= self.config.zone_min_value) & (hsv[:, :, 1] >= self.config.zone_min_saturation)
        if kind == "blue":
            visible &= hsv[:, :, 2] <= self.config.blue_zone_max_value
        return ((color_active & hue_active & visible).astype(np.uint8) * 255)

    @staticmethod
    def _sample(mask: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        sampled = cv2.remap(mask, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return sampled.astype(np.float32).mean(axis=0) / 255.0

    def _smooth(self, scores: np.ndarray) -> np.ndarray:
        width = max(1, int(self.config.smooth_bins))
        if width == 1:
            return scores
        pad = width // 2
        wrapped = np.concatenate((scores[-pad:], scores, scores[:pad]))
        kernel = np.ones(width, dtype=np.float32) / width
        return np.convolve(wrapped, kernel, mode="valid")[: len(scores)]

    def _extract_zones(self, active: np.ndarray, scores: np.ndarray, kind: str) -> list[AngularZone]:
        n = len(active)
        if not active.any():
            return []
        if active.all():
            return []
        first_false = int(np.flatnonzero(~active)[0])
        ordered = np.roll(active, -first_false - 1)
        ordered_scores = np.roll(scores, -first_false - 1)
        changes = np.diff(np.r_[False, ordered, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        degrees_per_bin = 360.0 / n
        zones: list[AngularZone] = []
        offset = first_false + 1
        for start_idx, end_idx in zip(starts, ends):
            width = (end_idx - start_idx) * degrees_per_bin
            if width < self.config.zone_min_width_deg:
                continue
            start = float(((start_idx + offset) % n) * degrees_per_bin)
            end = float(((end_idx + offset) % n) * degrees_per_bin)
            confidence = float(ordered_scores[start_idx:end_idx].mean())
            zones.append(AngularZone(start, end, kind, confidence))
        return zones

    @staticmethod
    def _near_pointer(zone: AngularZone, pointer_angle: float) -> bool:
        distance = abs(signed_delta(pointer_angle, zone.center))
        return distance <= zone.width / 2.0 + 5.0

    @staticmethod
    def _same_place(left: AngularZone, right: AngularZone) -> bool:
        tolerance = max(6.0, min(18.0, (left.width + right.width) / 2.0))
        return abs(signed_delta(left.center, right.center)) <= tolerance

    def _filter_blue_pointer_artifacts(self, zones: list[AngularZone], pointer_angle: float | None) -> list[AngularZone]:
        """Reject blue glow that follows the pointer and bridge brief pointer occlusion.

        A genuine blue zone is first observed away from the pointer. When the pointer
        reaches it, its most recent trusted geometry is retained for a few frames.
        """
        if pointer_angle is None:
            self._trusted_blue = [(zone, 0) for zone in zones]
            return zones

        visible = [zone for zone in zones if not self._near_pointer(zone, pointer_angle)]
        next_tracks: list[tuple[AngularZone, int]] = [(zone, 0) for zone in visible]
        output = list(visible)
        for previous, age in self._trusted_blue:
            if any(self._same_place(previous, current) for current in visible):
                continue
            if age < 8 and self._near_pointer(previous, pointer_angle):
                output.append(previous)
                next_tracks.append((previous, age + 1))
        self._trusted_blue = next_tracks
        return output

    def detect(self, bgr: np.ndarray) -> VisionResult:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        pointer_scores = self._smooth(self._sample(self._mask(hsv, self.config.pointer_hsv), self._pointer_maps))
        pointer_idx = int(np.argmax(pointer_scores))
        pointer_conf = float(pointer_scores[pointer_idx])
        pointer_angle = pointer_idx * 360.0 / self.dial.angle_bins if pointer_conf >= self.config.pointer_min_score else None

        yellow_zones: list[AngularZone] = []
        blue_zones: list[AngularZone] = []
        for kind, hsv_range in (("yellow", self.config.yellow_hsv), ("blue", self.config.blue_hsv)):
            if self.config.adaptive_zone_color:
                mask = self._adaptive_zone_mask(bgr, hsv, hsv_range, kind)
            else:
                mask = self._mask(hsv, hsv_range)
            scores = self._smooth(self._sample(mask, self._zone_maps))
            extracted = self._extract_zones(scores >= self.config.zone_pixel_ratio, scores, kind)
            if kind == "yellow":
                yellow_zones.extend(extracted)
            else:
                blue_zones.extend(extracted)
        blue_zones = self._filter_blue_pointer_artifacts(blue_zones, pointer_angle)
        return VisionResult(pointer_angle, pointer_conf, tuple(yellow_zones + blue_zones))
