from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import VisionConfig


@dataclass(frozen=True)
class PaddleDetection:
    center_x: float
    width: float
    confidence: float


@dataclass(frozen=True)
class BootDetection:
    x: float
    y: float
    confidence: float
    source: str
    width: float = 0.0
    height: float = 0.0
    arc_score: float = 0.0


@dataclass(frozen=True)
class AimDetection:
    points: tuple[tuple[float, float], ...]
    slope_dx_dy: float
    top_x: float
    confidence: float


@dataclass(frozen=True)
class VisionResult:
    paddle: PaddleDetection | None
    boot: BootDetection | None
    aim: AimDetection | None
    motion_pixels: int
    prompt_visible: bool = False
    prompt_confidence: float = 0.0
    prompt_stability: float = 0.0


class BootBreakerVision:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.previous_gray: np.ndarray | None = None
        self.last_boot: BootDetection | None = None
        self.missed_boot_frames = 0
        self.rejected_regions: list[tuple[float, float, int]] = []
        self.pending_boot: BootDetection | None = None
        self.pending_boot_frames = 0
        self.prompt_on_frames = 0
        self.prompt_off_frames = 0
        self.prompt_latched = False
        self.prompt_latched_confidence = 0.0
        self.previous_prompt_mask: np.ndarray | None = None
        self.launch_aim: AimDetection | None = None
        self.launch_prior_frames_left = 0
        self.last_aim: AimDetection | None = None

    def reset_tracking(self) -> None:
        self.previous_gray = None
        self.last_boot = None
        self.missed_boot_frames = 0
        self.pending_boot = None
        self.pending_boot_frames = 0
        self.launch_aim = None
        self.launch_prior_frames_left = 0
        self.last_aim = None

    def reject_boot_location(self, detection: BootDetection | None) -> None:
        if detection is None:
            return
        self.last_boot = None
        self.missed_boot_frames = 0
        self.pending_boot = None
        self.pending_boot_frames = 0
        self.rejected_regions.append(
            (detection.x, detection.y, self.config.rejected_region_frames)
        )

    def _age_rejected_regions(self) -> None:
        self.rejected_regions = [
            (x, y, frames - 1)
            for x, y, frames in self.rejected_regions
            if frames > 1
        ]

    def _is_rejected_region(self, detection: BootDetection) -> bool:
        return any(
            float(np.hypot(detection.x - x, detection.y - y)) <= self.config.rejected_region_radius_px
            for x, y, _ in self.rejected_regions
        )

    @staticmethod
    def _red_mask(hsv: np.ndarray) -> np.ndarray:
        hue = hsv[:, :, 0]
        return (((hue <= 10) | (hue >= 170)) & (hsv[:, :, 1] >= 145) & (hsv[:, :, 2] >= 90)).astype(np.uint8)

    def _detect_paddle(self, hsv: np.ndarray) -> PaddleDetection | None:
        top, bottom = self.config.paddle_band_top, self.config.paddle_band_bottom
        band = self._red_mask(hsv[top:bottom])
        ys, xs = np.nonzero(band)
        if len(xs) < 80:
            return None
        histogram = np.bincount(xs, minlength=hsv.shape[1])
        active = histogram >= 2
        indices = np.flatnonzero(active)
        if not len(indices):
            return None
        # Split unrelated red decorations and retain the widest cluster.
        splits = np.flatnonzero(np.diff(indices) > 12) + 1
        clusters = np.split(indices, splits)
        cluster = max(clusters, key=lambda values: values[-1] - values[0])
        width = float(cluster[-1] - cluster[0] + 1)
        if width < self.config.paddle_min_width:
            return None
        confidence = min(1.0, len(xs) / 900.0)
        return PaddleDetection(float(cluster[0] + cluster[-1]) / 2.0, width, confidence)

    def _detect_aim(self, hsv: np.ndarray) -> AimDetection | None:
        top, bottom = self.config.aim_band_top, self.config.aim_band_bottom
        h, s, v = cv2.split(hsv[top:bottom])
        mask = ((h >= 3) & (h <= 35) & (s >= 120) & (v >= 95)).astype(np.uint8)
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        points: list[tuple[float, float]] = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if 8 <= area <= 60 and width <= 12 and height <= 8 and x > 8:
                points.append((float(centers[index][0]), float(centers[index][1] + top)))
        if len(points) < self.config.aim_min_points:
            return None
        points.sort(key=lambda point: point[1])
        ys = np.array([point[1] for point in points], dtype=np.float64)
        xs = np.array([point[0] for point in points], dtype=np.float64)
        matrix = np.c_[ys, np.ones_like(ys)]
        slope, intercept = np.linalg.lstsq(matrix, xs, rcond=None)[0]
        residual = float(np.mean(np.abs(xs - (slope * ys + intercept))))
        if residual > 4.0:
            return None
        top_x = float(slope * self.config.aim_band_top + intercept)
        confidence = min(1.0, len(points) / 6.0) * max(0.0, 1.0 - residual / 4.0)
        return AimDetection(tuple(points), float(slope), top_x, confidence)

    def _prompt_mask_stability(self, mask: np.ndarray) -> float:
        previous = self.previous_prompt_mask
        self.previous_prompt_mask = mask.copy()
        if previous is None:
            return 0.0
        intersection = int(np.count_nonzero(mask & previous))
        union = int(np.count_nonzero(mask | previous))
        return intersection / union if union else 0.0

    def _detect_prompt(self, hsv: np.ndarray) -> tuple[bool, float, float]:
        cfg = self.config
        region = hsv[cfg.prompt_top : cfg.prompt_bottom, cfg.prompt_left : cfg.prompt_right]
        if region.size == 0:
            return False, 0.0, 0.0
        bright_text = (region[:, :, 2] >= 110) & (region[:, :, 1] <= 180)
        pixels = int(np.count_nonzero(bright_text))
        stability = self._prompt_mask_stability(bright_text)
        pixel_confidence = min(1.0, pixels / max(cfg.prompt_min_pixels * 2.0, 1.0))
        confidence = pixel_confidence * stability
        visible = pixels >= cfg.prompt_min_pixels and stability >= cfg.prompt_min_mask_iou
        return visible, confidence, stability

    def _debounce_prompt(self, visible: bool, confidence: float) -> tuple[bool, float]:
        if visible:
            self.prompt_on_frames += 1
            self.prompt_off_frames = 0
            self.prompt_latched_confidence = max(self.prompt_latched_confidence, confidence)
            if self.prompt_on_frames >= self.config.prompt_confirm_frames:
                self.prompt_latched = True
        else:
            self.prompt_on_frames = 0
            self.prompt_off_frames += 1
            if self.prompt_off_frames >= self.config.prompt_release_frames:
                self.prompt_latched = False
                self.prompt_latched_confidence = 0.0
        return self.prompt_latched, self.prompt_latched_confidence

    def acknowledge_launch(self, aim: AimDetection | None = None) -> None:
        """Release the prompt latch immediately after our own space input.

        The next captured frame may establish a boot track. If the input did
        not launch, the still-visible stationary prompt will reconfirm after
        the normal three frames and can be retried safely.
        """
        self.prompt_on_frames = 0
        self.prompt_off_frames = 0
        self.prompt_latched = False
        self.prompt_latched_confidence = 0.0
        selected_aim = aim if aim is not None else self.last_aim
        self.launch_aim = selected_aim
        self.launch_prior_frames_left = self.config.launch_prior_frames if selected_aim is not None else 0
        self.last_aim = None

    def _launch_line_error(self, candidate: BootDetection) -> float | None:
        if self.launch_aim is None or self.launch_prior_frames_left <= 0:
            return None
        expected_x = self.launch_aim.top_x + self.launch_aim.slope_dx_dy * (
            candidate.y - self.config.aim_band_top
        )
        return abs(candidate.x - expected_x)

    def _cyan_candidates(self, hsv: np.ndarray) -> list[BootDetection]:
        # Include the paddle collision band. Limiting this to y=700 made a
        # rejected track impossible to reacquire during the final descent.
        h, s, v = cv2.split(hsv[: self.config.boot_search_bottom])
        mask = (
            (h >= self.config.boot_cyan_h_min)
            & (h <= self.config.boot_cyan_h_max)
            & (s >= 95)
            & (v >= 90)
        ).astype(np.uint8)
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        results: list[BootDetection] = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if 5 <= area <= 180 and width <= 35 and height <= 35:
                center_x = float(centers[index][0])
                center_y = float(centers[index][1])
                arc_score = self._boot_arc_score(mask, center_x, center_y)
                results.append(BootDetection(center_x, center_y, min(1.0, area / 45.0), "cyan", arc_score=arc_score))
        return results

    def _boot_arc_score(self, cyan_mask: np.ndarray, center_x: float, center_y: float) -> float:
        """Score the boot's invariant cyan crescent using a local circle fit.

        A circle is defined by three points; least squares uses all visible arc
        pixels to make that construction resistant to pixel-art gaps. Filled
        rectangular cyan bricks have poor radial agreement and score low.
        """
        radius = 36
        height, width = cyan_mask.shape
        x0 = max(0, int(center_x) - radius)
        y0 = max(0, int(center_y) - radius)
        x1 = min(width, int(center_x) + radius + 1)
        y1 = min(height, int(center_y) + radius + 1)
        ys, xs = np.nonzero(cyan_mask[y0:y1, x0:x1])
        if len(xs) < 18:
            return 0.0
        xs = xs.astype(np.float64) + x0
        ys = ys.astype(np.float64) + y0
        matrix = np.c_[2.0 * xs, 2.0 * ys, np.ones_like(xs)]
        squared = xs * xs + ys * ys
        circle_x, circle_y, constant = np.linalg.lstsq(matrix, squared, rcond=None)[0]
        fitted_radius_sq = circle_x * circle_x + circle_y * circle_y + constant
        if fitted_radius_sq <= 0.0:
            return 0.0
        fitted_radius = float(np.sqrt(fitted_radius_sq))
        if not self.config.boot_arc_min_radius_px <= fitted_radius <= self.config.boot_arc_max_radius_px:
            return 0.0
        errors = np.abs(np.hypot(xs - circle_x, ys - circle_y) - fitted_radius)
        median_error = float(np.median(errors))
        if median_error > self.config.boot_arc_max_median_error_px:
            return 0.0
        inlier_ratio = float(np.mean(errors <= 3.0))
        residual_score = max(0.0, 1.0 - median_error / self.config.boot_arc_max_median_error_px)
        return inlier_ratio * residual_score

    def _refine_boot_center(self, hsv: np.ndarray, anchor: BootDetection) -> BootDetection:
        """Expand a cyan sole feature into the full colored boot silhouette.

        The sprite rotates around its physical body, while the cyan sole is an
        off-center feature. A local silhouette bounding-box center is a much
        less oscillatory proxy for collision tracking.
        """
        radius = 42
        height, width = hsv.shape[:2]
        x0 = max(0, int(anchor.x) - radius)
        y0 = max(0, int(anchor.y) - radius)
        x1 = min(width, int(anchor.x) + radius + 1)
        y1 = min(height, int(anchor.y) + radius + 1)
        patch = hsv[y0:y1, x0:x1]
        h, s, v = cv2.split(patch)
        brown = (h >= 3) & (h <= 32) & (s >= 70) & (v >= 50)
        cyan = (
            (h >= self.config.boot_cyan_h_min)
            & (h <= self.config.boot_cyan_h_max)
            & (s >= 95)
            & (v >= 90)
        )
        mask = ((brown | cyan).astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        choices: list[tuple[float, int]] = []
        for index in range(1, count):
            x, y, box_width, box_height, area = stats[index]
            center_x = float(centers[index][0] + x0)
            center_y = float(centers[index][1] + y0)
            distance = float(np.hypot(center_x - anchor.x, center_y - anchor.y))
            if 45 <= area <= 2400 and 8 <= box_width <= 72 and 8 <= box_height <= 72 and distance <= 38:
                choices.append((area - distance * 8.0, index))
        if not choices:
            return anchor
        _, index = max(choices)
        x, y, box_width, box_height, area = stats[index]
        center_x = x0 + x + box_width / 2.0
        center_y = y0 + y + box_height / 2.0
        confidence = min(1.0, anchor.confidence * 0.55 + area / 900.0)
        return BootDetection(
            center_x,
            center_y,
            confidence,
            "silhouette",
            float(box_width),
            float(box_height),
            anchor.arc_score,
        )

    def _motion_candidates(self, frame: np.ndarray, gray: np.ndarray) -> tuple[list[BootDetection], int]:
        if self.previous_gray is None:
            return [], 0
        bottom = self.config.boot_search_bottom
        difference = cv2.absdiff(gray[:bottom], self.previous_gray[:bottom])
        mask = (difference >= self.config.motion_threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        results: list[BootDetection] = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if 18 <= area <= 2600 and 4 <= width <= 90 and 4 <= height <= 90:
                # Persistent brick animations tend to be very wide rectangles.
                aspect = width / max(height, 1)
                if 0.22 <= aspect <= 4.0:
                    confidence = min(0.85, 0.25 + area / 900.0)
                    results.append(BootDetection(float(centers[index][0]), float(centers[index][1]), confidence, "motion"))
        return results, int(np.count_nonzero(mask))

    @staticmethod
    def _distance(left: BootDetection, right: BootDetection) -> float:
        return float(np.hypot(left.x - right.x, left.y - right.y))

    def _select_boot(self, cyan: list[BootDetection], motion: list[BootDetection]) -> BootDetection | None:
        if not motion:
            self.pending_boot = None
            self.pending_boot_frames = 0
            return None
        supported_cyan = [
            candidate
            for candidate in cyan
            if any(self._distance(candidate, moving) <= 55.0 for moving in motion)
        ]
        if self.last_boot is not None:
            # Motion alone is not an identity feature: near the paddle it can
            # silently transfer the track to impact effects or the cart.
            nearby = [
                candidate
                for candidate in supported_cyan
                if self._distance(candidate, self.last_boot)
                <= (
                    self.config.boot_rebound_track_max_distance
                    if candidate.y >= self.config.boot_safe_zone_top
                    and self.last_boot.y >= self.config.boot_safe_zone_top
                    else self.config.boot_track_max_distance
                )
            ]
            if nearby:
                self.pending_boot = None
                self.pending_boot_frames = 0
                return max(
                    nearby,
                    key=lambda item: item.confidence + item.arc_score * 0.8 - self._distance(item, self.last_boot) / 300.0,
                )
        if supported_cyan:
            launch_candidates = [
                candidate
                for candidate in supported_cyan
                if (error := self._launch_line_error(candidate)) is not None
                and error <= self.config.launch_prior_tolerance_px
                and not self._is_rejected_region(candidate)
            ]
            if launch_candidates:
                selected = min(
                    launch_candidates,
                    key=lambda item: self._launch_line_error(item) or 0.0,
                )
                self.launch_aim = None
                self.launch_prior_frames_left = 0
                self.pending_boot = None
                self.pending_boot_frames = 0
                return selected

            # In the lower no-brick zone, cyan+motion is sufficient for rapid
            # post-impact recovery. Above it, require the invariant arc so a
            # moving brick highlight cannot become a new boot identity.
            allowed = [
                candidate
                for candidate in supported_cyan
                if (
                    candidate.y >= self.config.boot_safe_zone_top
                    or candidate.arc_score >= self.config.boot_arc_min_score
                )
                and not self._is_rejected_region(candidate)
            ]
            if allowed:
                selected = max(allowed, key=lambda item: item.confidence + item.arc_score * 0.8)
                self.pending_boot = None
                self.pending_boot_frames = 0
                return selected

            # The crescent can disappear at some sprite rotations. A moving
            # cyan candidate may still establish identity after three
            # consecutive, non-stationary observations. Static cyan bricks do
            # not pass this temporal displacement test.
            candidate = max(supported_cyan, key=lambda item: item.confidence)
            if self.pending_boot is not None:
                displacement = self._distance(candidate, self.pending_boot)
                if (
                    self.config.boot_motion_confirm_min_displacement_px <= displacement
                    <= self.config.boot_track_max_distance
                ):
                    self.pending_boot_frames += 1
                else:
                    self.pending_boot_frames = 1
            else:
                self.pending_boot_frames = 1
            self.pending_boot = candidate
            if self.pending_boot_frames >= self.config.boot_motion_confirm_frames:
                self.pending_boot = None
                self.pending_boot_frames = 0
                return candidate
        else:
            self.pending_boot = None
            self.pending_boot_frames = 0
        return None

    def detect(self, frame: np.ndarray, allow_boot_tracking: bool = True) -> VisionResult:
        self._age_rejected_regions()
        if self.launch_prior_frames_left > 0:
            self.launch_prior_frames_left -= 1
            if self.launch_prior_frames_left == 0:
                self.launch_aim = None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        paddle = self._detect_paddle(hsv)
        aim = self._detect_aim(hsv)
        if aim is not None:
            self.last_aim = aim
        prompt_raw, prompt_confidence, prompt_stability = self._detect_prompt(hsv)
        if paddle is None:
            prompt_raw = False
        prompt_visible, prompt_confidence = self._debounce_prompt(prompt_raw, prompt_confidence)
        if allow_boot_tracking and not prompt_visible:
            cyan = self._cyan_candidates(hsv)
            motion, motion_pixels = self._motion_candidates(frame, gray)
            boot = self._select_boot(cyan, motion)
            if boot is not None and boot.source == "cyan":
                boot = self._refine_boot_center(hsv, boot)
        else:
            boot = None
            motion_pixels = 0
            self.last_boot = None
            self.missed_boot_frames = 0
        if paddle is None:
            # HUD, end screens and menus contain cyan/orange decorations but no playable paddle.
            boot = None
            aim = None
            prompt_visible = False
            prompt_confidence = 0.0
        self.previous_gray = gray
        if boot is not None:
            self.last_boot = boot
            self.missed_boot_frames = 0
        else:
            self.missed_boot_frames += 1
            if self.missed_boot_frames >= 10:
                self.last_boot = None
        return VisionResult(paddle, boot, aim, motion_pixels, prompt_visible, prompt_confidence, prompt_stability)
