from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ScoreConfig


@dataclass(frozen=True)
class ScoreResult:
    value: int | None
    delta: int | None
    changed: bool
    confidence: float
    fingerprint: str


class ScoreMonitor:
    """Small fixed-font OCR plus a font-independent stable-change signal."""

    def __init__(self, config: ScoreConfig):
        self.config = config
        self.templates = self._build_templates()
        self.candidate_key: str | None = None
        self.candidate_count = 0
        self.stable_key: str | None = None
        self.stable_value: int | None = None
        self.ocr_cache: dict[str, tuple[int | None, float]] = {}
        self.last_crop: np.ndarray | None = None

    @staticmethod
    def _normalize(glyph: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(glyph)
        output = np.zeros((32, 20), dtype=np.uint8)
        if not len(xs):
            return output
        glyph = glyph[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        scale = min(18.0 / glyph.shape[1], 30.0 / glyph.shape[0])
        resized = cv2.resize(glyph, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        y = (32 - resized.shape[0]) // 2
        x = (20 - resized.shape[1]) // 2
        output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return output

    def _build_templates(self) -> dict[int, list[np.ndarray]]:
        templates: dict[int, list[np.ndarray]] = {digit: [] for digit in range(10)}
        fonts = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_TRIPLEX)
        for digit in range(10):
            for font in fonts:
                for scale in (0.7, 0.8, 0.9, 1.0, 1.1):
                    for thickness in (1, 2, 3):
                        canvas = np.zeros((50, 40), dtype=np.uint8)
                        cv2.putText(canvas, str(digit), (2, 38), font, scale, 255, thickness, cv2.LINE_AA)
                        templates[digit].append(self._normalize(canvas))
        # Exact fixed-font samples visible in the supplied 1920x1080 frames.
        literals = {
            0: (
                "00001110000", "00111111100", "01111111110", "01111111110",
                "11110001111", "11110001111", "11110000111", "11110000111",
                "11110000111", "11110000111", "11110000111", "11110001111",
                "01110001111", "01111111110", "00111111110", "00011111100",
            ),
            5: (
                "0111111110", "0111111111", "0111111111", "0111111110",
                "0111000000", "0111000000", "0111111000", "0111111100",
                "0111111110", "0000011111", "0000001111", "0000001111",
                "0000001111", "1111111110", "1111111110", "1111111100",
            ),
            2: (
                "01111000", "11111110", "11111110", "00001110",
                "00001110", "00001110", "00001100", "00011100",
                "00111000", "01111000", "11111111", "11111111",
            ),
            1: (
                "00011", "01111", "11111", "11111",
                "00111", "00111", "00111", "00111",
                "00111", "00111", "00111", "00111",
            ),
            3: (
                "0111000", "1111110", "1111111", "0000111",
                "0001110", "0111100", "0011100", "0000110",
                "0000111", "1001111", "1111110", "1111100",
            ),
            8: (
                "00011000", "01111110", "01111111", "01100111",
                "01110110", "00111110", "00111110", "01100111",
                "11100011", "11100111", "01111110", "00111100",
            ),
        }
        for digit, rows in literals.items():
            glyph = np.array([[255 if cell == "1" else 0 for cell in row] for row in rows], dtype=np.uint8)
            templates[digit].append(self._normalize(glyph))
        extra_literals = (
            (2, (
                "01111000", "11111110", "11111110", "00001110",
                "00001110", "00001110", "00001100", "00011100",
                "00111000", "01111110", "11111111", "11111111",
            )),
            (0, (
                "00011000", "01111100", "11111110", "11100110",
                "11000111", "11000111", "11000111", "11000110",
                "11000110", "11101110", "01111110", "00111000",
            )),
        )
        for digit, rows in extra_literals:
            glyph = np.array([[255 if cell == "1" else 0 for cell in row] for row in rows], dtype=np.uint8)
            templates[digit].append(self._normalize(glyph))
        return templates

    def _recognize(self, mask: np.ndarray) -> tuple[int | None, float]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        glyphs: list[tuple[int, np.ndarray]] = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if height >= self.config.min_digit_height and area >= 25:
                glyphs.append((x, mask[y : y + height, x : x + width]))
        glyphs.sort(key=lambda item: item[0])
        if not glyphs or len(glyphs) > 7:
            return None, 0.0
        digits: list[int] = []
        confidences: list[float] = []
        for _, glyph in glyphs:
            normalized = self._normalize(glyph)
            scores: dict[int, float] = {}
            for digit, variants in self.templates.items():
                scores[digit] = min(float(np.mean(np.abs(normalized.astype(np.float32) - template.astype(np.float32)))) for template in variants)
            ordered = sorted(scores.items(), key=lambda item: item[1])
            digits.append(ordered[0][0])
            separation = max(0.0, ordered[1][1] - ordered[0][1])
            confidences.append(min(1.0, separation / 18.0))
        return int("".join(map(str, digits))), float(np.mean(confidences))

    def update(self, frame: np.ndarray) -> ScoreResult:
        cfg = self.config
        crop = frame[cfg.top : cfg.bottom, cfg.left : cfg.right]
        self.last_crop = crop.copy()
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = (((hsv[:, :, 2] >= 100) & (hsv[:, :, 1] <= 180)).astype(np.uint8) * 255)
        small = cv2.resize(mask, (30, 16), interpolation=cv2.INTER_AREA)
        fingerprint = np.packbits(small >= 80).tobytes().hex()
        cached = self.ocr_cache.get(fingerprint)
        if cached is None:
            cached = self._recognize(mask)
            self.ocr_cache[fingerprint] = cached
        candidate_value, confidence = cached
        value = candidate_value
        key = f"{value}:{fingerprint}"
        if key == self.candidate_key:
            self.candidate_count += 1
        else:
            self.candidate_key = key
            self.candidate_count = 1

        changed = False
        delta: int | None = None
        if self.candidate_count >= cfg.stable_frames and key != self.stable_key:
            previous = self.stable_value
            self.stable_key = key
            changed = previous is not None
            # Total score is monotonic during a run. Reject cropped/misread
            # lower values without suppressing the font-independent event.
            reliable = value is not None and confidence >= 0.12
            if reliable and (previous is None or value >= previous):
                self.stable_value = value
                if previous is not None:
                    candidate_delta = value - previous
                    if candidate_delta <= cfg.max_delta_per_event:
                        delta = candidate_delta
        return ScoreResult(self.stable_value, delta, changed, confidence, fingerprint)

    def sample_image(self) -> np.ndarray | None:
        return None if self.last_crop is None else self.last_crop.copy()
