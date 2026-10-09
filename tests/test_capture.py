from __future__ import annotations

import unittest

import cv2
import numpy as np

from lockpick_ai.capture import detect_content_rect, fitted_region, scaled_region
from lockpick_ai.config import ROIConfig


class CaptureScalingTests(unittest.TestCase):
    def test_2560x1440_scales_reference_roi(self):
        region = scaled_region(ROIConfig(644, 62, 635, 956), 0, 0, 2560, 1440, 1920, 1080)
        self.assertEqual(region, {"left": 859, "top": 83, "width": 847, "height": 1275})

    def test_secondary_monitor_origin_is_preserved(self):
        region = scaled_region(ROIConfig(100, 50, 200, 100), 1920, -100, 1920, 1080, 1920, 1080)
        self.assertEqual(region, {"left": 2020, "top": -50, "width": 200, "height": 100})

    def test_window_client_is_center_fitted_when_not_16_by_9(self):
        region = fitted_region(ROIConfig(644, 62, 635, 956), 100, 50, 1920, 1200, 1920, 1080)
        self.assertEqual(region, {"left": 744, "top": 172, "width": 635, "height": 956})

    def test_non_16_by_9_display_is_rejected(self):
        with self.assertRaises(ValueError):
            scaled_region(ROIConfig(), 0, 0, 1920, 1200, 1920, 1080)

    def test_detects_minigame_frame_inside_oversized_capture(self):
        frame = np.zeros((956, 635, 3), dtype=np.uint8)
        left, top, width = 75, 90, 480
        height = round(width * 956 / 635)
        cv2.line(frame, (left, top), (left, top + height - 1), (0, 0, 255), 6)
        cv2.line(frame, (left + width - 1, top), (left + width - 1, top + height - 1), (0, 0, 255), 6)
        cv2.line(frame, (left, top), (left + width - 1, top), (0, 190, 255), 6)
        detected = detect_content_rect(frame, 635, 956)
        self.assertIsNotNone(detected)
        x, y, found_width, found_height = detected or (0, 0, 0, 0)
        self.assertAlmostEqual(x, left, delta=5)
        self.assertAlmostEqual(y, top, delta=5)
        self.assertAlmostEqual(found_width, width, delta=8)
        self.assertAlmostEqual(found_height, height, delta=12)

    def test_already_aligned_frame_is_not_recropped(self):
        frame = np.zeros((956, 635, 3), dtype=np.uint8)
        cv2.line(frame, (0, 0), (0, 955), (0, 0, 255), 5)
        cv2.line(frame, (634, 0), (634, 955), (0, 0, 255), 5)
        self.assertEqual(detect_content_rect(frame, 635, 956), (0, 0, 635, 956))


if __name__ == "__main__":
    unittest.main()
