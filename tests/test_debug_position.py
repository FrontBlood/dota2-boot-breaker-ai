from __future__ import annotations

import unittest

from boot_breaker.debug_view import choose_window_position


class DebugPositionTests(unittest.TestCase):
    def test_places_monitor_to_right_of_capture(self):
        capture = {"left": 644, "top": 62, "width": 635, "height": 956}
        desktop = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        self.assertEqual(choose_window_position(capture, desktop, 457, 688), (1289, 62))

    def test_uses_left_side_when_right_has_no_room(self):
        capture = {"left": 700, "top": 100, "width": 600, "height": 700}
        desktop = {"left": 0, "top": 0, "width": 1400, "height": 900}
        self.assertEqual(choose_window_position(capture, desktop, 400, 600), (290, 100))


if __name__ == "__main__":
    unittest.main()
