from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass


VK_ESCAPE = 0x1B
VK_F8 = 0x77
VK_F9 = 0x78
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_A = 0x41
VK_D = 0x44


class MouseOutput:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("Mouse output is currently implemented for Windows only")
        self.user32 = ctypes.windll.user32
        self.right_is_down = False

    def click_left(self) -> None:
        self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def set_right(self, down: bool) -> None:
        if down == self.right_is_down:
            return
        flag = MOUSEEVENTF_RIGHTDOWN if down else MOUSEEVENTF_RIGHTUP
        self.user32.mouse_event(flag, 0, 0, 0, 0)
        self.right_is_down = down

    def release_all(self) -> None:
        self.set_right(False)


@dataclass
class HotkeyEdges:
    previous: dict[int, bool]

    def __init__(self):
        self.previous = {}
        self.user32 = ctypes.windll.user32 if sys.platform == "win32" else None

    def pressed(self, key_code: int) -> bool:
        if self.user32 is None:
            return False
        current = bool(self.user32.GetAsyncKeyState(key_code) & 0x8000)
        result = current and not self.previous.get(key_code, False)
        self.previous[key_code] = current
        return result


class KeyboardOutput:
    def __init__(self, mode: str = "arrows"):
        if sys.platform != "win32":
            raise RuntimeError("Keyboard output is currently implemented for Windows only")
        self.user32 = ctypes.windll.user32
        self.left_key = VK_A if mode.lower() == "ad" else VK_LEFT
        self.right_key = VK_D if mode.lower() == "ad" else VK_RIGHT
        self.horizontal = 0
        self.space_is_down = False
        self.space_release_at = 0.0

    def _key(self, key_code: int, down: bool) -> None:
        flags = KEYEVENTF_EXTENDEDKEY if key_code in (VK_LEFT, VK_RIGHT) else 0
        if not down:
            flags |= KEYEVENTF_KEYUP
        self.user32.keybd_event(key_code, 0, flags, 0)

    def set_horizontal(self, direction: int) -> None:
        direction = -1 if direction < 0 else (1 if direction > 0 else 0)
        if direction == self.horizontal:
            return
        if self.horizontal < 0:
            self._key(self.left_key, False)
        elif self.horizontal > 0:
            self._key(self.right_key, False)
        if direction < 0:
            self._key(self.left_key, True)
        elif direction > 0:
            self._key(self.right_key, True)
        self.horizontal = direction

    def press_space(self, hold_ms: float = 90.0) -> None:
        if not self.space_is_down:
            self._key(VK_SPACE, True)
            self.space_is_down = True
        self.space_release_at = max(self.space_release_at, time.perf_counter() + hold_ms / 1000.0)

    def update(self, timestamp: float | None = None) -> None:
        now = time.perf_counter() if timestamp is None else timestamp
        if self.space_is_down and now >= self.space_release_at:
            self._key(VK_SPACE, False)
            self.space_is_down = False
            self.space_release_at = 0.0

    def release_all(self) -> None:
        self.set_horizontal(0)
        if self.space_is_down:
            self._key(VK_SPACE, False)
            self.space_is_down = False
            self.space_release_at = 0.0
