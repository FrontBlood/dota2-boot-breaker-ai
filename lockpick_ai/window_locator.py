from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from ctypes import wintypes


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


@dataclass(frozen=True)
class WindowBounds:
    hwnd: int
    left: int
    top: int
    width: int
    height: int
    executable: str
    title: str


def refresh_window_bounds(previous: WindowBounds) -> WindowBounds | None:
    """Refresh a known HWND without repeating process enumeration."""
    if os.name != "nt":
        return None
    user32 = ctypes.windll.user32
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    hwnd = wintypes.HWND(previous.hwnd)
    if not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        return None
    client = wintypes.RECT()
    origin = wintypes.POINT(0, 0)
    if not user32.GetClientRect(hwnd, ctypes.byref(client)) or not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    width = int(client.right - client.left)
    height = int(client.bottom - client.top)
    if width < 640 or height < 360:
        return None
    title_length = user32.GetWindowTextLengthW(hwnd)
    title_buffer = ctypes.create_unicode_buffer(title_length + 1)
    user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
    return WindowBounds(
        previous.hwnd,
        int(origin.x),
        int(origin.y),
        width,
        height,
        previous.executable,
        title_buffer.value,
    )


def find_process_window(executable: str) -> WindowBounds | None:
    """Return the largest visible, non-minimized client window for an EXE."""
    if os.name != "nt":
        return None
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    matches: list[WindowBounds] = []
    expected = executable.casefold()

    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    @callback_type
    def visit(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id.value)
        if not handle:
            return True
        try:
            path_buffer = ctypes.create_unicode_buffer(32768)
            path_size = wintypes.DWORD(len(path_buffer))
            if not kernel32.QueryFullProcessImageNameW(handle, 0, path_buffer, ctypes.byref(path_size)):
                return True
            process_name = os.path.basename(path_buffer.value).casefold()
            if process_name != expected:
                return True
        finally:
            kernel32.CloseHandle(handle)

        client = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(client)):
            return True
        origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return True
        width = int(client.right - client.left)
        height = int(client.bottom - client.top)
        if width < 640 or height < 360:
            return True
        title_length = user32.GetWindowTextLengthW(hwnd)
        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
        matches.append(
            WindowBounds(
                int(hwnd),
                int(origin.x),
                int(origin.y),
                width,
                height,
                executable,
                title_buffer.value,
            )
        )
        return True

    user32.EnumWindows(visit, 0)
    return max(matches, key=lambda item: item.width * item.height, default=None)
