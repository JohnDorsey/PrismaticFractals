"""Keeping the program window on screen at startup: where the window is, where the usable part of its monitor is (the work area, taskbar excluded), and the smallest move that brings it inside."""
import re
import sys


def shift_into_area(rect, area):
    """The smallest (dx, dy) that moves `rect` (left, top, right, bottom) inside
    `area` (same form). A rect larger than the area is lined up by its left and
    top edges instead, so the title bar stays reachable."""
    left, top, right, bottom = rect
    a_left, a_top, a_right, a_bottom = area
    dx = dy = 0
    if right > a_right:
        dx = a_right - right
    if bottom > a_bottom:
        dy = a_bottom - bottom
    if left + dx < a_left:
        dx = a_left - left
    if top + dy < a_top:
        dy = a_top - top
    return dx, dy


def parse_geometry_position(geometry):
    """(x, y) from a Tk geometry string like "1904x936+0+104" or "1904x936-8+-3";
    None if it has no position."""
    match = re.fullmatch(r"\d+x\d+([+-])(-?\d+)([+-])(-?\d+)", geometry)
    if match is None:
        return None
    sign_x, x, sign_y, y = match.groups()
    return (int(x) if sign_x == "+" else -int(x)), (int(y) if sign_y == "+" else -int(y))


def window_rect_and_work_area(root):
    """((left, top, right, bottom) of the whole window including its frame, and the
    (left, top, right, bottom) of the usable area - the work area, taskbar excluded -
    of the monitor it is mostly on), in screen pixels. On Windows both come from
    the operating system; elsewhere the window's client rectangle and the whole
    screen are the best available."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class MonitorInfo(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                            ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(root.winfo_id())
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            info = MonitorInfo()
            info.cbSize = ctypes.sizeof(MonitorInfo)
            monitor = user32.MonitorFromWindow(hwnd, 2)       # MONITOR_DEFAULTTONEAREST
            user32.GetMonitorInfoW(monitor, ctypes.byref(info))
            work = info.rcWork
            return ((rect.left, rect.top, rect.right, rect.bottom),
                    (work.left, work.top, work.right, work.bottom))
        except Exception:                           # noqa: BLE001
            pass
    x, y = root.winfo_rootx(), root.winfo_rooty()
    return ((x, y, x + root.winfo_width(), y + root.winfo_height()),
            (0, 0, root.winfo_screenwidth(), root.winfo_screenheight()))
