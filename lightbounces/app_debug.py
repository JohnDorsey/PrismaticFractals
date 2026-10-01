"""The Debug menu, crash hooks and startup captures (screenshot and state dump, see debug.py), and keeping the window on screen at startup."""
import sys
import threading
import traceback
import tkinter as tk
from tkinter import messagebox
from .constants import MAX_CRASH_DUMPS
from .debug import take_ui_screenshot
from .window import parse_geometry_position, shift_into_area, window_rect_and_work_area
from . import debug


class DebugMixin:
    def build_menu_bar(self):
        """The menu bar: File (projects) and Debug (the dev aids; they do nothing
        unless the dev/ folder exists)."""
        bar = tk.Menu(self.root, tearoff=False)
        file_menu = tk.Menu(bar, tearoff=False)
        file_menu.add_command(label="New project", accelerator="Ctrl+N",
                              command=self.new_project)
        file_menu.add_command(label="Open project...", accelerator="Ctrl+O",
                              command=self.open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Save project", accelerator="Ctrl+S",
                              command=self.save_project)
        file_menu.add_command(label="Save project as...", accelerator="Ctrl+Shift+S",
                              command=self.save_project_as)
        bar.add_cascade(label="File", menu=file_menu)
        debug = tk.Menu(bar, tearoff=False)
        debug.add_command(label="Take screenshot", command=self.manual_screenshot)
        debug.add_command(label="Dump (without large arrays)",
                          command=lambda: self.manual_dump(False))
        debug.add_command(label="Dump (with large arrays)",
                          command=lambda: self.manual_dump(True))
        bar.add_cascade(label="Debug", menu=debug)
        self.root.config(menu=bar)

    def report_debug_result(self, what, path):
        if path is None:
            messagebox.showinfo("Debug", f"No {what} was saved (it needs the dev folder "
                                         "next to the program" +
                                (", and Pillow" if what == "screenshot" else "") + ").")
        else:
            messagebox.showinfo("Debug", f"Saved {what}:\n{path}")

    def manual_screenshot(self):
        try:
            path = take_ui_screenshot(self.root, "manual")
        except Exception as exc:                   # noqa: BLE001
            print(f"Screenshot failed: {exc}", file=sys.stderr)
            path = None
        self.report_debug_result("screenshot", path)

    def manual_dump(self, include_large_arrays):
        self.report_debug_result("dump", debug.dump_app_state(self, "manual", include_large_arrays))

    def on_callback_exception(self, exc_type, exc_value, exc_tb):
        """Tk calls this for an exception escaping any callback (replacing its
        default, which only prints it). Prints it the same way, then leaves a crash
        dump (without large arrays; never a screenshot), at most MAX_CRASH_DUMPS
        per launch."""
        print("Exception in Tkinter callback", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.dump_crash((exc_type, exc_value, exc_tb))

    def dump_crash(self, exc_info):
        if self._crash_dumps >= MAX_CRASH_DUMPS:
            return
        self._crash_dumps += 1
        debug.dump_app_state(self, "crash", False, exc_info)

    def install_crash_hooks(self):
        """Also dump on an exception that ends the program (sys.excepthook) or a
        background thread (run later on the main thread, since Tk variables
        must not be read from another one). The previous hooks still run."""
        previous_hook, previous_thread_hook = sys.excepthook, threading.excepthook

        def hook(exc_type, exc_value, exc_tb):
            self.dump_crash((exc_type, exc_value, exc_tb))
            previous_hook(exc_type, exc_value, exc_tb)

        def thread_hook(args):
            info = (args.exc_type, args.exc_value, args.exc_traceback)
            try:
                self.root.after(0, lambda: self.dump_crash(info))
            except Exception:                      # noqa: BLE001
                pass
            previous_thread_hook(args)

        sys.excepthook = hook
        threading.excepthook = thread_hook

    def keep_window_on_screen(self):
        """Move the window (never resize it) so all of it is on the screen it is
        on, e.g. when the operating system placed it with its bottom edge below the
        screen. A window bigger than the screen ends up with its top-left corner at
        the screen's. Startup only; never raises."""
        try:
            self.root.update_idletasks()
            rect, area = window_rect_and_work_area(self.root)
            dx, dy = shift_into_area(rect, area)
            position = parse_geometry_position(self.root.geometry())
            if (dx or dy) and position is not None:
                self.root.geometry(f"+{position[0] + dx}+{position[1] + dy}")
                self.root.update_idletasks()
        except Exception as exc:                    # noqa: BLE001
            print(f"Could not reposition the window: {exc}", file=sys.stderr)

    def take_startup_captures(self):
        """Once per launch, shortly after the window settles: a screenshot, then a
        dump without large arrays."""
        self.keep_window_on_screen()                # so the screenshot shows all of it
        self.take_startup_screenshot()
        if getattr(self, "_startup_dump_taken", False):
            return
        self._startup_dump_taken = True
        debug.dump_app_state(self, "startup", False)

    def take_startup_screenshot(self):
        """At most once per launch (see take_ui_screenshot for the conditions).
        A development aid, so any failure is reported and otherwise ignored."""
        if getattr(self, "_startup_screenshot_taken", False):
            return
        self._startup_screenshot_taken = True
        try:
            take_ui_screenshot(self.root, "startup")
        except Exception as exc:                   # noqa: BLE001
            print(f"Startup screenshot failed: {exc}", file=sys.stderr)
