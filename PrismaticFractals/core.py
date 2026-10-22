"""The window: class App, assembled from the mixins of the app_*.py modules (each holds one area of its methods, which use
`self` and the attributes their area's init_<area>_state method creates), and main(). Each area creates its own state (App.__init__ calls them
in an order that matters only where one needs another).

The program's feature description is the docstring of the package (PrismaticFractals/__init__.py)."""


import os
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from .constants import ERROR_RED, REFRESH_HZ
from .app_projects import ProjectsMixin
from .app_export import ExportMixin
from .app_engine import EngineFlowMixin
from .engines import EMPTY_ENGINE_MEMORY
from .app_framing import FramingMixin
from .app_render import RenderMixin
from .app_video import VideoMixin
from .app_timeline import TimelineMixin
from .app_debug import DebugMixin
from .app_generation import GenerationMixin
from .app_preview import PreviewMixin
from .app_recolor import RecolorMixin
from .app_obstacles import ObstaclesMixin
from .app_coloration import ColorationMixin
from .app_buddha import BuddhaMixin
from .app_controls import ControlsMixin
from .app_layout import LayoutMixin


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App(ProjectsMixin, EngineFlowMixin, ExportMixin, FramingMixin, RenderMixin, VideoMixin, TimelineMixin, DebugMixin, GenerationMixin,
          PreviewMixin, RecolorMixin, BuddhaMixin, ObstaclesMixin, ColorationMixin, ControlsMixin, LayoutMixin):
    PERIOD = 1.0 / REFRESH_HZ

    def __init__(self, root):
        self.root = root
        root.title(os.path.basename(sys.argv[0]) or "Prismatic Fractals")
        root.resizable(False, False)
        self.build_menu_bar()
        self._crash_dumps = 0
        root.report_callback_exception = self.on_callback_exception
        self.install_crash_hooks()
        root.after(100, self.keep_window_on_screen)   # once the window has its real size

        # Slightly smaller font for the dense metrics list
        base_font = tkfont.nametofont("TkDefaultFont")
        self.small_font = base_font.copy()
        size = base_font.cget("size")
        self.small_font.configure(size=size - 1 if size > 0 else size + 1)
        self.small_bold_font = self.small_font.copy()
        self.small_bold_font.configure(weight="bold")

        # Style used to flag an invalid number entry
        ttk.Style().configure("Invalid.TEntry", foreground=ERROR_RED)

        # ---- state ---- (each area creates its own, before any widget can call back into us; the order matters only where one
        # needs another: the settings model first, the colorations before anything that reads the live one)
        self.built = set()                # names of widgets that exist yet: what a method called while the window is still being built asks
        self.init_controls_state()
        self.init_engine_state()
        self.init_coloration_state()
        self.init_generation_state()
        self.init_buddha_state()
        self.init_preview_state()
        self.init_timeline_state()
        self.init_video_state()
        self.init_obstacle_state()
        self.init_framing_state()

        # ---- the image on screen ---- a LiveImage: its framing, size in pixels, the world rectangle it covers, the per-pixel data
        # and the colours (see set_image_framing)
        self.init_live_image()

        self.build_image_quadrant()
        self.build_controls_quadrant()
        self.build_timeline_panel()
        self.build_info_quadrant()
        self.build_status_quadrant()
        self.build_video_panel()
        self.update_fuzz_enabled_state()

        # Watch the number entries (registered last, once everything exists)
        self.bind_model_controls()
        self.confine_method_var.trace_add("write", self.on_confine_method_changed)

        self.draw()
        self.update_buttons()
        self.start_project_handling()
        root.update()                              # show the window before compiling
        root.after(20, self.warm_up)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
