"""The window: class App, assembled from the mixins of the app_*.py modules (each holds one area of its methods, which use
`self` and the attributes __init__ creates), and main(). All of the application's state is created in App.__init__.

The program's feature description is the docstring of the package (lightbounces/__init__.py)."""


import os
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import numpy as np


from .image_io import make_grid
from .kernels import N_POINTS, N_RAW, N_STATS, ObstacleArrays, SIZE
from .constants import (
    BACKGROUND, DEFAULT_CONFINE_THRESHOLD, DEFAULT_CONFINE_WINDOW, DEFAULT_DIST,
    DEFAULT_ESCAPE_RADIUS,
    DEFAULT_EMIT_TARGET_CX, DEFAULT_EMIT_TARGET_CY, DEFAULT_EMIT_TARGET_R,
    DEFAULT_EMIT_TARGET_THETA, DEFAULT_FUZZ_SEED, DEFAULT_ITER, EMIT_TARGET_FIELDS, ERROR_RED,
    ITER_LIMIT_MAX, OBST_BOUND_LEN, REFRESH_HZ
)
from .settings import (
    DEFAULT_FUZZ, DEFAULT_NATURAL_STOP_SETTINGS, DEFAULT_RAY_SETTINGS, EMISSION_DEFAULT, NATURAL_STOP_DEFAULT,
    fuzz_rule, ray_args
)
from .shapes import default_starting_obstacle
from .metrics import make_mapping
from .views import ViewConfig
from .app_projects import ProjectsMixin
from .app_export import ExportMixin
from .app_render import RenderMixin
from .app_video import VideoMixin
from .app_timeline import TimelineMixin
from .app_debug import DebugMixin
from .app_generation import GenerationMixin
from .app_preview import PreviewMixin
from .app_recolor import RecolorMixin
from .app_obstacles import ObstaclesMixin
from .app_coloration import ColorationMixin
from .app_controls import ControlsMixin
from .app_layout import LayoutMixin


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App(ProjectsMixin, ExportMixin, RenderMixin, VideoMixin, TimelineMixin, DebugMixin, GenerationMixin,
          PreviewMixin, RecolorMixin, ObstaclesMixin, ColorationMixin, ControlsMixin, LayoutMixin):
    PERIOD = 1.0 / REFRESH_HZ

    def __init__(self, root):
        self.root = root
        root.title(os.path.basename(sys.argv[0]) or "Light bounces")
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

        # ---- state ---- (defined before any widget can call back into us)
        self.ready = False                # numba kernels compiled?
        self.running = False              # generation in progress?
        self.done = False                 # image fully generated?
        self.generated_settings = None    # ray settings the current image was started with
        self.generated_natural_stop = None    # ...and its natural stopping settings (None together)
        self.baseline = DEFAULT_RAY_SETTINGS  # what "Revert" restores
        self.baseline_natural_stop = DEFAULT_NATURAL_STOP_SETTINGS
        # What the current image was generated with is generated_settings and
        # generated_natural_stop (see settings_of_current_image); gen_stop and gen_ray are only
        # the kernel-ready tuples made from them (and from the obstacles, see below).
        # The Buddha blur is not part of that: it is applied whenever Buddha channels are
        # colored, so the image always uses the blur controls' current values.
        self.gen_stop = NATURAL_STOP_DEFAULT
        self.last_iter = DEFAULT_ITER     # most recent valid entry values
        self.last_dist = DEFAULT_DIST
        self.last_escape_radius = DEFAULT_ESCAPE_RADIUS
        self.last_emit_target_r = DEFAULT_EMIT_TARGET_R
        self.last_emit_target_theta = DEFAULT_EMIT_TARGET_THETA
        self.last_emit_target_cx = DEFAULT_EMIT_TARGET_CX
        self.last_emit_target_cy = DEFAULT_EMIT_TARGET_CY
        self.last_confine_window = DEFAULT_CONFINE_WINDOW
        self.last_confine_threshold = DEFAULT_CONFINE_THRESHOLD
        self.last_seed = DEFAULT_FUZZ_SEED
        self.views = [ViewConfig(0)]      # list[ViewConfig], in export/composition order
        self._view_counter = 1
        self.selected_view_id = 0         # which view is selected - both for editing in
                                           # the Coloration tab AND shown in the preview
        self.mapping = make_mapping(self.live_view().kinds, self.live_view().point_metric,
                                    self.live_view().other_metric, self.live_view().chan_src,
                                    self.live_view().constant_choice)
        self.buddha_counts = [None, None, None]   # per channel of the live view: (SIZE, SIZE)
                                                   # float64 RAW (unblurred) visit counts,
                                                   # or None while unused/before Start
        self.buddha_counts_blurred = [None, None, None]   # per channel: the lazily-computed,
                                                           # cached blurred version - see
                                                           # blurred_buddha_counts()
        self.view_thumb_cache = {}        # non-live view id -> (rows_done, thumbnail array)
                                          # at the time it was computed - see
                                          # render_view_thumbnail(). The live view is
                                          # never cached here; it always reuses self.img
                                          # directly instead.
        self.deco = {}                    # decoration key -> BooleanVar
        self.rows_done = 0
        self.batch_rows = 1
        self.tick_id = None
        self.outline_job = None
        self.blur_job = None              # a pending refresh_blur (slider events coalesce)
        self.seed_job = None              # a pending refresh_buddha_seeds (changes of one moment coalesce)
        self.start_time = 0.0
        self.elapsed = 0.0
        self.next_tick = 0.0
        self.hover_px = None
        self.pinned_px = None             # pixel pinned by a click, or None
        self.fuzz_tk_widgets = []         # classic Tk widgets in the Fuzzing section of the Mutations tab
        self.fuzz_ttk_widgets = []        # ttk widgets in the Fuzzing section of the Mutations tab
        self.hover_pts = np.empty((ITER_LIMIT_MAX + 2, 2))
        self.hover_kinds = np.empty(ITER_LIMIT_MAX + 2)
        self.hover_raw = np.empty(N_RAW)
        self.hover_vec = np.empty(N_STATS)
        self.hover_sp = np.empty((N_POINTS, 2))

        # ---- timeline / video state ----
        self.keyframes = []               # list[Keyframe], in the order shown
        self._kf_counter = 0
        self.selected_keyframe = None
        self.kf_modified = set()          # setting keys the user changed since the selected
                                          # keyframe was selected / last saved (blue outlines)
        self.kf_last_seen = None          # capture_all_settings() at the previous check
        self.kf_applying = False          # True while a keyframe's values are being loaded
        self.outlines = {}                # widget -> ModifiedOutline
        self.diff_expanded = set()        # titles of the expanded sections of the keyframe
                                          # settings list (everything starts collapsed)
        self.render_thread = None
        self.render_cancel = None

        # ---- obstacles ---- (the scene: not part of Settings, but every obstacle's
        # own fields are keyframeable). The app starts with one obstacle: the
        # unit circle, which reflects rays and fires them inside itself.
        self.obstacles = [default_starting_obstacle(0)]   # list[Shape], in list order
        self._obst_counter = 1
        self.selected_obstacle = self.obstacles[0]
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_bounds = np.empty((0, OBST_BOUND_LEN), dtype=np.float64)
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=np.bool_)
        self.gen_ray = ray_args(              # the ray engine's arguments as generated (see start), with the
            DEFAULT_RAY_SETTINGS.seg_mult, fuzz_rule(DEFAULT_FUZZ), EMISSION_DEFAULT,   # obstacles as generated
            ObstacleArrays(shapes=empty_i64, bounds=empty_bounds, cx=empty_f64, cy=empty_f64,
                           effects=np.empty((0, 2), dtype=np.int64), fire_inside=empty_bool))
        self.gen_obstacles_sig = ()
        self.gen_obstacle_snapshots = []   # frozen copies of the obstacles as generated,
                                           # for the export's figure settings text
        self.gen_obstacles_by_id = {}      # obstacle id -> frozen settings, for the
                                           # dashed "changed since last generation" guide

        # ---- per-pixel data ----
        self.stats = np.zeros((SIZE, SIZE, N_STATS), dtype=np.float32)
        self.points = np.zeros((SIZE, SIZE, N_POINTS, 2), dtype=np.float32)
        self.status = np.zeros((SIZE, SIZE), dtype=np.uint8)   # 0 = not computed yet

        # ---- image buffers ----
        self.img = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
        self.img[:] = BACKGROUND
        self.frame = np.empty_like(self.img)      # img + white outline; this is what
                                                   # gets exported/saved, so it must never
                                                   # hold a preview-only decoration
        self.display_frame = np.empty_like(self.img)   # self.frame + the view-name
                                                        # overlay; only ever pushed to
                                                        # self.photo, never written to disk

        # World coordinates of every pixel centre (also each ray's start point)
        self.gx, self.gy = make_grid(SIZE, SIZE)

        self.build_image_quadrant()
        self.build_controls_quadrant()
        self.build_timeline_panel()
        self.build_info_quadrant()
        self.build_status_quadrant()
        self.build_video_panel()
        self.update_fuzz_enabled_state()

        # Watch the number entries (registered last, once everything exists)
        self.iter_var.trace_add("write", self.on_limit_var_changed)
        self.dist_var.trace_add("write", self.on_limit_var_changed)
        self.escape_radius_var.trace_add("write", self.on_limit_var_changed)
        self.confine_window_var.trace_add("write", self.on_limit_var_changed)
        self.confine_threshold_var.trace_add("write", self.on_limit_var_changed)
        self.confine_method_var.trace_add("write", self.on_confine_method_changed)
        for field in EMIT_TARGET_FIELDS:
            getattr(self, f"emit_target_{field}_var").trace_add(
                "write", self.on_limit_var_changed)
        self.emit_mode_var.trace_add("write", self.on_emit_mode_changed)
        self.seed_var.trace_add("write", self.on_seed_var_changed)

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
