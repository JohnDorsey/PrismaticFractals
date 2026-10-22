"""Generating the image: compiling the kernels at startup, Start, the tick loop that traces rows within each refresh period, and what the generated image was made with."""
import time
import numpy as np
from .constants import (
    COMPUTE_FRACTION, DEFAULT_HEIGHT, DEFAULT_ROTATION, DEFAULT_SEGMENT_MULT,
    DEFAULT_ESCAPE_RADIUS, DEFAULT_WIDTH, DIST_LIMIT_MAX, OBST_BOUND_LEN, WARM_ITER
)
from .geometry import ellipse_bound
from .kernels import ObstacleArrays, SHAPE_ELLIPSE, fires_ray, inside_boundary, measure_path, trace_rows, trace_engine_path, ENGINE_RAY_BOUNCING
from .metrics import CHAN_SRC_BUDDHA
from .settings import (
    CONFINE_DEFAULT, DEFAULT_FUZZ, EMISSION_DEFAULT, fuzz_rule, natural_termination_rule,
    path_args, ray_args
)
from .figure import PLACEHOLDER_FIGURE, generated_figure
from .framings import snapshot_framing
from .live_image import LiveImage


class GenerationMixin:
    def init_generation_state(self):
        """Whether the kernels are compiled, whether a generation is running or done, the figure the image on screen was generated
        from (None until the first Start: see current_figure) and the tick loop's timing. The Buddha blur is not part of the figure:
        it is applied whenever Buddha channels are colored, so the image always uses the blur controls' current values."""
        self.ready = False
        self.running = False
        self.done = False
        self.generated = None
        self.tick_id = None
        self.start_time = 0.0
        self.elapsed = 0.0
        self.next_tick = 0.0

    # ---- generation ------------------------------------------------------- #
    def init_live_image(self):
        """The image on screen: a blank one of the first framing, laid out for the figure before the first Start (the placeholder one), and
        the window's buffers for decorating it. Made after the framings, and before anything that reads it."""
        self.live = LiveImage(snapshot_framing(self.framings[0]), self.current_figure().layout)
        self.allocate_screen_buffers()

    def warm_up(self):
        """Trigger numba compilation (slow only the first time; then cached)."""
        big = float(DIST_LIMIT_MAX)
        bound = ellipse_bound(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_ROTATION)
        fz = fuzz_rule(DEFAULT_FUZZ)
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_bounds = np.empty((0, OBST_BOUND_LEN), dtype=np.float64)
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=np.bool_)
        inside_boundary(0.0, 0.0, SHAPE_ELLIPSE, bound)
        no_obstacles = ObstacleArrays(
            shapes=empty_i64, bounds=empty_bounds, cx=empty_f64, cy=empty_f64,
            effects=np.empty((0, 2), dtype=np.int64), fire_inside=empty_bool)
        fires_ray(0.0, 0.0, no_obstacles)
        termination = natural_termination_rule(WARM_ITER, big, DEFAULT_ESCAPE_RADIUS, CONFINE_DEFAULT)
        engine_args = path_args(ENGINE_RAY_BOUNCING, ray_args(DEFAULT_SEGMENT_MULT, fz, EMISSION_DEFAULT), no_obstacles)
        trace_engine_path(0.0, 0.0, self.hover_path_points, self.hover_path_point_tags, termination, engine_args, self.hover_raw)
        measure_path(ENGINE_RAY_BOUNCING, self.hover_path_points, 4, self.hover_path_point_tags, self.hover_raw, WARM_ITER,
                     DEFAULT_SEGMENT_MULT, engine_args.tables, self.hover_vec, self.hover_ray_characteristic_points)
        trace_rows(self.live.stats, self.live.characteristic_points, self.live.pixel_state, 0, 1, self.live.img_w, self.live.img_h, self.live.world_rect,
                       termination, engine_args, 0)
        self.live.reset(self.live.framing, self.current_figure().layout)           # (blank again)
        self.ready = True
        self.update_buttons()
        self.root.after(500, self.take_startup_captures)   # let the UI settle first

    def start(self):
        """(Re)start generation using the current settings."""
        self.correct_controls_for_render()
        snap = self.current_generation_settings()
        if snap is None:
            return
        if self.tick_id is not None:
            self.root.after_cancel(self.tick_id)
            self.tick_id = None

        settings, natural_termination = snap
        engine = self.current_engine_settings()
        mandelbrot = self.current_mandelbrot_settings()
        self.generated = generated_figure(settings, natural_termination, self.obstacles, engine, mandelbrot)
        self.baseline = settings                   # Revert now returns to these values
        self.baseline_natural_termination = natural_termination
        self.baseline_engine = engine
        self.baseline_mandelbrot = mandelbrot
        self.begin_generation()

    def begin_generation(self):
        """Generate, from its first row, an image of the figure this program holds (the GeneratedFigure
        `generated`: set by start, which is what makes a new figure) as seen by the
        selected framing - which is also what a change of framing does, with the same figure."""
        if self.tick_id is not None:
            self.root.after_cancel(self.tick_id)
            self.tick_id = None
        self.canvas.delete("pending")
        self.set_image_framing(self.selected_framing_snapshot())    # (also clears everything computed so far)
        self.rebuild_buddha_counts()
        self.thumb_sims = {}
        self.coloration_thumb_cache = {}   # every non-live coloration's cached thumbnail is for the
                                     # previous generation's data - none of it applies here
        self.done = False
        self.running = True
        if self.pinned_px is None:
            self.clear_overlay()
        self.start_time = time.perf_counter()
        self.elapsed = 0.0
        self.next_tick = self.start_time
        self.draw()
        self.update_buttons()
        self.tick()

    def tick(self):
        """Compute for a slice of the refresh period, redraw, reschedule."""
        t_start = time.perf_counter()
        rows_before = self.live.advance(self.current_figure(), self.PERIOD * COMPUTE_FRACTION)

        self.scatter_buddha_rows(rows_before, self.live.rows_done)
        if any(s == CHAN_SRC_BUDDHA for s in self.live_coloration().chan_src):
            self.recolor(0, self.live.rows_done)         # a Buddha channel's counts can
                                                     # affect any already-drawn pixel
        else:
            self.recolor(rows_before, self.live.rows_done)
        self.elapsed = time.perf_counter() - self.start_time
        self.draw()

        if self.live.rows_done >= self.live.img_h:
            self.on_finished()
            return

        # Keep a steady 10 Hz cadence; idle time lets Tk handle mouse events
        now = time.perf_counter()
        self.next_tick += self.PERIOD
        if self.next_tick < now:
            self.next_tick = now
        delay_ms = max(1, int((self.next_tick - now) * 1000))
        self.tick_id = self.root.after(delay_ms, self.tick)

    def on_finished(self):
        self.tick_id = None
        self.running = False
        self.done = True
        self.update_buttons()
        if self.pinned_px is not None:             # refresh the pinned preview
            self.hover_px = None                   # with the settings just used
            self.update_preview(*self.pinned_px)
        self.rebuild_coloration_list()                   # every coloration's thumbnail reflects the
                                                    # simulation that just finished, even
                                                    # if the Colorations tab isn't showing

    def current_figure(self):
        """The GeneratedFigure the image on screen was generated from (the PLACEHOLDER_FIGURE - the defaults and no
        obstacle - before the first Start). Coloring a coloration from the stored data needs all of it, not just the
        limits: Buddha coloration re-traces rays with the generated emission rule, fuzzing and obstacles."""
        return PLACEHOLDER_FIGURE if self.generated is None else self.generated

    def settings_of_current_image(self):
        """The RaySettings the current image was generated with (the defaults, for the placeholder image before the
        first Start)."""
        return self.current_figure().ray

    def natural_termination_of_current_image(self):
        """The NaturalTerminationSettings the current image was generated with (the defaults, for the placeholder
        image before the first Start)."""
        return self.current_figure().natural_termination
