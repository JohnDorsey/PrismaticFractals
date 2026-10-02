"""Generating the image: compiling the kernels at startup, Start, the tick loop that traces rows within each refresh period, and what the generated image was made with."""
import time
import numpy as np
from .constants import (
    BACKGROUND, COMPUTE_FRACTION, DEFAULT_HEIGHT, DEFAULT_ROTATION, DEFAULT_SEGMENT_MULT,
    DEFAULT_WIDTH, DIST_LIMIT_MAX, OBST_BOUND_LEN, WARM_ITER
)
from .geometry import ellipse_bound
from .kernels import (
    ObstacleArrays, SHAPE_ELLIPSE, SIZE, compute_points, compute_stats, emits_ray,
    inside_boundary, render_rows, trace_path
)
from .metrics import CHAN_SRC_BUDDHA
from .settings import (
    CONFINE_DEFAULT, DEFAULT_FUZZ, DEFAULT_RAY_SETTINGS, EMISSION_DEFAULT, effective_dist,
    fuzz_rule, settings_confine, settings_emission
)
from .shapes import build_obstacle_arrays, obstacles_signature


class GenerationMixin:
    # ---- generation ------------------------------------------------------- #
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
        emits_ray(0.0, 0.0, no_obstacles)
        trace_path(0.0, 0.0, self.hover_pts, self.hover_kinds, WARM_ITER, big,
                   DEFAULT_SEGMENT_MULT, fz, CONFINE_DEFAULT, EMISSION_DEFAULT, no_obstacles,
                   self.hover_raw)
        compute_stats(self.hover_pts, 4, self.hover_raw, WARM_ITER,
                      DEFAULT_SEGMENT_MULT, self.hover_vec)
        compute_points(self.hover_pts, 4, self.hover_raw, self.hover_kinds, self.hover_sp)
        render_rows(self.stats, self.points, self.status, 0, 1, WARM_ITER, big,
                    DEFAULT_SEGMENT_MULT, fz, CONFINE_DEFAULT, EMISSION_DEFAULT, no_obstacles, 0)
        self.stats[:] = 0
        self.points[:] = 0
        self.status[:] = 0
        self.ready = True
        self.update_buttons()
        self.root.after(500, self.take_startup_captures)   # let the UI settle first

    def start(self):
        """(Re)start generation using the current settings."""
        snap = self.current_ray_settings()
        if snap is None:
            return
        if self.tick_id is not None:
            self.root.after_cancel(self.tick_id)
            self.tick_id = None

        settings = snap
        self.generated_settings = settings
        self.baseline = settings                   # Revert now returns to these values
        self.gen_fuzz = fuzz_rule(settings.fuzz)
        self.gen_confine = settings_confine(settings)
        self.gen_emission = settings_emission(settings)
        self.gen_obstacles = build_obstacle_arrays(self.obstacles)
        self.gen_obstacles_sig = obstacles_signature(self.obstacles)
        self.gen_obstacles_by_id = {o.id: o.state() for o in self.obstacles}
        self.gen_obstacle_snapshots = [o.snapshot() for o in self.obstacles]
        self.canvas.delete("pending")
        self.stats[:] = 0
        self.points[:] = 0
        self.status[:] = 0
        self.img[:] = BACKGROUND
        self.rows_done = 0
        self.batch_rows = 1
        self.rebuild_buddha_counts()
        self.view_thumb_cache = {}   # every non-live view's cached thumbnail is for the
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
        budget = self.PERIOD * COMPUTE_FRACTION
        target = budget / 3.0                      # aim for ~3 batches per tick
        rows_before = self.rows_done
        g = self.settings_of_current_image()
        max_dist = effective_dist(g.max_dist)

        while self.rows_done < SIZE:
            t0 = time.perf_counter()
            row1 = min(self.rows_done + self.batch_rows, SIZE)
            render_rows(self.stats, self.points, self.status, self.rows_done, row1,
                        g.max_iter, max_dist, g.seg_mult, self.gen_fuzz,
                        self.gen_confine, self.gen_emission, self.gen_obstacles,
                        g.fuzz.seed)
            self.rows_done = row1
            now = time.perf_counter()

            # Adapt batch size so each batch takes about `target` seconds
            dt = max(now - t0, 1e-6)
            scale = min(max(target / dt, 0.5), 2.0)
            self.batch_rows = max(1, int(self.batch_rows * scale))

            if now - t_start >= budget:
                break

        self.scatter_buddha_rows(rows_before, self.rows_done)
        if any(s == CHAN_SRC_BUDDHA for s in self.live_view().chan_src):
            self.recolor(0, self.rows_done)         # a Buddha channel's counts can
                                                     # affect any already-drawn pixel
        else:
            self.recolor(rows_before, self.rows_done)
        self.elapsed = time.perf_counter() - self.start_time
        self.draw()

        if self.rows_done >= SIZE:
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
        self.rebuild_view_list()                   # every view's thumbnail reflects the
                                                    # simulation that just finished, even
                                                    # if the Coloration tab isn't showing

    def settings_of_current_image(self):
        """The RaySettings the current image was generated with (the defaults, for
        the placeholder image before the first Start). Coloring a view from the
        stored data needs all of them, not just the limits: Buddha coloration
        re-traces rays with the generated confinement and emission rules."""
        if self.generated_settings is not None:
            return self.generated_settings
        return DEFAULT_RAY_SETTINGS
