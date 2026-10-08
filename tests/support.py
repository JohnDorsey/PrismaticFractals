"""Shared by the test files of this folder: the one loaded copy of the package (`m`), the root folder, the
skip exception, and every helper, fake app and fixture used by more than one area."""
import dataclasses
import os
import shutil
import sys
import tempfile
import threading
import traceback

import numpy as np

from test_harness import load_module

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the folder with light_bounces.py
m = load_module()


class SkipTest(Exception):
    """Raised by a test that can't run in this environment (e.g. needs a real external tool that isn't
    installed); reported as SKIP, not PASS."""


# --------------------------------------------------------------------------- #
# The shape kinds' parameter tables (Shape subclasses)
# --------------------------------------------------------------------------- #
def _breaks_a_self_check(fresh, expected_text):
    try:
        fresh.run_self_checks()
    except AssertionError as e:
        assert expected_text in str(e), str(e)
    else:
        raise AssertionError("expected run_self_checks() to raise")


# --------------------------------------------------------------------------- #
# Snapshot/Settings construction (keyword-argument refactor)
# --------------------------------------------------------------------------- #
def _make_fake_app_for_snapshot():
    """A minimal stand-in for App with just enough stub controls for
    read_fuzz_controls() and current_ray_settings() to run for real, unmodified,
    off the actual App class."""
    class _Get:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    class FakeApp:
        pass

    app = FakeApp()
    app.read_fuzz_controls = m.App.read_fuzz_controls.__get__(app)
    app.current_ray_settings = m.App.current_ray_settings.__get__(app)
    app.current_natural_termination_settings = m.App.current_natural_termination_settings.__get__(app)
    app.current_generation_settings = m.App.current_generation_settings.__get__(app)
    app.current_blur_settings = m.App.current_blur_settings.__get__(app)
    app.current_buddha_seed_settings = lambda: m.DEFAULT_BUDDHA_SEED

    app.rng_seed_var = _Get("0")
    app.last_rng_seed = 0
    app.fuzz_enabled_var = _Get(False)
    for field in ["emit_jitter", "bnd_offset", "bnd_jitter", "bnd_mult",
                 "self_offset", "self_jitter", "self_mult"]:
        setattr(app, f"{field}_scale", _Get(1.0 if field.endswith("mult") else 0.0))

    app.iter_var = _Get("32")
    app.dist_var = _Get("0")
    app.escape_radius_var = _Get(str(m.DEFAULT_ESCAPE_RADIUS))
    app.seg_mult_scale = _Get(0.75)
    app.buddha_blur_radius_scale = _Get(m.DEFAULT_BUDDHA_BLUR_RADIUS)
    app.buddha_blur_type_var = _Get(m.BUDDHA_BLUR_LABELS[m.DEFAULT_BUDDHA_BLUR_TYPE])
    app.confine_lookback_var = _Get(str(m.DEFAULT_CONFINE_LOOKBACK))
    app.confine_method_var = _Get(m.CONFINE_METHOD_NAMES[m.DEFAULT_CONFINE_METHOD])
    app.confine_threshold_var = _Get(str(m.DEFAULT_CONFINE_THRESHOLD))
    app.emit_mode_var = _Get(m.EMIT_MODE_NAMES[m.DEFAULT_EMIT_MODE])
    for field in m.EMIT_TARGET_FIELDS:
        setattr(app, f"emit_target_{field}_var", _Get("0.0"))
    app.emit_offset_scale = _Get(m.DEFAULT_EMIT_OFFSET)
    _give_framings(app)
    return app


# --------------------------------------------------------------------------- #
# Colorations (multiple side-by-side colorations of the same simulation)
# --------------------------------------------------------------------------- #
def _framing(**changes):
    values = dict(id=0, x=0.0, y=0.0, diagonal=m.DEFAULT_FRAMING_DIAGONAL, aspect=1.0)
    values.update(changes)
    return m.FramingSnapshot(**values)


def _give_image_geometry(app, framing=None):
    """Give a fake app the state of the image on screen that the real App has (its framing, size in pixels and world
    rectangle, buffers and grid: see App.set_image_framing) and the conversions between pixels and the world."""
    app.set_image_framing = m.App.set_image_framing.__get__(app)
    app.make_image_buffers = m.App.make_image_buffers.__get__(app)
    app.pixel_center = m.App.pixel_center.__get__(app)
    app.world_to_canvas = m.App.world_to_canvas.__get__(app)
    app.set_image_framing(m.DEFAULT_FRAMING if framing is None else framing)
    return app


def _give_framings(app):
    """Give a fake app the list of framings the real App holds (the whole-world framing, selected) and the methods
    that find the selected one."""
    app.framings = [m.FramingConfig(0)]
    app._framing_counter = 1
    app.selected_framing_id = 0
    app.framing_by_id = m.App.framing_by_id.__get__(app)
    app.selected_framing_obj = m.App.selected_framing_obj.__get__(app)
    app.selected_framing_snapshot = m.App.selected_framing_snapshot.__get__(app)
    app.all_framing_ids = m.App.all_framing_ids.__get__(app)
    app.framings_loaded = False
    app.on_framings_loaded = lambda: None
    return app


def _fake_render_app():
    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    _give_framings(app)
    app.render_full_image = m.App.render_full_image.__get__(app)
    app.apply_coloration = m.App.apply_coloration.__get__(app)
    return app


def _starting_obst_arrays():
    """The scene the app starts with: one obstacle, the unit circle."""
    return m.build_obstacle_arrays([m.default_starting_obstacle(0)])


def _coloration_settings_from_config(v):
    return m.ColorationSettings(mode=v.mode, metric_types=v.metric_types, point_metric=v.point_metric,
                          other_metric=v.other_metric, point1_choice=v.point1_choice,
                          point2_choice=v.point2_choice, chan_src=v.chan_src,
                          buddha_points=v.buddha_points, buddha_compression=v.buddha_compression,
                          buddha_brightness=v.buddha_brightness,
                          constant_choice=v.constant_choice)


def _fake_coloration_list_app():
    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    app.colorations = [m.ColorationConfig(0)]
    app._coloration_counter = 1
    app.selected_coloration_id = 0
    app.coloration_by_id = m.App.coloration_by_id.__get__(app)
    app.live_coloration = m.App.live_coloration.__get__(app)
    app.selected_coloration_obj = m.App.selected_coloration_obj.__get__(app)
    app.all_coloration_ids = m.App.all_coloration_ids.__get__(app)
    app.new_coloration = m.App.new_coloration.__get__(app)
    app.duplicate_coloration = m.App.duplicate_coloration.__get__(app)
    app.delete_coloration = m.App.delete_coloration.__get__(app)
    app.select_coloration = m.App.select_coloration.__get__(app)
    app.refresh_live_preview_state = m.App.refresh_live_preview_state.__get__(app)
    app.rebuild_coloration_list = lambda: None
    app.refresh_color_controls_for_selected_coloration = lambda: None
    # Stubs for the live-rendering cascade every selection change now
    # triggers (self.mapping is set directly, not via apply_mapping_change,
    # so no stub is needed for that part).
    app.rebuild_buddha_counts = lambda: None
    app.update_stat_tags = lambda: None
    app.recolor = lambda r0, r1: None
    app.draw = lambda: None
    app.update_preview = lambda i, j: None
    app.rows_done = 0
    app.pinned_px = None
    app.coloration_thumb_cache = {}
    app.keyframes = []
    app.refresh_timeline_diffs = lambda: None
    app.purge_keyframe_settings = m.App.purge_keyframe_settings.__get__(app)
    return app


# --------------------------------------------------------------------------- #
# Export includes every coloration, side by side (the actual save/render call
# chains, not just render_full_image in isolation - these exercise the real
# path from "how many colorations exist right now" through to the output file)
# --------------------------------------------------------------------------- #
class _StubJxlControls:
    """Stands in for JxlQualityControls (the video panel's JPEG XL choices)."""
    def __init__(self, lossless=True, quality=90):
        self._lossless, self._quality = lossless, quality

    def lossless(self):
        return self._lossless

    def quality(self):
        return self._quality


class _StubVar:
    def __init__(self, v=None):
        self._v = v
    def get(self):
        return self._v
    def set(self, v):
        self._v = v


class _StubWidget:
    def __getattr__(self, _name):
        return lambda *a, **k: None
    def state(self, *a, **k):
        pass


class _TrackedFrame:
    """Like _StubWidget, but actually tracks grid()/grid_remove() calls -
    for tests that need to check which of several frames is currently
    shown."""
    def __init__(self):
        self.shown = True
    def grid(self, *a, **k):
        self.shown = True
    def grid_remove(self, *a, **k):
        self.shown = False
    def __getattr__(self, _name):
        return lambda *a, **k: None


def _fake_export_app():
    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    app.colorations = [m.ColorationConfig(0)]
    app._coloration_counter = 1
    app.selected_coloration_id = 0
    app.include_all_colorations_var = _StubVar(True)
    app.color_info_location_var = _StubVar("None")
    app.figure_info_location_var = _StubVar("None")
    app.framing_info_location_var = _StubVar("None")
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    app.generated_natural_termination = m.DEFAULT_NATURAL_TERMINATION_SETTINGS
    app.coloration_by_id = m.App.coloration_by_id.__get__(app)
    app.live_coloration = m.App.live_coloration.__get__(app)
    app.selected_coloration_obj = m.App.selected_coloration_obj.__get__(app)
    app.all_coloration_ids = m.App.all_coloration_ids.__get__(app)
    app.colorations_to_export = m.App.colorations_to_export.__get__(app)
    _give_framings(app)
    app.new_coloration = m.App.new_coloration.__get__(app)
    app.refresh_live_preview_state = m.App.refresh_live_preview_state.__get__(app)
    app.rebuild_buddha_counts = lambda: None
    app.update_stat_tags = lambda: None
    app.recolor = lambda r0, r1: None
    app.draw = lambda: None
    app.update_preview = lambda i, j: None
    app.rows_done = 0
    app.pinned_px = None
    app.coloration_thumb_cache = {}
    app.rebuild_coloration_list = lambda: None
    app.refresh_color_controls_for_selected_coloration = lambda: None
    app.obstacles = []
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    app.current_ray_settings = lambda: m.DEFAULT_RAY_SETTINGS
    app.current_generation_settings = lambda: (m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_TERMINATION_SETTINGS)
    app.current_blur_settings = lambda: m.DEFAULT_BUDDHA_BLUR
    app.current_buddha_seed_settings = lambda: m.DEFAULT_BUDDHA_SEED
    app.deco = {"outline": _StubVar(False)}
    app.render_full_image = m.App.render_full_image.__get__(app)
    app.apply_coloration = m.App.apply_coloration.__get__(app)
    return app


def _run_save_new_image_synchronously(app):
    """save_new_image() spawns a background thread; run its target inline so
    the test can inspect the result without dealing with real threading."""
    class SyncThread:
        def __init__(self, target, args, daemon=True):
            self.target, self.args = target, args
        def start(self):
            self.target(*self.args)
    orig_thread = threading.Thread
    threading.Thread = SyncThread
    try:
        app.save_new_image(None)
    finally:
        threading.Thread = orig_thread


# --------------------------------------------------------------------------- #
# "Current image" export (_write/render_current_image): reuses the
# already-traced simulation rather than re-simulating, and respects
# "Include all colorations" the same way the other two export paths do
# --------------------------------------------------------------------------- #
def _fake_current_image_app():
    app = _fake_export_app()
    app.stats = np.random.RandomState(0).rand(m.SIZE, m.SIZE, m.N_STATS).astype(np.float32)
    app.characteristic_points = (np.random.RandomState(1).rand(m.SIZE, m.SIZE, m.N_POINTS, 2)
                 .astype(np.float32) * 2 - 1)
    app.pixel_state = np.full((m.SIZE, m.SIZE), m.PIXEL_TRACED, dtype=np.uint8)
    app.current_blur_settings = lambda: m.DEFAULT_BUDDHA_BLUR
    app.current_buddha_seed_settings = lambda: m.DEFAULT_BUDDHA_SEED
    app.settings_of_current_image = m.App.settings_of_current_image.__get__(app)
    app.natural_termination_of_current_image = m.App.natural_termination_of_current_image.__get__(app)
    app.gen_termination = m.NATURAL_TERMINATION_DEFAULT
    empty_i = np.empty(0, dtype=np.int64)
    empty_b = np.empty((0, m.OBST_BOUND_LEN))
    empty_f = np.empty(0)
    empty_bool = np.empty(0, dtype=bool)
    app.gen_engine_args = _engine_args(m.ObstacleArrays(
        shapes=empty_i, bounds=empty_b, cx=empty_f, cy=empty_f,
        effects=np.empty((0, 2), dtype=np.int64), fire_inside=empty_bool))
    app.gx, app.gy = m.make_grid(m.SIZE, m.SIZE)
    app.obstacle_ring_specs = lambda: []
    app.gen_obstacle_snapshots = [m.default_starting_obstacle(0).snapshot()]
    app.frame = np.full((m.SIZE, m.SIZE, 3), 42, dtype=np.uint8)   # the "already
                                                                   # rendered" preview
    app.render_current_image = m.App.render_current_image.__get__(app)
    app.render_coloration_thumbnail = m.App.render_coloration_thumbnail.__get__(app)
    app._write = m.App._write.__get__(app)
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    return app


# --------------------------------------------------------------------------- #
# figure_info_lines / the "Figure settings location" margin: every setting
# that defines the figure (limits, segment length, fuzzing, every obstacle)
# --------------------------------------------------------------------------- #
def _settings(**overrides):
    """A RaySettings: the defaults, then `overrides` (any fields, by name)."""
    return m.DEFAULT_RAY_SETTINGS.replace(**overrides)


def _termination_settings(**overrides):
    """A NaturalTerminationSettings: the defaults, with 64 iterations and a distance limit of 0,
    then `overrides` (any fields, by name)."""
    return m.DEFAULT_NATURAL_TERMINATION_SETTINGS.replace(**{"max_iter": 64, "max_dist": 0.0, **overrides})


def _generation(**overrides):
    """(RaySettings, NaturalTerminationSettings) - the two records that, with the obstacles, define a figure - from
    `overrides` naming fields of either, by name (see _settings and _stop_settings)."""
    termination_fields = {f.name for f in dataclasses.fields(m.NaturalTerminationSettings)}
    return (_settings(**{k: v for k, v in overrides.items() if k not in termination_fields}),
            _termination_settings(**{k: v for k, v in overrides.items() if k in termination_fields}))


def _termination(max_iter, max_dist=0.0, escape_radius=None, confine=None):
    """The kernels' NaturalTerminationRule: `max_dist` 0 = no limit, the other defaults the program's."""
    return m.natural_termination_rule(
        max_iter, m.effective_dist(max_dist),
        m.DEFAULT_ESCAPE_RADIUS if escape_radius is None else escape_radius,
        m.CONFINE_DEFAULT if confine is None else confine)


def _engine_args(obstacles, seg_mult=1.0, fuzz=None, emission=None):
    """The kernels' RayArgs for an ObstacleArrays; `fuzz` is a FuzzParams (default: fuzzing off)."""
    return m.ray_args(seg_mult, m.fuzz_rule(m.DEFAULT_FUZZ if fuzz is None else fuzz),
                      m.EMISSION_DEFAULT if emission is None else emission, obstacles)


def _start_scene():
    return [m.default_starting_obstacle(0)]


# --------------------------------------------------------------------------- #
# The main bounding shape's own ray behavior is an ordinary keyframeable setting
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Ray effects on crossing an edge (incoming / outgoing: Reflect, Terminate or
# Pass through) and rays that escape (nothing left to hit)
# --------------------------------------------------------------------------- #
R, T, P = m.OBSTACLE_EFFECT_REFLECT, m.OBSTACLE_EFFECT_TERMINATE, m.OBSTACLE_EFFECT_PASS
UNIT_CIRCLE = m.ellipse_bound(2.0, 2.0, 0.0)


@m.njit
def _seed_kernel_rng(seed):
    """Seeds the RNG the compiled kernels use - which is NOT the one
    np.random.seed() seeds from ordinary Python code."""
    np.random.seed(seed)


def _circle_obstacle(cx, cy, radius, in_effect, out_effect, fire=False, id_=0):
    o = m.EllipseShape(id_)
    o.width = o.height = 2.0 * radius
    o.r, o.theta, o.cx, o.cy = 0.0, 0.0, cx, cy
    o.in_effect, o.out_effect, o.fire_inside = in_effect, out_effect, fire
    return o


# --------------------------------------------------------------------------- #
# The new ray behavior end to end: the real render paths, not just trace_path
# --------------------------------------------------------------------------- #
def _terminated_coloration(**_):
    """A coloration whose channel 0 is the termination-reason metric and whose other
    channels are constant 0: a pixel's red value then says why its ray stopped
    (255 = escaped)."""
    v = m.ColorationConfig(0)
    v.metric_types[0] = m.METRIC_TYPE_OTHER
    v.other_metric[0] = m.ST_TERMINATION
    v.chan_src[1] = v.chan_src[2] = m.CHAN_SRC_CONSTANT
    v.constant_choice[1] = v.constant_choice[2] = m.NONE_ZERO
    return _coloration_settings_from_config(v)


# --------------------------------------------------------------------------- #
# Ray emission direction: Up / Towards point, a target point, an angle offset
# --------------------------------------------------------------------------- #
UP, TOWARDS = m.EMIT_UP, m.EMIT_TOWARDS


def _emission(mode=UP, x=0.0, y=0.0, offset=-45.0):
    return m.emission_rule(mode, x, y, offset)


def _first_direction(x0, y0, emission, fuzz=None, seed=0):
    """The unit direction of the first segment of a ray fired from (x0, y0),
    read off the real kernel (an obstacle-free, escaping scene: the first
    recorded point is then straight down the ray, on the image edge)."""
    arrays = m.build_obstacle_arrays([_circle_obstacle(x0, y0, 0.001, P, P, fire=True)])
    path_points = np.empty((6, 2))
    path_point_tags = np.empty(6)
    raw = np.empty(m.N_RAW)
    fz = m.fuzz_rule(m.DEFAULT_FUZZ if fuzz is None else fuzz)
    _seed_kernel_rng(seed)
    term, npts = m.trace_path(x0, y0, path_points, path_point_tags, _termination(4),
                              m.ray_args(1.0, fz, emission, arrays), raw)
    assert term == m.TERM_ESCAPED and npts == 2
    delta = path_points[1] - path_points[0]
    return delta / np.hypot(*delta)


def _angle_deg(direction):
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


# --------------------------------------------------------------------------- #
# Editing a non-live selected coloration must never touch the live coloration (this is
# the actual mechanism behind the reported bug: thumbnails silently going
# stale, or two different-looking colorations becoming identical, because edits
# were always landing on self.live_coloration() regardless of what was selected)
# --------------------------------------------------------------------------- #
def _fake_color_editor_app():
    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    _give_framings(app)
    app.colorations = [m.ColorationConfig(0), m.ColorationConfig(1)]
    app.colorations[1].mode = "HSV"
    app._coloration_counter = 2
    app.selected_coloration_id = 0
    app.coloration_by_id = m.App.coloration_by_id.__get__(app)
    app.live_coloration = m.App.live_coloration.__get__(app)
    app.selected_coloration_obj = m.App.selected_coloration_obj.__get__(app)
    app.all_coloration_ids = m.App.all_coloration_ids.__get__(app)
    app.select_coloration = m.App.select_coloration.__get__(app)
    app.refresh_live_preview_state = m.App.refresh_live_preview_state.__get__(app)
    app._color_editor_loading = False

    app.mode_var = _StubVar("RGB")
    app.mode_combo = _StubWidget()
    app.comp_labels = [_StubWidget() for _ in range(3)]
    app.map_help_var = _StubVar("")
    app.chan_src_vars = [_StubVar(m.CHAN_SRC_QUERY) for _ in range(3)]
    app.chan_src_combos = [_StubWidget() for _ in range(3)]
    app.metric_type_vars = [_StubVar(m.METRIC_TYPE_POINT_PAIR) for _ in range(3)]
    app.metric_type_combos = [_StubWidget() for _ in range(3)]
    app.map_vars = [_StubVar("") for _ in range(3)]
    app.map_combos = [_StubWidget() for _ in range(3)]
    app.point1_vars = [_StubVar(m.POINT_LABELS[m.POINT_START]) for _ in range(3)]
    app.point1_combos = [_StubWidget() for _ in range(3)]
    app.point2_vars = [_StubVar(m.POINT_LABELS[m.POINT_END]) for _ in range(3)]
    app.point2_combos = [_StubWidget() for _ in range(3)]
    app.query_frames = [_TrackedFrame() for _ in range(3)]
    app.buddha_frames = [_TrackedFrame() for _ in range(3)]
    app.buddha_point_vars = [[_StubVar(False) for _ in range(m.N_BUDDHA_POINTS)]
                             for _ in range(3)]
    app.buddha_compression_vars = [_StubVar(m.BUDDHA_COMPRESSION_LABELS[0]) for _ in range(3)]
    app.buddha_compression_combos = [_StubWidget() for _ in range(3)]
    app.buddha_brightness_vars = [_StubVar(m.BUDDHA_BRIGHTNESS_OPTIONS[2]) for _ in range(3)]
    app.buddha_brightness_combos = [_StubWidget() for _ in range(3)]
    app.constant_frames = [_TrackedFrame() for _ in range(3)]
    app.constant_vars = [_StubVar(m.CHAN_CONSTANT_LABELS[0]) for _ in range(3)]
    app.constant_combos = [_StubWidget() for _ in range(3)]

    app.mapping = m.make_mapping(m.DEFAULT_METRIC_TYPES, m.DEFAULT_POINT_METRICS,
                                 m.DEFAULT_OTHER_METRICS, m.DEFAULT_CHAN_SRCS,
                                 m.DEFAULT_CONSTANT_CHOICES)
    app.rows_done = 0
    app.pinned_px = None
    app.coloration_thumb_cache = {}
    app.recolor_calls = []
    app.recolor = lambda r0, r1: app.recolor_calls.append((r0, r1))
    app.draw = lambda: None
    app.update_preview = lambda i, j: None
    app.rebuild_buddha_counts_calls = []
    app.rebuild_buddha_counts = lambda: app.rebuild_buddha_counts_calls.append(1)
    app.stat_tag_calls = []
    app.update_stat_tags = lambda: app.stat_tag_calls.append(1)
    app.rebuild_calls = []
    app.rebuild_coloration_list = lambda: app.rebuild_calls.append(1)
    app.refresh_timeline_diffs = lambda: None
    app.refresh_color_controls_for_selected_coloration = (
        m.App.refresh_color_controls_for_selected_coloration.__get__(app))
    app.refresh_row = m.App.refresh_row.__get__(app)
    app.on_mode_changed = m.App.on_mode_changed.__get__(app)
    app.apply_mapping_change = m.App.apply_mapping_change.__get__(app)
    return app


# --------------------------------------------------------------------------- #
# Debug dumps (dev/debug_dump.py), the Debug menu, and their triggers
# --------------------------------------------------------------------------- #
def _load_dumper():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "debug_dump_under_test", os.path.join(ROOT,
                                              "dev", "debug_dump.py"))
    dumper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dumper)
    return dumper


class _Unpicklable:
    """Like real Tk objects, which hold a live interpreter and never pickle."""
    def __reduce__(self):
        raise TypeError(f"cannot pickle {type(self).__name__!r} object")


class _FakeVar(_Unpicklable):
    _name = "PY_VAR7"

    def get(self):
        return "256"

    def set(self, v):
        pass

    def trace_add(self, *a):
        pass


class _FakeWidget(_Unpicklable):
    def winfo_class(self):
        return "TEntry"

    def winfo_children(self):
        return []

    def configure(self):
        return {"width": ("width", "w", "W", 20, 8), "text": ("text", "t", "T", "", "hi")}

    def __str__(self):
        return ".frame.entry"


class _ModuleRegistered:
    """Pickle finds a class by looking its module up in sys.modules, but every load_module() copy
    keeps its modules out of sys.modules (so the next copy starts clean); put the modules of the
    copy these tests use there for the duration."""
    def __enter__(self):
        self.saved = {name: sys.modules.get(name) for name in m.__lb_modules__}
        sys.modules.update(m.__lb_modules__)

    def __exit__(self, *exc):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


# --------------------------------------------------------------------------- #
# Strict records (FuzzParams, RaySettings, BuddhaBlurSettings, KeyframeData,
# ColorationSettings, ObstacleSnapshot, ColorationSnapshot): keyword-only, frozen, no
# indexing, checked types
# --------------------------------------------------------------------------- #
def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception as other:                                   # noqa: BLE001
        raise AssertionError(f"expected {exc}, got {type(other).__name__}: {other}")
    raise AssertionError(f"expected {exc}, nothing was raised")




# --------------------------------------------------------------------------- #
# The squircles and the Archimedean spiral (used by the shapes and the kernels tests)
# --------------------------------------------------------------------------- #
NEW_CLASSES = [m.SuperellipseShape, m.FernandezGuastiShape, m.ArchimedeanSpiralShape,
               m.PythagorasSnailShape, m.HugelschafferEggShape, m.SineFlowerShape]


def _new_kind_cases():
    """Several settings of each new kind, rotated and not."""
    cases = []
    for width, height, exponent, rotation in ((1.6, 1.0, 4.0, 0.0), (1.2, 1.2, 2.0, 37.0),
                                              (1.8, 0.7, 1.3, 120.0), (1.5, 1.1, 12.0, -20.0)):
        o = m.SuperellipseShape(0)
        o.width, o.height, o.exponent, o.rotation = width, height, exponent, rotation
        cases.append(o)
    for radius, squareness, rotation in ((1.0, 0.0, 0.0), (0.9, 0.5, 30.0), (1.1, 0.95, 70.0),
                                         (0.8, 1.0, 15.0)):
        o = m.FernandezGuastiShape(0)
        o.radius, o.squareness, o.rotation = radius, squareness, rotation
        cases.append(o)
    for radius, pitch, fill, rotation in ((1.0, 0.25, 0.5, 0.0), (1.3, 0.1, 0.7, 40.0),
                                          (0.9, 0.5, 0.2, -75.0), (1.0, 0.3, 0.94, 10.0)):
        o = m.ArchimedeanSpiralShape(0)
        o.radius, o.spiral_pitch, o.spiral_fill, o.rotation = radius, pitch, fill, rotation
        cases.append(o)
    for segments, rotation in ((17, 0.0), (5, 35.0), (12, -110.0), (30, 20.0), (1, 0.0)):
        o = m.PythagorasSnailShape(0)
        o.snail_segments, o.rotation = segments, rotation
        cases.append(o)
    for a, b, shift, rotation in ((1.0, 0.7, 0.4, 0.0), (1.2, 0.5, -0.8, 50.0), (0.9, 1.1, 0.0, 15.0),
                                  (1.0, 0.3, 0.9, -70.0), (0.6, 0.6, 0.6, 200.0)):
        o = m.HugelschafferEggShape(0)
        o.egg_a, o.egg_b, o.egg_shift, o.rotation = a, b, shift, rotation
        cases.append(o)
    for radius, petals, amplitude, rotation in ((1.0, 6, 0.3, 0.0), (0.8, 1, 0.5, 40.0), (1.1, 3, 0.9, -25.0),
                                                (0.9, 12, 0.6, 10.0), (1.0, 25, 0.2, 77.0), (0.7, 5, 0.0, 0.0)):
        o = m.SineFlowerShape(0)
        o.radius, o.flower_petals, o.flower_amplitude, o.rotation = radius, petals, amplitude, rotation
        cases.append(o)
    return cases


def _independent_membership(o):
    """The shape as a formula written straight from its definition (not from the packed bound):
    point -> inside?"""
    t = np.radians(o.rotation)
    cs, sn = np.cos(t), np.sin(t)
    if isinstance(o, m.SuperellipseShape):
        def member(x, y):
            u, v = x * cs + y * sn, -x * sn + y * cs
            return abs(u / (o.width / 2)) ** o.exponent + abs(v / (o.height / 2)) ** o.exponent < 1.0
        return member
    if isinstance(o, m.FernandezGuastiShape):
        r, k = o.radius, o.squareness ** 2 / o.radius ** 2

        def member(x, y):
            u, v = x * cs + y * sn, -x * sn + y * cs
            return abs(u) < r and abs(v) < r and u * u + v * v - k * u * u * v * v < r * r
        return member
    if isinstance(o, m.SineFlowerShape):
        def member(x, y):
            u, v = x * cs + y * sn, -x * sn + y * cs
            return np.hypot(u, v) < o.radius * (1.0 + o.flower_amplitude * np.sin(o.flower_petals * np.arctan2(v, u)))
        return member
    if isinstance(o, m.HugelschafferEggShape):
        # the equation as the page gives it, in its own x (the middle of the egg's length is at x = d)
        a, b, d = o.egg_a, o.egg_b, o.egg_shift * o.egg_a

        def member(x, y):
            u, v = x * cs + y * sn, -x * sn + y * cs
            px = u + d
            return (abs(u) < a and (px - d) ** 2 / a ** 2
                    + v * v / b ** 2 * (1.0 + (2.0 * d * px - d * d) / a ** 2) < 1.0)
        return member
    if isinstance(o, m.PythagorasSnailShape):
        # the construction itself: A_1 = (1, 0), and A_{k+1} is A_k plus a unit step turned a quarter turn left
        # of the direction O -> A_k; triangle k is O, A_k, A_{k+1}
        corners = [np.array([1.0, 0.0])]
        for _k in range(o.snail_segments):
            a = corners[-1]
            corners.append(a + np.array([-a[1], a[0]]) / np.hypot(*a))
        scale = 1.0 / np.hypot(*corners[-1])

        def member(x, y):
            u, v = (x * cs + y * sn) / scale, (-x * sn + y * cs) / scale
            for a, b in zip(corners[:-1], corners[1:]):
                d1 = a[0] * v - a[1] * u
                d2 = u * b[1] - v * b[0]
                d3 = (b[0] - a[0]) * (v - a[1]) - (b[1] - a[1]) * (u - a[0])
                if d1 > 0 and d2 > 0 and d3 > 0:
                    return True
            return False
        return member
    pitch, fill = o.spiral_pitch, o.spiral_fill
    b, w = pitch / (2 * np.pi), fill * pitch
    theta_max = (o.radius - w) / b

    def member(x, y):
        u, v = x * cs + y * sn, -x * sn + y * cs
        rho, phi = np.hypot(u, v), np.arctan2(v, u)
        for k in range(-1, 400):                       # every turn the point could lie on
            theta = phi + 2 * np.pi * k
            if 0.0 <= theta <= theta_max and b * theta <= rho < b * theta + w:
                return True
        return False
    return member


def _buddha_choice(*points):
    """A channel's buddha_points tuple with exactly the entries `points` (indexes into BUDDHA_POINT_CATALOG - a stored
    point's own number, such as POINT_SELF_MEAN, is its entry) switched on."""
    return tuple(i in points for i in range(m.N_BUDDHA_POINTS))


__all__ = [n for n in dir() if not n.startswith("__")]

