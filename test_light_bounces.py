"""
Regression test suite for light_bounces.py.

Run with:  python3 test_light_bounces.py

Every test is a plain function taking no arguments, named test_*, that
raises AssertionError on failure. run_all() below discovers and runs them
all, reporting a pass/fail summary, so a future change can be checked
against the *whole* accumulated history of things that have gone wrong in
this project so far - not just whatever the current change happens to touch.

This is not meant to be exhaustive. It's meant to grow: whenever a real bug
is found (by hand, by the person using the program, or by Claude while
making a change), the fix belongs here too, as a test that would have caught
it. Several of the tests below exist for exactly that reason and say so in
their docstring.
"""
import os
import shutil
import sys
import tempfile
import threading
import traceback

import numpy as np

from test_harness import load_module

m = load_module()


# --------------------------------------------------------------------------- #
# Registry self-checks
# --------------------------------------------------------------------------- #
def test_self_checks_pass_on_the_real_file():
    """run_self_checks() (in light_bounces.py itself) already runs at import
    time, so if we got this far it already passed - this just re-runs it
    explicitly so a failure here points straight at the registry check
    rather than an opaque import error."""
    m.run_self_checks()


def test_self_checks_catch_a_missing_termination_color():
    """Regression test: TERM_PERPENDICULAR was once missing from the
    ray-end-color dict, causing a KeyError the first time a ray terminated
    that way. Confirms the check that would have caught it actually fires."""
    fresh = load_module()
    del fresh.TERM_END_COLOR[fresh.TERM_PERPENDICULAR]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "TERM_END_COLOR" in str(e)


def test_self_checks_catch_an_unknown_float_key():
    fresh = load_module()
    fresh.SETTING_FLOAT_KEYS = fresh.SETTING_FLOAT_KEYS | {"not_a_real_setting"}
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "SETTING_FLOAT_KEYS" in str(e)


def test_self_checks_catch_a_setting_info_mismatch():
    fresh = load_module()
    del fresh.DEFAULT_SETTING_VALUES["dist"]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "SETTING_INFO" in str(e)


def test_self_checks_catch_an_obstacle_field_mismatch():
    fresh = load_module()
    fresh.OBST_FIELD_INFO = [f for f in fresh.OBST_FIELD_INFO if f[0] != "moss_size"]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "OBST_FIELD_INFO" in str(e)


# --------------------------------------------------------------------------- #
# Distance limit: non-integer support (regression coverage for two bugs)
# --------------------------------------------------------------------------- #
def test_parse_dist_limit_accepts_non_integers():
    assert m.parse_dist_limit("5.5") == 5.5
    assert m.parse_dist_limit("0.4") == 0.4
    assert m.parse_dist_limit("7") == 7 and isinstance(m.parse_dist_limit("7"), int)
    assert m.parse_dist_limit("-1") is None
    assert m.parse_dist_limit("abc") is None
    assert m.parse_dist_limit("") is None
    assert m.parse_dist_limit(f"{m.DIST_LIMIT_MAX + 1}") is None


def test_dist_is_not_rounded_during_keyframe_resolution():
    """Regression test: resolve_frame_settings() used to force dist through
    int(round(...)), so a keyframe with dist=0.4 resolved to max_dist=0 -
    which means "no limit" in this program's convention. A short-distance
    keyframe would silently become an unlimited-distance one."""
    kf = m.Keyframe(1, 0, {**m.DEFAULT_SETTING_VALUES, "dist": 0.4}, "Linear", None)
    data = m.resolve_frame_settings([kf], 0)
    assert data.snapshot.max_dist == 0.4
    assert m.effective_dist(data.snapshot.max_dist) == 0.4


def test_dist_animates_smoothly_between_keyframes():
    """Regression test: "dist" wasn't in SETTING_FLOAT_KEYS, so it was held
    (jumping only at the second keyframe's exact frame) instead of
    interpolated - the actual bug report was "Buddha colors don't appear
    until the very last frame" of a video animating the distance limit."""
    kf0 = m.Keyframe(1, 0, {**m.DEFAULT_SETTING_VALUES, "dist": 0.01}, "Linear", None)
    kf1 = m.Keyframe(2, 59, {**m.DEFAULT_SETTING_VALUES, "dist": 16}, "Linear", None)
    dists = [m.resolve_frame_settings([kf0, kf1], fn).snapshot.max_dist for fn in range(60)]
    assert dists[0] == 0.01 and dists[59] == 16
    assert all(dists[i] < dists[i + 1] for i in range(59)), \
        "dist should increase strictly monotonically across every frame"


# --------------------------------------------------------------------------- #
# Segment length multiplier: core ray-tracing math
# --------------------------------------------------------------------------- #
def _trace(seg_mult, x0=-0.3, y0=-0.2, max_iter=8, width=2.0, height=1.4):
    bound = m.ellipse_bound(width, height, 0.0)
    fz = m.fuzz_array(m.DEFAULT_FUZZ)
    empty_i = np.empty(0, dtype=np.int64)
    empty_b = np.empty((0, m.OBST_BOUND_LEN))
    empty_f = np.empty(0)
    pts = np.empty((max_iter + 2, 2))
    kinds = np.empty(max_iter + 2)
    raw = np.empty(m.N_RAW)
    term, npts = m.trace_path(x0, y0, pts, kinds, max_iter, m.effective_dist(0.0),
                              m.SHAPE_ELLIPSE, bound, m.MAIN_CTL_DEFAULT, seg_mult, fz,
                              empty_i, empty_b,
                              empty_f, empty_f, np.empty((0, 2), dtype=np.int64), raw)
    return pts[:npts], kinds[:npts], term, raw, bound


def test_segment_multiplier_shortens_to_the_right_point():
    """The recorded bounce point must land at exactly seg_mult of the way
    from the segment's start to its true (unshortened) hit point."""
    pts_full, _, _, _, _ = _trace(1.0)
    pts_half, _, _, _, _ = _trace(0.5)
    assert len(pts_full) >= 3 and len(pts_half) >= 3
    start = pts_full[0]
    true_hit = pts_full[1]
    expected = start + 0.5 * (true_hit - start)
    assert np.allclose(pts_half[1], expected, atol=1e-9)


def test_segment_multiplier_keeps_the_true_bounce_direction():
    """The ray must continue in the direction calculated at the TRUE
    (full-length) hit point, not one recalculated from the shortened point -
    this is the "changes course in mid-air" behavior that was requested."""
    pts_full, _, _, _, _ = _trace(1.0)
    pts_half, _, _, _, _ = _trace(0.5)
    dir_full = pts_full[2] - pts_full[1]
    dir_full /= np.linalg.norm(dir_full)
    dir_half = pts_half[2] - pts_half[1]
    dir_half /= np.linalg.norm(dir_half)
    assert np.allclose(dir_full, dir_half, atol=1e-9)


def test_segment_multiplier_of_one_matches_unshortened_behavior():
    """1.0 must be a true no-op - anyone's existing (pre-feature) images and
    videos should render identically."""
    pts_a, kinds_a, term_a, _, _ = _trace(1.0)
    pts_b, kinds_b, term_b, _, _ = _trace(1.0)
    assert np.array_equal(pts_a, pts_b) and term_a == term_b


def test_self_loop_area_zeroed_when_multiplier_active():
    """The self-bounce polygon area metric must read exactly 0 whenever the
    segment length multiplier isn't exactly 1 - the polygon stops meaning
    "the ray's own enclosed area" once bounce points are pulled back."""
    pts, kinds, term, raw, bound = _trace(1.0, max_iter=10)
    out = np.empty(m.N_STATS)
    m.compute_stats(pts, len(pts), raw, 10, 1.0, out)
    assert int(raw[m.RAW_SELF_K]) >= 0, "test setup needs an actual self-bounce"
    assert raw[m.RAW_AREA] > 0, "seg_mult=1.0 should compute a real nonzero area"

    for bad_mult in (0.99, 0.5, m.SEGMENT_MULT_MIN):
        pts2, kinds2, term2, raw2, bound2 = _trace(bad_mult, max_iter=10)
        out2 = np.empty(m.N_STATS)
        m.compute_stats(pts2, len(pts2), raw2, 10, bad_mult, out2)
        assert raw2[m.RAW_AREA] == 0.0
        assert out2[m.ST_LOOP_AREA] == 0.0


# --------------------------------------------------------------------------- #
# Moss's egg size
# --------------------------------------------------------------------------- #
def test_moss_egg_area_scales_quadratically_with_size():
    area_1x = m.moss_egg_bound(90.0, 1.0, 0.0)[0]
    area_2x = m.moss_egg_bound(90.0, 2.0, 0.0)[0]
    area_half = m.moss_egg_bound(90.0, 0.5, 0.0)[0]
    assert np.isclose(area_2x, area_1x * 4, rtol=1e-3)
    assert np.isclose(area_half, area_1x / 4, rtol=1e-3)


def test_moss_egg_params_round_trip_through_describe_and_suffix():
    """Regression test: describe_shape() and settings_suffix() both still
    unpacked Moss's egg params as a 2-tuple (angle, rotation) right after
    moss_size was added, which would have raised ValueError the moment
    either function ran."""
    m.describe_shape(m.SHAPE_MOSS_EGG, (90.0, 1.0, 0.0))
    settings = m.Settings(20, 0, 1.0, m.SHAPE_MOSS_EGG, (90.0, 1.0, 0.0), m.DEFAULT_FUZZ,
                          m.DEFAULT_BUDDHA_BLUR_RADIUS, m.DEFAULT_BUDDHA_BLUR_TYPE)
    m.settings_suffix(settings)


def test_obstacle_and_snapshot_agree_on_moss_egg_param_order():
    o = m.Obstacle(1)
    o.shape = m.SHAPE_MOSS_EGG
    o.moss_angle, o.moss_size, o.moss_rotation = 100.0, 1.3, 15.0
    via_dispatch = m.make_bound(m.SHAPE_MOSS_EGG, o.params())
    direct = m.moss_egg_bound(100.0, 1.3, 15.0)
    assert np.allclose(via_dispatch, direct)


# --------------------------------------------------------------------------- #
# Image encoders
# --------------------------------------------------------------------------- #
def _qoi_reference_decode(data):
    """Independent QOI decoder (not sharing any code with the encoder under
    test), used only to verify round-trips here."""
    import struct
    assert data[:4] == b"qoif"
    width, height, _channels, _cs = struct.unpack(">IIBB", data[4:14])
    pixels = np.zeros((height * width, 4), dtype=np.uint8)
    pixels[:, 3] = 255
    index = np.zeros((64, 4), dtype=np.uint8)
    pos = 14
    px = np.array([0, 0, 0, 255], dtype=np.int32)
    i, n = 0, height * width
    while i < n:
        tag = data[pos]
        if tag == 0xFE:
            px[0], px[1], px[2] = data[pos + 1], data[pos + 2], data[pos + 3]
            pos += 4
            pixels[i] = px
            idx = (px[0] * 3 + px[1] * 5 + px[2] * 7 + px[3] * 11) % 64
            index[idx] = px
            i += 1
        elif tag == 0xFF:
            px[0], px[1], px[2], px[3] = (data[pos + 1], data[pos + 2],
                                          data[pos + 3], data[pos + 4])
            pos += 5
            pixels[i] = px
            idx = (px[0] * 3 + px[1] * 5 + px[2] * 7 + px[3] * 11) % 64
            index[idx] = px
            i += 1
        elif (tag & 0xC0) == 0x00:
            px[:] = index[tag & 0x3F]
            pos += 1
            pixels[i] = px
            i += 1
        elif (tag & 0xC0) == 0x40:
            dr = ((tag >> 4) & 0x03) - 2
            dg = ((tag >> 2) & 0x03) - 2
            db = (tag & 0x03) - 2
            px[0] = (px[0] + dr) & 0xFF
            px[1] = (px[1] + dg) & 0xFF
            px[2] = (px[2] + db) & 0xFF
            pos += 1
            pixels[i] = px
            idx = (px[0] * 3 + px[1] * 5 + px[2] * 7 + px[3] * 11) % 64
            index[idx] = px
            i += 1
        elif (tag & 0xC0) == 0x80:
            b2 = data[pos + 1]
            dg = (tag & 0x3F) - 32
            dr_dg = (b2 >> 4) - 8
            db_dg = (b2 & 0x0F) - 8
            px[1] = (px[1] + dg) & 0xFF
            px[0] = (px[0] + dg + dr_dg) & 0xFF
            px[2] = (px[2] + dg + db_dg) & 0xFF
            pos += 2
            pixels[i] = px
            idx = (px[0] * 3 + px[1] * 5 + px[2] * 7 + px[3] * 11) % 64
            index[idx] = px
            i += 1
        else:  # QOI_OP_RUN
            run = (tag & 0x3F) + 1
            for _ in range(run):
                pixels[i] = px
                i += 1
            pos += 1
    return pixels[:, :3].reshape(height, width, 3)


def test_qoi_round_trip():
    rng = np.random.RandomState(0)
    cases = {
        "random": rng.randint(0, 256, (37, 53, 3), dtype=np.uint8),
        "solid": np.full((10, 100, 3), 123, dtype=np.uint8),
        "palette": np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
                            dtype=np.uint8)[rng.randint(0, 4, (20, 30))],
        "run_boundary_62": np.full((1, 62, 3), 50, dtype=np.uint8),
        "run_boundary_63": np.full((1, 63, 3), 50, dtype=np.uint8),
        "single_pixel": np.array([[[7, 8, 9]]], dtype=np.uint8),
    }
    for name, img in cases.items():
        data = m.encode_qoi_bytes(img)
        decoded = _qoi_reference_decode(data)
        assert np.array_equal(decoded, img), f"QOI round-trip failed for case: {name}"


def test_png_round_trip():
    rng = np.random.RandomState(1)
    img = rng.randint(0, 256, (40, 60, 3), dtype=np.uint8)
    data = m.encode_png_bytes(img)
    decoded = m.decode_png_bytes(data)
    assert np.array_equal(decoded, img)


def test_jxl_quality_to_distance_mapping():
    assert m.jxl_quality_to_distance(90) == 1.0   # ffmpeg/libjxl's own default
    assert m.jxl_quality_to_distance(100) < m.jxl_quality_to_distance(50)
    assert m.jxl_quality_to_distance(0) == 10.0


# --------------------------------------------------------------------------- #
# Downsampling
# --------------------------------------------------------------------------- #
def test_downsample_sum_clamps_on_overflow():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0], img[0, 1], img[1, 0], img[1, 1] = (200, 10, 0), (100, 250, 0), (50, 5, 0), (10, 0, 0)
    out = m.downsample_image(img, 2, "Sum")
    assert tuple(out[0, 0]) == (255, 255, 0)


def test_downsample_max_is_per_channel_independent():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0], img[0, 1], img[1, 0], img[1, 1] = (200, 10, 0), (100, 250, 0), (50, 5, 0), (10, 0, 0)
    out = m.downsample_image(img, 2, "Max")
    assert tuple(out[0, 0]) == (200, 250, 0)


# --------------------------------------------------------------------------- #
# Declarative shape-parameter table (SHAPE_PARAM_INFO)
# --------------------------------------------------------------------------- #
def test_shape_param_table_matches_snapshot_fields():
    non_shape = {"max_iter", "max_dist", "seg_mult", "shape", "fuzz",
                "buddha_blur_radius", "buddha_blur_type", "main_fire", "main_in_effect",
                "main_out_effect"}
    table_fields = {p.field for p in m.SHAPE_PARAM_INFO}
    assert table_fields == set(m.Snapshot._fields) - non_shape


def test_self_checks_catch_a_shape_param_table_snapshot_mismatch():
    fresh = load_module()
    fresh.SHAPE_PARAM_INFO = fresh.SHAPE_PARAM_INFO[:-1]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "Snapshot" in str(e)


def test_self_checks_catch_a_shape_param_table_duplicate():
    fresh = load_module()
    dup = fresh.SHAPE_PARAM_INFO[0]._replace(field="height")
    fresh.SHAPE_PARAM_INFO = [dup] + fresh.SHAPE_PARAM_INFO[1:]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "duplicate" in str(e)


def test_load_value_dispatches_every_shape_param_to_its_widget():
    """Regression coverage for the load_value/apply_snapshot_to_controls
    refactor: each of the 23 shape parameters used to have its own hand-
    written elif/assignment line; this confirms the table-driven version
    still reaches the exact same widget for every one of them."""
    class _Get:
        def __init__(self, value=None):
            self.value = value
        def get(self):
            return self.value
        def set(self, v):
            self.value = v

    app = _make_fake_app_for_snapshot()
    for p in m.SHAPE_PARAM_INFO:
        setattr(app, p.widget, _Get())
    app.load_value = m.App.load_value.__get__(app)

    for p in m.SHAPE_PARAM_INFO:
        test_val = 3 if p.fmt == "int" else 1.75
        app.load_value(p.key, test_val)
        assert getattr(app, p.widget).get() == test_val, \
            f"load_value({p.key!r}, ...) didn't reach {p.widget}"


def test_apply_snapshot_to_controls_reaches_every_shape_param_widget():
    class _Get:
        def __init__(self, value=None):
            self.value = value
        def get(self):
            return self.value
        def set(self, v):
            self.value = v

    app = _make_fake_app_for_snapshot()
    for p in m.SHAPE_PARAM_INFO:
        setattr(app, p.widget, _Get())
    for attr in ["iter_var", "dist_var", "seg_mult_scale", "shape_var", "fuzz_enabled_var",
                "emit_angle_scale", "emit_jitter_scale", "bnd_offset_scale",
                "bnd_jitter_scale", "bnd_mult_scale", "self_offset_scale",
                "self_jitter_scale", "self_mult_scale", "seed_var",
                "buddha_blur_radius_scale", "buddha_blur_type_var", "main_fire_var",
                "main_in_effect_var", "main_out_effect_var"]:
        setattr(app, attr, _Get())
    app.show_shape_controls = lambda: None
    app.update_fuzz_enabled_state = lambda: None
    app.apply_snapshot_to_controls = m.App.apply_snapshot_to_controls.__get__(app)

    s = m.DEFAULT_SNAPSHOT._replace(moss_size=1.9, lens_radius1=0.4, reuleaux_sides=7)
    app.apply_snapshot_to_controls(s)
    for p in m.SHAPE_PARAM_INFO:
        assert getattr(app, p.widget).get() == getattr(s, p.field), \
            f"apply_snapshot_to_controls didn't set {p.widget} from s.{p.field}"


# --------------------------------------------------------------------------- #
# Baseline behavior lock-in for the shape/obstacle de-duplication refactor:
# the bounding shape's own controls (read_shape_controls, load_value/
# apply_snapshot_to_controls) are already table-driven off SHAPE_PARAM_INFO;
# the obstacle side (refresh_obstacle_editor, on_obstacle_param_changed) is
# still 25+ hand-written lines each, and describe_shape/describe_obstacle and
# params()/snapshot_shape_params are still separate implementations. These
# tests capture their exact current behavior so the refactor can be verified
# against it rather than against assumptions.
# --------------------------------------------------------------------------- #
def _fake_obstacle_editor_app():
    """A minimal stand-in for App with stub controls for
    refresh_obstacle_editor() and on_obstacle_param_changed() to run for
    real, unmodified, off the actual App class."""
    class FakeApp:
        pass
    app = FakeApp()
    app.refresh_obstacle_editor = m.App.refresh_obstacle_editor.__get__(app)
    app.on_obstacle_param_changed = m.App.on_obstacle_param_changed.__get__(app)
    app.show_obstacle_shape_controls = m.App.show_obstacle_shape_controls.__get__(app)

    app.obs_shape_var = _StubVar(m.SHAPE_NAMES[m.SHAPE_ELLIPSE])
    app.obs_in_effect_var = _StubVar(m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_REFLECT])
    app.obs_out_effect_var = _StubVar(m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_PASS])
    app.obs_fire_inside_var = _StubVar(False)

    app.obs_ellipse_group = _TrackedFrame()
    app.obs_polygon_group = _TrackedFrame()
    app.obs_moss_group = _TrackedFrame()
    app.obs_sector_group = _TrackedFrame()
    app.obs_segment_group = _TrackedFrame()
    app.obs_lens_group = _TrackedFrame()
    app.obs_reuleaux_group = _TrackedFrame()
    app.obs_capsule_group = _TrackedFrame()

    for p in m.SHAPE_PARAM_INFO:
        setattr(app, "obs_" + p.widget, _StubVar(p.default))
    app.obs_radius_scale = _StubVar(0.5)            # poly_radius (obstacle-only)
    app.obs_reuleaux_radius_scale = _StubVar(0.5)   # reuleaux_radius (obstacle-only)
    app.obs_r_scale = _StubVar(0.0)
    app.obs_theta_scale = _StubVar(0.0)
    app.obs_cx_scale = _StubVar(0.0)
    app.obs_cy_scale = _StubVar(0.0)

    app.rebuild_calls = []
    app.rebuild_obstacle_list = lambda: app.rebuild_calls.append(1)
    app.param_changed_calls = []
    app.on_shape_param_changed = lambda: app.param_changed_calls.append(1)
    return app


def _make_distinct_obstacle(shape=None):
    """An Obstacle with a distinct, easily-checked value in every shared
    field, an obstacle-only value in the radius/position fields, and a
    non-default effect/fire_inside - so a test can tell every field apart
    from every other."""
    o = m.Obstacle(1)
    if shape is not None:
        o.shape = shape
    values = {"width": 1.11, "height": 1.22, "ell_rotation": 10.0, "sides": 6,
             "poly_rotation": 20.0, "moss_angle": 91.0, "moss_size": 1.33,
             "moss_rotation": 30.0, "sector_radius": 1.44, "sector_span": 70.0,
             "sector_rotation": 40.0, "segment_radius": 1.55, "segment_span": 80.0,
             "segment_rotation": 50.0, "lens_radius1": 1.66, "lens_radius2": 1.77,
             "lens_sep": 0.88, "lens_rotation": 60.0, "reuleaux_sides": 7,
             "reuleaux_rotation": 70.0, "capsule_length": 1.99, "capsule_radius": 0.55,
             "capsule_rotation": 80.0}
    for field, val in values.items():
        setattr(o, field, val)
    o.poly_radius = 0.61
    o.reuleaux_radius = 0.62
    o.r = 0.5
    o.theta = 45.0
    o.cx = 0.1
    o.cy = -0.2
    o.in_effect = m.OBSTACLE_EFFECT_TERMINATE
    o.out_effect = m.OBSTACLE_EFFECT_REFLECT
    o.fire_inside = True
    return o


def test_refresh_obstacle_editor_loads_every_shared_field():
    app = _fake_obstacle_editor_app()
    o = _make_distinct_obstacle()
    app.selected_obstacle = o
    app.refresh_obstacle_editor()
    for p in m.SHAPE_PARAM_INFO:
        got = getattr(app, "obs_" + p.widget).get()
        expected = getattr(o, p.field)
        assert got == expected, f"obs_{p.widget} = {got}, expected {expected} (o.{p.field})"


def test_refresh_obstacle_editor_loads_obstacle_only_fields():
    app = _fake_obstacle_editor_app()
    o = _make_distinct_obstacle()
    app.selected_obstacle = o
    app.refresh_obstacle_editor()
    assert app.obs_shape_var.get() == m.SHAPE_NAMES[o.shape]
    assert app.obs_radius_scale.get() == o.poly_radius
    assert app.obs_reuleaux_radius_scale.get() == o.reuleaux_radius
    assert app.obs_r_scale.get() == o.r
    assert app.obs_theta_scale.get() == o.theta
    assert app.obs_cx_scale.get() == o.cx
    assert app.obs_cy_scale.get() == o.cy
    assert app.obs_in_effect_var.get() == m.OBSTACLE_EFFECT_NAMES[o.in_effect]
    assert app.obs_out_effect_var.get() == m.OBSTACLE_EFFECT_NAMES[o.out_effect]
    assert app.obs_fire_inside_var.get() == o.fire_inside


def test_refresh_obstacle_editor_shows_only_the_selected_shapes_group():
    app = _fake_obstacle_editor_app()
    o = _make_distinct_obstacle(shape=m.SHAPE_LENS)
    app.selected_obstacle = o
    app.refresh_obstacle_editor()
    assert app.obs_lens_group.shown
    for other in (app.obs_ellipse_group, app.obs_polygon_group, app.obs_moss_group,
                 app.obs_sector_group, app.obs_segment_group, app.obs_reuleaux_group,
                 app.obs_capsule_group):
        assert not other.shown


def test_refresh_obstacle_editor_does_nothing_without_a_selected_obstacle():
    app = _fake_obstacle_editor_app()
    app.selected_obstacle = None
    app.refresh_obstacle_editor()   # must not raise


def test_on_obstacle_param_changed_writes_every_shared_field_with_correct_rounding():
    app = _fake_obstacle_editor_app()
    o = m.Obstacle(1)
    app.selected_obstacle = o
    app._obst_editor_loading = False
    for p in m.SHAPE_PARAM_INFO:
        widget = getattr(app, "obs_" + p.widget)
        widget.set(3 if p.fmt == "int" else 1.23456)
    app.on_obstacle_param_changed()
    for p in m.SHAPE_PARAM_INFO:
        got = getattr(o, p.field)
        expected = 3 if p.fmt == "int" else 1.23
        assert got == expected, f"o.{p.field} = {got}, expected {expected}"


def test_on_obstacle_param_changed_writes_obstacle_only_fields():
    app = _fake_obstacle_editor_app()
    o = m.Obstacle(1)
    app.selected_obstacle = o
    app._obst_editor_loading = False
    app.obs_radius_scale.set(0.6789)
    app.obs_reuleaux_radius_scale.set(0.7654)
    app.obs_r_scale.set(0.98765)
    app.obs_theta_scale.set(12.345)
    app.obs_cx_scale.set(0.1111)
    app.obs_cy_scale.set(-0.2222)
    app.on_obstacle_param_changed()
    assert o.poly_radius == 0.68
    assert o.reuleaux_radius == 0.77
    assert o.r == 0.99
    assert o.theta == 12.35 or o.theta == 12.34   # banker's-rounding-safe
    assert o.cx == 0.11
    assert o.cy == -0.22


def test_on_obstacle_param_changed_calls_rebuild_and_param_changed():
    app = _fake_obstacle_editor_app()
    app.selected_obstacle = m.Obstacle(1)
    app._obst_editor_loading = False
    app.on_obstacle_param_changed()
    assert len(app.rebuild_calls) == 1
    assert len(app.param_changed_calls) == 1


def test_on_obstacle_param_changed_skips_while_loading():
    app = _fake_obstacle_editor_app()
    o = m.Obstacle(1)
    o.width = 9.0
    app.selected_obstacle = o
    app._obst_editor_loading = True
    app.obs_width_scale.set(1.0)
    app.on_obstacle_param_changed()
    assert o.width == 9.0, "must not write anything while the editor is loading"
    assert app.rebuild_calls == []


def test_on_obstacle_param_changed_skips_without_a_selected_obstacle():
    app = _fake_obstacle_editor_app()
    app.selected_obstacle = None
    app._obst_editor_loading = False
    app.on_obstacle_param_changed()   # must not raise
    assert app.rebuild_calls == []


def test_obstacle_editor_round_trip_preserves_already_rounded_values():
    """refresh then immediately re-read-and-write must be a no-op, since
    _make_distinct_obstacle's values are already at 2 decimal places."""
    app = _fake_obstacle_editor_app()
    o = _make_distinct_obstacle()
    app.selected_obstacle = o
    app.refresh_obstacle_editor()
    app._obst_editor_loading = False
    app.on_obstacle_param_changed()
    for p in m.SHAPE_PARAM_INFO:
        assert getattr(o, p.field) == getattr(_make_distinct_obstacle(), p.field)


# describe_shape / describe_obstacle: exact current text for every shape,
# captured before unifying the two implementations
_DESCRIBE_SHAPE_CASES = [
    (m.SHAPE_ELLIPSE, (1.5, 2.5, 30.0), "Ellipse 1.50 \u00d7 2.50, rotated 30.0\u00b0"),
    (m.SHAPE_POLYGON, (6, 15.0), "Regular 6-gon, rotated 15.0\u00b0"),
    (m.SHAPE_MOSS_EGG, (91.0, 1.30, 5.0),
     "Moss's egg, angle ACB 91.0\u00b0, size 1.30, rotated 5.0\u00b0"),
    (m.SHAPE_SECTOR, (1.2, 70.0, 10.0),
     "Circular sector, radius 1.20, span 70.0\u00b0, rotated 10.0\u00b0"),
    (m.SHAPE_SEGMENT, (1.3, 80.0, 20.0),
     "Circular segment, radius 1.30, span 80.0\u00b0, rotated 20.0\u00b0"),
    (m.SHAPE_LENS, (1.1, 1.4, 0.6, 25.0),
     "Two-circle lens, radii 1.10/1.40, separation 0.60, rotated 25.0\u00b0"),
    (m.SHAPE_REULEAUX, (5, 35.0), "Reuleaux 5-gon, rotated 35.0\u00b0"),
    (m.SHAPE_CAPSULE, (2.0, 0.4, 45.0), "Capsule, length 2.00, radius 0.40, rotated 45.0\u00b0"),
]


def test_describe_shape_exact_text_for_every_shape():
    for shape, params, expected in _DESCRIBE_SHAPE_CASES:
        assert m.describe_shape(shape, params) == expected


def _make_obstacle_from_params(shape, params):
    """An Obstacle configured to match a _DESCRIBE_SHAPE_CASES params tuple,
    for the equivalent describe_obstacle case - fills in the one extra
    radius field polygon/Reuleaux obstacles have that the bounding shape's
    tuple doesn't carry."""
    o = m.Obstacle(1)
    o.shape = shape
    if shape == m.SHAPE_ELLIPSE:
        o.width, o.height, o.ell_rotation = params
    elif shape == m.SHAPE_POLYGON:
        o.sides, o.poly_rotation = params
        o.poly_radius = 0.5
    elif shape == m.SHAPE_MOSS_EGG:
        o.moss_angle, o.moss_size, o.moss_rotation = params
    elif shape == m.SHAPE_SECTOR:
        o.sector_radius, o.sector_span, o.sector_rotation = params
    elif shape == m.SHAPE_SEGMENT:
        o.segment_radius, o.segment_span, o.segment_rotation = params
    elif shape == m.SHAPE_LENS:
        o.lens_radius1, o.lens_radius2, o.lens_sep, o.lens_rotation = params
    elif shape == m.SHAPE_REULEAUX:
        o.reuleaux_sides, o.reuleaux_rotation = params
        o.reuleaux_radius = 0.5
    else:
        o.capsule_length, o.capsule_radius, o.capsule_rotation = params
    return o


_DESCRIBE_OBSTACLE_SHAPE_TEXT = {
    m.SHAPE_ELLIPSE: "Ellipse 1.50\u00d72.50, rot 30.0\u00b0",
    m.SHAPE_POLYGON: "6-gon r0.50, rot 15.0\u00b0",
    m.SHAPE_MOSS_EGG: "Moss's egg 91\u00b0, size 1.30, rot 5.0\u00b0",
    m.SHAPE_SECTOR: "Sector r1.20 70\u00b0",
    m.SHAPE_SEGMENT: "Segment r1.30 80\u00b0",
    m.SHAPE_LENS: "Lens r1.10/1.40 d0.60",
    m.SHAPE_REULEAUX: "Reuleaux-5 r0.50",
    m.SHAPE_CAPSULE: "Capsule L2.00 r0.40",
}


def test_describe_obstacle_exact_shape_text_for_every_shape():
    for shape, params, _ in _DESCRIBE_SHAPE_CASES:
        o = _make_obstacle_from_params(shape, params)
        o.r, o.theta, o.cx, o.cy = 0.0, 0.0, 0.0, 0.0
        text = m.describe_obstacle(o)
        assert text.startswith(_DESCRIBE_OBSTACLE_SHAPE_TEXT[shape]), \
            f"got: {text!r}"


def test_describe_obstacle_includes_position_and_effect():
    o = _make_obstacle_from_params(m.SHAPE_ELLIPSE, (1.0, 1.0, 0.0))
    o.r, o.theta, o.cx, o.cy = 0.5, 45.0, 0.1, -0.2
    o.in_effect = m.OBSTACLE_EFFECT_TERMINATE
    o.out_effect = m.OBSTACLE_EFFECT_REFLECT
    text = m.describe_obstacle(o)
    assert "r0.50 @ 45.0\u00b0" in text
    assert "+(+0.10, -0.20)" in text
    assert f"in: {m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_TERMINATE]}" in text
    assert f"out: {m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_REFLECT]}" in text


# params() / snapshot_shape_params: exact tuples for every shape, including
# the polygon/Reuleaux asymmetry (an obstacle carries its own radius; the
# bounding shape doesn't, since it's always inscribed in the unit circle)
def test_snapshot_shape_params_matches_obstacle_params_for_shared_fields():
    for shape, params, _ in _DESCRIBE_SHAPE_CASES:
        o = _make_obstacle_from_params(shape, params)
        snap = m.DEFAULT_SNAPSHOT._replace(shape=shape)
        # apply the same field values to the Snapshot side
        field_map = {
            m.SHAPE_ELLIPSE: ("width", "height", "ell_rotation"),
            m.SHAPE_POLYGON: ("sides", "poly_rotation"),
            m.SHAPE_MOSS_EGG: ("moss_angle", "moss_size", "moss_rotation"),
            m.SHAPE_SECTOR: ("sector_radius", "sector_span", "sector_rotation"),
            m.SHAPE_SEGMENT: ("segment_radius", "segment_span", "segment_rotation"),
            m.SHAPE_LENS: ("lens_radius1", "lens_radius2", "lens_sep", "lens_rotation"),
            m.SHAPE_REULEAUX: ("reuleaux_sides", "reuleaux_rotation"),
            m.SHAPE_CAPSULE: ("capsule_length", "capsule_radius", "capsule_rotation"),
        }[shape]
        snap = snap._replace(**dict(zip(field_map, params)))
        assert m.snapshot_shape_params(snap) == params

        obst_params = o.params()
        if shape == m.SHAPE_POLYGON:
            assert obst_params == params + (o.poly_radius,), \
                "obstacle polygon params must carry an extra trailing radius"
        elif shape == m.SHAPE_REULEAUX:
            assert obst_params == params + (o.reuleaux_radius,), \
                "obstacle Reuleaux params must carry an extra trailing radius"
        else:
            assert obst_params == params, \
                "for every other shape, obstacle and bounding-shape params must match exactly"


def test_obstacle_params_polygon_and_reuleaux_have_one_more_field_than_snapshot():
    """Explicit regression lock for the asymmetry itself, independent of the
    parametrized comparison above."""
    o_poly = m.Obstacle(1)
    o_poly.shape = m.SHAPE_POLYGON
    assert len(o_poly.params()) == 3   # sides, poly_rotation, poly_radius
    snap_poly = m.DEFAULT_SNAPSHOT._replace(shape=m.SHAPE_POLYGON)
    assert len(m.snapshot_shape_params(snap_poly)) == 2   # sides, poly_rotation

    o_reu = m.Obstacle(1)
    o_reu.shape = m.SHAPE_REULEAUX
    assert len(o_reu.params()) == 3   # reuleaux_sides, reuleaux_rotation, reuleaux_radius
    snap_reu = m.DEFAULT_SNAPSHOT._replace(shape=m.SHAPE_REULEAUX)
    assert len(m.snapshot_shape_params(snap_reu)) == 2   # reuleaux_sides, reuleaux_rotation


def test_shape_params_tuple_works_off_either_a_snapshot_or_an_obstacle():
    """The whole point of the shared function: it reads the same field
    names off either kind of object without needing to know which."""
    snap = m.DEFAULT_SNAPSHOT._replace(shape=m.SHAPE_MOSS_EGG, moss_angle=91.0,
                                       moss_size=1.3, moss_rotation=5.0)
    o = m.Obstacle(1)
    o.shape = m.SHAPE_MOSS_EGG
    o.moss_angle, o.moss_size, o.moss_rotation = 91.0, 1.3, 5.0
    expected = (91.0, 1.3, 5.0)
    assert m.shape_params_tuple(m.SHAPE_MOSS_EGG, snap) == expected
    assert m.shape_params_tuple(m.SHAPE_MOSS_EGG, o) == expected


def test_shape_params_tuple_include_obstacle_only_flag_for_every_shape():
    o = m.Obstacle(1)
    o.poly_radius = 0.6
    o.reuleaux_radius = 0.7
    for shape in (m.SHAPE_ELLIPSE, m.SHAPE_POLYGON, m.SHAPE_MOSS_EGG, m.SHAPE_SECTOR,
                 m.SHAPE_SEGMENT, m.SHAPE_LENS, m.SHAPE_REULEAUX, m.SHAPE_CAPSULE):
        o.shape = shape
        without_extra = m.shape_params_tuple(shape, o, include_obstacle_only=False)
        with_extra = m.shape_params_tuple(shape, o, include_obstacle_only=True)
        if shape == m.SHAPE_POLYGON:
            assert with_extra == without_extra + (0.6,)
        elif shape == m.SHAPE_REULEAUX:
            assert with_extra == without_extra + (0.7,)
        else:
            assert with_extra == without_extra, f"shape={shape} must be unaffected by the flag"


# Keyframe capture/load round-trip for obstacle fields
def test_obstacle_keyframe_capture_and_load_round_trip():
    app = _fake_render_app()
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    app.current_snapshot = lambda: m.DEFAULT_SNAPSHOT
    app.deco = {"outline": _StubVar(True)}
    app.views = [m.ViewConfig(0)]
    o1 = _make_distinct_obstacle(shape=m.SHAPE_MOSS_EGG)
    o1.id = 1
    o2 = _make_distinct_obstacle(shape=m.SHAPE_LENS)
    o2.id = 2
    o2.width = 4.44   # a value o1 doesn't have, to catch id cross-contamination
    app.obstacles = [o1, o2]

    captured = app.capture_all_settings()
    for field, _label, _is_float in m.OBST_FIELD_INFO:
        assert captured[f"obst:1:{field}"] == getattr(o1, field)
        assert captured[f"obst:2:{field}"] == getattr(o2, field)

    # load onto fresh obstacles with matching ids
    fresh1 = m.Obstacle(1)
    fresh2 = m.Obstacle(2)
    load_app = _fake_obstacle_editor_app()
    load_app.load_value = m.App.load_value.__get__(load_app)
    load_app.obstacles = [fresh1, fresh2]
    load_app.selected_obstacle = None
    load_app.rebuild_obstacle_list = lambda: None
    for key, val in captured.items():
        if key.startswith("obst:"):
            load_app.load_value(key, val)
    for field, _label, _is_float in m.OBST_FIELD_INFO:
        assert getattr(fresh1, field) == getattr(o1, field), f"field={field}"
        assert getattr(fresh2, field) == getattr(o2, field), f"field={field}"


def test_obstacle_keyframe_load_only_touches_the_matching_id():
    app = _fake_obstacle_editor_app()
    app.load_value = m.App.load_value.__get__(app)
    o1 = m.Obstacle(1)
    o2 = m.Obstacle(2)
    o1.width, o2.width = 1.0, 2.0
    app.obstacles = [o1, o2]
    app.selected_obstacle = None
    app.rebuild_obstacle_list = lambda: None
    app.load_value("obst:1:width", 9.0)
    assert o1.width == 9.0
    assert o2.width == 2.0


# --------------------------------------------------------------------------- #
# Snapshot/Settings construction (keyword-argument refactor)
# --------------------------------------------------------------------------- #
def _make_fake_app_for_snapshot():
    """A minimal stand-in for App with just enough stub controls for
    read_shape_controls(), read_fuzz_controls() and current_snapshot() to
    run for real, unmodified, off the actual App class."""
    class _Get:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    class FakeApp:
        pass

    app = FakeApp()
    app.read_shape_controls = m.App.read_shape_controls.__get__(app)
    app.read_fuzz_controls = m.App.read_fuzz_controls.__get__(app)
    app.control_shape = m.App.control_shape.__get__(app)
    app.current_snapshot = m.App.current_snapshot.__get__(app)

    app.shape_var = _Get(m.SHAPE_NAMES[m.SHAPE_MOSS_EGG])
    shape_fields = {
        "width": 2.0, "height": 2.0, "ell_rot": 0.0, "sides": 5, "poly_rot": 0.0,
        "moss_angle": 90.0, "moss_size": 1.3, "moss_rot": 0.0, "sector_radius": 1.0,
        "sector_span": 90.0, "sector_rot": 0.0, "segment_radius": 1.0,
        "segment_span": 120.0, "segment_rot": 0.0, "lens_r1": 1.0, "lens_r2": 1.0,
        "lens_sep": 1.0, "lens_rot": 0.0, "reuleaux_sides": 3, "reuleaux_rot": 0.0,
        "capsule_length": 1.0, "capsule_radius": 0.5, "capsule_rot": 0.0,
    }
    for field, value in shape_fields.items():
        setattr(app, f"{field}_scale", _Get(value))

    app.seed_var = _Get("0")
    app.last_seed = 0
    app.fuzz_enabled_var = _Get(False)
    for field in ["emit_angle", "emit_jitter", "bnd_offset", "bnd_jitter", "bnd_mult",
                 "self_offset", "self_jitter", "self_mult"]:
        setattr(app, f"{field}_scale", _Get(1.0 if field.endswith("mult") else 0.0))

    app.iter_var = _Get("32")
    app.dist_var = _Get("0")
    app.seg_mult_scale = _Get(0.75)
    app.buddha_blur_radius_scale = _Get(m.DEFAULT_BUDDHA_BLUR_RADIUS)
    app.buddha_blur_type_var = _Get(m.BUDDHA_BLUR_LABELS[m.DEFAULT_BUDDHA_BLUR_TYPE])
    app.main_fire_var = _Get(True)
    app.main_in_effect_var = _Get(m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_REFLECT])
    app.main_out_effect_var = _Get(m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_REFLECT])
    return app


def test_read_shape_controls_feeds_snapshot_by_keyword():
    """Locks in the keyword-construction refactor: read_shape_controls()
    returns a dict, and current_snapshot()/control_shape() build a real
    Snapshot from it with Snapshot(**...). If a key here were ever renamed on
    one side without the other, or a field forgotten, this fails with a
    TypeError naming the exact problem - not a silently misplaced value."""
    app = _make_fake_app_for_snapshot()
    snap = app.current_snapshot()
    assert snap is not None
    assert snap.max_iter == 32 and snap.max_dist == 0 and snap.seg_mult == 0.75
    assert snap.shape == m.SHAPE_MOSS_EGG
    assert snap.moss_angle == 90.0 and snap.moss_size == 1.3

    shape, params = app.control_shape()
    assert shape == m.SHAPE_MOSS_EGG
    assert params == (90.0, 1.3, 0.0)   # (angle, size, rotation), see snapshot_shape_params


# --------------------------------------------------------------------------- #
# Views (multiple side-by-side colorations of the same simulation)
# --------------------------------------------------------------------------- #
def _fake_render_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.render_full_image = m.App.render_full_image.__get__(app)
    app.colorize_view = m.App.colorize_view.__get__(app)
    app.compute_ring = m.App.compute_ring.__get__(app)
    return app


def _empty_obst_arrays():
    empty_i = np.empty(0, dtype=np.int64)
    empty_b = np.empty((0, m.OBST_BOUND_LEN))
    empty_f = np.empty(0)
    empty_bool = np.empty(0, dtype=bool)
    return (empty_i, empty_b, empty_f, empty_f, np.empty((0, 2), dtype=np.int64),
            empty_bool)


def _view_settings_from_config(v):
    return m.ViewSettings(v.mode, v.kinds, v.point_metric, v.other_metric, v.point1_choice,
                          v.point2_choice, v.chan_src, v.buddha_points, v.buddha_compression,
                          v.buddha_brightness, v.constant_choice)


def test_render_full_image_with_a_single_view_matches_pre_views_behavior():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES), "Linear", None)
    data = m.resolve_frame_settings([kf], 0)
    views = (_view_settings_from_config(m.ViewConfig(0)),)
    img = app.render_full_image(data, 40, 40, _empty_obst_arrays(), views)
    assert img.shape == (40, 40, 3)


def test_two_identical_views_render_as_two_identical_halves():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES), "Linear", None)
    data = m.resolve_frame_settings([kf], 0)
    obst_arrays = _empty_obst_arrays()
    one_view = (_view_settings_from_config(m.ViewConfig(0)),)
    single = app.render_full_image(data, 50, 50, obst_arrays, one_view)
    doubled = app.render_full_image(data, 50, 50, obst_arrays, one_view * 2)
    assert doubled.shape == (50, 100, 3)
    assert np.array_equal(doubled[:, :50], doubled[:, 50:])
    assert np.array_equal(doubled[:, :50], single)


def test_two_different_views_render_as_two_different_halves():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES), "Linear", None)
    data = m.resolve_frame_settings([kf], 0)
    v0 = m.ViewConfig(0)
    v1 = m.ViewConfig(1)
    v1.mode = "HSV"
    views = (_view_settings_from_config(v0), _view_settings_from_config(v1))
    combined = app.render_full_image(data, 50, 50, _empty_obst_arrays(), views)
    assert not np.array_equal(combined[:, :50], combined[:, 50:])


def test_view_config_defaults_match_the_module_defaults():
    v = m.ViewConfig(1)
    assert v.mode == m.DEFAULT_MODE
    assert list(v.chan_src) == list(m.DEFAULT_CHAN_SRCS)
    assert list(v.kinds) == list(m.DEFAULT_KINDS)


def test_view_config_per_channel_fields_are_independent_per_instance():
    """Regression coverage for the shallow-copy hazard: two ViewConfigs must
    never share the same underlying list."""
    a = m.ViewConfig(1)
    b = m.ViewConfig(2)
    a.chan_src[0] = 999
    assert b.chan_src[0] != 999


def test_resolve_views_falls_back_to_live_state():
    v = m.ViewConfig(7)
    snap = m.snapshot_view(v)
    resolved = m.resolve_views([snap], [], 0)
    assert len(resolved) == 1
    assert resolved[0].mode == v.mode
    assert list(resolved[0].chan_src) == list(v.chan_src)


def test_resolve_views_works_uniformly_for_id_zero():
    """id 0 must not be treated specially - it's an ordinary view like any
    other, resolved the same way."""
    v = m.ViewConfig(0)
    v.mode = "HSV"
    snap = m.snapshot_view(v)
    kf = m.Keyframe(1, 0, {"view:0:mode": "Grayscale"}, "Linear", None)
    resolved = m.resolve_views([snap], [kf], 0)[0]
    assert resolved.mode == "Grayscale"


def test_resolve_views_honors_keyframe_overrides_per_channel():
    """A keyframe can override one of a view's channels without disturbing
    the others."""
    v = m.ViewConfig(9)
    snap = m.snapshot_view(v)
    custom_chan = (m.CHAN_SRC_QUERY, m.KIND_OTHER, m.ST_BOUNCES, m.POINT_START, m.POINT_END,
                  m.DEFAULT_BUDDHA_POINTS, m.BUDDHA_COMPRESS_MODULUS, 3, m.NONE_ZERO)
    kf = m.Keyframe(1, 0, {"view:9:mode": "HSV", "view:9:chan0": custom_chan}, "Linear", None)
    resolved = m.resolve_views([snap], [kf], 0)[0]
    assert resolved.mode == "HSV"
    assert resolved.kinds[0] == m.KIND_OTHER and resolved.buddha_brightness[0] == 3
    assert resolved.kinds[1] == m.DEFAULT_KINDS[1]   # untouched channel keeps the live default


def test_capture_all_settings_has_no_first_view_special_case():
    app = _fake_render_app()
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    app.current_snapshot = lambda: m.DEFAULT_SNAPSHOT
    app.deco = {"outline": type("X", (), {"get": lambda self: True})()}
    v0 = m.ViewConfig(0)
    v1 = m.ViewConfig(5)
    app.obstacles = []
    app.views = [v0, v1]
    d = app.capture_all_settings()
    assert d is not None
    assert "mode" not in d and "chan0" not in d, \
        "the first view must not get fixed bare keys anymore"
    for v in (v0, v1):
        assert f"view:{v.id}:mode" in d and d[f"view:{v.id}:mode"] == v.mode
        assert all(f"view:{v.id}:chan{c}" in d for c in range(3))


def test_setting_fallback_handles_view_keys_without_crashing():
    """Regression test: setting_fallback used to fall through to
    DEFAULT_SETTING_VALUES[key] for anything that wasn't an obst: key,
    which raised KeyError for any view: key (view fields are dynamic, never
    in that dict) - this is exactly the path load_animated_setting takes."""
    app = _fake_render_app()
    app.views = [m.ViewConfig(0)]
    app.obstacles = []
    app.setting_fallback = m.App.setting_fallback.__get__(app)
    app.setting_is_float = m.App.setting_is_float.__get__(app)
    app.view_by_id = m.App.view_by_id.__get__(app)

    assert app.setting_fallback("view:0:mode") == m.DEFAULT_MODE
    assert app.setting_fallback("view:0:chan0") is not None
    assert app.setting_is_float("view:0:mode") is False
    assert app.setting_fallback("view:99:mode") is None   # a deleted view: no crash


def _fake_view_list_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.views = [m.ViewConfig(0)]
    app._view_counter = 1
    app.selected_view_id = 0
    app.view_by_id = m.App.view_by_id.__get__(app)
    app.live_view = m.App.live_view.__get__(app)
    app.selected_view_obj = m.App.selected_view_obj.__get__(app)
    app.all_view_ids = m.App.all_view_ids.__get__(app)
    app.new_view = m.App.new_view.__get__(app)
    app.duplicate_view = m.App.duplicate_view.__get__(app)
    app.delete_view = m.App.delete_view.__get__(app)
    app.select_view = m.App.select_view.__get__(app)
    app.refresh_live_preview_state = m.App.refresh_live_preview_state.__get__(app)
    app.rebuild_view_list = lambda: None
    app.refresh_color_controls_for_selected_view = lambda: None
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
    app.view_thumb_cache = {}
    app.keyframes = []
    app.refresh_timeline_diffs = lambda: None
    app.purge_keyframe_settings = m.App.purge_keyframe_settings.__get__(app)
    return app


def test_new_view_inserts_right_after_the_selected_one():
    app = _fake_view_list_app()
    app.new_view()
    assert app.all_view_ids() == [0, 1]
    assert app.selected_view_id == 1
    app.new_view()
    assert app.all_view_ids() == [0, 1, 2]
    assert app.selected_view_id == 2


def test_duplicate_view_copies_settings_and_inserts_after_source():
    app = _fake_view_list_app()
    app.new_view()
    app.new_view()
    app.select_view(0)
    app.duplicate_view()
    assert app.all_view_ids() == [0, 3, 1, 2]
    assert app.selected_view_id == 3
    dup = app.view_by_id(3)
    src = app.view_by_id(0)
    assert dup.mode == src.mode and list(dup.chan_src) == list(src.chan_src)


def test_duplicate_view_copy_does_not_share_list_state_with_source():
    """Regression coverage for the shallow-copy hazard in duplicate_view."""
    app = _fake_view_list_app()
    app.duplicate_view()
    dup = app.selected_view_obj()
    src = app.view_by_id(0)
    dup.chan_src[0] = 999
    assert src.chan_src[0] != 999


def test_delete_view_refuses_the_only_remaining_view():
    app = _fake_view_list_app()
    before = app.all_view_ids()
    app.delete_view()
    assert app.all_view_ids() == before, "the last remaining view must never be deletable"


def test_delete_view_allows_deleting_the_first_view_when_others_exist():
    """The core payoff of unifying the data model: deleting position 0 is
    now completely ordinary - whichever view is now first becomes the one
    live_view() (and so the top-left preview) shows, with no promotion step
    and no keyframe-key renaming needed."""
    app = _fake_view_list_app()
    app.new_view()                      # ids: [0, 1]
    app.select_view(0)
    app.delete_view()
    assert app.all_view_ids() == [1]
    assert app.selected_view_id == 1
    assert app.live_view().id == 1


def test_delete_view_removes_an_extra_view_and_keeps_the_first():
    app = _fake_view_list_app()
    app.new_view()
    app.delete_view()
    assert app.all_view_ids() == [0]
    assert app.selected_view_id == 0


# --------------------------------------------------------------------------- #
# Export includes every view, side by side (the actual save/render call
# chains, not just render_full_image in isolation - these exercise the real
# path from "how many views exist right now" through to the output file)
# --------------------------------------------------------------------------- #
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
    app.views = [m.ViewConfig(0)]
    app._view_counter = 1
    app.selected_view_id = 0
    app.include_all_views_var = _StubVar(True)
    app.color_info_location_var = _StubVar("None")
    app.sim_info_location_var = _StubVar("None")
    app.generated_settings = m.effective_settings(m.DEFAULT_SNAPSHOT)
    app.view_by_id = m.App.view_by_id.__get__(app)
    app.live_view = m.App.live_view.__get__(app)
    app.selected_view_obj = m.App.selected_view_obj.__get__(app)
    app.all_view_ids = m.App.all_view_ids.__get__(app)
    app.views_to_export = m.App.views_to_export.__get__(app)
    app.new_view = m.App.new_view.__get__(app)
    app.refresh_live_preview_state = m.App.refresh_live_preview_state.__get__(app)
    app.rebuild_buddha_counts = lambda: None
    app.update_stat_tags = lambda: None
    app.recolor = lambda r0, r1: None
    app.draw = lambda: None
    app.update_preview = lambda i, j: None
    app.rows_done = 0
    app.pinned_px = None
    app.view_thumb_cache = {}
    app.rebuild_view_list = lambda: None
    app.refresh_color_controls_for_selected_view = lambda: None
    app.obstacles = []
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    app.current_snapshot = lambda: m.DEFAULT_SNAPSHOT
    app.deco = {"outline": _StubVar(False)}
    app.render_full_image = m.App.render_full_image.__get__(app)
    app.colorize_view = m.App.colorize_view.__get__(app)
    app.compute_ring = m.App.compute_ring.__get__(app)
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


def test_new_image_export_widens_when_a_second_view_is_added_and_all_views_included():
    """The scenario the person actually cares about: does exporting a "New
    figure" actually get wider when you add a view AND turn on "Include all
    views" - through the real save_new_image -> _save_new_image_worker ->
    render_full_image chain, not just render_full_image called directly
    with a hand-built views list."""
    app = _fake_export_app()
    app.include_all_views_var = _StubVar(True)
    app.save_new_w_var = _StubVar("20")
    app.save_new_h_var = _StubVar("20")
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.save_btn = _StubWidget()
    app.save_as_btn = _StubWidget()
    app.save_new_image = m.App.save_new_image.__get__(app)
    app._save_new_image_worker = m.App._save_new_image_worker.__get__(app)
    app._save_new_image_done = m.App._save_new_image_done.__get__(app)
    app._save_new_image_failed = m.App._save_new_image_failed.__get__(app)
    app.root = type("R", (), {"after": lambda self, delay, fn: fn()})()

    tmpdir = tempfile.mkdtemp()
    orig_export_dir = m.EXPORT_DIR
    m.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img_one_view = m.read_png(os.path.join(tmpdir, files[0]))
        assert img_one_view.shape == (20, 20, 3)
        os.remove(os.path.join(tmpdir, files[0]))

        app.new_view()
        assert app.all_view_ids() == [0, 1]
        _run_save_new_image_synchronously(app)
        files2 = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files2) == 1
        img_two_views = m.read_png(os.path.join(tmpdir, files2[0]))
        assert img_two_views.shape == (20, 40, 3), \
            f"adding a second view should double the export width, got {img_two_views.shape}"
    finally:
        m.EXPORT_DIR = orig_export_dir


def test_include_all_views_defaults_to_true():
    app = _fake_export_app()
    assert app.include_all_views_var.get() is True


def test_new_image_export_stays_single_width_when_all_views_not_included():
    """With "Include all views" explicitly off, adding a second view must
    NOT widen the export - only the selected/live view goes out. (The
    checkbox now defaults to on, so this test turns it off explicitly
    rather than relying on a default.)"""
    app = _fake_export_app()
    app.include_all_views_var = _StubVar(False)
    app.save_new_w_var = _StubVar("20")
    app.save_new_h_var = _StubVar("20")
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.save_btn = _StubWidget()
    app.save_as_btn = _StubWidget()
    app.save_new_image = m.App.save_new_image.__get__(app)
    app._save_new_image_worker = m.App._save_new_image_worker.__get__(app)
    app._save_new_image_done = m.App._save_new_image_done.__get__(app)
    app._save_new_image_failed = m.App._save_new_image_failed.__get__(app)
    app.root = type("R", (), {"after": lambda self, delay, fn: fn()})()

    app.new_view()
    tmpdir = tempfile.mkdtemp()
    orig_export_dir = m.EXPORT_DIR
    m.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img = m.read_png(os.path.join(tmpdir, files[0]))
        assert img.shape == (20, 20, 3), \
            f"with 'Include all views' off, export must stay single-width, got {img.shape}"
    finally:
        m.EXPORT_DIR = orig_export_dir


def test_video_frame_render_widens_when_a_second_view_is_added_and_included():
    """Same property, for the video-render call chain: snapshot
    views_to_export() the way the render launch site now does, resolve
    them, and render - before and after adding a view, with 'Include all
    views' on."""
    app = _fake_export_app()
    app.include_all_views_var = _StubVar(True)
    kf = m.Keyframe(0, 0, dict(m.DEFAULT_SETTING_VALUES), "Linear", None)
    keyframes = [kf]
    obst_snapshot = [m.snapshot_obstacle(o) for o in app.obstacles]
    data = m.resolve_frame_settings(keyframes, 0)
    obst_arrays = m.resolve_obstacle_arrays(obst_snapshot, keyframes, 0)

    view_snapshot = [m.snapshot_view(v) for v in app.views_to_export()]
    views = m.resolve_views(view_snapshot, keyframes, 0)
    img_one_view = app.render_full_image(data, 20, 20, obst_arrays, views)
    assert img_one_view.shape == (20, 20, 3)

    app.new_view()
    view_snapshot2 = [m.snapshot_view(v) for v in app.views_to_export()]   # as the render
                                                                           # launch site
                                                                           # re-snapshots
    views2 = m.resolve_views(view_snapshot2, keyframes, 0)
    img_two_views = app.render_full_image(data, 20, 20, obst_arrays, views2)
    assert img_two_views.shape == (20, 40, 3), \
        f"adding a second view should double the video frame width, got {img_two_views.shape}"


# --------------------------------------------------------------------------- #
# "Current figure" export (_write/render_current_figure): reuses the
# already-traced simulation rather than re-simulating, and respects
# "Include all views" the same way the other two export paths do
# --------------------------------------------------------------------------- #
def _fake_current_figure_app():
    app = _fake_export_app()
    app.stats = np.random.RandomState(0).rand(m.SIZE, m.SIZE, m.N_STATS).astype(np.float32)
    app.points = (np.random.RandomState(1).rand(m.SIZE, m.SIZE, m.N_POINTS, 2)
                 .astype(np.float32) * 2 - 1)
    app.status = np.ones((m.SIZE, m.SIZE), dtype=np.uint8)
    app.max_iter = m.DEFAULT_ITER
    app.max_dist = m.effective_dist(0.0)
    app.gen_seg_mult = 1.0
    app.gen_shape = m.SHAPE_ELLIPSE
    app.gen_bound = m.ellipse_bound(2.0, 2.0, 0.0)
    app.gen_fuzz = m.fuzz_array(m.DEFAULT_FUZZ)
    app.gen_seed_base = 0
    app.gen_buddha_blur_radius = m.DEFAULT_BUDDHA_BLUR_RADIUS
    app.gen_buddha_blur_type = m.DEFAULT_BUDDHA_BLUR_TYPE
    empty_i = np.empty(0, dtype=np.int64)
    empty_b = np.empty((0, m.OBST_BOUND_LEN))
    empty_f = np.empty(0)
    empty_bool = np.empty(0, dtype=bool)
    app.gen_obst_shapes, app.gen_obst_bounds = empty_i, empty_b
    app.gen_obst_cx, app.gen_obst_cy = empty_f, empty_f
    app.gen_obst_effects = np.empty((0, 2), dtype=np.int64)
    app.gen_obst_fire_inside = empty_bool
    app.ring = np.zeros((m.SIZE, m.SIZE), dtype=bool)
    app.gx, app.gy = m.make_grid(m.SIZE, m.SIZE)
    app.obstacle_ring_specs = lambda: []
    app.frame = np.full((m.SIZE, m.SIZE, 3), 42, dtype=np.uint8)   # the "already
                                                                   # rendered" preview
    app.render_current_figure = m.App.render_current_figure.__get__(app)
    app.render_view_thumbnail = m.App.render_view_thumbnail.__get__(app)
    app._write = m.App._write.__get__(app)
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    return app


def test_render_view_thumbnail_does_not_crash_for_every_view_kind():
    """Regression test: render_view_thumbnail builds its own ViewSettings
    manually (like render_current_figure does) and had no coverage at all -
    that's exactly how a missing constant_choice argument shipped
    undetected. Exercises it for Query, Buddha, and Constant channels."""
    app = _fake_current_figure_app()
    v = m.ViewConfig(0)
    v.chan_src[0] = m.CHAN_SRC_CONSTANT
    v.constant_choice[0] = m.NONE_ONE
    v.chan_src[1] = m.CHAN_SRC_BUDDHA
    thumb = app.render_view_thumbnail(v, m.THUMB_SIZE)
    assert thumb.shape == (m.THUMB_SIZE, m.THUMB_SIZE, 3)
    assert (thumb[..., 0] == 255).all()   # channel 0 is Constant, Always 1


def test_render_view_thumbnail_reuses_self_img_for_the_live_view():
    """The performance fix itself: the live view's thumbnail must be an
    exact downsample of self.img (already kept current by recolor()), not
    a fresh colorize_view computation - this is what avoids redundantly
    re-tracing every ray for a "multi" Buddha point option on every single
    color-setting change."""
    app = _fake_current_figure_app()
    app.img = np.random.RandomState(0).randint(0, 256, (m.SIZE, m.SIZE, 3), dtype=np.uint8)
    live = app.live_view()
    assert live is app.views[0]
    thumb = app.render_view_thumbnail(live, m.THUMB_SIZE)
    ys = (np.arange(m.THUMB_SIZE) * m.SIZE / m.THUMB_SIZE).astype(np.int64)
    xs = (np.arange(m.THUMB_SIZE) * m.SIZE / m.THUMB_SIZE).astype(np.int64)
    expected = app.img[np.ix_(ys, xs)]
    assert np.array_equal(thumb, expected)


def test_render_view_thumbnail_never_calls_colorize_view_for_the_live_view():
    """Belt-and-suspenders on the same guarantee: even a "multi" Buddha
    point (the expensive, ray-re-tracing case) must not reach
    colorize_view at all when thumbnailing the live view."""
    app = _fake_current_figure_app()
    app.img = np.zeros((m.SIZE, m.SIZE, 3), dtype=np.uint8)
    live = app.live_view()
    live.chan_src[0] = m.CHAN_SRC_BUDDHA
    live.buddha_points[0] = tuple(i == 7 for i in range(m.N_BUDDHA_POINTS))  # a "multi" point

    def _boom(*a, **k):
        raise AssertionError("colorize_view must not be called for the live view's thumbnail")
    app.colorize_view = _boom
    app.render_view_thumbnail(live, m.THUMB_SIZE)   # must not raise


def test_render_view_thumbnail_still_uses_colorize_view_for_a_non_live_view():
    """The other half: a view that ISN'T live has no incrementally-updated
    self.img to reuse, so it must still go through colorize_view as
    before."""
    app = _fake_current_figure_app()
    app.img = np.full((m.SIZE, m.SIZE, 3), 42, dtype=np.uint8)   # sentinel: must be unused
    other = m.ViewConfig(99)
    other.mode = "HSV"
    assert other is not app.live_view()
    thumb = app.render_view_thumbnail(other, m.THUMB_SIZE)
    assert not np.array_equal(thumb, np.full((m.THUMB_SIZE, m.THUMB_SIZE, 3), 42, dtype=np.uint8))


def test_render_view_thumbnail_caches_a_non_live_view_across_calls():
    """The third performance fix: rendering the SAME non-live view's
    thumbnail twice, with nothing having changed (rows_done unchanged),
    must only actually compute it once - this is what keeps editing one
    view fast even when a DIFFERENT, unrelated view uses an expensive
    "multi" Buddha point option, since rebuild_view_list re-renders every
    view's thumbnail on every single color edit."""
    app = _fake_current_figure_app()
    other = m.ViewConfig(99)
    calls = []
    real_colorize = app.colorize_view
    def counting_colorize(*a, **k):
        calls.append(1)
        return real_colorize(*a, **k)
    app.colorize_view = counting_colorize
    thumb1 = app.render_view_thumbnail(other, m.THUMB_SIZE)
    thumb2 = app.render_view_thumbnail(other, m.THUMB_SIZE)
    assert len(calls) == 1, "the second call should reuse the cache, not recompute"
    assert np.array_equal(thumb1, thumb2)


def test_render_view_thumbnail_recomputes_a_non_live_view_after_rows_done_advances():
    """When new simulation data becomes available, a cached non-live
    thumbnail must not be reused, even though the view's own settings
    haven't changed - it may now cover more of the image."""
    app = _fake_current_figure_app()
    other = m.ViewConfig(99)
    calls = []
    real_colorize = app.colorize_view
    def counting_colorize(*a, **k):
        calls.append(1)
        return real_colorize(*a, **k)
    app.colorize_view = counting_colorize
    app.render_view_thumbnail(other, m.THUMB_SIZE)
    app.rows_done = m.SIZE // 2   # simulate more rows having completed
    app.render_view_thumbnail(other, m.THUMB_SIZE)
    assert len(calls) == 2, "rows_done advancing must invalidate the cache"


def test_select_view_invalidates_the_incoming_views_stale_cache():
    """Regression test for the specific staleness bug this design has to
    avoid: a view that was cached while non-live, then selected (becomes
    live, gets edited), then deselected again must NOT fall back to its
    pre-edit cached thumbnail just because rows_done happens to match."""
    app = _fake_color_editor_app()
    app.render_view_thumbnail = m.App.render_view_thumbnail.__get__(app)
    app.colorize_view = m.App.colorize_view.__get__(app)
    app.obstacles = []
    app.stats = np.zeros((m.SIZE, m.SIZE, m.N_STATS), dtype=np.float32)
    app.points = np.zeros((m.SIZE, m.SIZE, m.N_POINTS, 2), dtype=np.float32)
    app.status = np.ones((m.SIZE, m.SIZE), dtype=np.uint8)
    app.max_iter = m.DEFAULT_ITER
    app.max_dist = m.effective_dist(0.0)
    app.gen_seg_mult = 1.0
    app.gen_shape = m.SHAPE_ELLIPSE
    app.gen_bound = m.ellipse_bound(2.0, 2.0, 0.0)
    app.gen_fuzz = m.fuzz_array(m.DEFAULT_FUZZ)
    app.gen_seed_base = 0
    app.gen_buddha_blur_radius = 0.0
    app.gen_buddha_blur_type = m.BUDDHA_BLUR_GAUSSIAN
    app.img = np.zeros((m.SIZE, m.SIZE, 3), dtype=np.uint8)

    # view 1 starts non-live; cache its (default RGB) thumbnail first
    view1 = app.views[1]
    thumb_before = app.render_view_thumbnail(view1, m.THUMB_SIZE)
    assert view1.id in app.view_thumb_cache

    # select it (becomes live), "edit" it, then select view 0 again (view 1
    # becomes non-live) - all through the real select_view, which must
    # invalidate view 1's stale entry when it becomes live. view1 is HSV
    # mode (the fixture's default for views[1]), so channel 2 is Value -
    # editing it is what actually changes the visible output; Hue alone
    # wouldn't, since Value stays 0 either way.
    app.select_view(1)
    view1.chan_src[2] = m.CHAN_SRC_CONSTANT
    view1.constant_choice[2] = m.NONE_ONE
    app.select_view(0)
    thumb_after = app.render_view_thumbnail(view1, m.THUMB_SIZE)
    assert not np.array_equal(thumb_before, thumb_after), \
        "must reflect the edit made while live, not the stale pre-edit cache"


def test_write_uses_self_frame_directly_for_a_single_view():
    """No need to recompute anything when only one view is being exported -
    self.frame already holds exactly that."""
    app = _fake_current_figure_app()
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    assert img.shape == (m.SIZE, m.SIZE, 3)
    assert (img == 42).all(), "single-view export should be exactly self.frame"


def test_write_composes_all_views_when_include_all_views_is_on():
    app = _fake_current_figure_app()
    app.include_all_views_var = _StubVar(True)
    app.new_view()
    app.views[1].mode = "HSV"
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    assert img.shape == (m.SIZE, m.SIZE * 2, 3)
    half = m.SIZE
    assert not np.array_equal(img[:, :half], img[:, half:]), \
        "two views with different modes should produce different halves"


def test_render_current_figure_matches_self_frame_for_identical_single_view():
    """A sanity check that the two code paths agree: composing just the live
    view through render_current_figure gives the same picture self.frame
    already has (modulo the outline, which render_current_figure also
    applies - both are off here for a clean, direct pixel comparison isn't
    meaningful since self.frame's content comes from a different source in
    this fixture; instead just check shape and that it runs without error)."""
    app = _fake_current_figure_app()
    img = app.render_current_figure([app.live_view()])
    assert img.shape == (m.SIZE, m.SIZE, 3)


# --------------------------------------------------------------------------- #
# simulation_info_lines / the "Simulation info location" margin
# --------------------------------------------------------------------------- #
_ALL_SHAPES_AND_PARAMS = [
    (m.SHAPE_ELLIPSE, (2.0, 1.5, 30.0)),
    (m.SHAPE_POLYGON, (6, 15.0)),
    (m.SHAPE_MOSS_EGG, (60.0, 1.0, 0.0)),
    (m.SHAPE_SECTOR, (1.0, 90.0, 45.0)),
    (m.SHAPE_SEGMENT, (1.0, 120.0, 0.0)),
    (m.SHAPE_LENS, (1.0, 1.2, 0.5, 10.0)),
    (m.SHAPE_REULEAUX, (5, 20.0)),
    (m.SHAPE_CAPSULE, (2.0, 0.5, 0.0)),
]


def test_simulation_info_lines_renders_for_every_shape_and_fuzz_state():
    """Every shape/fuzz combination must produce lines that are actually
    renderable - a missing glyph would KeyError mid-export."""
    for shape, params in _ALL_SHAPES_AND_PARAMS:
        for fuzz_on in (False, True):
            fuzz = m.DEFAULT_FUZZ._replace(enabled=fuzz_on)
            settings = m.Settings(max_iter=64, max_dist=0.0, seg_mult=0.75, shape=shape,
                                  params=params, fuzz=fuzz,
                                  buddha_blur_radius=m.DEFAULT_BUDDHA_BLUR_RADIUS,
                                  buddha_blur_type=m.DEFAULT_BUDDHA_BLUR_TYPE)
            for line in m.simulation_info_lines(settings):
                m.render_text_line(line, scale=1)   # must not raise


def test_simulation_info_lines_shows_limits_and_shape():
    settings = m.Settings(max_iter=42, max_dist=5.0, seg_mult=1.0, shape=m.SHAPE_ELLIPSE,
                          params=(2.0, 2.0, 0.0), fuzz=m.DEFAULT_FUZZ,
                          buddha_blur_radius=m.DEFAULT_BUDDHA_BLUR_RADIUS,
                          buddha_blur_type=m.DEFAULT_BUDDHA_BLUR_TYPE)
    lines = m.simulation_info_lines(settings)
    assert "Iter limit:42" in lines
    assert any("5.00" in l for l in lines)
    assert "Shape:" in lines
    assert any("Fuzz:Off" == l for l in lines)


def test_simulation_info_lines_shows_none_for_unlimited_distance():
    settings = m.Settings(max_iter=42, max_dist=0.0, seg_mult=1.0, shape=m.SHAPE_ELLIPSE,
                          params=(2.0, 2.0, 0.0), fuzz=m.DEFAULT_FUZZ,
                          buddha_blur_radius=m.DEFAULT_BUDDHA_BLUR_RADIUS,
                          buddha_blur_type=m.DEFAULT_BUDDHA_BLUR_TYPE)
    lines = m.simulation_info_lines(settings)
    assert "Dist limit:none" in lines


def test_simulation_info_lines_shows_fuzz_details_when_enabled():
    fuzz = m.DEFAULT_FUZZ._replace(enabled=True, emit_jitter=5.0)
    settings = m.Settings(max_iter=42, max_dist=0.0, seg_mult=1.0, shape=m.SHAPE_ELLIPSE,
                          params=(2.0, 2.0, 0.0), fuzz=fuzz,
                          buddha_blur_radius=m.DEFAULT_BUDDHA_BLUR_RADIUS,
                          buddha_blur_type=m.DEFAULT_BUDDHA_BLUR_TYPE)
    lines = m.simulation_info_lines(settings)
    assert "Fuzz:On" in lines
    assert any("Seed" in l for l in lines)


def test_simulation_info_lines_omits_segment_mult_when_default():
    settings_default = m.Settings(64, 0.0, 1.0, m.SHAPE_ELLIPSE, (2.0, 2.0, 0.0),
                                  m.DEFAULT_FUZZ, m.DEFAULT_BUDDHA_BLUR_RADIUS,
                                  m.DEFAULT_BUDDHA_BLUR_TYPE)
    settings_custom = m.Settings(64, 0.0, 0.5, m.SHAPE_ELLIPSE, (2.0, 2.0, 0.0),
                                 m.DEFAULT_FUZZ, m.DEFAULT_BUDDHA_BLUR_RADIUS,
                                 m.DEFAULT_BUDDHA_BLUR_TYPE)
    assert not any("Segment" in l for l in m.simulation_info_lines(settings_default))
    assert any("Segment" in l for l in m.simulation_info_lines(settings_custom))


def test_add_left_simulation_margin_does_not_mutate_input():
    img = np.full((40, 60, 3), 33, dtype=np.uint8)
    original = img.copy()
    settings = m.effective_settings(m.DEFAULT_SNAPSHOT)
    m.add_left_simulation_margin(img, settings)
    assert np.array_equal(img, original)


def test_add_left_simulation_margin_does_not_crash_for_any_shape():
    settings_list = [m.Settings(64, 0.0, 1.0, shape, params, m.DEFAULT_FUZZ,
                                m.DEFAULT_BUDDHA_BLUR_RADIUS, m.DEFAULT_BUDDHA_BLUR_TYPE)
                     for shape, params in _ALL_SHAPES_AND_PARAMS]
    img = np.full((30, 30, 3), 10, dtype=np.uint8)
    for settings in settings_list:
        m.add_left_simulation_margin(img, settings)   # must not raise


def test_font_5x7_covers_every_character_any_coloration_label_could_use():
    """Every string that coloration_info_lines could ever produce must be
    renderable - a missing glyph would KeyError mid-export."""
    strings = list(m.MODE_NAMES)
    for mode in m.MODE_NAMES:
        strings += [name for name, _color in m.MODES[mode]]
    strings += m.CHAN_SRC_LABELS
    strings += m.STAT_LABELS
    strings += m.POINT_SHORT
    strings += m.BUDDHA_POINT_LABELS
    strings += m.BUDDHA_COMPRESSION_LABELS
    strings += m.BUDDHA_BRIGHTNESS_OPTIONS
    strings += ["Mode:", "P1:", "P2:", "(none)", "0", "1"]
    missing = set()
    for s in strings:
        for ch in s:
            if ch.lower() not in m.FONT_5X7 and ch not in m.FONT_5X7:
                missing.add(ch)
    assert not missing, f"FONT_5X7 is missing glyphs for: {sorted(missing)}"


def test_font_5x7_new_glyphs_are_not_blank():
    """Every new glyph must actually light up some pixels (catches a
    copy-paste that left a row all zeros)."""
    for ch in "abcdfstuxyzABCEGHLMPQRSTV()+-/:" + "\u00b7\u00b0":
        mask = m.render_text_line(ch, scale=1)
        assert mask.shape[0] == 7
        assert mask.any(), f"glyph {ch!r} renders as entirely blank"


def test_coloration_info_lines_omits_metric_kind_and_shows_the_metric_name():
    v = m.ViewConfig(0)
    lines = m.coloration_info_lines(v)
    assert lines[0] == f"Mode:{v.mode}"
    assert not any("Point comparison" in l or "Other" in l for l in lines), \
        "the metric's kind must never be shown - it's implied by the metric's name"
    letter = m.MODES[v.mode][0][0][0]
    expected_metric = m.STAT_LABELS[v.point_metric[0]]
    assert f"{letter}:{expected_metric}" in lines


def test_coloration_info_lines_uses_long_names_not_short_ones():
    """The point of this turn's change: dropdown-length names, not the
    compact forms used elsewhere (e.g. the keyframe list)."""
    v = m.ViewConfig(0)
    lines = m.coloration_info_lines(v)
    joined = " ".join(lines)
    assert "H dist" not in joined and "Horizontal distance" in joined


def test_point_comparison_metric_names_no_longer_say_between_points():
    for label in m.STAT_LABELS[:4]:      # the four point-comparison metrics
        assert "between points" not in label
    assert all("between points" not in label for label in m.POINT_METRIC_LABELS)


def test_coloration_info_lines_shows_points_only_for_point_comparison_metrics():
    v = m.ViewConfig(0)
    v.kinds[2] = m.KIND_OTHER
    v.other_metric[2] = m.ST_BOUNCES
    lines = m.coloration_info_lines(v)
    letter2 = m.MODES[v.mode][2][0][0]
    idx = lines.index(f"{letter2}:{m.STAT_LABELS[m.ST_BOUNCES]}")
    # the very next line (if any - this may be the last line) must NOT be a
    # Point 1/2 line for this channel
    assert idx + 1 == len(lines) or not lines[idx + 1].strip().startswith("P1:")


def test_coloration_info_lines_buddha_channel_shows_points_compression_brightness():
    v = m.ViewConfig(0)
    v.chan_src[1] = m.CHAN_SRC_BUDDHA
    v.buddha_points[1] = tuple(i in (0, 3) for i in range(m.N_BUDDHA_POINTS))
    v.buddha_compression[1] = m.BUDDHA_COMPRESS_MODULUS
    v.buddha_brightness[1] = 3
    lines = m.coloration_info_lines(v)
    letter = m.MODES[v.mode][1][0][0]
    assert f"{letter}:Buddha" in lines
    assert f"  {m.BUDDHA_POINT_LABELS[0]}" in lines
    assert f"  {m.BUDDHA_POINT_LABELS[3]}" in lines
    settings_line = next(l for l in lines if "Modulus" in l)
    assert m.BUDDHA_BRIGHTNESS_OPTIONS[3] in settings_line
    assert "x" in settings_line and "xx" not in settings_line, \
        "brightness labels already include 'x' - must not be doubled"


def test_coloration_info_lines_works_for_resolved_view_settings_too():
    """New figure / video export pass resolved ViewSettings (read-only
    tuples), not live ViewConfig objects - the field names must line up."""
    vc = m.ViewConfig(0)
    vc.chan_src[1] = m.CHAN_SRC_BUDDHA
    vs = m.ViewSettings(vc.mode, vc.kinds, vc.point_metric, vc.other_metric,
                        vc.point1_choice, vc.point2_choice, vc.chan_src, vc.buddha_points,
                        vc.buddha_compression, vc.buddha_brightness, vc.constant_choice)
    assert m.coloration_info_lines(vc) == m.coloration_info_lines(vs)


def test_draw_coloration_info_uses_pure_black_background_and_not_pure_white_text():
    assert m.COLORATION_INFO_BG == (0, 0, 0)
    assert m.COLORATION_INFO_FG != (255, 255, 255)
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img, [m.ViewConfig(0), m.ViewConfig(1)], 100)
    assert (img == m.COLORATION_INFO_BG).all(axis=-1).any()
    assert (img == m.COLORATION_INFO_FG).all(axis=-1).any()
    assert not (img == (255, 255, 255)).all(axis=-1).any()


def test_draw_coloration_info_stays_within_each_views_own_portion():
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img, [m.ViewConfig(0), m.ViewConfig(1)], 100)
    left_has_info = (img[:, :100] == m.COLORATION_INFO_BG).all(axis=-1).any()
    right_has_info = (img[:, 100:] == m.COLORATION_INFO_BG).all(axis=-1).any()
    assert left_has_info and right_has_info


def test_draw_coloration_info_does_not_crash_on_a_tiny_image():
    img = np.full((15, 20, 3), 50, dtype=np.uint8)
    v = m.ViewConfig(0)
    v.chan_src[1] = m.CHAN_SRC_BUDDHA
    m.draw_coloration_info(img, [v], 20)   # must not raise


def _text_row_extent(img, background):
    """How many rows contain any non-background pixel - a rough proxy for
    the overlay's total rendered height."""
    return (img != background).any(axis=-1).any(axis=1).sum()


def test_coloration_info_text_scales_up_above_size_512():
    v = m.ViewConfig(0)
    img_1x = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img_1x, [v], m.SIZE)
    img_2x = np.full((m.SIZE * 2, m.SIZE * 2, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img_2x, [v], m.SIZE * 2)
    h1 = _text_row_extent(img_1x, 128)
    h2 = _text_row_extent(img_2x, 128)
    assert h2 == h1 * 2, \
        f"text should be exactly 2x taller at 2x the view resolution, got {h1} vs {h2}"


def test_coloration_info_text_never_shrinks_below_the_size_512_baseline():
    v = m.ViewConfig(0)
    img_baseline = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img_baseline, [v], m.SIZE)
    img_small = np.full((200, 200, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img_small, [v], 200)
    assert _text_row_extent(img_small, 128) == _text_row_extent(img_baseline, 128), \
        "text must stay at the baseline size for any resolution at or below SIZE"


def test_coloration_info_text_scale_uses_the_smaller_of_width_and_height():
    """An unusually elongated export (very wide, short view) shouldn't get
    oversized text just because one dimension is large."""
    v = m.ViewConfig(0)
    img_square = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img_square, [v], m.SIZE)
    img_wide_short = np.full((m.SIZE, m.SIZE * 4, 3), 128, dtype=np.uint8)   # short height
    m.draw_coloration_info(img_wide_short, [v], m.SIZE * 4)
    assert _text_row_extent(img_wide_short, 128) == _text_row_extent(img_square, 128), \
        "text size should follow the smaller dimension (height here), not the wide one"


def test_bottom_margin_text_scales_with_the_views_own_size_not_the_margin():
    """Regression test: the margin's own height is only a quarter of
    view_width by design, but the text size must track view_width itself -
    it was previously computed from the (always smaller) margin height,
    which meant the text barely grew until an absurdly large export
    (view_width beyond ~3072) rather than scaling normally."""
    v = m.ViewConfig(0)
    img_512 = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    out_512 = m.add_bottom_color_margin(img_512, [v], m.SIZE)
    h1 = _text_row_extent(out_512[m.SIZE:], m.COLORATION_INFO_BG)

    img_1080 = np.full((1080, 1080, 3), 128, dtype=np.uint8)
    out_1080 = m.add_bottom_color_margin(img_1080, [v], 1080)
    h2 = _text_row_extent(out_1080[1080:], m.COLORATION_INFO_BG)

    assert h2 == h1 * 2, f"expected exactly 2x text height at 1080 vs 512, got {h1} -> {h2}"


def test_left_margin_text_scales_with_the_views_own_size_not_the_margin():
    """Measures vertical extent (line height/spacing), not horizontal -
    horizontal extent can be legitimately clipped once text no longer fits
    the (narrow) margin width at a larger scale, which isn't itself a bug."""
    settings = m.effective_settings(m.DEFAULT_SNAPSHOT)
    img_512 = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    out_512 = m.add_left_simulation_margin(img_512, settings)
    margin_w_512 = m.SIZE // 2
    h1 = (out_512[:, :margin_w_512] != m.COLORATION_INFO_BG).any(axis=-1).any(axis=1).sum()

    img_1080 = np.full((1080, 1080, 3), 128, dtype=np.uint8)
    out_1080 = m.add_left_simulation_margin(img_1080, settings)
    margin_w_1080 = 1080 // 2
    h2 = (out_1080[:, :margin_w_1080] != m.COLORATION_INFO_BG).any(axis=-1).any(axis=1).sum()

    assert h2 == h1 * 2, f"expected exactly 2x text height at 1080 vs 512, got {h1} -> {h2}"


def test_write_draws_coloration_info_without_mutating_self_frame():
    """self.frame is the shared, reused live-preview buffer - drawing the
    overlay must go into a copy, never into self.frame itself."""
    app = _fake_current_figure_app()
    app.color_info_location_var = _StubVar("Overlay")
    frame_before = app.frame.copy()
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    assert np.array_equal(app.frame, frame_before), \
        "self.frame must never be mutated by an export"
    img = m.read_png(path)
    assert (img == m.COLORATION_INFO_BG).all(axis=-1).any(), \
        "the exported file should carry the coloration-info overlay"


def test_save_new_image_draws_coloration_info_when_enabled():
    app = _fake_export_app()
    app.color_info_location_var = _StubVar("Overlay")
    app.save_new_w_var = _StubVar("40")
    app.save_new_h_var = _StubVar("40")
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.save_btn = _StubWidget()
    app.save_as_btn = _StubWidget()
    app.save_new_image = m.App.save_new_image.__get__(app)
    app._save_new_image_worker = m.App._save_new_image_worker.__get__(app)
    app._save_new_image_done = m.App._save_new_image_done.__get__(app)
    app._save_new_image_failed = m.App._save_new_image_failed.__get__(app)
    app.root = type("R", (), {"after": lambda self, delay, fn: fn()})()

    tmpdir = tempfile.mkdtemp()
    orig_export_dir = m.EXPORT_DIR
    m.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img = m.read_png(os.path.join(tmpdir, files[0]))
        assert (img == m.COLORATION_INFO_BG).all(axis=-1).any()
    finally:
        m.EXPORT_DIR = orig_export_dir


def test_video_render_worker_draws_coloration_info_when_enabled():
    """Exercises the per-frame drawing step _render_worker takes, without
    needing a real ffmpeg encode: build the same rgb array it would, with
    the same call (apply_info_locations, as _render_worker actually calls
    it), and check the overlay landed."""
    app = _fake_export_app()
    kf = m.Keyframe(0, 0, dict(m.DEFAULT_SETTING_VALUES), "Linear", None)
    keyframes = [kf]
    obst_snapshot = [m.snapshot_obstacle(o) for o in app.obstacles]
    data = m.resolve_frame_settings(keyframes, 0)
    obst_arrays = m.resolve_obstacle_arrays(obst_snapshot, keyframes, 0)
    view_snapshot = [m.snapshot_view(v) for v in app.views_to_export()]
    views = m.resolve_views(view_snapshot, keyframes, 0)

    rgb = app.render_full_image(data, 40, 40, obst_arrays, views)
    rgb = m.apply_info_locations(rgb, views, 40, m.effective_settings(data.snapshot),
                                 "Overlay", "None", mutate_ok=True)
    assert (rgb == m.COLORATION_INFO_BG).all(axis=-1).any()


def test_video_render_worker_adds_simulation_left_margin_when_enabled():
    app = _fake_export_app()
    kf = m.Keyframe(0, 0, dict(m.DEFAULT_SETTING_VALUES), "Linear", None)
    keyframes = [kf]
    obst_snapshot = [m.snapshot_obstacle(o) for o in app.obstacles]
    data = m.resolve_frame_settings(keyframes, 0)
    obst_arrays = m.resolve_obstacle_arrays(obst_snapshot, keyframes, 0)
    view_snapshot = [m.snapshot_view(v) for v in app.views_to_export()]
    views = m.resolve_views(view_snapshot, keyframes, 0)

    rgb = app.render_full_image(data, 40, 40, obst_arrays, views)
    original_width = rgb.shape[1]
    rgb = m.apply_info_locations(rgb, views, 40, m.effective_settings(data.snapshot),
                                 "None", "Left margin", mutate_ok=True)
    assert rgb.shape[1] == original_width + 40 // 2
    assert (rgb == m.COLORATION_INFO_BG).all(axis=-1).any()


def test_color_and_sim_info_location_default_to_none():
    app = _fake_export_app()
    assert app.color_info_location_var.get() == "None"
    assert app.sim_info_location_var.get() == "None"


def test_apply_info_locations_none_leaves_the_image_unchanged():
    settings = m.effective_settings(m.DEFAULT_SNAPSHOT)
    v = m.ViewConfig(0)
    img = np.full((40, 40, 3), 77, dtype=np.uint8)
    out = m.apply_info_locations(img, [v], 40, settings, "None", "None", mutate_ok=True)
    assert out.shape == img.shape
    assert (out == 77).all()


def test_apply_info_locations_margin_cases_never_mutate_when_not_ok():
    """Bottom/left margins always return a new array anyway (they can't
    mutate in place - the image gets bigger), but this locks in that
    mutate_ok=False is honored for "Overlay" specifically, which otherwise
    would draw straight into the caller's array."""
    settings = m.effective_settings(m.DEFAULT_SNAPSHOT)
    v = m.ViewConfig(0)
    img = np.full((40, 40, 3), 77, dtype=np.uint8)
    original = img.copy()
    m.apply_info_locations(img, [v], 40, settings, "Overlay", "None", mutate_ok=False)
    assert np.array_equal(img, original), "mutate_ok=False must not draw into the input array"


def test_write_with_bottom_margin_produces_a_taller_image():
    app = _fake_current_figure_app()
    app.color_info_location_var = _StubVar("Bottom margin")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    assert img.shape == (m.SIZE + m.SIZE // 4, m.SIZE, 3)
    assert (img[m.SIZE:] == m.COLORATION_INFO_BG).all(axis=-1).mean() > 0.5
    assert (img[:m.SIZE] == 42).all(), "the original view content must be untouched"


def test_write_with_left_margin_produces_a_wider_image():
    app = _fake_current_figure_app()
    app.sim_info_location_var = _StubVar("Left margin")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    assert img.shape == (m.SIZE, m.SIZE + m.SIZE // 2, 3)
    assert (img[:, :m.SIZE // 2] == m.COLORATION_INFO_BG).all(axis=-1).any()
    assert (img[:, m.SIZE // 2:] == 42).all(), "the original view content must be untouched"


def test_write_with_both_margins_composes_correctly():
    app = _fake_current_figure_app()
    app.color_info_location_var = _StubVar("Bottom margin")
    app.sim_info_location_var = _StubVar("Left margin")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    expected_h = m.SIZE + m.SIZE // 4
    expected_w = m.SIZE + m.SIZE // 2
    assert img.shape == (expected_h, expected_w, 3)


# --------------------------------------------------------------------------- #
# FFV1 video export (lossless, MKV container)
# --------------------------------------------------------------------------- #
def _fake_ffmpeg_format_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.ffmpeg_output_kwargs = m.App.ffmpeg_output_kwargs.__get__(app)
    app.output_extension_and_suffix = m.App.output_extension_and_suffix.__get__(app)
    return app


def test_ffv1_uses_mkv_extension_with_ffv1_suffix():
    app = _fake_ffmpeg_format_app()
    ext, suffix = app.output_extension_and_suffix({"fmt": "mkv (ffv1)"})
    assert ext == "mkv"
    assert suffix == ".ffv1"
    filename = f"light_bounces-video-20260101-000000{suffix}.{ext}"
    assert filename.endswith(".ffv1.mkv")


def test_ffv1_kwargs_are_lossless_full_rgb():
    app = _fake_ffmpeg_format_app()
    kwargs = app.ffmpeg_output_kwargs("mkv (ffv1)", "Constant quality", 75, 0, 0, 0)
    assert kwargs["vcodec"] == "ffv1"
    assert kwargs["pix_fmt"] == "gbrp", \
        "FFV1 should use planar RGB (no chroma subsampling) to stay truly lossless"
    assert "crf" not in kwargs and "video_bitrate" not in kwargs, \
        "a lossless codec has no bitrate/quality knobs to set"


def test_other_formats_unaffected_by_adding_ffv1():
    app = _fake_ffmpeg_format_app()
    assert app.output_extension_and_suffix({"fmt": "mp4 (h264)"}) == ("mp4", "")
    assert app.output_extension_and_suffix({"fmt": "GIF"}) == ("gif", "")
    assert app.output_extension_and_suffix({"fmt": "APNG"}) == ("apng", "")
    mp4_kwargs = app.ffmpeg_output_kwargs("mp4 (h264)", "Constant quality", 75, 0, 0, 0)
    assert mp4_kwargs["vcodec"] == "libx264" and mp4_kwargs["pix_fmt"] == "yuv420p"
    assert app.ffmpeg_output_kwargs("GIF", "Constant quality", 75, 0, 0, 0) == {}
    assert app.ffmpeg_output_kwargs("APNG", "Constant quality", 75, 0, 0, 0) == {}


# --------------------------------------------------------------------------- #
# JXL video: ffmpeg can't produce an animated JXL file directly (its libjxl
# wrapper is still-image only), so this renders to an intermediate APNG and
# hands that to the external `cjxl` tool, which does support conversion
# --------------------------------------------------------------------------- #
FAKE_CJXL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dev", "testing")
REAL_CJXL_PATH = shutil.which("cjxl")   # captured before any test alters PATH


def _path_entries_without_cjxl():
    """The current PATH's directories, minus any that contain a cjxl
    executable, so a test's view of PATH can't be affected by whether the
    machine running it happens to have the real tool installed. (Everything
    else on PATH - ffmpeg, the system directory cmd.exe needs - is kept.)"""
    keep = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and any(os.path.exists(os.path.join(entry, name))
                         for name in ("cjxl", "cjxl.exe", "cjxl.cmd", "cjxl.bat")):
            continue
        keep.append(entry)
    return keep


class _PathContext:
    """For the duration of a `with` block, replaces os.environ["PATH"] with
    `dir_path` (if given) followed by every original PATH directory that
    doesn't contain a cjxl executable - so `dir_path` decides on its own
    whether any `cjxl` is findable: None means none at all, and
    FAKE_CJXL_DIR means the fake from dev/testing/. `fail_on` sets
    FAKE_CJXL_FAIL_ON for the fake (see dev/testing/cjxl.cmd). Both are
    restored afterward."""
    def __init__(self, dir_path=None, fail_on=None):
        self.dir_path = dir_path
        self.fail_on = fail_on

    def __enter__(self):
        self.orig_path = os.environ.get("PATH", "")
        self.orig_fail_on = os.environ.get("FAKE_CJXL_FAIL_ON")
        entries = _path_entries_without_cjxl()
        if self.dir_path is not None:
            entries.insert(0, self.dir_path)
        os.environ["PATH"] = os.pathsep.join(entries)
        if self.fail_on is None:
            os.environ.pop("FAKE_CJXL_FAIL_ON", None)
        else:
            os.environ["FAKE_CJXL_FAIL_ON"] = self.fail_on
        return self

    def __exit__(self, *exc):
        os.environ["PATH"] = self.orig_path
        if self.orig_fail_on is None:
            os.environ.pop("FAKE_CJXL_FAIL_ON", None)
        else:
            os.environ["FAKE_CJXL_FAIL_ON"] = self.orig_fail_on


def _fake_cjxl(fail_on=None):
    """A _PathContext in which `cjxl` is the fake from dev/testing/: it
    copies its input file to its output path and exits 0, unless the input
    path contains `fail_on`, in which case it prints to stderr and exits 1 -
    lets tests exercise both convert_apng_to_jxl's success and failure paths
    without depending on a real libjxl installation."""
    if os.name != "nt":   # the POSIX fake needs its execute bit
        os.chmod(os.path.join(FAKE_CJXL_DIR, "cjxl"), 0o755)
    return _PathContext(FAKE_CJXL_DIR, fail_on)


def _no_cjxl():
    """A _PathContext in which no `cjxl` is findable at all."""
    return _PathContext(None)


def test_output_extension_and_suffix_for_jxl():
    app = _fake_ffmpeg_format_app()
    assert app.output_extension_and_suffix({"fmt": "JXL"}) == ("jxl", "")


def test_ffmpeg_output_kwargs_for_jxl_matches_apng():
    app = _fake_ffmpeg_format_app()
    assert app.ffmpeg_output_kwargs("JXL", "Constant quality", 75, 0, 0, 0) == {}


def test_cjxl_available_false_without_a_real_binary():
    with _no_cjxl():
        assert m.cjxl_available() is False


def test_cjxl_available_true_with_a_fake_binary_on_path():
    with _fake_cjxl():
        assert m.cjxl_available() is True


def test_convert_apng_to_jxl_success_produces_the_output_file():
    work_dir = tempfile.mkdtemp()
    apng_path = os.path.join(work_dir, "source.apng")
    jxl_path = os.path.join(work_dir, "out.jxl")
    with open(apng_path, "wb") as f:
        f.write(b"fake apng bytes")
    with _fake_cjxl():
        m.convert_apng_to_jxl(apng_path, jxl_path)
    assert os.path.exists(jxl_path)
    with open(jxl_path, "rb") as f:
        assert f.read() == b"fake apng bytes"


def test_convert_apng_to_jxl_failure_raises_with_stderr_and_makes_no_file():
    work_dir = tempfile.mkdtemp()
    apng_path = os.path.join(work_dir, "shouldfail.apng")
    jxl_path = os.path.join(work_dir, "out.jxl")
    with open(apng_path, "wb") as f:
        f.write(b"fake apng bytes")
    with _fake_cjxl("shouldfail"):
        try:
            m.convert_apng_to_jxl(apng_path, jxl_path)
            assert False, "should have raised"
        except RuntimeError as e:
            assert "simulated encoder failure" in str(e)
    assert not os.path.exists(jxl_path)


def test_convert_apng_to_jxl_raises_clearly_when_cjxl_is_missing():
    """convert_apng_to_jxl resolves cjxl via shutil.which (so .cmd shims work
    on Windows); with nothing on PATH that must be a clear RuntimeError, not
    a bare FileNotFoundError from subprocess."""
    work_dir = tempfile.mkdtemp()
    apng_path = os.path.join(work_dir, "source.apng")
    with open(apng_path, "wb") as f:
        f.write(b"unused")
    with _no_cjxl():
        try:
            m.convert_apng_to_jxl(apng_path, os.path.join(work_dir, "out.jxl"))
            assert False, "should have raised"
        except RuntimeError as e:
            assert "not found" in str(e)


def test_convert_apng_to_jxl_with_the_real_cjxl_produces_a_jxl_file():
    """The one test here that uses the real tools instead of the fake: builds
    a genuine 2-frame APNG (this program's own PNG writer, then the real
    ffmpeg) and converts it with the real cjxl. Catches what the fake can't -
    cjxl rejecting our APNG, or its command-line interface changing. Skipped
    when cjxl or ffmpeg isn't installed."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if REAL_CJXL_PATH is None:
        raise SkipTest("real cjxl not on PATH")
    if ffmpeg_exe is None:
        raise SkipTest("real ffmpeg not on PATH")
    import subprocess
    work_dir = tempfile.mkdtemp()
    for i, value in enumerate((40, 200)):
        frame = np.full((16, 16, 3), value, dtype=np.uint8)
        m.write_png(os.path.join(work_dir, f"frame_{i:05d}.png"), frame)
    apng_path = os.path.join(work_dir, "source.apng")
    jxl_path = os.path.join(work_dir, "out.jxl")
    subprocess.run([ffmpeg_exe, "-y", "-loglevel", "error", "-framerate", "10",
                    "-i", os.path.join(work_dir, "frame_%05d.png"),
                    "-plays", "0", apng_path], check=True)
    m.convert_apng_to_jxl(apng_path, jxl_path)
    with open(jxl_path, "rb") as f:
        head = f.read(12)
    bare_codestream = head[:2] == bytes.fromhex("ff0a")
    iso_container = head == bytes.fromhex("0000000c4a584c200d0a870a")
    assert bare_codestream or iso_container, f"not a JXL file: {head!r}"


def test_read_render_settings_blocks_jxl_without_cjxl():
    class FakeApp:
        pass
    app = FakeApp()
    app.fps_var = _StubVar("30")
    app.video_w_var = _StubVar("40")
    app.video_h_var = _StubVar("40")
    app.export_format_var = _StubVar("JXL")
    app.bitrate_mode_var = _StubVar("Constant quality")
    app.quality_scale = _StubVar(75)
    app.downsample_factor_var = _StubVar(list(m.DOWNSAMPLE_FACTOR_VALUES)[0])
    app.downsample_method_var = _StubVar("Mean")
    app.read_render_settings = m.App.read_render_settings.__get__(app)
    with _no_cjxl():
        rs = app.read_render_settings()
    assert rs is None


def test_read_render_settings_allows_jxl_with_cjxl_available():
    class FakeApp:
        pass
    app = FakeApp()
    app.fps_var = _StubVar("30")
    app.video_w_var = _StubVar("40")
    app.video_h_var = _StubVar("40")
    app.export_format_var = _StubVar("JXL")
    app.bitrate_mode_var = _StubVar("Constant quality")
    app.quality_scale = _StubVar(75)
    app.downsample_factor_var = _StubVar(list(m.DOWNSAMPLE_FACTOR_VALUES)[0])
    app.downsample_method_var = _StubVar("Mean")
    app.read_render_settings = m.App.read_render_settings.__get__(app)
    orig_ffmpeg = m.ffmpeg
    m.ffmpeg = orig_ffmpeg or object()   # only this sandbox lacks ffmpeg-python
    try:
        with _fake_cjxl():
            rs = app.read_render_settings()
    finally:
        m.ffmpeg = orig_ffmpeg
    assert rs is not None
    assert rs["fmt"] == "JXL"


class _FakeFfmpegStream:
    def __init__(self, path=None):
        self.path = path


class _FakeFfmpegModule:
    """Stands in for the real ffmpeg-python module (unavailable in this
    sandbox): input()/output() just track the target path through a
    trivial stream object, and run() simulates a successful encode by
    actually writing bytes to that path, so the rest of encode_video_output
    (the cjxl handoff, temp-file cleanup) can be exercised for real."""
    class Error(Exception):
        pass

    def input(self, *a, **k):
        return _FakeFfmpegStream()

    def output(self, stream, path, **kwargs):
        return _FakeFfmpegStream(path)

    def filter(self, stream, *a, **k):
        return stream

    def run(self, stream, **kwargs):
        with open(stream.path, "wb") as f:
            f.write(b"fake ffmpeg output bytes")


def _fake_encode_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.apply_downsample_filter = m.App.apply_downsample_filter.__get__(app)
    app.ffmpeg_output_kwargs = m.App.ffmpeg_output_kwargs.__get__(app)
    app.encode_video_output = m.App.encode_video_output.__get__(app)
    app.status_calls = []
    app.set_video_status = lambda text: app.status_calls.append(text)
    app.root = type("R", (), {"after": lambda self, delay, fn: fn()})()
    return app


def _base_rs(fmt):
    return {"fps": 30, "fmt": fmt, "mode": "Constant quality", "quality": 75,
           "bitrate": 0, "vbr_target": 0, "vbr_max": 0,
           "downsample_factor": 1, "downsample_method": "Mean"}


def test_encode_video_output_writes_directly_for_non_jxl_formats():
    app = _fake_encode_app()
    work_dir = tempfile.mkdtemp()
    out_path = os.path.join(work_dir, "out.mp4")
    orig_ffmpeg = m.ffmpeg
    m.ffmpeg = _FakeFfmpegModule()
    try:
        app.encode_video_output("pattern_%05d.png", 0, _base_rs("mp4 (h264)"), out_path)
    finally:
        m.ffmpeg = orig_ffmpeg
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
        assert f.read() == b"fake ffmpeg output bytes"


def test_encode_video_output_routes_jxl_through_apng_and_cjxl():
    app = _fake_encode_app()
    work_dir = tempfile.mkdtemp()
    out_path = os.path.join(work_dir, "out.jxl")
    orig_ffmpeg = m.ffmpeg
    orig_temp_dir = m.TEMP_DIR
    m.ffmpeg = _FakeFfmpegModule()
    m.TEMP_DIR = tempfile.mkdtemp()
    try:
        with _fake_cjxl():
            app.encode_video_output("pattern_%05d.png", 0, _base_rs("JXL"), out_path)
    finally:
        m.ffmpeg = orig_ffmpeg
        m.TEMP_DIR = orig_temp_dir
    assert os.path.exists(out_path), "the final .jxl file must exist"
    with open(out_path, "rb") as f:
        assert f.read() == b"fake ffmpeg output bytes", \
            "the fake cjxl just copies its input, so this confirms the chain ran: " \
            "ffmpeg wrote the APNG, cjxl copied it to the final path"
    temp_apng = os.path.join(m.TEMP_DIR, "_jxl_video_source.apng")
    assert not os.path.exists(temp_apng), "the intermediate APNG must be cleaned up"
    assert any("cjxl" in s for s in app.status_calls), \
        "the status should mention the cjxl conversion step"


def test_encode_video_output_cleans_up_temp_apng_even_on_cjxl_failure():
    app = _fake_encode_app()
    work_dir = tempfile.mkdtemp()
    out_path = os.path.join(work_dir, "out.jxl")
    orig_ffmpeg = m.ffmpeg
    orig_temp_dir = m.TEMP_DIR
    m.ffmpeg = _FakeFfmpegModule()
    m.TEMP_DIR = tempfile.mkdtemp()
    try:
        with _fake_cjxl("_jxl_video_source"):
            try:
                app.encode_video_output("pattern_%05d.png", 0, _base_rs("JXL"), out_path)
                assert False, "should have raised"
            except RuntimeError:
                pass
    finally:
        m.ffmpeg = orig_ffmpeg
        m.TEMP_DIR = orig_temp_dir
    temp_apng = os.path.join(m.TEMP_DIR, "_jxl_video_source.apng")
    assert not os.path.exists(temp_apng), \
        "the intermediate APNG must still be cleaned up even when cjxl fails"
    assert not os.path.exists(out_path)


# --------------------------------------------------------------------------- #
# clear_temp_frames: deletes only frame_NNNNN.png files from TEMP_DIR - the
# one place in the whole program that deletes anything at all
# --------------------------------------------------------------------------- #
class _ClearTempFramesContext:
    """Patches m.TEMP_DIR to a fresh scratch directory and m.messagebox's
    dialog functions to controllable, call-counting stubs, for the duration
    of a `with` block - and restores both afterward, since the test module
    loads light_bounces once and shares it across every test."""
    def __init__(self, confirm=True):
        self.confirm = confirm
        self.showinfo_calls = []
        self.showerror_calls = []
        self.askyesno_calls = []

    def __enter__(self):
        self.orig_temp_dir = m.TEMP_DIR
        self.orig_showinfo = m.messagebox.showinfo
        self.orig_showerror = m.messagebox.showerror
        self.orig_askyesno = m.messagebox.askyesno
        self.tmpdir = tempfile.mkdtemp()
        m.TEMP_DIR = self.tmpdir
        m.messagebox.showinfo = lambda *a, **k: self.showinfo_calls.append((a, k))
        m.messagebox.showerror = lambda *a, **k: self.showerror_calls.append((a, k))
        m.messagebox.askyesno = lambda *a, **k: (self.askyesno_calls.append((a, k))
                                                 or self.confirm)
        return self

    def __exit__(self, *exc):
        m.TEMP_DIR = self.orig_temp_dir
        m.messagebox.showinfo = self.orig_showinfo
        m.messagebox.showerror = self.orig_showerror
        m.messagebox.askyesno = self.orig_askyesno


def _fake_app_for_clear_temp_frames(busy=False):
    class FakeApp:
        pass
    app = FakeApp()
    app.busy_with_render_or_reencode = lambda: busy
    app.video_status_calls = []
    app.set_video_status = lambda text: app.video_status_calls.append(text)
    app.clear_temp_frames = m.App.clear_temp_frames.__get__(app)
    return app


def test_clear_temp_frames_only_deletes_the_strict_frame_pattern():
    with _ClearTempFramesContext(confirm=True) as ctx:
        names = ["frame_00000.png", "frame_00001.png", "frame_abc.png",
                "other_frame_00000.png", "notes.txt", "frame_00002.PNG"]
        for name in names:
            open(os.path.join(ctx.tmpdir, name), "w").close()
        app = _fake_app_for_clear_temp_frames()
        app.clear_temp_frames()
        remaining = set(os.listdir(ctx.tmpdir))
        assert remaining == {"frame_abc.png", "other_frame_00000.png", "notes.txt",
                             "frame_00002.PNG"}, \
            f"only the two strict frame_NNNNN.png files should be gone, got {remaining}"
        assert len(ctx.askyesno_calls) == 1


def test_clear_temp_frames_deletes_nothing_without_confirmation():
    with _ClearTempFramesContext(confirm=False) as ctx:
        open(os.path.join(ctx.tmpdir, "frame_00000.png"), "w").close()
        app = _fake_app_for_clear_temp_frames()
        app.clear_temp_frames()
        assert os.listdir(ctx.tmpdir) == ["frame_00000.png"], \
            "declining the confirmation dialog must delete nothing"


def test_clear_temp_frames_refuses_while_busy():
    with _ClearTempFramesContext(confirm=True) as ctx:
        open(os.path.join(ctx.tmpdir, "frame_00000.png"), "w").close()
        app = _fake_app_for_clear_temp_frames(busy=True)
        app.clear_temp_frames()
        assert os.listdir(ctx.tmpdir) == ["frame_00000.png"], \
            "must not delete anything while a render or re-encode is running"
        assert len(ctx.askyesno_calls) == 0, "must not even ask while busy"
        assert len(ctx.showerror_calls) == 1


def test_clear_temp_frames_handles_missing_temp_dir():
    with _ClearTempFramesContext(confirm=True) as ctx:
        shutil.rmtree(ctx.tmpdir)   # TEMP_DIR itself doesn't exist
        app = _fake_app_for_clear_temp_frames()
        app.clear_temp_frames()   # must not raise
        assert len(ctx.showinfo_calls) == 1
        assert len(ctx.askyesno_calls) == 0


def test_clear_temp_frames_handles_no_matching_files():
    with _ClearTempFramesContext(confirm=True) as ctx:
        open(os.path.join(ctx.tmpdir, "notes.txt"), "w").close()
        app = _fake_app_for_clear_temp_frames()
        app.clear_temp_frames()
        assert os.listdir(ctx.tmpdir) == ["notes.txt"]
        assert len(ctx.showinfo_calls) == 1
        assert len(ctx.askyesno_calls) == 0


def test_clear_temp_frames_reports_the_count_deleted():
    with _ClearTempFramesContext(confirm=True) as ctx:
        for i in range(3):
            open(os.path.join(ctx.tmpdir, f"frame_{i:05d}.png"), "w").close()
        app = _fake_app_for_clear_temp_frames()
        app.clear_temp_frames()
        assert os.listdir(ctx.tmpdir) == []
        assert any("3" in msg for msg in app.video_status_calls)


# --------------------------------------------------------------------------- #
# Fixed (non-ray-dependent) point choices: Origin and Top left corner
# --------------------------------------------------------------------------- #
def test_extract_point_xy_returns_constants_for_the_fixed_points():
    n = 5
    x, y = m.extract_point_xy(None, m.POINT_ORIGIN, None, n)
    assert len(x) == n and (x == 0.0).all() and (y == 0.0).all()
    x, y = m.extract_point_xy(None, m.POINT_TOP_LEFT_CORNER, None, n)
    assert (x == -1.0).all() and (y == 1.0).all()


def test_extract_point_xy_still_reads_real_points_from_the_array():
    points_arr = np.zeros((3, 3, m.N_POINTS, 2))
    points_arr[..., m.POINT_START, :] = [0.5, 0.25]
    computed = np.ones((3, 3), dtype=bool)
    x, y = m.extract_point_xy(points_arr, m.POINT_START, computed, 9)
    assert x[0] == 0.5 and y[0] == 0.25


def test_resolve_point_xy_handles_both_constant_and_real_points():
    assert m.resolve_point_xy(np.zeros((m.N_POINTS, 2)), m.POINT_ORIGIN) == (0.0, 0.0)
    assert m.resolve_point_xy(np.zeros((m.N_POINTS, 2)), m.POINT_TOP_LEFT_CORNER) == (-1.0, 1.0)
    sp = np.zeros((m.N_POINTS, 2))
    sp[m.POINT_END] = [0.3, -0.4]
    assert m.resolve_point_xy(sp, m.POINT_END) == (0.3, -0.4)


def test_point_lists_include_the_two_new_fixed_points():
    assert m.POINT_LABELS[m.POINT_ORIGIN] == "Origin"
    assert m.POINT_LABELS[m.POINT_TOP_LEFT_CORNER] == "Top left corner"
    assert len(m.POINT_LABELS) == len(m.POINT_SHORT) == len(m.POINT_DESCRIPTIONS)
    assert len(m.POINT_LABELS) == m.N_POINT_CHOICES == m.N_POINTS + 2


def test_colorize_view_produces_uniform_color_for_two_constant_points():
    """The real, end-to-end proof: a channel comparing two fixed points must
    give the SAME value everywhere, regardless of where any ray went -
    since the metric no longer depends on the ray at all."""
    app = _fake_render_app()
    stats = np.zeros((4, 4, m.N_STATS), dtype=np.float32)
    points = np.zeros((4, 4, m.N_POINTS, 2), dtype=np.float32)
    status = np.ones((4, 4), dtype=np.uint8)
    v = m.ViewSettings(m.DEFAULT_MODE, list(m.DEFAULT_KINDS), list(m.DEFAULT_POINT_METRICS),
                       list(m.DEFAULT_OTHER_METRICS), list(m.DEFAULT_POINT1S),
                       list(m.DEFAULT_POINT2S), list(m.DEFAULT_CHAN_SRCS),
                       list(m.DEFAULT_BUDDHA_POINTS_PER_CHAN),
                       list(m.DEFAULT_BUDDHA_COMPRESSIONS), list(m.DEFAULT_BUDDHA_BRIGHTNESSES),
                       list(m.DEFAULT_CONSTANT_CHOICES))
    v.point1_choice[0] = m.POINT_ORIGIN
    v.point2_choice[0] = m.POINT_TOP_LEFT_CORNER
    v.point_metric[0] = m.ST_HDIST
    mini_settings = m.Settings(30, m.effective_dist(0.0), 1.0, m.SHAPE_ELLIPSE, (),
                               m.DEFAULT_FUZZ, m.DEFAULT_BUDDHA_BLUR_RADIUS,
                               m.DEFAULT_BUDDHA_BLUR_TYPE)
    bound = m.ellipse_bound(2.0, 2.0, 0.0)
    fz = m.fuzz_array(m.DEFAULT_FUZZ)
    obst_arrays = m.build_obstacle_arrays([])
    img = app.colorize_view(v, stats, points, status, 4, 4, mini_settings, bound, fz,
                            obst_arrays)
    assert (img[..., 0] == img[0, 0, 0]).all()
    assert 126 <= img[0, 0, 0] <= 129   # H dist(origin, top-left) = 0.5 -> ~127.5


def test_coloration_info_lines_shows_the_new_point_names():
    v = m.ViewConfig(0)
    v.point1_choice[0] = m.POINT_ORIGIN
    v.point2_choice[0] = m.POINT_TOP_LEFT_CORNER
    lines = m.coloration_info_lines(v)
    assert any("Origin" in l and "TopLeft" in l for l in lines)


# --------------------------------------------------------------------------- #
# "Constant" channel source - a dedicated ViewConfig field (constant_choice),
# never packed into a slot whose meaning depends on context
# --------------------------------------------------------------------------- #
def test_other_labels_no_longer_offer_the_constant_options():
    assert "None (always 0)" not in m.OTHER_LABELS
    assert "None (always 1)" not in m.OTHER_LABELS
    assert len(m.OTHER_LABELS) == len(m.OTHER_STATS)


def test_chan_src_labels_include_constant():
    assert m.CHAN_SRC_LABELS[m.CHAN_SRC_CONSTANT] == "Constant"
    assert len(m.CHAN_SRC_LABELS) == 3


def test_constant_label_stat_round_trip():
    assert m.constant_label_to_stat("Always 0") == m.NONE_ZERO
    assert m.constant_label_to_stat("Always 1") == m.NONE_ONE
    assert m.constant_stat_to_label(m.NONE_ZERO) == "Always 0"
    assert m.constant_stat_to_label(m.NONE_ONE) == "Always 1"


def test_view_config_has_a_dedicated_constant_choice_field():
    v = m.ViewConfig(0)
    assert v.constant_choice == list(m.DEFAULT_CONSTANT_CHOICES)
    # setting it must never touch other_metric - they are fully independent
    v.constant_choice[0] = m.NONE_ONE
    assert v.other_metric[0] == m.DEFAULT_OTHER_METRICS[0]


def test_make_mapping_raises_if_constant_channel_has_no_constant_choices():
    """The specific safety property requested: a mistake here must crash,
    not silently misread some other slot."""
    try:
        m.make_mapping([m.KIND_OTHER] * 3, [0] * 3, [0] * 3,
                       chan_src=[m.CHAN_SRC_CONSTANT] * 3)
        assert False, "should have raised"
    except ValueError:
        pass


def test_make_mapping_resolves_constant_channels_from_constant_choices():
    kinds = [m.KIND_OTHER] * 3
    point_metrics = [m.ST_HDIST] * 3
    other_metrics = [m.ST_BOUNCES, m.ST_BOUNCES, m.ST_BOUNCES]   # never consulted
    chan_src = [m.CHAN_SRC_CONSTANT, m.CHAN_SRC_CONSTANT, m.CHAN_SRC_QUERY]
    constant_choices = [m.NONE_ZERO, m.NONE_ONE, m.NONE_ZERO]    # [2] unused (Query)
    mapping = m.make_mapping(kinds, point_metrics, other_metrics, chan_src, constant_choices)
    assert mapping == (m.NONE_ZERO, m.NONE_ONE, m.ST_BOUNCES)


def test_view_channel_value_and_decompose_chans_round_trip_constant_choice():
    v = m.ViewConfig(0)
    v.chan_src[1] = m.CHAN_SRC_CONSTANT
    v.constant_choice[1] = m.NONE_ONE
    chans = [m.view_channel_value(v, c) for c in range(3)]
    assert len(chans[1]) == 9
    vs = m.decompose_chans(v.mode, chans)
    assert vs.constant_choice[1] == m.NONE_ONE
    assert vs.constant_choice[0] == m.DEFAULT_CONSTANT_CHOICES[0]   # untouched channel


def test_colorize_view_renders_constant_channels_correctly():
    """End-to-end: Always-0 and Always-1 channels render as pure 0/255
    everywhere, regardless of the actual simulation data."""
    app = _fake_render_app()
    stats = np.random.RandomState(0).rand(4, 4, m.N_STATS).astype(np.float32)
    points = np.zeros((4, 4, m.N_POINTS, 2), dtype=np.float32)
    status = np.ones((4, 4), dtype=np.uint8)
    v = m.ViewSettings(m.DEFAULT_MODE, list(m.DEFAULT_KINDS), list(m.DEFAULT_POINT_METRICS),
                       list(m.DEFAULT_OTHER_METRICS), list(m.DEFAULT_POINT1S),
                       list(m.DEFAULT_POINT2S), list(m.DEFAULT_CHAN_SRCS),
                       list(m.DEFAULT_BUDDHA_POINTS_PER_CHAN),
                       list(m.DEFAULT_BUDDHA_COMPRESSIONS), list(m.DEFAULT_BUDDHA_BRIGHTNESSES),
                       list(m.DEFAULT_CONSTANT_CHOICES))
    v.chan_src[0] = m.CHAN_SRC_CONSTANT
    v.constant_choice[0] = m.NONE_ONE
    v.chan_src[1] = m.CHAN_SRC_CONSTANT
    v.constant_choice[1] = m.NONE_ZERO
    mini_settings = m.Settings(30, m.effective_dist(0.0), 1.0, m.SHAPE_ELLIPSE, (),
                               m.DEFAULT_FUZZ, m.DEFAULT_BUDDHA_BLUR_RADIUS,
                               m.DEFAULT_BUDDHA_BLUR_TYPE)
    bound = m.ellipse_bound(2.0, 2.0, 0.0)
    fz = m.fuzz_array(m.DEFAULT_FUZZ)
    obst_arrays = m.build_obstacle_arrays([])
    img = app.colorize_view(v, stats, points, status, 4, 4, mini_settings, bound, fz,
                            obst_arrays)
    assert (img[..., 0] == 255).all()
    assert (img[..., 1] == 0).all()


def test_coloration_info_lines_describes_a_constant_channel():
    v = m.ViewConfig(0)
    v.chan_src[2] = m.CHAN_SRC_CONSTANT
    v.constant_choice[2] = m.NONE_ONE
    lines = m.coloration_info_lines(v)
    assert "B:Constant" in lines
    assert "  Always 1" in lines
    for line in lines:
        m.render_text_line(line, scale=1)   # every line must actually render


def test_coloration_info_lines_raises_if_stat_is_a_constant_sentinel_outside_constant_source():
    """The other half of the safety property: if other_metric somehow ends
    up holding a Constant-only sentinel while chan_src is Query, that must
    crash loudly rather than render as if it were a real metric."""
    v = m.ViewConfig(0)
    v.kinds[0] = m.KIND_OTHER
    v.other_metric[0] = m.NONE_ZERO      # chan_src[0] is still CHAN_SRC_QUERY
    try:
        m.coloration_info_lines(v)
        assert False, "should have raised"
    except ValueError:
        pass


def test_short_value_text_describes_a_constant_channel():
    v = m.ViewConfig(0)
    v.chan_src[0] = m.CHAN_SRC_CONSTANT
    v.constant_choice[0] = m.NONE_ZERO
    val = m.view_channel_value(v, 0)
    text = m.short_value_text("view:0:chan0", val)
    assert text == "Const:0"


def test_short_value_text_raises_if_stat_is_a_constant_sentinel_outside_constant_source():
    v = m.ViewConfig(0)
    v.kinds[0] = m.KIND_OTHER
    v.other_metric[0] = m.NONE_ONE       # chan_src[0] is still CHAN_SRC_QUERY
    val = m.view_channel_value(v, 0)
    try:
        m.short_value_text("view:0:chan0", val)
        assert False, "should have raised"
    except ValueError:
        pass


def test_on_constant_changed_updates_the_dedicated_field_only():
    app = _fake_color_editor_app()
    view = app.selected_view_obj()
    view.chan_src[0] = m.CHAN_SRC_CONSTANT
    original_other_metric = view.other_metric[0]
    app.constant_vars[0].set("Always 1")
    app.on_constant_changed = m.App.on_constant_changed.__get__(app)
    app.on_constant_changed(0)
    assert view.constant_choice[0] == m.NONE_ONE
    assert view.other_metric[0] == original_other_metric, \
        "on_constant_changed must never touch other_metric"


def test_refresh_row_shows_constant_frame_only_for_constant_source():
    app = _fake_color_editor_app()
    view = app.selected_view_obj()
    view.chan_src[0] = m.CHAN_SRC_CONSTANT
    app.refresh_row(0)
    assert app.constant_frames[0].shown
    assert not app.query_frames[0].shown
    assert not app.buddha_frames[0].shown
    view.chan_src[0] = m.CHAN_SRC_QUERY
    app.refresh_row(0)
    assert not app.constant_frames[0].shown
    assert app.query_frames[0].shown


# --------------------------------------------------------------------------- #
# Resolution controls round to the nearest multiple of 4 (ffmpeg needs even
# dimensions to encode most video formats)
# --------------------------------------------------------------------------- #
def test_round_to_multiple_of_4_table():
    cases = [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 8), (7, 8), (8, 8),
            (1080, 1080), (1081, 1080), (1082, 1080), (1083, 1084)]
    for n, expected in cases:
        assert m.round_to_multiple_of_4(n) == expected, f"n={n}"


def test_round_to_multiple_of_4_never_returns_zero_or_negative():
    for n in range(1, 20):
        assert m.round_to_multiple_of_4(n) >= 4


def test_parse_resolution_dimension_rounds_a_valid_value():
    assert m.parse_resolution_dimension("1081") == 1080
    assert m.parse_resolution_dimension("1") == 4


def test_parse_resolution_dimension_rejects_invalid_text():
    assert m.parse_resolution_dimension("abc") is None
    assert m.parse_resolution_dimension("0") is None
    assert m.parse_resolution_dimension("") is None


def test_commit_resolution_var_rewrites_displayed_text_to_rounded_value():
    class FakeApp:
        pass
    app = FakeApp()
    app.commit_resolution_var = m.App.commit_resolution_var.__get__(app)
    var = _StubVar("1081")
    app.commit_resolution_var(var)
    assert var.get() == "1080"


def test_commit_resolution_var_leaves_invalid_text_alone():
    class FakeApp:
        pass
    app = FakeApp()
    app.commit_resolution_var = m.App.commit_resolution_var.__get__(app)
    var = _StubVar("not a number")
    app.commit_resolution_var(var)
    assert var.get() == "not a number"


def test_save_new_image_rounds_a_non_multiple_of_4_resolution():
    app = _fake_export_app()
    app.save_new_w_var = _StubVar("21")   # not a multiple of 4 -> rounds to 20
    app.save_new_h_var = _StubVar("22")   # not a multiple of 4 -> rounds to 24
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.save_btn = _StubWidget()
    app.save_as_btn = _StubWidget()
    app.save_new_image = m.App.save_new_image.__get__(app)
    app._save_new_image_worker = m.App._save_new_image_worker.__get__(app)
    app._save_new_image_done = m.App._save_new_image_done.__get__(app)
    app._save_new_image_failed = m.App._save_new_image_failed.__get__(app)
    app.root = type("R", (), {"after": lambda self, delay, fn: fn()})()

    tmpdir = tempfile.mkdtemp()
    orig_export_dir = m.EXPORT_DIR
    m.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img = m.read_png(os.path.join(tmpdir, files[0]))
        assert img.shape == (24, 20, 3), \
            f"21x22 should round to 20x24, got {img.shape}"
    finally:
        m.EXPORT_DIR = orig_export_dir


def test_read_render_settings_rounds_a_non_multiple_of_4_resolution():
    class FakeApp:
        pass
    app = FakeApp()
    app.fps_var = _StubVar("30")
    app.video_w_var = _StubVar("1081")
    app.video_h_var = _StubVar("1082")
    app.export_format_var = _StubVar("mp4 (h264)")
    app.bitrate_mode_var = _StubVar("Constant quality")
    app.quality_scale = _StubVar(75)
    app.downsample_factor_var = _StubVar(list(m.DOWNSAMPLE_FACTOR_VALUES)[0])
    app.downsample_method_var = _StubVar("Mean")
    app.read_render_settings = m.App.read_render_settings.__get__(app)
    orig_ffmpeg = m.ffmpeg
    m.ffmpeg = orig_ffmpeg or object()   # only this sandbox lacks ffmpeg-python; the
                                        # function under test doesn't actually call it
    try:
        rs = app.read_render_settings()
    finally:
        m.ffmpeg = orig_ffmpeg
    assert rs is not None
    assert rs["video_w"] == 1080
    assert rs["video_h"] == 1080


# --------------------------------------------------------------------------- #
# Buddha blur: spreads a scattered point's weight across a neighborhood of
# pixels rather than a single one
# --------------------------------------------------------------------------- #
def test_world_radius_to_pixels_matches_the_spec_example():
    # 0.1 -> 0.2 world units; the "2x2 or 2x3" wording means the SMALLEST
    # dimension governs, and it's 2.0 either way here.
    assert abs(m.world_radius_to_pixels(0.1, 512, 512) - 51.2) < 1e-9
    assert abs(m.world_radius_to_pixels(0.1, 512, 768) - 51.2) < 1e-9


def test_zero_radius_kernel_is_a_single_pixel():
    k = m.build_buddha_blur_kernel(0.0, m.BUDDHA_BLUR_GAUSSIAN)
    assert k.shape == (1, 3)
    assert tuple(k[0]) == (0.0, 0.0, 1.0)


def test_every_blur_kernel_normalizes_to_weight_one():
    for bt in range(len(m.BUDDHA_BLUR_LABELS)):
        k = m.build_buddha_blur_kernel(5.0, bt)
        assert abs(k[:, 2].sum() - 1.0) < 1e-9


def test_gaussian_kernel_peaks_at_center_and_extends_past_radius():
    k = m.build_buddha_blur_kernel(5.0, m.BUDDHA_BLUR_GAUSSIAN)
    weights = {(row[0], row[1]): row[2] for row in k}
    assert weights[(0.0, 0.0)] == k[:, 2].max()
    assert weights.get((5.0, 0.0), 0.0) > 0.0, \
        "gaussian must still have weight at the nominal radius, unlike the others"


def test_triangular_kernel_is_exactly_zero_at_radius():
    k = m.build_buddha_blur_kernel(5.0, m.BUDDHA_BLUR_TRIANGULAR)
    weights = {(row[0], row[1]) for row in k}
    assert (5.0, 0.0) not in weights


def test_bokeh_kernel_is_uniform_inside_the_disc():
    k = m.build_buddha_blur_kernel(5.0, m.BUDDHA_BLUR_BOKEH)
    assert len(np.unique(np.round(k[:, 2], 10))) == 1


def test_catadioptric_kernel_has_a_hole_at_the_center():
    k = m.build_buddha_blur_kernel(5.0, m.BUDDHA_BLUR_CATADIOPTRIC)
    weights = {(row[0], row[1]): row[2] for row in k}
    assert weights.get((0.0, 0.0), 0.0) == 0.0
    assert weights.get((1.0, 0.0), 0.0) == 0.0
    assert weights.get((4.0, 0.0), 0.0) > 0.0


def test_raw_scatter_then_blur_conserves_total_weight_and_spreads():
    counts = np.zeros((10, 10), dtype=np.float64)
    rows = np.array([5, 5])
    cols = np.array([5, 5])
    valid = np.array([True, True])
    m.scatter_points_raw(counts, rows, cols, valid)
    kernel = m.build_buddha_blur_kernel(2.0, m.BUDDHA_BLUR_BOKEH)
    blurred = m.blur_counts_image(counts, kernel)
    assert abs(blurred.sum() - 2.0) < 1e-6
    assert (blurred > 0).sum() > 1


def test_raw_scatter_then_blur_with_no_blur_matches_original_behavior():
    counts = np.zeros((10, 10), dtype=np.float64)
    rows = np.array([5, 5])
    cols = np.array([5, 5])
    valid = np.array([True, True])
    m.scatter_points_raw(counts, rows, cols, valid)
    kernel = m.build_buddha_blur_kernel(0.0, m.BUDDHA_BLUR_GAUSSIAN)
    blurred = m.blur_counts_image(counts, kernel)
    assert blurred[5, 5] == 2.0
    assert (blurred > 0).sum() == 1


def test_blur_counts_image_matches_the_old_per_point_reference_for_every_kernel():
    """The core correctness guarantee behind the performance rewrite: raw
    scatter + one blur_counts_image pass must give EXACTLY the same result
    (to floating-point precision) as blurring each point individually and
    summing - since convolution distributes over sums."""
    rows = np.array([20, 35, 10])
    cols = np.array([20, 35, 50])
    for bt in range(len(m.BUDDHA_BLUR_LABELS)):
        kernel = m.build_buddha_blur_kernel(6.0, bt)
        ref = np.zeros((60, 60), dtype=np.float64)
        for r, c in zip(rows, cols):
            for dr, dc, wt in kernel:
                rr, cc = r + int(dr), c + int(dc)
                if 0 <= rr < 60 and 0 <= cc < 60:
                    ref[rr, cc] += wt
        raw = np.zeros((60, 60), dtype=np.float64)
        m.scatter_points_raw(raw, rows, cols, np.ones(3, dtype=bool))
        new = m.blur_counts_image(raw, kernel)
        assert np.abs(ref - new).max() < 1e-6, \
            f"{m.BUDDHA_BLUR_LABELS[bt]} deferred blur doesn't match the per-point reference"


def test_blur_counts_image_no_blur_is_a_true_no_op():
    counts = np.random.RandomState(0).rand(20, 20)
    kernel = m.build_buddha_blur_kernel(0.0, m.BUDDHA_BLUR_BOKEH)
    result = m.blur_counts_image(counts, kernel)
    assert np.array_equal(result, counts)


def test_snapshot_and_settings_carry_buddha_blur_fields():
    assert m.DEFAULT_SNAPSHOT.buddha_blur_radius == m.DEFAULT_BUDDHA_BLUR_RADIUS
    assert m.DEFAULT_SNAPSHOT.buddha_blur_type == m.DEFAULT_BUDDHA_BLUR_TYPE
    settings = m.effective_settings(m.DEFAULT_SNAPSHOT)
    assert settings.buddha_blur_radius == m.DEFAULT_BUDDHA_BLUR_RADIUS
    assert settings.buddha_blur_type == m.DEFAULT_BUDDHA_BLUR_TYPE


def test_capture_and_resolve_round_trip_buddha_blur_settings():
    app = _make_fake_app_for_snapshot()
    app.buddha_blur_radius_scale = _StubVar(0.2)
    app.buddha_blur_type_var = _StubVar(m.BUDDHA_BLUR_LABELS[m.BUDDHA_BLUR_BOKEH])
    app.deco = {"outline": _StubVar(True)}
    app.obstacles = []
    app.views = []
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    data = app.capture_all_settings()
    assert data["buddha_blur_radius"] == 0.2
    assert data["buddha_blur_type"] == m.BUDDHA_BLUR_BOKEH

    kf = m.Keyframe(0, 0, data, "Linear", None)
    resolved = m.resolve_frame_settings([kf], 0)
    assert resolved.snapshot.buddha_blur_radius == 0.2
    assert resolved.snapshot.buddha_blur_type == m.BUDDHA_BLUR_BOKEH


def test_colorize_view_spreads_a_buddha_point_with_blur_enabled():
    """End-to-end: a single ray's Buddha point, with blur on, must leave a
    nonzero count on more of its (fired) neighboring pixels than with blur
    off. Two things this test design has to account for, both found by
    running it and checking the actual numbers rather than guessing:
    colorize_view only colors pixels that are themselves "computed" (fired
    their own ray), so the neighborhood around the scattered point must
    also be marked as fired for the spread to be visible at all - and every
    computed pixel gets assigned a value (even exactly 0), overwriting the
    background, so "differs from background" is the wrong signal; "nonzero"
    is the right one."""
    app = _fake_render_app()
    size = 20
    stats = np.zeros((size, size, m.N_STATS), dtype=np.float32)
    points = np.full((size, size, m.N_POINTS, 2), 100.0, dtype=np.float32)   # far outside
                                                                             # the [-1,1] grid
    status = np.zeros((size, size), dtype=np.uint8)
    status[8:13, 8:13] = 1                          # a small fired neighborhood
    points[10, 10, m.POINT_ALL_MEAN] = [0.0, 0.0]   # only this one actually lands in-frame

    v = m.ViewSettings(m.DEFAULT_MODE, list(m.DEFAULT_KINDS), list(m.DEFAULT_POINT_METRICS),
                       list(m.DEFAULT_OTHER_METRICS), list(m.DEFAULT_POINT1S),
                       list(m.DEFAULT_POINT2S), list(m.DEFAULT_CHAN_SRCS),
                       list(m.DEFAULT_BUDDHA_POINTS_PER_CHAN),
                       list(m.DEFAULT_BUDDHA_COMPRESSIONS), list(m.DEFAULT_BUDDHA_BRIGHTNESSES),
                       list(m.DEFAULT_CONSTANT_CHOICES))
    v.chan_src[0] = m.CHAN_SRC_BUDDHA
    # index 0 in BUDDHA_POINT_CATALOG is "All bounces mean" -> POINT_ALL_MEAN
    v.buddha_points[0] = tuple(i == 0 for i in range(m.N_BUDDHA_POINTS))

    bound = m.ellipse_bound(2.0, 2.0, 0.0)
    fz = m.fuzz_array(m.DEFAULT_FUZZ)
    obst_arrays = m.build_obstacle_arrays([])

    settings_no_blur = m.Settings(30, m.effective_dist(0.0), 1.0, m.SHAPE_ELLIPSE, (),
                                  m.DEFAULT_FUZZ, 0.0, m.BUDDHA_BLUR_BOKEH)
    img_no_blur = app.colorize_view(v, stats, points, status, size, size, settings_no_blur,
                                    bound, fz, obst_arrays)
    lit_no_blur = (img_no_blur[8:13, 8:13, 0] > 0).sum()

    settings_blur = m.Settings(30, m.effective_dist(0.0), 1.0, m.SHAPE_ELLIPSE, (),
                               m.DEFAULT_FUZZ, 0.1, m.BUDDHA_BLUR_BOKEH)
    img_blur = app.colorize_view(v, stats, points, status, size, size, settings_blur,
                                 bound, fz, obst_arrays)
    lit_blur = (img_blur[8:13, 8:13, 0] > 0).sum()

    assert lit_no_blur == 1, f"with no blur, exactly one pixel should be nonzero, got {lit_no_blur}"
    assert lit_blur > lit_no_blur, \
        f"blur should leave more pixels nonzero ({lit_blur}) than no blur ({lit_no_blur})"


def test_blurred_buddha_counts_is_cached_until_invalidated():
    """The mechanism that keeps the live preview cheap: blurring is a whole-
    image pass, so it must not be recomputed on every recolor - only when
    the raw counts actually changed."""
    app = _fake_current_figure_app()
    app.live_view = lambda: app.views[0]
    app.views[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.buddha_counts = [np.zeros((m.SIZE, m.SIZE), dtype=np.float64), None, None]
    app.buddha_counts_blurred = [None, None, None]
    app.gen_buddha_blur_radius = 0.05
    app.gen_buddha_blur_type = m.BUDDHA_BLUR_BOKEH
    app.blurred_buddha_counts = m.App.blurred_buddha_counts.__get__(app)

    first = app.blurred_buddha_counts(0)
    second = app.blurred_buddha_counts(0)
    assert first is second, "must return the cached array, not recompute it"

    app.buddha_counts_blurred[0] = None      # simulate scatter_buddha_rows invalidating it
    third = app.blurred_buddha_counts(0)
    assert third is not first, "must recompute once actually invalidated"


def test_scatter_buddha_rows_invalidates_only_touched_channels():
    app = _fake_current_figure_app()
    app.live_view = lambda: app.views[0]
    view = app.views[0]
    view.chan_src[0] = m.CHAN_SRC_BUDDHA
    view.buddha_points[0] = tuple(i == 0 for i in range(m.N_BUDDHA_POINTS))
    view.chan_src[1] = m.CHAN_SRC_BUDDHA
    view.buddha_points[1] = tuple(False for _ in range(m.N_BUDDHA_POINTS))   # nothing selected
    app.buddha_counts = [np.zeros((m.SIZE, m.SIZE), dtype=np.float64),
                         np.zeros((m.SIZE, m.SIZE), dtype=np.float64), None]
    app.buddha_counts_blurred = ["sentinel0", "sentinel1", None]
    app.status = np.ones((m.SIZE, m.SIZE), dtype=np.uint8)
    app.points = np.zeros((m.SIZE, m.SIZE, m.N_POINTS, 2), dtype=np.float32)
    app.stats = np.zeros((m.SIZE, m.SIZE, m.N_STATS), dtype=np.float32)
    app.scatter_buddha_rows = m.App.scatter_buddha_rows.__get__(app)
    app.scatter_buddha_rows(0, m.SIZE)
    assert app.buddha_counts_blurred[0] is None, "touched channel must be invalidated"
    assert app.buddha_counts_blurred[1] == "sentinel1", \
        "untouched channel (nothing selected) must not be invalidated"


def test_large_blur_radius_completes_quickly():
    """Regression guard for the performance bug itself: applying a large
    blur must not scale with the number of scattered points. Before the
    fix, this configuration took upwards of ten seconds; it should now
    complete in a small fraction of a second."""
    import time
    app = _fake_render_app()
    size = 128
    stats = np.zeros((size, size, m.N_STATS), dtype=np.float32)
    points = np.random.RandomState(0).rand(size, size, m.N_POINTS, 2).astype(np.float32) * 2 - 1
    status = np.ones((size, size), dtype=np.uint8)   # every pixel fires
    v = m.ViewSettings(m.DEFAULT_MODE, list(m.DEFAULT_KINDS), list(m.DEFAULT_POINT_METRICS),
                       list(m.DEFAULT_OTHER_METRICS), list(m.DEFAULT_POINT1S),
                       list(m.DEFAULT_POINT2S), list(m.DEFAULT_CHAN_SRCS),
                       list(m.DEFAULT_BUDDHA_POINTS_PER_CHAN),
                       list(m.DEFAULT_BUDDHA_COMPRESSIONS), list(m.DEFAULT_BUDDHA_BRIGHTNESSES),
                       list(m.DEFAULT_CONSTANT_CHOICES))
    v.chan_src[0] = m.CHAN_SRC_BUDDHA
    v.buddha_points[0] = tuple(i == 0 for i in range(m.N_BUDDHA_POINTS))
    bound = m.ellipse_bound(2.0, 2.0, 0.0)
    fz = m.fuzz_array(m.DEFAULT_FUZZ)
    obst_arrays = m.build_obstacle_arrays([])
    settings = m.Settings(30, m.effective_dist(0.0), 1.0, m.SHAPE_ELLIPSE, (), m.DEFAULT_FUZZ,
                          0.15, m.BUDDHA_BLUR_GAUSSIAN)   # a large, expensive-looking radius
    t0 = time.time()
    app.colorize_view(v, stats, points, status, size, size, settings, bound, fz, obst_arrays)
    elapsed = time.time() - t0
    assert elapsed < 3.0, f"large-radius blur took {elapsed:.2f}s - performance regression?"


def test_save_new_image_with_bottom_margin_produces_a_taller_file():
    app = _fake_export_app()
    app.color_info_location_var = _StubVar("Bottom margin")
    app.save_new_w_var = _StubVar("40")
    app.save_new_h_var = _StubVar("40")
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.save_btn = _StubWidget()
    app.save_as_btn = _StubWidget()
    app.save_new_image = m.App.save_new_image.__get__(app)
    app._save_new_image_worker = m.App._save_new_image_worker.__get__(app)
    app._save_new_image_done = m.App._save_new_image_done.__get__(app)
    app._save_new_image_failed = m.App._save_new_image_failed.__get__(app)
    app.root = type("R", (), {"after": lambda self, delay, fn: fn()})()

    tmpdir = tempfile.mkdtemp()
    orig_export_dir = m.EXPORT_DIR
    m.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img = m.read_png(os.path.join(tmpdir, files[0]))
        assert img.shape == (40 + 40 // 4, 40, 3)
    finally:
        m.EXPORT_DIR = orig_export_dir


# --------------------------------------------------------------------------- #
# Deleting an obstacle or view removes its "obst:<id>:*" / "view:<id>:*"
# settings from every keyframe immediately (they used to be left behind as
# orphans: invisible in the Timeline's settings list, so impossible to remove)
# --------------------------------------------------------------------------- #
def _kf(id_, frame_index, data):
    return m.Keyframe(id_, frame_index, dict(data))


def test_remove_keyframe_keys_removes_only_that_ids_keys_from_every_keyframe():
    kfs = [_kf(0, 0, {"iter": 5, "obst:3:r": 0.5, "obst:3:theta": 10.0, "obst:30:r": 0.9}),
           _kf(1, 10, {"obst:3:r": 0.7, "view:3:mode": "RGB"}),
           _kf(2, 20, {"iter": 9})]
    removed = m.remove_keyframe_keys(kfs, "obst:3:")
    assert removed == 3
    assert kfs[0].data == {"iter": 5, "obst:30:r": 0.9}, \
        "obst:30:* must survive removal of obst:3:* (ids that merely start alike)"
    assert kfs[1].data == {"view:3:mode": "RGB"}, "view:3 is not obstacle 3"
    assert kfs[2].data == {"iter": 9}


def test_remove_keyframe_keys_with_no_matches_changes_nothing():
    kfs = [_kf(0, 0, {"iter": 5})]
    assert m.remove_keyframe_keys(kfs, "obst:1:") == 0
    assert kfs[0].data == {"iter": 5}


def _fake_delete_obstacle_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.obstacles = []
    app._obst_counter = 0
    app.selected_obstacle = None
    app.keyframes = []
    app.delete_obstacle = m.App.delete_obstacle.__get__(app)
    app.purge_keyframe_settings = m.App.purge_keyframe_settings.__get__(app)
    app.rebuild_obstacle_list = lambda: None
    app.on_shape_param_changed = lambda: None
    app.diff_refreshes = []
    app.refresh_timeline_diffs = lambda: app.diff_refreshes.append(1)
    return app


def test_delete_obstacle_purges_its_settings_from_every_keyframe():
    app = _fake_delete_obstacle_app()
    keep, doomed = m.Obstacle(0), m.Obstacle(1)
    app.obstacles = [keep, doomed]
    app.selected_obstacle = doomed
    app.keyframes = [_kf(0, 0, {"iter": 5, "obst:0:r": 0.1, "obst:1:r": 0.5,
                                "view:1:mode": "RGB"}),
                     _kf(1, 10, {"obst:1:theta": 45.0, "obst:1:effect": 1})]
    app.delete_obstacle()
    assert app.obstacles == [keep]
    assert app.keyframes[0].data == {"iter": 5, "obst:0:r": 0.1, "view:1:mode": "RGB"}
    assert app.keyframes[1].data == {}
    assert app.diff_refreshes, "the Timeline's settings list must be refreshed"


def test_delete_obstacle_with_nothing_selected_changes_no_keyframe():
    app = _fake_delete_obstacle_app()
    o = m.Obstacle(0)
    app.obstacles = [o]
    app.keyframes = [_kf(0, 0, {"obst:0:r": 0.1})]
    app.delete_obstacle()
    assert app.obstacles == [o]
    assert app.keyframes[0].data == {"obst:0:r": 0.1}


def test_delete_view_purges_its_settings_from_every_keyframe():
    app = _fake_view_list_app()
    app.new_view()                       # view id 1, selected
    app.keyframes = [_kf(0, 0, {"iter": 5, "view:0:mode": "RGB", "view:1:mode": "HSV",
                                "view:1:chan0": ("x",), "obst:1:r": 0.2}),
                     _kf(1, 10, {"view:1:chan2": ("y",)})]
    app.delete_view()
    assert app.all_view_ids() == [0]
    assert app.keyframes[0].data == {"iter": 5, "view:0:mode": "RGB", "obst:1:r": 0.2}
    assert app.keyframes[1].data == {}


def test_refused_delete_of_the_only_view_purges_nothing():
    app = _fake_view_list_app()
    app.keyframes = [_kf(0, 0, {"view:0:mode": "RGB"})]
    app.delete_view()
    assert app.all_view_ids() == [0]
    assert app.keyframes[0].data == {"view:0:mode": "RGB"}


def test_render_keyframe_snapshot_is_unaffected_by_later_edits_to_the_keyframes():
    """Regression guard for the immediate cleanup above: editing (or purging)
    keyframe settings while a render is running must not change that
    render, so the render works from independent copies."""
    live = [_kf(0, 0, {"obst:1:r": 0.5}), _kf(1, 10, {"obst:1:r": 0.9})]
    snap = m.snapshot_keyframes(live)
    m.remove_keyframe_keys(live, "obst:1:")
    live[0].data["iter"] = 99
    assert [(k.frame_index, k.data) for k in snap] == [(0, {"obst:1:r": 0.5}),
                                                     (10, {"obst:1:r": 0.9})]
    assert snap[0].interp == live[0].interp and snap[0].id == live[0].id


# --------------------------------------------------------------------------- #
# Obstacle incoming / outgoing ray effects (replacing the single "effect")
# --------------------------------------------------------------------------- #
def test_new_obstacle_defaults_to_reflecting_in_and_passing_out():
    o = m.Obstacle(0)
    assert o.in_effect == m.OBSTACLE_EFFECT_REFLECT
    assert o.out_effect == m.OBSTACLE_EFFECT_PASS
    assert not hasattr(o, "effect")


def test_effect_names_and_colors_cover_every_effect():
    assert m.OBSTACLE_EFFECT_NAMES == ["Reflect", "Terminate", "Pass through"]
    assert m.OBST_EFFECT_RGB == {m.OBSTACLE_EFFECT_REFLECT: (179, 229, 252),
                                 m.OBSTACLE_EFFECT_TERMINATE: (121, 85, 72),
                                 m.OBSTACLE_EFFECT_PASS: (0, 0, 0)}


def test_obstacle_field_info_has_both_effects_and_no_old_effect_field():
    fields = [f for f, _label, _is_float in m.OBST_FIELD_INFO]
    assert "in_effect" in fields and "out_effect" in fields
    assert "effect" not in fields


def test_build_obstacle_arrays_effects_are_n_by_2_and_keep_pass_through_obstacles():
    a, b, c = m.Obstacle(0), m.Obstacle(1), m.Obstacle(2)
    a.in_effect, a.out_effect = m.OBSTACLE_EFFECT_TERMINATE, m.OBSTACLE_EFFECT_REFLECT
    b.in_effect, b.out_effect = m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_PASS
    c.in_effect, c.out_effect = m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_TERMINATE
    shapes, bounds, cx, cy, effects, fire = m.build_obstacle_arrays([a, b, c])
    assert len(shapes) == len(bounds) == len(cx) == len(cy) == len(fire) == 3
    assert effects.shape == (3, 2)
    assert effects.tolist() == [[m.OBSTACLE_EFFECT_TERMINATE, m.OBSTACLE_EFFECT_REFLECT],
                                [m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_PASS],
                                [m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_TERMINATE]]


def test_build_obstacle_arrays_with_no_obstacles_gives_an_empty_n_by_2_effects_array():
    effects = m.build_obstacle_arrays([])[4]
    assert effects.shape == (0, 2)


def test_resolve_obstacle_arrays_animates_nothing_but_holds_each_effect_per_keyframe():
    o = m.Obstacle(3)
    snap = m.snapshot_obstacle(o)
    kf0 = m.Keyframe(0, 0, {"obst:3:in_effect": m.OBSTACLE_EFFECT_TERMINATE,
                            "obst:3:out_effect": m.OBSTACLE_EFFECT_REFLECT})
    kf1 = m.Keyframe(1, 10, {"obst:3:in_effect": m.OBSTACLE_EFFECT_PASS,
                             "obst:3:out_effect": m.OBSTACLE_EFFECT_TERMINATE})
    kfs = [kf0, kf1]
    for frame, expect in [(0, [1, 0]), (5, [1, 0]), (10, [2, 1])]:
        effects = m.resolve_obstacle_arrays([snap], kfs, frame)[4]
        assert effects.tolist() == [expect], (frame, effects.tolist())


def test_short_value_text_names_both_effects():
    for field in ("in_effect", "out_effect"):
        assert m.short_value_text(f"obst:1:{field}", m.OBSTACLE_EFFECT_PASS) == "Pass through"


def _grid_around(bound_extent=1.3, n=400):
    xs = np.linspace(-bound_extent, bound_extent, n)
    return np.meshgrid(xs, xs[::-1])


def test_ring_layers_for_every_shape_kind_are_disjoint_outer_outside_inner_inside():
    gx, gy = _grid_around()
    cases = [(m.SHAPE_ELLIPSE, m.ellipse_bound(1.6, 1.0, 20.0)),
             (m.SHAPE_POLYGON, m.polygon_bound(5, 10.0, 0.9)),
             (m.SHAPE_CAPSULE, m.capsule_bound(1.0, 0.4, 30.0)),
             (m.SHAPE_LENS, m.lens_bound(1.0, 1.0, 1.0, 0.0))]
    for shape, bound in cases:
        outer, inner = m.ring_layer_masks(shape, bound, gx, gy, layer_px=8.0)
        inside = m.inside_boundary_grid(gx, gy, shape, bound)
        assert outer.any() and inner.any(), shape
        assert not (outer & inner).any(), shape
        assert not (outer & inside).any(), f"outer layer must lie outside (shape {shape})"
        assert not (inner & ~inside).any(), f"inner layer must lie inside (shape {shape})"


def test_ring_layers_have_the_requested_thickness_for_a_circle():
    gx, gy = _grid_around(1.2, 1200)
    step = float(gx[0, 1] - gx[0, 0])
    layer_px = 10.0
    outer, inner = m.ring_layer_masks(m.SHAPE_ELLIPSE, m.ellipse_bound(2.0, 2.0, 0.0),
                                      gx, gy, layer_px=layer_px)
    r = np.hypot(gx, gy)
    width = layer_px * m.PIXEL
    assert r[outer].min() >= 1.0 - 2 * step and r[outer].max() <= 1.0 + width + 2 * step
    assert r[inner].max() <= 1.0 + 2 * step and r[inner].min() >= 1.0 - width - 2 * step


def test_draw_obstacle_outline_paints_outer_layer_incoming_and_inner_layer_outgoing():
    gx, gy = _grid_around(1.2, 500)
    bound = m.ellipse_bound(1.0, 1.0, 0.0)
    img = np.full(gx.shape + (3,), 77, dtype=np.uint8)
    m.draw_obstacle_outline(img, m.SHAPE_ELLIPSE, bound, gx, gy,
                            m.OBSTACLE_EFFECT_TERMINATE, m.OBSTACLE_EFFECT_PASS)
    r = np.hypot(gx, gy)
    outer_px = (img == np.array(m.OBST_TERMINATE_RGB)).all(axis=-1)
    inner_px = (img == np.array(m.OBST_PASS_RGB)).all(axis=-1)
    assert outer_px.any() and inner_px.any()
    assert (r[outer_px] > 0.5 - 0.01).all(), "incoming color must be the OUTER layer"
    assert (r[inner_px] < 0.5 + 0.01).all(), "outgoing color must be the INNER layer"


def test_obstacle_thumbnail_shows_both_effect_colors():
    o = m.Obstacle(0)
    o.in_effect, o.out_effect = m.OBSTACLE_EFFECT_TERMINATE, m.OBSTACLE_EFFECT_REFLECT
    thumb = m.render_obstacle_thumbnail(o, m.THUMB_SIZE)
    assert thumb.shape == (m.THUMB_SIZE, m.THUMB_SIZE, 3)
    assert (thumb == np.array(m.OBST_TERMINATE_RGB)).all(axis=-1).any()
    assert (thumb == np.array(m.OBST_REFLECT_RGB)).all(axis=-1).any()


def test_a_pass_through_both_ways_obstacle_counts_as_pending_even_if_unchanged():
    class FakeApp:
        pass
    app = FakeApp()
    o = m.Obstacle(0)
    o.in_effect = o.out_effect = m.OBSTACLE_EFFECT_PASS
    app.obstacle_state = m.App.obstacle_state.__get__(app)
    app.obstacle_is_pending = m.App.obstacle_is_pending.__get__(app)
    app.gen_obstacles_by_id = {0: app.obstacle_state(o)}     # unchanged since generation
    assert app.obstacle_is_pending(o)
    o.out_effect = m.OBSTACLE_EFFECT_REFLECT
    app.gen_obstacles_by_id = {0: app.obstacle_state(o)}
    assert not app.obstacle_is_pending(o)


# --------------------------------------------------------------------------- #
# "Fire rays inside": a ray starts at a point iff the point is inside ANY
# shape (the main bounding shape, or an obstacle) that has it switched on
# --------------------------------------------------------------------------- #
def _emit_fixture(main_fire=True, obstacles=()):
    """(emits(x, y) -> bool) for a unit-circle main shape plus `obstacles`, a
    list of (cx, cy, radius, fire) circles."""
    shape, bound = m.SHAPE_ELLIPSE, m.ellipse_bound(2.0, 2.0, 0.0)
    obs = []
    for i, (cx, cy, radius, fire) in enumerate(obstacles):
        o = m.Obstacle(i)
        o.width = o.height = 2.0 * radius
        o.r, o.theta = 0.0, 0.0
        o.cx, o.cy = cx, cy
        o.fire_inside = fire
        obs.append(o)
    shapes, bounds, ocx, ocy, _eff, ofire = m.build_obstacle_arrays(obs)
    ctl = m.main_ctl_array(main_fire, m.OBSTACLE_EFFECT_REFLECT, m.OBSTACLE_EFFECT_REFLECT)
    return lambda x, y: bool(m.emits_ray(x, y, shape, bound, ctl, shapes, bounds, ocx, ocy,
                                         ofire))


def test_main_shape_fires_rays_inside_by_default_and_not_outside():
    emits = _emit_fixture()
    assert emits(0.0, 0.0) and emits(0.5, -0.5)
    assert not emits(0.9, 0.9)          # outside the unit circle


def test_main_shape_with_fire_off_fires_nothing_on_its_own():
    emits = _emit_fixture(main_fire=False)
    assert not emits(0.0, 0.0)


def test_obstacle_with_fire_on_fires_inside_it_even_if_main_is_off():
    emits = _emit_fixture(main_fire=False, obstacles=[(0.4, 0.0, 0.2, True)])
    assert emits(0.4, 0.0)
    assert not emits(-0.4, 0.0)


def test_obstacle_with_fire_off_no_longer_punches_a_hole_in_the_main_shape():
    """Being inside a shape that has fire off makes no difference: a point
    inside both the main shape (fire on) and a fire-off obstacle still fires."""
    emits = _emit_fixture(obstacles=[(0.4, 0.0, 0.2, False)])
    assert emits(0.4, 0.0)


def test_obstacle_with_fire_off_alone_fires_nothing():
    emits = _emit_fixture(main_fire=False, obstacles=[(0.4, 0.0, 0.2, False)])
    assert not emits(0.4, 0.0)


def test_obstacle_with_fire_on_can_fire_outside_the_main_shape():
    emits = _emit_fixture(obstacles=[(1.2, 0.0, 0.2, True)])
    assert emits(1.2, 0.0)
    assert not emits(0.0, 1.2)


# --------------------------------------------------------------------------- #
# The main bounding shape's own ray behavior is an ordinary keyframeable setting
# --------------------------------------------------------------------------- #
def test_main_ray_behavior_defaults_preserve_the_old_behavior():
    s = m.DEFAULT_SNAPSHOT
    assert s.main_fire is True
    assert s.main_in_effect == m.OBSTACLE_EFFECT_REFLECT
    assert s.main_out_effect == m.OBSTACLE_EFFECT_REFLECT
    assert m.settings_main_ctl(m.effective_settings(s)).tolist() == \
        m.MAIN_CTL_DEFAULT.tolist() == [1, m.OBSTACLE_EFFECT_REFLECT, m.OBSTACLE_EFFECT_REFLECT]


def test_settings_built_the_old_way_still_get_the_default_main_ray_behavior():
    """Settings/Snapshot gained fields with defaults, so code constructing
    them without the new fields (like older call sites) keeps working."""
    settings = m.Settings(30, m.effective_dist(0.0), 1.0, m.SHAPE_ELLIPSE, (),
                          m.DEFAULT_FUZZ, 0.0, m.BUDDHA_BLUR_GAUSSIAN)
    assert (settings.main_fire, settings.main_in_effect, settings.main_out_effect) == \
        (True, m.OBSTACLE_EFFECT_REFLECT, m.OBSTACLE_EFFECT_REFLECT)


def test_effective_settings_carries_the_main_ray_behavior():
    snap = m.DEFAULT_SNAPSHOT._replace(main_fire=False,
                                       main_in_effect=m.OBSTACLE_EFFECT_PASS,
                                       main_out_effect=m.OBSTACLE_EFFECT_TERMINATE)
    settings = m.effective_settings(snap)
    assert settings.main_fire is False
    assert settings.main_in_effect == m.OBSTACLE_EFFECT_PASS
    assert settings.main_out_effect == m.OBSTACLE_EFFECT_TERMINATE
    assert m.settings_main_ctl(settings).tolist() == [0, m.OBSTACLE_EFFECT_PASS,
                                                      m.OBSTACLE_EFFECT_TERMINATE]


def test_main_ray_settings_are_registered_in_the_keyframe_tables():
    keys = {k for k, _tab, _label in m.SETTING_INFO}
    for key in ("main_fire", "main_in_effect", "main_out_effect"):
        assert key in keys and key in m.DEFAULT_SETTING_VALUES
        assert key not in m.SETTING_FLOAT_KEYS, "never interpolated: held until the next keyframe"


def test_capture_and_resolve_round_trip_the_main_ray_settings():
    app = _make_fake_app_for_snapshot()
    app.main_fire_var = _StubVar(False)
    app.main_in_effect_var = _StubVar(m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_PASS])
    app.main_out_effect_var = _StubVar(m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_TERMINATE])
    app.deco = {"outline": _StubVar(True)}
    app.obstacles = []
    app.views = []
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    data = app.capture_all_settings()
    assert data["main_fire"] is False
    assert data["main_in_effect"] == m.OBSTACLE_EFFECT_PASS
    assert data["main_out_effect"] == m.OBSTACLE_EFFECT_TERMINATE
    snap = app.current_snapshot()
    assert (snap.main_fire, snap.main_in_effect, snap.main_out_effect) == \
        (False, m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_TERMINATE)

    kf = m.Keyframe(0, 0, data, "Linear", None)
    resolved = m.resolve_frame_settings([kf], 0).snapshot
    assert (resolved.main_fire, resolved.main_in_effect, resolved.main_out_effect) == \
        (False, m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_TERMINATE)


def test_main_ray_settings_hold_between_keyframes_rather_than_blend():
    kf0 = m.Keyframe(0, 0, {"main_fire": True, "main_out_effect": m.OBSTACLE_EFFECT_REFLECT})
    kf1 = m.Keyframe(1, 10, {"main_fire": False, "main_out_effect": m.OBSTACLE_EFFECT_PASS})
    mid = m.resolve_frame_settings([kf0, kf1], 5).snapshot
    assert mid.main_fire is True and mid.main_out_effect == m.OBSTACLE_EFFECT_REFLECT
    end = m.resolve_frame_settings([kf0, kf1], 10).snapshot
    assert end.main_fire is False and end.main_out_effect == m.OBSTACLE_EFFECT_PASS


def test_short_value_text_for_the_main_ray_settings():
    assert m.short_value_text("main_fire", True) == "On"
    assert m.short_value_text("main_fire", False) == "Off"
    assert m.short_value_text("main_in_effect", m.OBSTACLE_EFFECT_PASS) == "Pass through"
    assert m.short_value_text("main_out_effect", m.OBSTACLE_EFFECT_TERMINATE) == "Terminate"


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
    o = m.Obstacle(id_)
    o.width = o.height = 2.0 * radius
    o.r, o.theta, o.cx, o.cy = 0.0, 0.0, cx, cy
    o.in_effect, o.out_effect, o.fire_inside = in_effect, out_effect, fire
    return o


def _kernel_trace(x0, y0, angle_deg, obstacles=(), main=(R, R), main_fire=True,
                  main_shape=None, max_iter=12, max_dist=0.0, seg_mult=1.0):
    """One ray through the real trace_path, heading `angle_deg` (fuzzing is
    switched on with all jitter zero purely to set the direction, and is
    otherwise the identity). Returns (points, kinds, term, raw)."""
    shape, bound = main_shape or (m.SHAPE_ELLIPSE, UNIT_CIRCLE)
    arrays = m.build_obstacle_arrays(list(obstacles))
    fz = m.fuzz_array(m.FuzzParams(True, angle_deg, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0))
    ctl = m.main_ctl_array(main_fire, main[0], main[1])
    pts = np.empty((max_iter + 2, 2))
    kinds = np.empty(max_iter + 2)
    raw = np.empty(m.N_RAW)
    term, npts = m.trace_path(x0, y0, pts, kinds, max_iter, m.effective_dist(max_dist), shape,
                              bound, ctl, seg_mult, fz, *arrays[:5], raw)
    return pts[:npts].copy(), kinds[:npts].copy(), term, raw


def test_a_trapped_obstacle_keeps_its_ray_inside_forever():
    """Reflect both ways = a container: the ray started inside never leaves."""
    box = _circle_obstacle(0.2, 0.0, 0.5, R, R, fire=True)
    pts, _k, term, raw = _kernel_trace(0.2, -0.3, 0.0, [box], main=(P, P), main_fire=False,
                                       max_iter=40)
    assert term in (m.TERM_ITER, m.TERM_CONFINED)   # its own segments may confine it
    assert int(raw[m.RAW_BOUNCES]) >= 8
    assert (np.hypot(pts[:, 0] - 0.2, pts[:, 1]) <= 0.5 + 1e-9).all()
    assert np.isclose(np.hypot(pts[1, 0] - 0.2, pts[1, 1]), 0.5)


def test_an_obstacle_reflecting_both_ways_behaves_exactly_like_the_bounding_shape():
    """The differential test for the whole design: for every shape kind, a
    Reflect/Reflect obstacle of that shape (with the bounding shape itself
    switched to pass-through and fire off) traces every ray identically to
    the classic bounding shape with no obstacle - including with fuzzing and
    a segment length multiplier."""
    cases = [(m.SHAPE_ELLIPSE, m.ellipse_bound(1.7, 1.1, 25.0)),
             (m.SHAPE_POLYGON, m.polygon_bound(5, 12.0)),
             (m.SHAPE_MOSS_EGG, m.moss_egg_bound(90.0, 1.0, 10.0)),
             (m.SHAPE_SECTOR, m.sector_bound(1.0, 100.0, 15.0)),
             (m.SHAPE_LENS, m.lens_bound(1.0, 1.0, 1.0, 20.0)),
             (m.SHAPE_CAPSULE, m.capsule_bound(1.0, 0.5, 30.0))]
    fuzz = m.fuzz_array(m.FuzzParams(True, 45.0, 3.0, 2.0, 5.0, 1.1, 1.0, 4.0, 0.9, 7))
    rng = np.random.RandomState(3)
    for shape, bound in cases:
        obst_shapes = np.array([shape], dtype=np.int64)
        obst_bounds = np.zeros((1, m.OBST_BOUND_LEN))
        obst_bounds[0, :len(bound)] = bound
        zero, one_eff = np.zeros(1), np.array([[R, R]], dtype=np.int64)
        classic_ctl = m.main_ctl_array(True, R, R)
        absent_ctl = m.main_ctl_array(False, P, P)
        empty = m.build_obstacle_arrays([])
        checked = 0
        for _ in range(60):
            x, y = rng.uniform(-1, 1, 2)
            if not m.inside_boundary(x, y, shape, bound):
                continue
            for seg_mult, fz in ((1.0, m.fuzz_array(m.DEFAULT_FUZZ)), (0.6, fuzz)):
                results = []
                for ctl, arrs in ((classic_ctl, empty[:5]),
                                  (absent_ctl, (obst_shapes, obst_bounds, zero, zero, one_eff))):
                    m_shape, m_bound = (shape, bound) if ctl is classic_ctl else \
                        (m.SHAPE_ELLIPSE, UNIT_CIRCLE)
                    pts = np.empty((32, 2))
                    kinds = np.empty(32)
                    raw = np.empty(m.N_RAW)
                    _seed_kernel_rng(0)
                    term, n = m.trace_path(x, y, pts, kinds, 30, m.effective_dist(0.0), m_shape,
                                           m_bound, ctl, seg_mult, fz, *arrs, raw)
                    results.append((term, n, pts[:n].copy(), kinds[:n].copy()))
                (t1, n1, p1, k1), (t2, n2, p2, k2) = results
                assert (t1, n1) == (t2, n2), (shape, x, y, seg_mult)
                assert np.allclose(p1, p2, atol=1e-9) and np.allclose(k1, k2), (shape, x, y)
                checked += 1
        assert checked > 20, shape


def test_an_obstacle_passing_rays_through_both_ways_is_as_if_it_were_not_there():
    ghost = _circle_obstacle(0.3, 0.1, 0.3, P, P, fire=False)
    with_it = _kernel_trace(-0.6, -0.4, 30.0, [ghost])
    without = _kernel_trace(-0.6, -0.4, 30.0)
    assert with_it[2] == without[2]
    assert np.allclose(with_it[0], without[0]) and np.allclose(with_it[1], without[1])


def test_enter_but_never_exit_traps_a_ray_that_started_outside():
    """Incoming Pass through, outgoing Reflect: rays enter freely, then stay."""
    trap = _circle_obstacle(0.5, 0.0, 0.3, P, R)
    pts, _k, term, raw = _kernel_trace(0.6, -0.7, 90.0, [trap], max_iter=30)
    wall_y = np.sqrt(0.09 - 0.01)
    # the entry crossing isn't a bounce and doesn't shorten the segment: the
    # first recorded point is the far wall, reached after crossing the trap
    assert np.allclose(pts[1], (0.6, wall_y))
    assert (np.hypot(pts[1:, 0] - 0.5, pts[1:, 1]) <= 0.3 + 1e-9).all()
    assert term in (m.TERM_ITER, m.TERM_CONFINED)   # its own segments may confine it
    assert int(raw[m.RAW_BOUNCES]) >= 5


def test_incoming_terminate_stops_a_ray_at_the_edge_it_meets_from_outside():
    stop = _circle_obstacle(0.5, 0.0, 0.3, T, P)
    pts, kinds, term, _raw = _kernel_trace(0.6, -0.7, 90.0, [stop])
    assert term == m.TERM_OBSTACLE
    assert np.allclose(pts[-1], (0.6, -np.sqrt(0.09 - 0.01)))
    assert kinds[-1] == m.KIND_OBSTACLE_TERM


def test_outgoing_terminate_stops_a_ray_fired_inside_at_the_edge_it_reaches():
    stop = _circle_obstacle(0.5, 0.0, 0.3, R, T, fire=True)
    pts, kinds, term, _raw = _kernel_trace(0.6, -0.1, 90.0, [stop], main=(P, P), main_fire=False)
    assert term == m.TERM_OBSTACLE
    assert np.allclose(pts[-1], (0.6, np.sqrt(0.09 - 0.01)))
    assert kinds[-1] == m.KIND_OBSTACLE_TERM


def test_outgoing_pass_through_lets_a_ray_fired_inside_carry_on_to_the_bounding_shape():
    hole = _circle_obstacle(0.0, 0.0, 0.3, R, P, fire=True)
    pts, _k, _term, _raw = _kernel_trace(0.1, 0.2, 0.0, [hole], main_fire=False)
    assert np.isclose(np.hypot(*pts[1]), 1.0), "first bounce is on the bounding shape"


def test_a_ray_that_leaves_the_bounding_shape_escapes_at_the_image_edge():
    pts, kinds, term, raw = _kernel_trace(0.0, -0.5, 0.0, main=(R, P))
    assert term == m.TERM_ESCAPED
    assert len(pts) == 2 and int(raw[m.RAW_BOUNCES]) == 0
    assert np.allclose(pts[-1], (1.0, -0.5)), "ends where it leaves the [-1, 1] square"
    assert kinds[-1] == m.KIND_ESCAPED


def test_a_ray_escapes_after_bouncing_when_it_finally_meets_nothing():
    wall = _circle_obstacle(0.0, 0.0, 0.3, R, P, fire=False)
    # leaves the (pass-through) bounding shape... bounces off the obstacle first
    pts, _k, term, raw = _kernel_trace(-0.9, 0.2, 0.0, [wall], main=(R, P))
    assert term == m.TERM_ESCAPED
    assert int(raw[m.RAW_BOUNCES]) == 1
    assert abs(pts[-1, 0]) == 1.0 or abs(pts[-1, 1]) == 1.0


def test_an_obstacle_outside_the_bounding_shape_can_fire_rays_that_meet_it_from_outside():
    far = _circle_obstacle(1.3, 0.3, 0.2, R, P, fire=True)
    pts, _k, term, _raw = _kernel_trace(1.3, 0.3, 180.0, [far], max_iter=4)
    hit = (np.sqrt(1.0 - 0.09), 0.3)
    assert np.allclose(pts[1], hit), "meets the bounding shape's OUTSIDE and reflects"
    assert not np.allclose(pts[2], (0.0, 0.0))


def test_a_ray_can_pass_into_the_bounding_shape_from_outside_and_cross_it():
    far = _circle_obstacle(1.3, 0.3, 0.2, R, P, fire=True)
    pts, _k, _term, _raw = _kernel_trace(1.3, 0.3, 180.0, [far], main=(P, R), max_iter=3)
    assert np.allclose(pts[1], (-np.sqrt(1.0 - 0.09), 0.3)), "reflects off the FAR inner wall"


def test_a_distance_limit_cuts_an_escaping_ray_before_the_image_edge():
    pts, kinds, term, raw = _kernel_trace(0.0, -0.5, 0.0, main=(R, P), max_dist=0.4)
    assert term == m.TERM_DIST
    assert np.allclose(pts[-1], (0.4, -0.5))
    assert np.isclose(raw[m.RAW_LAST_FULL], 1.0), "the uncut segment reached the image edge"


def test_the_segment_multiplier_treats_pass_through_crossings_as_part_of_one_segment():
    """With a multiplier, the recorded point is pulled back along the WHOLE
    segment from its start - past the crossings in it - and the ray's inside/
    outside state must follow, or it would carry on as if still inside a
    shape it was pulled back out of."""
    trap = _circle_obstacle(0.5, 0.0, 0.3, P, R)
    full = _kernel_trace(0.6, -0.7, 90.0, [trap], max_iter=3)[0]
    pulled = _kernel_trace(0.6, -0.7, 90.0, [trap], max_iter=3, seg_mult=0.1)[0]
    total = full[1, 1] - (-0.7)
    assert np.allclose(pulled[1], (0.6, -0.7 + 0.1 * total)), "pulled back along the whole segment"
    # It is now back OUTSIDE the trap, so its next segment must run to the
    # bounding shape - not off an edge of a shape it was pulled back out of.
    # (Every recorded point is pulled back to 10% of its true hit, so undo that
    # to find where this segment really ended.)
    true_hit = pulled[1] + (pulled[2] - pulled[1]) / 0.1
    assert np.isclose(np.hypot(*true_hit), 1.0), true_hit


def test_coincident_pass_through_walls_cannot_hang_a_ray():
    a = _circle_obstacle(0.5, 0.0, 0.3, P, R, id_=0)
    b = _circle_obstacle(0.5, 0.0, 0.3, P, R, id_=1)
    _pts, _k, term, _raw = _kernel_trace(0.6, -0.7, 90.0, [a, b], max_iter=10)
    assert term in (m.TERM_ITER, m.TERM_CONFINED, m.TERM_PERPENDICULAR)


def test_termination_tables_cover_escaped_and_are_evenly_spaced():
    assert m.TERM_ESCAPED in m.TERM_CODES
    order = [m.TERM_ITER, m.TERM_CONFINED, m.TERM_DIST, m.TERM_OBSTACLE,
             m.TERM_PERPENDICULAR, m.TERM_ESCAPED]
    assert [m.TERM_STAT_VALUE[t] for t in order] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert m.TERM_ESCAPED in m.TERM_END_COLOR and m.TERM_ESCAPED in m.TERM_OUTCOME_TEXT
    assert m.KIND_ESCAPED in m.KIND_COLOR


def test_stats_and_points_are_sane_for_an_escaped_ray():
    pts, kinds, term, raw = _kernel_trace(0.0, -0.5, 0.0, main=(R, P))
    out = np.empty(m.N_STATS)
    sp = np.empty((m.N_POINTS, 2))
    m.compute_stats(pts, len(pts), raw, 12, 1.0, out)
    m.compute_points(pts, len(pts), raw, kinds, sp)
    assert out[m.ST_TERMINATION] == 1.0
    assert np.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()
    assert np.allclose(sp[m.POINT_END], (1.0, -0.5)) and np.allclose(sp[m.POINT_START], (0.0, -0.5))
    assert out[m.ST_BOUNCES] == 0.0


def test_every_termination_reason_maps_to_its_own_stat_value_in_compute_stats():
    seen = {}
    for term, value in m.TERM_STAT_VALUE.items():
        pts = np.array([[0.0, 0.0], [0.5, 0.1], [0.6, 0.3]])
        raw = np.zeros(m.N_RAW)
        raw[m.RAW_TERM] = term
        raw[m.RAW_BOUNCES] = 1
        out = np.empty(m.N_STATS)
        m.compute_stats(pts, 3, raw, 10, 1.0, out)
        seen[term] = out[m.ST_TERMINATION]
        assert np.isclose(out[m.ST_TERMINATION], value), term
    assert len(set(np.round(list(seen.values()), 6))) == len(m.TERM_CODES)


# --------------------------------------------------------------------------- #
# The new ray behavior end to end: the real render paths, not just trace_path
# --------------------------------------------------------------------------- #
def _terminated_view(**_):
    """A view whose channel 0 is the termination-reason metric and whose other
    channels are constant 0: a pixel's red value then says why its ray stopped
    (255 = escaped)."""
    v = m.ViewConfig(0)
    v.kinds[0] = m.KIND_OTHER
    v.other_metric[0] = m.ST_TERMINATION
    v.chan_src[1] = v.chan_src[2] = m.CHAN_SRC_CONSTANT
    v.constant_choice[1] = v.constant_choice[2] = m.NONE_ZERO
    return _view_settings_from_config(v)


def test_render_full_image_colors_escaped_rays_when_the_bounding_shape_lets_rays_out():
    app = _fake_render_app()
    values = dict(m.DEFAULT_SETTING_VALUES, main_out_effect=P)
    data = m.resolve_frame_settings([m.Keyframe(1, 0, values, "Linear", None)], 0)
    img = app.render_full_image(data, 40, 40, _empty_obst_arrays(), (_terminated_view(),))
    assert tuple(img[20, 20]) == (255, 0, 0), "escaped rays are termination value 1.0"
    assert tuple(img[0, 0]) == tuple(m.BACKGROUND), "outside the shape nothing fires"


def test_render_full_image_default_bounding_shape_produces_no_escaped_rays():
    app = _fake_render_app()
    data = m.resolve_frame_settings(
        [m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES), "Linear", None)], 0)
    img = app.render_full_image(data, 40, 40, _empty_obst_arrays(), (_terminated_view(),))
    inside = img[20, 20]
    assert inside[0] != 255


def test_render_rows_fires_only_from_shapes_with_fire_on():
    """The real (parallel) render kernel: with the bounding shape's fire off, only
    pixels inside an obstacle with fire on emit a ray."""
    obstacle = _circle_obstacle(0.3, 0.0, 0.25, R, R, fire=True)
    arrays = m.build_obstacle_arrays([obstacle])
    ctl = m.main_ctl_array(False, R, R)
    stats = np.zeros((m.SIZE, m.SIZE, m.N_STATS), dtype=np.float32)
    points = np.zeros((m.SIZE, m.SIZE, m.N_POINTS, 2), dtype=np.float32)
    status = np.zeros((m.SIZE, m.SIZE), dtype=np.uint8)
    m.render_rows(stats, points, status, 0, m.SIZE, 8, m.effective_dist(0.0), m.SHAPE_ELLIPSE,
                  UNIT_CIRCLE, ctl, 1.0, m.fuzz_array(m.DEFAULT_FUZZ), *arrays, 0)
    ys, xs = np.nonzero(status)
    wx = (xs + 0.5) * m.PIXEL - 1.0
    wy = 1.0 - (ys + 0.5) * m.PIXEL
    assert len(xs) > 1000
    assert (np.hypot(wx - 0.3, wy) <= 0.25 + m.PIXEL).all(), "only inside the obstacle"
    inside_count = int((np.hypot(*np.meshgrid((np.arange(m.SIZE) + 0.5) * m.PIXEL - 1.0 - 0.3,
                                              1.0 - (np.arange(m.SIZE) + 0.5) * m.PIXEL)) <= 0.25
                        ).sum())
    assert abs(len(xs) - inside_count) <= 4, "and from essentially every pixel inside it"


def test_buddha_multi_scatter_includes_the_escaping_segment():
    counts = np.zeros((m.SIZE, m.SIZE))
    ctl = m.main_ctl_array(True, R, P)              # rays leave the bounding shape
    m.scatter_multi_buddha(250, 252, 8, m.effective_dist(0.0), m.SHAPE_ELLIPSE, UNIT_CIRCLE,
                           ctl, 1.0, m.fuzz_array(m.DEFAULT_FUZZ),
                           *m.build_obstacle_arrays([]), 0, False, True, False, False, counts)
    # Nothing to bounce off, so every firing ray is a single segment (start to
    # the image edge) with exactly one midpoint: the total is the firing pixels.
    xs = (np.arange(m.SIZE) + 0.5) * m.PIXEL - 1.0
    gx, gy = np.meshgrid(xs, 1.0 - (np.array([250, 251]) + 0.5) * m.PIXEL)
    assert counts.sum() == int((np.hypot(gx, gy) < 1.0).sum())


def test_obstacle_effect_dropdowns_write_both_effects_to_the_selected_obstacle():
    class FakeApp:
        pass
    app = FakeApp()
    app._obst_editor_loading = False
    app.selected_obstacle = m.Obstacle(0)
    app.obs_in_effect_var = _StubVar(m.OBSTACLE_EFFECT_NAMES[T])
    app.obs_out_effect_var = _StubVar(m.OBSTACLE_EFFECT_NAMES[R])
    app.obs_in_effect_combo = _StubWidget()
    app.obs_out_effect_combo = _StubWidget()
    app.rebuild_obstacle_list = lambda: None
    changed = []
    app.on_shape_param_changed = lambda: changed.append(1)
    app.on_obstacle_effect_changed = m.App.on_obstacle_effect_changed.__get__(app)
    app.on_obstacle_effect_changed()
    assert app.selected_obstacle.in_effect == T and app.selected_obstacle.out_effect == R
    assert changed


def test_simulation_info_text_mentions_bounding_shape_ray_behavior_only_when_not_default():
    default = m.simulation_info_lines(m.effective_settings(m.DEFAULT_SNAPSHOT))
    assert not any(line.startswith("Bound rays") for line in default)
    custom = m.effective_settings(m.DEFAULT_SNAPSHOT._replace(
        main_fire=False, main_in_effect=P, main_out_effect=T))
    lines = m.simulation_info_lines(custom)
    at = lines.index("Bound rays:")
    assert lines[at + 1:at + 4] == ["  Fire:Off", "  In:Pass through", "  Out:Terminate"]


def test_every_glyph_in_the_new_info_text_exists_in_the_pixel_font():
    for text in ("Bound rays:", "  Fire:On", "  Fire:Off", "  In:Reflect", "  In:Pass through",
                 "  Out:Terminate"):
        missing = [c for c in text if c not in m.FONT_5X7]
        assert not missing, (text, missing)


# --------------------------------------------------------------------------- #
# Loop-area metric: the first self-loop's area divided by the area of a circle
# with the same perimeter as the loop (the isoperimetric ratio), so it never
# depends on any shape's size and 1 means a perfect circle
# --------------------------------------------------------------------------- #
def _loop_stats(loop_vertices, seg_mult=1.0):
    """compute_stats() for a synthetic path: a start point, then `loop_vertices`
    as bounce points 1..n whose polygon is the first self-loop (self-touch at
    the last one, hitting segment 0)."""
    loop = np.asarray(loop_vertices, dtype=float)
    pts = np.vstack([[-5.0, -5.0], loop])
    raw = np.zeros(m.N_RAW)
    raw[m.RAW_BOUNCES] = len(loop)
    raw[m.RAW_SELF_K] = 0
    raw[m.RAW_SELF_M] = len(loop)
    raw[m.RAW_TERM] = m.TERM_ITER
    out = np.empty(m.N_STATS)
    m.compute_stats(pts, len(pts), raw, 20, seg_mult, out)
    return out[m.ST_LOOP_AREA], raw[m.RAW_AREA]


def _regular_polygon(n, radius=1.0):
    ang = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack((radius * np.cos(ang), radius * np.sin(ang)))


def test_loop_area_metric_is_the_isoperimetric_ratio_for_regular_polygons():
    for n in (3, 4, 6, 12, 60):
        ratio, area = _loop_stats(_regular_polygon(n, 0.4))
        expected = np.pi / (n * np.tan(np.pi / n))
        assert np.isclose(ratio, expected, atol=1e-6), (n, ratio, expected)
        assert area > 0
    assert np.isclose(_loop_stats(_regular_polygon(4, 0.3))[0], np.pi / 4)   # a square


def test_loop_area_metric_does_not_depend_on_the_loops_size():
    small = _loop_stats(_regular_polygon(6, 0.05))[0]
    large = _loop_stats(_regular_polygon(6, 0.9))[0]
    assert np.isclose(small, large, atol=1e-6)


def test_loop_area_metric_is_lower_for_a_thin_loop_than_a_round_one():
    thin = _loop_stats([(0.0, 0.0), (0.9, 0.05), (0.9, -0.05), (0.0, -0.1)])[0]
    round_ = _loop_stats(_regular_polygon(24, 0.4))[0]
    assert 0.0 < thin < 0.5 < round_ <= 1.0


def test_loop_area_metric_stays_within_zero_and_one_even_for_a_self_crossing_loop():
    bowtie = _loop_stats([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)])[0]
    assert 0.0 <= bowtie <= 1.0


def test_loop_area_metric_is_zero_when_the_ray_never_touches_itself():
    pts = np.array([[0.0, 0.0], [0.5, 0.1], [0.6, 0.3]])
    raw = np.zeros(m.N_RAW)
    raw[m.RAW_SELF_K] = -1
    raw[m.RAW_BOUNCES] = 1
    raw[m.RAW_TERM] = m.TERM_ITER
    out = np.empty(m.N_STATS)
    m.compute_stats(pts, 3, raw, 10, 1.0, out)
    assert out[m.ST_LOOP_AREA] == 0.0


def test_loop_area_metric_is_zero_not_nan_for_a_zero_length_loop():
    ratio, _area = _loop_stats([(0.2, 0.2), (0.2, 0.2), (0.2, 0.2)])
    assert ratio == 0.0


def test_loop_area_metric_is_still_zero_when_the_segment_multiplier_is_active():
    assert _loop_stats(_regular_polygon(6, 0.4), seg_mult=0.5)[0] == 0.0


def test_loop_area_metric_is_described_as_relative_to_a_same_perimeter_circle():
    label = m.STAT_LABELS[m.ST_LOOP_AREA].lower()
    assert "circle" in label and "boundary" not in label
    assert "perimeter" in m.STAT_DESCRIPTIONS[m.ST_LOOP_AREA]


# --------------------------------------------------------------------------- #
# Tabs whose contents can outgrow them (depending on font size and display
# scaling) scroll instead of squeezing their contents
# --------------------------------------------------------------------------- #
def test_make_scrollable_tab_returns_a_canvas_and_a_separate_inner_frame():
    tab = object.__new__(m.tk.Frame)
    canvas, inner = m.make_scrollable_tab(tab)
    assert canvas is not None and inner is not None
    assert canvas is not inner, "widgets go in the inner frame, not the canvas itself"


def test_the_simulation_and_obstacles_tabs_are_built_inside_scrollable_frames():
    import inspect
    source = inspect.getsource(m.App.build_controls_quadrant)
    assert "self.sim_canvas, sim_inner = make_scrollable_tab(sim_tab)" in source
    assert "self.build_limits_controls(sim_inner)" in source
    assert "self.obst_canvas, obst_inner = make_scrollable_tab(obst_tab)" in source
    assert "self.build_obstacles_controls(obst_inner)" in source


# --------------------------------------------------------------------------- #
# Startup screenshot (development aid): once per launch, and only if a dev/
# folder exists next to the program, the app grabs a picture of its own
# window into dev/screenshots/<FILE_VERSION>-YYYYMMDD-hhmmss.png
# --------------------------------------------------------------------------- #
class _FakeGrabbedImage:
    def __init__(self):
        self.saved_to = None

    def save(self, path):
        self.saved_to = path
        with open(path, "wb") as f:
            f.write(b"fake png bytes")


class _FakeImageGrab:
    def __init__(self):
        self.bboxes = []
        self.images = []

    def grab(self, bbox=None):
        self.bboxes.append(bbox)
        img = _FakeGrabbedImage()
        self.images.append(img)
        return img


class _FakeGeometryRoot:
    """winfo_* values a real root would report: window at (100, 50), 1500x900."""
    updates = 0

    def update(self):
        self.updates += 1

    def winfo_rootx(self):
        return 100

    def winfo_rooty(self):
        return 50

    def winfo_width(self):
        return 1500

    def winfo_height(self):
        return 900


class _ScreenshotEnv:
    """Points m.DEV_DIR at a scratch directory (created only if
    `with_dev_dir`) and m.ImageGrab at a fake (or None), restoring both."""
    def __init__(self, with_dev_dir=True, grab="fake"):
        self.with_dev_dir = with_dev_dir
        self.grab = _FakeImageGrab() if grab == "fake" else grab

    def __enter__(self):
        self.parent = tempfile.mkdtemp()
        self.dev_dir = os.path.join(self.parent, "dev")
        if self.with_dev_dir:
            os.makedirs(self.dev_dir)
        self.orig_dev, self.orig_grab = m.DEV_DIR, m.ImageGrab
        m.DEV_DIR, m.ImageGrab = self.dev_dir, self.grab
        return self

    def __exit__(self, *exc):
        m.DEV_DIR, m.ImageGrab = self.orig_dev, self.orig_grab


def test_dev_dir_is_dev_next_to_the_program():
    assert m.DEV_DIR == os.path.join(m.SCRIPT_DIR, "dev")


def test_screenshot_is_skipped_without_a_dev_dir():
    with _ScreenshotEnv(with_dev_dir=False) as env:
        assert m.take_ui_screenshot(_FakeGeometryRoot()) is None
        assert env.grab.bboxes == [], "must not even attempt a grab"
        assert not os.path.exists(env.dev_dir), "must not create dev/ itself"


def test_screenshot_is_skipped_without_pillow():
    with _ScreenshotEnv(grab=None) as env:
        assert m.take_ui_screenshot(_FakeGeometryRoot()) is None
        assert not os.path.exists(os.path.join(env.dev_dir, "screenshots"))


def test_screenshot_grabs_the_windows_own_rectangle_and_saves_it_named_by_version():
    import re
    root = _FakeGeometryRoot()
    with _ScreenshotEnv() as env:
        path = m.take_ui_screenshot(root)
        shots_dir = os.path.join(env.dev_dir, "screenshots")
        assert os.path.dirname(path) == shots_dir
        assert re.fullmatch(rf"{m.FILE_VERSION}-\d{{8}}-\d{{6}}\.png",
                            os.path.basename(path)), os.path.basename(path)
        assert os.path.isfile(path)
        assert env.grab.bboxes == [(100, 50, 1600, 950)]
        assert env.grab.images[0].saved_to == path
    assert root.updates >= 1, "must let Tk finish drawing before grabbing"


def test_startup_screenshot_happens_once_per_launch():
    class FakeApp:
        pass
    app = FakeApp()
    app.root = _FakeGeometryRoot()
    app.take_startup_screenshot = m.App.take_startup_screenshot.__get__(app)
    with _ScreenshotEnv() as env:
        app.take_startup_screenshot()
        app.take_startup_screenshot()
        assert len(env.grab.bboxes) == 1


def test_startup_screenshot_failure_never_breaks_the_app():
    class BoomGrab:
        def grab(self, bbox=None):
            raise OSError("screen unavailable")
    class FakeApp:
        pass
    app = FakeApp()
    app.root = _FakeGeometryRoot()
    app.take_startup_screenshot = m.App.take_startup_screenshot.__get__(app)
    with _ScreenshotEnv(grab=BoomGrab()):
        app.take_startup_screenshot()      # must not raise


# --------------------------------------------------------------------------- #
# Output locations: everything exported goes to exports/ next to the program,
# and the video frames' TEMP folder lives inside it
# --------------------------------------------------------------------------- #
def test_export_dir_is_exports_next_to_the_program_and_temp_is_inside_it():
    assert m.EXPORT_DIR == os.path.join(m.SCRIPT_DIR, "exports")
    assert m.TEMP_DIR == os.path.join(m.EXPORT_DIR, "TEMP")


def test_ensure_export_dir_creates_a_missing_directory_and_returns_it():
    parent = tempfile.mkdtemp()
    target = os.path.join(parent, "exports")
    orig = m.EXPORT_DIR
    m.EXPORT_DIR = target
    try:
        assert not os.path.exists(target)
        assert m.ensure_export_dir() == target
        assert os.path.isdir(target)
        assert m.ensure_export_dir() == target      # already existing is fine
    finally:
        m.EXPORT_DIR = orig


def test_export_writes_the_current_figure_into_the_export_dir():
    """Regression test for the exports/ move: "Export" (no dialog) used to
    write next to the program itself. Also confirms the directory is created
    if it doesn't exist yet."""
    class FakeApp:
        pass
    app = FakeApp()
    app.save_source_var = _StubVar(m.SAVE_SOURCE_OPTIONS[0])
    app.save_format_var = _StubVar("png")
    app.generated_settings = m.effective_settings(m.DEFAULT_SNAPSHOT)
    written = []
    app._write = lambda path: written.append(path)
    app.save = m.App.save.__get__(app)
    parent = tempfile.mkdtemp()
    target = os.path.join(parent, "exports")
    orig = m.EXPORT_DIR
    m.EXPORT_DIR = target
    try:
        app.save()
    finally:
        m.EXPORT_DIR = orig
    assert len(written) == 1
    assert os.path.dirname(written[0]) == target
    assert os.path.basename(written[0]).endswith(".png")
    assert os.path.isdir(target)


def test_export_as_dialog_opens_in_the_export_dir():
    class FakeApp:
        pass
    app = FakeApp()
    app.root = None
    app.save_source_var = _StubVar(m.SAVE_SOURCE_OPTIONS[0])
    app.save_format_var = _StubVar("png")
    app.generated_settings = m.effective_settings(m.DEFAULT_SNAPSHOT)
    app._write = lambda path: None
    app.save_as = m.App.save_as.__get__(app)
    seen = {}
    orig_dialog = m.filedialog.asksaveasfilename
    m.filedialog.asksaveasfilename = lambda *a, **k: seen.update(k) or ""
    parent = tempfile.mkdtemp()
    target = os.path.join(parent, "exports")
    orig = m.EXPORT_DIR
    m.EXPORT_DIR = target
    try:
        app.save_as()
    finally:
        m.EXPORT_DIR = orig
        m.filedialog.asksaveasfilename = orig_dialog
    assert seen["initialdir"] == target
    assert os.path.isdir(target)


def test_new_image_export_creates_the_export_dir_if_missing():
    app = _fake_export_app()
    app.color_info_location_var = _StubVar("None")
    app.save_new_w_var = _StubVar("20")
    app.save_new_h_var = _StubVar("20")
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.save_btn = _StubWidget()
    app.save_as_btn = _StubWidget()
    app.save_new_image = m.App.save_new_image.__get__(app)
    app._save_new_image_worker = m.App._save_new_image_worker.__get__(app)
    app._save_new_image_done = m.App._save_new_image_done.__get__(app)
    app._save_new_image_failed = m.App._save_new_image_failed.__get__(app)
    app.root = type("R", (), {"after": lambda self, delay, fn: fn()})()
    parent = tempfile.mkdtemp()
    target = os.path.join(parent, "exports")
    orig = m.EXPORT_DIR
    m.EXPORT_DIR = target
    try:
        _run_save_new_image_synchronously(app)
    finally:
        m.EXPORT_DIR = orig
    assert [f for f in os.listdir(target) if f.endswith(".png")], \
        "the exported image should have been written into the freshly created dir"


# --------------------------------------------------------------------------- #
# Export button gating (update_buttons): disabled with an explanatory
# tooltip until a figure has actually finished generating
# --------------------------------------------------------------------------- #
def _fake_button_gating_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.ready = True
    app.current_snapshot = lambda: m.DEFAULT_SNAPSHOT
    app.generated_settings = None
    app.gen_obstacles_sig = None
    app.obstacles = []
    app.baseline = m.DEFAULT_SNAPSHOT
    app.running = False
    app.rows_done = 0
    app.start_btn = _StubWidget()
    app.revert_btn = _StubWidget()
    app.save_btn = _StubWidget()
    app.save_as_btn = _StubWidget()
    app.save_btn_tooltip = m.Tooltip.__new__(m.Tooltip)
    app.save_btn_tooltip.text = ""
    app.save_btn_tooltip.tip = None
    app.save_as_btn_tooltip = m.Tooltip.__new__(m.Tooltip)
    app.save_as_btn_tooltip.text = ""
    app.save_as_btn_tooltip.tip = None
    app.update_buttons = m.App.update_buttons.__get__(app)
    return app


def test_export_disabled_before_any_generation():
    app = _fake_button_gating_app()
    app.update_buttons()
    assert app.save_btn_tooltip.text != ""


def test_export_enabled_after_generation_finishes():
    app = _fake_button_gating_app()
    app.generated_settings = m.effective_settings(m.DEFAULT_SNAPSHOT)
    app.rows_done = m.SIZE
    app.running = False
    app.update_buttons()
    assert app.save_btn_tooltip.text == ""


def test_export_disabled_while_still_generating():
    app = _fake_button_gating_app()
    app.generated_settings = m.effective_settings(m.DEFAULT_SNAPSHOT)
    app.rows_done = m.SIZE // 2
    app.running = True
    app.update_buttons()
    assert app.save_btn_tooltip.text != ""


def test_update_buttons_tolerates_export_widgets_not_built_yet():
    """update_buttons() is called from __init__ before build_status_quadrant
    has run; it must not crash just because save_btn doesn't exist yet."""
    app = _fake_button_gating_app()
    del app.save_btn
    del app.save_btn_tooltip
    del app.save_as_btn
    del app.save_as_btn_tooltip
    app.update_buttons()   # must not raise


# --------------------------------------------------------------------------- #
# Editing a non-live selected view must never touch the live view (this is
# the actual mechanism behind the reported bug: thumbnails silently going
# stale, or two different-looking views becoming identical, because edits
# were always landing on self.live_view() regardless of what was selected)
# --------------------------------------------------------------------------- #
def _fake_color_editor_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.views = [m.ViewConfig(0), m.ViewConfig(1)]
    app.views[1].mode = "HSV"
    app._view_counter = 2
    app.selected_view_id = 0
    app.view_by_id = m.App.view_by_id.__get__(app)
    app.live_view = m.App.live_view.__get__(app)
    app.selected_view_obj = m.App.selected_view_obj.__get__(app)
    app.all_view_ids = m.App.all_view_ids.__get__(app)
    app.select_view = m.App.select_view.__get__(app)
    app.refresh_live_preview_state = m.App.refresh_live_preview_state.__get__(app)
    app._color_editor_loading = False

    app.mode_var = _StubVar("RGB")
    app.mode_combo = _StubWidget()
    app.comp_labels = [_StubWidget() for _ in range(3)]
    app.map_help_var = _StubVar("")
    app.chan_src_vars = [_StubVar(m.CHAN_SRC_QUERY) for _ in range(3)]
    app.chan_src_combos = [_StubWidget() for _ in range(3)]
    app.kind_vars = [_StubVar(m.KIND_POINT) for _ in range(3)]
    app.kind_combos = [_StubWidget() for _ in range(3)]
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

    app.mapping = m.make_mapping(m.DEFAULT_KINDS, m.DEFAULT_POINT_METRICS,
                                 m.DEFAULT_OTHER_METRICS, m.DEFAULT_CHAN_SRCS,
                                 m.DEFAULT_CONSTANT_CHOICES)
    app.rows_done = 0
    app.pinned_px = None
    app.view_thumb_cache = {}
    app.recolor_calls = []
    app.recolor = lambda r0, r1: app.recolor_calls.append((r0, r1))
    app.draw = lambda: None
    app.update_preview = lambda i, j: None
    app.rebuild_buddha_counts_calls = []
    app.rebuild_buddha_counts = lambda: app.rebuild_buddha_counts_calls.append(1)
    app.stat_tag_calls = []
    app.update_stat_tags = lambda: app.stat_tag_calls.append(1)
    app.rebuild_calls = []
    app.rebuild_view_list = lambda: app.rebuild_calls.append(1)
    app.refresh_timeline_diffs = lambda: None
    app.refresh_color_controls_for_selected_view = (
        m.App.refresh_color_controls_for_selected_view.__get__(app))
    app.refresh_row = m.App.refresh_row.__get__(app)
    app.on_mode_changed = m.App.on_mode_changed.__get__(app)
    app.apply_mapping_change = m.App.apply_mapping_change.__get__(app)
    return app


def test_selecting_a_view_makes_it_the_live_one():
    """The core simplification: there's no separate "focus" step anymore -
    selecting a view for editing is what makes it the one shown in the
    preview."""
    app = _fake_color_editor_app()
    assert app.live_view().id == 0
    app.select_view(1)
    assert app.live_view().id == 1
    assert app.live_view() is app.selected_view_obj()


def test_editing_the_selected_view_triggers_the_recolor_cascade():
    app = _fake_color_editor_app()
    app.mode_var.set("HSV")
    app.on_mode_changed()
    assert app.views[0].mode == "HSV"
    assert len(app.recolor_calls) == 1
    assert len(app.stat_tag_calls) == 1


# --------------------------------------------------------------------------- #
# Performance fix: a Buddha rescatter (which can involve re-tracing every
# fired ray for a "multi" point option) must only happen when the specific
# thing that changed could actually affect Buddha data - never for an
# unrelated channel's edit, and never for a mode change
# --------------------------------------------------------------------------- #
def test_apply_mapping_change_skips_buddha_rebuild_for_unrelated_channel():
    """The core performance fix: editing a channel that isn't Buddha must
    not trigger a full Buddha rescatter, even if some OTHER channel on the
    same view IS Buddha (whose rescatter could be very expensive for a
    "multi" point option like All self bounces)."""
    app = _fake_color_editor_app()
    app.views[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.apply_mapping_change(1)   # channel 1 is plain Query, untouched
    assert app.rebuild_buddha_counts_calls == []


def test_apply_mapping_change_rebuilds_when_the_edited_channel_is_buddha():
    app = _fake_color_editor_app()
    app.views[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.apply_mapping_change(0)
    assert len(app.rebuild_buddha_counts_calls) == 1


def test_load_value_skips_buddha_rebuild_for_mode_field():
    """Mode (RGB<->HSV) never affects Buddha scattering at all - matches
    on_mode_changed's own behavior, which never calls this either."""
    app = _fake_color_editor_app()
    app.views[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.load_value = m.App.load_value.__get__(app)
    app.load_value("view:0:mode", "HSV")
    assert app.rebuild_buddha_counts_calls == []


def test_load_value_skips_buddha_rebuild_for_unrelated_channel_field():
    app = _fake_color_editor_app()
    app.views[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.load_value = m.App.load_value.__get__(app)
    chan1_value = (m.CHAN_SRC_QUERY, m.KIND_POINT, m.ST_HDIST, m.POINT_START, m.POINT_END,
                  m.DEFAULT_BUDDHA_POINTS, m.BUDDHA_COMPRESS_ARCTAN, 0, m.NONE_ZERO)
    app.load_value("view:0:chan1", chan1_value)
    assert app.rebuild_buddha_counts_calls == []


def test_load_value_rebuilds_when_the_loaded_channel_is_buddha():
    app = _fake_color_editor_app()
    app.load_value = m.App.load_value.__get__(app)
    chan0_value = (m.CHAN_SRC_BUDDHA, m.KIND_OTHER, m.ST_BOUNCES, m.POINT_START, m.POINT_END,
                  m.DEFAULT_BUDDHA_POINTS, m.BUDDHA_COMPRESS_ARCTAN, 0, m.NONE_ZERO)
    app.load_value("view:0:chan0", chan0_value)
    assert len(app.rebuild_buddha_counts_calls) == 1


def test_delete_view_recolors_the_preview_to_match_the_new_selection():
    app = _fake_view_list_app()
    app.new_view()                     # ids: [0, 1]; 1 is now selected/live
    app.select_view(0)
    app.mapping = m.make_mapping(m.DEFAULT_KINDS, m.DEFAULT_POINT_METRICS,
                                 m.DEFAULT_OTHER_METRICS, m.DEFAULT_CHAN_SRCS)
    recolor_calls = []
    app.recolor = lambda r0, r1: recolor_calls.append((r0, r1))
    app.delete_view()
    assert app.live_view().id == 1
    assert len(recolor_calls) == 1, \
        "deleting the selected (live) view must recolor the preview to match the new one"


def test_on_finished_refreshes_the_views_list():
    """The thumbnails must reflect a just-finished generation even if the
    Coloration tab isn't the active notebook tab - rebuild_view_list is a
    plain method call, not gated on tab visibility."""
    class FakeApp:
        pass
    app = FakeApp()
    app.tick_id = "irrelevant"
    app.update_status = lambda: None
    app.update_buttons = lambda: None
    app.pinned_px = None
    rebuild_calls = []
    app.rebuild_view_list = lambda: rebuild_calls.append(1)
    m.App.on_finished(app)
    assert len(rebuild_calls) == 1


# --------------------------------------------------------------------------- #
# The view-name pixel-font overlay on the live preview
# --------------------------------------------------------------------------- #
def test_font_5x7_covers_every_character_needed_for_view_names():
    for ch in "view0123456789 ":
        assert ch in m.FONT_5X7, f"FONT_5X7 is missing {ch!r}, needed for 'view N' text"


def test_render_text_line_produces_legible_glyphs_for_every_digit():
    """Each digit glyph must actually light up some pixels (not be blank/a
    typo'd all-zero row) and must be exactly 7 rows tall."""
    for ch in "0123456789":
        mask = m.render_text_line(ch, scale=1)
        assert mask.shape[0] == 7
        assert mask.any(), f"digit {ch!r} renders as entirely blank"


def test_view_name_overlay_draws_in_the_bottom_left_corner_only():
    app = type("FakeApp", (), {})()
    app.display_frame = np.full((m.SIZE, m.SIZE, 3), 100, dtype=np.uint8)
    app.views = [m.ViewConfig(0), m.ViewConfig(1)]
    app.selected_view_id = 1
    app.view_by_id = m.App.view_by_id.__get__(app)
    app.selected_view_obj = m.App.selected_view_obj.__get__(app)
    app.live_view = m.App.live_view.__get__(app)
    app.all_view_ids = m.App.all_view_ids.__get__(app)
    app.draw_view_name_overlay = m.App.draw_view_name_overlay.__get__(app)

    app.draw_view_name_overlay()

    # A background-box pixel near the bottom-left corner must be the
    # overlay's background color, and some pixels nearby must be its
    # foreground color (the text itself) - while the rest of the image is
    # untouched.
    assert tuple(app.display_frame[m.SIZE - 10, 10]) == m.VIEW_NAME_OVERLAY_BG
    region = app.display_frame[m.SIZE - 40:m.SIZE, 0:100]
    assert ((region == m.VIEW_NAME_OVERLAY_FG).all(axis=-1)).sum() > 0
    assert tuple(app.display_frame[10, m.SIZE - 10]) == (100, 100, 100)   # untouched elsewhere


def test_view_name_overlay_reflects_the_live_views_position():
    app = type("FakeApp", (), {})()
    app.display_frame = np.zeros((m.SIZE, m.SIZE, 3), dtype=np.uint8)
    app.views = [m.ViewConfig(0), m.ViewConfig(1), m.ViewConfig(2)]
    app.selected_view_id = 2
    app.view_by_id = m.App.view_by_id.__get__(app)
    app.selected_view_obj = m.App.selected_view_obj.__get__(app)
    app.live_view = m.App.live_view.__get__(app)
    app.all_view_ids = m.App.all_view_ids.__get__(app)
    app.draw_view_name_overlay = m.App.draw_view_name_overlay.__get__(app)

    app.draw_view_name_overlay()
    text_pixels_with_id_2 = (app.display_frame == m.VIEW_NAME_OVERLAY_FG).all(axis=-1).sum()

    app2 = type("FakeApp", (), {})()
    app2.display_frame = np.zeros((m.SIZE, m.SIZE, 3), dtype=np.uint8)
    app2.views = app.views
    app2.selected_view_id = 0
    app2.view_by_id = m.App.view_by_id.__get__(app2)
    app2.selected_view_obj = m.App.selected_view_obj.__get__(app2)
    app2.live_view = m.App.live_view.__get__(app2)
    app2.all_view_ids = m.App.all_view_ids.__get__(app2)
    app2.draw_view_name_overlay = m.App.draw_view_name_overlay.__get__(app2)
    app2.draw_view_name_overlay()
    text_pixels_with_id_0 = (app2.display_frame == m.VIEW_NAME_OVERLAY_FG).all(axis=-1).sum()

    # "view 2" and "view 0" are different-length strings of lit pixels in
    # general, but at minimum the two renders should differ - same digit
    # count here, so compare the actual pixel patterns are not identical.
    assert not np.array_equal(app.display_frame, app2.display_frame)


def test_draw_never_lets_the_view_name_overlay_leak_into_the_export_buffer():
    """Regression test: draw() used to draw the overlay directly into
    self.frame, which _write() (the "Current preview image" export path)
    saves straight to disk - meaning the preview-only overlay was silently
    ending up in exported files. self.frame must stay untouched by the
    overlay; only self.display_frame (never written to disk) should carry it."""
    app = type("FakeApp", (), {})()
    app.img = np.full((m.SIZE, m.SIZE, 3), 50, dtype=np.uint8)
    app.frame = np.empty_like(app.img)
    app.display_frame = np.empty_like(app.img)
    app.deco = {"outline": type("X", (), {"get": lambda self: False})()}
    app.ring = np.zeros((m.SIZE, m.SIZE), dtype=bool)
    app.views = [m.ViewConfig(0)]
    app.selected_view_id = 0
    app.view_by_id = m.App.view_by_id.__get__(app)
    app.selected_view_obj = m.App.selected_view_obj.__get__(app)
    app.live_view = m.App.live_view.__get__(app)
    app.all_view_ids = m.App.all_view_ids.__get__(app)
    app.obstacle_ring_specs = lambda: []
    app.draw_view_name_overlay = m.App.draw_view_name_overlay.__get__(app)
    app.draw = m.App.draw.__get__(app)
    app.photo = type("P", (), {"configure": lambda self, **kw: None})()

    app.draw()

    assert tuple(app.display_frame[m.SIZE - 10, 10]) == m.VIEW_NAME_OVERLAY_BG
    assert tuple(app.frame[m.SIZE - 10, 10]) == (50, 50, 50), \
        "the overlay leaked into self.frame, which gets written to disk on export"


# --------------------------------------------------------------------------- #
# Buddha coloration catalog
# --------------------------------------------------------------------------- #
def test_buddha_catalog_lengths_agree():
    assert len(m.DEFAULT_BUDDHA_POINTS) == m.N_BUDDHA_POINTS
    assert len(m.BUDDHA_POINT_LABELS) == m.N_BUDDHA_POINTS


def test_buddha_confined_channel_present():
    assert any("confined" in label.lower() for label in m.BUDDHA_POINT_LABELS)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class SkipTest(Exception):
    """Raised by a test that can't run in this environment (e.g. needs a
    real external tool that isn't installed); reported as SKIP, not PASS."""


def run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items())
            if name.startswith("test_") and callable(fn)]
    passed, failed, skipped = [], [], []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
            print(f"  PASS  {name}")
        except SkipTest as e:
            skipped.append(name)
            print(f"  SKIP  {name}: {e}")
        except Exception:                          # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc(limit=3, file=sys.stdout)
            print()
    print(f"\n{len(passed)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped, {len(tests)} total")
    return len(failed) == 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ok = run_all()
    sys.exit(0 if ok else 1)
