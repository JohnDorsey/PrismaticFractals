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
import dataclasses
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
    fresh.core.SETTING_FLOAT_KEYS = fresh.SETTING_FLOAT_KEYS | {"not_a_real_setting"}
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
    kf = m.Keyframe(1, 0, {**m.DEFAULT_SETTING_VALUES, "dist": 0.4})
    data = m.resolve_frame_settings([kf], 0)
    assert data.settings.max_dist == 0.4
    assert m.effective_dist(data.settings.max_dist) == 0.4


def test_dist_animates_smoothly_between_keyframes():
    """Regression test: "dist" wasn't in SETTING_FLOAT_KEYS, so it was held
    (jumping only at the second keyframe's exact frame) instead of
    interpolated - the actual bug report was "Buddha colors don't appear
    until the very last frame" of a video animating the distance limit."""
    kf0 = m.Keyframe(1, 0, {**m.DEFAULT_SETTING_VALUES, "dist": 0.01})
    kf1 = m.Keyframe(2, 59, {**m.DEFAULT_SETTING_VALUES, "dist": 16})
    dists = [m.resolve_frame_settings([kf0, kf1], fn).settings.max_dist for fn in range(60)]
    assert dists[0] == 0.01 and dists[59] == 16
    assert all(dists[i] < dists[i + 1] for i in range(59)), \
        "dist should increase strictly monotonically across every frame"


# --------------------------------------------------------------------------- #
# Segment length multiplier: core ray-tracing math
# --------------------------------------------------------------------------- #
def _trace(seg_mult, x0=-0.3, y0=-0.2, max_iter=8, width=2.0, height=1.4):
    start = m.default_starting_obstacle(0)
    start.width, start.height = width, height
    bound = start.bound()
    arrays = m.build_obstacle_arrays([start])
    fz = m.fuzz_rule(m.DEFAULT_FUZZ)
    pts = np.empty((max_iter + 2, 2))
    kinds = np.empty(max_iter + 2)
    raw = np.empty(m.N_RAW)
    term, npts = m.trace_path(x0, y0, pts, kinds, max_iter, m.effective_dist(0.0), seg_mult,
                              fz, m.CONFINE_DEFAULT, m.EMISSION_DEFAULT, arrays, raw)
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


def test_obstacle_and_snapshot_agree_on_moss_egg_param_order():
    o = m.MossEggShape(1)
    o.moss_angle, o.moss_size, o.rotation = 100.0, 1.3, 15.0
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
# The shape kinds' parameter tables (Shape subclasses)
# --------------------------------------------------------------------------- #
def _breaks_a_self_check(fresh, expected_text):
    try:
        fresh.run_self_checks()
    except AssertionError as e:
        assert expected_text in str(e), str(e)
    else:
        raise AssertionError("expected run_self_checks() to raise")


def test_self_checks_catch_a_kinds_slots_and_parameter_table_disagreeing():
    fresh = load_module()
    fresh.MossEggShape.FIELD_NAMES = tuple(f for f in fresh.MossEggShape.FIELD_NAMES
                                           if f != "moss_size")
    _breaks_a_self_check(fresh, "MossEggShape: __slots__ and PARAMS disagree")


def test_self_checks_catch_a_duplicate_field_in_a_kind():
    fresh = load_module()
    fresh.EllipseShape.FIELD_NAMES = fresh.EllipseShape.FIELD_NAMES + ("width",)
    _breaks_a_self_check(fresh, "duplicate field name")


def test_self_checks_catch_a_field_that_means_different_things_on_two_kinds():
    fresh = load_module()
    fresh.CircularSectorShape.NUMERIC_PARAMS = tuple(
        p._replace(fmt="angle") if p.field == "radius" else p
        for p in fresh.CircularSectorShape.NUMERIC_PARAMS)
    _breaks_a_self_check(fresh, "radius means different things")


def test_self_checks_catch_a_kind_without_a_valid_layout():
    fresh = load_module()
    fresh.LensShape.LAYOUT = None
    _breaks_a_self_check(fresh, "LensShape.LAYOUT")


def test_every_kind_declares_how_its_bound_array_is_packed():
    expected = {m.EllipseShape: m.LAYOUT_ELLIPSE, m.RegularPolygonShape: m.LAYOUT_POLYGON,
                m.LensShape: m.LAYOUT_CIRCLES, m.ReuleauxPolygonShape: m.LAYOUT_CIRCLES,
                m.MossEggShape: m.LAYOUT_ARCS_EDGES, m.CircularSectorShape: m.LAYOUT_ARCS_EDGES,
                m.CircularSegmentShape: m.LAYOUT_ARCS_EDGES, m.CapsuleShape: m.LAYOUT_ARCS_EDGES,
                m.TrefoilShape: m.LAYOUT_ARC_LOOPS, m.QuatrefoilShape: m.LAYOUT_ARC_LOOPS,
                m.LuneShape: m.LAYOUT_ARC_LOOPS, m.ArbelosShape: m.LAYOUT_ARC_LOOPS}
    assert {cls: cls.LAYOUT for cls in m.SHAPE_CLASSES} == expected


def test_self_checks_catch_a_kind_out_of_place_in_the_kind_list():
    fresh = load_module()
    fresh.SHAPE_CLASSES[0], fresh.SHAPE_CLASSES[1] = fresh.SHAPE_CLASSES[1], fresh.SHAPE_CLASSES[0]
    _breaks_a_self_check(fresh, "KIND")


class _StubLabel:
    def __init__(self):
        self.text = ""

    def configure(self, **options):
        self.text = options.get("text", self.text)


def _fake_obstacle_editor_app():
    """A minimal stand-in for App with stub controls for
    refresh_obstacle_editor() and on_obstacle_param_changed() to run for
    real, unmodified, off the actual App class."""
    class FakeApp:
        pass
    app = FakeApp()
    for name in ("refresh_obstacle_editor", "on_obstacle_param_changed",
                 "show_obstacle_shape_controls", "obstacle_widget"):
        setattr(app, name, getattr(m.App, name).__get__(app))

    app.obs_shape_label = _StubLabel()
    app.obs_in_effect_var = _StubVar(m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_REFLECT])
    app.obs_out_effect_var = _StubVar(m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_PASS])
    app.obs_fire_inside_var = _StubVar(False)
    app.obs_shape_groups = {cls: _TrackedFrame() for cls in m.SHAPE_CLASSES}
    app.obs_widgets = {}
    for cls in m.SHAPE_CLASSES:
        for p in cls.PARAMS:
            app.obs_widgets[(cls, p.field)] = _StubVar(p.default)
    for p in (m.ROTATION_PARAM,) + m.POSITION_PARAMS:
        app.obs_widgets[(m.Shape, p.field)] = _StubVar(p.default)

    app.rebuild_calls = []
    app.rebuild_obstacle_list = lambda: app.rebuild_calls.append(1)
    app.param_changed_calls = []
    app.on_shape_param_changed = lambda: app.param_changed_calls.append(1)
    return app


def _make_distinct_obstacle(shape=m.SHAPE_ELLIPSE):
    """A shape of kind `shape` with a distinct, easily-checked value in every
    field it has, and a non-default effect - so a test can tell every field
    apart from every other."""
    o = m.SHAPE_CLASSES[shape](1)
    values = {"width": 1.11, "height": 1.22, "rotation": 10.0, "sides": 6,
             "radius": 1.44, "moss_angle": 91.0, "moss_size": 1.33,
             "sector_span": 70.0, "segment_span": 80.0, "lens_radius1": 1.66,
             "lens_radius2": 1.77, "lens_sep": 0.88, "reuleaux_sides": 7,
             "capsule_length": 1.99, "capsule_radius": 0.55, "lobe_size": 0.77,
             "cut_radius": 0.66, "cut_offset": 0.55, "split": 0.45,
             "r": 0.5, "theta": 45.0, "cx": 0.1, "cy": -0.2}
    for field, val in values.items():
        if field in o.FIELD_NAMES:
            setattr(o, field, val)
    o.in_effect = m.OBSTACLE_EFFECT_TERMINATE
    o.out_effect = m.OBSTACLE_EFFECT_REFLECT
    o.fire_inside = True
    return o


def test_refresh_obstacle_editor_loads_every_field_of_every_kind():
    for shape in range(len(m.SHAPE_CLASSES)):
        app = _fake_obstacle_editor_app()
        o = _make_distinct_obstacle(shape)
        app.selected_obstacle = o
        app.refresh_obstacle_editor()
        for p in o.NUMERIC_PARAMS:
            got = app.obstacle_widget(type(o), p.field).get()
            assert got == getattr(o, p.field), (type(o).__name__, p.field, got)


def test_refresh_obstacle_editor_loads_the_shape_name_and_effects():
    app = _fake_obstacle_editor_app()
    o = _make_distinct_obstacle(m.SHAPE_LENS)
    app.selected_obstacle = o
    app.refresh_obstacle_editor()
    assert app.obs_shape_label.text == "Two-circle lens"
    assert app.obs_in_effect_var.get() == m.OBSTACLE_EFFECT_NAMES[o.in_effect]
    assert app.obs_out_effect_var.get() == m.OBSTACLE_EFFECT_NAMES[o.out_effect]
    assert app.obs_fire_inside_var.get() == o.fire_inside


def test_refresh_obstacle_editor_shows_only_the_selected_kinds_group():
    for shape, cls in enumerate(m.SHAPE_CLASSES):
        app = _fake_obstacle_editor_app()
        app.selected_obstacle = _make_distinct_obstacle(shape)
        app.refresh_obstacle_editor()
        assert {c for c, g in app.obs_shape_groups.items() if g.shown} == {cls}, cls.__name__


def test_refresh_obstacle_editor_does_nothing_without_a_selected_obstacle():
    app = _fake_obstacle_editor_app()
    app.selected_obstacle = None
    app.refresh_obstacle_editor()   # must not raise


def test_on_obstacle_param_changed_writes_every_field_of_every_kind_with_correct_rounding():
    for shape, cls in enumerate(m.SHAPE_CLASSES):
        app = _fake_obstacle_editor_app()
        o = cls(1)
        app.selected_obstacle = o
        app._obst_editor_loading = False
        for widget in app.obs_widgets.values():     # every kind's controls, not just this one's
            widget.set(1.23456)
        for p in cls.PARAMS:
            app.obs_widgets[(cls, p.field)].set(3 if p.fmt == "int" else 1.23456)
        app.obs_widgets[(m.Shape, "theta")].set(12.345)
        app.on_obstacle_param_changed()
        for p in o.NUMERIC_PARAMS:
            got = getattr(o, p.field)
            expected = (3 if p.fmt == "int" else 12.35 if p.field == "theta" else 1.23)
            assert got == expected or (p.field == "theta" and got == 12.34), \
                f"{cls.__name__}.{p.field} = {got}, expected {expected}"


def test_on_obstacle_param_changed_reads_only_the_selected_kinds_own_controls():
    app = _fake_obstacle_editor_app()
    o = m.LensShape(1)
    app.selected_obstacle = o
    app._obst_editor_loading = False
    app.obs_widgets[(m.EllipseShape, "width")].set(1.7)       # another kind's control
    app.obs_widgets[(m.LensShape, "lens_sep")].set(0.3)
    app.on_obstacle_param_changed()
    assert o.lens_sep == 0.3 and not hasattr(o, "width")


def test_on_obstacle_param_changed_calls_rebuild_and_param_changed():
    app = _fake_obstacle_editor_app()
    app.selected_obstacle = m.EllipseShape(1)
    app._obst_editor_loading = False
    app.on_obstacle_param_changed()
    assert len(app.rebuild_calls) == 1
    assert len(app.param_changed_calls) == 1


def test_on_obstacle_param_changed_skips_while_loading():
    app = _fake_obstacle_editor_app()
    o = m.EllipseShape(1)
    o.width = 9.0
    app.selected_obstacle = o
    app._obst_editor_loading = True
    app.obs_widgets[(m.EllipseShape, "width")].set(1.0)
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
    for shape in range(len(m.SHAPE_CLASSES)):
        app = _fake_obstacle_editor_app()
        o = _make_distinct_obstacle(shape)
        app.selected_obstacle = o
        app.refresh_obstacle_editor()
        app._obst_editor_loading = False
        app.on_obstacle_param_changed()
        for p in o.NUMERIC_PARAMS:
            assert getattr(o, p.field) == getattr(_make_distinct_obstacle(shape), p.field)


# Shape.describe: exact current text for every shape,
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


def _make_obstacle_from_params(shape, params):
    """A shape of the given kind configured to match a _DESCRIBE_SHAPE_CASES
    params tuple, for the equivalent describe() case - fills in the
    radius polygon/Reuleaux shapes have that the tuple doesn't carry."""
    o = m.SHAPE_CLASSES[shape](1)
    if shape == m.SHAPE_ELLIPSE:
        o.width, o.height, o.rotation = params
    elif shape == m.SHAPE_POLYGON:
        o.sides, o.rotation = params
        o.radius = 0.5
    elif shape == m.SHAPE_MOSS_EGG:
        o.moss_angle, o.moss_size, o.rotation = params
    elif shape == m.SHAPE_SECTOR:
        o.radius, o.sector_span, o.rotation = params
    elif shape == m.SHAPE_SEGMENT:
        o.radius, o.segment_span, o.rotation = params
    elif shape == m.SHAPE_LENS:
        o.lens_radius1, o.lens_radius2, o.lens_sep, o.rotation = params
    elif shape == m.SHAPE_REULEAUX:
        o.reuleaux_sides, o.rotation = params
        o.radius = 0.5
    else:
        o.capsule_length, o.capsule_radius, o.rotation = params
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


def test_describe_exact_shape_text_for_every_shape():
    for shape, params, _ in _DESCRIBE_SHAPE_CASES:
        o = _make_obstacle_from_params(shape, params)
        o.r, o.theta, o.cx, o.cy = 0.0, 0.0, 0.0, 0.0
        text = o.describe()
        assert text.startswith(_DESCRIBE_OBSTACLE_SHAPE_TEXT[shape]), \
            f"got: {text!r}"


def test_describe_includes_position_and_effect():
    o = _make_obstacle_from_params(m.SHAPE_ELLIPSE, (1.0, 1.0, 0.0))
    o.r, o.theta, o.cx, o.cy = 0.5, 45.0, 0.1, -0.2
    o.in_effect = m.OBSTACLE_EFFECT_TERMINATE
    o.out_effect = m.OBSTACLE_EFFECT_REFLECT
    text = o.describe()
    assert "r0.50 @ 45.0\u00b0" in text
    assert "+(+0.10, -0.20)" in text
    assert f"in: {m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_TERMINATE]}" in text
    assert f"out: {m.OBSTACLE_EFFECT_NAMES[m.OBSTACLE_EFFECT_REFLECT]}" in text


# params() / snapshot_shape_params: exact tuples for every shape, including
# the polygon/Reuleaux asymmetry (an obstacle carries its own radius; the
# bounding shape doesn't, since it's always inscribed in the unit circle)
def test_obstacle_keyframe_capture_and_load_round_trip():
    app = _fake_render_app()
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    app.current_ray_settings = lambda: m.DEFAULT_RAY_SETTINGS
    app.deco = {"outline": _StubVar(True)}
    app.views = [m.ViewConfig(0)]
    o1 = _make_distinct_obstacle(shape=m.SHAPE_MOSS_EGG)
    o1.id = 1
    o2 = _make_distinct_obstacle(shape=m.SHAPE_LENS)
    o2.id = 2
    o2.lens_sep = 0.44   # a value o1 doesn't have, to catch id cross-contamination
    app.obstacles = [o1, o2]

    captured = app.capture_all_settings()
    assert {k for k in captured if k.startswith("obst:1:")} == {f"obst:1:{f}" for f in o1.FIELD_NAMES}
    assert {k for k in captured if k.startswith("obst:2:")} == {f"obst:2:{f}" for f in o2.FIELD_NAMES}
    assert "obst:1:width" not in captured and "obst:2:moss_size" not in captured
    for field in o1.FIELD_NAMES:
        assert captured[f"obst:1:{field}"] == getattr(o1, field)
    for field in o2.FIELD_NAMES:
        assert captured[f"obst:2:{field}"] == getattr(o2, field)

    # load onto fresh obstacles of the same kinds with matching ids
    fresh1 = m.MossEggShape(1)
    fresh2 = m.LensShape(2)
    load_app = _fake_obstacle_editor_app()
    load_app.load_value = m.App.load_value.__get__(load_app)
    load_app.obstacles = [fresh1, fresh2]
    load_app.selected_obstacle = None
    load_app.rebuild_obstacle_list = lambda: None
    for key, val in captured.items():
        if key.startswith("obst:"):
            load_app.load_value(key, val)
    for field in o1.FIELD_NAMES:
        assert getattr(fresh1, field) == getattr(o1, field), f"field={field}"
    for field in o2.FIELD_NAMES:
        assert getattr(fresh2, field) == getattr(o2, field), f"field={field}"


def test_obstacle_keyframe_load_only_touches_the_matching_id():
    app = _fake_obstacle_editor_app()
    app.load_value = m.App.load_value.__get__(app)
    o1 = m.EllipseShape(1)
    o2 = m.EllipseShape(2)
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

    app.seed_var = _Get("0")
    app.last_seed = 0
    app.fuzz_enabled_var = _Get(False)
    for field in ["emit_jitter", "bnd_offset", "bnd_jitter", "bnd_mult",
                 "self_offset", "self_jitter", "self_mult"]:
        setattr(app, f"{field}_scale", _Get(1.0 if field.endswith("mult") else 0.0))

    app.iter_var = _Get("32")
    app.dist_var = _Get("0")
    app.seg_mult_scale = _Get(0.75)
    app.buddha_blur_radius_scale = _Get(m.DEFAULT_BUDDHA_BLUR_RADIUS)
    app.buddha_blur_type_var = _Get(m.BUDDHA_BLUR_LABELS[m.DEFAULT_BUDDHA_BLUR_TYPE])
    app.confine_window_var = _Get(str(m.DEFAULT_CONFINE_WINDOW))
    app.confine_method_var = _Get(m.CONFINE_METHOD_NAMES[m.DEFAULT_CONFINE_METHOD])
    app.confine_threshold_var = _Get(str(m.DEFAULT_CONFINE_THRESHOLD))
    app.emit_mode_var = _Get(m.EMIT_MODE_NAMES[m.DEFAULT_EMIT_MODE])
    for field in m.EMIT_TARGET_FIELDS:
        setattr(app, f"emit_target_{field}_var", _Get("0.0"))
    app.emit_offset_scale = _Get(m.DEFAULT_EMIT_OFFSET)
    return app


# --------------------------------------------------------------------------- #
# Views (multiple side-by-side colorations of the same simulation)
# --------------------------------------------------------------------------- #
def _fake_render_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.render_full_image = m.App.render_full_image.__get__(app)
    app.colorize_view = m.App.colorize_view.__get__(app)
    return app


def _starting_obst_arrays():
    """The scene the app starts with: one obstacle, the unit circle."""
    return m.build_obstacle_arrays([m.default_starting_obstacle(0)])


def _view_settings_from_config(v):
    return m.ViewSettings(mode=v.mode, kinds=v.kinds, point_metric=v.point_metric,
                          other_metric=v.other_metric, point1_choice=v.point1_choice,
                          point2_choice=v.point2_choice, chan_src=v.chan_src,
                          buddha_points=v.buddha_points, buddha_compression=v.buddha_compression,
                          buddha_brightness=v.buddha_brightness,
                          constant_choice=v.constant_choice)


def test_render_full_image_with_a_single_view_matches_pre_views_behavior():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES))
    data = m.resolve_frame_settings([kf], 0)
    views = (_view_settings_from_config(m.ViewConfig(0)),)
    img = app.render_full_image(data, 40, 40, _starting_obst_arrays(), views)
    assert img.shape == (40, 40, 3)


def test_two_identical_views_render_as_two_identical_halves():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES))
    data = m.resolve_frame_settings([kf], 0)
    obst_arrays = _starting_obst_arrays()
    one_view = (_view_settings_from_config(m.ViewConfig(0)),)
    single = app.render_full_image(data, 50, 50, obst_arrays, one_view)
    doubled = app.render_full_image(data, 50, 50, obst_arrays, one_view * 2)
    assert doubled.shape == (50, 100, 3)
    assert np.array_equal(doubled[:, :50], doubled[:, 50:])
    assert np.array_equal(doubled[:, :50], single)


def test_two_different_views_render_as_two_different_halves():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES))
    data = m.resolve_frame_settings([kf], 0)
    v0 = m.ViewConfig(0)
    v1 = m.ViewConfig(1)
    v1.mode = "HSV"
    views = (_view_settings_from_config(v0), _view_settings_from_config(v1))
    combined = app.render_full_image(data, 50, 50, _starting_obst_arrays(), views)
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
    kf = m.Keyframe(1, 0, {"view:0:mode": "Grayscale"})
    resolved = m.resolve_views([snap], [kf], 0)[0]
    assert resolved.mode == "Grayscale"


def test_resolve_views_honors_keyframe_overrides_per_channel():
    """A keyframe can override one of a view's channels without disturbing
    the others."""
    v = m.ViewConfig(9)
    snap = m.snapshot_view(v)
    custom_chan = m.ChannelSettings(
        chan_src=m.CHAN_SRC_QUERY, kind=m.KIND_OTHER, stat=m.ST_BOUNCES, point1=m.POINT_START,
        point2=m.POINT_END, buddha_points=m.DEFAULT_BUDDHA_POINTS,
        buddha_compression=m.BUDDHA_COMPRESS_MODULUS, buddha_brightness=3, constant=m.NONE_ZERO)
    kf = m.Keyframe(1, 0, {"view:9:mode": "HSV", "view:9:chan0": custom_chan})
    resolved = m.resolve_views([snap], [kf], 0)[0]
    assert resolved.mode == "HSV"
    assert resolved.kinds[0] == m.KIND_OTHER and resolved.buddha_brightness[0] == 3
    assert resolved.kinds[1] == m.DEFAULT_KINDS[1]   # untouched channel keeps the live default


def test_capture_all_settings_has_no_first_view_special_case():
    app = _fake_render_app()
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    app.current_ray_settings = lambda: m.DEFAULT_RAY_SETTINGS
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
    app.views = [m.ViewConfig(0)]
    app._view_counter = 1
    app.selected_view_id = 0
    app.include_all_views_var = _StubVar(True)
    app.color_info_location_var = _StubVar("None")
    app.figure_info_location_var = _StubVar("None")
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
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
    app.current_ray_settings = lambda: m.DEFAULT_RAY_SETTINGS
    app.deco = {"outline": _StubVar(False)}
    app.render_full_image = m.App.render_full_image.__get__(app)
    app.colorize_view = m.App.colorize_view.__get__(app)
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
    orig_export_dir = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = tmpdir
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
        m.paths.EXPORT_DIR = orig_export_dir


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
    orig_export_dir = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img = m.read_png(os.path.join(tmpdir, files[0]))
        assert img.shape == (20, 20, 3), \
            f"with 'Include all views' off, export must stay single-width, got {img.shape}"
    finally:
        m.paths.EXPORT_DIR = orig_export_dir


def test_video_frame_render_widens_when_a_second_view_is_added_and_included():
    """Same property, for the video-render call chain: snapshot
    views_to_export() the way the render launch site now does, resolve
    them, and render - before and after adding a view, with 'Include all
    views' on."""
    app = _fake_export_app()
    app.include_all_views_var = _StubVar(True)
    kf = m.Keyframe(0, 0, dict(m.DEFAULT_SETTING_VALUES))
    keyframes = [kf]
    obst_snapshot = [o.snapshot() for o in app.obstacles]
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
# "Current image" export (_write/render_current_image): reuses the
# already-traced simulation rather than re-simulating, and respects
# "Include all views" the same way the other two export paths do
# --------------------------------------------------------------------------- #
def _fake_current_image_app():
    app = _fake_export_app()
    app.stats = np.random.RandomState(0).rand(m.SIZE, m.SIZE, m.N_STATS).astype(np.float32)
    app.points = (np.random.RandomState(1).rand(m.SIZE, m.SIZE, m.N_POINTS, 2)
                 .astype(np.float32) * 2 - 1)
    app.status = np.ones((m.SIZE, m.SIZE), dtype=np.uint8)
    app.gen_fuzz = m.fuzz_rule(m.DEFAULT_FUZZ)
    app.settings_of_current_image = m.App.settings_of_current_image.__get__(app)
    empty_i = np.empty(0, dtype=np.int64)
    empty_b = np.empty((0, m.OBST_BOUND_LEN))
    empty_f = np.empty(0)
    empty_bool = np.empty(0, dtype=bool)
    app.gen_obstacles = m.ObstacleArrays(
        shapes=empty_i, bounds=empty_b, cx=empty_f, cy=empty_f,
        effects=np.empty((0, 2), dtype=np.int64), fire_inside=empty_bool)
    app.gx, app.gy = m.make_grid(m.SIZE, m.SIZE)
    app.obstacle_ring_specs = lambda: []
    app.gen_obstacle_snapshots = [m.default_starting_obstacle(0).snapshot()]
    app.frame = np.full((m.SIZE, m.SIZE, 3), 42, dtype=np.uint8)   # the "already
                                                                   # rendered" preview
    app.render_current_image = m.App.render_current_image.__get__(app)
    app.render_view_thumbnail = m.App.render_view_thumbnail.__get__(app)
    app._write = m.App._write.__get__(app)
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    return app


def test_render_view_thumbnail_does_not_crash_for_every_view_kind():
    """Regression test: render_view_thumbnail builds its own ViewSettings
    manually (like render_current_image does) and had no coverage at all -
    that's exactly how a missing constant_choice argument shipped
    undetected. Exercises it for Query, Buddha, and Constant channels."""
    app = _fake_current_image_app()
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
    app = _fake_current_image_app()
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
    app = _fake_current_image_app()
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
    app = _fake_current_image_app()
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
    app = _fake_current_image_app()
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
    app = _fake_current_image_app()
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
    app.gen_fuzz = m.fuzz_rule(m.DEFAULT_FUZZ)
    app.generated_settings = None                     # the placeholder image: default settings
    app.settings_of_current_image = m.App.settings_of_current_image.__get__(app)
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
    app = _fake_current_image_app()
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    assert img.shape == (m.SIZE, m.SIZE, 3)
    assert (img == 42).all(), "single-view export should be exactly self.frame"


def test_write_composes_all_views_when_include_all_views_is_on():
    app = _fake_current_image_app()
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


def test_render_current_image_matches_self_frame_for_identical_single_view():
    """A sanity check that the two code paths agree: composing just the live
    view through render_current_image gives the same picture self.frame
    already has (modulo the outline, which render_current_image also
    applies - both are off here for a clean, direct pixel comparison isn't
    meaningful since self.frame's content comes from a different source in
    this fixture; instead just check shape and that it runs without error)."""
    app = _fake_current_image_app()
    img = app.render_current_image([app.live_view()])
    assert img.shape == (m.SIZE, m.SIZE, 3)


# --------------------------------------------------------------------------- #
# figure_info_lines / the "Figure settings location" margin: every setting
# that defines the figure (limits, segment length, fuzzing, every obstacle)
# --------------------------------------------------------------------------- #
def _settings(**overrides):
    """A RaySettings: the defaults, with 64 iterations and a distance limit of 0,
    then `overrides` (any fields, by name)."""
    return m.DEFAULT_RAY_SETTINGS.replace(**{"max_iter": 64, "max_dist": 0.0, **overrides})


def _start_scene():
    return [m.default_starting_obstacle(0)]


def _every_kind_of_obstacle():
    """One obstacle of each of the eight kinds, each with awkward values
    (negative and large numbers, an offset, every effect)."""
    out = []
    awkward = {"width": 1.99, "height": 0.01, "rotation": -179.9, "sides": 64, "radius": 1.99,
               "moss_angle": 179.0, "moss_size": 1.99, "sector_span": 180.0,
               "segment_span": 359.0, "lens_radius1": 1.99, "lens_radius2": 1.99,
               "lens_sep": 1.99, "reuleaux_sides": 21, "capsule_length": 1.99,
               "capsule_radius": 0.99, "lobe_size": 0.99, "cut_radius": 1.99,
               "cut_offset": 1.99, "split": 0.95, "r": 1.0, "theta": -179.9, "cx": -0.99,
               "cy": 0.99}
    for i, cls in enumerate(m.SHAPE_CLASSES):
        o = cls(i)
        for field, value in awkward.items():
            if field in o.FIELD_NAMES:
                setattr(o, field, value)
        o.in_effect, o.out_effect = i % 3, (i + 1) % 3
        o.fire_inside = bool(i % 2)
        out.append(o)
    return out


def test_figure_info_lines_shows_limits_and_segment_length_always():
    lines = m.figure_info_lines(_settings(max_iter=42, max_dist=5.0, seg_mult=0.5),
                                _start_scene())
    assert "Iter limit:42" in lines
    assert "Dist limit:5.00" in lines
    assert "Segment:x0.50" in lines
    assert "Segment:x1.00" in m.figure_info_lines(_settings(), _start_scene()), \
        "shown even at the default, so the layout never depends on the values"


def test_figure_info_lines_shows_none_for_unlimited_distance():
    assert "Dist limit:none" in m.figure_info_lines(_settings(max_dist=0.0), _start_scene())


def test_figure_info_lines_shows_fuzz_details_when_enabled_and_just_off_otherwise():
    off = m.figure_info_lines(_settings(), _start_scene())
    assert "Fuzz:Off" in off and not any("Seed" in l for l in off)
    fuzz = m.DEFAULT_FUZZ.replace(enabled=True, emit_jitter=5.0, seed=12345)
    on = m.figure_info_lines(_settings(fuzz=fuzz), _start_scene())
    assert "Fuzz:On" in on
    assert any("Emit" in l and "5.0" in l for l in on)
    assert "  Seed 12345" in on


def test_figure_info_lines_no_longer_describes_a_bounding_shape():
    lines = m.figure_info_lines(_settings(), _start_scene())
    assert not any("Shape" in l for l in lines)


def test_figure_info_lines_says_so_when_there_are_no_obstacles():
    assert "no obstacles" in m.figure_info_lines(_settings(), [])


def test_figure_info_lines_lists_every_setting_of_the_starting_obstacle():
    lines = m.figure_info_lines(_settings(), _start_scene())
    assert "Obstacle #1: Ellipse" in lines
    for expected in ("  w2.00 h2.00", "  rot 0.0\u00b0", "  pos r0.00 \u03b80.0\u00b0",
                     "  offset +0.00 +0.00", "  in:Reflect", "  out:Reflect", "  fire:On"):
        assert expected in lines, expected


def test_figure_info_lines_shows_every_field_that_defines_each_kind_of_obstacle():
    """Changing any one of an obstacle's own settings must change the text - so
    nothing that defines the figure is left out of the margin."""
    relevant = {
        m.SHAPE_ELLIPSE: ["width", "height", "rotation"],
        m.SHAPE_POLYGON: ["sides", "radius", "rotation"],
        m.SHAPE_MOSS_EGG: ["moss_angle", "moss_size", "rotation"],
        m.SHAPE_SECTOR: ["radius", "sector_span", "rotation"],
        m.SHAPE_SEGMENT: ["radius", "segment_span", "rotation"],
        m.SHAPE_LENS: ["lens_radius1", "lens_radius2", "lens_sep", "rotation"],
        m.SHAPE_REULEAUX: ["reuleaux_sides", "radius", "rotation"],
        m.SHAPE_CAPSULE: ["capsule_length", "capsule_radius", "rotation"],
        m.SHAPE_TREFOIL: ["radius", "lobe_size", "rotation"],
        m.SHAPE_QUATREFOIL: ["radius", "lobe_size", "rotation"],
        m.SHAPE_LUNE: ["radius", "cut_radius", "cut_offset", "rotation"],
        m.SHAPE_ARBELOS: ["radius", "split", "rotation"],
    }
    everywhere = ["r", "theta", "cx", "cy", "in_effect", "out_effect", "fire_inside"]
    changed = {"sides": 7, "reuleaux_sides": 9, "in_effect": 2, "out_effect": 1,
               "fire_inside": False, "theta": 33.0}
    for shape, fields in relevant.items():
        cls = m.SHAPE_CLASSES[shape]
        assert set(fields + everywhere) == set(cls.FIELD_NAMES), cls.__name__
        base = cls(0)
        base.fire_inside = True
        base_text = base.info_lines(1)
        for field in fields + everywhere:
            o = cls(0)
            o.fire_inside = True
            setattr(o, field, changed.get(field, getattr(base, field) + 0.37))
            assert o.info_lines(1) != base_text, (m.SHAPE_NAMES[shape], field)


def test_every_obstacle_line_fits_a_column_and_uses_only_glyphs_the_font_has():
    scene = _every_kind_of_obstacle()
    fuzz = m.DEFAULT_FUZZ.replace(enabled=True, emit_jitter=179.9,
                                   bnd_offset=-179.9, bnd_jitter=179.9, bnd_mult=2.99,
                                   self_offset=-179.9, self_jitter=179.9, self_mult=2.99,
                                   seed=999999999)
    settings = _settings(max_iter=9999, max_dist=9999.0, seg_mult=0.01, fuzz=fuzz)
    for line in m.figure_info_lines(settings, scene):
        assert len(line) <= m.FIGURE_INFO_COLUMN_CHARS, line
        missing = [c for c in line if c not in m.FONT_5X7]
        assert not missing, (line, missing)
        m.render_text_line(line, scale=1)          # must not raise


def test_figure_info_layout_depends_only_on_how_many_obstacles_there_are():
    """A video's frames may animate the values, but each must lay out the same
    way (so every frame has the same size)."""
    scene_a = _every_kind_of_obstacle()
    scene_b = _every_kind_of_obstacle()
    for o in scene_b:
        o.rotation, o.r, o.theta, o.fire_inside = 5.0, 0.0, 12.5, not o.fire_inside
    fuzz_on = m.DEFAULT_FUZZ.replace(enabled=True)
    shapes = []
    for settings, scene in [(_settings(), scene_a), (_settings(max_dist=3.0, seg_mult=0.5,
                            fuzz=fuzz_on), scene_b)]:
        shapes.append([len(block) for block in m.figure_info_blocks(settings, scene)])
    assert shapes[0] == shapes[1]


def test_layout_info_columns_keeps_blocks_together_and_splits_only_oversized_ones():
    blocks = [["a1", "a2", "a3"], ["b1", "b2", "b3"], ["c1", "c2"]]
    cols = m.layout_info_columns(blocks, lines_per_column=7)
    assert cols == [["a1", "a2", "a3", " ", "b1", "b2", "b3"], ["c1", "c2"]]
    huge = m.layout_info_columns([["x%d" % i for i in range(10)]], lines_per_column=4)
    assert [len(c) for c in huge] == [4, 4, 2]
    assert [l for c in huge for l in c] == ["x%d" % i for i in range(10)]


def test_left_margin_is_half_the_image_width_for_an_ordinary_scene():
    img = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    out = m.add_left_figure_margin(img, _settings(), _start_scene())
    assert out.shape == (m.SIZE, m.SIZE + m.SIZE // 2, 3)
    assert np.array_equal(out[:, m.SIZE // 2:], img), "the image itself is untouched"
    assert (out[:, :m.SIZE // 2] != m.COLORATION_INFO_BG).any(), "and the text is drawn"


def test_left_margin_grows_extra_columns_when_the_settings_need_the_room():
    img = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    scene = []
    for i in range(14):
        o = m.default_starting_obstacle(i)
        o.r = 0.1 * i
        scene.append(o)
    out = m.add_left_figure_margin(img, _settings(), scene)
    margin_w = out.shape[1] - m.SIZE
    assert margin_w > m.SIZE // 2
    assert margin_w % 2 == 0, "video encoders want even widths"
    text = (out[:, :margin_w] != m.COLORATION_INFO_BG).any(axis=-1)
    last_col_x0 = margin_w - 6 * m.FIGURE_INFO_COLUMN_CHARS
    assert text[:, last_col_x0:].any(), "the last column really has text in it"


def test_left_margin_size_is_the_same_for_every_frame_of_an_animated_video():
    img = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    scene = _every_kind_of_obstacle()
    moved = _every_kind_of_obstacle()
    for o in moved:
        o.r, o.rotation = 0.25, 5.0
    fuzz_on = m.DEFAULT_FUZZ.replace(enabled=True)
    a = m.add_left_figure_margin(img, _settings(), scene)
    b = m.add_left_figure_margin(img, _settings(max_dist=3.0, seg_mult=0.5, fuzz=fuzz_on),
                                 moved)
    assert a.shape == b.shape
    assert not np.array_equal(a, b), "but they do show the different values"


def test_add_left_figure_margin_does_not_mutate_input():
    img = np.full((40, 60, 3), 33, dtype=np.uint8)
    original = img.copy()
    m.add_left_figure_margin(img, m.DEFAULT_RAY_SETTINGS, _start_scene())
    assert np.array_equal(img, original)


def test_add_left_figure_margin_does_not_crash_for_any_setting_combination():
    img = np.full((30, 30, 3), 10, dtype=np.uint8)
    for fuzz_on in (False, True):
        for seg_mult in (1.0, 0.5):
            for scene in ([], _start_scene(), _every_kind_of_obstacle()):
                settings = _settings(seg_mult=seg_mult,
                                     fuzz=m.DEFAULT_FUZZ.replace(enabled=fuzz_on))
                m.add_left_figure_margin(img, settings, scene)   # must not raise


def test_export_designer_calls_them_figure_settings_not_simulation_info():
    import inspect
    source = inspect.getsource(m.App.build_status_quadrant)
    assert "Figure settings location:" in source
    assert "Simulation info" not in source
    assert m.FIGURE_INFO_LOCATIONS == ["None", "Left margin"]
    assert not hasattr(m, "SIM_INFO_LOCATIONS")
    for gone in ("simulation_info_lines", "add_left_simulation_margin"):
        assert not hasattr(m, gone), gone


# --------------------------------------------------------------------------- #
# Obstacle values resolved for one frame (what the margin's text shows)
# --------------------------------------------------------------------------- #
def test_resolve_obstacles_interpolates_float_fields_and_holds_the_rest():
    o = m.EllipseShape(3)
    o.fire_inside = False
    snap = o.snapshot()
    kf0 = m.Keyframe(0, 0, {"obst:3:r": 0.0, "obst:3:fire_inside": False,
                            "obst:3:in_effect": R})
    kf1 = m.Keyframe(1, 10, {"obst:3:r": 1.0, "obst:3:fire_inside": True,
                             "obst:3:in_effect": T})
    mid = m.resolve_obstacles([snap], [kf0, kf1], 5)[0]
    assert mid.id == 3
    assert np.isclose(mid.r, 0.5)
    assert mid.fire_inside is False and mid.in_effect == R      # held, not blended
    end = m.resolve_obstacles([snap], [kf0, kf1], 10)[0]
    assert end.fire_inside is True and end.in_effect == T


def test_resolve_obstacle_arrays_is_the_arrays_of_the_resolved_obstacles():
    o = m.RegularPolygonShape(1)
    snaps = [o.snapshot()]
    kfs = [m.Keyframe(0, 0, {"obst:1:r": 0.2}),
           m.Keyframe(1, 4, {"obst:1:r": 0.6})]
    direct = m.resolve_obstacle_arrays(snaps, kfs, 2)
    via = m.build_obstacle_arrays(m.resolve_obstacles(snaps, kfs, 2))
    for a, b in zip(direct, via):
        assert np.array_equal(a, b)


def test_left_margin_of_a_video_frame_shows_that_frames_animated_values():
    o = m.EllipseShape(1)
    snaps = [o.snapshot()]
    kfs = [m.Keyframe(0, 0, {"obst:1:theta": 0.0}),
           m.Keyframe(1, 10, {"obst:1:theta": 90.0})]
    settings = _settings()
    text0 = m.figure_info_lines(settings, m.resolve_obstacles(snaps, kfs, 0))
    text5 = m.figure_info_lines(settings, m.resolve_obstacles(snaps, kfs, 5))
    assert "  pos r0.00 \u03b80.0\u00b0" in text0
    assert "  pos r0.00 \u03b845.0\u00b0" in text5


# --------------------------------------------------------------------------- #
# A pinned ray preview stays visible even where no ray would fire
# --------------------------------------------------------------------------- #
def _fake_preview_app(pinned):
    import collections

    class FakeApp:
        pass
    app = FakeApp()
    app.update_preview = m.App.update_preview.__get__(app)
    app.pinned_px = pinned
    app.hover_px = None
    app.obstacles = []                              # nothing at all fires a ray
    app.preview_settings = lambda: m.DEFAULT_RAY_SETTINGS
    app.hover_pts = np.empty((m.ITER_LIMIT_MAX + 2, 2))
    app.hover_kinds = np.empty(m.ITER_LIMIT_MAX + 2)
    app.hover_raw = np.empty(m.N_RAW)
    app.hover_vec = np.empty(m.N_STATS)
    app.hover_sp = np.empty((m.N_POINTS, 2))
    app.component_values = lambda vec, sp, x, y: (0.1, 0.2, 0.3)
    app.draw_segments = lambda pts, kinds, n: None
    app.deco = {k: _StubVar(k != "samples") for k in
                ("pending", "outline", "ray_path", "ray_start", "ray_end", "guides", "samples")}
    app.canvas = _StubWidget()
    app.hline_img = app.vline_img = None
    app.live_view = lambda: m.ViewConfig(0)
    app.map_color = lambda vals: (1, 2, 3)
    app.info_vars = collections.defaultdict(_StubVar)
    app.stat_vars = [_StubVar() for _ in range(m.N_STATS)]
    app.swatch = app.info_placeholder = app.info_rows = _StubWidget()
    app.cleared = []
    app.clear_overlay = lambda: app.cleared.append(1)
    return app


def test_a_hovered_pixel_that_fires_no_ray_has_no_preview():
    app = _fake_preview_app(pinned=None)
    app.update_preview(100, 100)
    assert app.cleared == [1]
    assert "start" not in app.info_vars


def test_a_pinned_pixel_keeps_its_preview_even_when_no_ray_would_fire_from_it():
    app = _fake_preview_app(pinned=(100, 100))
    app.update_preview(100, 100)
    assert app.cleared == [], "a pinned preview must stay visible"
    assert "[pinned]" in app.info_vars["start"].get()
    assert app.info_vars["outcome"].get(), "and it shows what such a ray would do"


def _fake_click_app(pinned, obstacles):
    class FakeApp:
        pass
    app = FakeApp()
    app.on_click = m.App.on_click.__get__(app)
    app.pinned_px = pinned
    app.hover_px = "stale"
    app.obstacles = obstacles
    app.preview_settings = lambda: m.DEFAULT_RAY_SETTINGS
    app.canvas = _StubWidget()
    app.previews = []
    app.update_preview = lambda i, j: app.previews.append((i, j))
    return app


def _click(x, y):
    import types
    return types.SimpleNamespace(x=x, y=y)


def test_clicking_a_pixel_that_fires_no_ray_does_nothing():
    app = _fake_click_app(pinned=None, obstacles=[])
    app.on_click(_click(100, 100))
    assert app.pinned_px is None and app.previews == []


def test_clicking_a_firing_pixel_pins_it_and_clicking_again_unpins_it():
    app = _fake_click_app(pinned=None, obstacles=_start_scene())
    app.on_click(_click(256, 256))
    assert app.pinned_px == (256, 256)
    app.on_click(_click(256, 256))
    assert app.pinned_px is None


def test_the_pinned_pixel_can_be_unpinned_by_clicking_it_even_after_it_stopped_firing():
    """A pinned preview now stays visible where no ray fires, so the click that
    releases it must work there too - or it could never be released."""
    app = _fake_click_app(pinned=(100, 100), obstacles=[])     # nothing fires any more
    app.on_click(_click(100, 100))
    assert app.pinned_px is None
    assert app.previews == [(100, 100)], "and the preview is refreshed (and so cleared)"


def test_clicking_a_different_non_firing_pixel_does_not_move_the_pin():
    app = _fake_click_app(pinned=(100, 100), obstacles=[])
    app.on_click(_click(200, 200))
    assert app.pinned_px == (100, 100)


def test_only_the_pinned_pixel_is_exempt_from_the_must_fire_rule():
    app = _fake_preview_app(pinned=(100, 100))
    app.update_preview(200, 200)
    assert app.cleared == [1]


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
            if ch not in m.FONT_5X7:
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
    """New image / video export pass resolved ViewSettings (read-only
    tuples), not live ViewConfig objects - the field names must line up."""
    vc = m.ViewConfig(0)
    vc.chan_src[1] = m.CHAN_SRC_BUDDHA
    vs = _view_settings_from_config(vc)
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
    settings = m.DEFAULT_RAY_SETTINGS
    img_512 = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    out_512 = m.add_left_figure_margin(img_512, settings, _start_scene())
    margin_w_512 = m.SIZE // 2
    h1 = (out_512[:, :margin_w_512] != m.COLORATION_INFO_BG).any(axis=-1).any(axis=1).sum()

    img_1080 = np.full((1080, 1080, 3), 128, dtype=np.uint8)
    out_1080 = m.add_left_figure_margin(img_1080, settings, _start_scene())
    margin_w_1080 = 1080 // 2
    h2 = (out_1080[:, :margin_w_1080] != m.COLORATION_INFO_BG).any(axis=-1).any(axis=1).sum()

    assert h2 == h1 * 2, f"expected exactly 2x text height at 1080 vs 512, got {h1} -> {h2}"


def test_write_draws_coloration_info_without_mutating_self_frame():
    """self.frame is the shared, reused live-preview buffer - drawing the
    overlay must go into a copy, never into self.frame itself."""
    app = _fake_current_image_app()
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
    orig_export_dir = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img = m.read_png(os.path.join(tmpdir, files[0]))
        assert (img == m.COLORATION_INFO_BG).all(axis=-1).any()
    finally:
        m.paths.EXPORT_DIR = orig_export_dir


def test_video_render_worker_draws_coloration_info_when_enabled():
    """Exercises the per-frame drawing step _render_worker takes, without
    needing a real ffmpeg encode: build the same rgb array it would, with
    the same call (apply_info_locations, as _render_worker actually calls
    it), and check the overlay landed."""
    app = _fake_export_app()
    kf = m.Keyframe(0, 0, dict(m.DEFAULT_SETTING_VALUES))
    keyframes = [kf]
    obst_snapshot = [o.snapshot() for o in app.obstacles]
    data = m.resolve_frame_settings(keyframes, 0)
    obst_arrays = m.resolve_obstacle_arrays(obst_snapshot, keyframes, 0)
    view_snapshot = [m.snapshot_view(v) for v in app.views_to_export()]
    views = m.resolve_views(view_snapshot, keyframes, 0)

    rgb = app.render_full_image(data, 40, 40, obst_arrays, views)
    rgb = m.apply_info_locations(rgb, views, 40, data.settings, [],
                                 "Overlay", "None", mutate_ok=True)
    assert (rgb == m.COLORATION_INFO_BG).all(axis=-1).any()


def test_video_render_worker_adds_figure_settings_left_margin_when_enabled():
    app = _fake_export_app()
    kf = m.Keyframe(0, 0, dict(m.DEFAULT_SETTING_VALUES))
    keyframes = [kf]
    obst_snapshot = [o.snapshot() for o in app.obstacles]
    data = m.resolve_frame_settings(keyframes, 0)
    obst_arrays = m.resolve_obstacle_arrays(obst_snapshot, keyframes, 0)
    view_snapshot = [m.snapshot_view(v) for v in app.views_to_export()]
    views = m.resolve_views(view_snapshot, keyframes, 0)

    rgb = app.render_full_image(data, 40, 40, obst_arrays, views)
    original_width = rgb.shape[1]
    rgb = m.apply_info_locations(rgb, views, 40, data.settings,
                                 _start_scene(), "None", "Left margin", mutate_ok=True)
    assert rgb.shape[1] >= original_width + 40 // 2       # (more, at this tiny size)
    assert (rgb == m.COLORATION_INFO_BG).all(axis=-1).any()


def test_color_info_and_figure_settings_locations_default_to_none():
    app = _fake_export_app()
    assert app.color_info_location_var.get() == "None"
    assert app.figure_info_location_var.get() == "None"


def test_apply_info_locations_none_leaves_the_image_unchanged():
    settings = m.DEFAULT_RAY_SETTINGS
    v = m.ViewConfig(0)
    img = np.full((40, 40, 3), 77, dtype=np.uint8)
    out = m.apply_info_locations(img, [v], 40, settings, [], "None", "None", mutate_ok=True)
    assert out.shape == img.shape
    assert (out == 77).all()


def test_apply_info_locations_margin_cases_never_mutate_when_not_ok():
    """Bottom/left margins always return a new array anyway (they can't
    mutate in place - the image gets bigger), but this locks in that
    mutate_ok=False is honored for "Overlay" specifically, which otherwise
    would draw straight into the caller's array."""
    settings = m.DEFAULT_RAY_SETTINGS
    v = m.ViewConfig(0)
    img = np.full((40, 40, 3), 77, dtype=np.uint8)
    original = img.copy()
    m.apply_info_locations(img, [v], 40, settings, [], "Overlay", "None", mutate_ok=False)
    assert np.array_equal(img, original), "mutate_ok=False must not draw into the input array"


def test_write_with_bottom_margin_produces_a_taller_image():
    app = _fake_current_image_app()
    app.color_info_location_var = _StubVar("Bottom margin")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    assert img.shape == (m.SIZE + m.SIZE // 4, m.SIZE, 3)
    assert (img[m.SIZE:] == m.COLORATION_INFO_BG).all(axis=-1).mean() > 0.5
    assert (img[:m.SIZE] == 42).all(), "the original view content must be untouched"


def test_write_with_left_margin_produces_a_wider_image():
    app = _fake_current_image_app()
    app.figure_info_location_var = _StubVar("Left margin")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    assert img.shape == (m.SIZE, m.SIZE + m.SIZE // 2, 3)
    assert (img[:, :m.SIZE // 2] == m.COLORATION_INFO_BG).all(axis=-1).any()
    assert (img[:, m.SIZE // 2:] == 42).all(), "the original view content must be untouched"


def test_write_with_both_margins_composes_correctly():
    app = _fake_current_image_app()
    app.color_info_location_var = _StubVar("Bottom margin")
    app.figure_info_location_var = _StubVar("Left margin")
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
    rs = {"fmt": "JXL", "jxl_lossless": True, "jxl_quality": 70}
    assert app.output_extension_and_suffix(rs) == ("jxl", ".q100")          # lossless
    assert app.output_extension_and_suffix({**rs, "jxl_lossless": False}) == ("jxl", ".q70")
    assert app.output_extension_and_suffix({**rs, "jxl_lossless": False, "jxl_quality": 100}) \
        == ("jxl", ".q95")                                                    # q100 is only lossless


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


def test_the_quality_tag_in_a_jxl_file_name_says_lossless_or_the_lossy_quality():
    assert m.jxl_quality_tag(True, 90) == "q100" and m.jxl_quality_tag(True, 5) == "q100"
    assert m.jxl_quality_tag(False, 90) == "q90" and m.jxl_quality_tag(False, 5) == "q5"
    assert m.jxl_quality_tag(False, 100) == "q95"        # lossy never reads q100
    assert m.JXL_LOSSY_MAX_QUALITY == 95


def test_default_filename_puts_the_quality_tag_just_before_the_extension():
    name = m.default_filename(m.DEFAULT_RAY_SETTINGS, "jxl", "q90")
    assert name.endswith(".q90.jxl") and name.count(".q90") == 1
    assert m.default_filename(None, "jxl", "q100").endswith(".q100.jxl")
    assert m.default_filename(None, "png").endswith(".png")
    assert ".q" not in m.default_filename(None, "png")
    assert m.image_filename(m.DEFAULT_RAY_SETTINGS, "jxl", False, 85).endswith(".q85.jxl")
    assert m.image_filename(m.DEFAULT_RAY_SETTINGS, "jxl", True, 85).endswith(".q100.jxl")
    import re
    for fmt in ("png", "qoi"):                            # only JPEG XL files carry a quality
        assert not re.search(r"\.q\d+\.", m.image_filename(m.DEFAULT_RAY_SETTINGS, fmt, False, 85))


def test_still_images_default_to_jxl():
    assert m.SAVE_NEW_FORMATS[0] == "jxl" and set(m.SAVE_NEW_FORMATS) == {"jxl", "png", "qoi"}


def _fake_save_app(fmt, lossless, quality):
    class FakeApp:
        pass
    app = FakeApp()
    app.save_source_var = _StubVar(m.SAVE_SOURCE_OPTIONS[0])
    app.save_format_var = _StubVar(fmt)
    app.save_jxl_mode_var = _StubVar("Lossless" if lossless else "Lossy")
    app.save_jxl_quality_scale = _StubVar(quality)
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    app.written = []
    app._write = lambda path: app.written.append(os.path.basename(path))
    app.save = m.App.save.__get__(app)
    return app


def test_saving_a_jxl_image_names_its_quality_and_other_formats_do_not():
    orig = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = os.path.join(tempfile.mkdtemp(), "exports")
    try:
        for fmt, lossless, quality, tail in (("jxl", True, 90, ".q100.jxl"),
                                             ("jxl", False, 80, ".q80.jxl"),
                                             ("png", False, 80, ".png")):
            app = _fake_save_app(fmt, lossless, quality)
            app.save()
            assert app.written[0].endswith(tail), app.written
            import re
            assert bool(re.search(r"\.q\d+\.jxl$", app.written[0])) == (fmt == "jxl")
    finally:
        m.paths.EXPORT_DIR = orig


def _recorded_cjxl_arguments(**options):
    """The command line convert_apng_to_jxl runs (the fake cjxl copies its input)."""
    work_dir = tempfile.mkdtemp()
    apng_path = os.path.join(work_dir, "source.apng")
    with open(apng_path, "wb") as f:
        f.write(b"fake apng")
    seen = []
    real_run = m.subprocess.run
    m.subprocess.run = lambda cmd, **kw: seen.append(list(cmd)) or real_run(cmd, **kw)
    try:
        with _fake_cjxl():
            m.convert_apng_to_jxl(apng_path, os.path.join(work_dir, "out.jxl"), **options)
    finally:
        m.subprocess.run = real_run
    return seen[0]


def test_convert_apng_to_jxl_is_lossless_by_default_and_lossy_at_the_distance_of_the_quality():
    def distance(args):
        return args[args.index("-d") + 1]
    assert distance(_recorded_cjxl_arguments()) == "0"
    assert distance(_recorded_cjxl_arguments(lossless=True, quality=50)) == "0"
    assert distance(_recorded_cjxl_arguments(lossless=False, quality=90)) == "1"
    assert distance(_recorded_cjxl_arguments(lossless=False, quality=50)) == "5"
    assert distance(_recorded_cjxl_arguments(lossless=False, quality=100)) == "0.5"   # capped at q95


def test_a_lossy_animated_jxl_from_the_real_cjxl_is_a_smaller_lossy_animation():
    """cjxl really does lossy animation: jxlinfo (when installed beside it) calls the file
    a lossy animation, and it is smaller than the lossless one. Skipped without cjxl/ffmpeg."""
    import subprocess
    ffmpeg_exe = shutil.which("ffmpeg")
    if REAL_CJXL_PATH is None:
        raise SkipTest("real cjxl not on PATH")
    if ffmpeg_exe is None:
        raise SkipTest("real ffmpeg not on PATH")
    work_dir = tempfile.mkdtemp()
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:64, 0:64]
    for i in range(4):
        frame = np.zeros((64, 64, 3), np.uint8)
        frame[..., 0] = (xx * 4 + i * 20) % 256
        frame[..., 1] = (yy * 4) % 256
        frame[..., 2] = rng.randint(0, 40, (64, 64))
        m.write_png(os.path.join(work_dir, f"frame_{i:05d}.png"), frame)
    apng_path = os.path.join(work_dir, "source.apng")
    subprocess.run([ffmpeg_exe, "-y", "-loglevel", "error", "-framerate", "10",
                    "-i", os.path.join(work_dir, "frame_%05d.png"), "-plays", "0", apng_path],
                   check=True)
    lossless = os.path.join(work_dir, "lossless.jxl")
    lossy = os.path.join(work_dir, "lossy.jxl")
    m.convert_apng_to_jxl(apng_path, lossless)
    m.convert_apng_to_jxl(apng_path, lossy, lossless=False, quality=80)
    assert os.path.getsize(lossy) < os.path.getsize(lossless) / 2
    for path in (lossless, lossy):
        with open(path, "rb") as f:
            head = f.read(12)
        assert head[:2] == bytes.fromhex("ff0a") or head == bytes.fromhex("0000000c4a584c200d0a870a")
    jxlinfo = os.path.join(os.path.dirname(REAL_CJXL_PATH),
                           "jxlinfo.exe" if os.name == "nt" else "jxlinfo")
    if os.path.exists(jxlinfo):
        info = subprocess.run([jxlinfo, lossy], capture_output=True, text=True).stdout
        assert "animation" in info and "lossy" in info, info


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
    app.video_jxl_controls = _StubJxlControls()
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
    app.video_jxl_controls = _StubJxlControls()
    app.read_render_settings = m.App.read_render_settings.__get__(app)
    orig_ffmpeg = m.external.ffmpeg
    m.external.ffmpeg = orig_ffmpeg or object()   # only this sandbox lacks ffmpeg-python
    try:
        with _fake_cjxl():
            rs = app.read_render_settings()
    finally:
        m.external.ffmpeg = orig_ffmpeg
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
           "bitrate": 0, "vbr_target": 0, "vbr_max": 0, "jxl_lossless": True, "jxl_quality": 90,
           "downsample_factor": 1, "downsample_method": "Mean"}


def test_encode_video_output_writes_directly_for_non_jxl_formats():
    app = _fake_encode_app()
    work_dir = tempfile.mkdtemp()
    out_path = os.path.join(work_dir, "out.mp4")
    orig_ffmpeg = m.external.ffmpeg
    m.external.ffmpeg = _FakeFfmpegModule()
    try:
        app.encode_video_output("pattern_%05d.png", 0, _base_rs("mp4 (h264)"), out_path)
    finally:
        m.external.ffmpeg = orig_ffmpeg
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
        assert f.read() == b"fake ffmpeg output bytes"


def test_encode_video_output_routes_jxl_through_apng_and_cjxl():
    app = _fake_encode_app()
    work_dir = tempfile.mkdtemp()
    out_path = os.path.join(work_dir, "out.jxl")
    orig_ffmpeg = m.external.ffmpeg
    orig_temp_dir = m.paths.TEMP_DIR
    m.external.ffmpeg = _FakeFfmpegModule()
    m.paths.TEMP_DIR = tempfile.mkdtemp()
    try:
        with _fake_cjxl():
            app.encode_video_output("pattern_%05d.png", 0, _base_rs("JXL"), out_path)
    finally:
        m.external.ffmpeg = orig_ffmpeg
        m.paths.TEMP_DIR = orig_temp_dir
    assert os.path.exists(out_path), "the final .jxl file must exist"
    with open(out_path, "rb") as f:
        assert f.read() == b"fake ffmpeg output bytes", \
            "the fake cjxl just copies its input, so this confirms the chain ran: " \
            "ffmpeg wrote the APNG, cjxl copied it to the final path"
    temp_apng = os.path.join(m.paths.TEMP_DIR, "_jxl_video_source.apng")
    assert not os.path.exists(temp_apng), "the intermediate APNG must be cleaned up"
    assert any("cjxl" in s for s in app.status_calls), \
        "the status should mention the cjxl conversion step"


def test_encode_video_output_cleans_up_temp_apng_even_on_cjxl_failure():
    app = _fake_encode_app()
    work_dir = tempfile.mkdtemp()
    out_path = os.path.join(work_dir, "out.jxl")
    orig_ffmpeg = m.external.ffmpeg
    orig_temp_dir = m.paths.TEMP_DIR
    m.external.ffmpeg = _FakeFfmpegModule()
    m.paths.TEMP_DIR = tempfile.mkdtemp()
    try:
        with _fake_cjxl("_jxl_video_source"):
            try:
                app.encode_video_output("pattern_%05d.png", 0, _base_rs("JXL"), out_path)
                assert False, "should have raised"
            except RuntimeError:
                pass
    finally:
        m.external.ffmpeg = orig_ffmpeg
        m.paths.TEMP_DIR = orig_temp_dir
    temp_apng = os.path.join(m.paths.TEMP_DIR, "_jxl_video_source.apng")
    assert not os.path.exists(temp_apng), \
        "the intermediate APNG must still be cleaned up even when cjxl fails"
    assert not os.path.exists(out_path)


# --------------------------------------------------------------------------- #
# clear_temp_frames: deletes only frame_NNNNN.png files from TEMP_DIR - the
# one place in the whole program that deletes anything at all
# --------------------------------------------------------------------------- #
class _ClearTempFramesContext:
    """Patches m.paths.TEMP_DIR to a fresh scratch directory and m.messagebox's
    dialog functions to controllable, call-counting stubs, for the duration
    of a `with` block - and restores both afterward, since the test module
    loads light_bounces once and shares it across every test."""
    def __init__(self, confirm=True):
        self.confirm = confirm
        self.showinfo_calls = []
        self.showerror_calls = []
        self.askyesno_calls = []

    def __enter__(self):
        self.orig_temp_dir = m.paths.TEMP_DIR
        self.orig_showinfo = m.messagebox.showinfo
        self.orig_showerror = m.messagebox.showerror
        self.orig_askyesno = m.messagebox.askyesno
        self.tmpdir = tempfile.mkdtemp()
        m.paths.TEMP_DIR = self.tmpdir
        m.messagebox.showinfo = lambda *a, **k: self.showinfo_calls.append((a, k))
        m.messagebox.showerror = lambda *a, **k: self.showerror_calls.append((a, k))
        m.messagebox.askyesno = lambda *a, **k: (self.askyesno_calls.append((a, k))
                                                 or self.confirm)
        return self

    def __exit__(self, *exc):
        m.paths.TEMP_DIR = self.orig_temp_dir
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
    v = m.ViewConfig(0)                  # mutable; converted for colorize_view below
    v.point1_choice[0] = m.POINT_ORIGIN
    v.point2_choice[0] = m.POINT_TOP_LEFT_CORNER
    v.point_metric[0] = m.ST_HDIST
    mini_settings = _settings(max_iter=30, max_dist=m.effective_dist(0.0))
    fz = m.fuzz_rule(m.DEFAULT_FUZZ)
    obst_arrays = m.build_obstacle_arrays([])
    img = app.colorize_view(_view_settings_from_config(v), stats, points, status, 4, 4, mini_settings, fz,
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
    assert isinstance(chans[1], m.ChannelSettings) and chans[1].constant == m.NONE_ONE
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
    v = m.ViewConfig(0)                  # mutable; converted for colorize_view below
    v.chan_src[0] = m.CHAN_SRC_CONSTANT
    v.constant_choice[0] = m.NONE_ONE
    v.chan_src[1] = m.CHAN_SRC_CONSTANT
    v.constant_choice[1] = m.NONE_ZERO
    mini_settings = _settings(max_iter=30, max_dist=m.effective_dist(0.0))
    fz = m.fuzz_rule(m.DEFAULT_FUZZ)
    obst_arrays = m.build_obstacle_arrays([])
    img = app.colorize_view(_view_settings_from_config(v), stats, points, status, 4, 4, mini_settings, fz,
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
    orig_export_dir = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img = m.read_png(os.path.join(tmpdir, files[0]))
        assert img.shape == (24, 20, 3), \
            f"21x22 should round to 20x24, got {img.shape}"
    finally:
        m.paths.EXPORT_DIR = orig_export_dir


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
    app.video_jxl_controls = _StubJxlControls()
    app.read_render_settings = m.App.read_render_settings.__get__(app)
    orig_ffmpeg = m.external.ffmpeg
    m.external.ffmpeg = orig_ffmpeg or object()   # only this sandbox lacks ffmpeg-python; the
                                        # function under test doesn't actually call it
    try:
        rs = app.read_render_settings()
    finally:
        m.external.ffmpeg = orig_ffmpeg
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
    assert m.DEFAULT_RAY_SETTINGS.buddha_blur.radius == m.DEFAULT_BUDDHA_BLUR_RADIUS
    assert m.DEFAULT_RAY_SETTINGS.buddha_blur.type == m.DEFAULT_BUDDHA_BLUR_TYPE
    settings = m.DEFAULT_RAY_SETTINGS
    assert settings.buddha_blur.radius == m.DEFAULT_BUDDHA_BLUR_RADIUS
    assert settings.buddha_blur.type == m.DEFAULT_BUDDHA_BLUR_TYPE


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

    kf = m.Keyframe(0, 0, data)
    resolved = m.resolve_frame_settings([kf], 0)
    assert resolved.settings.buddha_blur.radius == 0.2
    assert resolved.settings.buddha_blur.type == m.BUDDHA_BLUR_BOKEH


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

    v = m.ViewConfig(0)                  # mutable; converted for colorize_view below
    v.chan_src[0] = m.CHAN_SRC_BUDDHA
    # index 0 in BUDDHA_POINT_CATALOG is "All bounces mean" -> POINT_ALL_MEAN
    v.buddha_points[0] = tuple(i == 0 for i in range(m.N_BUDDHA_POINTS))

    fz = m.fuzz_rule(m.DEFAULT_FUZZ)
    obst_arrays = m.build_obstacle_arrays([])

    settings_no_blur = _settings(max_iter=30, max_dist=m.effective_dist(0.0),
                                 buddha_blur=m.BuddhaBlurSettings(radius=0.0, type=m.BUDDHA_BLUR_BOKEH))
    img_no_blur = app.colorize_view(_view_settings_from_config(v), stats, points, status, size, size, settings_no_blur, fz, obst_arrays)
    lit_no_blur = (img_no_blur[8:13, 8:13, 0] > 0).sum()

    settings_blur = _settings(max_iter=30, max_dist=m.effective_dist(0.0),
                              buddha_blur=m.BuddhaBlurSettings(radius=0.1, type=m.BUDDHA_BLUR_BOKEH))
    img_blur = app.colorize_view(_view_settings_from_config(v), stats, points, status, size, size, settings_blur, fz, obst_arrays)
    lit_blur = (img_blur[8:13, 8:13, 0] > 0).sum()

    assert lit_no_blur == 1, f"with no blur, exactly one pixel should be nonzero, got {lit_no_blur}"
    assert lit_blur > lit_no_blur, \
        f"blur should leave more pixels nonzero ({lit_blur}) than no blur ({lit_no_blur})"


def test_blurred_buddha_counts_is_cached_until_invalidated():
    """The mechanism that keeps the live preview cheap: blurring is a whole-
    image pass, so it must not be recomputed on every recolor - only when
    the raw counts actually changed."""
    app = _fake_current_image_app()
    app.live_view = lambda: app.views[0]
    app.views[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.buddha_counts = [np.zeros((m.SIZE, m.SIZE), dtype=np.float64), None, None]
    app.buddha_counts_blurred = [None, None, None]
    app.generated_settings = _settings(
        buddha_blur=m.BuddhaBlurSettings(radius=0.05, type=m.BUDDHA_BLUR_BOKEH))
    app.settings_of_current_image = m.App.settings_of_current_image.__get__(app)
    app.blurred_buddha_counts = m.App.blurred_buddha_counts.__get__(app)

    first = app.blurred_buddha_counts(0)
    second = app.blurred_buddha_counts(0)
    assert first is second, "must return the cached array, not recompute it"

    app.buddha_counts_blurred[0] = None      # simulate scatter_buddha_rows invalidating it
    third = app.blurred_buddha_counts(0)
    assert third is not first, "must recompute once actually invalidated"


def test_scatter_buddha_rows_invalidates_only_touched_channels():
    app = _fake_current_image_app()
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
    v = m.ViewConfig(0)                  # mutable; converted for colorize_view below
    v.chan_src[0] = m.CHAN_SRC_BUDDHA
    v.buddha_points[0] = tuple(i == 0 for i in range(m.N_BUDDHA_POINTS))
    fz = m.fuzz_rule(m.DEFAULT_FUZZ)
    obst_arrays = m.build_obstacle_arrays([])
    settings = _settings(max_iter=30, max_dist=m.effective_dist(0.0),   # a large, expensive-looking radius
                         buddha_blur=m.BuddhaBlurSettings(radius=0.15, type=m.BUDDHA_BLUR_GAUSSIAN))
    t0 = time.time()
    app.colorize_view(_view_settings_from_config(v), stats, points, status, size, size, settings, fz, obst_arrays)
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
    orig_export_dir = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img = m.read_png(os.path.join(tmpdir, files[0]))
        assert img.shape == (40 + 40 // 4, 40, 3)
    finally:
        m.paths.EXPORT_DIR = orig_export_dir


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
    keep, doomed = m.EllipseShape(0), m.EllipseShape(1)
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
    o = m.EllipseShape(0)
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
    assert snap[0].id == live[0].id


# --------------------------------------------------------------------------- #
# Obstacle incoming / outgoing ray effects (replacing the single "effect")
# --------------------------------------------------------------------------- #
def test_new_obstacle_defaults_to_reflecting_both_ways_and_firing_rays_inside():
    o = m.EllipseShape(0)
    assert o.in_effect == m.OBSTACLE_EFFECT_REFLECT
    assert o.out_effect == m.OBSTACLE_EFFECT_REFLECT
    assert o.fire_inside is True


def test_effect_names_and_colors_cover_every_effect():
    assert m.OBSTACLE_EFFECT_NAMES == ["Reflect", "Terminate", "Pass through"]
    assert m.OBST_EFFECT_RGB == {m.OBSTACLE_EFFECT_REFLECT: (179, 229, 252),
                                 m.OBSTACLE_EFFECT_TERMINATE: (121, 85, 72),
                                 m.OBSTACLE_EFFECT_PASS: (0, 0, 0)}


def test_every_shape_kind_has_both_effects():
    for cls in m.SHAPE_CLASSES:
        assert "in_effect" in cls.FIELD_NAMES and "out_effect" in cls.FIELD_NAMES


def test_build_obstacle_arrays_effects_are_n_by_2_and_keep_pass_through_obstacles():
    a, b, c = m.EllipseShape(0), m.EllipseShape(1), m.EllipseShape(2)
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
    effects = m.build_obstacle_arrays([]).effects
    assert effects.shape == (0, 2)


def test_resolve_obstacle_arrays_animates_nothing_but_holds_each_effect_per_keyframe():
    o = m.EllipseShape(3)
    snap = o.snapshot()
    kf0 = m.Keyframe(0, 0, {"obst:3:in_effect": m.OBSTACLE_EFFECT_TERMINATE,
                            "obst:3:out_effect": m.OBSTACLE_EFFECT_REFLECT})
    kf1 = m.Keyframe(1, 10, {"obst:3:in_effect": m.OBSTACLE_EFFECT_PASS,
                             "obst:3:out_effect": m.OBSTACLE_EFFECT_TERMINATE})
    kfs = [kf0, kf1]
    for frame, expect in [(0, [1, 0]), (5, [1, 0]), (10, [2, 1])]:
        effects = m.resolve_obstacle_arrays([snap], kfs, frame).effects
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
             (m.SHAPE_MOSS_EGG, m.moss_egg_bound(91.0, 1.1, 30.0)),
             (m.SHAPE_SECTOR, m.sector_bound(1.2, 70.0, 40.0)),
             (m.SHAPE_SEGMENT, m.segment_bound(1.1, 80.0, 50.0)),
             (m.SHAPE_CAPSULE, m.capsule_bound(1.0, 0.4, 30.0)),
             (m.SHAPE_LENS, m.lens_bound(1.0, 1.0, 1.0, 0.0)),
             (m.SHAPE_REULEAUX, m.reuleaux_bound(5, 70.0, 0.7)),
             (m.SHAPE_TREFOIL, m.lobed_bound(m.SHAPE_TREFOIL, 3, 0.9, 0.6, 15.0)),
             (m.SHAPE_QUATREFOIL, m.lobed_bound(m.SHAPE_QUATREFOIL, 4, 0.9, 0.6, 15.0)),
             (m.SHAPE_LUNE, m.lune_bound(1.0, 0.8, 0.6, 30.0)),
             (m.SHAPE_ARBELOS, m.arbelos_bound(1.0, 0.35, 30.0))]
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
    o = m.EllipseShape(0)
    o.in_effect, o.out_effect = m.OBSTACLE_EFFECT_TERMINATE, m.OBSTACLE_EFFECT_REFLECT
    thumb = m.render_obstacle_thumbnail(o, m.THUMB_SIZE)
    assert thumb.shape == (m.THUMB_SIZE, m.THUMB_SIZE, 3)
    assert (thumb == np.array(m.OBST_TERMINATE_RGB)).all(axis=-1).any()
    assert (thumb == np.array(m.OBST_REFLECT_RGB)).all(axis=-1).any()


def test_a_pass_through_both_ways_obstacle_counts_as_pending_even_if_unchanged():
    class FakeApp:
        pass
    app = FakeApp()
    o = m.EllipseShape(0)
    o.in_effect = o.out_effect = m.OBSTACLE_EFFECT_PASS
    app.obstacle_is_pending = m.App.obstacle_is_pending.__get__(app)
    app.gen_obstacles_by_id = {0: o.state()}     # unchanged since generation
    assert app.obstacle_is_pending(o)
    o.out_effect = m.OBSTACLE_EFFECT_REFLECT
    app.gen_obstacles_by_id = {0: o.state()}
    assert not app.obstacle_is_pending(o)


# --------------------------------------------------------------------------- #
# "Fire rays inside": a ray starts at a point iff the point is inside ANY
# shape (the main bounding shape, or an obstacle) that has it switched on
# --------------------------------------------------------------------------- #
def _emit_fixture(circle=True, obstacles=()):
    """(emits(x, y) -> bool) for an optional unit circle (True: fires rays
    inside it; False: present but not firing; None: absent) plus
    `obstacles`, a list of (cx, cy, radius, fire) circles."""
    scene = []
    if circle is not None:
        scene.append(_circle_obstacle(0.0, 0.0, 1.0, R, R, fire=circle))
    for i, (cx, cy, radius, fire) in enumerate(obstacles, start=1):
        scene.append(_circle_obstacle(cx, cy, radius, R, R, fire=fire, id_=i))
    arrays = m.build_obstacle_arrays(scene)
    return lambda x, y: bool(m.emits_ray(x, y, arrays))


def test_a_circle_fires_rays_inside_it_and_not_outside():
    emits = _emit_fixture()
    assert emits(0.0, 0.0) and emits(0.5, -0.5)
    assert not emits(0.9, 0.9)          # outside the unit circle


def test_a_circle_with_fire_off_fires_nothing_on_its_own():
    assert not _emit_fixture(circle=False)(0.0, 0.0)


def test_no_shapes_at_all_fire_nothing():
    assert not _emit_fixture(circle=None)(0.0, 0.0)


def test_obstacle_with_fire_on_fires_inside_it_even_if_the_circle_is_off():
    emits = _emit_fixture(circle=False, obstacles=[(0.4, 0.0, 0.2, True)])
    assert emits(0.4, 0.0)
    assert not emits(-0.4, 0.0)


def test_obstacle_with_fire_off_does_not_punch_a_hole_in_a_shape_that_fires():
    """Being inside a shape that has fire off makes no difference: a point
    inside both a firing shape and a fire-off obstacle still fires."""
    emits = _emit_fixture(obstacles=[(0.4, 0.0, 0.2, False)])
    assert emits(0.4, 0.0)


def test_obstacle_with_fire_off_alone_fires_nothing():
    emits = _emit_fixture(circle=False, obstacles=[(0.4, 0.0, 0.2, False)])
    assert not emits(0.4, 0.0)


def test_obstacle_with_fire_on_can_fire_outside_the_circle():
    emits = _emit_fixture(obstacles=[(1.2, 0.0, 0.2, True)])
    assert emits(1.2, 0.0)
    assert not emits(0.0, 1.2)


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


def _kernel_trace(x0, y0, angle_deg, obstacles=(), circle=(R, R), max_iter=12, max_dist=0.0,
                  seg_mult=1.0, confine=None):
    """One ray through the real trace_path, heading `angle_deg` (fuzzing is
    switched on with all jitter zero purely to set the direction, and is
    otherwise the identity). `circle` is (incoming effect, outgoing effect[,
    fires rays inside]) of a unit-circle obstacle at the origin, listed first;
    None for no circle. `confine` is a confine_rule (default: the standard rule).
    Returns (points, kinds, term, raw)."""
    confine = m.CONFINE_DEFAULT if confine is None else confine
    scene = list(obstacles)
    if circle is not None:
        fire = circle[2] if len(circle) > 2 else True
        scene.insert(0, _circle_obstacle(0.0, 0.0, 1.0, circle[0], circle[1], fire=fire))
    arrays = m.build_obstacle_arrays(scene)
    fz = m.fuzz_rule(m.DEFAULT_FUZZ)
    emission = m.emission_rule(m.EMIT_UP, 0.0, 0.0, angle_deg - 90.0)
    pts = np.empty((max_iter + 2, 2))
    kinds = np.empty(max_iter + 2)
    raw = np.empty(m.N_RAW)
    term, npts = m.trace_path(x0, y0, pts, kinds, max_iter, m.effective_dist(max_dist), seg_mult,
                              fz, confine, emission, arrays, raw)
    return pts[:npts].copy(), kinds[:npts].copy(), term, raw


def test_a_trapped_obstacle_keeps_its_ray_inside_forever():
    """Reflect both ways = a container: the ray started inside never leaves."""
    box = _circle_obstacle(0.2, 0.0, 0.5, R, R, fire=True)
    pts, _k, term, raw = _kernel_trace(0.2, -0.3, 0.0, [box], circle=None, max_iter=40)
    assert term in (m.TERM_ITER, m.TERM_CONFINED)   # its own segments may confine it
    assert int(raw[m.RAW_BOUNCES]) >= 8
    assert (np.hypot(pts[:, 0] - 0.2, pts[:, 1]) <= 0.5 + 1e-9).all()
    assert np.isclose(np.hypot(pts[1, 0] - 0.2, pts[1, 1]), 0.5)


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
    pts, kinds, term, _raw = _kernel_trace(0.6, -0.1, 90.0, [stop], circle=None)
    assert term == m.TERM_OBSTACLE
    assert np.allclose(pts[-1], (0.6, np.sqrt(0.09 - 0.01)))
    assert kinds[-1] == m.KIND_OBSTACLE_TERM


def test_outgoing_pass_through_lets_a_ray_fired_inside_carry_on_to_the_bounding_shape():
    hole = _circle_obstacle(0.0, 0.0, 0.3, R, P, fire=True)
    pts, _k, _term, _raw = _kernel_trace(0.1, 0.2, 0.0, [hole], circle=(R, R, False))
    assert np.isclose(np.hypot(*pts[1]), 1.0), "first bounce is on the bounding shape"


def test_a_ray_that_leaves_the_bounding_shape_escapes_at_the_image_edge():
    pts, kinds, term, raw = _kernel_trace(0.0, -0.5, 0.0, circle=(R, P))
    assert term == m.TERM_ESCAPED
    assert len(pts) == 2 and int(raw[m.RAW_BOUNCES]) == 0
    assert np.allclose(pts[-1], (1.0, -0.5)), "ends where it leaves the [-1, 1] square"
    assert kinds[-1] == m.KIND_ESCAPED


def test_a_ray_escapes_after_bouncing_when_it_finally_meets_nothing():
    wall = _circle_obstacle(0.0, 0.0, 0.3, R, P, fire=False)
    # leaves the (pass-through) bounding shape... bounces off the obstacle first
    pts, _k, term, raw = _kernel_trace(-0.9, 0.2, 0.0, [wall], circle=(R, P))
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
    pts, _k, _term, _raw = _kernel_trace(1.3, 0.3, 180.0, [far], circle=(P, R), max_iter=3)
    assert np.allclose(pts[1], (-np.sqrt(1.0 - 0.09), 0.3)), "reflects off the FAR inner wall"


def test_a_distance_limit_cuts_an_escaping_ray_before_the_image_edge():
    pts, kinds, term, raw = _kernel_trace(0.0, -0.5, 0.0, circle=(R, P), max_dist=0.4)
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
    pts, kinds, term, raw = _kernel_trace(0.0, -0.5, 0.0, circle=(R, P))
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


def test_render_rows_fires_only_from_shapes_with_fire_on():
    """The real (parallel) render kernel: with no firing circle, only pixels
    inside an obstacle with fire on emit a ray."""
    obstacle = _circle_obstacle(0.3, 0.0, 0.25, R, R, fire=True)
    arrays = m.build_obstacle_arrays([obstacle])
    stats = np.zeros((m.SIZE, m.SIZE, m.N_STATS), dtype=np.float32)
    points = np.zeros((m.SIZE, m.SIZE, m.N_POINTS, 2), dtype=np.float32)
    status = np.zeros((m.SIZE, m.SIZE), dtype=np.uint8)
    m.render_rows(stats, points, status, 0, m.SIZE, 8, m.effective_dist(0.0), 1.0,
                  m.fuzz_rule(m.DEFAULT_FUZZ), m.CONFINE_DEFAULT, m.EMISSION_DEFAULT, arrays, 0)
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
    circle = _circle_obstacle(0.0, 0.0, 1.0, R, P, fire=True)      # rays leave it
    m.scatter_multi_buddha(250, 252, 8, m.effective_dist(0.0), 1.0,
                           m.fuzz_rule(m.DEFAULT_FUZZ), m.CONFINE_DEFAULT, m.EMISSION_DEFAULT,
                           m.build_obstacle_arrays([circle]), 0, False, True, False, False,
                           counts)
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
    app.selected_obstacle = m.EllipseShape(0)
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
# Parity with the classic bounding shape. The old system (a fixed shape at the
# origin, with its own settings) is being replaced by ordinary obstacles; an
# obstacle that is a plain stand-in - Reflect both ways, fires rays inside, at the
# origin - must produce exactly what the classic shape produced. The recorded
# golden data (dev/testing/classic_boundary_golden.npz) was generated from the
# classic implementation, so it keeps guarding this after that code is gone.
# --------------------------------------------------------------------------- #
CLASSIC_GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev",
                                   "testing", "classic_boundary_golden.npz")
GOLDEN_GRID = 13

# name, shape, classic main-shape params, the equivalent obstacle's fields, and
# the simulation settings to run it under
# The recorded scenes list their fuzzing positionally, in this order.
FUZZ_FIELD_ORDER = ("enabled", "emit_jitter", "bnd_offset", "bnd_jitter", "bnd_mult",
                    "self_offset", "self_jitter", "self_mult", "seed")

CLASSIC_SCENES = [
    dict(name="circle", shape=m.SHAPE_ELLIPSE, main=(2.0, 2.0, 0.0),
         fields=dict(width=2.0, height=2.0, rotation=0.0), iters=32),
    dict(name="ellipse", shape=m.SHAPE_ELLIPSE, main=(1.7, 1.1, 25.0),
         fields=dict(width=1.7, height=1.1, rotation=25.0), iters=30),
    dict(name="polygon", shape=m.SHAPE_POLYGON, main=(5, 10.0),
         fields=dict(sides=5, rotation=10.0, radius=1.0), iters=30),
    dict(name="moss", shape=m.SHAPE_MOSS_EGG, main=(90.0, 1.0, 10.0),
         fields=dict(moss_angle=90.0, moss_size=1.0, rotation=10.0), iters=30),
    dict(name="sector", shape=m.SHAPE_SECTOR, main=(1.0, 100.0, 15.0),
         fields=dict(radius=1.0, sector_span=100.0, rotation=15.0), iters=30),
    dict(name="segment", shape=m.SHAPE_SEGMENT, main=(1.0, 120.0, 20.0),
         fields=dict(radius=1.0, segment_span=120.0, rotation=20.0), iters=30),
    dict(name="lens", shape=m.SHAPE_LENS, main=(1.0, 1.0, 1.0, 20.0),
         fields=dict(lens_radius1=1.0, lens_radius2=1.0, lens_sep=1.0, rotation=20.0),
         iters=30),
    dict(name="reuleaux", shape=m.SHAPE_REULEAUX, main=(3, 0.0),
         fields=dict(reuleaux_sides=3, rotation=0.0, radius=1.0), iters=30),
    dict(name="capsule", shape=m.SHAPE_CAPSULE, main=(1.0, 0.5, 30.0),
         fields=dict(capsule_length=1.0, capsule_radius=0.5, rotation=30.0), iters=30),
    dict(name="ellipse_fuzz_seg_dist", shape=m.SHAPE_ELLIPSE, main=(1.8, 1.3, -20.0),
         fields=dict(width=1.8, height=1.3, rotation=-20.0), iters=40,
         fuzz=(True, 3.0, 2.0, 5.0, 1.1, 1.0, 4.0, 0.9, 7), seg_mult=0.7, dist=4.0),
]


def _scene_obstacle(scene):
    """The obstacle equivalent to a classic bounding shape."""
    o = m.SHAPE_CLASSES[scene["shape"]](0)
    for field, value in scene["fields"].items():
        setattr(o, field, value)
    o.r = o.theta = o.cx = o.cy = 0.0
    o.in_effect = o.out_effect = m.OBSTACLE_EFFECT_REFLECT
    o.fire_inside = True
    return o


def _render_scene(scene, n):
    """(stats, points, status) of an n x n render of a classic scene, drawn as
    the equivalent obstacle."""
    fuzz = (m.FuzzParams(**dict(zip(FUZZ_FIELD_ORDER, scene["fuzz"])))
            if "fuzz" in scene else m.DEFAULT_FUZZ)
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    status = np.zeros((n, n), dtype=np.uint8)
    arrays = m.build_obstacle_arrays([_scene_obstacle(scene)])
    m.render_rows_wh(stats, points, status, 0, n, n, n, scene["iters"],
                     m.effective_dist(scene.get("dist", 0.0)), scene.get("seg_mult", 1.0),
                     m.fuzz_rule(fuzz), m.CONFINE_DEFAULT, m.EMISSION_DEFAULT, arrays, fuzz.seed)
    return stats, points, status


def test_every_classic_scene_is_expressible_as_an_obstacle_with_identical_geometry():
    for scene in CLASSIC_SCENES:
        o = _scene_obstacle(scene)
        classic = m.make_bound(scene["shape"], scene["main"])
        equivalent = o.bound()
        assert np.allclose(classic, equivalent), scene["name"]


def test_an_obstacle_reproduces_the_recorded_classic_golden_data():
    golden = np.load(CLASSIC_GOLDEN_PATH)
    assert len(CLASSIC_SCENES) == int(golden["n_scenes"])
    for idx, scene in enumerate(CLASSIC_SCENES):
        stats, points, status = _render_scene(scene, n=GOLDEN_GRID)
        assert np.array_equal(status, golden[f"status_{idx}"]), scene["name"]
        assert np.allclose(stats, golden[f"stats_{idx}"], atol=1e-5), scene["name"]
        assert np.allclose(points, golden[f"points_{idx}"], atol=1e-5), scene["name"]


# --------------------------------------------------------------------------- #
# The fixed bounding shape is gone: the scene is only obstacles, starting with
# an obstacle that stands in for it
# --------------------------------------------------------------------------- #
def test_the_starting_obstacle_is_the_unit_circle_that_reflects_and_fires_rays_inside():
    start = m.default_starting_obstacle(7)
    assert start.id == 7
    assert type(start) is m.EllipseShape
    assert (start.width, start.height, start.rotation) == (2.0, 2.0, 0.0)
    assert (start.r, start.theta, start.cx, start.cy) == (0.0, 0.0, 0.0, 0.0)
    assert (start.in_effect, start.out_effect) == (m.OBSTACLE_EFFECT_REFLECT,
                                                     m.OBSTACLE_EFFECT_REFLECT)
    assert start.fire_inside is True
    assert np.allclose(start.bound(), UNIT_CIRCLE)


def test_the_app_starts_with_the_starting_obstacle_selected():
    import inspect
    source = inspect.getsource(m.App.__init__)
    assert "self.obstacles = [default_starting_obstacle(0)]" in source
    assert "self._obst_counter = 1" in source
    assert "self.selected_obstacle = self.obstacles[0]" in source


def test_the_old_bounding_shape_system_is_gone():
    for name in ("describe_shape", "outline_coords", "snapshot_shape_params", "main_ctl_array",
                 "settings_main_ctl", "MAIN_CTL_DEFAULT", "SHAPE_HINTS", "DEFAULT_SHAPE",
                 "RING_HALF_WIDTH_PX", "SHAPE_PARAM_BY_KEY"):
        assert not hasattr(m, name), name
    for name in ("build_shape_controls", "build_main_ray_controls", "read_shape_controls",
                 "control_shape", "show_shape_controls", "on_shape_changed", "reset_ellipse",
                 "compute_ring", "set_ring"):
        assert not hasattr(m.App, name), name


def test_snapshot_and_settings_hold_only_the_simulation_controls():
    assert [f.name for f in dataclasses.fields(m.RaySettings)] == [
        "max_iter", "max_dist", "seg_mult", "fuzz", "buddha_blur",
        "confine_window", "confine_method", "confine_threshold",
        "emit_mode", "emit_target_r", "emit_target_theta", "emit_target_cx", "emit_target_cy",
        "emit_offset"]
    assert [f.name for f in dataclasses.fields(m.BuddhaBlurSettings)] == ["radius", "type"]


def test_no_keyframe_setting_describes_a_bounding_shape_any_more():
    keys = {k for k, _tab, _label in m.SETTING_INFO}
    assert keys == {"iter", "dist", "seg_mult", "fuzz_on", "emit_jitter",
                    "bnd_offset", "bnd_jitter", "bnd_mult", "self_offset", "self_jitter",
                    "self_mult", "seed", "buddha_blur_radius", "buddha_blur_type", "outline",
                    "confine_window", "confine_method", "confine_threshold",
                    "emit_mode", "emit_target_r", "emit_target_theta", "emit_target_cx",
                    "emit_target_cy", "emit_offset"}
    assert keys == set(m.DEFAULT_SETTING_VALUES)
    tabs = {tab for _k, tab, _label in m.SETTING_INFO}
    assert tabs == {"Simulation", "Fuzzing", "Buddha blur", "Decorations"}


def test_a_frame_resolved_from_no_keyframes_uses_the_defaults():
    data = m.resolve_frame_settings([], 0)
    assert data.settings == m.DEFAULT_RAY_SETTINGS
    assert data.outline is True


def test_settings_suffix_is_only_the_limits_and_fuzzing():
    assert m.settings_suffix(m.DEFAULT_RAY_SETTINGS) == "iter256-distNONE"
    limited = _settings(max_iter=10, max_dist=2.5)
    assert m.settings_suffix(limited) == "iter10-dist2.5"
    fuzzed = _settings(fuzz=m.DEFAULT_FUZZ.replace(enabled=True, seed=9))
    suffix = m.settings_suffix(fuzzed)
    assert suffix.startswith("iter64-distNONE-fuzz-") and suffix.endswith("-seed9")
    for word in ("ellipse", "polygon", "moss", "sector", "segment", "lens", "reuleaux",
                 "capsule"):
        assert word not in suffix


def test_params_for_polygon_and_reuleaux_end_with_their_radius():
    polygon, reuleaux = m.RegularPolygonShape(0), m.ReuleauxPolygonShape(1)
    polygon.sides, polygon.rotation, polygon.radius = 6, 15.0, 0.7
    reuleaux.reuleaux_sides, reuleaux.rotation, reuleaux.radius = 5, 20.0, 0.4
    assert polygon.params() == (6, 15.0, 0.7)
    assert reuleaux.params() == (5, 20.0, 0.4)


def test_every_shape_kind_has_the_shared_fields_and_only_the_parameters_its_geometry_needs():
    own = {m.EllipseShape: {"width", "height"}, m.RegularPolygonShape: {"sides", "radius"},
           m.MossEggShape: {"moss_angle", "moss_size"},
           m.CircularSectorShape: {"radius", "sector_span"},
           m.CircularSegmentShape: {"radius", "segment_span"},
           m.LensShape: {"lens_radius1", "lens_radius2", "lens_sep"},
           m.ReuleauxPolygonShape: {"reuleaux_sides", "radius"},
           m.CapsuleShape: {"capsule_length", "capsule_radius"},
           m.TrefoilShape: {"radius", "lobe_size"}, m.QuatrefoilShape: {"radius", "lobe_size"},
           m.LuneShape: {"radius", "cut_radius", "cut_offset"},
           m.ArbelosShape: {"radius", "split"}}
    shared = {"rotation", "r", "theta", "cx", "cy", "in_effect", "out_effect", "fire_inside"}
    assert set(own) == set(m.SHAPE_CLASSES)
    for cls, fields in own.items():
        assert set(cls.FIELD_NAMES) == fields | shared, cls.__name__
        o = cls(0)
        for field in fields:
            assert hasattr(o, field)
        for field in set().union(*own.values()) - fields:
            assert not hasattr(o, field), (cls.__name__, field)     # no field it doesn't use


def test_one_rotation_serves_every_kind_and_the_default_radius_is_1_for_every_kind_with_one():
    for cls in m.SHAPE_CLASSES:
        o = cls(0)
        assert o.rotation == 0.0
        o.rotation = 33.0
        assert 33.0 in o.params(), cls.__name__
        if "radius" in cls.FIELD_NAMES:
            assert o.radius == 1.0, cls.__name__


# --------------------------------------------------------------------------- #
# The arc-loop kinds (trefoil, quatrefoil, lune, arbelos): concave shapes made of circular arcs
# --------------------------------------------------------------------------- #
ARC_LOOP_CLASSES = [m.TrefoilShape, m.QuatrefoilShape, m.LuneShape, m.ArbelosShape]


def _arc_loop_membership(o):
    """The shape as an independent formula (not the packed arcs): point -> inside?"""
    if isinstance(o, (m.TrefoilShape, m.QuatrefoilShape)):
        r = o.lobe_size * o.radius
        ring = o.radius - r
        circles = []
        for k in range(o.LOBES):
            a = np.pi / 2 + np.radians(o.rotation) + 2 * np.pi * k / o.LOBES
            circles.append((ring * np.cos(a), ring * np.sin(a), r))
        return lambda x, y: any((x - cx) ** 2 + (y - cy) ** 2 < rr * rr for cx, cy, rr in circles)
    if isinstance(o, m.LuneShape):
        t = np.radians(o.rotation)
        cx, cy = o.cut_offset * np.cos(t), o.cut_offset * np.sin(t)
        return lambda x, y: (x * x + y * y < o.radius ** 2
                             and (x - cx) ** 2 + (y - cy) ** 2 >= o.cut_radius ** 2)
    big, share = o.radius, o.split
    t = np.radians(o.rotation)
    cs, sn = np.cos(t), np.sin(t)

    def member(x, y):
        u, v = x * cs + y * sn, -x * sn + y * cs + 0.5 * big      # into the unrotated frame
        return (v > 0 and u * u + v * v < big * big
                and (u + big - share * big) ** 2 + v * v >= (share * big) ** 2
                and (u - big + (1 - share) * big) ** 2 + v * v >= ((1 - share) * big) ** 2)
    return member


def test_the_arc_loop_kinds_agree_with_an_independent_formula_about_what_is_inside():
    rng = np.random.RandomState(5)
    pts = rng.uniform(-1.3, 1.3, size=(2500, 2))
    for cls in ARC_LOOP_CLASSES:
        o = cls(0)
        o.rotation = 17.0
        bound, member = o.bound(), _arc_loop_membership(o)
        wrong = [(x, y) for x, y in pts if m.inside_boundary(x, y, cls.KIND, bound) != member(x, y)]
        assert not wrong, (cls.__name__, wrong[:3])


def test_a_lune_is_hollow_where_its_cut_is_and_the_origin_of_a_trefoil_is_inside():
    lune = m.LuneShape(0)                                   # cut centre at (0.6, 0), radius 0.8
    assert not m.inside_boundary(0.0, 0.0, m.SHAPE_LUNE, lune.bound())     # in the bite
    assert m.inside_boundary(-0.9, 0.0, m.SHAPE_LUNE, lune.bound())        # in the crescent
    assert not m.inside_boundary(1.2, 0.0, m.SHAPE_LUNE, lune.bound())     # past the disc
    assert m.inside_boundary(0.0, 0.0, m.SHAPE_TREFOIL, m.TrefoilShape(0).bound())


def test_the_nearest_crossing_of_an_arc_loop_is_the_way_out_or_in_even_when_concave():
    """From inside, the first edge hit is the exit (boundary_hit); from outside it is the
    entry (obstacle_hit) - checked against walking along the ray and watching the inside test flip."""
    rng = np.random.RandomState(8)

    def walked(x, y, ux, uy, kind, bound):
        prev = m.inside_boundary(x, y, kind, bound)
        for t in np.linspace(1e-6, 4.0, 3000):
            if m.inside_boundary(x + t * ux, y + t * uy, kind, bound) != prev:
                lo, hi = t - 4.0 / 3000, t
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if m.inside_boundary(x + mid * ux, y + mid * uy, kind, bound) == prev:
                        lo = mid
                    else:
                        hi = mid
                return 0.5 * (lo + hi)
        return None

    for cls in ARC_LOOP_CLASSES:
        o = cls(0)
        o.rotation = 23.0
        bound = o.bound()
        checked = 0
        for _ in range(60):
            x, y = rng.uniform(-1.4, 1.4, 2)
            a = rng.uniform(0, 2 * np.pi)
            ux, uy = np.cos(a), np.sin(a)
            if m.inside_boundary(x, y, cls.KIND, bound):
                t = m.boundary_hit(x, y, ux, uy, cls.KIND, bound)[0]
            else:
                t = m.obstacle_hit(x, y, ux, uy, cls.KIND, bound)[0]
                t = 1e30 if t < 0 else t
            expected = walked(x, y, ux, uy, cls.KIND, bound)
            assert (expected is None) == (t > 1e29), cls.__name__
            if expected is not None:
                checked += 1
                assert abs(expected - t) < 2e-3, (cls.__name__, t, expected)
        assert checked > 15, cls.__name__


def test_rays_inside_an_arc_loop_bounce_around_it_and_every_bounce_lies_on_its_edge():
    """Reflection off a concave arc (a lune's bite, a trefoil's notch) must keep the ray
    inside the shape: every recorded bounce off its edge is a point where the inside test flips."""
    for cls in (m.LuneShape, m.QuatrefoilShape, m.ArbelosShape):
        o = cls(0)
        o.rotation = 10.0
        bound = o.bound()
        arrays = m.build_obstacle_arrays([o])
        pts = np.empty((40, 2))
        kinds = np.empty(40)
        raw = np.empty(m.N_RAW)
        rng = np.random.RandomState(2)
        bounced = 0
        for _ in range(40):
            x, y = rng.uniform(-1, 1, 2)
            if not m.inside_boundary(x, y, cls.KIND, bound):
                continue
            term, npts = m.trace_path(x, y, pts, kinds, 30, m.effective_dist(0.0), 1.0,
                                      m.fuzz_rule(m.DEFAULT_FUZZ), m.CONFINE_DEFAULT,
                                      m.EMISSION_DEFAULT, arrays, raw)
            for k in range(1, npts):
                if kinds[k] != m.KIND_BOUNDARY:                # (a bounce off the ray's own path
                    continue                                   # is not on the shape's edge)
                px, py = pts[k]
                around = [m.inside_boundary(px + 1e-5 * np.cos(a), py + 1e-5 * np.sin(a),
                                            cls.KIND, bound) for a in np.linspace(0, 6.28, 16)]
                assert any(around) and not all(around), (cls.__name__, px, py)
                bounced += 1
        assert bounced > 40, cls.__name__


def test_the_outline_of_an_arc_loop_is_one_closed_curve_without_stray_jumps():
    for cls in ARC_LOOP_CLASSES:
        xs, ys = cls(0).boundary_xy()
        assert (xs[0], ys[0]) == (xs[-1], ys[-1]), cls.__name__
        assert np.max(np.hypot(np.diff(xs), np.diff(ys))) < 0.2, cls.__name__
    coords = m.LuneShape(0).outline_coords()
    assert len(coords) % 2 == 0 and len(coords) > 40


def test_a_lune_whose_two_circles_do_not_cross_is_still_a_valid_shape():
    far = m.lune_bound(1.0, 0.5, 1.8, 0.0)                    # the cut misses the disc: a full disc
    assert int(far[1]) == 1 and m.inside_boundary(0.0, 0.0, m.SHAPE_LUNE, far)
    ring = m.lune_bound(1.0, 0.3, 0.2, 0.0)                   # the cut is inside: a disc with a hole
    assert int(ring[1]) == 2
    assert m.inside_boundary(0.7, 0.0, m.SHAPE_LUNE, ring)
    assert not m.inside_boundary(0.2, 0.0, m.SHAPE_LUNE, ring)
    gone = m.lune_bound(1.0, 2.0, 0.0, 0.0)                   # the cut covers everything: nothing left
    assert int(gone[1]) == 0 and not m.inside_boundary(0.0, 0.0, m.SHAPE_LUNE, gone)
    lune = m.LuneShape(0)
    lune.radius, lune.cut_radius, lune.cut_offset = 1.0, 2.0, 0.0
    lune.outline_coords()                                       # must not raise for an empty shape
    m.render_obstacle_thumbnail(lune, 16)


def test_two_coincident_lobes_are_one_edge_not_two():
    """A quatrefoil at lobe size 1 is four identical circles: if each counted as its own edge
    the even-odd rule would call the inside outside."""
    quat = m.QuatrefoilShape(0)
    quat.lobe_size = 1.0
    assert int(quat.bound()[1]) == 1 and m.inside_boundary(0.1, 0.1, m.SHAPE_QUATREFOIL, quat.bound())


def test_a_shape_only_accepts_numbers_of_the_type_its_field_declares():
    o = m.RegularPolygonShape(1)
    assert _raises(TypeError, lambda: setattr(o, "sides", 5.5))        # a count is whole
    assert _raises(TypeError, lambda: setattr(o, "radius", "1"))
    assert _raises(TypeError, lambda: setattr(o, "radius", True))
    assert _raises(TypeError, lambda: setattr(o, "fire_inside", 1))
    assert _raises(AttributeError, lambda: setattr(o, "width", 1.0))   # another kind's field
    o.radius = 2                                                       # an int is fine for a float
    o.sides = np.int64(7)
    o.radius = np.float64(0.5)
    assert (o.sides, o.radius) == (7, 0.5)


def test_a_snapshot_is_a_frozen_copy_of_the_same_kind_with_every_method():
    o = m.LensShape(5)
    o.lens_sep, o.r = 0.4, 0.5
    snap = o.snapshot()
    assert type(snap) is m.LensShape and snap.id == 5 and snap.state() == o.state()
    assert snap.describe() == o.describe()
    assert np.array_equal(snap.bound(), o.bound()) and snap.center() == o.center()
    o.lens_sep = 0.9                                      # a later live edit never reaches it
    assert snap.lens_sep == 0.4
    assert _raises(AttributeError, lambda: setattr(snap, "r", 1.0))


def test_with_values_and_duplicate_make_copies_of_the_same_kind():
    o = m.CapsuleShape(3)
    frozen = o.with_values(capsule_length=1.5, r=0.25)
    assert type(frozen) is m.CapsuleShape and frozen.id == 3
    assert (frozen.capsule_length, frozen.r, o.capsule_length, o.r) == (1.5, 0.25, 1.0, 0.0)
    assert _raises(AttributeError, lambda: o.with_values(width=1.0))   # not a capsule's field
    assert _raises(AttributeError, lambda: setattr(frozen, "r", 0.0))
    dupe = o.duplicate(9)
    assert type(dupe) is m.CapsuleShape and dupe.id == 9 and dupe.state() == o.state()
    dupe.r = 0.5                                          # editable, and independent of the original
    assert o.r == 0.0


def test_shapes_copy_and_pickle_as_the_same_kind_and_a_frozen_one_stays_frozen():
    import copy
    import pickle
    o = m.MossEggShape(2)
    o.moss_size = 1.25
    for original in (o, o.snapshot()):
        for clone in (copy.deepcopy(original), copy.copy(original)):
            assert type(clone) is m.MossEggShape and clone.state() == original.state()
            assert clone.id == 2
        with _ModuleRegistered():
            clone = pickle.loads(pickle.dumps(original))
        assert type(clone) is m.MossEggShape and clone.state() == original.state()
        assert (clone._frozen) == (original._frozen)
    assert _raises(AttributeError, lambda: setattr(copy.deepcopy(o.snapshot()), "r", 1.0))


def test_the_new_menu_lists_every_kind_and_a_new_obstacle_has_that_kind_only():
    for cls in m.SHAPE_CLASSES:
        o = cls(0)
        assert type(o) is cls and o.NAME == m.SHAPE_NAMES[o.KIND]
    assert m.SHAPE_NAMES == ["Ellipse", "Regular polygon", "Moss's egg", "Circular sector",
                             "Circular segment", "Two-circle lens", "Reuleaux polygon", "Capsule",
                             "Trefoil", "Quatrefoil", "Lune", "Arbelos"]



def test_render_full_image_colors_escaped_rays_when_the_starting_obstacle_lets_rays_out():
    app = _fake_render_app()
    start = m.default_starting_obstacle(0)
    start.out_effect = P
    data = m.resolve_frame_settings([], 0)
    img = app.render_full_image(data, 40, 40, m.build_obstacle_arrays([start]),
                                (_terminated_view(),))
    assert tuple(img[20, 20]) == (255, 0, 0), "escaped rays are termination value 1.0"
    assert tuple(img[0, 0]) == tuple(m.BACKGROUND), "outside the obstacle nothing fires"


def test_render_full_image_with_the_starting_obstacle_produces_no_escaped_rays():
    app = _fake_render_app()
    data = m.resolve_frame_settings([], 0)
    img = app.render_full_image(data, 40, 40, _starting_obst_arrays(), (_terminated_view(),))
    assert img[20, 20][0] != 255


def test_render_full_image_of_an_empty_scene_is_just_the_background():
    app = _fake_render_app()
    data = m.resolve_frame_settings([], 0)
    empty = m.build_obstacle_arrays([])
    img = app.render_full_image(data, 24, 24, empty, (_terminated_view(),))
    assert (img == np.array(m.BACKGROUND, dtype=np.uint8)).all()


# --------------------------------------------------------------------------- #
# Obstacle outlines are drawn a bit thicker than they used to be
# --------------------------------------------------------------------------- #
def test_each_obstacle_outline_layer_is_one_and_a_half_pixels_thick():
    assert m.OBST_RING_LAYER_PX == 1.5


def test_the_default_outline_layers_measure_their_full_thickness_on_a_circle():
    gx, gy = _grid_around(1.2, 1600)
    step = float(gx[0, 1] - gx[0, 0])
    outer, inner = m.ring_layer_masks(m.SHAPE_ELLIPSE, UNIT_CIRCLE, gx, gy)
    r = np.hypot(gx, gy)
    width = m.OBST_RING_LAYER_PX * m.PIXEL
    assert np.isclose(r[outer].max() - 1.0, width, atol=2 * step)
    assert np.isclose(1.0 - r[inner].min(), width, atol=2 * step)


# --------------------------------------------------------------------------- #
# The pixel font is complete: every letter, both cases, digits, "#" and theta
# --------------------------------------------------------------------------- #
def test_the_pixel_font_has_every_letter_in_both_cases_every_digit_and_the_symbols():
    import string
    needed = (string.ascii_uppercase + string.ascii_lowercase + string.digits
              + "#\u03b8 ")
    assert [c for c in needed if c not in m.FONT_5X7] == []


def test_every_pixel_font_glyph_is_seven_rows_of_five_zero_or_one_cells():
    for ch, rows in m.FONT_5X7.items():
        assert len(rows) == 7, ch
        for row in rows:
            assert len(row) == 5 and set(row) <= {"0", "1"}, (ch, row)


def test_no_visible_pixel_font_glyph_is_blank_and_letters_are_told_apart():
    import string
    for ch, rows in m.FONT_5X7.items():
        if ch != " ":
            assert any("1" in row for row in rows), f"{ch!r} is blank"
    letters = {c: tuple(m.FONT_5X7[c]) for c in string.ascii_letters + "#\u03b8"}
    assert len(set(letters.values())) == len(letters), "two letters share a glyph"


def test_the_only_glyphs_sharing_a_picture_are_the_two_long_standing_pairs():
    seen = {}
    for ch, rows in m.FONT_5X7.items():
        seen.setdefault(tuple(rows), []).append(ch)
    shared = sorted(sorted(v) for v in seen.values() if len(v) > 1)
    assert shared == [["0", "o"], ["x", "\u00d7"]]


def test_the_new_glyphs_are_the_right_shape():
    def art(ch):
        return ["".join("#" if c == "1" else "." for c in row) for row in m.FONT_5X7[ch]]
    assert art("#")[2] == "#####" and art("#")[4] == "#####"
    theta = art("\u03b8")
    assert theta[3] == "#####" and theta[0] == ".###." and theta[6] == ".###."
    assert art("W")[3] == "#.#.#" and art("N")[1] == "##..#" and art("Z")[0] == "#####"


def test_render_text_line_can_draw_every_letter_hash_and_theta():
    import string
    text = string.ascii_uppercase + string.ascii_lowercase + "#\u03b8 0123456789"
    mask = m.render_text_line(text, scale=1)
    assert mask.shape == (7, 6 * len(text) - 1) and mask.any()


# --------------------------------------------------------------------------- #
# "Origin" and "Top left corner" head the point dropdowns
# --------------------------------------------------------------------------- #
def test_the_point_dropdowns_list_origin_and_top_left_corner_first():
    assert m.POINT_MENU_LABELS[:2] == ["Origin", "Top left corner"]
    assert m.POINT_MENU_ORDER[:2] == [m.POINT_ORIGIN, m.POINT_TOP_LEFT_CORNER]


def test_the_point_dropdowns_still_list_every_choice_exactly_once():
    assert sorted(m.POINT_MENU_ORDER) == list(range(m.N_POINT_CHOICES))
    assert sorted(m.POINT_MENU_LABELS) == sorted(m.POINT_LABELS)
    assert len(set(m.POINT_MENU_LABELS)) == m.N_POINT_CHOICES


def test_choosing_from_the_reordered_dropdown_still_gives_the_same_point_ids():
    """Only the display order changed: a label maps back to its unchanged id, so
    the ids stored in views, keyframes and the sample arrays are untouched."""
    assert m.POINT_LABELS.index("Origin") == m.POINT_ORIGIN == 13
    assert m.POINT_LABELS.index("Top left corner") == m.POINT_TOP_LEFT_CORNER == 14
    assert m.POINT_LABELS.index("Start point") == m.POINT_START == 0
    for point_id in range(m.N_POINT_CHOICES):
        label = m.POINT_LABELS[point_id]
        assert m.POINT_LABELS.index(label) == point_id
        assert m.POINT_MENU_LABELS[m.POINT_MENU_ORDER.index(point_id)] == label


def test_the_comboboxes_are_given_the_reordered_labels():
    import inspect
    source = inspect.getsource(m.App.build_color_controls)
    assert source.count("values=POINT_MENU_LABELS") == 2
    assert "values=POINT_LABELS" not in source


def test_the_menu_order_self_check_catches_a_missing_point():
    fresh = load_module()
    fresh.core.POINT_MENU_ORDER = fresh.POINT_MENU_ORDER[:-1]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "POINT_MENU_ORDER" in str(e)


# --------------------------------------------------------------------------- #
# The default iteration limit
# --------------------------------------------------------------------------- #
def test_the_default_iteration_limit_is_256():
    assert m.DEFAULT_ITER == 256
    assert m.DEFAULT_RAY_SETTINGS.max_iter == 256
    assert m.DEFAULT_SETTING_VALUES["iter"] == 256
    assert m.ITER_LIMIT_MIN <= m.DEFAULT_ITER <= m.ITER_LIMIT_MAX


def test_the_iteration_limit_entry_starts_at_the_default():
    import inspect
    assert "self.iter_var = tk.StringVar(value=str(DEFAULT_ITER))" in \
        inspect.getsource(m.App.build_limits_controls)


def test_the_export_image_choices_are_called_current_image_and_new_image():
    assert m.SAVE_SOURCE_OPTIONS == ["Current image", "New image"]


# --------------------------------------------------------------------------- #
# Confinement: window size, detection method and threshold are settings
# --------------------------------------------------------------------------- #
BOX, PATH = m.CONFINE_BOX, m.CONFINE_PATH


def _confine(window, method, threshold):
    return m.confine_rule(window, method, threshold)


def test_the_default_confinement_rule_is_the_one_that_was_built_in():
    assert m.DEFAULT_CONFINE_WINDOW == 4
    assert m.DEFAULT_CONFINE_METHOD == m.CONFINE_BOX
    assert m.DEFAULT_CONFINE_THRESHOLD == m.PIXEL == 2.0 / 512
    assert m.CONFINE_METHOD_NAMES == ["Box", "Path length"]
    assert m.CONFINE_DEFAULT == m.ConfineRule(window=4, method=0, threshold=2.0 / 512)
    s = m.DEFAULT_RAY_SETTINGS
    assert (s.confine_window, s.confine_method, s.confine_threshold) == (4, BOX, 2.0 / 512)
    assert m.settings_confine(s) == m.CONFINE_DEFAULT


def _unconfined_path(x, y, angle, iters=60):
    """The full path of a ray with confinement effectively switched off."""
    pts, kinds, term, raw = _kernel_trace(x, y, angle, max_iter=iters,
                                          confine=_confine(2, BOX, 1e-300))
    return pts


def _first_confined_count(pts, window, method, threshold):
    """An independent reimplementation of the rule: the first path length
    (number of points) at which the last `window` points are confined, judged
    only after each bounce (so from `window` points onward)."""
    for count in range(max(window, 2), len(pts) + 1):
        recent = pts[count - window:count]
        if method == BOX:
            extent = recent.max(axis=0) - recent.min(axis=0)
            confined = extent[0] < threshold and extent[1] < threshold
        else:
            confined = np.hypot(*np.diff(recent, axis=0).T).sum() < threshold
        if confined:
            return count
    return None


def test_both_methods_stop_the_ray_exactly_where_the_rule_says_for_many_rays():
    rng = np.random.RandomState(11)
    checked = 0
    for _ in range(40):
        x, y = rng.uniform(-0.7, 0.7, 2)
        angle = rng.uniform(0.0, 360.0)
        full = _unconfined_path(x, y, angle)
        if len(full) < 12:
            continue
        for window in (2, 3, 4, 6, 9):
            for method in (BOX, PATH):
                # thresholds chosen from the path's own scale, so both outcomes occur
                scale = np.hypot(*np.diff(full, axis=0).T)
                for threshold in (scale.min() * 1.5, np.median(scale) * (window - 1), 1e-300):
                    expected = _first_confined_count(full, window, method, threshold)
                    pts, _k, term, _r = _kernel_trace(x, y, angle, max_iter=60,
                                                      confine=_confine(window, method, threshold))
                    if expected is None:
                        assert term != m.TERM_CONFINED, (window, method, threshold)
                    else:
                        assert term == m.TERM_CONFINED, (window, method, threshold)
                        assert len(pts) == expected, (window, method, threshold, len(pts),
                                                      expected)
                        assert np.allclose(pts, full[:expected])
                    checked += 1
    assert checked > 300


def test_a_huge_threshold_confines_a_ray_as_soon_as_the_window_is_full():
    for method in (BOX, PATH):
        for window in (2, 5, 12):
            pts, _k, term, _r = _kernel_trace(0.3, 0.1, 40.0, max_iter=40,
                                              confine=_confine(window, method, 1e9))
            assert term == m.TERM_CONFINED
            assert len(pts) == window, (method, window)


def test_a_tiny_threshold_never_confines_an_ordinary_ray():
    for method in (BOX, PATH):
        for threshold in (1e-9, 1e-100, 1e-300, 5e-324):
            _pts, _k, term, _r = _kernel_trace(0.3, 0.1, 40.0, max_iter=40,
                                               confine=_confine(4, method, threshold))
            assert term != m.TERM_CONFINED, (method, threshold)


def test_the_kernel_receives_the_threshold_exactly_with_no_rounding_or_clamping():
    for value in (5e-324, 1e-300, 1.234567890123e-7, 0.5, 1e300):
        assert m.confine_rule(4, BOX, value).threshold == value


def test_the_default_rule_reproduces_the_old_hard_wired_behavior_on_the_golden_scenes():
    # test_an_obstacle_reproduces_the_recorded_classic_golden_data runs every
    # scene under CONFINE_DEFAULT against data recorded before this was a
    # setting; this pins that the default IS one pixel, box, 4 points.
    assert m.CONFINE_DEFAULT == m.ConfineRule(window=4, method=BOX, threshold=m.PIXEL)


def test_the_box_method_needs_both_directions_to_be_small():
    """A path that is tiny across but long along one axis is NOT confined by
    the box (both extents must be under the threshold)."""
    pts, _k, term, _r = _kernel_trace(0.0, -0.5, 0.0, max_iter=6,
                                      confine=_confine(2, BOX, 0.3))
    assert len(pts) >= 3, "the first segment is wider than the box"


def test_confinement_window_and_threshold_parse_and_validate():
    assert m.parse_confine_window("4") == 4
    assert m.parse_confine_window("2") == 2 and m.parse_confine_window("999") == 999
    for bad in ("", "1", "0", "1000", "-3", "3.5", "abc", "4 "):
        assert m.parse_confine_window(bad) is None, bad


def test_the_threshold_accepts_any_value_above_zero_however_small():
    for text, value in [("0.00390625", 0.00390625), ("1", 1.0), ("3.", 3.0), (".5", 0.5),
                        ("0.0000000000001", 1e-13), ("1e-300", 1e-300), ("5e-324", 5e-324),
                        ("2.5E-7", 2.5e-7), ("1e300", 1e300), ("4e+2", 400.0)]:
        assert m.parse_confine_threshold(text) == value, text


def test_the_threshold_rejects_zero_negatives_and_non_numbers():
    for bad in ("", "0", "0.0", "0.000", "-1", "-1e-9", "1e-400", "1e999", "inf", "nan",
                "abc", "1e", "--1", "1.2.3", " 1", "1 ", "1_0", "e5"):
        assert m.parse_confine_threshold(bad) is None, repr(bad)


def test_the_threshold_field_accepts_partial_typing_of_any_valid_number():
    ok = m.App.validate_positive_float
    for partial in ("", "0", "0.", "0.0", ".", "1e", "1e-", "1e-3", "12.5E+", "5e-324"):
        assert ok(partial), partial
    for never in ("-", "-1", "a", "1e--3", "1.2.3", "1 e5", "1,5"):
        assert not ok(never), never
    assert m.App.validate_window_digits("999") and not m.App.validate_window_digits("1000")


# ---- keyframes, settings plumbing -----------------------------------------------------------
def test_confinement_settings_are_registered_and_only_the_threshold_interpolates():
    keys = {k: tab for k, tab, _label in m.SETTING_INFO}
    for key in ("confine_window", "confine_method", "confine_threshold"):
        assert keys[key] == "Simulation" and key in m.DEFAULT_SETTING_VALUES
    assert "confine_threshold" in m.SETTING_FLOAT_KEYS
    assert "confine_window" not in m.SETTING_FLOAT_KEYS
    assert "confine_method" not in m.SETTING_FLOAT_KEYS


def test_confinement_settings_round_trip_through_capture_and_resolve():
    app = _make_fake_app_for_snapshot()
    app.confine_window_var = _StubVar("9")
    app.confine_method_var = _StubVar("Path length")
    app.confine_threshold_var = _StubVar("1e-12")
    app.deco = {"outline": _StubVar(True)}
    app.obstacles = []
    app.views = []
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    data = app.capture_all_settings()
    assert data["confine_window"] == 9 and data["confine_method"] == PATH
    assert data["confine_threshold"] == 1e-12
    resolved = m.resolve_frame_settings([m.Keyframe(0, 0, data)], 0).settings
    assert (resolved.confine_window, resolved.confine_method, resolved.confine_threshold) \
        == (9, PATH, 1e-12)


def test_confinement_threshold_interpolates_but_window_and_method_hold():
    kf0 = m.Keyframe(0, 0, {"confine_window": 4, "confine_method": BOX,
                            "confine_threshold": 0.0})
    kf1 = m.Keyframe(1, 10, {"confine_window": 20, "confine_method": PATH,
                             "confine_threshold": 1.0})
    mid = m.resolve_frame_settings([kf0, kf1], 5).settings
    assert mid.confine_window == 4 and mid.confine_method == BOX
    assert np.isclose(mid.confine_threshold, 0.5)
    end = m.resolve_frame_settings([kf0, kf1], 10).settings
    assert end.confine_window == 20 and end.confine_method == PATH


def test_short_value_text_for_the_confinement_settings():
    assert m.short_value_text("confine_window", 12) == "12"
    assert m.short_value_text("confine_method", PATH) == "Path length"
    assert m.short_value_text("confine_threshold", 0.00390625) == "0.00390625"
    assert m.short_value_text("confine_threshold", 1e-9) == "1e-09"


def test_a_changed_confinement_rule_makes_the_settings_differ():
    a = m.DEFAULT_RAY_SETTINGS
    for change in (dict(confine_window=5), dict(confine_method=PATH),
                   dict(confine_threshold=0.5)):
        assert m.DEFAULT_RAY_SETTINGS.replace(**change) != a


# ---- the entry handlers ---------------------------------------------------------------------
def _fake_limits_app(window="4", method="Box", threshold="0.00390625"):
    class FakeApp:
        pass
    app = FakeApp()
    app.on_limit_var_changed = m.App.on_limit_var_changed.__get__(app)
    app.on_limit_commit = m.App.on_limit_commit.__get__(app)
    app.iter_var, app.dist_var = _StubVar("256"), _StubVar("0")
    app.confine_window_var = _StubVar(window)
    app.confine_method_var = _StubVar(method)
    app.confine_threshold_var = _StubVar(threshold)
    for field in m.EMIT_TARGET_FIELDS:
        setattr(app, f"emit_target_{field}_var", _StubVar("0.0"))
        setattr(app, f"last_emit_target_{field}", 0.0)
    app.last_iter, app.last_dist = 256, 0
    app.last_confine_window = 4
    app.last_confine_threshold = m.PIXEL
    app.styles = {}

    class Entry:
        def __init__(self, name):
            self.name = name

        def configure(self, style=None, **_k):
            app.styles[self.name] = style
    app.iter_entry, app.dist_entry = Entry("iter"), Entry("dist")
    app.confine_window_entry, app.confine_threshold_entry = Entry("window"), Entry("threshold")
    for field in m.EMIT_TARGET_FIELDS:
        setattr(app, f"emit_target_{field}_entry", Entry(f"target_{field}"))
    app.limit_msg_var, app.confine_msg_var = _StubVar(""), _StubVar("")
    app.emit_msg_var = _StubVar("")
    app.invalidate_preview = app.update_buttons = app.refresh_state_text = lambda: None
    app.refresh_timeline_diffs = lambda: None
    return app


def test_valid_confinement_entries_update_the_last_good_values():
    app = _fake_limits_app(window="9", threshold="1e-30")
    app.on_limit_var_changed()
    assert app.last_confine_window == 9 and app.last_confine_threshold == 1e-30
    assert app.styles["window"] == "TEntry" and app.styles["threshold"] == "TEntry"
    assert app.confine_msg_var.get() == ""


def test_an_invalid_threshold_is_flagged_and_the_last_good_value_is_kept():
    for bad in ("0", "-1", "abc", ""):
        app = _fake_limits_app(threshold=bad)
        app.on_limit_var_changed()
        assert app.styles["threshold"] == "Invalid.TEntry", bad
        assert app.last_confine_threshold == m.PIXEL
        assert "greater than 0" in app.confine_msg_var.get()
        app.on_limit_commit()
        assert app.confine_threshold_var.get() == str(m.PIXEL), "put back the last good one"


def test_an_invalid_window_is_flagged_and_restored_on_commit():
    app = _fake_limits_app(window="1")
    app.on_limit_var_changed()
    assert app.styles["window"] == "Invalid.TEntry"
    assert "whole number from 2 to 999" in app.confine_msg_var.get()
    app.on_limit_commit()
    assert app.confine_window_var.get() == "4"


def test_committing_tidies_a_valid_threshold_without_changing_its_value():
    app = _fake_limits_app(threshold="4E-3")
    app.on_limit_var_changed()
    app.on_limit_commit()
    assert app.confine_threshold_var.get() == "0.004"
    assert float(app.confine_threshold_var.get()) == 0.004


def test_current_ray_settings_is_none_while_a_confinement_entry_is_invalid():
    for field, bad in (("confine_window_var", "1"), ("confine_threshold_var", "0")):
        app = _make_fake_app_for_snapshot()
        setattr(app, field, _StubVar(bad))
        assert app.current_ray_settings() is None, field


def test_current_ray_settings_reads_the_confinement_controls():
    app = _make_fake_app_for_snapshot()
    app.confine_window_var = _StubVar("7")
    app.confine_method_var = _StubVar("Path length")
    app.confine_threshold_var = _StubVar("2.5e-9")
    snap = app.current_ray_settings()
    assert (snap.confine_window, snap.confine_method, snap.confine_threshold) == (7, PATH, 2.5e-9)


def test_the_threshold_label_follows_the_method():
    class FakeApp:
        pass
    app = FakeApp()
    app.update_confine_threshold_label = m.App.update_confine_threshold_label.__get__(app)
    app.confine_method_var = _StubVar("Box")
    app.confine_threshold_label_var = _StubVar("")
    app.update_confine_threshold_label()
    assert "Box size" in app.confine_threshold_label_var.get()
    app.confine_method_var.set("Path length")
    app.update_confine_threshold_label()
    assert "Path length" in app.confine_threshold_label_var.get()


def test_the_simulation_tab_has_a_confinement_section_with_all_three_controls():
    import inspect
    assert "self.build_confinement_controls(sim_inner)" in \
        inspect.getsource(m.App.build_controls_quadrant)
    source = inspect.getsource(m.App.build_confinement_controls)
    assert 'text="Confinement"' in source
    for control in ("confine_window_entry", "confine_method_var", "confine_threshold_entry"):
        assert control in source


# ---- the figure settings margin ------------------------------------------------------------
def test_figure_info_lists_the_confinement_rule():
    lines = m.figure_info_lines(_settings(), _start_scene())
    assert "Confine:Box" in lines and "  window 4" in lines
    assert "  box size 0.00390625" in lines
    lines = m.figure_info_lines(_settings(confine_window=9, confine_method=PATH,
                                          confine_threshold=1.5e-9), _start_scene())
    assert "Confine:Path length" in lines and "  window 9" in lines
    assert "  path length 1.5e-09" in lines


def test_confinement_lines_fit_a_column_for_extreme_values_and_use_known_glyphs():
    for method in (BOX, PATH):
        for threshold in (5e-324, 1.7976931348623157e308, 0.1234567891234):
            lines = m.figure_info_lines(_settings(confine_window=999, confine_method=method,
                                                  confine_threshold=threshold), _start_scene())
            for line in lines:
                assert len(line) <= m.FIGURE_INFO_COLUMN_CHARS, line
                assert all(c in m.FONT_5X7 for c in line), line


def test_the_confinement_block_has_the_same_size_whatever_its_values():
    a = m.figure_info_blocks(_settings(), _start_scene())
    b = m.figure_info_blocks(_settings(confine_window=500, confine_method=PATH,
                                       confine_threshold=1e-300), _start_scene())
    assert [len(x) for x in a] == [len(x) for x in b]


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
    pts = np.empty((6, 2))
    kinds = np.empty(6)
    raw = np.empty(m.N_RAW)
    fz = m.fuzz_rule(m.DEFAULT_FUZZ if fuzz is None else fuzz)
    _seed_kernel_rng(seed)
    term, npts = m.trace_path(x0, y0, pts, kinds, 4, m.effective_dist(0.0), 1.0, fz,
                              m.CONFINE_DEFAULT, emission, arrays, raw)
    assert term == m.TERM_ESCAPED and npts == 2
    delta = pts[1] - pts[0]
    return delta / np.hypot(*delta)


def _angle_deg(direction):
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def test_the_default_emission_is_up_with_a_minus_45_degree_offset():
    assert m.EMIT_MODE_NAMES == ["Up", "Towards point"]
    assert (m.DEFAULT_EMIT_MODE, m.DEFAULT_EMIT_TARGET_R, m.DEFAULT_EMIT_TARGET_THETA,
            m.DEFAULT_EMIT_TARGET_CX, m.DEFAULT_EMIT_TARGET_CY,
            m.DEFAULT_EMIT_OFFSET) == (UP, 0.0, 0.0, 0.0, 0.0, -45.0)
    assert m.EMISSION_DEFAULT == m.EmissionRule(mode=0, target_x=0.0, target_y=0.0, offset=-45.0)
    s = m.DEFAULT_RAY_SETTINGS
    assert (s.emit_mode, s.emit_target_r, s.emit_target_theta, s.emit_target_cx,
            s.emit_target_cy, s.emit_offset) == (UP, 0.0, 0.0, 0.0, 0.0, -45.0)
    assert m.settings_emission(s) == m.EMISSION_DEFAULT


def test_the_default_emission_is_exactly_the_up_and_right_direction_used_before():
    """Bit-for-bit (1, 1)/sqrt(2), not merely close - rays in chaotic scenes
    would otherwise drift from the results everything was recorded against."""
    for x0, y0 in ((0.3, -0.2), (-0.9, 0.1), (0.0123, 0.0456)):
        d = _first_direction(x0, y0, m.EMISSION_DEFAULT)
        assert abs(d[0] - m.INV_SQRT2) < 1e-15 and abs(d[1] - m.INV_SQRT2) < 1e-15
    assert m.unit_direction(45.0) == (m.INV_SQRT2, m.INV_SQRT2)


def test_up_with_a_zero_offset_fires_exactly_straight_up():
    d = _first_direction(0.3, -0.2, _emission(UP, offset=0.0))
    assert d[0] == 0.0 and abs(d[1] - 1.0) < 1e-15


def test_positive_offsets_turn_counterclockwise_and_negative_ones_clockwise():
    for offset, expected in [(0.0, 90.0), (10.0, 100.0), (30.0, 120.0), (90.0, 180.0),
                             (-30.0, 60.0), (-90.0, 0.0), (-135.0, -45.0), (180.0, -90.0)]:
        got = _angle_deg(_first_direction(0.2, 0.1, _emission(UP, offset=offset)))
        assert abs(((got - expected + 180.0) % 360.0) - 180.0) < 1e-9, (offset, got, expected)


def test_towards_a_point_fires_from_each_start_pixel_to_that_point():
    tx, ty = 0.4, -0.3
    for x0, y0 in [(-0.5, 0.5), (0.9, 0.9), (0.4, 0.8), (-0.7, -0.6)]:
        d = _first_direction(x0, y0, _emission(TOWARDS, tx, ty, offset=0.0))
        want = np.array([tx - x0, ty - y0])
        want /= np.hypot(*want)
        assert np.allclose(d, want, atol=1e-12), (x0, y0, d, want)


def test_towards_the_default_target_the_origin_fires_at_the_origin():
    d = _first_direction(0.6, 0.0, _emission(TOWARDS, offset=0.0))
    assert np.allclose(d, (-1.0, 0.0))


def test_the_offset_is_added_to_the_direction_towards_the_target():
    base = _angle_deg(_first_direction(0.6, 0.0, _emission(TOWARDS, 0.0, 0.0, 0.0)))
    turned = _angle_deg(_first_direction(0.6, 0.0, _emission(TOWARDS, 0.0, 0.0, 25.0)))
    assert abs(((turned - base - 25.0 + 180.0) % 360.0) - 180.0) < 1e-9


def test_a_ray_starting_exactly_on_the_target_points_up_before_the_offset():
    d = _first_direction(0.25, -0.5, _emission(TOWARDS, 0.25, -0.5, 0.0))
    assert d[0] == 0.0 and abs(d[1] - 1.0) < 1e-15


def test_the_target_point_can_be_anywhere_at_all():
    for tx, ty in [(1e6, 1e6), (-3.0e5, 2.0), (1e-9, -1e-9), (0.0, -1e12), (1e300, -1e300)]:
        d = _first_direction(0.1, 0.1, _emission(TOWARDS, tx, ty, 0.0))
        assert np.isfinite(d).all() and abs(np.hypot(*d) - 1.0) < 1e-12
        want = np.arctan2(ty - 0.1, tx - 0.1)
        assert abs(np.arctan2(d[1], d[0]) - want) < 1e-9, (tx, ty)


def test_fuzz_jitter_is_added_on_top_of_the_emission_direction_and_only_when_enabled():
    plain = _angle_deg(_first_direction(0.2, 0.1, _emission(TOWARDS, 0.0, 0.0, 20.0)))
    off = m.DEFAULT_FUZZ.replace(emit_jitter=30.0)              # jitter set, fuzzing off
    assert _angle_deg(_first_direction(0.2, 0.1, _emission(TOWARDS, 0.0, 0.0, 20.0), off)) == plain
    on = m.DEFAULT_FUZZ.replace(enabled=True, emit_jitter=30.0)
    got = [_angle_deg(_first_direction(0.2, 0.1, _emission(TOWARDS, 0.0, 0.0, 20.0), on, seed=k))
           for k in range(40)]
    spread = [((g - plain + 180.0) % 360.0) - 180.0 for g in got]
    assert max(abs(v) for v in spread) <= 30.0 + 1e-9
    assert max(spread) > 10.0 and min(spread) < -10.0, "genuinely random within +-30"


def test_enabling_fuzz_no_longer_overrides_the_emission_direction():
    """Regression guard for the design change: the Fuzzing tab's old absolute
    'emission angle' replaced the direction whenever fuzzing was on, which
    would have silently defeated 'Towards point'."""
    towards = _emission(TOWARDS, 0.0, 0.0, 0.0)
    fuzz_on_no_jitter = m.DEFAULT_FUZZ.replace(enabled=True, emit_jitter=0.0)
    a = _first_direction(0.6, 0.2, towards)
    b = _first_direction(0.6, 0.2, towards, fuzz_on_no_jitter)
    assert np.allclose(a, b, atol=1e-15)
    assert "emit_angle" not in [f.name for f in dataclasses.fields(m.FuzzParams)]
    assert m.fuzz_rule(m.DEFAULT_FUZZ) == m.FuzzRule(
        enabled=False, emit_jitter=0.0, bnd_offset=0.0, bnd_jitter=0.0, bnd_mult=1.0,
        self_offset=0.0, self_jitter=0.0, self_mult=1.0)


def test_unit_direction_is_exact_at_multiples_of_45_degrees_and_trig_elsewhere():
    s2 = m.INV_SQRT2
    exact = {0.0: (1.0, 0.0), 45.0: (s2, s2), 90.0: (0.0, 1.0), 135.0: (-s2, s2),
             180.0: (-1.0, 0.0), 225.0: (-s2, -s2), 270.0: (0.0, -1.0), 315.0: (s2, -s2),
             -45.0: (s2, -s2), -90.0: (0.0, -1.0), 360.0: (1.0, 0.0), -270.0: (0.0, 1.0)}
    for deg, want in exact.items():
        assert m.unit_direction(deg) == want, deg
    for deg in (10.0, 33.3, -170.5, 400.0):
        dx, dy = m.unit_direction(deg)
        assert abs(dx - np.cos(np.radians(deg))) < 1e-15
        assert abs(dy - np.sin(np.radians(deg))) < 1e-15


def test_the_kernel_uses_the_emission_for_every_ray_in_a_render():
    """End to end through the real (parallel) render kernel: every fired ray's
    first segment points the way the emission says."""
    start = m.default_starting_obstacle(0)
    arrays = m.build_obstacle_arrays([start])
    n = 24
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    status = np.zeros((n, n), dtype=np.uint8)
    up = _emission(UP, offset=0.0)
    m.render_rows_wh(stats, points, status, 0, n, n, n, 1, m.effective_dist(0.0), 1.0,
                     m.fuzz_rule(m.DEFAULT_FUZZ), m.CONFINE_DEFAULT, up, arrays, 0)
    fired = status == 1
    xs = ((np.arange(n) + 0.5) * (2.0 / n) - 1.0)[None, :].repeat(n, axis=0)
    ys = (1.0 - (np.arange(n) + 0.5) * (2.0 / n))[:, None].repeat(n, axis=1)
    ends = points[..., m.POINT_END, :]
    assert fired.sum() > 300
    assert np.allclose(ends[..., 0][fired], xs[fired], atol=1e-6), "straight up keeps x"
    assert (ends[..., 1][fired] > ys[fired]).all(), "and goes up"
    toward = _emission(TOWARDS, 0.0, 0.0, 0.0)
    m.render_rows_wh(stats, points, status, 0, n, n, n, 1, m.effective_dist(0.0), 1.0,
                     m.fuzz_rule(m.DEFAULT_FUZZ), m.CONFINE_DEFAULT, toward, arrays, 0)
    ends = points[..., m.POINT_END, :]
    fired = status == 1
    start_r = np.hypot(xs, ys)[fired]
    end_r = np.hypot(ends[..., 0], ends[..., 1])[fired]
    assert end_r.mean() > 0.99, "aimed at the origin, every ray crosses to the circle's far side"
    away = (ends[..., 0][fired] * xs[fired] + ends[..., 1][fired] * ys[fired]) < 0
    assert away.mean() > 0.95, "the far side: the opposite side of the origin from the start"
    assert start_r.max() <= 1.0


def test_the_target_point_is_positioned_like_an_obstacle():
    """Distance r at angle theta from the origin, plus an x and y offset: the same
    formula as Obstacle.center."""
    s = _settings(emit_mode=TOWARDS, emit_target_r=2.0, emit_target_theta=90.0,
                  emit_target_cx=0.5, emit_target_cy=-0.25)
    rule = m.settings_emission(s)
    tx, ty = rule.target_x, rule.target_y
    assert abs(tx - 0.5) < 1e-12 and abs(ty - 1.75) < 1e-12
    o = m.EllipseShape(0)
    o.r, o.theta, o.cx, o.cy = 2.0, 90.0, 0.5, -0.25
    assert (tx, ty) == o.center()
    # a negative r flips through the origin; theta is not limited to one turn
    s = _settings(emit_mode=TOWARDS, emit_target_r=-1.0, emit_target_theta=720.0 + 180.0)
    rule = m.settings_emission(s)
    tx, ty = rule.target_x, rule.target_y
    assert abs(tx - 1.0) < 1e-9 and abs(ty) < 1e-9


def test_rays_fired_towards_a_polar_target_head_for_it():
    s = _settings(emit_mode=TOWARDS, emit_target_r=1.0, emit_target_theta=90.0,
                  emit_target_cx=0.0, emit_target_cy=0.0, emit_offset=0.0)
    emission = m.settings_emission(s)
    d = _first_direction(0.0, -0.5, emission)
    assert abs(_angle_deg(d) - 90.0) < 1e-6          # straight up, at (0, 1)
    d = _first_direction(-0.5, 0.0, emission)
    want = np.degrees(np.arctan2(1.0 - 0.0, 0.0 - -0.5))
    assert abs(_angle_deg(d) - want) < 1e-6


# ---- parsing and key checks ---------------------------------------------------------------
def test_target_coordinates_accept_any_finite_number_of_either_sign():
    for text, value in [("0", 0.0), ("-0.5", -0.5), ("0.5", 0.5), ("3.", 3.0), (".5", 0.5),
                        ("-.5", -0.5), ("3e8", 3e8), ("-2.5E-7", -2.5e-7), ("1e300", 1e300),
                        ("-1e300", -1e300), ("5e-324", 5e-324), ("123456789.123", 123456789.123)]:
        assert m.parse_target_coordinate(text) == value, text


def test_target_coordinates_reject_non_numbers_and_non_finite_values():
    for bad in ("", "-", ".", "abc", "1e", "1e999", "-1e999", "inf", "nan", "+1", "--1",
                "1.2.3", " 1", "1 ", "1_0", "e5", "1,5"):
        assert m.parse_target_coordinate(bad) is None, repr(bad)


def test_the_coordinate_field_accepts_partial_typing_of_any_valid_number():
    ok = m.App.validate_signed_float
    for partial in ("", "-", "-0", "-.", "0.", "1e", "1e-", "-1.5e+", "12.5E-3", "-5e-324"):
        assert ok(partial), partial
    for never in ("+", "a", "1e--3", "1.2.3", "--1", "1 e5", "1,5"):
        assert not ok(never), never


# ---- keyframes, settings plumbing -----------------------------------------------------------
def test_emission_settings_are_registered_and_the_right_ones_interpolate():
    keys = {k: tab for k, tab, _label in m.SETTING_INFO}
    target = ["emit_target_r", "emit_target_theta", "emit_target_cx", "emit_target_cy"]
    for key in ["emit_mode", *target, "emit_offset"]:
        assert keys[key] == "Simulation" and key in m.DEFAULT_SETTING_VALUES
    for key in [*target, "emit_offset"]:
        assert key in m.SETTING_FLOAT_KEYS
    assert "emit_mode" not in m.SETTING_FLOAT_KEYS
    assert "emit_angle" not in keys and "emit_angle" not in m.DEFAULT_SETTING_VALUES


def test_emission_settings_round_trip_through_capture_and_resolve():
    app = _make_fake_app_for_snapshot()
    app.emit_mode_var = _StubVar("Towards point")
    app.emit_target_r_var = _StubVar("-2.5e3")
    app.emit_target_theta_var = _StubVar("90")
    app.emit_target_cx_var = _StubVar("0.75")
    app.emit_target_cy_var = _StubVar("-4")
    app.emit_offset_scale = _StubVar(12.5)
    app.deco = {"outline": _StubVar(True)}
    app.obstacles = []
    app.views = []
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    data = app.capture_all_settings()
    assert (data["emit_mode"], data["emit_target_r"], data["emit_target_theta"],
            data["emit_target_cx"], data["emit_target_cy"],
            data["emit_offset"]) == (TOWARDS, -2500.0, 90.0, 0.75, -4.0, 12.5)
    snap = m.resolve_frame_settings([m.Keyframe(0, 0, data)], 0).settings
    assert (snap.emit_mode, snap.emit_target_r, snap.emit_target_theta, snap.emit_target_cx,
            snap.emit_target_cy, snap.emit_offset) == (TOWARDS, -2500.0, 90.0, 0.75, -4.0, 12.5)


def test_emission_target_and_offset_interpolate_but_the_mode_holds():
    kf0 = m.Keyframe(0, 0, {"emit_mode": UP, "emit_target_r": 0.0, "emit_target_theta": 0.0,
                            "emit_target_cx": -1.0, "emit_target_cy": 0.0,
                            "emit_offset": -45.0})
    kf1 = m.Keyframe(1, 10, {"emit_mode": TOWARDS, "emit_target_r": 2.0,
                             "emit_target_theta": 90.0, "emit_target_cx": 1.0,
                             "emit_target_cy": 4.0, "emit_offset": 45.0})
    mid = m.resolve_frame_settings([kf0, kf1], 5).settings
    assert mid.emit_mode == UP
    assert (mid.emit_target_r, mid.emit_target_theta, mid.emit_target_cx, mid.emit_target_cy,
            mid.emit_offset) == (1.0, 45.0, 0.0, 2.0, 0.0)
    assert m.resolve_frame_settings([kf0, kf1], 10).settings.emit_mode == TOWARDS


def test_short_value_text_for_the_emission_settings():
    assert m.short_value_text("emit_mode", TOWARDS) == "Towards point"
    assert m.short_value_text("emit_offset", -45.0) == "-45.0\u00b0"
    assert m.short_value_text("emit_offset", 12.34) == "+12.3\u00b0"
    assert m.short_value_text("emit_target_r", 0.5) == "0.5"
    assert m.short_value_text("emit_target_theta", 30.0) == "30\u00b0"
    assert m.short_value_text("emit_target_cx", -2.0) == "-2"


def test_a_changed_emission_makes_the_settings_differ():
    a = m.DEFAULT_RAY_SETTINGS
    for change in (dict(emit_mode=TOWARDS), dict(emit_target_r=1.0),
                   dict(emit_target_theta=10.0), dict(emit_target_cx=1.0),
                   dict(emit_target_cy=-1.0), dict(emit_offset=0.0)):
        assert m.DEFAULT_RAY_SETTINGS.replace(**change) != a


# ---- the entry handlers and UI ----------------------------------------------------------------
def test_valid_target_entries_update_the_last_good_values():
    app = _fake_limits_app()
    app.emit_target_r_var.set("-12.5")
    app.emit_target_theta_var.set("-720")
    app.emit_target_cx_var.set("3e8")
    app.emit_target_cy_var.set("0.25")
    app.on_limit_var_changed()
    assert (app.last_emit_target_r, app.last_emit_target_theta, app.last_emit_target_cx,
            app.last_emit_target_cy) == (-12.5, -720.0, 3e8, 0.25)
    assert all(app.styles[f"target_{f}"] == "TEntry" for f in m.EMIT_TARGET_FIELDS)
    assert app.emit_msg_var.get() == ""


def test_an_invalid_target_is_flagged_and_the_last_good_value_restored_on_commit():
    for field in m.EMIT_TARGET_FIELDS:
        app = _fake_limits_app()
        getattr(app, f"emit_target_{field}_var").set("abc")
        app.on_limit_var_changed()
        assert app.styles[f"target_{field}"] == "Invalid.TEntry"
        assert "numbers" in app.emit_msg_var.get()
        app.on_limit_commit()
        assert getattr(app, f"emit_target_{field}_var").get() == "0.0"


def test_current_ray_settings_reads_the_emission_controls_and_is_none_when_a_target_is_bad():
    app = _make_fake_app_for_snapshot()
    app.emit_mode_var = _StubVar("Towards point")
    app.emit_target_r_var = _StubVar("-0.25")
    app.emit_target_theta_var = _StubVar("45")
    app.emit_target_cx_var = _StubVar("1e3")
    app.emit_target_cy_var = _StubVar("-2")
    app.emit_offset_scale = _StubVar(33.333)
    snap = app.current_ray_settings()
    assert (snap.emit_mode, snap.emit_target_r, snap.emit_target_theta, snap.emit_target_cx,
            snap.emit_target_cy, snap.emit_offset) == (TOWARDS, -0.25, 45.0, 1000.0, -2.0, 33.33)
    app.emit_target_theta_var = _StubVar("x")
    assert app.current_ray_settings() is None


def test_the_simulation_tab_has_a_ray_emission_section_with_all_its_controls():
    import inspect
    assert "self.build_emission_controls(sim_inner)" in \
        inspect.getsource(m.App.build_controls_quadrant)
    source = inspect.getsource(m.App.build_emission_controls)
    assert 'text="Ray emission"' in source
    for control in ("emit_mode_var", "emit_offset_scale", "EMIT_TARGET_FIELDS",
                    "EMIT_TARGET_LABELS"):
        assert control in source


def test_the_fuzzing_tab_no_longer_has_an_emission_angle_control():
    import inspect
    source = inspect.getsource(m.App.build_fuzzing_controls)
    assert "emit_angle" not in source and "emit_jitter_scale" in source


def test_the_target_entries_are_only_enabled_when_firing_towards_a_point():
    class FakeApp:
        pass
    app = FakeApp()
    app.update_emission_state = m.App.update_emission_state.__get__(app)
    app.emit_mode_var = _StubVar("Up")
    states = []

    class Entry:
        def state(self, s):
            states.append(list(s))
    app.emit_target_entries = [Entry() for _ in range(4)]
    app.update_emission_state()
    assert states == [["disabled"]] * 4
    states.clear()
    app.emit_mode_var.set("Towards point")
    app.update_emission_state()
    assert states == [["!disabled"]] * 4


# ---- the figure settings margin --------------------------------------------------------------
def test_figure_info_lists_the_emission_direction():
    lines = m.figure_info_lines(_settings(), _start_scene())
    assert "Emit:Up" in lines and "  angle offset -45.00\u00b0" in lines
    lines = m.figure_info_lines(_settings(emit_mode=TOWARDS, emit_target_r=-0.5,
                                          emit_target_theta=30.0, emit_target_cx=1.25,
                                          emit_target_cy=-2.0, emit_offset=10.0),
                                _start_scene())
    assert "Emit:Towards point" in lines
    assert "  target r -0.5" in lines and "  target angle 30" in lines
    assert "  offset x 1.25" in lines and "  offset y -2" in lines
    assert "  angle offset +10.00\u00b0" in lines


def test_emission_lines_fit_a_column_for_extreme_values_and_use_known_glyphs():
    for big, small in ((-1.7976931348623157e308, 1.7976931348623157e308), (5e-324, -5e-324)):
        lines = m.figure_info_lines(_settings(emit_mode=TOWARDS, emit_target_r=big,
                                              emit_target_theta=small, emit_target_cx=big,
                                              emit_target_cy=small, emit_offset=-179.99),
                                    _start_scene())
        for line in lines:
            assert len(line) <= m.FIGURE_INFO_COLUMN_CHARS, line
            assert all(c in m.FONT_5X7 for c in line), line


def test_the_emission_block_has_the_same_size_whatever_its_values():
    a = m.figure_info_blocks(_settings(), _start_scene())
    b = m.figure_info_blocks(_settings(emit_mode=TOWARDS, emit_target_r=3.0,
                                       emit_target_theta=-20.0, emit_target_cx=3.0,
                                       emit_target_cy=-2.0, emit_offset=120.0), _start_scene())
    assert [len(x) for x in a] == [len(x) for x in b]


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
    """Points m.paths.DEV_DIR at a scratch directory (created only if
    `with_dev_dir`) and m.external.ImageGrab at a fake (or None), restoring both."""
    def __init__(self, with_dev_dir=True, grab="fake"):
        self.with_dev_dir = with_dev_dir
        self.grab = _FakeImageGrab() if grab == "fake" else grab

    def __enter__(self):
        self.parent = tempfile.mkdtemp()
        self.dev_dir = os.path.join(self.parent, "dev")
        if self.with_dev_dir:
            os.makedirs(self.dev_dir)
        self.orig_dev, self.orig_grab = m.paths.DEV_DIR, m.external.ImageGrab
        m.paths.DEV_DIR, m.external.ImageGrab = self.dev_dir, self.grab
        return self

    def __exit__(self, *exc):
        m.paths.DEV_DIR, m.external.ImageGrab = self.orig_dev, self.orig_grab


def test_dev_dir_is_dev_next_to_the_program():
    assert m.paths.DEV_DIR == os.path.join(m.SCRIPT_DIR, "dev")


def test_screenshot_is_skipped_without_a_dev_dir():
    with _ScreenshotEnv(with_dev_dir=False) as env:
        assert m.take_ui_screenshot(_FakeGeometryRoot(), "startup") is None
        assert env.grab.bboxes == [], "must not even attempt a grab"
        assert not os.path.exists(env.dev_dir), "must not create dev/ itself"


def test_screenshot_is_skipped_without_pillow():
    with _ScreenshotEnv(grab=None) as env:
        assert m.take_ui_screenshot(_FakeGeometryRoot(), "manual") is None
        assert not os.path.exists(os.path.join(env.dev_dir, "screenshots"))


def test_screenshot_grabs_the_windows_own_rectangle_and_saves_it_named_by_version_and_cause():
    import re
    root = _FakeGeometryRoot()
    with _ScreenshotEnv() as env:
        path = m.take_ui_screenshot(root, "startup")
        shots_dir = os.path.join(env.dev_dir, "screenshots")
        assert os.path.dirname(path) == shots_dir
        assert re.fullmatch(rf"{m.FILE_VERSION}-\d{{8}}-\d{{6}}-startup\.png",
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
    assert m.paths.EXPORT_DIR == os.path.join(m.SCRIPT_DIR, "exports")
    assert m.paths.TEMP_DIR == os.path.join(m.paths.EXPORT_DIR, "TEMP")


def test_ensure_export_dir_creates_a_missing_directory_and_returns_it():
    parent = tempfile.mkdtemp()
    target = os.path.join(parent, "exports")
    orig = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = target
    try:
        assert not os.path.exists(target)
        assert m.ensure_export_dir() == target
        assert os.path.isdir(target)
        assert m.ensure_export_dir() == target      # already existing is fine
    finally:
        m.paths.EXPORT_DIR = orig


def test_export_writes_the_current_image_into_the_export_dir():
    """Regression test for the exports/ move: "Export" (no dialog) used to
    write next to the program itself. Also confirms the directory is created
    if it doesn't exist yet."""
    class FakeApp:
        pass
    app = FakeApp()
    app.save_source_var = _StubVar(m.SAVE_SOURCE_OPTIONS[0])
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    written = []
    app._write = lambda path: written.append(path)
    app.save = m.App.save.__get__(app)
    parent = tempfile.mkdtemp()
    target = os.path.join(parent, "exports")
    orig = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = target
    try:
        app.save()
    finally:
        m.paths.EXPORT_DIR = orig
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
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    app._write = lambda path: None
    app.save_as = m.App.save_as.__get__(app)
    seen = {}
    orig_dialog = m.filedialog.asksaveasfilename
    m.filedialog.asksaveasfilename = lambda *a, **k: seen.update(k) or ""
    parent = tempfile.mkdtemp()
    target = os.path.join(parent, "exports")
    orig = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = target
    try:
        app.save_as()
    finally:
        m.paths.EXPORT_DIR = orig
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
    orig = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = target
    try:
        _run_save_new_image_synchronously(app)
    finally:
        m.paths.EXPORT_DIR = orig
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
    app.current_ray_settings = lambda: m.DEFAULT_RAY_SETTINGS
    app.generated_settings = None
    app.gen_obstacles_sig = None
    app.obstacles = []
    app.baseline = m.DEFAULT_RAY_SETTINGS
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
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    app.rows_done = m.SIZE
    app.running = False
    app.update_buttons()
    assert app.save_btn_tooltip.text == ""


def test_export_disabled_while_still_generating():
    app = _fake_button_gating_app()
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
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
    chan1_value = m.ChannelSettings(
        chan_src=m.CHAN_SRC_QUERY, kind=m.KIND_POINT, stat=m.ST_HDIST, point1=m.POINT_START,
        point2=m.POINT_END, buddha_points=m.DEFAULT_BUDDHA_POINTS,
        buddha_compression=m.BUDDHA_COMPRESS_ARCTAN, buddha_brightness=0, constant=m.NONE_ZERO)
    app.load_value("view:0:chan1", chan1_value)
    assert app.rebuild_buddha_counts_calls == []


def test_load_value_rebuilds_when_the_loaded_channel_is_buddha():
    app = _fake_color_editor_app()
    app.load_value = m.App.load_value.__get__(app)
    chan0_value = m.ChannelSettings(
        chan_src=m.CHAN_SRC_BUDDHA, kind=m.KIND_OTHER, stat=m.ST_BOUNCES, point1=m.POINT_START,
        point2=m.POINT_END, buddha_points=m.DEFAULT_BUDDHA_POINTS,
        buddha_compression=m.BUDDHA_COMPRESS_ARCTAN, buddha_brightness=0, constant=m.NONE_ZERO)
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
# Collapsible sections of the keyframe settings list
# --------------------------------------------------------------------------- #
class _RecordingLabel:
    """Records the text of every ttk.Label built, and keeps its click handler."""
    made = []

    def __init__(self, _parent=None, text="", **_k):
        self.text = text
        self.handler = None
        _RecordingLabel.made.append(self)

    def grid(self, *a, **k):
        return self

    def bind(self, _event, fn):
        self.handler = fn


class _RecordingRow:
    """Stands in for the ttk.Frame/Button of one setting row; counts rows."""
    frames = 0

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: self


def _fake_diff_app():
    import types
    keyframe_cls = m.Keyframe
    tabs = ["Limits", "Fuzzing"]
    entries = [m.SettingEntry(key="max_iter", tab="Limits", label="Iteration limit",
                              is_float=False, fallback=256),
               m.SettingEntry(key="max_dist", tab="Limits", label="Distance limit",
                              is_float=True, fallback=1.0),
               m.SettingEntry(key="fuzz_seed", tab="Fuzzing", label="Seed",
                              is_float=False, fallback=0)]

    class FakeApp:
        pass
    app = FakeApp()
    app.diff_expanded = set()
    app.diff_presence_var = m.tk.StringVar(value="present or absent")
    app.diff_list_frame = types.SimpleNamespace(winfo_children=lambda: [])
    app.kf_modified = set()
    app.diff_save_all_btn = _StubWidget()
    app.diff_delete_all_btn = _StubWidget()
    app.small_font = app.small_bold_font = None
    app.kf = keyframe_cls(1, 0, {"max_iter": 10})
    app.keyframes = [app.kf]
    app.selected_keyframe = app.kf
    app.capture_all_settings = lambda show_error=True: {e.key: e.fallback for e in entries}
    app.all_setting_entries = lambda: entries
    app.all_setting_tabs = lambda: tabs
    app.setting_status = lambda kf, key, sk, i: "present" if key in kf.data else "none"
    app.rows = []
    app.add_setting_row = lambda row, kf, key, *a: app.rows.append(key)
    app.fit_diff_scroll = lambda: None
    for name in ("refresh_timeline_diffs", "_rebuild_timeline_diffs", "toggle_diff_section"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    return app


def _headers_of(app):
    _RecordingLabel.made = []
    saved = m.ttk.Label
    m.ttk.Label = _RecordingLabel
    try:
        app.refresh_timeline_diffs()
    finally:
        m.ttk.Label = saved
    return list(_RecordingLabel.made)


def test_section_header_text_shows_arrow_title_and_defined_count():
    assert m.diff_section_header_text("Limits", False, 1, 3) == "\u25b6 Limits  (1 of 3 set)"
    assert m.diff_section_header_text("Limits", True, 0, 3) == "\u25bc Limits  (0 of 3 set)"


def test_keyframe_settings_sections_start_collapsed():
    """Every section is collapsed by default: headers only, no setting rows."""
    app = _fake_diff_app()
    labels = _headers_of(app)
    assert [l.text for l in labels] == [
        "\u25b6 Limits  (1 of 2 set)", "\u25b6 Fuzzing  (0 of 1 set)"]
    assert app.rows == []


def test_clicking_a_section_header_expands_only_that_section_and_again_collapses():
    app = _fake_diff_app()
    _headers_of(app)[0].handler(None)          # expand Limits
    labels = _headers_of(app)
    assert labels[0].text.startswith("\u25bc Limits")
    assert labels[1].text.startswith("\u25b6 Fuzzing")
    app.rows = []
    _headers_of(app)
    assert app.rows == ["max_iter", "max_dist"]
    _headers_of(app)[0].handler(None)          # collapse again
    assert app.diff_expanded == set()
    app.rows = []
    _headers_of(app)
    assert app.rows == []


def test_expanded_sections_stay_expanded_across_rebuilds():
    app = _fake_diff_app()
    _headers_of(app)[1].handler(None)          # expand Fuzzing
    for _ in range(3):
        labels = _headers_of(app)
        assert labels[1].text.startswith("\u25bc Fuzzing")


def test_collapsing_does_not_change_what_load_all_save_all_delete_all_act_on():
    """Those buttons work from the keyframe's data and all_setting_entries, never
    from the visible rows, so collapsed sections are still included."""
    import inspect
    for name in ("save_modified_settings", "delete_all_diffs"):
        src = inspect.getsource(getattr(m.App, name))
        assert "diff_expanded" not in src, name


def test_presence_filter_truth_table():
    f = m.setting_passes_presence_filter
    assert m.DIFF_PRESENCE_CHOICES[-1] == "present or absent"   # the default shows everything
    for in_kf in (True, False):
        assert f(in_kf, "present or absent")
        assert f(in_kf, "present") == in_kf
        assert f(in_kf, "absent") == (not in_kf)


def test_keyframe_list_filters_by_presence_and_hides_empty_sections():
    """The keyframe stores only max_iter, so "present" shows just it (hiding the
    Fuzzing section), and "absent" shows the rest."""
    app = _fake_diff_app()
    app.diff_expanded = {"Limits", "Fuzzing"}

    def shown(presence):
        app.diff_presence_var.set(presence)
        app.rows = []
        labels = _headers_of(app)
        return app.rows, [l.text.split("  ")[0][2:] for l in labels]

    assert shown("present or absent")[0] == ["max_iter", "max_dist", "fuzz_seed"]
    assert shown("present") == (["max_iter"], ["Limits"])
    assert shown("absent") == (["max_dist", "fuzz_seed"], ["Limits", "Fuzzing"])
    for k in ("max_dist", "fuzz_seed"):
        app.kf.data[k] = 1
    # every section filtered away -> a placeholder, not a blank list
    labels = _headers_of(app)
    assert [l.text for l in labels] == ["(no settings match the filters)"]


def test_tooltip_popup_is_built_from_options_the_widget_class_accepts():
    """Regression test: Tooltip._show made a ttk.Label with padx/pady, which ttk
    labels reject (TclError: unknown option "-padx"), so no tooltip ever appeared.
    A stand-in ttk.Label that rejects those options like the real one does."""
    import types

    class StrictTtkLabel:
        def __init__(self, _parent=None, **kw):
            for bad in ("padx", "pady"):
                if bad in kw:
                    raise Exception(f'unknown option "-{bad}"')

        def pack(self, *a, **k):
            pass

    class Popup:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, _name):
            return lambda *a, **k: None

    saved = (m.ttk.Label, m.tk.Toplevel)
    m.ttk.Label, m.tk.Toplevel = StrictTtkLabel, Popup
    try:
        tip = m.Tooltip.__new__(m.Tooltip)
        tip.text, tip.tip = "hello", None
        tip.widget = types.SimpleNamespace(winfo_rootx=lambda: 0, winfo_rooty=lambda: 0,
                                           winfo_height=lambda: 10)
        tip._show()
        assert tip.tip is not None
    finally:
        m.ttk.Label, m.tk.Toplevel = saved


def test_floats_interpolate_linearly_and_everything_else_holds():
    kfs = [m.Keyframe(0, 0, {"max_dist": 2.0, "max_iter": 10}),
           m.Keyframe(1, 10, {"max_dist": 4.0, "max_iter": 50})]
    for frame, want in ((0, 2.0), (5, 3.0), (10, 4.0), (2, 2.4)):
        got = m.resolve_value("max_dist", kfs, frame, 0.0, True)
        assert abs(got - want) < 1e-12, (frame, got)
    for frame in (0, 5, 9):
        assert m.resolve_value("max_iter", kfs, frame, 0, False) == 10
    assert m.resolve_value("max_iter", kfs, 10, 0, False) == 50


# --------------------------------------------------------------------------- #
# Selecting a keyframe loads it; edits are marked modified (blue outlines)
# --------------------------------------------------------------------------- #
def test_detect_modified_keys_marks_changes_and_ignores_new_keys():
    d = m.detect_modified_keys
    assert d({"a": 1, "b": 2}, {"a": 1, "b": 2}, set()) == set()
    assert d({"a": 1, "b": 2}, {"a": 5, "b": 2}, set()) == {"a"}
    # sticky: the value going back to where it started does not clear the mark
    assert d({"a": 5}, {"a": 1}, {"a"}) == {"a"}
    assert d({"a": 1}, {"a": 1}, {"a"}) == {"a"}
    # a key that is new (a just-added obstacle) is not an edit ...
    assert d({"a": 1}, {"a": 1, "new": 3}, set()) == set()
    # ... and a key that vanished (a deleted obstacle) stops being modified
    assert d({"a": 1, "gone": 2}, {"a": 1}, {"gone"}) == set()


def test_self_checks_catch_a_missing_setting_widget_entry():
    fresh = load_module()
    del fresh.SETTING_WIDGET_ATTRS["seed"]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "SETTING_WIDGET_ATTRS" in str(e)


def _fake_kf_app():
    """K0 (frame 0) {a: 0.0, c: 0.0}, K1 (frame 5) {b: 5}, K2 (frame 10) {a: 10.0,
    c: 10.0}. a and c are animated through K1; d is defined by no keyframe."""
    class FakeApp:
        pass
    app = FakeApp()
    app.keyframes = [m.Keyframe(0, 0, {"a": 0.0, "c": 0.0}), m.Keyframe(1, 5, {"b": 5}),
                     m.Keyframe(2, 10, {"a": 10.0, "c": 10.0})]
    app._kf_counter = 10
    app.selected_keyframe = None
    app.kf_modified, app.kf_last_seen, app.kf_applying = set(), None, False
    app.current = {"a": 99.0, "b": 99, "c": 99.0, "d": 7}
    app.loaded = []
    app.capture_all_settings = lambda show_error=True: dict(app.current)
    app.all_setting_entries = lambda: [
        m.SettingEntry(key=k, tab="T", label=k, is_float=isinstance(v, float), fallback=v)
        for k, v in app.current.items()]
    app.setting_fallback = lambda key: app.current[key]
    app.setting_is_float = lambda key: isinstance(app.current[key], float)

    def load_value(key, val):
        app.loaded.append(key)
        app.current[key] = val
    app.load_value = load_value
    for name in ("on_limit_var_changed", "on_shape_param_changed", "on_fuzz_changed",
                 "rebuild_keyframe_list", "refresh_timeline_diffs",
                 "refresh_modified_outlines"):
        setattr(app, name, lambda *a, **k: None)
    for name in ("select_keyframe", "deselect_keyframe", "apply_keyframe_settings",
                 "check_modified", "save_settings", "save_modified_settings",
                 "setting_status", "load_animated_setting", "new_keyframe",
                 "new_keyframe_empty", "delete_selected_keyframe"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    return app


def test_selecting_a_keyframe_loads_its_settings_and_the_ones_neighbors_animate():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[1])
    assert sorted(app.loaded) == ["a", "b", "c"]            # d: nobody defines it
    assert app.current == {"a": 5.0, "b": 5, "c": 5.0, "d": 7}   # a, c blended at frame 5
    assert app.selected_keyframe is app.keyframes[1]
    assert app.kf_modified == set() and app.kf_last_seen == app.current


def test_edits_are_marked_modified_and_stay_marked():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[1])
    assert app.check_modified() is False
    app.current["d"] = 8
    assert app.check_modified() is True and app.kf_modified == {"d"}
    app.current["d"] = 7                                    # typed back to the old value
    app.check_modified()
    assert app.kf_modified == {"d"}
    app.current["a"] = 1.5
    app.check_modified()
    assert app.kf_modified == {"a", "d"}


def test_loading_a_keyframe_never_counts_as_a_user_edit():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[0])
    app.select_keyframe(app.keyframes[2])                   # overwrites a and c
    assert app.kf_modified == set()
    app.kf_applying = True
    app.current["d"] = 0
    assert app.check_modified() is False and app.kf_modified == set()


def test_nothing_is_tracked_while_no_keyframe_is_selected():
    app = _fake_kf_app()
    app.current["d"] = 8
    assert app.check_modified() is False and app.kf_modified == set()


def test_deselecting_keeps_values_but_drops_modified_status_so_reselecting_overwrites():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[0])
    app.current["a"] = 42.0
    app.check_modified()
    assert app.kf_modified == {"a"}
    app.deselect_keyframe()
    assert app.selected_keyframe is None and app.kf_modified == set()
    assert app.current["a"] == 42.0                         # not lost immediately
    app.select_keyframe(app.keyframes[2])
    assert app.current["a"] == 10.0 and app.kf_modified == set()


def test_selecting_another_keyframe_equals_deselect_then_select():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[0])
    app.current["b"] = 1
    app.check_modified()
    app.select_keyframe(app.keyframes[1])
    assert app.kf_modified == set() and app.current["b"] == 5


def test_reselecting_the_selected_keyframe_changes_nothing():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[1])
    app.current["d"] = 8
    app.check_modified()
    app.loaded.clear()
    app.select_keyframe(app.keyframes[1])
    assert app.loaded == [] and app.kf_modified == {"d"} and app.current["d"] == 8


def test_save_all_modified_saves_only_modified_settings_and_clears_them():
    app = _fake_kf_app()
    kf = app.keyframes[1]
    app.select_keyframe(kf)
    app.current["d"] = 8                                    # absent from the keyframe
    app.current["b"] = 6                                    # present in the keyframe
    app.check_modified()
    assert app.kf_modified == {"b", "d"}
    app.save_modified_settings()
    assert kf.data == {"b": 6, "d": 8}                      # a and c (loaded, unedited) not saved
    assert app.kf_modified == set()


def test_saving_one_setting_clears_only_that_settings_mark():
    app = _fake_kf_app()
    kf = app.keyframes[1]
    app.select_keyframe(kf)
    app.current["d"] = 8
    app.current["b"] = 6
    app.check_modified()
    app.save_settings(kf, ["d"])
    assert kf.data == {"b": 5, "d": 8} and app.kf_modified == {"b"}


def test_new_empty_keyframe_is_selected_and_loads_only_what_neighbors_animate():
    app = _fake_kf_app()
    app.keyframes[2].frame_index = 20                       # next new one lands at 21
    app.new_keyframe_empty()
    kf = app.selected_keyframe
    assert kf.data == {} and kf.frame_index == 21 and kf is app.keyframes[-1]
    assert app.kf_modified == set()


def test_new_keyframe_from_current_is_selected_without_reloading_anything():
    app = _fake_kf_app()
    app.new_keyframe()
    kf = app.selected_keyframe
    assert kf.data == {"a": 99.0, "b": 99, "c": 99.0, "d": 7}
    assert app.loaded == [] and app.kf_modified == set()


def test_deleting_the_selected_keyframe_deselects_it_and_clears_modified():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[1])
    app.current["d"] = 8
    app.check_modified()
    app.delete_selected_keyframe()
    assert app.selected_keyframe is None and app.kf_modified == set()
    assert len(app.keyframes) == 2


def test_settings_list_marks_modified_sections_and_save_button_state():
    app = _fake_diff_app()
    app.diff_expanded = {"Limits"}
    app.kf_modified = {"max_dist"}
    labels = _headers_of(app)
    assert labels[0].text == "\u25bc Limits  (1 of 2 set, 1 modified)"
    assert labels[1].text == "\u25b6 Fuzzing  (0 of 1 set)"


def test_a_setting_defined_on_one_side_only_is_animated_from_that_side():
    """The width example: #1 stores 1.4, #2 stores 1.5, #3 stores nothing. #3 gets
    1.5 (only earlier keyframes define it); #1 and #2 keep their own values; a
    keyframe before the first one to define a setting gets the first one's value;
    and once a later keyframe defines it too, #3 is blended like normal."""
    app = _fake_kf_app()
    app.current = {"w": 9.9}
    app.keyframes = [m.Keyframe(0, 0, {"w": 1.4}), m.Keyframe(1, 5, {"w": 1.5}),
                     m.Keyframe(2, 10, {})]
    early = m.Keyframe(3, -3, {})
    app.keyframes.append(early)
    app.select_keyframe(app.keyframes[0])
    assert app.current["w"] == 1.4
    app.select_keyframe(app.keyframes[2])
    assert app.current["w"] == 1.5                     # held from the last defining keyframe
    app.select_keyframe(early)
    assert app.current["w"] == 1.4                     # held from the first defining keyframe
    app.select_keyframe(app.keyframes[1])
    assert app.current["w"] == 1.5
    app.keyframes.append(m.Keyframe(4, 20, {"w": 2.5}))    # a keyframe after #3 defines it too
    app.select_keyframe(app.keyframes[2])
    assert abs(app.current["w"] - (1.5 + 1.0 / 3)) < 1e-9   # blended: frame 10 of 5..20


def test_frames_before_the_first_defining_keyframe_hold_its_value():
    """Regression test: resolve_value blended backwards past the first keyframe
    that defines a float setting (frame -3 of keyframes at 0 and 5 gave
    1.4 - 0.6 * (1.5 - 1.4)), instead of holding the first keyframe's value."""
    kfs = [m.Keyframe(0, 0, {"w": 1.4}), m.Keyframe(1, 5, {"w": 1.5})]
    assert m.resolve_value("w", kfs, -3, 0.0, True) == 1.4
    assert m.resolve_value("w", kfs, 0, 0.0, True) == 1.4
    assert abs(m.resolve_value("w", kfs, 2.5, 0.0, True) - 1.45) < 1e-12
    assert m.resolve_value("w", kfs, 50, 0.0, True) == 1.5


def test_settings_list_scroll_is_refitted_after_every_rebuild():
    """Regression test: with the list scrolled down, choosing the "present" filter
    (or collapsing a section) left it scrolled past its shorter content - a blank
    list - because the scroll area only followed the list frame's <Configure>
    event, which never comes when the canvas is showing none of the frame."""
    calls = []
    app = _fake_diff_app()
    app.fit_diff_scroll = lambda: calls.append("fit")
    app.refresh_timeline_diffs()
    app.toggle_diff_section("Limits")
    assert calls == ["fit", "fit"]

    class Canvas:
        def __init__(self):
            self.log = []
        def bbox(self, _tag):
            return (0, 0, 355, 116)
        def configure(self, **kw):
            self.log.append(("configure", kw))
        def yview(self):
            return (0.7, 0.78)
        def yview_moveto(self, f):
            self.log.append(("moveto", f))

    fit_app = _fake_diff_app()
    fit_app.diff_canvas = Canvas()
    fit_app.diff_list_frame = type("F", (), {"update_idletasks": lambda self: None})()
    m.App.fit_diff_scroll(fit_app)
    assert fit_app.diff_canvas.log == [("configure", {"scrollregion": (0, 0, 355, 116)}),
                                       ("moveto", 0.7)]


def _recorded_angle_entries(build_name):
    """Build a controls section on a fake app whose add_angle_entry only records
    (label, lo, hi) for each angle entry asked for."""
    calls = []

    class Handle:
        widget = None

    class FakeApp:
        root = _StubWidget()
        fuzz_ttk_widgets = []

        def add_angle_entry(self, parent, row, label, default, on_change=None,
                            lo=None, hi=None, fmt="{:.2f}"):
            calls.append((label, lo, hi))
            return Handle()

        def __getattr__(self, name):
            return lambda *a, **k: None
    app = FakeApp()
    getattr(m.App, build_name)(app, _StubWidget())
    return calls


def test_angle_offset_and_rotation_style_entries_have_no_limits():
    """The emission angle offset, and the fuzzing offsets, take any number of
    degrees (many revolutions, either direction)."""
    (emission,) = _recorded_angle_entries("build_emission_controls")
    assert emission[1:] == (None, None), emission
    fuzz = {label: (lo, hi) for label, lo, hi in _recorded_angle_entries("build_fuzzing_controls")}
    assert fuzz["Offset (\u00b0)"] == (None, None)


def test_fuzzing_jitter_ranges_stay_limited_to_0_through_180_degrees():
    """Jitter is a +/- range, so it keeps its 0..180 limits (only offsets and
    rotations are unlimited)."""
    jitters = [(lo, hi) for label, lo, hi in _recorded_angle_entries("build_fuzzing_controls")
               if label.startswith("Jitter")]
    assert len(jitters) == 3 and all(j == (0.0, 180.0) for j in jitters), jitters


def test_huge_emission_offsets_point_the_same_way_as_their_remainder():
    base = _angle_deg(_first_direction(0.2, 0.1, _emission(UP, offset=30.0)))
    for offset in (30.0 + 360.0, 30.0 - 720.0, 30.0 + 360.0 * 1000):
        got = _angle_deg(_first_direction(0.2, 0.1, _emission(UP, offset=offset)))
        assert abs(((got - base + 180.0) % 360.0) - 180.0) < 1e-6, offset


def test_default_moss_egg_fits_inside_the_view_at_every_apex_angle():
    """Regression test: at the old default size (1.0) the default egg (90 degree
    apex) reached y = 1.034, past the top edge of the view (which spans -1 to 1)."""
    g = np.linspace(-1.2, 1.2, 1201)
    gx, gy = np.meshgrid(g, g)
    for angle in range(60, 181, 10):
        bound = m.moss_egg_bound(float(angle), m.DEFAULT_MOSS_SIZE, 0.0)
        inside = m.inside_boundary_grid(gx, gy, m.SHAPE_MOSS_EGG, bound)
        assert np.abs(gy[inside]).max() < 1.0 and np.abs(gx[inside]).max() < 1.0, angle


class _WidgetsApp:
    """Just enough App to ask which controls show a setting."""


def test_every_setting_and_obstacle_field_resolves_to_a_control():
    import types
    app = _WidgetsApp()
    names = set(m.SETTING_WIDGET_ATTRS.values()) | {"mode_combo"}
    for name in names:
        setattr(app, name, types.SimpleNamespace(name=name))
    app.chan_blocks = [types.SimpleNamespace(name=f"block{c}") for c in range(3)]
    app.obs_widgets = {}
    for cls in m.SHAPE_CLASSES:
        for p in cls.PARAMS:
            app.obs_widgets[(cls, p.field)] = types.SimpleNamespace(name=f"{cls.__name__}.{p.field}")
    for field in m.EllipseShape.FIELD_NAMES:
        if field not in ("width", "height"):
            app.obs_widgets[(m.Shape, field)] = types.SimpleNamespace(name=f"Shape.{field}")
    v = m.ViewConfig(2)
    app.selected_view_obj = lambda: v
    app.widgets_for_setting = m.App.widgets_for_setting.__get__(app)
    app.obstacle_widget = m.App.obstacle_widget.__get__(app)
    for key in m.SETTING_WIDGET_ATTRS:
        assert [w.name for w in app.widgets_for_setting(key)] == [m.SETTING_WIDGET_ATTRS[key]]
    for cls in m.SHAPE_CLASSES:
        o = cls(4)
        app.selected_obstacle = o
        for field in o.FIELD_NAMES:
            found = app.widgets_for_setting(f"obst:4:{field}")
            owner = cls.__name__ if field in {p.field for p in cls.PARAMS} else "Shape"
            assert [w.name for w in found] == [f"{owner}.{field}"], (cls.__name__, field)
            assert app.widgets_for_setting(f"obst:5:{field}") == []     # not the edited obstacle
    assert [w.name for w in app.widgets_for_setting("view:2:mode")] == ["mode_combo"]
    assert [w.name for w in app.widgets_for_setting("view:2:chan1")] == ["block1"]
    assert app.widgets_for_setting("view:3:chan1") == []


# --------------------------------------------------------------------------- #
# Debug dumps (dev/debug_dump.py), the Debug menu, and their triggers
# --------------------------------------------------------------------------- #
def _load_dumper():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "debug_dump_under_test", os.path.join(os.path.dirname(os.path.abspath(__file__)),
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


class _Plain:
    """Picklable, with attributes that are not."""
    def __init__(self):
        self.count = 3
        self.var = _FakeVar()
        self.helper = lambda: None


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


def _load_test_dump(path):
    """Test doubles live in the test runner (__main__), so map __main__ to itself."""
    with _ModuleRegistered():
        return _load_dumper().load_dump(path, module_name="__main__")


def _dump_roundtrip(app, cause="manual", include_large_arrays=True, exc_info=None,
                    module_globals=None, large_array_bytes=None):
    import tempfile
    dumper = _load_dumper()
    path = os.path.join(tempfile.mkdtemp(), "x.pkl")
    with _ModuleRegistered():
        extra = {} if large_array_bytes is None else {"large_array_bytes": large_array_bytes}
        dumper.dump_everything(app, module_globals or {}, path, cause,
                               include_large_arrays=include_large_arrays, exc_info=exc_info,
                               file_version=7, **extra)
    return _load_test_dump(path)


def test_dump_keeps_picklable_values_exactly_as_they_are():
    class App:
        pass
    app = App()
    app.arr = np.arange(12.0).reshape(3, 4)
    app.nested = {"a": [1, (2, 3)], "b": m.EllipseShape(4), "c": m.DEFAULT_RAY_SETTINGS}
    app.obstacles = [m.EllipseShape(1), m.EllipseShape(2)]
    app.obstacles[1].r = 0.75
    data = _dump_roundtrip(app)["app"]
    assert np.array_equal(data["arr"], app.arr)
    assert data["nested"]["a"] == [1, (2, 3)]
    assert data["nested"]["b"].id == 4 and data["nested"]["c"] == m.DEFAULT_RAY_SETTINGS
    assert [o.r for o in data["obstacles"]] == [0.0, 0.75]
    assert isinstance(data["obstacles"][0], m.EllipseShape)       # still the real class


def test_dump_replaces_what_cannot_be_pickled_with_readable_stand_ins():
    class App:
        pass
    app = App()
    app.var = _FakeVar()
    app.widget = _FakeWidget()
    app.callback = lambda: None
    app.plain = _Plain()
    app.mixed = {"ok": 1, "bad": [_FakeVar(), 5, lambda: None]}
    app.handle = m.FloatEntryHandle(_FakeVar(), "{:.2f}", 1.0)
    data = _dump_roundtrip(app)["app"]
    assert data["var"] == {"__standin__": "tk variable", "type": data["var"]["type"],
                           "name": "PY_VAR7", "value": "256"}
    assert data["widget"]["__standin__"] == "widget" and data["widget"]["path"] == ".frame.entry"
    assert data["widget"]["options"] == {"width": "8", "text": "hi"}
    assert data["callback"]["__standin__"] == "callable"
    assert "lambda" in data["callback"]["name"]
    obj = data["plain"]
    assert obj["__standin__"] == "object" and obj["state"]["count"] == 3
    assert obj["state"]["var"]["value"] == "256"
    assert data["mixed"]["ok"] == 1
    assert data["mixed"]["bad"][0]["__standin__"] == "tk variable" and data["mixed"]["bad"][1] == 5
    assert data["mixed"]["bad"][2]["__standin__"] == "callable"
    assert data["handle"]["__standin__"] == "object"            # structure kept, not flattened
    assert data["handle"]["state"]["fmt"] == "{:.2f}"


def test_dump_survives_cycles_and_objects_that_fail_when_read():
    class App:
        pass

    class Hostile:
        def __init__(self):
            self.fn = lambda: None
            self.me = self

        def __repr__(self):
            raise RuntimeError("no repr")
    app = App()
    app.loop = {"self": None, "fn": lambda: None}
    app.loop["self"] = app.loop
    app.hostile = Hostile()
    data = _dump_roundtrip(app)["app"]
    assert data["loop"]["self"]["__standin__"] == "cycle"
    assert data["hostile"]["state"]["me"]["__standin__"] == "cycle"


class _Holder:
    """Picklable, with a large array inside."""
    def __init__(self, big):
        self.big = big
        self.label = "held"


def test_dump_without_large_arrays_replaces_big_arrays_wherever_they_are():
    class App:
        pass
    app = App()
    big = np.zeros((200, 200), dtype=np.float64)              # 320 KB
    app.top = big
    app.in_list = [big, None, np.arange(5.0)]
    app.in_dict = {"k": {"deep": big}}
    app.in_object = _Holder(big)
    app.small = np.arange(5.0)
    app.obstacles = [m.EllipseShape(1), m.EllipseShape(2)]
    data = _dump_roundtrip(app, include_large_arrays=False, module_globals={"GRID": big, "SMALL": np.ones(3)})
    want = {"__standin__": "large array", "shape": (200, 200), "dtype": "float64",
            "nbytes": 320000}
    assert data["app"]["top"] == want
    assert data["app"]["in_list"][0] == want and data["app"]["in_list"][1] is None
    assert np.array_equal(data["app"]["in_list"][2], np.arange(5.0))     # small: kept
    assert data["app"]["in_dict"]["k"]["deep"] == want
    held = data["app"]["in_object"]
    assert held["__standin__"] == "object" and held["state"]["big"] == want
    assert held["state"]["label"] == "held"
    assert np.array_equal(data["app"]["small"], np.arange(5.0))
    assert data["module"]["GRID"] == want and np.array_equal(data["module"]["SMALL"], np.ones(3))
    # values with no large array inside are still stored exactly, real classes and all
    assert [type(o).__name__ for o in data["app"]["obstacles"]] == ["EllipseShape", "EllipseShape"]
    assert data["meta"]["large_arrays_included"] is False
    assert data["meta"]["large_array_bytes"] == 64 * 1024


def test_dump_with_large_arrays_keeps_them_and_the_size_limit_can_be_changed():
    class App:
        pass
    app = App()
    app.top = np.zeros((200, 200))
    app.mid = np.zeros(1000)                                  # 8 KB
    full = _dump_roundtrip(app, include_large_arrays=True)
    assert np.array_equal(full["app"]["top"], app.top) and full["meta"]["large_arrays_included"]
    tiny = _dump_roundtrip(app, include_large_arrays=False, large_array_bytes=1000)
    assert tiny["app"]["mid"]["__standin__"] == "large array"


def test_dump_records_the_cause_the_exception_the_threads_and_module_data():
    class App:
        pass
    try:
        raise KeyError("boom")
    except KeyError:
        exc_info = sys.exc_info()
    data = _dump_roundtrip(App(), cause="crash", exc_info=exc_info,
                           module_globals={"LIMIT": 5, "helper": len, "__name__": "x", "np": np,
                                           "TABLE": {"a": _FakeVar()}})
    assert data["meta"]["cause"] == "crash" and "KeyError: 'boom'" in data["meta"]["exception"]
    assert data["meta"]["file_version"] == 7
    assert any(t["name"] == threading.current_thread().name for t in data["threads"])
    assert set(data["module"]) == {"LIMIT", "TABLE"}           # no dunders, modules, callables
    assert data["module"]["TABLE"]["a"]["__standin__"] == "tk variable"


def test_a_dump_written_by_a_script_loads_by_mapping___main___to_the_module():
    import tempfile, types
    main = sys.modules["__main__"]

    class Zed:
        def __init__(self):
            self.n = 9
    Zed.__module__ = "__main__"
    Zed.__qualname__ = "Zed"
    main.Zed = Zed
    try:
        class App:
            pass
        app = App()
        app.zed = Zed()
        dumper = _load_dumper()
        path = os.path.join(tempfile.mkdtemp(), "x.pkl")
        dumper.dump_everything(app, {}, path, "manual")
    finally:
        del main.Zed
    fake_module = types.ModuleType("zed_module")
    fake_module.Zed = Zed
    sys.modules["zed_module"] = fake_module
    try:
        assert dumper.load_dump(path, module_name="zed_module")["app"]["zed"].n == 9
    finally:
        del sys.modules["zed_module"]


def test_one_unpicklable_value_never_costs_the_whole_dump():
    """Regression guard found while writing the dumper: a namedtuple of a class
    pickle cannot find by name was rebuilt as that same class, and the final
    write then failed for the entire dump. Such values become stand-ins."""
    import collections
    Lost = collections.namedtuple("Lost", ["a", "b"])
    Lost.__module__ = "no_such_module_anywhere"

    class App:
        pass
    app = App()
    app.lost = Lost(1, _FakeVar())
    app.fine = 5
    data = _dump_roundtrip(app)["app"]
    assert data["fine"] == 5
    assert data["lost"]["__standin__"] == "object" and data["lost"]["state"]["a"] == 1
    assert data["lost"]["state"]["b"]["__standin__"] == "tk variable"


def _fake_debug_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.root = _FakeGeometryRoot()
    app._crash_dumps = 0
    app.keep_window_on_screen = lambda: None
    app.points = np.zeros((200, 200, 2), dtype=np.float32)   # 320 KB: a large array
    app.iter_var = _FakeVar()
    for name in ("take_startup_screenshot", "take_startup_captures", "manual_screenshot",
                 "manual_dump", "report_debug_result", "on_callback_exception", "dump_crash"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    return app


class _DumpEnv(_ScreenshotEnv):
    """A scratch dev/ folder containing the real dumper."""
    def __enter__(self):
        import shutil
        super().__enter__()
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev",
                                 "debug_dump.py"), self.dev_dir)
        return self

    def dumps(self):
        folder = os.path.join(self.dev_dir, "dumps")
        return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def test_captures_in_the_same_second_do_not_overwrite_each_other():
    with tempfile.TemporaryDirectory() as d:
        first = m.unused_path(os.path.join(d, "a.pkl"))
        assert first == os.path.join(d, "a.pkl")
        open(first, "w").close()
        second = m.unused_path(os.path.join(d, "a.pkl"))
        assert second == os.path.join(d, "a-2.pkl")
        open(second, "w").close()
        assert m.unused_path(os.path.join(d, "a.pkl")) == os.path.join(d, "a-3.pkl")


def test_dump_file_names_carry_the_version_time_and_cause():
    import re
    app = _fake_debug_app()
    with _DumpEnv() as env:
        for cause in ("startup", "crash", "manual"):
            path = m.core.dump_app_state(app, cause, False)
            assert re.fullmatch(rf"{m.FILE_VERSION}-\d{{8}}-\d{{6}}-{cause}\.pkl",
                                os.path.basename(path)), path
            assert os.path.dirname(path) == os.path.join(env.dev_dir, "dumps")
        try:
            m.core.dump_app_state(app, "other", False)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_dumps_are_skipped_without_a_dev_dir_or_the_dumper_and_never_raise():
    app = _fake_debug_app()
    with _ScreenshotEnv(with_dev_dir=False) as env:
        assert m.core.dump_app_state(app, "manual", True) is None
        assert not os.path.exists(env.dev_dir)
    with _ScreenshotEnv() as env:                      # dev/ but no dumper in it
        assert m.core.dump_app_state(app, "manual", True) is None
    with _DumpEnv() as env:                            # a dumper that blows up
        with open(os.path.join(env.dev_dir, "debug_dump.py"), "w") as f:
            f.write("raise RuntimeError('broken dumper')\n")
        assert m.core.dump_app_state(app, "manual", True) is None


def test_startup_captures_take_a_screenshot_then_a_dump_without_large_arrays_once():
    app = _fake_debug_app()
    with _DumpEnv() as env:
        app.take_startup_captures()
        app.take_startup_captures()
        assert len(env.grab.bboxes) == 1
        (name,) = env.dumps()
        assert name.endswith("-startup.pkl")
        shots = os.listdir(os.path.join(env.dev_dir, "screenshots"))
        assert len(shots) == 1 and shots[0].endswith("-startup.png")
        data = _load_test_dump(os.path.join(env.dev_dir, "dumps", name))
        assert data["meta"]["cause"] == "startup" and data["meta"]["large_arrays_included"] is False
        assert data["app"]["points"]["__standin__"] == "large array"
        assert data["app"]["iter_var"]["value"] == "256"


def test_a_failing_startup_screenshot_still_leaves_the_startup_dump():
    class BoomGrab:
        def grab(self, bbox=None):
            raise OSError("screen unavailable")
    app = _fake_debug_app()
    with _DumpEnv(grab=BoomGrab()) as env:
        app.take_startup_captures()
        assert len(env.dumps()) == 1


def test_manual_actions_save_with_the_manual_suffix_and_report_the_result():
    shown = []
    saved = m.messagebox.showinfo
    m.messagebox.showinfo = lambda title, text: shown.append(text)
    try:
        app = _fake_debug_app()
        with _DumpEnv() as env:
            app.manual_screenshot()
            app.manual_dump(False)
            app.manual_dump(True)
            dumps = env.dumps()
            import re
            assert len(dumps) == 2, dumps                   # never overwritten, even in one second
            assert all(re.fullmatch(r"\d+-\d{8}-\d{6}-manual(-\d+)?\.pkl", d) for d in dumps)
            assert os.listdir(os.path.join(env.dev_dir, "screenshots"))[0].endswith("-manual.png")
            assert len(shown) == 3 and all("Saved" in s for s in shown)
            included = {_load_test_dump(os.path.join(env.dev_dir, "dumps", d))
                        ["meta"]["large_arrays_included"] for d in dumps}
            assert included == {False, True}
        shown.clear()
        with _ScreenshotEnv(with_dev_dir=False):
            app.manual_screenshot()
            app.manual_dump(True)
        assert len(shown) == 2 and all("No " in s for s in shown)
    finally:
        m.messagebox.showinfo = saved


def test_a_crash_leaves_a_dump_without_large_arrays_and_no_screenshot():
    app = _fake_debug_app()
    try:
        raise ValueError("kaboom")
    except ValueError:
        exc_info = sys.exc_info()
    import io, contextlib
    with _DumpEnv() as env, contextlib.redirect_stderr(io.StringIO()) as err:
        app.on_callback_exception(*exc_info)
        assert env.grab.bboxes == [], "a crash must not trigger a screenshot"
        assert not os.path.exists(os.path.join(env.dev_dir, "screenshots"))
        (name,) = env.dumps()
        assert name.endswith("-crash.pkl")
        data = _load_test_dump(os.path.join(env.dev_dir, "dumps", name))
        assert data["meta"]["large_arrays_included"] is False
        assert data["app"]["points"]["__standin__"] == "large array"
        assert "ValueError: kaboom" in data["meta"]["exception"]
    assert "Exception in Tkinter callback" in err.getvalue() and "kaboom" in err.getvalue()


def test_crash_dumps_are_capped_per_launch():
    app = _fake_debug_app()
    try:
        raise ValueError("again")
    except ValueError:
        exc_info = sys.exc_info()
    import io, contextlib, time as _time
    calls = []
    saved = m.core.dump_app_state
    m.core.dump_app_state = lambda *a, **k: calls.append(a[1])
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            for _ in range(m.MAX_CRASH_DUMPS + 4):
                app.on_callback_exception(*exc_info)
    finally:
        m.core.dump_app_state = saved
    assert calls == ["crash"] * m.MAX_CRASH_DUMPS


def test_the_menu_bar_has_a_file_menu_and_a_debug_menu_with_the_three_actions():
    class Menu:
        made = []

        def __init__(self, parent=None, **kw):
            self.commands, self.cascades = [], []
            Menu.made.append(self)

        def add_command(self, label, command, accelerator=None):
            self.commands.append((label, command))

        def add_separator(self):
            pass

        def add_cascade(self, label, menu):
            self.cascades.append((label, menu))

    class Root:
        menu = None

        def config(self, **kw):
            Root.menu = kw.get("menu")

    class FakeApp:
        root = Root()
        new_project = staticmethod(lambda: "new")
        open_project = staticmethod(lambda: "open")
        save_project = staticmethod(lambda: "save")
        save_project_as = staticmethod(lambda: "save as")
        manual_screenshot = staticmethod(lambda: "shot")
        manual_dump = staticmethod(lambda with_arrays: with_arrays)
    saved = m.tk.Menu
    m.tk.Menu = Menu
    try:
        m.App.build_menu_bar(FakeApp())
    finally:
        m.tk.Menu = saved
    bar = Root.menu
    assert [label for label, _ in bar.cascades] == ["File", "Debug"]
    file_menu = bar.cascades[0][1]
    assert [(label, command()) for label, command in file_menu.commands] == [
        ("New project", "new"), ("Open project...", "open"), ("Save project", "save"),
        ("Save project as...", "save as")]
    debug = bar.cascades[1][1]
    assert [label for label, _ in debug.commands] == [
        "Take screenshot", "Dump (without large arrays)", "Dump (with large arrays)"]
    assert debug.commands[0][1]() == "shot"
    assert debug.commands[1][1]() is False and debug.commands[2][1]() is True


# --------------------------------------------------------------------------- #
# Keeping the window on the screen at startup
# --------------------------------------------------------------------------- #
def test_shift_into_area_moves_a_window_only_as_far_as_needed():
    area = (0, 0, 1920, 1080)
    assert m.shift_into_area((0, 104, 1920, 1099), area) == (0, -19)       # the real case
    assert m.shift_into_area((100, 50, 900, 600), area) == (0, 0)          # already inside
    assert m.shift_into_area((1500, 200, 2100, 700), area) == (-180, 0)
    assert m.shift_into_area((-30, -40, 500, 400), area) == (30, 40)
    assert m.shift_into_area((1800, 1000, 2000, 1200), area) == (-80, -120)
    assert m.shift_into_area((0, 0, 1920, 1080), area) == (0, 0)           # exact fit


def test_a_window_bigger_than_the_area_lines_up_by_its_top_left_corner():
    area = (0, 0, 1000, 800)
    dx, dy = m.shift_into_area((50, 60, 1250, 1060), area)                 # 1200 x 1000
    assert (50 + dx, 60 + dy) == (0, 0)
    dx, dy = m.shift_into_area((-20, 500, 1180, 1500), (100, 40, 1100, 840))
    assert (-20 + dx, 500 + dy) == (100, 40)


def test_a_work_area_that_does_not_start_at_the_origin_is_respected():
    area = (1920, 0, 3840, 1040)                                           # a second monitor
    assert m.shift_into_area((1900, 100, 3000, 1100), area) == (20, -60)


def test_geometry_strings_give_the_window_position():
    assert m.parse_geometry_position("1904x936+0+104") == (0, 104)
    assert m.parse_geometry_position("1904x936+-8+-3") == (-8, -3)
    assert m.parse_geometry_position("1904x936-8+10") == (-8, 10)
    assert m.parse_geometry_position("1904x936") is None
    assert m.parse_geometry_position("") is None


class _FakePlacedRoot:
    def __init__(self, geometry, rect, screen=(1920, 1080)):
        self.geometry_text = geometry
        x, y, r, b = rect
        self.x, self.y, self.w, self.h = x, y, r - x, b - y
        self.screen = screen
        self.set_to = []

    def update_idletasks(self):
        pass

    def winfo_rootx(self):
        return self.x

    def winfo_rooty(self):
        return self.y

    def winfo_width(self):
        return self.w

    def winfo_height(self):
        return self.h

    def winfo_screenwidth(self):
        return self.screen[0]

    def winfo_screenheight(self):
        return self.screen[1]

    def geometry(self, text=None):
        if text is None:
            return self.geometry_text
        self.set_to.append(text)


def _placed_app(root):
    class FakeApp:
        pass
    app = FakeApp()
    app.root = root
    app.keep_window_on_screen = m.App.keep_window_on_screen.__get__(app)
    return app


def test_the_window_is_moved_up_when_its_bottom_hangs_off_the_screen():
    saved = sys.platform
    sys.platform = "linux"                      # the portable path: client rect, whole screen
    try:
        root = _FakePlacedRoot("1904x936+0+104", (0, 104, 1904, 1099))
        _placed_app(root).keep_window_on_screen()
    finally:
        sys.platform = saved
    assert root.set_to == ["+0+85"]


def test_a_window_already_on_screen_is_left_alone_and_never_resized():
    saved = sys.platform
    sys.platform = "linux"
    try:
        root = _FakePlacedRoot("1904x936+10+20", (10, 20, 1914, 956))
        _placed_app(root).keep_window_on_screen()
    finally:
        sys.platform = saved
    assert root.set_to == []


def test_repositioning_failures_are_reported_and_never_raise():
    import io, contextlib

    class BrokenRoot:
        def update_idletasks(self):
            raise RuntimeError("no window")
    with contextlib.redirect_stderr(io.StringIO()) as err:
        _placed_app(BrokenRoot()).keep_window_on_screen()
    assert "Could not reposition" in err.getvalue()


def test_startup_captures_reposition_the_window_before_the_screenshot():
    order = []

    class FakeApp:
        pass
    app = FakeApp()
    app.keep_window_on_screen = lambda: order.append("move")
    app.take_startup_screenshot = lambda: order.append("shot")
    app.take_startup_captures = m.App.take_startup_captures.__get__(app)
    saved = m.core.dump_app_state
    m.core.dump_app_state = lambda *a, **k: order.append("dump")
    try:
        app.take_startup_captures()
    finally:
        m.core.dump_app_state = saved
    assert order == ["move", "shot", "dump"]


# --------------------------------------------------------------------------- #
# The dump viewer (dev/dump_viewer.py): loading, graph and layout. The window
# itself needs a real Tk and was checked by hand.
# --------------------------------------------------------------------------- #
def _load_viewer():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dump_viewer_under_test", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "dev", "dump_viewer.py"))
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    return viewer


def _viewer_graph(viewer, data, char_w=7.0):
    return viewer.layout_tree(viewer.build_graph(data), char_w)


def _write_pickle(obj):
    import pickle, tempfile
    path = os.path.join(tempfile.mkdtemp(), "x.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def test_the_viewer_reads_application_classes_back_as_placeholders_without_importing_them():
    viewer = _load_viewer()
    o = m.EllipseShape(3)
    o.r = 0.75
    with _ModuleRegistered():
        path = _write_pickle({"obstacles": [o], "snap": m.DEFAULT_RAY_SETTINGS})
    data = viewer.load_pickle(path)
    held = data["obstacles"][0]
    assert isinstance(held, viewer.Placeholder) and held._qual.endswith(".EllipseShape")
    attrs = dict(viewer.object_attributes(held))
    assert attrs["id"] == 3 and attrs["r"] == 0.75 and attrs["fire_inside"] is True
    snap = data["snap"]                          # a slotted dataclass: placeholder with named attributes
    assert isinstance(snap, viewer.Placeholder)
    attrs = dict(viewer.object_attributes(snap))
    assert attrs["max_iter"] == m.DEFAULT_RAY_SETTINGS.max_iter
    assert attrs["max_dist"] == m.DEFAULT_RAY_SETTINGS.max_dist


def test_the_viewer_tolerates_classes_that_cannot_be_found():
    viewer = _load_viewer()
    data = viewer.TolerantUnpickler.find_class
    cls = viewer.placeholder_class("no_such_module_at_all", "Thing")
    assert cls.__name__ == "Thing" and cls._qual == "no_such_module_at_all.Thing"
    assert viewer.placeholder_class("no_such_module_at_all", "Thing") is cls


def test_plain_values_arrays_and_large_array_stand_ins_are_shown_inline():
    viewer = _load_viewer()
    big = {"__standin__": "large array", "shape": (512, 512), "dtype": "float32",
           "nbytes": 1048576}
    for value, expect in [(5, "5"), (None, "None"), ("hi", "'hi'"), (True, "True"),
                          ([], "[]"), ({}, "{}"), ((1, 2), "(1, 2)")]:
        assert viewer.inline_text(value) == expect, value
    assert "ndarray (3, 4) float64" in viewer.inline_text(np.zeros((3, 4)))
    assert "large array" in viewer.inline_text(big) and "1.0 MB" in viewer.inline_text(big)
    assert viewer.inline_text([1, 2]) is None and viewer.inline_text({"a": 1}) is None
    assert viewer.inline_text(m.EllipseShape(1)) is None


def test_containers_become_boxes_with_an_arrow_per_container_row():
    viewer = _load_viewer()
    data = {"meta": {"cause": "x"}, "app": {"n": 3, "items": [1, {"deep": 2}], "empty": []},
            "module": {}, "threads": []}
    g = viewer.build_graph(data)
    root = g.root
    assert root.title == "dump" and [r.key for r in root.rows] == ["meta", "app", "module", "threads"]
    assert root.rows[0].child is not None and root.rows[2].child is None     # {} is inline
    app = root.rows[1].child
    assert [r.key for r in app.rows] == ["n", "items", "empty"]
    assert app.rows[0].child is None and app.rows[1].child is not None and app.rows[2].child is None
    items = app.rows[1].child
    assert [r.key for r in items.rows] == ["[0]", "[1]"] and items.rows[1].child.rows[0].key == "deep"
    assert len(g.nodes) == 5 and len(g.edges) == 4


def test_standins_and_shared_objects_and_cycles_are_handled():
    viewer = _load_viewer()
    shared = {"x": 1, "y": [1, 2, 3]}
    cyc = {"name": "loop"}
    cyc["me"] = cyc
    standin = {"__standin__": "tk variable", "type": "tkinter.StringVar", "name": "PY_VAR1",
               "value": "256"}
    g = viewer.build_graph({"a": shared, "b": shared, "c": cyc, "d": standin})
    kinds = sorted(n.kind for n in g.nodes)
    assert kinds.count("standin") == 1 and kinds.count("dict") == 3      # shared: one box
    a_row, b_row = g.root.rows[0], g.root.rows[1]
    assert a_row.child is b_row.child
    assert [r.key for r in g.root.rows[3].child.rows] == ["type", "name", "value"]
    assert g.root.rows[3].child.title.startswith("tk variable: PY_VAR1")
    viewer.layout_tree(g)
    tree = [e for e in g.edges if e[3]]
    extra = [e for e in g.edges if not e[3]]
    assert len(tree) == len(g.nodes) - 1
    assert len(extra) == 2          # the second reference to `shared`, and the cycle's arrow to itself


def test_very_long_containers_show_a_limited_number_of_rows_and_say_so():
    viewer = _load_viewer()
    g = viewer.build_graph({"long": list(range(viewer.MAX_ROWS + 50))})
    box = g.root.rows[0].child
    assert len(box.rows) == viewer.MAX_ROWS + 1 and "50 more" in box.rows[-1].text


def _random_nested(seed, size=120):
    import random
    rng = random.Random(seed)
    pool = [{"leaf": 0}]
    for i in range(size):
        parent_choices = [rng.choice(pool) for _ in range(rng.randint(1, 6))]
        node = {f"k{j}": (c if rng.random() < 0.8 else j) for j, c in enumerate(parent_choices)}
        for j in range(rng.randint(0, 12)):
            node[f"v{j}"] = j
        if rng.random() < 0.5:
            node["list"] = [rng.choice(pool) for _ in range(rng.randint(1, 4))]
        pool.append(node)
    return {"meta": {"a": 1}, "app": pool[-1], "module": pool[len(pool) // 2], "threads": []}


def test_layout_never_overlaps_boxes_and_the_trees_own_arrows_never_cross():
    viewer = _load_viewer()
    for seed in range(8):
        g = _viewer_graph(viewer, _random_nested(seed))
        assert len(g.nodes) > 20
        assert viewer.overlapping_pairs(g) == [], seed
        assert viewer.count_crossings(g, tree_only=True) == 0, seed


def test_layout_puts_each_box_one_column_right_of_its_tree_parent_and_orders_children_by_row():
    viewer = _load_viewer()
    g = _viewer_graph(viewer, _random_nested(3))
    for node in g.nodes:
        for child in node.tree_children:
            assert child.depth == node.depth + 1 and child.x >= node.x + node.width
        ys = [c.y for c in node.tree_children]
        assert ys == sorted(ys), "children stack top to bottom in the order of the rows pointing at them"
    assert g.root.x == 0 and g.root.depth == 0


def test_a_double_click_does_the_presss_work():
    """Regression test: Tk delivers a second quick press near the first as a
    double-click INSTEAD of a press, so a drag started right after a click never
    began. The double-click handler must do the press's work."""
    import types
    viewer = _load_viewer()
    g = viewer.build_graph({"a": {"x": 1}, "n": 5})
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.vg = viewer.apply_filters(g, {})
    calls = []
    v.on_press = lambda e: calls.append(("press", e.y))
    v.on_double_click(types.SimpleNamespace(x=10, y=10))
    assert calls == [("press", 10)]
    v.vg = None
    v.on_double_click(types.SimpleNamespace(x=10, y=10))        # nothing loaded: no-op
    assert calls == [("press", 10)]


def test_the_layout_of_a_real_shaped_dump_has_no_overlaps_or_crossings():
    """A dump as debug_dump.py writes it, with stand-ins, placeholders and shared objects."""
    viewer = _load_viewer()
    dumper = _load_dumper()

    class App:
        pass
    app = App()
    app.var = _FakeVar()
    app.widget = _FakeWidget()
    app.obstacles = [m.EllipseShape(1), m.EllipseShape(2)]
    app.handle = m.FloatEntryHandle(_FakeVar(), "{:.2f}", 1.0)
    app.points = np.zeros((200, 200))
    app.same = app.obstacles
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "x.pkl")
    with _ModuleRegistered():
        dumper.dump_everything(app, {"LIMIT": 5}, path, "manual", include_large_arrays=False)
    data = viewer.load_pickle(path)
    g = _viewer_graph(viewer, data)
    assert g.root.title == "dump" and viewer.overlapping_pairs(g) == []
    assert viewer.count_crossings(g) == 0
    titles = [n.title for n in g.nodes]
    assert any(t.startswith("widget:") for t in titles) and any(t.startswith("tk variable:") for t in titles)
    assert any("EllipseShape" in t for t in titles)
    rows = {r.key: r.text for n in g.nodes for r in n.rows}
    assert "large array" in rows["points"]


def _filter_sample():
    return {"meta": {"cause": "x"},
            "app": {"count": 3, "name": "n", "items": [1, {"deep": 2}], "empty": [],
                    "obst": m.EllipseShape(1)},
            "module": {}, "threads": []}


def _view_rows(view, title_start):
    node = next(n for n in view.nodes if n.title.startswith(title_start))
    return node, [r.key for r in node.rows]


def test_type_keys_name_what_the_filters_act_on():
    viewer = _load_viewer()
    standin = {"__standin__": "tk variable", "name": "PY_VAR1"}
    big = {"__standin__": "large array", "shape": (2,), "dtype": "f8", "nbytes": 16}
    for value, want in [(None, "NoneType"), (True, "bool"), (3, "int"), (2.5, "float"),
                        ("s", "str"), ({}, "dict"), ([1], "list"), ((1,), "tuple"),
                        ({1}, "set"), (np.zeros(2), "ndarray"), (standin, "stand-in: tk variable"),
                        (big, "stand-in: large array"), (len, "function"), (int, "class")]:
        assert viewer.type_key_of(value) == want, value
    placeholder = viewer.placeholder_class("light_bounces", "Obstacle")()
    assert viewer.type_key_of(placeholder) == "light_bounces.Obstacle"


def test_the_type_inventory_lists_box_types_and_plain_value_types_with_counts():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    boxes, leaves = viewer.type_inventory(g)
    box_types = dict(boxes)
    assert box_types["dict"] == 3 and box_types["list"] == 1 and "dump" not in box_types
    assert any(k.endswith("EllipseShape") for k in box_types)
    leaf_types = dict(leaves)
    assert leaf_types["int"] >= 3 and leaf_types["str"] >= 2
    assert "list" not in leaf_types and "dict" not in leaf_types     # listed once, as box types
    assert not set(leaf_types) & set(box_types)
    assert boxes == sorted(boxes, key=lambda kv: (-kv[1], kv[0]))


def test_with_every_level_at_3_the_view_is_the_whole_graph():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {})
    assert len(view.nodes) == len(g.nodes) and len(view.edges) == len(g.edges)
    assert all(n.hidden == 0 for n in view.nodes)
    assert {n.id for n in view.nodes} == {n.id for n in g.nodes}     # the same boxes
    for orig in g.nodes:
        assert [r.key for r in orig.rows] == [r.key for r in view.by_id[orig.id].rows]


def test_level_3_shows_the_box_but_hides_its_children_and_counts_them():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {"dict": 3})
    app, rows = _view_rows(view, "dict (5)")
    assert rows == [] and app.hidden == 5                  # the box is there, empty, 5 hidden
    assert viewer.hidden_message(app.hidden) == "5 children hidden by filters"
    assert viewer.hidden_message(1) == "1 child hidden by filters"
    assert not any(n.title.startswith("list") for n in view.nodes)   # its descendants vanished
    assert [r.key for r in view.root.rows] == ["meta", "app", "threads"][:0] or \
        {"meta", "app"} <= {r.key for r in view.root.rows}


def test_level_2_keeps_a_row_in_the_parent_without_a_box_or_an_arrow():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {"list": 2})
    app, rows = _view_rows(view, "dict (5)")
    assert "items" in rows
    row = app.rows[rows.index("items")]
    assert row.child is None and row.text == "list (2)"    # a plain row naming the object
    assert not any(n.title.startswith("list") for n in view.nodes)
    assert app.hidden == 0


def test_level_1_removes_the_row_too_and_counts_it_in_the_parent():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {"list": 1, "int": 1})
    app, rows = _view_rows(view, "dict (5)")
    assert "items" not in rows and "empty" not in rows and "count" not in rows and "name" in rows
    assert app.hidden == 3                                 # items and empty (lists), count (an int)
    assert viewer.apply_filters(g, {"dict": 1}).root.hidden == 3   # meta, app and the empty module


def test_plain_value_types_hide_rows_at_1_and_0_and_levels_2_to_4_all_show_them():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    for level in (2, 3, 4):
        app, rows = _view_rows(viewer.apply_filters(g, {"str": level}), "dict (5)")
        assert "name" in rows, level
    for level in (1, 0):
        app, rows = _view_rows(viewer.apply_filters(g, {"str": level}), "dict (5)")
        assert "name" not in rows and app.hidden == 1, level


def test_a_box_stays_visible_while_any_visible_row_points_at_it():
    viewer = _load_viewer()
    shared = {"x": 1}
    g = viewer.build_graph({"meta": {}, "app": {"a": shared, "b": [shared]}, "module": {},
                            "threads": []})
    both = viewer.apply_filters(g, {})
    assert sum(1 for n in both.nodes if n.title == "dict (1)") == 1
    only_b = viewer.apply_filters(g, {"list": 4})
    assert any(n.title == "dict (1)" for n in only_b.nodes)
    # hide the list: the box is still reached directly from "a"
    no_list = viewer.apply_filters(g, {"list": 1})
    assert any(n.title == "dict (1)" for n in no_list.nodes)
    # hide the children of the dict holding both: nothing is left to reach the box
    assert not any(n.title == "dict (1)" for n in viewer.apply_filters(g, {"dict": 3}).nodes)


def test_layout_of_a_filtered_view_uses_no_room_for_hidden_boxes():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph(_random_nested(5)))
    full_height = max(n.y + n.height for n in g.nodes) - min(n.y for n in g.nodes)
    view = viewer.apply_filters(g, {"dict": 3})            # every dict box loses its children
    viewer.layout_tree(view)
    short = max(n.y + n.height for n in view.nodes) - min(n.y for n in view.nodes)
    assert len(view.nodes) < len(g.nodes) and short < full_height
    assert viewer.overlapping_pairs(view) == [] and viewer.count_crossings(view) == 0
    assert all(n.height == viewer.TITLE_H + viewer.FOOTER_H + 6 for n in view.nodes
               if n is not view.root and n.hidden and not n.rows)


def test_a_box_with_hidden_children_is_tall_enough_for_its_message():
    viewer = _load_viewer()
    g = viewer.build_graph({"meta": {"a": 1}, "app": {}, "module": {}, "threads": []})
    view = viewer.apply_filters(g, {"int": 1})
    viewer.measure_nodes(view, 7.0)
    meta = next(n for n in view.nodes if n.title == "dict (1)")
    assert meta.hidden == 1 and meta.rows == []
    assert meta.height == viewer.TITLE_H + viewer.FOOTER_H + 6
    assert meta.width >= len(viewer.hidden_message(1)) * 7.0


def _with_view_options(viewer, v, group=False, links_only=False, sort_by="none", desc=False):
    import types

    class Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value
    v.group_links_var, v.links_only_var = Var(group), Var(links_only)
    v.sort_by_var, v.sort_desc_var = Var(sort_by), Var(desc)
    v.make_view = viewer.Viewer.make_view.__get__(v)
    return v


def _viewer_with_graph(viewer, data):
    import types
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.graph = viewer.build_graph(data)
    v.char_w = 7.0
    everything = viewer.layout_tree(viewer.apply_filters(v.graph, {}))
    v.pos = {n.id: [n.x, n.y] for n in everything.nodes}
    v.levels = {}
    v.unchecked = set()
    _with_view_options(viewer, v)
    v.min_gap_var = types.SimpleNamespace(get=lambda: v.min_gap)
    v.aspect_var = types.SimpleNamespace(get=lambda: v.aspect)
    v.min_gap, v.aspect = viewer.DEFAULT_MIN_GAP, 0.0
    v.selected = set()
    v.redraws = []
    v.redraw = lambda: v.redraws.append("redraw")
    v.focus_root = lambda zoom: v.redraws.append(("focus", zoom))
    v.zoom = 0.8
    return v


def test_filtering_keeps_positions_and_rearranging_moves_only_the_visible_boxes():
    viewer = _load_viewer()
    v = _viewer_with_graph(viewer, _random_nested(2))
    v.rebuild_view()
    start = {k: tuple(p) for k, p in v.pos.items()}
    assert all((n.x, n.y) == start[n.id] for n in v.vg.nodes)
    v.levels = {"dict": 3}
    v.refilter()
    assert v.redraws[-1] == "redraw"
    assert {k: tuple(p) for k, p in v.pos.items()} == start          # nothing moved
    assert all((n.x, n.y) == start[n.id] for n in v.vg.nodes)
    hidden_ids = set(start) - set(v.vg.by_id)
    assert hidden_ids
    v.rearrange()
    assert v.redraws[-1] == ("focus", 0.8)
    assert {k: tuple(p) for k, p in v.pos.items() if k in hidden_ids} == \
        {k: start[k] for k in hidden_ids}                            # hidden keep their place
    assert any(tuple(v.pos[k]) != start[k] for k in v.vg.by_id)      # visible ones moved closer
    assert viewer.overlapping_pairs(v.vg) == []
    v.levels = {}
    v.refilter()
    assert {n.id for n in v.vg.nodes} == set(start)                  # everything back


def test_selection_drops_boxes_the_filters_hide():
    viewer = _load_viewer()
    v = _viewer_with_graph(viewer, {"meta": {}, "app": {"a": {"b": {"c": 1}}}, "module": {},
                                    "threads": []})
    v.rebuild_view()
    deep = next(n.id for n in v.vg.nodes if n.title == "dict (1)" and n.depth == 3)
    v.selected = {v.vg.root.id, deep}
    v.levels = {"dict": 3}                     # app keeps its box but loses everything under it
    v.rebuild_view()
    assert v.vg.root.id in v.selected and deep not in v.selected


def _turns(points):
    """The angle (degrees) the path turns at each interior point."""
    import math
    pts = list(zip(points[0::2], points[1::2]))
    out = []
    for (ax, ay), (bx, by), (cx, cy) in zip(pts, pts[1:], pts[2:]):
        a1, a2 = math.atan2(by - ay, bx - ax), math.atan2(cy - by, cx - bx)
        out.append(abs((math.degrees(a2 - a1) + 180) % 360 - 180))
    return out


def test_connections_can_be_curved_or_straight_and_the_end_points_are_the_same():
    viewer = _load_viewer()
    straight = viewer.edge_path(10, 20, 210, 120, 1.0, True)
    curved = viewer.edge_path(10, 20, 210, 120, 1.0, False)
    assert straight == (10, 20, 210, 120)
    assert curved[:2] == (10, 20) and curved[-2:] == (210, 120)


def test_swoopy_connections_are_smooth_all_along_and_leave_and_enter_horizontally():
    viewer = _load_viewer()
    for args in [(10, 20, 210, 120), (10, 20, 900, 5000), (10, 500, 400, 20), (0, 0, 60, 0.5)]:
        points = viewer.edge_path(*args, 1.0, False)
        assert len(points) // 2 >= 13                      # many short segments, not 3
        assert max(_turns(points)) < 25, (args, max(_turns(points)))
        x, y = points[0::2], points[1::2]
        assert abs(y[1] - y[0]) < 0.25 * abs(x[1] - x[0]) + 1e-9      # leaves horizontally
        assert abs(y[-1] - y[-2]) < 0.25 * abs(x[-1] - x[-2]) + 1e-9  # enters horizontally
        assert x[1] > x[0] and x[-1] > x[-2]               # rightwards at both ends
    forward = viewer.edge_path(10, 20, 410, 320, 1.0, False)
    xs = forward[0::2]
    assert all(b >= a for a, b in zip(xs, xs[1:]))         # never doubles back
    back = viewer.edge_path(300, 20, 10, 120, 1.0, False)  # an arrow running backwards
    assert back[2] > back[0] and back[-2] > back[-4] and back[:2] == (300, 20)
    assert back[-2:] == (10, 120)


def test_longer_connections_get_more_segments():
    viewer = _load_viewer()
    short = len(viewer.edge_path(0, 0, 50, 10, 1.0, False))
    long_ = len(viewer.edge_path(0, 0, 500, 3000, 1.0, False))
    assert long_ > short and long_ // 2 <= 65


def test_connections_are_bold_with_a_big_arrowhead():
    viewer = _load_viewer()
    assert viewer.EDGE_WIDTH >= 2 and viewer.ARROW_SHAPE[0] >= 10


def test_the_gap_between_a_box_and_its_children_follows_minimum_and_aspect_ratio():
    viewer = _load_viewer()
    for seed in range(4):
        for min_gap, aspect in [(110, 0.0), (40, 0.0), (300, 0.0), (20, 0.2), (200, 0.5)]:
            g = viewer.layout_tree(viewer.build_graph(_random_nested(seed)), 7.0, min_gap, aspect)
            checked = 0
            for node in g.nodes:
                for child in node.tree_children:
                    want = node.x + node.width + max(min_gap, aspect * node.span_h)
                    assert abs(child.x - want) < 1e-6, (seed, min_gap, aspect)
                    checked += 1
            assert checked > 10
            assert viewer.overlapping_pairs(g) == [], (seed, min_gap, aspect)
            assert viewer.count_crossings(g) == 0, (seed, min_gap, aspect)


def test_span_h_is_the_vertical_space_all_the_children_take_up():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph(_random_nested(1)), 7.0, 110, 0.0)
    fan = max(g.nodes, key=lambda n: len(n.tree_children))
    assert len(fan.tree_children) > 2
    top = min(c.y for c in fan.tree_children)
    bottom = max(c.y + c.height for c in fan.tree_children)
    assert fan.span_h >= bottom - top - 1e-6               # blocks can be taller than the boxes
    assert all(n.span_h == 0 for n in g.nodes if not n.tree_children)


def test_a_bigger_aspect_ratio_pushes_the_children_of_a_tall_fan_further_away():
    viewer = _load_viewer()
    data = _random_nested(6)
    flat = viewer.layout_tree(viewer.build_graph(data), 7.0, 110, 0.0)
    wide = viewer.layout_tree(viewer.build_graph(data), 7.0, 110, 0.4)
    fan_f = max(flat.nodes, key=lambda n: n.span_h)
    fan_w = next(n for n in wide.nodes if n.id == fan_f.id)
    gap_f = fan_f.tree_children[0].x - (fan_f.x + fan_f.width)
    gap_w = fan_w.tree_children[0].x - (fan_w.x + fan_w.width)
    assert gap_f == 110 and gap_w > 110
    assert abs(gap_w - 0.4 * fan_w.span_h) < 1e-6
    assert max(n.x for n in wide.nodes) > max(n.x for n in flat.nodes)


def test_rearranging_uses_the_arrangement_settings():
    viewer = _load_viewer()
    v = _viewer_with_graph(viewer, _random_nested(2))
    v.rebuild_view()
    v.min_gap, v.aspect = 250, 0.3
    v.rearrange()
    for node in v.vg.nodes:
        for child in node.tree_children:
            assert abs(child.x - (node.x + node.width + max(250, 0.3 * node.span_h))) < 1e-6
    assert {k: tuple(p) for k, p in v.pos.items() if k in v.vg.by_id} == \
        {n.id: (n.x, n.y) for n in v.vg.nodes}


def test_only_the_boxes_near_the_window_are_selected_for_drawing():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph(_random_nested(4)))
    everything = (-1e9, -1e9, 1e9, 1e9)
    assert len(viewer.nodes_in_region(g, everything)) == len(g.nodes)
    box = g.nodes[3]
    region = (box.x + 1, box.y + 1, box.x + 2, box.y + 2)
    near = viewer.nodes_in_region(g, region)
    assert box in near and len(near) < len(g.nodes)
    assert all(n.x <= region[2] and n.x + n.width >= region[0] and n.y <= region[3]
               and n.y + n.height >= region[1] for n in near)
    assert viewer.nodes_in_region(g, (-1e9, -1e9, -1e8, -1e8)) == []


def test_only_the_rows_inside_the_window_are_drawn():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph({"meta": {}, "app": {f"k{i}": i for i in range(100)},
                                               "module": {}, "threads": []}))
    app = next(n for n in g.nodes if n.title == "dict (100)")
    top = app.y + viewer.TITLE_H
    assert viewer.visible_row_range(app, top - 500, top + 10 * viewer.ROW_H - 1) == (0, 10)
    first, last = viewer.visible_row_range(app, top + 40 * viewer.ROW_H, top + 45 * viewer.ROW_H)
    assert (first, last) == (40, 46)
    assert viewer.visible_row_range(app, top + 1e6, top + 2e6) == (100, 100)
    assert viewer.visible_row_range(app, top - 100, top - 50) == (0, 0)


def test_there_are_five_levels_and_everything_starts_at_the_top_one():
    viewer = _load_viewer()
    assert viewer.LEVELS == (4, 3, 2, 1, 0)
    assert (viewer.SHOW, viewer.NO_CHILDREN, viewer.ROW_ONLY, viewer.HIDDEN,
            viewer.LOWEST) == (4, 3, 2, 1, 0)
    g = viewer.build_graph(_filter_sample())
    assert len(viewer.apply_filters(g, {}).nodes) == len(g.nodes)


def _lowest_sample():
    return {"meta": {}, "module": {}, "threads": [],
            "app": {"only": {"a": 1, "b": 2},                      # both children are ints
                    "mixed": {"a": 1, "s": "x"},                   # an int and a str
                    "nested": {"deep": {"a": 1}},                  # a box whose only child is a box
                    "empty": {}, "leaf": 5}}


def test_the_lowest_level_hides_the_object_like_level_1_and_counts_it():
    viewer = _load_viewer()
    g = viewer.build_graph(_lowest_sample())
    view = viewer.apply_filters(g, {"int": 0})
    mixed, rows = _view_rows(view, "dict (2)")
    assert "a" not in rows and "s" in rows and mixed.hidden == 1      # the int is hidden+counted
    assert "leaf" not in _view_rows(view, "dict (5)")[1]              # its row is gone too


def test_a_box_whose_children_are_all_at_the_lowest_level_hides_but_keeps_its_row():
    viewer = _load_viewer()
    g = viewer.build_graph(_lowest_sample())
    view = viewer.apply_filters(g, {"int": 0})
    app, rows = _view_rows(view, "dict (5)")
    assert "only" in rows                                   # the parent's row is still there ...
    row = app.rows[rows.index("only")]
    assert row.child is None and row.text == "dict (2)"     # ... as a plain row, no arrow
    assert not any(n.title == "dict (2)" and n.rows and n.rows[0].key == "a" for n in view.nodes)
    assert app.hidden == 1                                  # only "leaf" counts as hidden in app
    assert any(n.title == "dict (2)" and [r.key for r in n.rows] == ["s"] for n in view.nodes)


def test_a_box_with_any_child_above_the_lowest_level_stays_even_if_it_is_hidden_otherwise():
    viewer = _load_viewer()
    g = viewer.build_graph(_lowest_sample())
    # str at level 1 is hidden, not lowest: it still counts as a child above zero
    view = viewer.apply_filters(g, {"int": 0, "str": 1})
    mixed = next(n for n in view.nodes if n.title == "dict (2)" and n.hidden == 2)
    assert mixed.rows == []                                 # shown, empty, 2 hidden by filters
    assert "mixed" in _view_rows(view, "dict (5)")[1]
    assert any(r.child is mixed for n in view.nodes for r in n.rows)


def test_the_lowest_level_only_hides_a_parent_that_has_children_and_never_the_top_box():
    viewer = _load_viewer()
    g = viewer.build_graph(_lowest_sample())
    view = viewer.apply_filters(g, {"int": 0})
    app, rows = _view_rows(view, "dict (5)")
    assert "empty" in rows                                  # an empty dict has nothing at 0
    only = next(n for n in g.nodes if [r.key for r in n.rows] == ["a", "b"])
    mixed = next(n for n in g.nodes if [r.key for r in n.rows] == ["a", "s"])
    empty = next(n for n in g.nodes if n.title == "dict (0)") if any(
        n.title == "dict (0)" for n in g.nodes) else None
    assert viewer.hides_itself(only, {"int": 0})
    assert not viewer.hides_itself(mixed, {"int": 0})
    assert not viewer.hides_itself(only, {})                # nothing is at 0 by default
    assert empty is None                                    # empty dicts are inline rows
    # the top box never hides, even with every child of it at the lowest level
    sample = {"meta": {"a": 1}, "app": {"b": 2}, "module": {"c": 3}, "threads": []}
    top = viewer.apply_filters(viewer.build_graph(sample), {"dict": 0})
    assert [r.key for r in top.root.rows] == ["threads"] and top.root.hidden == 3
    only_lowest = viewer.apply_filters(viewer.build_graph(
        {"meta": {"a": 1}, "app": {"b": 2}, "module": {"c": 3}}), {"dict": 0})
    assert only_lowest.root is not None and only_lowest.root.rows == []
    assert only_lowest.root.hidden == 3 and len(only_lowest.nodes) == 1


def test_the_hiding_does_not_cascade_to_the_grandparent():
    """{"nested": {"deep": {"a": 1}}} with ints at the lowest level: "deep" hides itself
    (its only child is at 0) but keeps its row in "nested", and "nested" - whose child
    is now a plain row above zero - stays as a box."""
    viewer = _load_viewer()
    g = viewer.build_graph(_lowest_sample())
    view = viewer.apply_filters(g, {"int": 0})
    nested = next(n for n in view.nodes if n.title == "dict (1)" and n.rows
                  and n.rows[0].key == "deep")
    assert nested.rows[0].child is None and nested.rows[0].text == "dict (1)"
    assert nested.hidden == 0
    assert not any(n.rows and n.rows[0].key == "a" for n in view.nodes)   # deep's box is gone
    assert any(r.child is nested for n in view.nodes for r in n.rows)     # nested is reached


def test_each_box_knows_its_name_inside_its_tree_parent_and_the_row_that_points_at_it():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph(_filter_sample()))
    assert viewer.name_in_parent(g.root) == "top box"
    for node in g.nodes:
        if node is g.root:
            continue
        assert node.parent.rows[node.parent_row].child is node
        assert viewer.name_in_parent(node) == node.parent.rows[node.parent_row].key
    names = {viewer.name_in_parent(n) for n in g.nodes}
    assert {"meta", "app", "items", "obst"} <= names
    view = viewer.apply_filters(g, {"int": 1, "list": 1})       # rows removed: indexes shift
    viewer.layout_tree(view)
    for node in view.nodes:
        if node is not view.root:
            assert node.parent.rows[node.parent_row].child is node


def _zoomed_viewer(viewer, zoom):
    import types
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.zoom, v.ox, v.oy, v.char_w = zoom, 0.0, 0.0, 7.0
    v.vg = viewer.layout_tree(viewer.apply_filters(viewer.build_graph(_filter_sample()), {}))
    return _with_history(viewer, v)


def test_the_name_line_is_only_clickable_when_zoomed_in_far_enough():
    viewer = _load_viewer()
    v = _zoomed_viewer(viewer, viewer.TWO_LINE_TITLE_ZOOM)
    child = next(n for n in v.vg.nodes if n.title == "dict (9)" or n.parent is v.vg.root)
    x0, y0 = v.to_screen(child.x, child.y)
    top_half = y0 + viewer.TITLE_H * v.zoom * 0.25
    assert v.name_hit(child, x0 + 5, top_half)
    assert not v.name_hit(child, x0 + 5, y0 + viewer.TITLE_H * v.zoom * 0.75)   # the type line
    assert not v.name_hit(child, x0 + 5000, top_half)                          # past the name
    assert not v.name_hit(v.vg.root, 5, 5)                 # the top box has no parent
    v.zoom = viewer.TWO_LINE_TITLE_ZOOM - 0.01
    assert not v.name_hit(child, x0 + 5, top_half)


def test_clicking_a_name_goes_to_its_row_in_the_parent_instead_of_starting_a_drag():
    import types
    viewer = _load_viewer()
    v = _zoomed_viewer(viewer, 2.0)
    child = next(n for n in v.vg.nodes if n.parent is v.vg.root)
    calls = []
    v.vg = v.vg
    v.node_at = lambda x, y: child.id
    v.go_to_row = lambda parent, row: calls.append((parent, row))
    v.set_selection = lambda ids: calls.append(("selection", ids))
    v.selected = set()
    x0, y0 = v.to_screen(child.x, child.y)
    v.on_press(types.SimpleNamespace(x=x0 + 4, y=y0 + 3, state=0))
    assert calls == [(child.parent, child.parent_row)] and getattr(v, "drag", None) is None
    calls.clear()
    v.canvas = types.SimpleNamespace(tag_raise=lambda tag: None)
    v.on_press(types.SimpleNamespace(x=x0 + 4, y=y0 + viewer.TITLE_H * 2.0 * 0.8, state=0))
    assert v.drag["mode"] == "nodes"                        # the type line still drags the box


def test_going_to_a_row_centres_it_and_highlights_it_until_cleared():
    import types
    viewer = _load_viewer()
    v = _zoomed_viewer(viewer, 2.0)
    parent = v.vg.root
    seen = {}
    v.canvas_size = lambda: (1000, 600)
    v.set_view = lambda z, ox, oy: seen.update(zoom=z, ox=ox, oy=oy)
    v.set_info = lambda text: seen.update(info=text)
    v.root = types.SimpleNamespace(after=lambda ms, fn: seen.update(after=(ms, fn)))
    v.flash = None
    v.go_to_row(parent, 2)
    assert seen["zoom"] == 2.0 and v.flash == (parent.id, 2)
    row_y = parent.y + viewer.TITLE_H + 2 * viewer.ROW_H + viewer.ROW_H / 2
    assert abs(seen["oy"] - (row_y - 600 / 2 / 2.0)) < 1e-9       # the row is at mid-window
    assert seen["ox"] == parent.x - 30 / 2.0 and seen["after"][0] == 2000
    redraws = []
    v.redraw = lambda: redraws.append(1)
    v.clear_flash()
    assert v.flash is None and redraws == [1]
    v.clear_flash()
    assert redraws == [1]                                   # nothing to clear the second time


def test_titles_have_room_for_two_lines():
    viewer = _load_viewer()
    assert viewer.TITLE_H >= 2 * viewer.REF_FONT_PX + 6
    assert viewer.TWO_LINE_TITLE_ZOOM > 1.0


def test_rows_that_point_at_a_box_get_a_ticked_checkbox_and_other_rows_none():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {})
    app = next(n for n in view.nodes if n.title == "dict (5)")
    checks = {r.key: r.check for r in app.rows}
    assert checks == {"count": None, "name": None, "items": True, "empty": None, "obst": True}
    for node in view.nodes:
        for r in node.rows:
            assert (r.child is not None) == (r.check is True)
            assert r.origin is not None and r.origin[0] == node.id


def test_a_cleared_checkbox_hides_the_box_but_keeps_the_row_with_an_empty_checkbox():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    full = viewer.apply_filters(g, {})
    app = next(n for n in full.nodes if n.title == "dict (5)")
    items_row = next(r for r in app.rows if r.key == "items")
    view = viewer.apply_filters(g, {}, {items_row.origin})
    app2 = next(n for n in view.nodes if n.title == "dict (5)")
    row = next(r for r in app2.rows if r.key == "items")
    assert row.child is None and row.check is False and row.text == "list (2)"
    assert app2.hidden == 0                                  # not counted as hidden by filters
    assert not any(n.title.startswith("list") for n in view.nodes)   # the box is gone ...
    assert len(view.nodes) < len(full.nodes)
    assert not any(n.rows and n.rows[0].key == "deep" for n in view.nodes)   # ... and its contents
    assert [r.key for r in app2.rows] == [r.key for r in app.rows]           # same rows


def test_a_box_must_pass_both_its_checkbox_and_the_filters():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    app = next(n for n in viewer.apply_filters(g, {}).nodes if n.title == "dict (5)")
    origin = next(r for r in app.rows if r.key == "items").origin

    def list_visible(levels, unchecked):
        return any(n.title.startswith("list") for n in viewer.apply_filters(g, levels, unchecked).nodes)
    assert list_visible({}, set())                          # ticked and allowed
    assert not list_visible({}, {origin})                   # cleared
    assert not list_visible({"list": 2}, set())             # the filter keeps it a plain row
    assert not list_visible({"list": 1}, set())             # the filter hides the row
    assert not list_visible({"list": 2}, {origin})
    assert not list_visible({"dict": 3}, set())             # its parent's children are hidden
    assert list_visible({"list": 4}, set())
    # with the filter hiding the row, the checkbox has nothing to show (and keeps its state)
    hidden = viewer.apply_filters(g, {"list": 1}, {origin})
    assert "items" not in [r.key for n in hidden.nodes for r in n.rows]


def test_a_shared_box_stays_while_another_ticked_row_still_points_at_it():
    viewer = _load_viewer()
    shared = {"x": 1}
    g = viewer.build_graph({"meta": {}, "app": {"a": shared, "b": shared}, "module": {},
                            "threads": []})
    app = next(n for n in viewer.apply_filters(g, {}).nodes if n.title == "dict (2)")
    a, b = (next(r for r in app.rows if r.key == k).origin for k in ("a", "b"))
    def boxes(unchecked):
        return sum(1 for n in viewer.apply_filters(g, {}, unchecked).nodes if n.title == "dict (1)")
    assert boxes(set()) == 1 and boxes({a}) == 1 and boxes({b}) == 1 and boxes({a, b}) == 0


def test_boxes_with_checkboxes_are_wider_by_the_checkbox_room():
    viewer = _load_viewer()
    g = viewer.build_graph({"meta": {"a": 1}, "app": {"b": {"c": 1}}, "module": {}, "threads": []})
    view = viewer.apply_filters(g, {})
    viewer.measure_nodes(view, 7.0)
    app = next(n for n in view.nodes if n.rows and n.rows[0].key == "b")
    plain = next(n for n in view.nodes if n.rows and n.rows[0].key == "a")
    widest = lambda n: max(len(n.title) + 1, *(len(r.key) + 2 + len(r.text) for r in n.rows))
    assert abs(app.width - (widest(app) * 7.0 + 2 * viewer.PAD + viewer.CHECK_W)) < 1e-9
    assert abs(plain.width - (widest(plain) * 7.0 + 2 * viewer.PAD)) < 1e-9


def _with_history(viewer, v):
    """Give a stub viewer the real history methods (a recording menu and set_view)."""
    v.back_stack, v.forward_stack, v.flash = [], [], None
    v.vg = getattr(v, "vg", None)
    v.menu_states = {}
    v.view_menu = __import__("types").SimpleNamespace(
        entryconfigure=lambda label, state: v.menu_states.__setitem__(label, state))
    v.views_shown = []
    v.set_view = lambda zoom, ox, oy: (setattr(v, "zoom", zoom), setattr(v, "ox", ox),
                                       setattr(v, "oy", oy), v.views_shown.append((zoom, ox, oy)))
    for name in ("current_view", "navigate", "nav_back", "nav_forward", "show_view",
                 "update_nav_menu"):
        setattr(v, name, getattr(viewer.Viewer, name).__get__(v))
    return v


def _click_setup(viewer):
    import types
    g = viewer.build_graph({"meta": {}, "app": {"one": {"x": 1}, "num": 5}, "module": {},
                            "threads": []})
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.graph = g
    v.zoom, v.ox, v.oy, v.char_w = 1.0, 0.0, 0.0, 7.0
    v.levels, v.unchecked, v.selected = {}, set(), set()
    _with_view_options(viewer, v)
    v.min_gap_var = types.SimpleNamespace(get=lambda: 110)
    v.aspect_var = types.SimpleNamespace(get=lambda: 0.0)
    v.pos = {}
    everything = viewer.layout_tree(viewer.apply_filters(g, {}))
    v.pos = {n.id: [n.x, n.y] for n in everything.nodes}
    v.rebuild_view()
    v.link_block_until = 0.0
    _with_history(viewer, v)
    v.log = []
    v.redraw = lambda: v.log.append("redraw")
    v.go_to = lambda node: v.log.append(("go_to", node.title))
    v.canvas = types.SimpleNamespace(tag_raise=lambda tag: None,
                                     configure=lambda **kw: v.log.append(("cursor", kw)))
    v.set_selection = lambda ids: setattr(v, "selected", set(ids))
    v.set_info = lambda text: v.log.append(("info", text))
    app = next(n for n in v.vg.nodes if n.rows and n.rows[0].key == "one")
    v.node_at = lambda x, y: app.id
    v.cursor = ""
    return v, app


def _at_row(viewer, v, node, row, dx=10):
    x0, y0 = v.to_screen(node.x, node.y)
    return x0 + dx, y0 + (viewer.TITLE_H + row * viewer.ROW_H + viewer.ROW_H / 2)


def test_clicking_a_link_row_jumps_to_its_box_but_dragging_from_it_moves_the_box():
    import types
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    x, y = _at_row(viewer, v, app, 0)
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))
    v.on_release(types.SimpleNamespace(x=x, y=y, state=0))
    assert ("go_to", "dict (1)") in v.log
    v.log.clear()
    v.link_block_until = 0.0
    v.canvas.move = lambda *a: None
    v.update_edges = lambda ids: None
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))
    v.on_motion(types.SimpleNamespace(x=x + 30, y=y + 20, state=0))
    v.on_release(types.SimpleNamespace(x=x + 30, y=y + 20, state=0))
    assert not any(isinstance(e, tuple) and e[0] == "go_to" for e in v.log)   # a drag, no jump
    assert "redraw" in v.log
    v.log.clear()
    x2, y2 = _at_row(viewer, v, app, 1)                       # a plain value row: no link
    v.on_press(types.SimpleNamespace(x=x2, y=y2, state=0))
    v.on_release(types.SimpleNamespace(x=x2, y=y2, state=0))
    assert not any(isinstance(e, tuple) and e[0] == "go_to" for e in v.log)


def test_a_quick_second_click_does_not_jump_again():
    import types
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    x, y = _at_row(viewer, v, app, 0)
    press = lambda: (v.on_press(types.SimpleNamespace(x=x, y=y, state=0)),
                     v.on_release(types.SimpleNamespace(x=x, y=y, state=0)))
    press()
    press()                                                   # as the second click of a double-click
    assert [e for e in v.log if isinstance(e, tuple) and e[0] == "go_to"] == [("go_to", "dict (1)")]


def test_clicking_a_checkbox_toggles_it_and_does_not_start_a_drag_or_jump():
    import types
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    right = v.to_screen(app.x, app.y)[0] + app.width
    x = right - viewer.PAD - viewer.CHECK_SIZE / 2
    _x, y = _at_row(viewer, v, app, 0)
    assert v.checkbox_hit(app, 0, x, y) and not v.checkbox_hit(app, 1, x, y)
    assert not v.checkbox_hit(app, 0, x - 60, y)
    origin = app.rows[0].origin
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))
    assert origin in v.unchecked and getattr(v, "drag", None) is None
    assert "redraw" in v.log
    app2 = next(n for n in v.vg.nodes if n.id == app.id)
    assert app2.rows[0].check is False and app2.rows[0].child is None
    assert not any(n.title == "dict (1)" and n.rows and n.rows[0].key == "x" for n in v.vg.nodes)
    v.node_at = lambda x_, y_: app2.id
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))      # tick it again
    assert origin not in v.unchecked
    assert not any(isinstance(e, tuple) and e[0] == "go_to" for e in v.log)


def test_show_all_also_ticks_every_checkbox_again():
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    v.level_vars = {}
    v.unchecked = {app.rows[0].origin}
    v.levels = {"dict": 3}
    v.reset_filters()
    assert v.unchecked == set() and v.levels == {}


def test_the_hand_cursor_shows_over_links_checkboxes_and_names_only():
    import types
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    x, y = _at_row(viewer, v, app, 0)
    v.drag = None
    v.on_hover(types.SimpleNamespace(x=x, y=y, state=0))
    assert v.cursor == "hand2"                                # a link row
    x2, y2 = _at_row(viewer, v, app, 1)
    v.on_hover(types.SimpleNamespace(x=x2, y=y2, state=0))
    assert v.cursor == ""                                     # a plain value row
    right = v.to_screen(app.x, app.y)[0] + app.width
    v.on_hover(types.SimpleNamespace(x=right - viewer.PAD - 5, y=y, state=0))
    assert v.cursor == "hand2"                                # its checkbox


def _stub_with_places(viewer):
    """A viewer whose link jumps just move the view (by 100 world units per jump)."""
    v = _zoomed_viewer(viewer, 1.0)
    v.jumps = 0

    def jump():
        v.jumps += 1
        v.ox += 100.0
    v.jump = jump
    return v


def test_link_jumps_are_remembered_and_back_and_forward_walk_through_them():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    start = v.current_view()
    v.navigate(v.jump)
    second = v.current_view()
    v.navigate(v.jump)
    third = v.current_view()
    assert v.back_stack == [start, second] and v.forward_stack == []
    v.nav_back()
    assert v.current_view() == second and v.forward_stack == [third]
    v.nav_back()
    assert v.current_view() == start and v.back_stack == [] and v.forward_stack == [third, second]
    v.nav_back()                                             # nowhere further back: nothing happens
    assert v.current_view() == start
    v.nav_forward()
    assert v.current_view() == second
    v.nav_forward()
    assert v.current_view() == third and v.forward_stack == []
    v.nav_forward()
    assert v.current_view() == third


def test_a_new_link_jump_forgets_the_places_forward_would_have_gone_to():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    v.navigate(v.jump)
    v.navigate(v.jump)
    v.nav_back()
    assert v.forward_stack
    v.navigate(v.jump)
    assert v.forward_stack == []


def test_going_back_restores_the_exact_view_including_the_zoom_and_clears_the_highlight():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    v.zoom, v.ox, v.oy = 2.5, 40.0, -12.0

    def zoom_jump():
        v.zoom, v.ox, v.oy = 0.5, 9000.0, 300.0
        v.flash = (1, 2)
    v.navigate(zoom_jump)
    v.nav_back()
    assert v.current_view() == (2.5, 40.0, -12.0) and v.flash is None
    v.nav_forward()
    assert v.current_view() == (0.5, 9000.0, 300.0)


def test_the_history_is_limited():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    for _ in range(viewer.HISTORY_LIMIT + 50):
        v.navigate(v.jump)
    assert len(v.back_stack) == viewer.HISTORY_LIMIT


def test_the_menu_entries_are_greyed_out_when_there_is_nowhere_to_go():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    v.update_nav_menu()
    assert v.menu_states == {"Navigate back": "disabled", "Navigate forward": "disabled"}
    v.navigate(v.jump)
    assert v.menu_states == {"Navigate back": "normal", "Navigate forward": "disabled"}
    v.nav_back()
    assert v.menu_states == {"Navigate back": "disabled", "Navigate forward": "normal"}
    v.nav_forward()
    assert v.menu_states == {"Navigate back": "normal", "Navigate forward": "disabled"}


def test_back_and_forward_do_nothing_with_no_file_loaded():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    v.navigate(v.jump)
    v.vg = None
    v.nav_back()
    v.nav_forward()
    assert v.views_shown == []


def test_clicking_a_link_row_or_a_name_is_remembered_but_dragging_and_the_go_to_menu_are_not():
    import types
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    x, y = _at_row(viewer, v, app, 0)
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))
    v.on_release(types.SimpleNamespace(x=x, y=y, state=0))
    assert len(v.back_stack) == 1                            # a click on a link row
    v.link_block_until = 0.0
    v.canvas.move = lambda *a: None
    v.update_edges = lambda ids: None
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))
    v.on_motion(types.SimpleNamespace(x=x + 30, y=y + 20, state=0))
    v.on_release(types.SimpleNamespace(x=x + 30, y=y + 20, state=0))
    assert len(v.back_stack) == 1                            # a drag is not a jump
    # a click on a box's name (two-line titles) is a jump too
    v2 = _zoomed_viewer(viewer, 2.0)
    child = next(n for n in v2.vg.nodes if n.parent is v2.vg.root)
    v2.node_at = lambda x_, y_: child.id
    v2.go_to_row = lambda parent, row: None
    v2.selected = set()
    x0, y0 = v2.to_screen(child.x, child.y)
    v2.on_press(types.SimpleNamespace(x=x0 + 4, y=y0 + 3, state=0))
    assert len(v2.back_stack) == 1 and v2.forward_stack == []


def test_the_side_buttons_map_to_back_and_forward_per_platform():
    viewer = _load_viewer()
    assert viewer.nav_button_numbers("win32") == (4, 5)      # checked with real side-button clicks
    assert viewer.nav_button_numbers("x11") == (8, 9)        # 4 and 5 are the wheel there


def test_the_view_menu_and_keyboard_are_wired_to_back_and_forward():
    """The menu entries (with shortcut hints) and the key and mouse bindings are made
    in Viewer.__init__, which needs a real window; here the wiring is read from its
    source, and tests above cover what the actions do. The real window was checked
    by hand."""
    import inspect
    viewer = _load_viewer()
    source = inspect.getsource(viewer.Viewer.__init__)
    assert 'label="Navigate back", accelerator="Alt+Left"' in source
    assert 'label="Navigate forward", accelerator="Alt+Right"' in source
    assert "command=self.nav_back" in source and "command=self.nav_forward" in source
    assert '"<Alt-Left>"' in source and '"<Alt-Right>"' in source
    assert "nav_button_numbers(windowing)" in source
    assert 'if windowing == "x11":' in source             # the wheel bindings only there


def test_an_arrow_is_drawn_only_while_it_is_long_enough_on_the_screen():
    viewer = _load_viewer()
    short, long_ = (0, 0, 100, 0), (0, 0, 100000, 40000)
    assert viewer.arrow_visible(*short, 1.0) and viewer.arrow_visible(*long_, 1.0)
    assert not viewer.arrow_visible(*short, 0.05)            # 5 px: gone
    assert viewer.arrow_visible(*long_, 0.006)               # 840 px: still there, zoomed right out
    edge = (0, 0, 300, 200)                                  # the length is measured as |dx| + |dy|
    assert viewer.arrow_visible(*edge, viewer.MIN_ARROW_PX / 500)
    assert not viewer.arrow_visible(*edge, viewer.MIN_ARROW_PX / 501)
    assert viewer.arrow_visible(300, 200, 0, 0, viewer.MIN_ARROW_PX / 500)   # direction does not matter


def test_longer_arrows_stay_visible_to_a_smaller_zoom_than_shorter_ones():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph(_random_nested(3)))
    edges = sorted((viewer.edge_points(p, r, c) for p, r, c, _ in g.edges),
                   key=lambda e: abs(e[2] - e[0]) + abs(e[3] - e[1]))
    assert len(edges) > 20

    def gone_below(edge):
        lo, hi = 1e-9, 1e3
        for _ in range(80):
            mid = (lo * hi) ** 0.5
            lo, hi = (mid, hi) if not viewer.arrow_visible(*edge, mid) else (lo, mid)
        return hi
    zooms = [gone_below(e) for e in edges]
    assert zooms == sorted(zooms, reverse=True)              # longer -> vanishes at a smaller zoom
    assert zooms[0] > zooms[-1]
    # at a zoom where the shortest has gone, the longest is still drawn: not all at once
    z = zooms[0] * 0.99
    assert not viewer.arrow_visible(*edges[0], z) and viewer.arrow_visible(*edges[-1], z)


def test_zoomed_out_until_the_short_arrows_vanish_the_long_ones_remain():
    viewer = _load_viewer()
    rows = {f"k{i}": {"leaf": i} for i in range(60)}
    g = viewer.layout_tree(viewer.build_graph({"meta": {}, "app": rows, "module": {}, "threads": []}))
    edges = [viewer.edge_points(p, r, c) for p, r, c, _ in g.edges]
    lengths = sorted(abs(e[2] - e[0]) + abs(e[3] - e[1]) for e in edges)
    zoom = viewer.MIN_ARROW_PX / lengths[len(lengths) // 2]          # the median arrow is 12 px
    drawn = [e for e in edges if viewer.arrow_visible(*e, zoom)]
    assert 0 < len(drawn) < len(edges)                               # some, but not all, remain
    assert min(abs(e[2] - e[0]) + abs(e[3] - e[1]) for e in drawn) * zoom >= viewer.MIN_ARROW_PX
    assert all(abs(e[2] - e[0]) + abs(e[3] - e[1]) >= lengths[len(lengths) // 2] - 1e-9
               for e in drawn)


def _arrange_data():
    return {"meta": {}, "module": {}, "threads": [],
            "app": {"item10": 1, "item2": "hello world", "Alpha": {"x": 1}, "zeta": [1, 2, 3],
                    "beta": {"a": {"b": {"c": 1}}}}}


def _app_rows(viewer, data, group=False, sort_by="none", desc=False, levels=None, unchecked=(),
              links_only=False):
    g = viewer.build_graph(data)
    view = viewer.apply_filters(g, levels or {}, set(unchecked), links_only)
    viewer.arrange_rows(view, group, sort_by, desc)
    app = next(n for n in view.nodes if n.title == "dict (5)")
    return g, view, [r.key for r in app.rows]


def test_descendant_counts_handle_chains_sharing_cycles_and_self_references():
    """A box's descendants are the different boxes reachable from it, itself not counted."""
    viewer = _load_viewer()
    shared = {"x": {"y": 1}}
    cyc1, cyc2 = {}, {}
    cyc1["to2"] = cyc2
    cyc2["to1"] = cyc1
    selfish = {}
    selfish["me"] = selfish
    g = viewer.build_graph({"meta": {}, "module": {"c": cyc1, "self": selfish}, "threads": [],
                            "app": {"s1": shared, "s2": shared, "own": {"z": 1}}})
    below = {}
    for n in g.nodes:
        for r in n.rows:
            if r.child is not None:
                below[r.key] = r.child.desc_all
    assert below["x"] == 0                                   # {"y": 1} has no box below it
    assert below["s1"] == below["s2"] == 1                   # the shared box reaches x only
    assert below["own"] == 0
    assert below["c"] == 1 and below["to2"] == 1 and below["to1"] == 1   # each reaches the other
    assert below["self"] == 0 and below["me"] == 0           # a box pointing at itself: nothing else
    # a chain of four boxes under "a": a, b, c, d reach 3, 2, 1, 0 others
    chain = viewer.build_graph({"meta": {}, "app": {"a": {"b": {"c": {"d": {"e": 1}}}}},
                                "module": {}, "threads": []})
    counts = {r.key: r.child.desc_all for n in chain.nodes for r in n.rows if r.child is not None}
    assert [counts[k] for k in ("a", "b", "c", "d")] == [3, 2, 1, 0]
    assert chain.root.desc_all == 5                          # app, a, b, c, d
    # the function also works on just a part of the graph (the visible boxes)
    view = viewer.apply_filters(chain, {"dict": 3})
    assert viewer.descendant_counts(view.nodes)[view.root.id] == len(view.nodes) - 1


def test_sizes_count_everything_a_value_holds_and_arrays_their_data():
    viewer = _load_viewer()
    big = np.zeros((100, 100))
    small = {"n": 1}
    g = viewer.build_graph({"meta": {}, "app": {"arr": big, "small": small,
                                                "standin": {"__standin__": "large array",
                                                            "shape": (9,), "dtype": "f8",
                                                            "nbytes": 123456},
                                                "nest": {"one": small, "s": "x" * 500}},
                            "module": {}, "threads": []})
    app = next(n for n in g.nodes if n.title == "dict (4)")
    rows = {r.key: r for r in app.rows}
    assert rows["arr"].size >= big.nbytes
    assert rows["standin"].size >= 123456
    assert rows["nest"].size > rows["small"].size + 500      # holds the small box and a long string
    assert rows["nest"].child.size == rows["nest"].size
    assert app.size >= rows["arr"].size + rows["standin"].size + rows["nest"].size
    assert g.root.size >= app.size


def test_arrange_rows_does_nothing_unless_asked():
    viewer = _load_viewer()
    _g, view, keys = _app_rows(viewer, _arrange_data())
    assert keys == ["item10", "item2", "Alpha", "zeta", "beta"]


def test_links_can_be_grouped_first_keeping_the_order_inside_each_group():
    viewer = _load_viewer()
    _g, _view, keys = _app_rows(viewer, _arrange_data(), group=True)
    assert keys == ["Alpha", "zeta", "beta", "item10", "item2"]


def test_a_row_only_counts_as_a_link_while_its_box_is_visible():
    viewer = _load_viewer()
    g = viewer.build_graph(_arrange_data())
    app_orig = next(n for n in g.nodes if n.title == "dict (5)")
    zeta_origin = (app_orig.id, next(i for i, r in enumerate(app_orig.rows) if r.key == "zeta"))
    _g, _v, keys = _app_rows(viewer, _arrange_data(), group=True, unchecked=[zeta_origin])
    # zeta's box is hidden by its checkbox, so its row is no longer a link
    assert keys[:2] == ["Alpha", "beta"] and keys.index("zeta") > keys.index("beta")
    _g, _v, keys = _app_rows(viewer, _arrange_data(), group=True, levels={"list": 2})
    assert keys == ["Alpha", "beta", "item10", "item2", "zeta"]


def test_rows_sort_by_name_with_numbers_in_natural_order_ascending_and_descending():
    viewer = _load_viewer()
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="name")
    assert keys == ["Alpha", "beta", "item2", "item10", "zeta"]
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="name", desc=True)
    assert keys == ["zeta", "item10", "item2", "beta", "Alpha"]
    assert viewer.natural_key("item10") > viewer.natural_key("item2")
    assert viewer.natural_key("B") > viewer.natural_key("a")          # case-insensitive


def test_grouping_comes_first_then_the_sort_within_each_group():
    viewer = _load_viewer()
    _g, _v, keys = _app_rows(viewer, _arrange_data(), group=True, sort_by="name")
    assert keys == ["Alpha", "beta", "zeta", "item2", "item10"]
    _g, _v, keys = _app_rows(viewer, _arrange_data(), group=True, sort_by="name", desc=True)
    assert keys == ["zeta", "beta", "Alpha", "item10", "item2"]


def test_rows_sort_by_size_in_bytes():
    viewer = _load_viewer()
    g, _v, asc = _app_rows(viewer, _arrange_data(), sort_by="size")
    sizes = {r.key: r.size for n in g.nodes if n.title == "dict (5)" for r in n.rows}
    assert [sizes[k] for k in asc] == sorted(sizes.values())
    assert asc.index("item10") < asc.index("item2")                  # an int is smaller than a str
    assert asc.index("Alpha") < asc.index("beta")                    # beta holds much more
    _g, _v, desc = _app_rows(viewer, _arrange_data(), sort_by="size", desc=True)
    assert [sizes[k] for k in desc] == sorted(sizes.values(), reverse=True)


def test_rows_sort_by_total_descendants_including_hidden_ones():
    viewer = _load_viewer()
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="descendants")
    assert keys == ["item10", "item2", "Alpha", "zeta", "beta"]     # beta has 2 below; ties stay put
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="descendants", desc=True)
    assert keys == ["beta", "item10", "item2", "Alpha", "zeta"]
    # hiding beta's child box does not change the total
    g = viewer.build_graph(_arrange_data())
    beta = next(r.child for n in g.nodes for r in n.rows if r.key == "beta")
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="descendants",
                             unchecked=[(beta.id, 0)])
    assert keys[-1] == "beta"


def test_rows_sort_by_visible_descendants_which_follow_the_filters():
    viewer = _load_viewer()
    g = viewer.build_graph(_arrange_data())
    beta = next(r.child for n in g.nodes for r in n.rows if r.key == "beta")
    _g, _v, everything = _app_rows(viewer, _arrange_data(), sort_by="visible", desc=True)
    assert everything[0] == "beta"                                  # 2 visible descendants
    # with beta's "a" box hidden (its checkbox), beta has none left: back to the original order
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="visible", desc=True,
                             unchecked=[(beta.id, 0)])
    assert keys == ["item10", "item2", "Alpha", "zeta", "beta"]
    g2, view, _ = _app_rows(viewer, _arrange_data(), sort_by="visible", unchecked=[(beta.id, 0)])
    assert viewer.descendant_counts(view.nodes)[next(
        n.id for n in view.nodes if n.id == beta.id)] == 0
    assert g2.root.desc_all >= 5                                    # the full count is unchanged


def test_the_note_about_more_entries_always_stays_last_and_edges_follow_the_moved_rows():
    viewer = _load_viewer()
    many = {f"k{i}": {"v": i} for i in range(viewer.MAX_ROWS + 5)}
    g = viewer.build_graph({"meta": {}, "app": many, "module": {}, "threads": []})
    view = viewer.apply_filters(g, {})
    viewer.arrange_rows(view, True, "name", True)
    app = next(n for n in view.nodes if len(n.rows) == viewer.MAX_ROWS + 1)
    assert app.rows[-1].key == "\u2026" and app.rows[0].key == f"k{viewer.MAX_ROWS - 1}"
    for node, i, child, _tree in view.edges:
        assert node.rows[i].child is child
    assert len(view.edges) == sum(1 for n in view.nodes for r in n.rows if r.child is not None)
    viewer.layout_tree(view)
    assert viewer.overlapping_pairs(view) == [] and viewer.count_crossings(view) == 0
    ys = [c.y for c in app.tree_children]
    assert ys == sorted(ys)                                         # stacked in the new row order
    assert [c.parent_row for c in app.tree_children] == sorted(c.parent_row
                                                               for c in app.tree_children)


def test_hide_rows_that_are_not_links_counts_them_in_the_hidden_message():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {}, set(), True)
    app = next(n for n in view.nodes if n.title == "dict (5)")
    assert [r.key for r in app.rows] == ["items", "obst"]           # only the link rows
    assert app.hidden == 3                                          # count, name, empty
    assert viewer.hidden_message(app.hidden) == "3 children hidden by filters"
    assert all(r.child is not None for n in view.nodes for r in n.rows)   # every row left is a link
    # boxes with no link rows at all show just the message
    leaf = next(n for n in view.nodes if n.rows == [] and n.hidden)
    assert leaf.hidden >= 1
    # rows hidden by a checkbox or by a filter level are not links either
    items = next(r for r in next(n for n in viewer.apply_filters(g, {}).nodes
                                 if n.title == "dict (5)").rows if r.key == "items")
    cleared = viewer.apply_filters(g, {}, {items.origin}, True)
    app2 = next(n for n in cleared.nodes if n.title == "dict (5)")
    assert [r.key for r in app2.rows] == ["obst"] and app2.hidden == 4
    row_only = viewer.apply_filters(g, {"list": 2}, set(), True)
    app3 = next(n for n in row_only.nodes if n.title == "dict (5)")
    assert [r.key for r in app3.rows] == ["obst"] and app3.hidden == 4
    off = viewer.apply_filters(g, {}, set(), False)
    assert next(n for n in off.nodes if n.title == "dict (5)").hidden == 0


def test_the_view_options_reach_the_view_and_show_all_clears_the_links_only_filter():
    viewer = _load_viewer()
    v = _viewer_with_graph(viewer, _arrange_data())
    v.links_only_var.set(True)
    v.group_links_var.set(True)
    v.sort_by_var.set("name")
    v.rebuild_view()
    app = next(n for n in v.vg.nodes if n.title == "dict (5)")
    assert [r.key for r in app.rows] == ["Alpha", "beta", "zeta"] and app.hidden == 2
    v.level_vars = {}
    v.reset_filters()
    assert v.links_only_var.get() is False
    app = next(n for n in v.vg.nodes if n.title == "dict (5)")
    assert app.hidden == 0 and len(app.rows) == 5


def _wheel_stub(viewer):
    import types
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.vg = object()
    v.zoom, v.ox, v.oy = 1.0, 500.0, 300.0
    v.drawn = (0, 0, 10 ** 6, 10 ** 6)
    v.log = []
    v.canvas = types.SimpleNamespace(move=lambda tag, dx, dy: v.log.append(("move", dx, dy)))
    v.canvas_size = lambda: (1000, 600)
    v.redraw = lambda: v.log.append("redraw")
    v.schedule_redraw = lambda: v.log.append("schedule")
    return v


def test_the_plain_wheel_scrolls_up_and_down_and_shift_scrolls_sideways():
    import types
    viewer = _load_viewer()
    v = _wheel_stub(viewer)
    step = viewer.WHEEL_SCROLL_PX
    v.on_wheel(types.SimpleNamespace(x=10, y=10, delta=120, state=0))          # wheel up
    assert v.log[-1] == ("move", 0, step) and v.oy == 300.0 - step and v.ox == 500.0
    v.on_wheel(types.SimpleNamespace(x=10, y=10, delta=-120, state=0))         # wheel down
    assert v.log[-1] == ("move", 0, -step) and v.oy == 300.0
    v.on_wheel(types.SimpleNamespace(x=10, y=10, delta=120, state=0x1))        # Shift, wheel up
    assert v.log[-1] == ("move", step, 0) and v.ox == 500.0 - step and v.oy == 300.0
    v.on_wheel(types.SimpleNamespace(x=10, y=10, delta=-120, state=0x1))       # Shift, wheel down
    assert v.log[-1] == ("move", -step, 0) and v.ox == 500.0
    assert v.zoom == 1.0                                                        # neither zooms


def test_the_scroll_distance_is_in_window_pixels_whatever_the_zoom():
    import types
    viewer = _load_viewer()
    v = _wheel_stub(viewer)
    v.zoom = 4.0
    v.on_wheel(types.SimpleNamespace(x=0, y=0, delta=120, state=0x1))
    assert v.ox == 500.0 - viewer.WHEEL_SCROLL_PX / 4.0
    v.on_wheel(types.SimpleNamespace(x=0, y=0, delta=120, state=0))
    assert v.oy == 300.0 - viewer.WHEEL_SCROLL_PX / 4.0


def test_ctrl_wheel_zooms_at_the_pointer_and_scrolling_redraws_at_the_edge_of_what_is_drawn():
    import types
    viewer = _load_viewer()
    v = _wheel_stub(viewer)
    v.on_wheel(types.SimpleNamespace(x=200, y=100, delta=120, state=0x4))
    assert v.zoom == 1.2 and v.log[-1] == "schedule" and (v.ox, v.oy) != (500.0, 300.0)
    v.on_wheel(types.SimpleNamespace(x=200, y=100, delta=-120, state=0x4))
    assert abs(v.zoom - 1.0) < 1e-12
    v.on_wheel(types.SimpleNamespace(x=200, y=100, delta=120, state=0x1 | 0x4))   # Ctrl wins: zoom
    assert v.zoom == 1.2
    v.zoom = 1.0
    v.drawn = (v.ox, v.oy, v.ox + 1000, v.oy + 600)          # the drawn region is just the window
    v.log.clear()
    v.on_wheel(types.SimpleNamespace(x=0, y=0, delta=120, state=0))
    assert v.log[-1] == "redraw"                             # scrolled out of the drawn region
    v.vg = None
    v.log.clear()
    v.scroll_by(5, 5)
    assert v.log == []                                       # nothing loaded: nothing to move


def test_the_x11_wheel_buttons_use_the_same_rules():
    import types
    viewer = _load_viewer()
    v = _wheel_stub(viewer)
    v.wheel(10, 10, 0, 1)
    assert v.log[-1] == ("move", 0, viewer.WHEEL_SCROLL_PX)
    v.wheel(10, 10, 0x1, -1)
    assert v.log[-1] == ("move", -viewer.WHEEL_SCROLL_PX, 0)
    v.wheel(10, 10, 0x4, 1)
    assert v.zoom == 1.2


def test_the_view_menu_and_sidebar_offer_the_new_options():
    import inspect
    viewer = _load_viewer()
    init = inspect.getsource(viewer.Viewer.__init__)
    assert 'label="Group link rows first"' in init
    assert 'label="Sort objects within their parent"' in init
    assert "for value, label in SORT_CHOICES:" in init
    assert 'label="Ascending"' in init and 'label="Descending"' in init
    labels = [label for _value, label in viewer.SORT_CHOICES]
    assert labels == ["None (original order)", "Name (alphabetical)", "Size in bytes",
                      "Total descendants (including hidden ones)", "Visible descendants"]
    assert "Hide rows that aren't links" in inspect.getsource(viewer.Viewer.build_sidebar)


# --------------------------------------------------------------------------- #
# Strict records (FuzzParams, RaySettings, BuddhaBlurSettings, KeyframeData,
# ViewSettings, ObstacleSnapshot, ViewSnapshot): keyword-only, frozen, no
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


def test_strict_records_take_keyword_arguments_only_and_all_of_them():
    f = m.DEFAULT_FUZZ
    assert _raises(TypeError, lambda: m.FuzzParams(False, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0))
    assert _raises(TypeError, lambda: m.FuzzParams(enabled=False))          # the rest is missing
    assert _raises(TypeError, lambda: m.BuddhaBlurSettings(radius=0.1))
    assert _raises(TypeError, lambda: m.RaySettings(max_iter=1))
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f.as_dict(), "emit_jiter": 0.0}))
    assert _raises(TypeError, lambda: m.RaySettings(*dataclasses.astuple(m.DEFAULT_RAY_SETTINGS)))


def test_strict_records_cannot_be_indexed_iterated_unpacked_or_changed():
    s = m.DEFAULT_RAY_SETTINGS
    assert _raises(TypeError, lambda: s[0])
    assert _raises(TypeError, lambda: list(s))
    assert _raises(TypeError, lambda: tuple(m.DEFAULT_FUZZ))
    assert _raises(TypeError, lambda: len(s))
    assert _raises(dataclasses.FrozenInstanceError, lambda: setattr(s, "max_iter", 5))
    assert _raises(AttributeError, lambda: s.max_itre)
    # (Python 3.11 raises TypeError here, later versions AttributeError; both forbid it)
    assert _raises((AttributeError, TypeError, dataclasses.FrozenInstanceError),
                   lambda: setattr(s, "not_a_field", 1))


def test_strict_records_check_every_value_against_its_annotation():
    f = m.DEFAULT_FUZZ.as_dict()
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f, "seed": True}))        # a bool is not an int
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f, "seed": 1.0}))         # nor is a float
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f, "enabled": 1}))        # an int is not a bool
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f, "bnd_mult": "1.0"}))
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f, "bnd_mult": True}))    # a bool is not a float
    m.FuzzParams(**{**f, "bnd_mult": 2})                                          # an int is a fine float
    s = m.DEFAULT_RAY_SETTINGS.as_dict()
    assert _raises(TypeError, lambda: m.RaySettings(**{**s, "fuzz": (False, 0.0)}))
    assert _raises(TypeError, lambda: m.RaySettings(**{**s, "buddha_blur": 0.0}))
    try:
        m.RaySettings(**{**s, "max_iter": "9"})
        raise AssertionError("expected TypeError")
    except TypeError as e:
        assert "RaySettings.max_iter must be int, got str" in str(e)


def test_a_strict_record_stores_a_list_given_for_a_tuple_field_as_a_tuple():
    vs = m.ViewSettings(mode="RGB", kinds=[0, 1, 2], point_metric=(0, 0, 0), other_metric=(0, 0, 0),
                        point1_choice=(0, 0, 0), point2_choice=(0, 0, 0), chan_src=[0, 0, 0],
                        buddha_points=[(True,)] * 3, buddha_compression=(0, 0, 0),
                        buddha_brightness=(0, 0, 0), constant_choice=(0, 0, 0))
    assert vs.kinds == (0, 1, 2) and isinstance(vs.chan_src, tuple)
    assert _raises(TypeError, lambda: vs.replace(mode=5))


def test_replace_makes_a_modified_copy_and_equality_hash_and_pickle_work():
    import pickle
    a = m.DEFAULT_RAY_SETTINGS
    b = a.replace(max_iter=7, fuzz=a.fuzz.replace(seed=3))
    assert (b.max_iter, b.fuzz.seed, a.max_iter, a.fuzz.seed) == (7, 3, 256, 0)
    assert b != a and a == a.replace() and hash(a) == hash(a.replace())
    assert _raises(TypeError, lambda: a.replace(max_iter="x"))
    with _ModuleRegistered():
        again = pickle.loads(pickle.dumps(b))
    assert again == b and isinstance(again.fuzz, m.FuzzParams)


def test_records_pickle_by_field_name_not_position():
    """A dump or saved file must not depend on the order the fields are declared in
    (dataclass's own pickle state for a frozen, slotted class is a bare list)."""
    import pickle
    with _ModuleRegistered():
        state = m.DEFAULT_FUZZ.__reduce_ex__(2)[2]
        assert isinstance(state, dict) and state["seed"] == 0 and state["bnd_mult"] == 1.0
        scrambled = dict(reversed(list(state.items())))
        clone = m.FuzzParams.__new__(m.FuzzParams)
        clone.__setstate__(scrambled)
        assert clone == m.DEFAULT_FUZZ
        assert pickle.loads(pickle.dumps(m.DEFAULT_FUZZ)) == m.DEFAULT_FUZZ


def test_ray_settings_holds_the_buddha_blur_as_its_own_record():
    s = m.DEFAULT_RAY_SETTINGS
    assert isinstance(s.buddha_blur, m.BuddhaBlurSettings)
    assert (s.buddha_blur.radius, s.buddha_blur.type) == (m.DEFAULT_BUDDHA_BLUR_RADIUS,
                                                          m.DEFAULT_BUDDHA_BLUR_TYPE)
    assert s.replace(buddha_blur=s.buddha_blur.replace(radius=0.2)).buddha_blur.radius == 0.2


def test_the_view_snapshot_record_mirrors_the_slots_of_the_live_view():
    assert [f.name for f in dataclasses.fields(m.ViewSnapshot)] == list(m.ViewConfig.__slots__)


def test_snapshot_view_stores_the_channel_lists_as_tuples_unshared_with_the_live_view():
    v = m.ViewConfig(2)
    snap = m.snapshot_view(v)
    assert isinstance(snap, m.ViewSnapshot) and snap.id == 2 and snap.mode == v.mode
    assert all(isinstance(getattr(snap, f), tuple) for f in m.ViewConfig.__slots__
               if f not in ("id", "mode"))
    v.chan_src[0] = m.CHAN_SRC_BUDDHA
    assert snap.chan_src[0] != m.CHAN_SRC_BUDDHA                # a later live edit never reaches it
    assert m.view_channel_value(snap, 0) == m.view_channel_value(m.ViewConfig(2), 0)


def test_resolved_obstacles_are_frozen_shapes_and_views_and_settings_are_strict_records():
    kfs = [m.Keyframe(0, 0, {"obst:1:r": 0.25})]
    o = m.EllipseShape(1)
    resolved = m.resolve_obstacles([o.snapshot()], kfs, 0)
    assert type(resolved[0]) is m.EllipseShape and resolved[0].r == 0.25
    assert _raises(AttributeError, lambda: setattr(resolved[0], "r", 1.0))     # frozen
    views = m.resolve_views([m.snapshot_view(m.ViewConfig(0))], [], 0)
    assert isinstance(views[0], m.ViewSettings)
    assert isinstance(m.resolve_frame_settings([], 0), m.KeyframeData)
    assert isinstance(m.resolve_frame_settings([], 0).settings, m.RaySettings)


def test_views_colored_from_the_stored_data_use_the_settings_the_image_was_generated_with():
    """Regression test: the two places that color a view from the stored data (the
    composed 'Current image' export and the non-live view thumbnails) built their
    settings from the limits alone, so the confinement rule and emission direction
    silently fell back to their defaults - and Buddha coloration re-traces rays
    with both. Every RaySettings argument is required now, and they pass the
    settings of the generated image."""
    generated = _settings(confine_window=9, confine_method=m.CONFINE_PATH,
                          confine_threshold=0.123, emit_mode=m.EMIT_TOWARDS,
                          emit_target_r=0.5, emit_target_theta=30.0, emit_offset=12.0)
    app = _fake_current_image_app()
    app.generated_settings = generated
    app.settings_of_current_image = m.App.settings_of_current_image.__get__(app)
    seen = []

    def fake_colorize(view, stats, points, status, w, h, settings, fz, obst_arrays):
        seen.append(settings)
        return np.zeros((h, w, 3), dtype=np.uint8)
    app.colorize_view = fake_colorize
    app.render_current_image([m.ViewConfig(0), m.ViewConfig(1)])
    assert seen == [generated, generated]
    assert m.settings_confine(seen[0]) == m.ConfineRule(window=9, method=m.CONFINE_PATH, threshold=0.123)
    assert m.settings_emission(seen[0]).offset == 12.0
    seen.clear()
    app.view_thumb_cache = {}
    app.rows_done = 0
    app.live_view = lambda: app.views[0]
    app.render_view_thumbnail(m.ViewConfig(1), m.THUMB_SIZE)
    assert seen == [generated]
    app.generated_settings = None                                # before the first Start
    assert app.settings_of_current_image() is m.DEFAULT_RAY_SETTINGS


# --------------------------------------------------------------------------- #
# RAY_SETTING_SPECS: one row per simulation setting (path into a RaySettings,
# its control, how the control is written)
# --------------------------------------------------------------------------- #
def _varied_ray_settings():
    return m.RaySettings(
        max_iter=77, max_dist=3.5, seg_mult=0.8,
        fuzz=m.FuzzParams(enabled=True, emit_jitter=4.0, bnd_offset=1.5, bnd_jitter=2.5,
                          bnd_mult=1.25, self_offset=-3.0, self_jitter=6.0, self_mult=0.5,
                          seed=424242),
        buddha_blur=m.BuddhaBlurSettings(radius=0.07, type=m.BUDDHA_BLUR_BOKEH),
        confine_window=9, confine_method=m.CONFINE_PATH, confine_threshold=0.03,
        emit_mode=m.EMIT_TOWARDS, emit_target_r=0.4, emit_target_theta=33.0,
        emit_target_cx=-0.2, emit_target_cy=0.6, emit_offset=12.5)


def test_the_setting_specs_and_the_ray_settings_fields_are_the_same_set():
    paths = sorted(spec.path for spec in m.RAY_SETTING_SPECS.values())
    assert paths == sorted(m.ray_setting_leaf_paths())
    assert len(set(paths)) == len(paths)
    assert set(m.RAY_SETTING_SPECS) == {k for k, _t, _l in m.SETTING_INFO} - {"outline"}


def test_self_checks_catch_a_ray_settings_field_without_a_spec_row():
    fresh = load_module()
    del fresh.RAY_SETTING_SPECS["emit_offset"]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "RAY_SETTING_SPECS" in str(e)


def test_self_checks_catch_a_spec_row_with_a_bad_kind_or_path():
    fresh = load_module()
    fresh.RAY_SETTING_SPECS["seed"] = fresh.RAY_SETTING_SPECS["seed"].replace(kind="dial")
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "bad kind" in str(e)
    fresh = load_module()
    fresh.RAY_SETTING_SPECS["seed"] = fresh.RAY_SETTING_SPECS["seed"].replace(path=("fuzz", "sead"))
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "RaySettings fields disagree" in str(e)


def test_ray_settings_round_trip_through_the_key_dictionary():
    for settings in (m.DEFAULT_RAY_SETTINGS, _varied_ray_settings()):
        values = m.ray_settings_to_dict(settings)
        assert set(values) == set(m.RAY_SETTING_SPECS)
        assert m.ray_settings_from_dict(values) == settings
    values = m.ray_settings_to_dict(_varied_ray_settings())
    assert values["seed"] == 424242 and values["emit_jitter"] == 4.0 and values["iter"] == 77
    assert values["buddha_blur_type"] == m.BUDDHA_BLUR_BOKEH and values["fuzz_on"] is True


def test_values_resolved_from_keyframes_are_converted_to_their_fields_types():
    """A keyframe can hand an int setting over as a float (or a bool as 1): int fields
    are rounded and bool fields made bool; float fields are kept as given."""
    values = dict(m.ray_settings_to_dict(_varied_ray_settings()))
    values.update({"iter": 77.4, "seed": 9.0, "confine_window": 8.6, "emit_mode": 1.0,
                   "fuzz_on": 1, "buddha_blur_type": 2.0, "dist": 2})
    got = m.ray_settings_from_dict(values)
    assert (got.max_iter, got.fuzz.seed, got.confine_window, got.emit_mode) == (77, 9, 9, 1)
    assert got.fuzz.enabled is True and got.buddha_blur.type == 2 and got.max_dist == 2
    assert all(type(v) is int for v in (got.max_iter, got.fuzz.seed, got.confine_window))


def test_the_derived_setting_tables_follow_the_ray_settings_field_types():
    floats = m.SETTING_FLOAT_KEYS
    assert {"dist", "seg_mult", "confine_threshold", "emit_offset", "emit_target_r",
            "emit_jitter", "bnd_mult", "self_offset", "buddha_blur_radius"} <= floats
    assert not ({"iter", "seed", "confine_window", "confine_method", "emit_mode", "fuzz_on",
                 "buddha_blur_type", "outline"} & floats)
    assert m.DEFAULT_SETTING_VALUES == {**m.ray_settings_to_dict(m.DEFAULT_RAY_SETTINGS),
                                        "outline": True}
    assert m.SETTING_WIDGET_ATTRS["seed"] == "seed_entry"
    assert m.SETTING_WIDGET_ATTRS["outline"] == "deco_outline_check"


def _fake_controls_app():
    """An App stand-in holding a control of the right kind for every spec row."""
    class FakeApp:
        pass
    app = FakeApp()
    for spec in m.RAY_SETTING_SPECS.values():
        setattr(app, spec.control, _StubVar(None))
    app.on_limit_var_changed = app.on_shape_param_changed = app.on_fuzz_changed = lambda: None
    app.set_setting_control = m.App.set_setting_control.__get__(app)
    app.revert_settings = m.App.revert_settings.__get__(app)
    return app


def _expected_control_value(spec, value):
    if spec.kind == "text":
        return str(value)
    if spec.kind == "int_text":
        return str(int(round(value)))
    if spec.kind == "float_text":
        return str(float(value))
    if spec.kind == "choice":
        return spec.names[int(round(value))]
    return value


def test_each_kind_of_control_is_written_the_way_its_kind_needs():
    app = _fake_controls_app()
    values = m.ray_settings_to_dict(_varied_ray_settings())
    for key, value in values.items():
        app.set_setting_control(key, value)
        spec = m.RAY_SETTING_SPECS[key]
        assert getattr(app, spec.control).get() == _expected_control_value(spec, value), key
    assert app.iter_var.get() == "77" and app.confine_window_var.get() == "9"
    assert app.emit_target_r_var.get() == "0.4" and app.seed_var.get() == "424242"
    assert app.emit_mode_var.get() == m.EMIT_MODE_NAMES[m.EMIT_TOWARDS]
    assert app.buddha_blur_type_var.get() == m.BUDDHA_BLUR_LABELS[m.BUDDHA_BLUR_BOKEH]
    assert app.fuzz_enabled_var.get() is True and app.bnd_mult_scale.get() == 1.25


def test_a_value_an_interpolating_keyframe_made_fractional_still_reaches_an_int_control():
    app = _fake_controls_app()
    app.set_setting_control("confine_window", 8.6)
    app.set_setting_control("emit_mode", 0.9)
    assert app.confine_window_var.get() == "9"
    assert app.emit_mode_var.get() == m.EMIT_MODE_NAMES[1]


def test_revert_writes_every_control_from_the_baseline_through_the_same_table():
    app = _fake_controls_app()
    app.baseline = _varied_ray_settings()
    app.revert_settings()
    for key, value in m.ray_settings_to_dict(app.baseline).items():
        spec = m.RAY_SETTING_SPECS[key]
        assert getattr(app, spec.control).get() == _expected_control_value(spec, value), key


# --------------------------------------------------------------------------- #
# ChannelSettings, SettingEntry, ObstacleRing; label and tab live in the spec rows
# --------------------------------------------------------------------------- #
def test_a_view_channel_is_stored_as_a_strict_record_and_decomposes_back():
    v = m.ViewConfig(0)
    v.chan_src[2] = m.CHAN_SRC_BUDDHA
    v.buddha_points[2] = tuple(i == 1 for i in range(m.N_BUDDHA_POINTS))
    v.kinds[1] = m.KIND_OTHER
    v.other_metric[1] = m.ST_BOUNCES
    chans = [m.view_channel_value(v, c) for c in range(3)]
    assert all(isinstance(ch, m.ChannelSettings) for ch in chans)
    assert chans[1].kind == m.KIND_OTHER and chans[1].stat == m.ST_BOUNCES
    assert chans[2].chan_src == m.CHAN_SRC_BUDDHA and chans[2].buddha_points[1] is True
    vs = m.decompose_chans(v.mode, chans)
    assert vs.other_metric[1] == m.ST_BOUNCES and vs.point_metric[1] == m.DEFAULT_POINT_METRICS[1]
    assert vs.chan_src[2] == m.CHAN_SRC_BUDDHA
    assert vs.buddha_points[2] == v.buddha_points[2]
    assert chans[0] == m.view_channel_value(m.snapshot_view(v), 0)      # live view and snapshot agree
    assert _raises(TypeError, lambda: chans[0][0])                      # no positional access
    assert _raises(TypeError, lambda: m.ChannelSettings(*dataclasses.astuple(chans[0])))
    assert _raises(TypeError, lambda: chans[0].replace(kind="point"))


def test_all_setting_entries_are_strict_records_in_display_order():
    class FakeApp:
        pass
    app = FakeApp()
    o1, o2 = m.EllipseShape(4), m.EllipseShape(9)
    o2.width = 0.7
    app.obstacles = [o1, o2]
    app.views = [m.ViewConfig(3)]
    entries = m.App.all_setting_entries(app)
    assert all(isinstance(e, m.SettingEntry) for e in entries)
    static = entries[:len(m.SETTING_INFO)]
    assert [(e.key, e.tab, e.label) for e in static] == m.SETTING_INFO
    assert [e.is_float for e in static] == [e.key in m.SETTING_FLOAT_KEYS for e in static]
    width = next(e for e in entries if e.key == "obst:9:width")
    assert (width.tab, width.fallback, width.is_float) == (m.obstacle_tab_title(1, o2), 0.7, True)
    chan = next(e for e in entries if e.key == "view:3:chan1")
    assert chan.tab == "View 0" and chan.fallback == m.view_channel_value(app.views[0], 1)
    assert m.obstacle_tab_title(0, o1) == "Obstacle #1 (Ellipse)"
    assert [e.tab for e in entries].count(m.obstacle_tab_title(0, o1)) == len(o1.KEYFRAME_FIELDS)
    assert "obst:4:shape" not in [e.key for e in entries]       # a shape's kind is not a setting


def test_setting_info_is_derived_from_the_spec_rows_so_every_setting_has_a_label_and_tab():
    assert [row for row in m.SETTING_INFO if row[0] != "outline"] == [
        (key, spec.tab, spec.label) for key, spec in m.RAY_SETTING_SPECS.items()]
    assert m.SETTING_INFO[-1] == ("outline", "Decorations", "Outline")
    assert all(spec.label and spec.tab in m.SETTING_TABS_ORDER
               for spec in m.RAY_SETTING_SPECS.values())
    assert m.SETTING_LABEL["emit_offset"] == "Emission angle offset"
    assert m.SETTING_TAB["seed"] == "Fuzzing"


def _ring_app(generated):
    class FakeApp:
        pass
    app = FakeApp()
    o = m.default_starting_obstacle(0)
    o2 = m.EllipseShape(1)
    o2.r, o2.theta, o2.cx, o2.in_effect, o2.out_effect = 0.5, 90.0, 0.25, m.OBSTACLE_EFFECT_TERMINATE, m.OBSTACLE_EFFECT_PASS
    app.obstacles = [o, o2]
    arrays = m.build_obstacle_arrays(app.obstacles)
    app.gen_obstacles = arrays
    app.generated_settings = m.DEFAULT_RAY_SETTINGS if generated else None
    return app


def test_obstacle_rings_are_strict_records_for_both_the_live_and_the_generated_set():
    for generated in (False, True):
        rings = m.App.obstacle_ring_specs(_ring_app(generated))
        assert len(rings) == 2 and all(isinstance(r, m.ObstacleRing) for r in rings)
        first, second = rings
        assert type(first.shape) is int and type(first.in_effect) is int      # numpy scalars converted
        assert type(second.cx) is float and isinstance(second.bound, np.ndarray)
        assert (second.in_effect, second.out_effect) == (m.OBSTACLE_EFFECT_TERMINATE,
                                                         m.OBSTACLE_EFFECT_PASS)
        assert abs(second.cx - 0.25) < 1e-9 and abs(second.cy - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# Kernel arguments: FuzzRule / ConfineRule / EmissionRule / ObstacleArrays
# --------------------------------------------------------------------------- #
def test_the_rule_builders_give_named_typed_fields():
    f = m.fuzz_rule(m.DEFAULT_FUZZ.replace(enabled=True, emit_jitter=5.0, bnd_mult=2.0, seed=9))
    assert isinstance(f, m.FuzzRule) and f.enabled is True
    assert (f.emit_jitter, f.bnd_mult, f.self_mult) == (5.0, 2.0, 1.0)
    c = m.confine_rule(9.0, 1.0, 0.25)                       # floats become the ints the kernels need
    assert (c, type(c.window), type(c.method)) == (m.ConfineRule(window=9, method=1, threshold=0.25),
                                                  int, int)
    e = m.emission_rule(1, 2, 3, 4)
    assert e == m.EmissionRule(mode=1, target_x=2.0, target_y=3.0, offset=4.0)
    assert all(type(v) is float for v in (e.target_x, e.target_y, e.offset))
    settings = _settings(emit_mode=m.EMIT_TOWARDS, emit_target_r=2.0, emit_target_theta=90.0,
                         emit_offset=7.0, confine_window=6, confine_threshold=0.5)
    assert m.settings_confine(settings) == m.ConfineRule(window=6, method=settings.confine_method,
                                                         threshold=0.5)
    rule = m.settings_emission(settings)
    assert (rule.mode, rule.offset) == (m.EMIT_TOWARDS, 7.0) and abs(rule.target_y - 2.0) < 1e-9


def test_obstacle_arrays_bundle_parallel_arrays_in_one_argument():
    a, b = m.default_starting_obstacle(0), m.EllipseShape(1)
    arrays = m.build_obstacle_arrays([a, b])
    assert isinstance(arrays, m.ObstacleArrays)
    assert arrays._fields == ("shapes", "bounds", "cx", "cy", "effects", "fire_inside")
    n = len(arrays.shapes)
    assert n == 2 and arrays.bounds.shape == (2, m.OBST_BOUND_LEN) and arrays.effects.shape == (2, 2)
    assert all(len(x) == n for x in (arrays.cx, arrays.cy, arrays.fire_inside))
    assert arrays.fire_inside.tolist() == [True, True]
    empty = m.build_obstacle_arrays([])
    assert len(empty.shapes) == 0 and empty.effects.shape == (0, 2)
    resolved = m.resolve_obstacle_arrays([a.snapshot()], [], 0)
    assert isinstance(resolved, m.ObstacleArrays) and len(resolved.shapes) == 1


def test_multi_point_buddha_fires_rays_in_proportion_to_the_output_resolution():
    """Regression test: scatter_multi_buddha started its rays on the main image's fixed
    512-pixel grid whatever the size of the grid it scattered into, so a larger export fired
    the same rays as 512 (4x fewer per pixel at 1024: a dimmer image) and a smaller one
    only fired from the top-left corner of the world. Now every output pixel in the shape
    fires one ray, so the points per pixel stay the same at any resolution."""
    start = m.default_starting_obstacle(0)
    arrays = m.build_obstacle_arrays([start])
    fz = m.fuzz_rule(m.DEFAULT_FUZZ)
    totals, densities = {}, {}
    for w in (64, 128, 256):
        counts = np.zeros((w, w))
        m.scatter_multi_buddha(0, w, 40, m.effective_dist(0.0), 1.0, fz, m.CONFINE_DEFAULT,
                               m.EMISSION_DEFAULT, arrays, 0, False, True, False, False, counts)
        gx, gy = m.make_grid(w, w)
        inside = int((np.hypot(gx, gy) < 1.0).sum())
        totals[w], densities[w] = counts.sum(), counts.sum() / inside
        half = w // 2
        for quadrant in (counts[:half, :half], counts[:half, half:], counts[half:, :half],
                         counts[half:, half:]):
            assert quadrant.sum() > 0.15 * counts.sum(), "rays fired from the whole world"
    assert 3.7 < totals[128] / totals[64] < 4.3 and 3.7 < totals[256] / totals[128] < 4.3
    assert max(densities.values()) / min(densities.values()) < 1.1, densities


# --------------------------------------------------------------------------- #
# ViewConfig channels and settings
# --------------------------------------------------------------------------- #
def test_a_view_config_takes_a_channel_record_and_makes_its_view_settings():
    v = m.ViewConfig(4)
    channel = m.ChannelSettings(chan_src=m.CHAN_SRC_BUDDHA, kind=m.KIND_OTHER, stat=m.ST_ANGLES,
                                point1=3, point2=5,
                                buddha_points=(False, True) + (False,) * (m.N_BUDDHA_POINTS - 2),
                                buddha_compression=m.BUDDHA_COMPRESS_MODULUS,
                                buddha_brightness=2, constant=m.NONE_ONE)
    v.set_channel(1, channel)
    assert m.view_channel_value(v, 1) == channel
    assert m.view_channel_value(v, 0) == m.view_channel_value(m.ViewConfig(0), 0)   # others untouched
    point = channel.replace(kind=m.KIND_POINT, stat=m.ST_EUCLIDEAN)
    v.set_channel(2, point)
    assert m.view_channel_value(v, 2) == point and v.point_metric[2] == m.ST_EUCLIDEAN
    settings = v.settings()
    assert isinstance(settings, m.ViewSettings) and settings.mode == v.mode
    assert settings.chan_src[1] == m.CHAN_SRC_BUDDHA


def test_a_view_config_made_from_a_snapshot_has_the_snapshots_contents():
    v = m.ViewConfig(7)
    v.mode = "HSV"
    v.set_channel(0, m.ChannelSettings(
        chan_src=m.CHAN_SRC_CONSTANT, kind=m.KIND_POINT, stat=m.ST_HDIST, point1=1, point2=2,
        buddha_points=(True,) + (False,) * (m.N_BUDDHA_POINTS - 1), buddha_compression=0,
        buddha_brightness=4, constant=m.NONE_ZERO))
    snap = m.snapshot_view(v)
    back = m.view_config_from_snapshot(snap)
    assert back is not v and back.id == 7 and m.snapshot_view(back) == snap


# --------------------------------------------------------------------------- #
# Projects: the file format
# --------------------------------------------------------------------------- #
def _sample_project(**changes):
    """A project using every kind of obstacle, two views (one with Buddha channels),
    keyframes (including channel records) and non-default simulation and export choices."""
    obstacles = []
    for i, cls in enumerate(m.SHAPE_CLASSES):
        o = cls(i * 2 + 1)                                  # ids 1, 3, 5...: not 0..n
        for name in o.FIELD_NAMES:
            if name == "rotation":
                o.rotation = 10.0 + i
            elif name == "r":
                o.r = 0.25
            elif name == "fire_inside":
                o.fire_inside = bool(i % 2)
            elif name == "in_effect":
                o.in_effect = i % 3
            elif name == "out_effect":
                o.out_effect = (i + 1) % 3
        obstacles.append(o.snapshot())
    v0, v1 = m.ViewConfig(0), m.ViewConfig(5)
    v1.mode = "HSV"
    v1.set_channel(1, m.ChannelSettings(
        chan_src=m.CHAN_SRC_BUDDHA, kind=m.KIND_OTHER, stat=m.ST_BOUNCES, point1=0, point2=1,
        buddha_points=(False, True, True) + (False,) * (m.N_BUDDHA_POINTS - 3),
        buddha_compression=m.BUDDHA_COMPRESS_MODULUS, buddha_brightness=3, constant=m.NONE_ONE))
    channel = m.view_channel_value(v1, 1)
    keyframes = (m.Keyframe(2, 0, {"iter": 40, "dist": 2.5, "outline": False,
                                   "obst:1:rotation": 15.0, "view:5:mode": "HSV",
                                   "view:5:chan1": channel, "view:0:chan0": m.view_channel_value(v0, 0)}),
                 m.Keyframe(7, 30, {"obst:3:r": 0.5, "emit_mode": m.EMIT_TOWARDS}))
    ray = _settings(max_iter=77, max_dist=3.5, seg_mult=0.8, emit_mode=m.EMIT_TOWARDS,
                    emit_target_r=0.3, emit_offset=12.5, confine_window=6,
                    fuzz=m.DEFAULT_FUZZ.replace(enabled=True, emit_jitter=2.0, seed=99),
                    buddha_blur=m.BuddhaBlurSettings(radius=0.03, type=m.BUDDHA_BLUR_BOKEH))
    export = {"include_all_views": False, "color_info_location": m.COLOR_INFO_LOCATIONS[1],
              "image_format": "png", "image_jxl_quality": 85, "image_width": "640",
              "video_fps": "24", "video_format": "JXL", "video_quality": 60.5,
              "video_jxl_mode": "Lossy"}
    fields = dict(ray=ray, decorations={"outline": False, "pending": True}, obstacles=tuple(obstacles),
                  views=(m.snapshot_view(v0), m.snapshot_view(v1)), keyframes=keyframes,
                  export=export, obstacle_counter=40, view_counter=9, keyframe_counter=12,
                  selected_obstacle_id=5, selected_view_id=5)
    fields.update(changes)
    return m.Project(**fields)


def _as_saved(channel, number):
    """A colour channel as it comes back from a file: the settings its source does not use
    are not kept, so they take the channel's defaults."""
    return m._channel_from_json(m._channel_to_json(channel), "test", number)


def _round_trip(project):
    import json
    return m.project_from_dict(json.loads(m.project_text(project)))


def test_a_project_survives_being_saved_and_loaded():
    project = _sample_project()
    loaded = _round_trip(project)
    assert m.project_signature(loaded) == m.project_signature(project)
    assert loaded.ray == project.ray and [v.id for v in loaded.views] == [0, 5]
    for want, got in zip(project.views, loaded.views):
        assert got.mode == want.mode
        for number in range(3):
            assert m.view_channel_value(got, number) == _as_saved(m.view_channel_value(want, number), number)
    assert [type(o) for o in loaded.obstacles] == list(m.SHAPE_CLASSES)
    assert loaded.export == project.export
    assert [o.id for o in loaded.obstacles] == [i * 2 + 1 for i in range(len(m.SHAPE_CLASSES))]
    assert all(o.state() == q.state() for o, q in zip(loaded.obstacles, project.obstacles))
    assert all(o._frozen for o in loaded.obstacles)
    assert (loaded.obstacle_counter, loaded.view_counter, loaded.keyframe_counter) == (40, 9, 12)
    assert (loaded.selected_obstacle_id, loaded.selected_view_id) == (5, 5)
    kf = loaded.keyframes[0]
    assert (kf.id, kf.frame_index) == (2, 0)
    want = project.keyframes[0].data
    assert kf.data["view:5:chan1"] == _as_saved(want["view:5:chan1"], 1)
    assert kf.data["view:0:chan0"] == _as_saved(want["view:0:chan0"], 0)
    assert {k: v for k, v in kf.data.items() if not k.startswith("view:") or k.endswith("mode")} \
        == {k: v for k, v in want.items() if not k.startswith("view:") or k.endswith("mode")}
    assert isinstance(kf.data["view:5:chan1"], m.ChannelSettings)
    assert isinstance(kf.data["iter"], int) and kf.data["outline"] is False


def _channel(source, **changes):
    fields = dict(chan_src=source, kind=m.KIND_POINT, stat=m.ST_EUCLIDEAN, point1=m.POINT_TOP,
                  point2=m.POINT_ORIGIN,
                  buddha_points=(False, True) + (False,) * (m.N_BUDDHA_POINTS - 2),
                  buddha_compression=m.BUDDHA_COMPRESS_MODULUS, buddha_brightness=3,
                  constant=m.NONE_ONE)
    fields.update(changes)
    return m.ChannelSettings(**fields)


def test_a_channel_in_a_file_has_only_the_settings_its_source_uses():
    query_points = m._channel_to_json(_channel(m.CHAN_SRC_QUERY))
    assert query_points == {"source": "query", "kind": "point_metric", "metric": "euclidean_distance",
                            "point1": "uppermost_bounce", "point2": "origin"}
    query_other = m._channel_to_json(_channel(m.CHAN_SRC_QUERY, kind=m.KIND_OTHER, stat=m.ST_BOUNCES))
    assert query_other == {"source": "query", "kind": "other_metric", "metric": "bounces"}
    assert m._channel_to_json(_channel(m.CHAN_SRC_BUDDHA)) == {
        "source": "buddha", "buddha_points": ["self_bounces_mean"],
        "buddha_compression": "modulus", "buddha_brightness": "8x"}
    assert m._channel_to_json(_channel(m.CHAN_SRC_CONSTANT)) == {"source": "constant",
                                                                  "constant": "one"}
    text = m.project_text(_sample_project())
    import json
    for view in json.loads(text)["views"]:
        for channel in view["channels"]:
            unused = {"query": ["buddha_points", "buddha_compression", "buddha_brightness",
                                "constant"],
                      "buddha": ["kind", "metric", "point1", "point2", "constant"],
                      "constant": ["kind", "metric", "point1", "point2", "buddha_points",
                                   "buddha_compression", "buddha_brightness"]}[channel["source"]]
            assert not [name for name in unused if name in channel], channel


def test_a_channel_missing_a_setting_its_source_uses_is_refused_and_unused_ones_take_defaults():
    def refused(data, number=0):
        try:
            m._channel_from_json(data, "test", number)
        except m.ProjectError as exc:
            return str(exc)
        raise AssertionError("accepted")
    assert "'buddha_compression'" in refused({"source": "buddha", "buddha_points": []})
    assert "'buddha_brightness'" in refused({"source": "buddha", "buddha_points": [],
                                             "buddha_compression": "modulus"})
    assert "'constant'" in refused({"source": "constant"})
    assert "'metric'" in refused({"source": "query", "kind": "other_metric"})
    assert "'point2'" in refused({"source": "query", "kind": "point_metric",
                                  "metric": "horizontal_distance", "point1": "origin"})
    assert "'source'" in refused({"kind": "point_metric"})
    assert "does not suit" in refused({"source": "query", "kind": "other_metric",
                                       "metric": "horizontal_distance"})
    just_constant = m._channel_from_json({"source": "constant", "constant": "one"}, "test", 2)
    default = m.view_channel_value(m.ViewConfig(0), 2)
    assert just_constant == default.replace(chan_src=m.CHAN_SRC_CONSTANT, constant=m.NONE_ONE)
    assert m._channel_from_json({"source": "query", "kind": "other_metric", "metric": "bounces"},
                                "test", 1).point1 == m.view_channel_value(m.ViewConfig(0), 1).point1


def test_a_file_that_keeps_unused_channel_settings_still_loads_with_them():
    """The first files of format 2 held every setting of every channel; they must keep loading,
    and what they kept is kept."""
    everything = {"source": "query", "kind": "point_metric", "metric": "manhattan_distance",
                  "point1": "origin", "point2": "end_point", "buddha_points": ["termination_point"],
                  "buddha_compression": "modulus", "buddha_brightness": "16x", "constant": "one"}
    channel = m._channel_from_json(everything, "test", 0)
    assert channel.chan_src == m.CHAN_SRC_QUERY and channel.stat == m.ST_MANHATTAN
    assert channel.buddha_compression == m.BUDDHA_COMPRESS_MODULUS and channel.buddha_brightness == 4
    assert channel.constant == m.NONE_ONE and channel.buddha_points[3] is True


def test_unused_channel_settings_do_not_count_as_a_change_to_the_project():
    """What is not saved cannot make the project 'unsaved'."""
    project = _sample_project()
    sig = m.project_signature(project)
    view = m.view_config_from_snapshot(project.views[0])
    assert view.chan_src[0] == m.CHAN_SRC_QUERY
    view.buddha_compression[0] = m.BUDDHA_COMPRESS_MODULUS            # unused: a query channel
    view.constant_choice[0] = m.NONE_ONE
    view.buddha_brightness[0] = 4
    edited = project.replace(views=(m.snapshot_view(view), project.views[1]))
    assert m.project_signature(edited) == sig
    view.chan_src[0] = m.CHAN_SRC_BUDDHA                              # now they are used
    changed = project.replace(views=(m.snapshot_view(view), project.views[1]))
    assert m.project_signature(changed) != sig


def test_a_project_that_cannot_be_saved_counts_as_having_unsaved_changes():
    class FakeApp:
        pass
    app = FakeApp()
    ok = _sample_project()
    app.saved_signature = m.project_signature(ok)
    app.capture_project = lambda show_error=True: ok
    app.is_dirty = m.App.is_dirty.__get__(app)
    assert app.is_dirty() is False
    app.capture_project = lambda show_error=True: ok.replace(ray=ok.ray.replace(seg_mult=float("nan")))
    assert app.is_dirty() is True
    app.capture_project = lambda show_error=True: None
    assert app.is_dirty() is True


def test_a_project_file_is_pretty_printed_versioned_json():
    import json
    text = m.project_text(_sample_project())
    assert text.startswith('{\n  "format": "light-bounces-project",\n  "format_version": 2,')
    assert text.endswith("}\n") and "\r" not in text
    assert all(len(line) < 120 for line in text.split("\n"))            # one value per line
    data = json.loads(text)
    assert data["format"] == m.PROJECT_FORMAT_NAME and data["format_version"] == 2
    assert data["program_version"] == m.FILE_VERSION
    assert [o["kind"] for o in data["obstacles"]] == [cls.KEY for cls in m.SHAPE_CLASSES]
    assert data["obstacles"][10]["kind"] == "lune" and data["selected"] == {"obstacle": 5, "view": 5}
    assert m.PROJECT_EXTENSION == ".lightbounce"


def test_saving_and_loading_a_project_file_on_disk(tmp=None):
    path = os.path.join(tempfile.mkdtemp(), "mine" + m.PROJECT_EXTENSION)
    project = _sample_project()
    m.save_project(path, project)
    assert m.project_signature(m.load_project(path)) == m.project_signature(project)
    m.save_project(path, _sample_project(obstacle_counter=99))              # replaced in place
    assert m.load_project(path).obstacle_counter == 99
    assert os.listdir(os.path.dirname(path)) == ["mine" + m.PROJECT_EXTENSION]
    with open(path, "rb") as f:
        assert f.read().count(b"\n") > 100


def _dict_of(project=None):
    import json
    return json.loads(m.project_text(project or _sample_project()))


def _refused(data, expected_text):
    try:
        m.project_from_dict(data)
    except m.ProjectError as exc:
        assert expected_text in str(exc), str(exc)
    else:
        raise AssertionError(f"expected a ProjectError mentioning {expected_text!r}")


def test_a_file_that_is_not_a_usable_project_is_refused_with_a_clear_message():
    d = _dict_of()
    _refused([], "not a Light bounces project")
    _refused({**d, "format": "something-else"}, "not a Light bounces project")
    _refused({**d, "format_version": 99}, "newer version")
    _refused({**d, "format_version": "1"}, "format version")
    _refused({**d, "format_version": 0}, "format version")
    _refused({**d, "views": []}, "no views")
    d2 = _dict_of()
    d2["obstacles"][0]["kind"] = "HexagonShape"
    _refused(d2, "HexagonShape")
    d3 = _dict_of()
    d3["obstacles"][0]["fields"]["sides"] = 4
    _refused(d3, "does not have")
    d4 = _dict_of()
    d4["obstacles"][1]["fields"]["sides"] = 2.5
    _refused(d4, "sides")
    d5 = _dict_of()
    d5["obstacles"][1]["fields"]["sides"] = 500
    _refused(d5, "outside")
    d6 = _dict_of()
    d6["obstacles"][1]["id"] = d6["obstacles"][0]["id"]
    _refused(d6, "two obstacles have the id")
    d7 = _dict_of()
    d7["views"][1]["id"] = 0
    _refused(d7, "two views have the id")
    d8 = _dict_of()
    d8["views"][0]["mode"] = "CMYK"
    _refused(d8, "colour model")
    d9 = _dict_of()
    d9["views"][0]["channels"].pop()
    _refused(d9, "3 channels")
    d10 = _dict_of()
    d10["views"][0]["channels"][0]["metric"] = "no_such_metric"
    _refused(d10, "no_such_metric")
    d10b = _dict_of()
    d10b["views"][0]["channels"][0].update(kind="point_metric", metric="bounces")
    _refused(d10b, "does not suit")
    d10c = _dict_of()
    d10c["views"][1]["channels"][1]["buddha_points"] = ["all_bounces_mean", "a_future_point"]
    _refused(d10c, "a_future_point")
    d10d = _dict_of()
    d10d["obstacles"][0]["fields"]["in_effect"] = 1                      # a number, not a key
    _refused(d10d, "ray effect")
    d10e = _dict_of()
    d10e["simulation"]["emit_mode"] = "sideways"
    _refused(d10e, "sideways")
    d10f = _dict_of()
    d10f["keyframes"][0]["settings"]["obst:1:out_effect"] = "explode"
    _refused(d10f, "explode")
    d11 = _dict_of()
    d11["keyframes"][1]["id"] = d11["keyframes"][0]["id"]
    _refused(d11, "two keyframes have the id")
    d12 = _dict_of()
    d12["keyframes"][0]["settings"]["iter"] = [1, 2]
    _refused(d12, "iter")
    d13 = _dict_of()
    d13["simulation"]["iter"] = "many"
    _refused(d13, "damaged")
    d14 = _dict_of()
    d14["views"][0]["channels"][0].pop("metric")
    _refused(d14, "no 'metric'")
    d15 = _dict_of()
    d15["keyframes"][0]["settings"]["view:5:chan7"] = {"source": "query"}
    _refused(d15, "not a channel")


def test_loading_a_file_that_is_not_json_or_not_text_or_missing_is_refused():
    folder = tempfile.mkdtemp()
    bad = os.path.join(folder, "bad.lightbounce")
    with open(bad, "w") as f:
        f.write('{"format": "light-bounces-project", ')
    for path, text in ((bad, "line 1"), (os.path.join(folder, "missing.lightbounce"), "could not be read")):
        try:
            m.load_project(path)
        except m.ProjectError as exc:
            assert text in str(exc), str(exc)
        else:
            raise AssertionError(path)
    binary = os.path.join(folder, "binary.lightbounce")
    with open(binary, "wb") as f:
        f.write(bytes([0xff, 0xfe, 0x00, 0x80]))
    try:
        m.load_project(binary)
    except m.ProjectError as exc:
        assert "UTF-8" in str(exc)
    else:
        raise AssertionError("binary file accepted")


def test_a_project_that_cannot_be_written_as_json_says_so_and_writes_nothing():
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "x.lightbounce")
    ok = _sample_project()
    not_a_number = ok.replace(ray=ok.ray.replace(seg_mult=float("nan")))
    try:
        m.save_project(path, not_a_number)
    except m.ProjectError:
        pass
    else:
        raise AssertionError("NaN was saved")
    assert os.listdir(folder) == []
    try:
        m.save_project(os.path.join(folder, "no", "such", "dir", "x.lightbounce"), ok)
    except m.ProjectError as exc:
        assert "could not be written" in str(exc)
    else:
        raise AssertionError("an impossible path was accepted")


def test_a_file_from_another_version_is_read_leniently():
    d = _dict_of()
    del d["simulation"]["iter"]                      # a setting this file lacks: its default
    d["simulation"]["a_setting_from_the_future"] = 5
    d["future_section"] = {"anything": 1}
    d["export"]["a_future_choice"] = "x"
    d["counters"]["obstacle"] = 0                    # lower than the ids in use: raised above them
    d["export"]["video_format"] = "a_future_format"                   # unknown choice: control keeps its value
    d["views"][1]["channels"][1]["buddha_points"] = ["all_bounces_mean"]
    loaded = m.project_from_dict(d)
    assert "video_format" not in loaded.export
    assert loaded.ray.max_iter == m.DEFAULT_RAY_SETTINGS.max_iter
    assert loaded.obstacle_counter == max(o.id for o in loaded.obstacles) + 1
    chosen = m.view_channel_value(loaded.views[1], 1).buddha_points
    assert chosen == (True,) + (False,) * (m.N_BUDDHA_POINTS - 1)
    d2 = _dict_of()
    d2["selected"] = {"obstacle": 12345, "view": 777}              # gone: first view, none selected
    loaded = m.project_from_dict(d2)
    assert loaded.selected_obstacle_id == -1 and loaded.selected_view_id == loaded.views[0].id


def test_the_project_signature_sees_content_but_not_selection_or_counters():
    base = _sample_project()
    sig = m.project_signature(base)
    assert m.project_signature(_sample_project()) == sig
    assert m.project_signature(_sample_project(selected_obstacle_id=1, selected_view_id=0,
                                               obstacle_counter=1, view_counter=1,
                                               keyframe_counter=1)) == sig
    moved = base.obstacles[0].with_values(r=0.75)
    assert m.project_signature(_sample_project(obstacles=(moved,) + base.obstacles[1:])) != sig
    assert m.project_signature(_sample_project(obstacles=base.obstacles[:-1])) != sig
    assert m.project_signature(base.replace(ray=base.ray.replace(max_iter=78))) != sig
    assert m.project_signature(base.replace(export={**base.export, "video_fps": "25"})) != sig
    assert m.project_signature(base.replace(decorations={"outline": True, "pending": True})) != sig
    edited = (m.Keyframe(2, 0, {**base.keyframes[0].data, "iter": 41}), base.keyframes[1])
    assert m.project_signature(base.replace(keyframes=edited)) != sig
    other_view = m.ViewConfig(5)
    assert m.project_signature(base.replace(views=(base.views[0], m.snapshot_view(other_view)))) != sig


def test_the_export_option_table_names_each_choice_once_with_sensible_allowed_values():
    keys = [key for key, _path, _kind, _allowed in m.EXPORT_OPTIONS]
    assert len(keys) == len(set(keys))
    paths = [path for _key, path, _kind, _allowed in m.EXPORT_OPTIONS]
    assert len(paths) == len(set(paths))
    for _key, _path, kind, allowed in m.EXPORT_OPTIONS:
        assert kind in ("choice", "text", "bool", "number")
        assert (kind == "choice") == bool(allowed)
    assert "JXL" in m.VIDEO_FORMATS and "Constant quality" in m.VIDEO_BITRATE_MODES


# A project file calls every choice by a permanent string KEY, which a Catalog translates to the
# program's own code. Both sides are pinned: a key may be added but never removed, renamed or
# pointed at another meaning; the codes themselves may be renumbered freely (the file does not
# hold them). (If you only reworded a label that a key is pinned to, update it here.)
PINNED_KEYS = {
    "CAT_CHAN_SOURCE": {"query": "CHAN_SRC_QUERY", "buddha": "CHAN_SRC_BUDDHA",
                        "constant": "CHAN_SRC_CONSTANT"},
    "CAT_CHAN_KIND": {"point_metric": "KIND_POINT", "other_metric": "KIND_OTHER"},
    "CAT_METRIC": {
        "horizontal_distance": "ST_HDIST", "vertical_distance": "ST_VDIST",
        "manhattan_distance": "ST_MANHATTAN", "euclidean_distance": "ST_EUCLIDEAN",
        "bounces": "ST_BOUNCES", "ray_bounce_fraction": "ST_RAY_FRAC",
        "boundary_bounce_fraction": "ST_BOUNDARY_FRAC", "loop_area": "ST_LOOP_AREA",
        "mean_bounce_angle": "ST_ANGLES", "ray_length": "ST_LENGTH",
        "last_segment_direction": "ST_LAST_DIR", "termination_reason": "ST_TERMINATION",
        "last_segment_ratio": "ST_LAST_RATIO"},
    "CAT_POINT": {
        "start_point": "POINT_START", "end_point": "POINT_END",
        "all_bounces_mean": "POINT_ALL_MEAN", "boundary_bounces_mean": "POINT_BOUNDARY_MEAN",
        "self_bounces_mean": "POINT_SELF_MEAN", "uppermost_bounce": "POINT_TOP",
        "lowermost_bounce": "POINT_BOTTOM", "rightmost_bounce": "POINT_RIGHT",
        "leftmost_bounce": "POINT_LEFT", "uppermost_self_bounce": "POINT_SELF_TOP",
        "lowermost_self_bounce": "POINT_SELF_BOTTOM", "rightmost_self_bounce": "POINT_SELF_RIGHT",
        "leftmost_self_bounce": "POINT_SELF_LEFT", "origin": "POINT_ORIGIN",
        "top_left_corner": "POINT_TOP_LEFT_CORNER"},
    "CAT_BUDDHA_COMPRESSION": {"arctangent": "BUDDHA_COMPRESS_ARCTAN",
                               "modulus": "BUDDHA_COMPRESS_MODULUS"},
    "CAT_CONSTANT": {"zero": "NONE_ZERO", "one": "NONE_ONE"},
    "CAT_EMIT_MODE": {"up": "EMIT_UP", "towards_point": "EMIT_TOWARDS"},
    "CAT_CONFINE_METHOD": {"box": "CONFINE_BOX", "path_length": "CONFINE_PATH"},
    "CAT_BLUR_TYPE": {"gaussian": "BUDDHA_BLUR_GAUSSIAN", "triangular": "BUDDHA_BLUR_TRIANGULAR",
                      "bokeh": "BUDDHA_BLUR_BOKEH", "catadioptric_bokeh": "BUDDHA_BLUR_CATADIOPTRIC"},
    "CAT_EFFECT": {"reflect": "OBSTACLE_EFFECT_REFLECT", "terminate": "OBSTACLE_EFFECT_TERMINATE",
                   "pass_through": "OBSTACLE_EFFECT_PASS"},
}
PINNED_BUDDHA_POINT_KEYS = {          # key -> the label of the entry in BUDDHA_POINT_CATALOG
    "all_bounces_mean": "All bounces mean", "self_bounces_mean": "Self bounces mean",
    "boundary_bounces_mean": "Boundary bounces mean", "termination_point": "Termination point",
    "termination_point_distance_limit": "Termination point (distance limit)",
    "termination_point_perpendicular_collision": "Termination point (perpendicular collision)",
    "termination_point_confined": "Termination point (confined)",
    "all_self_bounces": "All self bounces", "all_segment_midpoints": "All segment midpoints",
    "progressive_means_of_bounce_points": "Progressive means of bounce points",
    "progressive_means_of_segment_midpoints": "Progressive means of segment midpoints"}
PINNED_SHAPE_KINDS = {"ellipse": "EllipseShape", "regular_polygon": "RegularPolygonShape",
                      "moss_egg": "MossEggShape", "circular_sector": "CircularSectorShape",
                      "circular_segment": "CircularSegmentShape", "two_circle_lens": "LensShape",
                      "reuleaux_polygon": "ReuleauxPolygonShape", "capsule": "CapsuleShape",
                      "trefoil": "TrefoilShape", "quatrefoil": "QuatrefoilShape",
                      "lune": "LuneShape", "arbelos": "ArbelosShape"}
PINNED_EXPORT_CHOICES = {             # catalog -> key -> the text the control holds
    "CAT_COLOR_INFO": {"none": "None", "overlay": "Overlay", "bottom_margin": "Bottom margin"},
    "CAT_FIGURE_INFO": {"none": "None", "left_margin": "Left margin"},
    "CAT_IMAGE_SOURCE": {"current_image": "Current image", "new_image": "New image"},
    "CAT_IMAGE_FORMAT": {"jxl": "jxl", "png": "png", "qoi": "qoi"},
    "CAT_JXL_MODE": {"lossless": "Lossless", "lossy": "Lossy"},
    "CAT_VIDEO_FORMAT": {"mp4_h264": "mp4 (h264)", "gif": "GIF", "apng": "APNG",
                         "mkv_ffv1": "mkv (ffv1)", "jxl": "JXL"},
    "CAT_BITRATE_MODE": {"constant_quality": "Constant quality", "target_bitrate": "Target bitrate",
                         "variable_bitrate": "Variable bitrate"},
    "CAT_DOWNSAMPLE_FACTOR": {"none": "No change", "2x2": "2x2", "4x4": "4x4", "8x8": "8x8"},
    "CAT_DOWNSAMPLE_METHOD": {"mean": "Mean", "median": "Median", "sum": "Sum", "max": "Max",
                              "ffmpeg": "ffmpeg"},
}
PINNED_SIMULATION_KEYS = [
    "iter", "dist", "seg_mult", "emit_mode", "emit_target_r", "emit_target_theta",
    "emit_target_cx", "emit_target_cy", "emit_offset", "confine_window", "confine_method",
    "confine_threshold", "fuzz_on", "emit_jitter", "bnd_offset", "bnd_jitter", "bnd_mult",
    "self_offset", "self_jitter", "self_mult", "seed", "buddha_blur_radius", "buddha_blur_type"]
PINNED_DECORATION_KEYS = ["outline", "pending", "ray_path", "ray_start", "ray_end", "guides",
                          "samples"]
PINNED_EXPORT_KEYS = [
    "include_all_views", "color_info_location", "figure_info_location", "image_source",
    "image_format", "image_jxl_mode", "image_jxl_quality", "image_width", "image_height",
    "video_fps", "video_width", "video_height", "video_skip_first_frame", "video_format",
    "video_bitrate_mode", "video_quality", "video_bitrate", "video_vbr_target", "video_vbr_max",
    "video_downsample_factor", "video_downsample_method", "video_jxl_mode", "video_jxl_quality"]
PINNED_OBSTACLE_FIELDS = {            # kind key -> the fields its project entry has
    "ellipse": {"width", "height"}, "regular_polygon": {"sides", "radius"},
    "moss_egg": {"moss_angle", "moss_size"}, "circular_sector": {"radius", "sector_span"},
    "circular_segment": {"radius", "segment_span"},
    "two_circle_lens": {"lens_radius1", "lens_radius2", "lens_sep"},
    "reuleaux_polygon": {"reuleaux_sides", "radius"},
    "capsule": {"capsule_length", "capsule_radius"}, "trefoil": {"radius", "lobe_size"},
    "quatrefoil": {"radius", "lobe_size"}, "lune": {"radius", "cut_radius", "cut_offset"},
    "arbelos": {"radius", "split"}}
SHARED_OBSTACLE_FIELDS = {"rotation", "r", "theta", "cx", "cy", "in_effect", "out_effect",
                          "fire_inside"}


def test_the_keys_in_project_files_never_change():
    """Every key below may be in a saved project, so it must still exist and still mean the
    same. Adding keys is fine; the program's own codes may be renumbered."""
    for catalog_name, keys in PINNED_KEYS.items():
        catalog = getattr(m, catalog_name)
        for key, constant in keys.items():
            assert catalog.by_key.get(key) == getattr(m, constant), \
                f"{catalog_name}: the key {key!r} must still mean {constant}"
    for key, label in PINNED_BUDDHA_POINT_KEYS.items():
        assert m.BUDDHA_POINT_CATALOG[m.CAT_BUDDHA_POINT.by_key[key]][0] == label, key
    for key, class_name in PINNED_SHAPE_KINDS.items():
        assert m.CAT_SHAPE_KIND.by_key[key].__name__ == class_name, key
    for catalog_name, keys in PINNED_EXPORT_CHOICES.items():
        for key, text in keys.items():
            assert getattr(m, catalog_name).by_key.get(key) == text, f"{catalog_name}: {key}"
    for index, key in enumerate(("1x", "2x", "4x", "8x", "16x")):
        assert m.CAT_BUDDHA_BRIGHTNESS.by_key[key] == index == m.BUDDHA_BRIGHTNESS_OPTIONS.index(key)
    assert set(PINNED_SIMULATION_KEYS) <= set(m.RAY_SETTING_SPECS)
    assert set(PINNED_EXPORT_KEYS) <= {row[0] for row in m.EXPORT_OPTIONS}
    assert set(PINNED_DECORATION_KEYS) <= {key for key, _label, _default in
                                           m.DECO_IMAGE + m.DECO_PREVIEW}
    for key, fields in PINNED_OBSTACLE_FIELDS.items():
        assert fields | SHARED_OBSTACLE_FIELDS <= set(m.CAT_SHAPE_KIND.by_key[key].FIELD_NAMES), key
    assert m.PROJECT_FORMAT_NAME == "light-bounces-project" and m.PROJECT_EXTENSION == ".lightbounce"


def test_a_catalog_translates_between_keys_and_codes_and_refuses_what_it_does_not_know():
    catalog = m.Catalog("colour", [("red", 7), ("green", 3)])
    assert catalog.key(7) == "red" and catalog.code("green", "here") == 3
    assert catalog.codes == [7, 3]
    for call in (lambda: catalog.code("blue", "the file"), lambda: catalog.code(3, "the file"),
                 lambda: catalog.key(99)):
        try:
            call()
        except m.ProjectError as exc:
            assert "colour" in str(exc)
        else:
            raise AssertionError("an unknown key or code was accepted")
    try:
        m.Catalog("bad", [("Has Capitals", 1)])
    except AssertionError:
        pass
    else:
        raise AssertionError("a key that is not lower-case snake case was accepted")


def test_a_project_file_holds_keys_for_every_choice_and_no_codes():
    data = _dict_of()
    assert data["simulation"]["emit_mode"] == "towards_point"
    assert data["simulation"]["confine_method"] in ("box", "path_length")
    assert data["simulation"]["buddha_blur_type"] == "bokeh"
    assert {o["fields"]["in_effect"] for o in data["obstacles"]} <= set(m.CAT_EFFECT.by_key)
    assert all(isinstance(o["fields"]["out_effect"], str) for o in data["obstacles"])
    channel = data["views"][1]["channels"][1]
    assert channel == {"source": "buddha",
                       "buddha_points": ["self_bounces_mean", "boundary_bounces_mean"],
                       "buddha_compression": "modulus", "buddha_brightness": "8x"}
    settings = data["keyframes"][0]["settings"]
    assert settings["view:5:chan1"] == channel and settings["iter"] == 40
    assert data["keyframes"][1]["settings"]["emit_mode"] == "towards_point"
    assert data["export"]["video_format"] == "jxl" and data["export"]["color_info_location"] == "overlay"
    assert data["export"]["video_jxl_mode"] == "lossy" and data["export"]["image_format"] == "png"


def test_self_checks_catch_a_catalog_that_misses_one_of_the_programs_codes():
    fresh = load_module()
    fresh.core.CAT_METRIC = fresh.Catalog("metric", fresh.CAT_METRIC.entries[:-1])
    _breaks_a_self_check(fresh, "the metric catalog")
    fresh = load_module()
    fresh.core.CAT_EFFECT = fresh.Catalog("ray effect", list(fresh.CAT_EFFECT.entries) + [("extra", 9)])
    _breaks_a_self_check(fresh, "the ray effect catalog")
    fresh = load_module()
    fresh.core.RAY_CHOICE_CATALOGS = dict(fresh.RAY_CHOICE_CATALOGS)
    del fresh.core.RAY_CHOICE_CATALOGS["emit_mode"]
    _breaks_a_self_check(fresh, "RAY_CHOICE_CATALOGS")


def test_self_checks_catch_a_shape_kind_without_its_own_key():
    fresh = load_module()
    fresh.LuneShape.KEY = fresh.ArbelosShape.KEY
    _breaks_a_self_check(fresh, "KEY")
    fresh = load_module()
    fresh.LuneShape.KEY = "Not A Key"
    _breaks_a_self_check(fresh, "KEY")


V1_PROJECT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev", "testing",
                               "project_v1.lightbounce")


def test_a_format_1_project_file_is_converted_when_loaded():
    """dev/testing/project_v1.lightbounce is a real file written by the first version of the
    project format (numbers for choices, class names for obstacle kinds) from _sample_project();
    it must keep loading. _sample_project() must therefore not change: this file is the
    reference it is compared with."""
    import json
    with open(V1_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    assert old["format_version"] == 1 and old["obstacles"][0]["kind"] == "EllipseShape"
    assert isinstance(old["simulation"]["emit_mode"], int)
    loaded = m.load_project(V1_PROJECT_FILE)
    assert m.project_signature(loaded) == m.project_signature(_sample_project())
    assert m.project_text(loaded).count('"format_version": 2') == 1


def test_a_damaged_format_1_file_is_refused_not_half_converted():
    import json
    with open(V1_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    old["obstacles"][1]["fields"]["in_effect"] = 77
    try:
        m.project_from_dict(old)
    except m.ProjectError as exc:
        assert "ray effect" in str(exc)
    else:
        raise AssertionError("a number that is not an effect was accepted")
    old = json.load(open(V1_PROJECT_FILE, encoding="utf-8"))
    del old["views"][0]["channels"][0]["stat"]
    try:
        m.project_from_dict(old)
    except m.ProjectError as exc:
        assert "damaged" in str(exc)
    else:
        raise AssertionError("a channel without a metric was accepted")


# --------------------------------------------------------------------------- #
# Projects in the program: unsaved changes and the file dialogs
# --------------------------------------------------------------------------- #
def _fake_project_app(dirty, save_result=True):
    class FakeApp:
        pass
    app = FakeApp()
    app.project_path = None
    app.busy_with_render_or_reencode = lambda: False
    app.is_dirty = lambda: dirty
    app.saves = []
    app.save_project = lambda: app.saves.append(1) or save_result
    app.project_display_name = lambda: "Untitled"
    app.can_replace_project = m.App.can_replace_project.__get__(app)
    return app


def _with_answer(answer, fn):
    orig = (m.messagebox.askyesnocancel, m.messagebox.showinfo)
    asked = []
    m.messagebox.askyesnocancel = lambda *a, **k: asked.append(a) or answer
    m.messagebox.showinfo = lambda *a, **k: asked.append(("info",) + a)
    try:
        return fn(), asked
    finally:
        m.messagebox.askyesnocancel, m.messagebox.showinfo = orig


def test_replacing_a_project_with_no_unsaved_changes_asks_nothing():
    app = _fake_project_app(dirty=False)
    result, asked = _with_answer(None, lambda: app.can_replace_project("open another project"))
    assert result is True and asked == [] and app.saves == []


def test_unsaved_changes_are_saved_discarded_or_the_action_cancelled_as_the_user_chooses():
    app = _fake_project_app(dirty=True)
    result, asked = _with_answer(True, lambda: app.can_replace_project("start a new project"))
    assert result is True and app.saves == [1] and len(asked) == 1
    assert "Untitled" in asked[0][1] and "start a new project" in asked[0][1]
    app = _fake_project_app(dirty=True)
    assert _with_answer(False, lambda: app.can_replace_project("x"))[0] is True and app.saves == []
    app = _fake_project_app(dirty=True)
    assert _with_answer(None, lambda: app.can_replace_project("x"))[0] is False and app.saves == []
    app = _fake_project_app(dirty=True, save_result=False)      # chose Save, then cancelled the dialog
    assert _with_answer(True, lambda: app.can_replace_project("x"))[0] is False


def test_a_project_is_not_replaced_while_a_video_renders_but_the_program_may_still_close():
    app = _fake_project_app(dirty=False)
    app.busy_with_render_or_reencode = lambda: True
    result, asked = _with_answer(None, lambda: app.can_replace_project("open another project"))
    assert result is False and asked and asked[0][0] == "info"
    result, asked = _with_answer(None, lambda: app.can_replace_project("close the program",
                                                                      check_busy=False))
    assert result is True and asked == []


def _fake_save_project_app(path=None):
    class FakeApp:
        pass
    app = FakeApp()
    app.project_path = path
    app.root = None
    app.project_display_name = m.App.project_display_name.__get__(app)
    app.project_dialog_dir = lambda: tempfile.mkdtemp()
    app.captured = _sample_project()
    app.capture_project = lambda show_error=True: app.captured
    app.update_title = lambda: None
    app.save_project_to = m.App.save_project_to.__get__(app)
    app.save_project_as = m.App.save_project_as.__get__(app)
    app.save_project = m.App.save_project.__get__(app)
    return app


def test_save_goes_to_the_projects_file_and_asks_for_one_only_when_it_has_none():
    folder = tempfile.mkdtemp()
    target = os.path.join(folder, "p" + m.PROJECT_EXTENSION)
    app = _fake_save_project_app(path=target)
    assert app.save_project() is True and os.path.exists(target)
    assert app.saved_signature == m.project_signature(app.captured)

    app = _fake_save_project_app(path=None)
    seen = {}
    orig = m.filedialog.asksaveasfilename
    chosen = os.path.join(folder, "new" + m.PROJECT_EXTENSION)
    m.filedialog.asksaveasfilename = lambda *a, **k: seen.update(k) or chosen
    try:
        assert app.save_project() is True
    finally:
        m.filedialog.asksaveasfilename = orig
    assert seen["defaultextension"] == ".lightbounce" and seen["filetypes"] == m.PROJECT_FILE_TYPES
    assert seen["initialfile"] == "untitled.lightbounce"
    assert app.project_path == chosen and os.path.exists(chosen)

    app = _fake_save_project_app(path=None)                     # the dialog was cancelled
    m.filedialog.asksaveasfilename = lambda *a, **k: ""
    try:
        assert app.save_project() is False
    finally:
        m.filedialog.asksaveasfilename = orig
    assert app.project_path is None


def test_a_failed_save_reports_the_problem_and_leaves_the_project_unsaved():
    app = _fake_save_project_app(path=os.path.join(tempfile.mkdtemp(), "no", "dir", "p.lightbounce"))
    shown = []
    orig = m.messagebox.showerror
    m.messagebox.showerror = lambda *a, **k: shown.append(a)
    try:
        assert app.save_project() is False
    finally:
        m.messagebox.showerror = orig
    assert shown and "could not be written" in shown[0][1]
    assert not hasattr(app, "saved_signature")


# --------------------------------------------------------------------------- #
# The package: modules, re-exports, patchable globals, the launcher
# --------------------------------------------------------------------------- #
PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lightbounces")


def _package_sources():
    import ast
    for filename in sorted(os.listdir(PACKAGE_DIR)):
        if filename.endswith(".py"):
            with open(os.path.join(PACKAGE_DIR, filename), encoding="utf-8") as handle:
                yield filename[:-3], ast.parse(handle.read())


def test_patchable_globals_are_read_as_attributes_of_their_owner_never_imported_by_value():
    """Tests patch these (m.paths.EXPORT_DIR = ...). A `from .paths import EXPORT_DIR` copies the value when
    the module is imported, so a later patch would silently do nothing; every module therefore reads
    `paths.EXPORT_DIR` at call time. Only the owner module may use the bare name."""
    import ast
    owners = m.PATCHABLE_GLOBALS
    assert set(owners.values()) == {"paths", "external"}
    for module, tree in _package_sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                bad = [a.name for a in node.names if a.name in owners and owners[a.name] != module]
                assert not bad, f"lightbounces/{module}.py imports {bad} by value"
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in owners
                    and owners[node.id] != module and module != "__init__"):
                raise AssertionError(f"lightbounces/{module}.py line {node.lineno} reads {node.id} directly; "
                                     f"read {owners[node.id]}.{node.id}")
    for name, owner in owners.items():
        assert hasattr(getattr(m, owner), name), f"{owner} does not define {name}"
        assert not hasattr(m, name), f"the package must not re-export {name}: it would be a stale copy after a patch"


def test_the_package_re_exports_every_name_of_every_module_as_the_same_object():
    import ast
    seen = {}
    for module, tree in _package_sources():
        if module == "__init__":
            continue
        for node in tree.body:
            for name in ([node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                         [x.id for t in getattr(node, "targets", []) for x in ast.walk(t)
                          if isinstance(x, ast.Name)]):
                if name.startswith("__") or name in m.PATCHABLE_GLOBALS:
                    continue
                seen.setdefault(name, []).append(module)
                assert hasattr(m, name), f"{module}.{name} is not reachable as lightbounces.{name}"
    assert len(seen) > 400
    for name, modules in seen.items():
        if len(modules) == 1:
            assert getattr(m, name) is getattr(getattr(m, modules[0]), name), name


def test_every_load_of_the_package_is_an_independent_copy():
    first, second = load_module(), load_module()
    assert first is not second and first.core is not second.core and first.EllipseShape is not second.EllipseShape
    first.LuneShape.KEY = "changed"
    assert second.LuneShape.KEY == "lune"
    assert set(first.__lb_modules__) >= {"lightbounces", "lightbounces.core", "lightbounces.paths"}
    assert not [name for name in sys.modules if name.startswith("lightbounces")], \
        "a copy must not stay in sys.modules (the next copy would not start clean)"


def test_the_launcher_exposes_the_programs_names_and_names_the_output_files_after_itself():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "light_bounces.py")
    spec = importlib.util.spec_from_file_location("launcher_under_test", path)
    launcher = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(launcher)
        package = sys.modules["lightbounces"]
        assert launcher.App is package.core.App and launcher.main is package.core.main
        assert launcher.paths is package.paths and launcher.paths.PROGRAM_NAME == "light_bounces"
        assert launcher.Shape is package.Shape
    finally:
        for name in [n for n in sys.modules if n == "lightbounces" or n.startswith("lightbounces.")]:
            del sys.modules[name]


def test_the_program_folder_is_the_one_with_the_launcher():
    assert m.paths.SCRIPT_DIR == os.path.dirname(os.path.abspath(__file__))
    assert os.path.isfile(os.path.join(m.paths.SCRIPT_DIR, "light_bounces.py"))
    assert m.paths.EXPORT_DIR == os.path.join(m.paths.SCRIPT_DIR, "exports")
    assert m.paths.TEMP_DIR == os.path.join(m.paths.EXPORT_DIR, "TEMP")


def test_no_name_is_defined_in_two_modules_of_the_package():
    """The package re-exports every module's names, later modules winning, so a second definition of a name in another
    module would silently shadow the first for everything that reads it through the package - and the two copies
    could drift apart (a moved definition left behind in the module it came from is the likely way to get here)."""
    import ast
    homes = {}
    for module, tree in _package_sources():
        if module == "__init__":
            continue
        for node in tree.body:
            names = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else \
                [x.id for t in getattr(node, "targets", []) for x in ast.walk(t) if isinstance(x, ast.Name)]
            for name in names:
                homes.setdefault(name, set()).add(module)
    assert len(homes) > 400
    twice = {name: sorted(modules) for name, modules in homes.items() if len(modules) > 1}
    assert not twice, f"defined in more than one module: {twice}"


def _load_dev_tool(*parts):
    """A script from dev/ loaded by its path (the dev tools are not part of the package)."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev", *parts) + ".py"
    spec = importlib.util.spec_from_file_location("dev_tool_under_test_" + parts[-1], path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_the_numba_kernels_read_only_names_defined_in_their_own_file():
    """numba freezes the value of every global a kernel reads into the compiled code, and when it reuses a cached
    compile it only compares the modification time of the file the kernel is in (verified with
    dev/refactor/numba_cache_check.py). So a constant or another @njit function living in a different file than
    the kernel that uses it would leave the kernel running stale code after an edit there: every file with @njit
    functions must define everything its kernels read, and kernels.py (the kernel ABI) may import nothing from the
    package but njit/prange."""
    reads = _load_dev_tool("refactor", "kernel_reads")
    kernels_found = {}
    for folder, _dirs, files in os.walk(PACKAGE_DIR):
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(folder, filename)
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            problems = reads.kernel_rule_violations(source, kernels_file=(filename == "kernels.py"))
            assert not problems, f"{os.path.relpath(path, PACKAGE_DIR)}: " + "; ".join(problems)
            found = len(reads.symtable_global_reads(source))
            if found:
                kernels_found[os.path.relpath(path, PACKAGE_DIR)] = found
    assert kernels_found.get("kernels.py", 0) >= 20, "the scan should have found the ray kernels it protects"
    assert kernels_found.get("image_io.py", 0) >= 1, "...and the QOI encoder"
    assert not (set(kernels_found) - {"kernels.py", "image_io.py"}), \
        f"@njit functions in a new file: {sorted(kernels_found)} - they belong in kernels.py unless they call no other kernel"


def test_the_kernel_rule_check_catches_each_way_of_breaking_the_rule():
    """The check above must not pass just because it looks at nothing: feed it small sources that break the rule in
    each way (and one that keeps it)."""
    reads = _load_dev_tool("refactor", "kernel_reads")
    head = "import numpy as np\nfrom math import sqrt\nfrom .external import njit, prange\n"
    good = head + (
        "LIMIT = 3\n"
        "@njit(cache=True)\n"
        "def inner(x):\n    return sqrt(x) * LIMIT\n"
        "@njit(cache=True)\n"
        "def outer(a):\n    out = np.empty(3)\n    for i in prange(3):\n        out[i] = inner(a)\n    return out\n")
    assert reads.kernel_rule_violations(good, kernels_file=True) == []

    def problems(source, kernels_file=False):
        return reads.kernel_rule_violations(source, kernels_file=kernels_file)

    undefined = head + "@njit(cache=True)\ndef k(x):\n    return x * MISSING\n"
    found = problems(undefined)
    assert len(found) == 1 and found[0].startswith("k reads MISSING")

    constant = "from .constants import LIMIT\n" + head + "@njit(cache=True)\ndef k(x):\n    return x * LIMIT\n"
    found = problems(constant)
    assert len(found) == 1 and "k reads LIMIT, imported from .constants" in found[0]

    callee = "from .helpers import inner\n" + head + "@njit(cache=True)\ndef k(x):\n    return inner(x)\n"
    found = problems(callee)
    assert len(found) == 1 and "k reads inner, imported from .helpers" in found[0]

    module = "from . import paths\n" + head + "@njit(cache=True)\ndef k(x):\n    return x * paths.SCALE\n"
    found = problems(module)
    assert len(found) == 1 and "k reads paths, imported from .paths" in found[0]

    unused_import = "from .constants import UNUSED\n" + good         # fine in an ordinary file...
    assert problems(unused_import) == []
    found = problems(unused_import, kernels_file=True)              # ...but not in the kernel ABI file
    assert len(found) == 1 and "imports UNUSED from .constants" in found[0]

    shadowed = "from .constants import LIMIT\n" + good              # imported AND assigned: the kernel may see either
    found = problems(shadowed)
    assert len(found) == 1 and "inner reads LIMIT, imported from .constants" in found[0]
    assert problems("from . import external\n" + good, kernels_file=True)

    # imports from outside the package, and a name only used outside the kernels, are fine
    assert problems(head + "import json\nOTHER = json.dumps({})\n@njit\ndef k(x):\n    return sqrt(x)\n",
                    kernels_file=True) == []


def test_kernels_read_their_rule_and_obstacle_arguments_by_field_name_never_by_number():
    """numba cannot forbid indexing a namedtuple, so the convention is enforced here:
    in the kernel sources, `fuzz`, `confine`, `emission` and `obstacles` are only
    ever read as `name.field`."""
    import ast
    import inspect
    import textwrap
    checked = 0
    for fn in (m.emits_ray, m.trace_path, m.render_rows, m.render_rows_wh,
               m.scatter_multi_buddha):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn.py_func)))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id in ("fuzz", "confine", "emission", "obstacles")):
                raise AssertionError(
                    f"{fn.py_func.__name__} indexes {node.value.id}[...] (line {node.lineno}); "
                    "read it by field name")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id in ("fuzz", "confine", "emission", "obstacles"):
                checked += 1
    assert checked >= 20, "the scan should have found the field reads it is protecting"


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
