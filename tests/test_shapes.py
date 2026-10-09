"""Tests: shapes."""
import dataclasses
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


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
                m.LuneShape: m.LAYOUT_ARC_LOOPS, m.ArbelosShape: m.LAYOUT_ARC_LOOPS,
                m.SuperellipseShape: m.LAYOUT_SQUIRCLE, m.FernandezGuastiShape: m.LAYOUT_SQUIRCLE,
                m.ArchimedeanSpiralShape: m.LAYOUT_SPIRAL, m.PythagorasSnailShape: m.LAYOUT_SNAIL,
                m.HugelschafferEggShape: m.LAYOUT_EGG, m.SineFlowerShape: m.LAYOUT_FLOWER}
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
    app.current_generation_settings = lambda: (m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_TERMINATION_SETTINGS)
    app.current_blur_settings = lambda: m.DEFAULT_BUDDHA_BLUR
    app.current_buddha_seed_settings = lambda: m.DEFAULT_BUDDHA_SEED
    app.current_engine_settings = lambda: m.DEFAULT_ENGINE_SETTINGS
    app.deco = {"outline": _StubVar(True)}
    app.colorations = [m.ColorationConfig(0)]
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
             (m.SHAPE_ARBELOS, m.arbelos_bound(1.0, 0.35, 30.0)),
             (m.SHAPE_SUPERELLIPSE, m.superellipse_bound(1.6, 1.0, 4.0, 20.0)),
             (m.SHAPE_FG_SQUIRCLE, m.fg_squircle_bound(0.9, 0.8, 15.0)),
             (m.SHAPE_SPIRAL, m.spiral_bound(1.0, 0.25, 0.5, 30.0)),
             (m.SHAPE_SNAIL, m.snail_bound(17, 20.0)),
             (m.SHAPE_SNAIL, m.snail_bound(9, 20.0)),
             (m.SHAPE_EGG, m.egg_bound(1.0, 0.6, 0.5, 25.0)),
             (m.SHAPE_FLOWER, m.flower_bound(0.8, 7, 0.4, 25.0))]
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
    width = layer_px * m.DEFAULT_PIXEL_WORLD_WIDTH
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
    app.generated = type("Generated", (), {})()
    app.generated.obstacle_states = {0: o.state()}     # unchanged since generation
    assert app.obstacle_is_pending(o)
    o.out_effect = m.OBSTACLE_EFFECT_REFLECT
    app.generated.obstacle_states = {0: o.state()}
    assert not app.obstacle_is_pending(o)


# --------------------------------------------------------------------------- #
# The fixed bounding shape is gone: the scene is only obstacles, starting with
# an obstacle that stands in for it
# --------------------------------------------------------------------------- #
def test_the_starting_obstacle_is_the_circle_of_radius_2_that_reflects_and_fires_rays_inside():
    """(It fills the default world [-2, 2], as the unit circle filled [-1, 1].)"""
    start = m.default_starting_obstacle(7)
    assert start.id == 7
    assert type(start) is m.EllipseShape
    assert (start.width, start.height, start.rotation) == (4.0, 4.0, 0.0)
    assert (start.r, start.theta, start.cx, start.cy) == (0.0, 0.0, 0.0, 0.0)
    assert (start.in_effect, start.out_effect) == (m.OBSTACLE_EFFECT_REFLECT,
                                                     m.OBSTACLE_EFFECT_REFLECT)
    assert start.fire_inside is True
    assert np.allclose(start.bound(), [4.0 * np.pi, 0.25, 0.0, 0.25]), "the circle of radius 2: area 4 pi"
    assert np.allclose(start.bound(), m.ellipse_bound(4.0, 4.0, 0.0))


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
        "seg_mult", "fuzz",
        "emit_mode", "emit_target_r", "emit_target_theta", "emit_target_cx", "emit_target_cy",
        "emit_offset"]
    assert [f.name for f in dataclasses.fields(m.NaturalTerminationSettings)] == [
        "max_iter", "max_dist", "escape_radius",
        "confine_lookback", "confine_method", "confine_threshold"]
    assert [f.name for f in dataclasses.fields(m.BuddhaBlurSettings)] == ["radius", "type"]


def test_no_keyframe_setting_describes_a_bounding_shape_any_more():
    keys = {k for k, _tab, _label in m.SETTING_INFO}
    assert keys == {"iter", "dist", "escape_radius", "seg_mult", "fuzz_on", "emit_jitter",
                    "bnd_offset", "bnd_jitter", "bnd_mult", "self_offset", "self_jitter",
                    "self_mult", "seed", "buddha_blur_radius", "buddha_blur_type", "outline",
                    "confine_lookback", "confine_method", "confine_threshold",
                    "emit_mode", "emit_target_r", "emit_target_theta", "emit_target_cx",
                    "emit_target_cy", "emit_offset", "buddha_seed_camera", "buddha_seed_x_start",
                    "buddha_seed_x_stop", "buddha_seed_y_start", "buddha_seed_y_stop", "engine"}
    assert keys == set(m.DEFAULT_SETTING_VALUES)
    tabs = {tab for _k, tab, _label in m.SETTING_INFO}
    assert tabs == {"Emissions", "Termination", "Mutations", "Buddha seeding", "Buddha blur", "Decorations", "Simulation"}


def test_a_frame_resolved_from_no_keyframes_uses_the_defaults():
    data = m.resolve_frame_settings([], 0, m.DEFAULT_SETTING_VALUES)
    assert data.ray == m.DEFAULT_RAY_SETTINGS
    assert data.buddha_blur == m.DEFAULT_BUDDHA_BLUR
    assert data.outline is True


def test_settings_suffix_is_only_the_limits_and_fuzzing():
    assert m.settings_suffix(m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_TERMINATION_SETTINGS) == "iter256-distNONE"
    limited = _generation(max_iter=10, max_dist=2.5)
    assert m.settings_suffix(*limited) == "iter10-dist2.5"
    fuzzed = _generation(fuzz=m.DEFAULT_FUZZ.replace(enabled=True, rng_seed=9))
    suffix = m.settings_suffix(*fuzzed)
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
           m.ArbelosShape: {"radius", "split"},
           m.SuperellipseShape: {"width", "height", "exponent"},
           m.FernandezGuastiShape: {"radius", "squareness"},
           m.ArchimedeanSpiralShape: {"radius", "spiral_pitch", "spiral_fill"},
           m.PythagorasSnailShape: {"snail_segments", "radius"},
           m.HugelschafferEggShape: {"egg_a", "egg_b", "egg_shift"},
           m.SineFlowerShape: {"radius", "flower_petals", "flower_amplitude"}}
    shared = {"rotation", "r", "theta", "cx", "cy", "in_effect", "out_effect", "fire_inside"}
    assert set(own) == set(m.SHAPE_CLASSES)
    for cls, fields in own.items():
        assert set(cls.FIELD_NAMES) == fields | shared, cls.__name__
        o = cls(0)
        for field in fields:
            assert hasattr(o, field)
        for field in set().union(*own.values()) - fields:
            assert not hasattr(o, field), (cls.__name__, field)     # no field it doesn't use


def test_one_rotation_serves_every_kind_and_the_default_radius_is_2_for_every_kind_with_one():
    for cls in m.SHAPE_CLASSES:
        o = cls(0)
        assert o.rotation == 0.0
        o.rotation = 33.0
        assert 33.0 in o.params(), cls.__name__
        if "radius" in cls.FIELD_NAMES:
            assert o.radius == 2.0 == m.DEFAULT_RADIUS, cls.__name__


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
        path_points = np.empty((40, 2))
        path_point_tags = np.empty(40)
        raw = np.empty(m.N_RAW)
        rng = np.random.RandomState(2)
        bounced = 0
        for _ in range(40):
            x, y = rng.uniform(-1, 1, 2)
            if not m.inside_boundary(x, y, cls.KIND, bound):
                continue
            term, npts = m.trace_path(x, y, path_points, path_point_tags, _termination(30), _engine_args(arrays), raw)
            for k in range(1, npts):
                if path_point_tags[k] != m.TAG_BOUNDARY:                # (a bounce off the ray's own path
                    continue                                   # is not on the shape's edge)
                px, py = path_points[k]
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
    assert (frozen.capsule_length, frozen.r, o.capsule_length, o.r) == (1.5, 0.25, 2.0, 0.0)
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
                             "Trefoil", "Quatrefoil", "Lune", "Arbelos", "Superelliptical squircle",
                             "Fernández-Guasti squircle", "Archimedean spiral", "Snail of Pythagoras",
                             "Hügelschäffer egg", "Sine flower"]



def test_render_full_image_colors_escaped_rays_when_the_starting_obstacle_lets_rays_out():
    app = _fake_render_app()
    start = m.default_starting_obstacle(0)
    start.out_effect = P
    data = m.resolve_frame_settings([], 0, m.DEFAULT_SETTING_VALUES)
    img = app.render_full_image(data, 40, 40, m.build_obstacle_arrays([start]),
                                (_terminated_coloration(),), m.DEFAULT_FRAMING)
    assert tuple(img[20, 20]) == (255, 0, 0), "escaped rays are termination value 1.0"
    assert tuple(img[0, 0]) == tuple(m.BACKGROUND), "outside the obstacle nothing fires"


def test_render_full_image_with_the_starting_obstacle_produces_no_escaped_rays():
    app = _fake_render_app()
    data = m.resolve_frame_settings([], 0, m.DEFAULT_SETTING_VALUES)
    img = app.render_full_image(data, 40, 40, _starting_obst_arrays(), (_terminated_coloration(),), m.DEFAULT_FRAMING)
    assert img[20, 20][0] != 255


def test_render_full_image_of_an_empty_scene_is_just_the_background():
    app = _fake_render_app()
    data = m.resolve_frame_settings([], 0, m.DEFAULT_SETTING_VALUES)
    empty = m.build_obstacle_arrays([])
    img = app.render_full_image(data, 24, 24, empty, (_terminated_coloration(),), m.DEFAULT_FRAMING)
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
    width = m.OBST_RING_LAYER_PX * m.DEFAULT_PIXEL_WORLD_WIDTH
    assert np.isclose(r[outer].max() - 1.0, width, atol=2 * step)
    assert np.isclose(1.0 - r[inner].min(), width, atol=2 * step)


# --------------------------------------------------------------------------- #
# The two squircles (superelliptical, Fernandez-Guasti) and the Archimedean spiral ribbon
# --------------------------------------------------------------------------- #
def test_the_squircles_and_the_spiral_agree_with_independent_formulas_about_what_is_inside():
    rng = np.random.RandomState(11)
    pts = rng.uniform(-1.5, 1.5, size=(3000, 2))
    for o in _new_kind_cases():
        member = _independent_membership(o)
        bound = o.bound()
        got = [bool(m.inside_boundary(x, y, o.KIND, bound)) for x, y in pts]
        want = [bool(member(x, y)) for x, y in pts]
        assert got == want, (type(o).__name__, o.state())
        assert any(want) and not all(want), (type(o).__name__, "the test points must hit both")


def test_the_new_kinds_have_the_documented_fields_defaults_and_names():
    assert [cls.KIND for cls in NEW_CLASSES] == [12, 13, 14, 15, 16, 17]
    assert [cls.KEY for cls in NEW_CLASSES] == ["superelliptical_squircle",
                                                "fernandez_guasti_squircle", "archimedean_spiral",
                                                "snail_of_pythagoras", "hugelschaffer_egg", "sine_flower"]
    assert [cls.NAME for cls in NEW_CLASSES] == ["Superelliptical squircle",
                                                 "Fernández-Guasti squircle",
                                                 "Archimedean spiral", "Snail of Pythagoras",
                                                 "Hügelschäffer egg", "Sine flower"]
    sup, fg, spiral, snail, egg, flower = (cls(0) for cls in NEW_CLASSES)
    assert (flower.radius, flower.flower_petals, flower.flower_amplitude) == (2.0, 6, 0.3)
    assert (egg.egg_a, egg.egg_b, egg.egg_shift) == (2.0, 1.4, 0.4)
    assert snail.snail_segments == 17 and snail.radius == 2.0
    assert (sup.width, sup.height, sup.exponent) == (2.0, 2.0, 4.0)
    assert (fg.radius, fg.squareness) == (2.0, 0.7)
    assert (spiral.radius, spiral.spiral_pitch, spiral.spiral_fill) == (2.0, 0.5, 0.5)
    for o in (sup, fg, spiral, snail, egg, flower):   # every default is inside its own slider range
        for p in o.NUMERIC_PARAMS:
            if p.lo is not None:
                assert p.lo <= getattr(o, p.field) <= p.hi, (type(o).__name__, p.field)


def test_the_squircles_reduce_to_the_ellipse_and_the_square_at_their_limits():
    sup = m.SuperellipseShape(0)
    sup.width, sup.height, sup.exponent = 1.4, 0.8, 2.0
    assert np.isclose(sup.bound()[0], np.pi * 0.7 * 0.4, rtol=1e-12)         # n = 2: an ellipse
    sup.exponent = 1.0
    assert np.isclose(sup.bound()[0], 2 * 0.7 * 0.4, rtol=1e-12)             # n = 1: a diamond
    sup.exponent = 16.0
    assert 0.97 * 4 * 0.7 * 0.4 < sup.bound()[0] < 4 * 0.7 * 0.4             # n large: the box
    fg = m.FernandezGuastiShape(0)
    fg.radius = 0.9
    fg.squareness = 0.0
    assert np.isclose(fg.bound()[0], np.pi * 0.81, rtol=1e-4)                # s = 0: a circle
    fg.squareness = 1.0
    assert np.isclose(fg.bound()[0], 4 * 0.81, rtol=1e-4)                    # s = 1: the square


def test_the_spiral_area_matches_a_count_of_independent_inside_points():
    o = m.ArchimedeanSpiralShape(0)
    o.radius, o.spiral_pitch, o.spiral_fill = 1.0, 0.25, 0.5
    member = _independent_membership(o)
    n = 500
    xs = (np.arange(n) + 0.5) * (2.2 / n) - 1.1
    count = sum(member(x, y) for x in xs for y in xs)
    assert np.isclose(o.bound()[0], count * (2.2 / n) ** 2, rtol=0.03)


def test_the_new_kinds_outlines_run_along_the_edge_and_stay_within_their_extent():
    for o in _new_kind_cases():
        bound = o.bound()
        xs, ys = o.boundary_xy()
        assert len(xs) == len(ys) > 3                # (a one-triangle snail has few corners)
        assert np.hypot(xs, ys).max() <= o.extent() * 1.001, (type(o).__name__, o.state())
        step = max(1, len(xs) // 60)
        for x, y in list(zip(xs, ys))[::step]:
            if isinstance(o, m.ArchimedeanSpiralShape) and np.hypot(x, y) < 0.02:
                continue            # the ribbon starts in a wedge too thin for a probe ring to find
            if isinstance(o, m.PythagorasSnailShape) and np.hypot(x, y) < 1e-3:
                continue            # the centre: all round it once the turns overlap
            around = [m.inside_boundary(x + 3e-5 * np.cos(a), y + 3e-5 * np.sin(a), o.KIND, bound)
                      for a in np.linspace(0.0, 2 * np.pi, 16, endpoint=False)]
            assert any(around) and not all(around), (type(o).__name__, o.state(), x, y)


def test_a_spiral_too_small_to_hold_its_ribbon_is_empty_and_harmless():
    """The ribbon is `spiral_fill` of a pitch wide, so a maximum radius under that width leaves no
    room for even a part turn: the shape holds nothing, fires nothing and is never hit."""
    o = m.ArchimedeanSpiralShape(0)
    o.radius, o.spiral_pitch, o.spiral_fill = 0.05, 0.25, 0.5
    bound = o.bound()
    assert bound[0] == 0.0
    for x, y in ((0.0, 0.0), (0.02, 0.01), (0.5, 0.5)):
        assert not m.inside_boundary(x, y, o.KIND, bound)
    assert m.obstacle_hit(-1.0, 0.0, 1.0, 0.0, o.KIND, bound)[0] < 0.0


def test_the_pixel_font_has_a_acute_for_the_fernandez_guasti_name():
    glyph = m.FONT_5X7["á"]
    assert len(glyph) == 7 and all(len(row) == 5 for row in glyph)
    assert glyph[1:] == m.FONT_5X7["a"][1:]                    # an a ...
    assert glyph[0] != m.FONT_5X7["a"][0] and "1" in glyph[0]  # ... with an accent above it
    assert m.render_text_line("Fernández", 1).any()


def test_the_snail_has_the_construction_s_right_triangles_and_fits_in_the_unit_circle():
    """Each triangle has its right angle at A_k, legs sqrt(k) and 1 (before scaling), and the farthest
    corner is one unit from the centre. The 17th triangle is the first to overlap the first turn."""
    bound = m.snail_bound(17, 0.0)
    scale = bound[2]
    corners = bound[3:].reshape(-1, 2) / scale
    assert len(corners) == 18 and np.allclose(corners[0], (1.0, 0.0))
    for k in range(1, 18):
        a, b = corners[k - 1], corners[k]
        assert np.isclose(np.hypot(*a), np.sqrt(k)) and np.isclose(np.hypot(*(b - a)), 1.0)
        assert np.isclose(a @ (b - a), 0.0, atol=1e-9)                     # the right angle is at A_k
    assert np.isclose(np.hypot(*bound[-2:]), 1.0)
    turned = np.arctan2(corners[16][1], corners[16][0])
    assert 0.0 > turned > -0.2 or 6.1 < turned + 2 * np.pi < 6.3          # A_17 is almost a whole turn on
    assert np.hypot(*m.PythagorasSnailShape(0).boundary_xy()).max() <= m.PythagorasSnailShape(0).radius + 1e-9


def test_the_snail_segment_count_is_a_whole_number_with_its_own_range():
    o = m.PythagorasSnailShape(0)
    assert o.FIELD_TYPES["snail_segments"] is int
    assert (m.SNAIL_MIN_SEGMENTS, m.SNAIL_MAX_SEGMENTS) == (1, 100)
    o.snail_segments = m.SNAIL_MAX_SEGMENTS
    assert len(o.bound()) <= m.OBST_BOUND_LEN
    assert m.build_obstacle_arrays([o]).bounds.shape == (1, m.OBST_BOUND_LEN)


def test_the_egg_is_an_ellipse_with_no_shift_and_its_outline_satisfies_the_pages_equation():
    egg = m.HugelschafferEggShape(0)
    egg.egg_a, egg.egg_b, egg.egg_shift = 1.2, 0.5, 0.0
    assert np.isclose(egg.bound()[0], np.pi * 1.2 * 0.5, rtol=1e-9)             # d = 0: the ellipse a x b
    for shift in (-0.8, 0.3, 0.9):
        egg.egg_a, egg.egg_b, egg.egg_shift, egg.rotation = 1.1, 0.6, shift, 0.0
        a, b, d = 1.1, 0.6, shift * 1.1
        xs, ys = egg.boundary_xy()
        x = xs + d                                                              # the page's x
        residual = (x - d) ** 2 / a ** 2 + ys ** 2 / b ** 2 * (1.0 + (2.0 * d * x - d * d) / a ** 2) - 1.0
        assert np.abs(residual).max() < 1e-9, shift
        assert np.isclose(np.abs(ys).max(), b, rtol=1e-6) and np.isclose(xs.min(), -a) and np.isclose(xs.max(), a)
        # the page's parametrization, x = cos(t) (sqrt(a^2 - d^2 sin^2 t) + d cos t), y = b sin t
        t = np.linspace(0.0, 2.0 * np.pi, 361)
        page_x = np.cos(t) * (np.sqrt(a * a - d * d * np.sin(t) ** 2) + d * np.cos(t))
        assert np.allclose(page_x - d, xs) and np.allclose(b * np.sin(t), ys)


def test_a_shift_the_egg_cannot_have_is_clamped_so_the_egg_stays_closed():
    egg = m.HugelschafferEggShape(0)
    egg.egg_shift = m.EGG_SHIFT_MAX
    assert egg.bound()[5] == 0.9 * egg.egg_a
    assert m.egg_bound(1.0, 0.5, 5.0, 0.0)[5] == 0.9


def test_the_eggs_ring_layers_stay_close_to_its_outline_even_at_a_large_shift():
    """Regression: at a shift near +-0.9 the curve's other branch comes in just past the egg's point, and the
    first-order distance estimate went to zero along its whole vertical asymptote: the outline layers drew a straight
    line right across the image. Every ring pixel must lie within a few layer widths of the egg's own outline."""
    xs = np.linspace(-1.0, 1.0, 400)
    gx, gy = np.meshgrid(xs, xs[::-1])
    for a, b, shift, rotation in ((0.46, 0.24, -0.9, 45.0), (0.46, 0.24, 0.9, 45.0), (0.3, 0.5, -0.9, 20.0),
                                  (1.0, 0.7, 0.4, 0.0), (0.8, 0.2, 0.0, 130.0)):
        egg = m.HugelschafferEggShape(0)
        egg.egg_a, egg.egg_b, egg.egg_shift, egg.rotation = a, b, shift, rotation
        outer, inner = m.ring_layer_masks(m.SHAPE_EGG, egg.bound(), gx, gy)
        ox, oy = egg.boundary_xy()
        curve = np.column_stack((np.interp(np.linspace(0, 1, 3000), np.linspace(0, 1, len(ox)), ox),
                                 np.interp(np.linspace(0, 1, 3000), np.linspace(0, 1, len(oy)), oy)))
        ring = np.column_stack((gx[outer | inner], gy[outer | inner]))
        assert len(ring) > 50, (a, b, shift)
        far = np.sqrt(((ring[:, None, :] - curve[None, ::3, :]) ** 2).sum(-1)).min(axis=1).max()
        assert far < 3 * m.OBST_RING_LAYER_PX * m.DEFAULT_PIXEL_WORLD_WIDTH + 0.01, (a, b, shift, far)


def test_the_sine_flower_has_the_exact_area_a_whole_number_of_petals_and_a_limited_amplitude():
    flower = m.SineFlowerShape(0)
    flower.radius, flower.flower_petals, flower.flower_amplitude = 0.9, 5, 0.4
    member = _independent_membership(flower)
    n = 700
    xs = (np.arange(n) + 0.5) * (2.6 / n) - 1.3
    count = sum(member(x, y) for x in xs for y in xs)
    assert np.isclose(flower.bound()[0], np.pi * 0.81 * (1 + 0.08), rtol=1e-12)
    assert np.isclose(flower.bound()[0], count * (2.6 / n) ** 2, rtol=0.01)
    assert flower.FIELD_TYPES["flower_petals"] is int
    assert m.flower_bound(1.0, 3, 5.0, 0.0)[4] == 0.95 and m.flower_bound(1.0, 999, 0.2, 0.0)[5] == 30
    xs, ys = flower.boundary_xy()
    assert np.hypot(xs, ys).max() <= flower.extent() + 1e-9
    assert np.isclose(np.hypot(xs, ys).max(), flower.extent(), rtol=1e-3)


def test_the_sine_flowers_ring_layers_stay_close_to_its_outline():
    xs = np.linspace(-2.0, 2.0, 500)
    gx, gy = np.meshgrid(xs, xs[::-1])
    for radius, petals, amplitude, rotation in ((1.0, 6, 0.3, 0.0), (0.8, 12, 0.9, 20.0), (1.0, 1, 0.95, 70.0)):
        flower = m.SineFlowerShape(0)
        flower.radius, flower.flower_petals, flower.flower_amplitude = radius, petals, amplitude
        flower.rotation = rotation
        outer, inner = m.ring_layer_masks(m.SHAPE_FLOWER, flower.bound(), gx, gy)
        ox, oy = flower.boundary_xy()
        along = np.linspace(0, 1, len(ox))
        dense = np.linspace(0, 1, 6000)
        curve = np.column_stack((np.interp(dense, along, ox), np.interp(dense, along, oy)))
        ring = np.column_stack((gx[outer | inner], gy[outer | inner]))
        assert len(ring) > 50 and outer.any() and inner.any(), (radius, petals, amplitude)
        far = np.sqrt(((ring[:, None, :] - curve[None, ::3, :]) ** 2).sum(-1)).min(axis=1).max()
        assert far < 0.02, (radius, petals, amplitude, far)


# --------------------------------------------------------------------------- #
# The snail of Pythagoras has a size: its radius
# --------------------------------------------------------------------------- #
def test_the_snail_radius_is_the_distance_to_its_farthest_corner_and_one_is_the_size_it_had():
    o = m.PythagorasSnailShape(0)
    assert o.radius == m.DEFAULT_RADIUS == 2.0 and "radius" in o.FIELD_NAMES and o.FIELD_TYPES["radius"] is float
    assert np.array_equal(m.snail_bound(17, 20.0), m.snail_bound(17, 20.0, 1.0)), "(the function's own default is the old size, 1)"
    for n, radius in ((17, 1.0), (9, 2.5), (60, 0.4)):
        bound = m.snail_bound(n, 33.0, radius)
        corners = bound[3:3 + 2 * (n + 1)].reshape(-1, 2)
        assert abs(np.hypot(corners[:, 0], corners[:, 1]).max() - radius) < 1e-12 * max(1.0, radius), (n, radius)
        assert np.allclose(corners, m.snail_bound(n, 33.0, 1.0)[3:3 + 2 * (n + 1)].reshape(-1, 2) * radius, rtol=1e-12, atol=0)
        assert abs(bound[0] / (m.snail_bound(n, 33.0, 1.0)[0] * radius * radius) - 1.0) < 0.05, "area grows as radius squared"
    o.radius = 1.0
    assert o.extent() == 1.0 and (o.extent(), type(o.extent())) == (1.0, float)
    o.radius = 2.0
    assert o.extent() == 2.0 and "r2.00" in o.info_param_line() and "r2.00" in o.shape_text()
    assert np.array_equal(o.bound(), m.snail_bound(o.snail_segments, o.rotation, 2.0))


def test_a_snail_of_radius_zero_is_nothing_and_a_larger_one_is_the_same_shape_scaled():
    nothing = m.snail_bound(17, 0.0, 0.0)
    assert nothing[0] == 0.0 and not any(m.inside_boundary(x, y, m.SHAPE_SNAIL, nothing) for x in (-0.3, 0.0, 0.2) for y in (-0.1, 0.0, 0.4))
    small, big = m.snail_bound(17, 20.0, 1.0), m.snail_bound(17, 20.0, 2.0)
    rng = np.random.RandomState(4)
    for x, y in rng.uniform(-1.3, 1.3, size=(400, 2)):
        assert m.inside_boundary(x, y, m.SHAPE_SNAIL, small) == m.inside_boundary(2.0 * x, 2.0 * y, m.SHAPE_SNAIL, big)


# --------------------------------------------------------------------------- #
# The default world is twice as large: every default length, and every maximum, doubled
# --------------------------------------------------------------------------- #
def test_every_default_obstacle_length_is_twice_what_it_was():
    """SAMPLE_SIZES holds the defaults of the world [-1, 1] (it is what the reference files in dev/testing say)."""
    for cls in m.SHAPE_CLASSES:
        o = cls(0)
        for field, old in SAMPLE_SIZES[cls.KEY].items():
            assert getattr(o, field) == 2.0 * old, (cls.__name__, field)
    assert m.default_starting_obstacle(0).width == m.WORLD_RECT_DEFAULT.x_stop * 2.0, "it fills the default world"


def test_every_maximum_of_an_obstacle_length_is_twice_what_it_was_and_a_minimum_stayed():
    new_maxima = {"width": 4.0, "height": 4.0, "radius": 4.0, "lens_radius1": 4.0, "lens_radius2": 4.0, "lens_sep": 4.0,
                  "capsule_length": 4.0, "capsule_radius": 2.0, "cut_radius": 4.0, "cut_offset": 4.0, "egg_a": 4.0,
                  "egg_b": 4.0, "moss_size": 4.0, "spiral_pitch": 2.0}
    seen = set()
    for cls in m.SHAPE_CLASSES:
        for p in cls.NUMERIC_PARAMS:
            if p.field in new_maxima:
                assert p.hi == new_maxima[p.field], (cls.__name__, p.field)
                assert p.lo == (0.04 if p.field == "spiral_pitch" else 0.0), (cls.__name__, p.field, "a minimum stays")
                seen.add(p.field)
    assert seen == set(new_maxima), set(new_maxima) - seen
    assert (m.OBST_POS_MIN, m.OBST_POS_MAX, m.OBST_RADIUS_MAX) == (-2.0, 2.0, 4.0)
    shared = {p.field: p for p in m.POSITION_PARAMS if p.lo is not None}
    assert (shared["r"].hi, shared["cx"].lo, shared["cx"].hi, shared["cy"].lo, shared["cy"].hi) == (2.0, -2.0, 2.0, -2.0, 2.0)
    for cls in m.SHAPE_CLASSES:                                 # and every default is inside its range
        o = cls(0)
        for p in cls.NUMERIC_PARAMS:
            if p.lo is not None:
                assert p.lo <= getattr(o, p.field) <= p.hi, (cls.__name__, p.field)
