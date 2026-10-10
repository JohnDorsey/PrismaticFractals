"""Tests: mandelbrot (the escape-time orbit engine: its path function, its measurements, its place in the drivers)."""
import cmath
import math
import random

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers

NO_CONFINE = m.ConfineRule(lookback=999, method=m.CONFINE_BOX, threshold=1e-300)     # (never judged: the orbit runs on)


def _args(obstacles=(), exponent=2.0, z0_is_c=False, julia=False, julia_re=0.0, julia_im=0.0, clip=True):
    orbit = m.orbit_args(exponent, z0_is_c, julia, julia_re, julia_im, clip)
    return m.path_args(m.ENGINE_MANDELBROT, m.settings_ray_args(m.DEFAULT_RAY_SETTINGS), m.build_obstacle_arrays(list(obstacles)),
                       orbit, m.tables_of_engine(m.ENGINE_MANDELBROT))


def _orbit(x, y, max_iter=50, radius=2.0, max_dist=0.0, confine=NO_CONFINE, obstacles=(), **orbit):
    """(termination reason, points, tags, raw) of the orbit of the pixel at (x, y), with the escape radius 2 and no confinement
    unless said otherwise."""
    termination = m.natural_termination_rule(max_iter, m.effective_dist(max_dist), radius, confine)
    points, tags, raw = np.zeros((max_iter + 2, 2)), np.zeros(max_iter + 2), np.zeros(m.N_RAW)
    term, n = m.trace_engine_path(x, y, points, tags, termination, _args(obstacles, **orbit), raw)
    return term, points[:n].copy(), tags[:n].copy(), raw


def _shape(cx, cy, radius, in_effect, out_effect, fire=True):
    o = m.EllipseShape(1)
    o.width = o.height = 2.0 * radius
    o.cx, o.cy = cx, cy
    o.in_effect, o.out_effect, o.fire_inside = in_effect, out_effect, fire
    return o


PASS, STOP, MIRROR = m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_TERMINATE, m.OBSTACLE_EFFECT_REFLECT


def _reference(c, z0, power, radius, max_iter):
    """The orbit as Python complex numbers: (points, ended by escape), the points up to the first beyond the radius (unclipped)."""
    z, points = z0, [z0]
    if z.real * z.real + z.imag * z.imag - radius * radius >= 0.0:
        return points, True
    for _ in range(max_iter):
        z = (z * z if power == 2 else z ** power) + c
        points.append(z)
        if z.real * z.real + z.imag * z.imag - radius * radius >= 0.0:
            return points, True
    return points, False


# --------------------------------------------------------------------------- #
# Known orbits of z -> z^2 + c with the escape radius 2 (D26)
# --------------------------------------------------------------------------- #
def test_c_equal_zero_never_escapes_and_stays_at_the_origin():
    term, points, tags, raw = _orbit(0.0, 0.0)
    assert term == m.TERM_ITER and len(points) == 51 and (points == 0.0).all()
    assert tags[0] == m.TAG_START and (tags[1:] == m.TAG_ORBIT).all() and raw[m.RAW_TERM] == m.TERM_ITER


def test_c_equal_one_escapes_at_the_third_point_where_the_orbit_reaches_the_radius():
    term, points, tags, _raw = _orbit(1.0, 0.0)
    assert term == m.TERM_ESCAPED and points.tolist() == [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    assert tags.tolist() == [m.TAG_START, m.TAG_ORBIT, m.TAG_ESCAPED]


def test_the_escape_test_is_at_least_the_radius_so_c_equal_two_and_minus_two_escape_at_the_second_point():
    for c in (2.0, -2.0):
        term, points, tags, _raw = _orbit(c, 0.0)
        assert term == m.TERM_ESCAPED and points.tolist() == [[0.0, 0.0], [c, 0.0]] and tags[-1] == m.TAG_ESCAPED


def test_c_equal_minus_one_cycles_without_escaping_exactly():
    term, points, _tags, _raw = _orbit(-1.0, 0.0, max_iter=40)
    assert term == m.TERM_ITER and len(points) == 41
    assert points[:, 0].tolist() == [0.0 if k % 2 == 0 else -1.0 for k in range(41)] and (points[:, 1] == 0.0).all()


def test_the_orbit_of_a_random_c_is_the_python_complex_orbit_and_ends_where_it_says():
    rng = random.Random(3)
    for _ in range(400):
        c = complex(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0))
        term, points, tags, _raw = _orbit(c.real, c.imag, max_iter=60, clip=False)
        want, escaped = _reference(c, 0j, 2, 2.0, 60)
        assert len(points) == len(want) and term == (m.TERM_ESCAPED if escaped else m.TERM_ITER), c
        assert [complex(*p) for p in points] == want, c
        assert (tags[-1] == m.TAG_ESCAPED) == escaped


def test_a_clipped_escape_ends_on_the_circle_between_the_last_two_points_and_an_unclipped_one_beyond_it():
    clipped = _orbit(1.5, 0.0)
    assert clipped[0] == m.TERM_ESCAPED and clipped[1].tolist() == [[0.0, 0.0], [1.5, 0.0], [2.0, 0.0]] and clipped[2][-1] == m.TAG_ESCAPED
    plain = _orbit(1.5, 0.0, clip=False)
    assert plain[0] == m.TERM_ESCAPED and plain[1].tolist() == [[0.0, 0.0], [1.5, 0.0], [3.75, 0.0]] and plain[2][-1] == m.TAG_ESCAPED
    rng = random.Random(5)
    for _ in range(200):
        c = complex(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0))
        term, points, _tags, _raw = _orbit(c.real, c.imag, max_iter=40)
        if term == m.TERM_ESCAPED:
            end, before = complex(*points[-1]), complex(*points[-2])
            assert abs(abs(end) - 2.0) < 1e-12, "(on the circle)"
            beyond, _ = _reference(c, 0j, 2, 2.0, 40)
            seg = beyond[-1] - before
            assert abs(((end - before) / seg).imag) < 1e-9 and 0.0 <= ((end - before) / seg).real <= 1.0 + 1e-12, "(on the segment)"


def test_a_confined_orbit_ends_by_the_shared_convergence_test_before_the_iteration_limit():
    termination_default = m.CONFINE_DEFAULT
    term, points, _tags, _raw = _orbit(-0.5, 0.0, max_iter=200, confine=termination_default)
    assert term == m.TERM_CONFINED and 4 < len(points) < 100
    assert _orbit(-0.5, 0.0, max_iter=200)[0] == m.TERM_ITER, "(without the rule it runs to the limit)"


def test_a_starting_point_already_beyond_the_radius_is_a_path_of_that_point_alone():
    term, points, tags, _raw = _orbit(2.5, 0.0, julia=True, julia_re=-1.0)
    assert term == m.TERM_ESCAPED and points.tolist() == [[2.5, 0.0]] and tags.tolist() == [m.TAG_START]


# --------------------------------------------------------------------------- #
# Julia sets, z_0, exponents
# --------------------------------------------------------------------------- #
def test_the_julia_form_starts_at_the_pixel_and_adds_the_constant():
    c = complex(-0.4, 0.6)
    rng = random.Random(8)
    for _ in range(200):
        z0 = complex(rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
        term, points, _tags, _raw = _orbit(z0.real, z0.imag, max_iter=50, clip=False, julia=True, julia_re=c.real, julia_im=c.imag)
        want, escaped = _reference(c, z0, 2, 2.0, 50)
        assert [complex(*p) for p in points] == want and (term == m.TERM_ESCAPED) == escaped, z0
    assert _orbit(0.5, 0.0, max_iter=6, julia=True, julia_re=0.0, julia_im=0.0)[1][:, 0].tolist() == [0.5, 0.25, 0.0625, 0.00390625, 1.52587890625e-05, 2.3283064365386963e-10, 5.421010862427522e-20]


def test_z0_may_be_c_instead_of_the_origin():
    term, points, _tags, _raw = _orbit(0.3, 0.2, max_iter=3, z0_is_c=True, clip=False)
    c = complex(0.3, 0.2)
    assert [complex(*p) for p in points] == [c, c * c + c, (c * c + c) ** 2 + c, ((c * c + c) ** 2 + c) ** 2 + c]


def test_a_whole_exponent_is_plain_multiplication_and_a_real_one_the_principal_value():
    rng = random.Random(21)
    for power in (3.0, 4.0, -1.0, -2.0, 1.0, 0.0, 2.5, 0.5, -0.5, 1.7):
        for _ in range(60):
            c = complex(rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2))
            term, points, _tags, _raw = _orbit(c.real, c.imag, max_iter=4, clip=False, z0_is_c=True, exponent=power)
            z, want = c, [c]
            for _step in range(len(points) - 1):
                z = (z ** power if z != 0 or power > 0 else complex("inf")) + c
                want.append(z)
            for k, p in enumerate(points):
                assert abs(complex(*p) - want[k]) < 1e-9 * max(1.0, abs(want[k])), (power, c, k)
    assert m.complex_power(1.0, 2.0, 2.0) == (-3.0, 4.0), "z^2 is exactly z*z"
    assert m.complex_power(1.0, 2.0, 0.0) == (1.0, 0.0), "z^0 is 1"
    assert m.complex_power(0.0, 2.0, -1.0) == (0.0, -0.5)
    assert m.complex_power(0.0, 0.0, 0.0) == (1.0, 0.0), "(so is 0^0)"
    assert m.complex_power(0.0, 0.0, 2.5) == (0.0, 0.0) and not math.isfinite(m.complex_power(0.0, 0.0, -2.5)[0])


def test_a_point_that_is_not_finite_ends_the_path_as_an_escape_at_its_last_finite_point():
    zero_power = _orbit(0.0, 0.0, exponent=-1.0)           # 0^-1 is infinite
    assert zero_power[0] == m.TERM_ESCAPED and zero_power[1].tolist() == [[0.0, 0.0]] and zero_power[2].tolist() == [m.TAG_START]
    term, points, tags, _raw = _orbit(1e200, 0.0, radius=1e300, max_iter=10)     # z^2 overflows at the second step
    assert term == m.TERM_ESCAPED and points.tolist() == [[0.0, 0.0], [1e200, 0.0]] and tags.tolist() == [m.TAG_START, m.TAG_ESCAPED]


# --------------------------------------------------------------------------- #
# The distance limit
# --------------------------------------------------------------------------- #
def test_the_distance_limit_cuts_a_segment_where_the_path_reaches_it():
    term, points, tags, raw = _orbit(1.0, 0.0, max_dist=1.5)
    assert term == m.TERM_DIST and points.tolist() == [[0.0, 0.0], [1.0, 0.0], [1.5, 0.0]] and tags[-1] == m.TAG_DISTCUT
    assert raw[m.RAW_LAST_FULL] == 1.0, "(the segment as long as it would have been)"
    term, points, tags, _raw = _orbit(1.5, 0.0, max_dist=1.8)          # the second segment is clipped at the radius first: 0.5 long
    assert term == m.TERM_DIST and abs(points[-1, 0] - 1.8) < 1e-12 and tags[-1] == m.TAG_DISTCUT
    assert _orbit(1.5, 0.0, max_dist=2.0)[0] == m.TERM_ESCAPED, "(reached exactly at the end: not cut)"


# --------------------------------------------------------------------------- #
# Obstacles: every segment is tested, not only the points (D12)
# --------------------------------------------------------------------------- #
def test_a_terminating_shape_ends_the_orbit_where_a_segment_crosses_it_though_no_point_is_inside():
    term, points, tags, _raw = _orbit(1.5, 0.0, obstacles=[_shape(0.75, 0.0, 0.25, STOP, PASS)])
    assert term == m.TERM_OBSTACLE and points.tolist() == [[0.0, 0.0], [0.5, 0.0]] and tags[-1] == m.TAG_OBSTACLE_TERM
    assert _orbit(1.5, 0.0, obstacles=[_shape(0.75, 0.0, 0.25, PASS, PASS)])[0] == m.TERM_ESCAPED, "(passing both ways: not there)"
    term, points, _tags, _raw = _orbit(1.5, 0.0, obstacles=[_shape(0.75, 0.0, 0.25, PASS, STOP)])
    assert term == m.TERM_OBSTACLE and abs(points[-1, 0] - 1.0) < 1e-12, "(passed in at 0.5, stopped on the way out at 1.0)"


def test_which_shapes_the_orbit_is_inside_is_remembered_from_one_segment_to_the_next():
    shape = _shape(1.4, 0.0, 0.3, PASS, STOP)                       # spans 1.1 to 1.7: the first segment (0 to 1.5) ends inside it
    assert _orbit(1.5, 0.0, obstacles=[_shape(1.4, 0.0, 0.3, PASS, PASS)])[0] == m.TERM_ESCAPED
    term, points, _tags, _raw = _orbit(1.5, 0.0, obstacles=[shape])
    assert term == m.TERM_OBSTACLE and points[:2].tolist() == [[0.0, 0.0], [1.5, 0.0]] and len(points) == 3
    assert abs(points[-1, 0] - 1.7) < 1e-12


def test_a_shape_passed_into_flips_the_orbit_to_inside_it_so_the_outgoing_effect_applies_later_on_the_same_segment():
    term, points, _tags, _raw = _orbit(1.5, 0.0, obstacles=[_shape(0.5, 0.0, 0.2, PASS, STOP)])
    assert term == m.TERM_OBSTACLE and abs(points[-1, 0] - 0.7) < 1e-12 and len(points) == 2


def test_a_reflecting_shape_terminates_an_orbit_which_cannot_reflect_the_last_resort():
    term, points, _tags, _raw = _orbit(1.5, 0.0, obstacles=[_shape(0.75, 0.0, 0.25, MIRROR, PASS)])
    assert term == m.TERM_OBSTACLE and points.tolist() == [[0.0, 0.0], [0.5, 0.0]]


def test_an_orbit_that_starts_inside_a_shape_leaves_it_through_the_outgoing_effect():
    term, points, _tags, _raw = _orbit(1.5, 0.0, obstacles=[_shape(0.0, 0.0, 0.4, PASS, STOP)])
    assert term == m.TERM_OBSTACLE and abs(points[-1, 0] - 0.4) < 1e-12


def test_a_terminating_shape_beyond_the_escape_clip_is_never_reached_and_one_before_it_wins():
    behind = _orbit(1.5, 0.0, obstacles=[_shape(2.6, 0.0, 0.2, STOP, STOP)])
    assert behind[0] == m.TERM_ESCAPED, "(the orbit ends at the radius, 2, before the shape at 2.4)"
    term, points, _tags, _raw = _orbit(1.5, 0.0, clip=False, obstacles=[_shape(3.0, 0.0, 0.5, STOP, STOP)])
    assert term == m.TERM_OBSTACLE and abs(points[-1, 0] - 2.5) < 1e-12, "(unclipped, the segment goes on to 3.75)"


def test_the_shapes_an_orbit_is_inside_are_decided_at_its_first_point_and_whether_it_begins_at_the_pixel():
    """D30: with z_0 = 0 the path begins at the origin, but only if the pixel (c) fires; flags come from the origin."""
    shape = _shape(1.2, 0.0, 0.5, STOP, PASS)                       # c = 1 is inside it, the origin is not
    n = 8
    state = np.zeros((n, n), np.uint8)
    layout = m.layout_of_engine(m.ENGINE_MANDELBROT)
    stats, points = layout.allocate(n, n)
    termination = m.natural_termination_rule(20, m.effective_dist(0.0), 2.0, NO_CONFINE)
    m.trace_rows(stats, points, state, 0, n, n, n, m.WorldRect(-1.0, 3.0, -2.0, 2.0), termination, _args([shape]), 0)
    cols = [i for i in range(n) if (state[:, i] == m.PIXEL_TRACED).any()]
    assert cols and set(np.unique(state)) == {m.PIXEL_TRACED, m.PIXEL_NO_RAY}, "only pixels inside the shape start a path"
    xs = -1.0 + (np.array(cols) + 0.5) * 0.5
    assert ((xs > 0.7) & (xs < 1.7)).all()
    traced = state == m.PIXEL_TRACED
    assert (points[..., layout.point_position(m.POINT_START), :][traced] == 0.0).all(), "(the paths begin at the origin)"
    term, path, tags, _raw = _orbit(1.0, 0.0, obstacles=[shape])
    assert term == m.TERM_OBSTACLE and abs(path[-1, 0] - 0.7) < 1e-12, \
        "(the origin is outside: the first segment enters the shape and is stopped; judged from c it would have been inside)"


# --------------------------------------------------------------------------- #
# What is stored, and the engine in the drivers
# --------------------------------------------------------------------------- #
def test_the_orbit_engine_stores_the_metrics_and_points_of_its_path_where_its_tables_say():
    tables = m.tables_of_engine(m.ENGINE_MANDELBROT)
    term, points, tags, raw = _orbit(0.4, 0.3, max_iter=30)
    out, rcp = np.full(tables.n_stats, np.nan), np.full((tables.n_points, 2), np.nan)
    m.measure_path(m.ENGINE_MANDELBROT, np.vstack([points, np.zeros((32 - len(points), 2))]), len(points), np.append(tags, np.zeros(32 - len(tags))),
                   raw, 30, 1.0, tables, out, rcp)
    assert not np.isnan(out).any() and not np.isnan(rcp).any(), "everything it stores is written"
    stat = lambda slot: out[tables.stat_pos[slot]]
    point = lambda slot: rcp[tables.point_pos[slot]]
    n = len(points)
    length = float(np.sum(np.hypot(*np.diff(points, axis=0).T)))
    assert stat(m.stat_slot(m.SEQ_PATH, m.SRED_COUNT)) == min(n / 30, 1.0)
    assert abs(stat(m.ST_LENGTH) - math.atan(length / m.LENGTH_METRIC_DIVISOR) * 2.0 / math.pi) < 1e-12
    assert stat(m.ST_TERMINATION) == m.termination_stat_value(term)
    d = points[-1] - points[0]
    assert abs(stat(m.ST_HDIST) - min(abs(d[0]) / m.DISTANCE_METRIC_DIVISOR, 1.0)) < 1e-12
    assert abs(stat(m.ST_EUCLIDEAN) - min(math.hypot(*d) / m.DISTANCE_METRIC_DIVISOR, 1.0)) < 1e-12
    assert point(m.point_slot(m.SEQ_PATH, m.PRED_FIRST)).tolist() == points[0].tolist()
    assert point(m.point_slot(m.SEQ_PATH, m.PRED_LAST)).tolist() == points[-1].tolist()
    assert np.allclose(point(m.point_slot(m.SEQ_PATH, m.PRED_MEAN)), points.mean(axis=0))
    assert raw[m.RAW_BOUNCES] == 0 and tables.stat_pos[m.ST_LOOP_AREA] == -1, "(no bounces, no loop area: not stored)"
    members = [[m.in_sequence(seq, k, tags) for k in range(n)] for seq in (m.SEQ_PATH, m.SEQ_BOUNCES, m.SEQ_BOUNDARY, m.SEQ_SELF)]
    assert all(members[0]) and not any(members[1]) and not any(members[2]) and not any(members[3]), "(orbit points are in the Path only)"


def test_the_termination_values_are_the_same_numbers_for_both_engines():
    for term in range(m.N_TERM):
        assert m.termination_stat_value(term) == m.TERM_STAT_VALUE[term]


def test_the_area_of_the_mandelbrot_set_is_estimated_by_the_drivers_as_about_1_5066():
    """The pixels of a 500 x 500 image of [-2, 2]^2 whose orbit does not escape within 200 iterations, times the area of a pixel (the
    exact area is 1.50659; a finite limit overestimates, a grid of pixel centres is within about a percent)."""
    start = m.default_starting_obstacle(0)
    start.in_effect = start.out_effect = PASS                         # (it still fires a path from every pixel inside it)
    n = 500
    layout = m.layout_of_engine(m.ENGINE_MANDELBROT)
    stats, points = layout.allocate(n, n)
    state = np.zeros((n, n), np.uint8)
    termination = m.natural_termination_rule(200, m.effective_dist(0.0), 2.0, NO_CONFINE)
    m.trace_rows(stats, points, state, 0, n, n, n, m.WORLD_RECT_DEFAULT, termination, _args([start]), 0)
    traced = state == m.PIXEL_TRACED
    assert traced.sum() > 0.7 * n * n, "(the starting shape is a disc of radius 2: the corners fire nothing)"
    inside = (stats[..., layout.stat_position(m.ST_TERMINATION)] < 1.0)[traced].sum()
    area = inside * (4.0 / n) ** 2
    assert 1.49 < area < 1.58, area


def test_the_app_renders_an_image_with_the_mandelbrot_engine_and_its_settings():
    app = _fake_render_app()
    base = dict(m.DEFAULT_SETTING_VALUES)
    start = m.default_starting_obstacle(0)
    start.in_effect = start.out_effect = PASS
    arrays = m.build_obstacle_arrays([start])
    config = m.ColorationConfig(0)
    config.metric_types[:] = [m.METRIC_TYPE_POINT_PAIR] * 3         # (the default third channel counts bounces: not stored by this engine)
    coloration = _coloration_settings_from_config(config)
    ray_data = m.resolve_frame_settings([], 0, base)
    images = {}
    for name, changes in (("ray", {}), ("mandelbrot", {"engine": m.ENGINE_MANDELBROT}),
                          ("julia", {"engine": m.ENGINE_MANDELBROT, "orbit_mapping": 1, "orbit_julia_re": -0.8, "orbit_julia_im": 0.156})):
        data = m.resolve_frame_settings([], 0, {**base, "outline": False, **changes})
        images[name] = app.render_full_image(data, 96, 96, arrays, (coloration,), m.DEFAULT_FRAMING)
    background = np.array(m.BACKGROUND, dtype=np.uint8)
    assert all((image != background).any() for image in images.values())
    assert not np.array_equal(images["ray"], images["mandelbrot"]) and not np.array_equal(images["mandelbrot"], images["julia"])
    assert ray_data.engine == m.DEFAULT_ENGINE_SETTINGS


# --------------------------------------------------------------------------- #
# What the engine offers, shown by the interface; the hover preview of an orbit
# --------------------------------------------------------------------------- #
class _Recorder:
    """A widget that remembers the last `values` it was configured with."""
    values = None

    def configure(self, **options):
        self.values = options.get("values", self.values)

    def __getattr__(self, _name):
        return lambda *a, **k: None


def test_the_lists_of_choices_hold_only_what_the_engine_offers_in_the_order_of_the_dropdowns():
    ray, orbit = m.RAY_BOUNCING_ENGINE, m.MANDELBROT_ENGINE
    assert m.point_metric_labels(ray) == m.POINT_METRIC_LABELS and m.other_metric_labels(ray) == m.OTHER_LABELS
    assert m.point_menu_labels(ray) == m.POINT_MENU_LABELS
    assert m.point_metric_labels(orbit) == m.POINT_METRIC_LABELS, "(all four distances)"
    assert m.other_metric_labels(orbit) == ["Termination reason", "Path: Point count / limit", "Path: Length (arctan)", "Path: Last segment direction"]
    assert m.point_menu_labels(orbit)[:3] == ["Origin", "Top left point", "(-1,1)"] and len(m.point_menu_labels(orbit)) == 3 + 7
    assert not any(label.startswith(("Self", "Boundary", "All bounces")) for label in m.point_menu_labels(orbit))
    assert m.effect_names(ray) == m.OBSTACLE_EFFECT_NAMES and m.effect_names(orbit) == ["Terminate", "Pass through"]


def test_the_colorations_editor_lists_only_the_selected_engines_choices_and_greys_the_options_it_lacks():
    app = _fake_color_editor_app()
    app.current_engine_description = lambda: m.MANDELBROT_ENGINE
    app.map_combos = [_Recorder() for _ in range(3)]
    app.point1_combos = [_Recorder() for _ in range(3)]
    app.point2_combos = [_Recorder() for _ in range(3)]
    coloration = app.selected_coloration_obj()
    coloration.metric_types[:] = [m.METRIC_TYPE_POINT_PAIR, m.METRIC_TYPE_OTHER, m.METRIC_TYPE_OTHER]
    for c in range(2):
        app.refresh_row(c)
    assert app.map_combos[0].values == m.POINT_METRIC_LABELS and app.map_combos[1].values == m.other_metric_labels(m.MANDELBROT_ENGINE)
    assert app.point1_combos[0].values == m.point_menu_labels(m.MANDELBROT_ENGINE)
    checks = [[_StateRecorder() for _ in range(m.N_BUDDHA_POINTS)] for _ in range(3)]
    app.buddha_point_checks = checks
    app.mode_var = object()
    app.buddha_kind_vars = [[_StubVar(True) for _ in range(3)] for _ in range(3)]
    m.App.refresh_color_controls_for_selected_coloration(app)
    offered = m.MANDELBROT_ENGINE.buddha_options
    assert all(checks[c][i].last == (["!disabled"] if i in offered else ["disabled"]) for c in range(3) for i in range(m.N_BUDDHA_POINTS))
    assert checks[0][m.POINT_ALL_MEAN].last == ["disabled"] and checks[0][m.point_slot(m.SEQ_PATH, m.PRED_MEAN)].last == ["!disabled"]


class _StateRecorder:
    last = None

    def state(self, flags):
        self.last = flags


def test_a_new_obstacle_gets_the_effects_the_selected_engine_uses_pass_through_for_mandelbrot():
    class FakeApp:
        pass
    for description, want in ((m.RAY_BOUNCING_ENGINE, (m.OBSTACLE_EFFECT_REFLECT,) * 2), (m.MANDELBROT_ENGINE, (PASS, PASS))):
        app = FakeApp()
        app.obstacles, app._obst_counter, app.current_engine_description = [], 1, lambda d=description: d
        app.rebuild_obstacle_list = app.refresh_obstacle_editor = app.on_shape_param_changed = lambda: None
        m.App.new_obstacle(app, m.EllipseShape)
        assert (app.obstacles[0].in_effect, app.obstacles[0].out_effect) == want


def test_every_tag_a_path_point_can_have_has_a_colour_for_the_graphs():
    tags = {value for name, value in vars(m).items() if name.startswith("TAG_") and isinstance(value, float) and name != "TAG_COLOR"}
    assert tags == set(m.TAG_COLOR) and m.TAG_ORBIT in tags


def test_a_buddhabrot_of_the_escaping_orbits_is_symmetric_about_the_real_axis_and_unlike_that_of_the_others():
    """The Buddhabrot (every point of every orbit, scattered into a histogram) from the app's own render path with the Mandelbrot
    engine: the orbit of the conjugate of c is the conjugate orbit, so the image is the same upside down; the escaping orbits and
    the non-escaping ones make different images."""
    app = _fake_render_app()
    start = m.default_starting_obstacle(0)
    start.in_effect = start.out_effect = PASS
    arrays = m.build_obstacle_arrays([start])
    data = m.resolve_frame_settings([], 0, {**m.DEFAULT_SETTING_VALUES, "outline": False, "engine": m.ENGINE_MANDELBROT, "iter": 60,
                                            "escape_radius": 2.0})
    all_path = m.BUDDHA_POINT_LABELS.index("All path points")
    images = {}
    for name, kinds in (("escaping", (True, False, False)), ("others", (False, True, True))):
        config = m.ColorationConfig(0)
        config.chan_src[:] = [m.CHAN_SRC_BUDDHA, m.CHAN_SRC_CONSTANT, m.CHAN_SRC_CONSTANT]
        config.buddha_points[0] = tuple(i == all_path for i in range(m.N_BUDDHA_POINTS))
        config.buddha_path_kinds[0] = kinds
        images[name] = app.render_full_image(data, 128, 128, arrays, (_coloration_settings_from_config(config),), m.DEFAULT_FRAMING)
    red = {name: image[..., 0].astype(float) for name, image in images.items()}
    assert red["escaping"].max() > 0 and red["others"].max() > 0
    for image in red.values():
        assert np.abs(image - image[::-1]).mean() < 0.02 * image.max(), "(the same upside down)"
    assert np.abs(red["escaping"] - red["others"]).mean() > 0.05 * red["escaping"].max()
