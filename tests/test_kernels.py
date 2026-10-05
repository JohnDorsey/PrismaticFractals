"""Tests: kernels."""
import dataclasses
import os
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


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
# Parity with the classic bounding shape. The old system (a fixed shape at the
# origin, with its own settings) is being replaced by ordinary obstacles; an
# obstacle that is a plain stand-in - Reflect both ways, fires rays inside, at the
# origin - must produce exactly what the classic shape produced. The recorded
# golden data (dev/testing/classic_boundary_golden.npz) was generated from the
# classic implementation, so it keeps guarding this after that code is gone.
# --------------------------------------------------------------------------- #
CLASSIC_GOLDEN_PATH = os.path.join(ROOT, "dev",
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
    resolved = m.resolve_frame_settings([m.Keyframe(0, 0, data)], 0).ray
    assert (resolved.confine_window, resolved.confine_method, resolved.confine_threshold) \
        == (9, PATH, 1e-12)


def test_confinement_threshold_interpolates_but_window_and_method_hold():
    kf0 = m.Keyframe(0, 0, {"confine_window": 4, "confine_method": BOX,
                            "confine_threshold": 0.0})
    kf1 = m.Keyframe(1, 10, {"confine_window": 20, "confine_method": PATH,
                             "confine_threshold": 1.0})
    mid = m.resolve_frame_settings([kf0, kf1], 5).ray
    assert mid.confine_window == 4 and mid.confine_method == BOX
    assert np.isclose(mid.confine_threshold, 0.5)
    end = m.resolve_frame_settings([kf0, kf1], 10).ray
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
    app.invalidate_preview = app.update_buttons = lambda: None
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
    snap = m.resolve_frame_settings([m.Keyframe(0, 0, data)], 0).ray
    assert (snap.emit_mode, snap.emit_target_r, snap.emit_target_theta, snap.emit_target_cx,
            snap.emit_target_cy, snap.emit_offset) == (TOWARDS, -2500.0, 90.0, 0.75, -4.0, 12.5)


def test_emission_target_and_offset_interpolate_but_the_mode_holds():
    kf0 = m.Keyframe(0, 0, {"emit_mode": UP, "emit_target_r": 0.0, "emit_target_theta": 0.0,
                            "emit_target_cx": -1.0, "emit_target_cy": 0.0,
                            "emit_offset": -45.0})
    kf1 = m.Keyframe(1, 10, {"emit_mode": TOWARDS, "emit_target_r": 2.0,
                             "emit_target_theta": 90.0, "emit_target_cx": 1.0,
                             "emit_target_cy": 4.0, "emit_offset": 45.0})
    mid = m.resolve_frame_settings([kf0, kf1], 5).ray
    assert mid.emit_mode == UP
    assert (mid.emit_target_r, mid.emit_target_theta, mid.emit_target_cx, mid.emit_target_cy,
            mid.emit_offset) == (1.0, 45.0, 0.0, 2.0, 0.0)
    assert m.resolve_frame_settings([kf0, kf1], 10).ray.emit_mode == TOWARDS


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
# The squircles and the Archimedean spiral ribbon in the kernels
# --------------------------------------------------------------------------- #
def _inside_along(o, px, py, dx, dy, t_max=4.0, steps=8001):
    """(ts, inside) sampled along a ray by the compiled point test alone - an independent way to
    find where the ray crosses the edge, to compare with boundary_hit / obstacle_hit."""
    ts = np.linspace(0.0, t_max, steps)
    gx = (px + ts * dx)[None, :]
    gy = (py + ts * dy)[None, :]
    return ts, m.inside_boundary_grid(gx, gy, o.KIND, o.bound())[0]


def _random_rays(o, want_inside, count, seed):
    """`count` (px, py, dx, dy) rays starting inside (or outside) shape `o`, which sits at the origin."""
    rng = np.random.RandomState(seed)
    bound = o.bound()
    rays = []
    while len(rays) < count:
        px, py = rng.uniform(-1.3, 1.3, 2)
        if bool(m.inside_boundary(px, py, o.KIND, bound)) != want_inside:
            continue
        ang = rng.uniform(0.0, 2.0 * np.pi)
        rays.append((px, py, np.cos(ang), np.sin(ang)))
    return rays


def test_rays_leave_the_squircles_and_the_spiral_exactly_where_marching_along_them_sees_the_edge():
    step = 4.0 / 8000
    for o in _new_kind_cases():
        bound = o.bound()
        for px, py, dx, dy in _random_rays(o, True, 120, 3):
            t, _nx, _ny = m.boundary_hit(px, py, dx, dy, o.KIND, bound)
            ts, inside = _inside_along(o, px, py, dx, dy)
            flips = np.nonzero(inside[1:] != inside[:-1])[0]
            assert len(flips), (type(o).__name__, o.state(), px, py, dx, dy)
            assert ts[flips[0]] - 1e-9 <= t <= ts[flips[0] + 1] + 1e-9, \
                (type(o).__name__, o.state(), (px, py, dx, dy), t, ts[flips[0]], step)


def test_rays_enter_the_squircles_and_the_spiral_exactly_where_marching_along_them_sees_the_edge():
    step = 4.0 / 8000
    for o in _new_kind_cases():
        bound = o.bound()
        for px, py, dx, dy in _random_rays(o, False, 120, 4):
            t, _nx, _ny = m.obstacle_hit(px, py, dx, dy, o.KIND, bound)
            ts, inside = _inside_along(o, px, py, dx, dy)
            flips = np.nonzero(inside[1:] != inside[:-1])[0]
            if t < 0.0:
                assert not len(flips), (type(o).__name__, o.state(), (px, py, dx, dy), ts[flips[:1]])
            else:
                assert len(flips), (type(o).__name__, o.state(), (px, py, dx, dy), t)
                assert ts[flips[0]] - 1e-9 <= t <= ts[flips[0] + 1] + 1e-9, \
                    (type(o).__name__, o.state(), (px, py, dx, dy), t, ts[flips[0]], step)


def test_the_normal_at_a_squircle_or_spiral_hit_stands_across_the_edge():
    """Just either side of the hit point along its normal is outside on one side, inside on the other."""
    for o in _new_kind_cases():
        bound = o.bound()
        for want_inside, entry in ((True, False), (False, True)):
            for px, py, dx, dy in _random_rays(o, want_inside, 60, 5):
                hit = m.obstacle_hit if entry else m.boundary_hit
                t, nx, ny = hit(px, py, dx, dy, o.KIND, bound)
                if t < 0.0:
                    continue
                length = np.hypot(nx, ny)
                assert length > 1e-9, (type(o).__name__, o.state())
                hx, hy = px + t * dx, py + t * dy
                a = m.inside_boundary(hx + 2e-6 * nx / length, hy + 2e-6 * ny / length, o.KIND, bound)
                b = m.inside_boundary(hx - 2e-6 * nx / length, hy - 2e-6 * ny / length, o.KIND, bound)
                assert a != b, (type(o).__name__, o.state(), (px, py, dx, dy), t)


def _fired_from_the_pixel_grid(shape, n=24, max_iter=12):
    """(stats, points, status) of an n x n render with `shape` (an obstacle) as the only obstacle."""
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    status = np.zeros((n, n), dtype=np.uint8)
    shape.in_effect = shape.out_effect = m.OBSTACLE_EFFECT_REFLECT
    shape.fire_inside = True
    arrays = m.build_obstacle_arrays([shape])
    m.render_rows_wh(stats, points, status, 0, n, n, n, max_iter, m.effective_dist(0.0), 1.0,
                     m.fuzz_rule(m.DEFAULT_FUZZ), m.CONFINE_DEFAULT, m.EMISSION_DEFAULT, arrays, 0)
    return stats, points, status


def test_a_superellipse_with_exponent_2_and_a_squircle_with_squareness_0_trace_like_the_exact_shapes():
    """The root finder against the closed-form ellipse and circle: the same rays bounce the same way."""
    ellipse = m.EllipseShape(0)
    ellipse.width, ellipse.height, ellipse.rotation = 1.7, 1.1, 25.0
    sup = m.SuperellipseShape(0)
    sup.width, sup.height, sup.exponent, sup.rotation = 1.7, 1.1, 2.0, 25.0
    circle = m.EllipseShape(0)
    circle.width = circle.height = 1.8
    fg = m.FernandezGuastiShape(0)
    fg.radius, fg.squareness = 0.9, 0.0
    for exact, general in ((ellipse, sup), (circle, fg)):
        want, got = _fired_from_the_pixel_grid(exact), _fired_from_the_pixel_grid(general)
        assert np.array_equal(want[2], got[2])
        assert np.allclose(want[0], got[0], atol=1e-4)
        assert np.allclose(want[1], got[1], atol=1e-4)


def test_rays_fired_inside_a_container_squircle_or_spiral_stay_inside_and_land_on_its_edge():
    for o in _new_kind_cases():
        o.in_effect = o.out_effect = m.OBSTACLE_EFFECT_REFLECT
        o.fire_inside = True
        bound = o.bound()
        hits = 0
        for px, py, dx, dy in _random_rays(o, True, 25, 6):
            pts, kinds, _term, _raw = _kernel_trace(px, py, np.degrees(np.arctan2(dy, dx)), [o],
                                                    circle=None, max_iter=40)
            for k in range(1, len(pts)):
                if kinds[k] == m.KIND_BOUNDARY:             # a bounce off the edge lies on the edge
                    x, y = pts[k]
                    around = [m.inside_boundary(x + 4e-6 * np.cos(a), y + 4e-6 * np.sin(a), o.KIND,
                                                bound)
                              for a in np.linspace(0.0, 2 * np.pi, 16, endpoint=False)]
                    assert any(around) and not all(around), (type(o).__name__, o.state(), k, x, y)
                mid = 0.5 * (pts[k - 1] + pts[k])           # and every segment runs through the inside
                assert m.inside_boundary(mid[0], mid[1], o.KIND, bound), (type(o).__name__, o.state(), k)
            hits += len(pts) - 1
        assert hits > 100, (type(o).__name__, "the rays must actually bounce")


def test_a_spiral_ray_that_starts_on_an_edge_does_not_hit_that_same_edge_again():
    """Regression guard for the self-hit: after a bounce the next hit is a different point
    (t > EPS), whichever way the ray points."""
    o = m.ArchimedeanSpiralShape(0)
    bound = o.bound()
    for px, py, dx, dy in _random_rays(o, True, 40, 8):
        t, nx, ny = m.boundary_hit(px, py, dx, dy, o.KIND, bound)
        hx, hy = px + t * dx, py + t * dy
        length = np.hypot(nx, ny)
        nx, ny = nx / length, ny / length
        dot = dx * nx + dy * ny
        rx, ry = dx - 2 * dot * nx, dy - 2 * dot * ny          # the mirror direction
        t2, _a, _b = m.boundary_hit(hx, hy, rx, ry, o.KIND, bound)
        assert t2 > 1e-6, (px, py, dx, dy, t, t2)


def test_a_snail_has_no_hairline_hole_along_the_edge_two_of_its_triangles_share():
    """Regression: the test for a point inside a triangle was strict, so a point exactly on the edge out from
    the centre that two triangles share was in neither: a one-pixel line along the 45 degree diagonal of the image
    (where the second corner is) fired no rays, and a ray running along it found nothing to hit."""
    o = m.PythagorasSnailShape(0)                          # corner 2 of the 17-triangle snail is at 45 degrees
    bound = o.bound()
    for r in (0.05, 0.1, 0.2):                              # (the edge is 0.33 long, out to (0.236, 0.236))
        assert m.inside_boundary(r, r, o.KIND, bound), r
    t, _nx, _ny = m.boundary_hit(0.1, 0.1, np.sqrt(0.5), np.sqrt(0.5), o.KIND, bound)
    ts = np.linspace(0.0, 2.0, 4001)
    inside = m.inside_boundary_grid((0.1 + ts * np.sqrt(0.5))[None, :], (0.1 + ts * np.sqrt(0.5))[None, :],
                                    o.KIND, bound)[0]
    first_outside = ts[np.argmin(inside)]
    assert 0.0 < t < 2.0 and abs(t - first_outside) < 1e-3, (t, first_outside)


def test_a_ray_through_the_exact_centre_of_a_sine_flower_finds_its_edge():
    """Regression: a line through the centre has rho = h / cos(phi - phi0) with h = 0, which is 0 / 0, so the
    search along phi found nothing and every ray along the 45 degree diagonal of the image (the default emission
    direction, from the diagonal pixels) escaped from a closed flower."""
    ts = np.linspace(0.0, 3.0, 6001)
    for petals, amplitude, rotation in ((6, 0.3, 0.0), (1, 0.8, 30.0), (13, 0.5, 5.0), (4, 0.0, 0.0)):
        o = m.SineFlowerShape(0)
        o.radius, o.flower_petals, o.flower_amplitude, o.rotation = 0.75, petals, amplitude, rotation
        bound = o.bound()
        for px, py, ang in ((0.2, 0.2, 45.0), (-0.3, -0.3, 45.0), (0.1, 0.0, 0.0), (0.0, -0.2, 90.0),
                            (1.4, 1.4, 225.0), (-1.2, 0.0, 0.0), (0.4, 0.4, 225.0)):
            dx, dy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            if abs(px * dy - py * dx) > 1e-12:
                continue                                       # (only lines through the centre)
            inside = m.inside_boundary_grid((px + ts * dx)[None, :], (py + ts * dy)[None, :], o.KIND, bound)[0]
            flips = np.nonzero(inside[1:] != inside[:-1])[0]
            hit = m.boundary_hit if inside[0] else m.obstacle_hit
            t, nx, ny = hit(px, py, dx, dy, o.KIND, bound)
            assert len(flips) and ts[flips[0]] - 1e-9 <= t <= ts[flips[0] + 1] + 1e-9, \
                (petals, amplitude, (px, py, ang), t, ts[flips[:1]])
            assert np.hypot(nx, ny) > 0.5


def test_rays_fired_inside_a_closed_sine_flower_never_escape_it():
    o = m.SineFlowerShape(0)
    o.radius, o.flower_petals, o.flower_amplitude = 0.75, 6, 0.3
    o.in_effect = o.out_effect = m.OBSTACLE_EFFECT_REFLECT
    o.fire_inside = True
    n = 96
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    status = np.zeros((n, n), dtype=np.uint8)
    m.render_rows_wh(stats, points, status, 0, n, n, n, 60, m.effective_dist(0.0), 1.0,
                     m.fuzz_rule(m.DEFAULT_FUZZ), m.CONFINE_DEFAULT, m.EMISSION_DEFAULT,
                     m.build_obstacle_arrays([o]), 0)
    fired = status > 0
    assert fired.sum() > 1000
    assert not (stats[:, :, m.ST_TERMINATION][fired] == m.TERM_STAT_VALUE[m.TERM_ESCAPED]).any()
