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
    path_points = np.empty((max_iter + 2, 2))
    path_point_tags = np.empty(max_iter + 2)
    raw = np.empty(m.N_RAW)
    term, npts = m.trace_path(x0, y0, path_points, path_point_tags, _stop(max_iter),
                              m.ray_args(seg_mult, fz, m.EMISSION_DEFAULT, arrays), raw)
    return path_points[:npts], path_point_tags[:npts], term, raw, bound


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
    path_points_a, tags_a, term_a, _, _ = _trace(1.0)
    path_points_b, tags_b, term_b, _, _ = _trace(1.0)
    assert np.array_equal(path_points_a, path_points_b) and term_a == term_b


def test_self_loop_area_zeroed_when_multiplier_active():
    """The self-bounce polygon area metric must read exactly 0 whenever the
    segment length multiplier isn't exactly 1 - the polygon stops meaning
    "the ray's own enclosed area" once bounce points are pulled back."""
    path_points, path_point_tags, term, raw, bound = _trace(1.0, max_iter=10)
    out = np.empty(m.N_STATS)
    m.compute_run_measurements(path_points, len(path_points), raw, 10, 1.0, out)
    assert int(raw[m.RAW_SELF_K]) >= 0, "test setup needs an actual self-bounce"
    assert raw[m.RAW_AREA] > 0, "seg_mult=1.0 should compute a real nonzero area"

    for bad_mult in (0.99, 0.5, m.SEGMENT_MULT_MIN):
        path_points2, tags2, term2, raw2, bound2 = _trace(bad_mult, max_iter=10)
        out2 = np.empty(m.N_STATS)
        m.compute_run_measurements(path_points2, len(path_points2), raw2, 10, bad_mult, out2)
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
                  seg_mult=1.0, confine=None, escape_radius=None):
    """One ray through the real trace_path, heading `angle_deg` (fuzzing is
    switched on with all jitter zero purely to set the direction, and is
    otherwise the identity). `circle` is (incoming effect, outgoing effect[,
    fires rays inside]) of a unit-circle obstacle at the origin, listed first;
    None for no circle. `confine` is a confine_rule (default: the standard rule).
    Returns (points, path_point_tags, term, raw)."""
    confine = m.CONFINE_DEFAULT if confine is None else confine
    escape_radius = m.DEFAULT_ESCAPE_RADIUS if escape_radius is None else escape_radius
    scene = list(obstacles)
    if circle is not None:
        fire = circle[2] if len(circle) > 2 else True
        scene.insert(0, _circle_obstacle(0.0, 0.0, 1.0, circle[0], circle[1], fire=fire))
    arrays = m.build_obstacle_arrays(scene)
    fz = m.fuzz_rule(m.DEFAULT_FUZZ)
    emission = m.emission_rule(m.EMIT_UP, 0.0, 0.0, angle_deg - 90.0)
    path_points = np.empty((max_iter + 2, 2))
    path_point_tags = np.empty(max_iter + 2)
    raw = np.empty(m.N_RAW)
    term, npts = m.trace_path(x0, y0, path_points, path_point_tags, _stop(max_iter, max_dist, escape_radius, confine),
                              m.ray_args(seg_mult, fz, emission, arrays), raw)
    return path_points[:npts].copy(), path_point_tags[:npts].copy(), term, raw


def test_a_trapped_obstacle_keeps_its_ray_inside_forever():
    """Reflect both ways = a container: the ray started inside never leaves."""
    box = _circle_obstacle(0.2, 0.0, 0.5, R, R, fire=True)
    path_points, _k, term, raw = _kernel_trace(0.2, -0.3, 0.0, [box], circle=None, max_iter=40)
    assert term in (m.TERM_ITER, m.TERM_CONFINED)   # its own segments may confine it
    assert int(raw[m.RAW_BOUNCES]) >= 8
    assert (np.hypot(path_points[:, 0] - 0.2, path_points[:, 1]) <= 0.5 + 1e-9).all()
    assert np.isclose(np.hypot(path_points[1, 0] - 0.2, path_points[1, 1]), 0.5)


def test_an_obstacle_passing_rays_through_both_ways_is_as_if_it_were_not_there():
    ghost = _circle_obstacle(0.3, 0.1, 0.3, P, P, fire=False)
    with_it = _kernel_trace(-0.6, -0.4, 30.0, [ghost])
    without = _kernel_trace(-0.6, -0.4, 30.0)
    assert with_it[2] == without[2]
    assert np.allclose(with_it[0], without[0]) and np.allclose(with_it[1], without[1])


def test_enter_but_never_exit_traps_a_ray_that_started_outside():
    """Incoming Pass through, outgoing Reflect: rays enter freely, then stay."""
    trap = _circle_obstacle(0.5, 0.0, 0.3, P, R)
    path_points, _k, term, raw = _kernel_trace(0.6, -0.7, 90.0, [trap], max_iter=30)
    wall_y = np.sqrt(0.09 - 0.01)
    # the entry crossing isn't a bounce and doesn't shorten the segment: the
    # first recorded point is the far wall, reached after crossing the trap
    assert np.allclose(path_points[1], (0.6, wall_y))
    assert (np.hypot(path_points[1:, 0] - 0.5, path_points[1:, 1]) <= 0.3 + 1e-9).all()
    assert term in (m.TERM_ITER, m.TERM_CONFINED)   # its own segments may confine it
    assert int(raw[m.RAW_BOUNCES]) >= 5


def test_incoming_terminate_stops_a_ray_at_the_edge_it_meets_from_outside():
    stop = _circle_obstacle(0.5, 0.0, 0.3, T, P)
    path_points, path_point_tags, term, _raw = _kernel_trace(0.6, -0.7, 90.0, [stop])
    assert term == m.TERM_OBSTACLE
    assert np.allclose(path_points[-1], (0.6, -np.sqrt(0.09 - 0.01)))
    assert path_point_tags[-1] == m.TAG_OBSTACLE_TERM


def test_outgoing_terminate_stops_a_ray_fired_inside_at_the_edge_it_reaches():
    stop = _circle_obstacle(0.5, 0.0, 0.3, R, T, fire=True)
    path_points, path_point_tags, term, _raw = _kernel_trace(0.6, -0.1, 90.0, [stop], circle=None)
    assert term == m.TERM_OBSTACLE
    assert np.allclose(path_points[-1], (0.6, np.sqrt(0.09 - 0.01)))
    assert path_point_tags[-1] == m.TAG_OBSTACLE_TERM


def test_outgoing_pass_through_lets_a_ray_fired_inside_carry_on_to_the_bounding_shape():
    hole = _circle_obstacle(0.0, 0.0, 0.3, R, P, fire=True)
    path_points, _k, _term, _raw = _kernel_trace(0.1, 0.2, 0.0, [hole], circle=(R, R, False))
    assert np.isclose(np.hypot(*path_points[1]), 1.0), "first bounce is on the bounding shape"


def test_a_ray_that_leaves_the_bounding_shape_escapes_at_the_escape_radius():
    path_points, path_point_tags, term, raw = _kernel_trace(0.0, -0.5, 0.0, circle=(R, P))
    assert term == m.TERM_ESCAPED
    assert len(path_points) == 2 and int(raw[m.RAW_BOUNCES]) == 0
    assert np.allclose(path_points[-1], (np.sqrt(m.DEFAULT_ESCAPE_RADIUS ** 2 - 0.25), -0.5)), \
        "ends where it is the escape radius from the origin"
    assert path_point_tags[-1] == m.TAG_ESCAPED


def test_the_escape_radius_decides_where_an_escaping_ray_ends():
    for radius in (1.5, 3.0, 40.0):
        path_points, _path_point_tags, term, _raw = _kernel_trace(0.0, -0.5, 0.0, circle=(R, P), escape_radius=radius)
        assert term == m.TERM_ESCAPED
        want_x = np.sqrt(radius ** 2 - 0.25)
        assert np.allclose(path_points[-1], (want_x, -0.5)), radius
        assert np.isclose(np.hypot(*path_points[-1]), radius)


def test_an_escaping_ray_never_ends_before_the_last_edge_it_passed_through():
    """The unit circle passes the ray out at x = 0.866, further out than a 0.8 escape radius."""
    path_points, _path_point_tags, term, _raw = _kernel_trace(0.0, -0.5, 0.0, circle=(R, P), escape_radius=0.8)
    assert term == m.TERM_ESCAPED and np.allclose(path_points[-1], (np.sqrt(0.75), -0.5))


def test_a_ray_that_starts_at_or_beyond_the_escape_radius_ends_at_its_last_edge():
    far = _circle_obstacle(1.3, 0.3, 0.2, R, P, fire=True)
    path_points, _path_point_tags, term, _raw = _kernel_trace(1.3, 0.3, 0.0, [far], circle=None, escape_radius=1.0)
    assert term == m.TERM_ESCAPED
    assert np.allclose(path_points[-1], (1.5, 0.3)), "it ends at the last edge it passed through (its own)"
    path_points, _path_point_tags, term, _raw = _kernel_trace(1.3, 0.3, 0.0, [far], circle=None, escape_radius=5.0)
    assert np.allclose(path_points[-1], (np.sqrt(25.0 - 0.09), 0.3)), "...unless the radius is further out"


def test_a_ray_escapes_after_bouncing_when_it_finally_meets_nothing():
    wall = _circle_obstacle(0.0, 0.0, 0.3, R, P, fire=False)
    # leaves the (pass-through) bounding shape... bounces off the obstacle first
    path_points, _k, term, raw = _kernel_trace(-0.9, 0.2, 0.0, [wall], circle=(R, P))
    assert term == m.TERM_ESCAPED
    assert int(raw[m.RAW_BOUNCES]) == 1
    assert np.isclose(np.hypot(*path_points[-1]), m.DEFAULT_ESCAPE_RADIUS)


def test_an_obstacle_outside_the_bounding_shape_can_fire_rays_that_meet_it_from_outside():
    far = _circle_obstacle(1.3, 0.3, 0.2, R, P, fire=True)
    path_points, _k, term, _raw = _kernel_trace(1.3, 0.3, 180.0, [far], max_iter=4)
    hit = (np.sqrt(1.0 - 0.09), 0.3)
    assert np.allclose(path_points[1], hit), "meets the bounding shape's OUTSIDE and reflects"
    assert not np.allclose(path_points[2], (0.0, 0.0))


def test_a_ray_can_pass_into_the_bounding_shape_from_outside_and_cross_it():
    far = _circle_obstacle(1.3, 0.3, 0.2, R, P, fire=True)
    path_points, _k, _term, _raw = _kernel_trace(1.3, 0.3, 180.0, [far], circle=(P, R), max_iter=3)
    assert np.allclose(path_points[1], (-np.sqrt(1.0 - 0.09), 0.3)), "reflects off the FAR inner wall"


def test_a_distance_limit_cuts_an_escaping_ray_before_the_escape_radius():
    path_points, path_point_tags, term, raw = _kernel_trace(0.0, -0.5, 0.0, circle=(R, P), max_dist=0.4)
    assert term == m.TERM_DIST
    assert np.allclose(path_points[-1], (0.4, -0.5))
    assert np.isclose(raw[m.RAW_LAST_FULL], np.sqrt(m.DEFAULT_ESCAPE_RADIUS ** 2 - 0.25)), \
        "the uncut segment reached the escape radius"


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
    assert m.TAG_ESCAPED in m.TAG_COLOR


def test_stats_and_points_are_sane_for_an_escaped_ray():
    path_points, path_point_tags, term, raw = _kernel_trace(0.0, -0.5, 0.0, circle=(R, P))
    out = np.empty(m.N_STATS)
    sp = np.empty((m.N_POINTS, 2))
    m.compute_run_measurements(path_points, len(path_points), raw, 12, 1.0, out)
    m.compute_sequences(path_points, len(path_points), path_point_tags, int(raw[m.RAW_BOUNCES]), 12, out, sp)
    assert out[m.ST_TERMINATION] == 1.0
    assert np.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()
    assert np.allclose(sp[m.POINT_END], (np.sqrt(m.DEFAULT_ESCAPE_RADIUS ** 2 - 0.25), -0.5))
    assert np.allclose(sp[m.POINT_START], (0.0, -0.5))
    assert out[m.ST_BOUNCES] == 0.0


def test_every_termination_reason_maps_to_its_own_stat_value_in_compute_stats():
    seen = {}
    for term, value in m.TERM_STAT_VALUE.items():
        path_points = np.array([[0.0, 0.0], [0.5, 0.1], [0.6, 0.3]])
        raw = np.zeros(m.N_RAW)
        raw[m.RAW_TERM] = term
        raw[m.RAW_BOUNCES] = 1
        out = np.empty(m.N_STATS)
        m.compute_run_measurements(path_points, 3, raw, 10, 1.0, out)
        seen[term] = out[m.ST_TERMINATION]
        assert np.isclose(out[m.ST_TERMINATION], value), term
    assert len(set(np.round(list(seen.values()), 6))) == len(m.TERM_CODES)


def test_render_rows_fires_only_from_shapes_with_fire_on():
    """The real (parallel) render kernel: with no firing circle, only pixels
    inside an obstacle with fire on emit a ray."""
    obstacle = _circle_obstacle(0.3, 0.0, 0.25, R, R, fire=True)
    arrays = m.build_obstacle_arrays([obstacle])
    stats = np.zeros((m.SIZE, m.SIZE, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((m.SIZE, m.SIZE, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((m.SIZE, m.SIZE), dtype=np.uint8)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, m.SIZE, m.SIZE, m.SIZE, m.WORLD_RECT_DEFAULT, _stop(8), _ray(arrays), 0)
    ys, xs = np.nonzero(pixel_state == m.PIXEL_TRACED)
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
    m.scatter_multi_buddha(250, 252, m.WORLD_RECT_DEFAULT, _stop(8), _ray(m.build_obstacle_arrays([circle])), 0,
                           False, True, False, False, counts)
    # Nothing to bounce off, so every firing ray is a single segment (start to
    # the escape radius) with exactly one midpoint, which is counted when it lands in the image.
    xs = (np.arange(m.SIZE) + 0.5) * m.PIXEL - 1.0
    gx, gy = np.meshgrid(xs, 1.0 - (np.array([250, 251]) + 0.5) * m.PIXEL)
    firing = np.hypot(gx, gy) < 1.0
    dx = dy = np.sqrt(0.5)                                    # the default emission direction
    b = gx * dx + gy * dy
    end = -b + np.sqrt(b * b - (gx * gx + gy * gy - m.DEFAULT_ESCAPE_RADIUS ** 2))
    mx, my = gx + 0.5 * end * dx, gy + 0.5 * end * dy
    in_image = (np.abs(mx) < 1.0) & (np.abs(my) < 1.0)
    assert counts.sum() == int((firing & in_image).sum()) and counts.sum() > 100


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
    """compute_run_measurements() for a synthetic path: a start point, then `loop_vertices`
    as bounce points 1..n whose polygon is the first self-loop (self-touch at
    the last one, hitting segment 0)."""
    loop = np.asarray(loop_vertices, dtype=float)
    path_points = np.vstack([[-5.0, -5.0], loop])
    raw = np.zeros(m.N_RAW)
    raw[m.RAW_BOUNCES] = len(loop)
    raw[m.RAW_SELF_K] = 0
    raw[m.RAW_SELF_M] = len(loop)
    raw[m.RAW_TERM] = m.TERM_ITER
    out = np.empty(m.N_STATS)
    m.compute_run_measurements(path_points, len(path_points), raw, 20, seg_mult, out)
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
    path_points = np.array([[0.0, 0.0], [0.5, 0.1], [0.6, 0.3]])
    raw = np.zeros(m.N_RAW)
    raw[m.RAW_SELF_K] = -1
    raw[m.RAW_BOUNCES] = 1
    raw[m.RAW_TERM] = m.TERM_ITER
    out = np.empty(m.N_STATS)
    m.compute_run_measurements(path_points, 3, raw, 10, 1.0, out)
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
                    "self_offset", "self_jitter", "self_mult", "rng_seed")

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
    """(stats, characteristic_points, pixel_state) of an n x n render of a classic scene, drawn as
    the equivalent obstacle."""
    fuzz = (m.FuzzParams(**dict(zip(FUZZ_FIELD_ORDER, scene["fuzz"])))
            if "fuzz" in scene else m.DEFAULT_FUZZ)
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((n, n), dtype=np.uint8)
    arrays = m.build_obstacle_arrays([_scene_obstacle(scene)])
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, m.WORLD_RECT_DEFAULT,
                     _stop(scene["iters"], scene.get("dist", 0.0)),
                     _ray(arrays, scene.get("seg_mult", 1.0), fuzz), fuzz.rng_seed)
    return stats, characteristic_points, pixel_state


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
        stats, characteristic_points, pixel_state = _render_scene(scene, n=GOLDEN_GRID)
        assert np.array_equal(pixel_state == m.PIXEL_TRACED, golden[f"status_{idx}"] == 1), scene["name"]
        # the golden data holds the 13 metrics and 13 points recorded before named sequences added slots after them
        assert golden[f"stats_{idx}"].shape[-1] == m.N_LEGACY_STATS and golden[f"points_{idx}"].shape[-2] == m.N_LEGACY_POINTS
        # (not the two fraction metrics: the data records the old count that also took in a final perpendicular hit, which
        # one capsule ray ends with - they are now plain sequence shares, see the test of that)
        recorded = [s for s in range(m.N_LEGACY_STATS) if s not in (m.ST_RAY_FRAC, m.ST_BOUNDARY_FRAC)]
        assert np.allclose(stats[..., recorded], golden[f"stats_{idx}"][..., recorded], atol=1e-5), scene["name"]
        assert np.allclose(characteristic_points[..., :m.N_LEGACY_POINTS, :], golden[f"points_{idx}"], atol=1e-5), scene["name"]


# --------------------------------------------------------------------------- #
# Confinement: lookback, detection method and threshold are settings
# --------------------------------------------------------------------------- #
BOX, PATH = m.CONFINE_BOX, m.CONFINE_PATH


def _confine(lookback, method, threshold):
    return m.confine_rule(lookback, method, threshold)


def test_the_default_confinement_rule_is_the_one_that_was_built_in():
    assert m.DEFAULT_CONFINE_LOOKBACK == 4
    assert m.DEFAULT_CONFINE_METHOD == m.CONFINE_BOX
    assert m.DEFAULT_CONFINE_THRESHOLD == m.PIXEL == 2.0 / 512
    assert m.CONFINE_METHOD_NAMES == ["Box", "Path length"]
    assert m.CONFINE_DEFAULT == m.ConfineRule(lookback=4, method=0, threshold=2.0 / 512)
    s = m.DEFAULT_NATURAL_STOP_SETTINGS
    assert (s.confine_lookback, s.confine_method, s.confine_threshold) == (4, BOX, 2.0 / 512)
    assert m.settings_confine(s) == m.CONFINE_DEFAULT


def _unconfined_path(x, y, angle, iters=60):
    """The full path of a ray with confinement effectively switched off."""
    path_points, path_point_tags, term, raw = _kernel_trace(x, y, angle, max_iter=iters,
                                          confine=_confine(2, BOX, 1e-300))
    return path_points


def _first_confined_count(path_points, lookback, method, threshold):
    """An independent reimplementation of the rule: the first path length
    (number of points) at which the last `lookback` points are confined, judged
    only after each bounce (so from `lookback` points onward)."""
    for count in range(max(lookback, 2), len(path_points) + 1):
        recent = path_points[count - lookback:count]
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
        for lookback in (2, 3, 4, 6, 9):
            for method in (BOX, PATH):
                # thresholds chosen from the path's own scale, so both outcomes occur
                scale = np.hypot(*np.diff(full, axis=0).T)
                for threshold in (scale.min() * 1.5, np.median(scale) * (lookback - 1), 1e-300):
                    expected = _first_confined_count(full, lookback, method, threshold)
                    path_points, _k, term, _r = _kernel_trace(x, y, angle, max_iter=60,
                                                      confine=_confine(lookback, method, threshold))
                    if expected is None:
                        assert term != m.TERM_CONFINED, (lookback, method, threshold)
                    else:
                        assert term == m.TERM_CONFINED, (lookback, method, threshold)
                        assert len(path_points) == expected, (lookback, method, threshold, len(path_points),
                                                      expected)
                        assert np.allclose(path_points, full[:expected])
                    checked += 1
    assert checked > 300


def test_a_huge_threshold_confines_a_ray_as_soon_as_the_lookback_is_full():
    for method in (BOX, PATH):
        for lookback in (2, 5, 12):
            path_points, _k, term, _r = _kernel_trace(0.3, 0.1, 40.0, max_iter=40,
                                              confine=_confine(lookback, method, 1e9))
            assert term == m.TERM_CONFINED
            assert len(path_points) == lookback, (method, lookback)


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
    assert m.CONFINE_DEFAULT == m.ConfineRule(lookback=4, method=BOX, threshold=m.PIXEL)


def test_the_box_method_needs_both_directions_to_be_small():
    """A path that is tiny across but long along one axis is NOT confined by
    the box (both extents must be under the threshold)."""
    path_points, _k, term, _r = _kernel_trace(0.0, -0.5, 0.0, max_iter=6,
                                      confine=_confine(2, BOX, 0.3))
    assert len(path_points) >= 3, "the first segment is wider than the box"


def test_confinement_lookback_and_threshold_parse_and_validate():
    assert m.parse_confine_lookback("4") == 4
    assert m.parse_confine_lookback("2") == 2 and m.parse_confine_lookback("999") == 999
    for bad in ("", "1", "0", "1000", "-3", "3.5", "abc", "4 "):
        assert m.parse_confine_lookback(bad) is None, bad


def test_the_threshold_accepath_points_any_value_above_zero_however_small():
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
    assert m.App.validate_lookback_digits("999") and not m.App.validate_lookback_digits("1000")


# ---- keyframes, settings plumbing -----------------------------------------------------------
def test_confinement_settings_are_registered_and_only_the_threshold_interpolates():
    keys = {k: tab for k, tab, _label in m.SETTING_INFO}
    for key in ("confine_lookback", "confine_method", "confine_threshold"):
        assert keys[key] == "Termination" and key in m.DEFAULT_SETTING_VALUES
    assert "confine_threshold" in m.SETTING_FLOAT_KEYS
    assert "confine_lookback" not in m.SETTING_FLOAT_KEYS
    assert "confine_method" not in m.SETTING_FLOAT_KEYS


def test_confinement_settings_round_trip_through_capture_and_resolve():
    app = _make_fake_app_for_snapshot()
    app.confine_lookback_var = _StubVar("9")
    app.confine_method_var = _StubVar("Path length")
    app.confine_threshold_var = _StubVar("1e-12")
    app.deco = {"outline": _StubVar(True)}
    app.obstacles = []
    app.colorations = []
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    data = app.capture_all_settings()
    assert data["confine_lookback"] == 9 and data["confine_method"] == PATH
    assert data["confine_threshold"] == 1e-12
    resolved = m.resolve_frame_settings([m.Keyframe(0, 0, data)], 0).natural_stop
    assert (resolved.confine_lookback, resolved.confine_method, resolved.confine_threshold) \
        == (9, PATH, 1e-12)


def test_confinement_threshold_interpolates_but_lookback_and_method_hold():
    kf0 = m.Keyframe(0, 0, {"confine_lookback": 4, "confine_method": BOX,
                            "confine_threshold": 0.0})
    kf1 = m.Keyframe(1, 10, {"confine_lookback": 20, "confine_method": PATH,
                             "confine_threshold": 1.0})
    mid = m.resolve_frame_settings([kf0, kf1], 5).natural_stop
    assert mid.confine_lookback == 4 and mid.confine_method == BOX
    assert np.isclose(mid.confine_threshold, 0.5)
    end = m.resolve_frame_settings([kf0, kf1], 10).natural_stop
    assert end.confine_lookback == 20 and end.confine_method == PATH


def test_short_value_text_for_the_confinement_settings():
    assert m.short_value_text("confine_lookback", 12) == "12"
    assert m.short_value_text("confine_method", PATH) == "Path length"
    assert m.short_value_text("confine_threshold", 0.00390625) == "0.00390625"
    assert m.short_value_text("confine_threshold", 1e-9) == "1e-09"


def test_a_changed_confinement_rule_makes_the_settings_differ():
    a = m.DEFAULT_NATURAL_STOP_SETTINGS
    for change in (dict(confine_lookback=5), dict(confine_method=PATH),
                   dict(confine_threshold=0.5)):
        assert m.DEFAULT_NATURAL_STOP_SETTINGS.replace(**change) != a


# ---- the entry handlers ---------------------------------------------------------------------
def _fake_limits_app(lookback="4", method="Box", threshold="0.00390625"):
    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    app.on_limit_var_changed = m.App.on_limit_var_changed.__get__(app)
    app.on_limit_commit = m.App.on_limit_commit.__get__(app)
    app.iter_var, app.dist_var = _StubVar("256"), _StubVar("0")
    app.escape_radius_var, app.last_escape_radius = _StubVar("2.0"), 2.0
    app.confine_lookback_var = _StubVar(lookback)
    app.confine_method_var = _StubVar(method)
    app.confine_threshold_var = _StubVar(threshold)
    for field in m.EMIT_TARGET_FIELDS:
        setattr(app, f"emit_target_{field}_var", _StubVar("0.0"))
        setattr(app, f"last_emit_target_{field}", 0.0)
    app.last_iter, app.last_dist = 256, 0
    app.last_confine_lookback = 4
    app.last_confine_threshold = m.PIXEL
    app.styles = {}

    class Entry:
        def __init__(self, name):
            self.name = name

        def configure(self, style=None, **_k):
            app.styles[self.name] = style
    app.iter_entry, app.dist_entry = Entry("iter"), Entry("dist")
    app.escape_radius_entry = Entry("escape_radius")
    app.confine_lookback_entry, app.confine_threshold_entry = Entry("lookback"), Entry("threshold")
    for field in m.EMIT_TARGET_FIELDS:
        setattr(app, f"emit_target_{field}_entry", Entry(f"target_{field}"))
    app.limit_msg_var, app.confine_msg_var = _StubVar(""), _StubVar("")
    app.emit_msg_var = _StubVar("")
    app.invalidate_preview = app.update_buttons = lambda: None
    app.refresh_timeline_diffs = lambda: None
    return app


def test_valid_confinement_entries_update_the_last_good_values():
    app = _fake_limits_app(lookback="9", threshold="1e-30")
    app.on_limit_var_changed()
    assert app.last_confine_lookback == 9 and app.last_confine_threshold == 1e-30
    assert app.styles["lookback"] == "TEntry" and app.styles["threshold"] == "TEntry"
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


def test_an_invalid_lookback_is_flagged_and_restored_on_commit():
    app = _fake_limits_app(lookback="1")
    app.on_limit_var_changed()
    assert app.styles["lookback"] == "Invalid.TEntry"
    assert "whole number from 2 to 999" in app.confine_msg_var.get()
    app.on_limit_commit()
    assert app.confine_lookback_var.get() == "4"


def test_committing_tidies_a_valid_threshold_without_changing_its_value():
    app = _fake_limits_app(threshold="4E-3")
    app.on_limit_var_changed()
    app.on_limit_commit()
    assert app.confine_threshold_var.get() == "0.004"
    assert float(app.confine_threshold_var.get()) == 0.004


def test_the_natural_stop_settings_are_none_while_a_confinement_entry_is_invalid():
    for field, bad in (("confine_lookback_var", "1"), ("confine_threshold_var", "0")):
        app = _make_fake_app_for_snapshot()
        setattr(app, field, _StubVar(bad))
        assert app.current_natural_stop_settings() is None, field
        assert app.current_generation_settings() is None, field
        assert app.current_ray_settings() is not None, "the ray engine's own settings are unaffected"


def test_the_natural_stop_settings_read_the_confinement_controls():
    app = _make_fake_app_for_snapshot()
    app.confine_lookback_var = _StubVar("7")
    app.confine_method_var = _StubVar("Path length")
    app.confine_threshold_var = _StubVar("2.5e-9")
    snap = app.current_natural_stop_settings()
    assert (snap.confine_lookback, snap.confine_method, snap.confine_threshold) == (7, PATH, 2.5e-9)


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


def test_the_termination_tab_has_a_confinement_section_with_all_three_controls():
    import inspect
    assert "self.build_confinement_controls(termination_inner)" in \
        inspect.getsource(m.App.build_controls_quadrant)
    source = inspect.getsource(m.App.build_confinement_controls)
    assert 'text="Confinement"' in source
    for control in ("confine_lookback_entry", "confine_method_var", "confine_threshold_entry"):
        assert control in source


# ---- the figure settings margin ------------------------------------------------------------
def test_figure_info_lists_the_confinement_rule():
    lines = m.figure_info_lines(*_generation(), _start_scene())
    assert "Confine:Box" in lines and "  lookback 4" in lines
    assert "  box size 0.00390625" in lines
    lines = m.figure_info_lines(*_generation(confine_lookback=9, confine_method=PATH,
                                          confine_threshold=1.5e-9), _start_scene())
    assert "Confine:Path length" in lines and "  lookback 9" in lines
    assert "  path length 1.5e-09" in lines


def test_confinement_lines_fit_a_column_for_extreme_values_and_use_known_glyphs():
    for method in (BOX, PATH):
        for threshold in (5e-324, 1.7976931348623157e308, 0.1234567891234):
            lines = m.figure_info_lines(*_generation(confine_lookback=999, confine_method=method,
                                                  confine_threshold=threshold), _start_scene())
            for line in lines:
                assert len(line) <= m.FIGURE_INFO_COLUMN_CHARS, line
                assert all(c in m.FONT_5X7 for c in line), line


def test_the_confinement_block_has_the_same_size_whatever_its_values():
    a = m.figure_info_blocks(*_generation(), _start_scene())
    b = m.figure_info_blocks(*_generation(confine_lookback=500, confine_method=PATH,
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
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((n, n), dtype=np.uint8)
    up = _emission(UP, offset=0.0)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, m.WORLD_RECT_DEFAULT, _stop(1), _ray(arrays, emission=up), 0)
    fired = pixel_state == m.PIXEL_TRACED
    xs = ((np.arange(n) + 0.5) * (2.0 / n) - 1.0)[None, :].repeat(n, axis=0)
    ys = (1.0 - (np.arange(n) + 0.5) * (2.0 / n))[:, None].repeat(n, axis=1)
    ends = characteristic_points[..., m.POINT_END, :]
    assert fired.sum() > 300
    assert np.allclose(ends[..., 0][fired], xs[fired], atol=1e-6), "straight up keeps x"
    assert (ends[..., 1][fired] > ys[fired]).all(), "and goes up"
    toward = _emission(TOWARDS, 0.0, 0.0, 0.0)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, m.WORLD_RECT_DEFAULT, _stop(1), _ray(arrays, emission=toward), 0)
    ends = characteristic_points[..., m.POINT_END, :]
    fired = pixel_state == m.PIXEL_TRACED
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
        assert keys[key] == "Emissions" and key in m.DEFAULT_SETTING_VALUES
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
    app.colorations = []
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


def test_the_emissions_tab_has_a_ray_emission_section_with_all_its_controls():
    import inspect
    assert "self.build_emission_controls(emissions_inner)" in \
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
    lines = m.figure_info_lines(*_generation(), _start_scene())
    assert "Emit:Up" in lines and "  angle offset -45.00\u00b0" in lines
    lines = m.figure_info_lines(*_generation(emit_mode=TOWARDS, emit_target_r=-0.5,
                                          emit_target_theta=30.0, emit_target_cx=1.25,
                                          emit_target_cy=-2.0, emit_offset=10.0),
                                _start_scene())
    assert "Emit:Towards point" in lines
    assert "  target r -0.5" in lines and "  target angle 30" in lines
    assert "  offset x 1.25" in lines and "  offset y -2" in lines
    assert "  angle offset +10.00\u00b0" in lines


def test_emission_lines_fit_a_column_for_extreme_values_and_use_known_glyphs():
    for big, small in ((-1.7976931348623157e308, 1.7976931348623157e308), (5e-324, -5e-324)):
        lines = m.figure_info_lines(*_generation(emit_mode=TOWARDS, emit_target_r=big,
                                              emit_target_theta=small, emit_target_cx=big,
                                              emit_target_cy=small, emit_offset=-179.99),
                                    _start_scene())
        for line in lines:
            assert len(line) <= m.FIGURE_INFO_COLUMN_CHARS, line
            assert all(c in m.FONT_5X7 for c in line), line


def test_the_emission_block_has_the_same_size_whatever_its_values():
    a = m.figure_info_blocks(*_generation(), _start_scene())
    b = m.figure_info_blocks(*_generation(emit_mode=TOWARDS, emit_target_r=3.0,
                                       emit_target_theta=-20.0, emit_target_cx=3.0,
                                       emit_target_cy=-2.0, emit_offset=120.0), _start_scene())
    assert [len(x) for x in a] == [len(x) for x in b]


# --------------------------------------------------------------------------- #
# Kernel arguments: FuzzRule / ConfineRule / EmissionRule / ObstacleArrays
# --------------------------------------------------------------------------- #
def test_the_rule_builders_give_named_typed_fields():
    f = m.fuzz_rule(m.DEFAULT_FUZZ.replace(enabled=True, emit_jitter=5.0, bnd_mult=2.0, rng_seed=9))
    assert isinstance(f, m.FuzzRule) and f.enabled is True
    assert (f.emit_jitter, f.bnd_mult, f.self_mult) == (5.0, 2.0, 1.0)
    c = m.confine_rule(9.0, 1.0, 0.25)                       # floats become the ints the kernels need
    assert (c, type(c.lookback), type(c.method)) == (m.ConfineRule(lookback=9, method=1, threshold=0.25),
                                                  int, int)
    e = m.emission_rule(1, 2, 3, 4)
    assert e == m.EmissionRule(mode=1, target_x=2.0, target_y=3.0, offset=4.0)
    assert all(type(v) is float for v in (e.target_x, e.target_y, e.offset))
    settings, stop = _generation(emit_mode=m.EMIT_TOWARDS, emit_target_r=2.0, emit_target_theta=90.0,
                                 emit_offset=7.0, confine_lookback=6, confine_threshold=0.5)
    assert m.settings_confine(stop) == m.ConfineRule(lookback=6, method=stop.confine_method,
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
        m.scatter_multi_buddha(0, w, m.WORLD_RECT_DEFAULT, _stop(40), _ray(arrays), 0, False, True, False, False, counts)
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
    """(stats, characteristic_points, pixel_state) of an n x n render with `shape` (an obstacle) as the only obstacle."""
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((n, n), dtype=np.uint8)
    shape.in_effect = shape.out_effect = m.OBSTACLE_EFFECT_REFLECT
    shape.fire_inside = True
    arrays = m.build_obstacle_arrays([shape])
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, m.WORLD_RECT_DEFAULT, _stop(max_iter), _ray(arrays), 0)
    return stats, characteristic_points, pixel_state


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
            path_points, path_point_tags, _term, _raw = _kernel_trace(px, py, np.degrees(np.arctan2(dy, dx)), [o],
                                                    circle=None, max_iter=40)
            for k in range(1, len(path_points)):
                if path_point_tags[k] == m.TAG_BOUNDARY:             # a bounce off the edge lies on the edge
                    x, y = path_points[k]
                    around = [m.inside_boundary(x + 4e-6 * np.cos(a), y + 4e-6 * np.sin(a), o.KIND,
                                                bound)
                              for a in np.linspace(0.0, 2 * np.pi, 16, endpoint=False)]
                    assert any(around) and not all(around), (type(o).__name__, o.state(), k, x, y)
                mid = 0.5 * (path_points[k - 1] + path_points[k])           # and every segment runs through the inside
                assert m.inside_boundary(mid[0], mid[1], o.KIND, bound), (type(o).__name__, o.state(), k)
            hits += len(path_points) - 1
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
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((n, n), dtype=np.uint8)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, m.WORLD_RECT_DEFAULT, _stop(60), _ray(m.build_obstacle_arrays([o])), 0)
    fired = pixel_state == m.PIXEL_TRACED
    assert fired.sum() > 1000
    assert not (stats[:, :, m.ST_TERMINATION][fired] == m.TERM_STAT_VALUE[m.TERM_ESCAPED]).any()


def test_an_invalid_escape_radius_is_flagged_and_the_last_good_value_restored_on_commit():
    app = _fake_limits_app()
    app.escape_radius_var.set("1.5e3")
    app.on_limit_var_changed()
    assert app.last_escape_radius == 1500.0 and app.styles["escape_radius"] == "TEntry"
    for bad in ("0", "", "-1", "1e999", "abc"):
        app.escape_radius_var.set(bad)
        app.on_limit_var_changed()
        assert app.styles["escape_radius"] == "Invalid.TEntry", bad
        assert "Escape radius" in app.limit_msg_var.get()
        assert app.last_escape_radius == 1500.0
    app.on_limit_commit()
    assert app.escape_radius_var.get() == "1500.0"


# --------------------------------------------------------------------------- #
# The natural stop rule and its shared checks: what any engine that makes sequences of points can use
# --------------------------------------------------------------------------- #
def test_is_confined_judges_the_last_lookback_of_the_points_by_box_or_by_path_length():
    path_points = np.array([[5.0, 5.0], [9.0, 9.0], [0.0, 0.0], [0.001, 0.0], [0.0, 0.001], [0.001, 0.001]])
    box, path = m.ConfineRule(lookback=4, method=m.CONFINE_BOX, threshold=0.01), \
        m.ConfineRule(lookback=4, method=m.CONFINE_PATH, threshold=0.01)
    assert m.is_confined(path_points, 6, box) and m.is_confined(path_points, 6, path)        # the last four are within 0.001
    assert not m.is_confined(path_points, 5, box)                                    # [9,9] is in that lookback
    assert not m.is_confined(path_points, 3, box), "fewer points than the lookback: not confined"
    assert m.is_confined(path_points, 4, m.ConfineRule(lookback=2, method=m.CONFINE_BOX, threshold=1.0))
    long_way_round = np.array([[0.0, 0.0], [0.4, 0.0], [0.0, 0.0], [0.4, 0.0]])
    # a point that goes back and forth over 0.4 three times: a small box, but a long way round
    assert m.is_confined(long_way_round, 4, m.ConfineRule(lookback=4, method=m.CONFINE_BOX, threshold=0.5))
    assert not m.is_confined(long_way_round, 4, m.ConfineRule(lookback=4, method=m.CONFINE_BOX, threshold=0.3))
    assert not m.is_confined(long_way_round, 4, m.ConfineRule(lookback=4, method=m.CONFINE_PATH, threshold=1.0))
    assert m.is_confined(long_way_round, 4, m.ConfineRule(lookback=4, method=m.CONFINE_PATH, threshold=1.3))


def test_beyond_escape_radius_is_at_or_past_the_radius():
    assert m.beyond_escape_radius(2.0, 0.0, 2.0) and m.beyond_escape_radius(0.0, -3.0, 2.0)
    assert m.beyond_escape_radius(1.5, 1.5, 2.0) and not m.beyond_escape_radius(1.0, 1.0, 2.0)
    assert not m.beyond_escape_radius(0.0, 0.0, 1e-3) and m.beyond_escape_radius(0.0, 0.0, 0.0)


_beyond_escape_radius = m.beyond_escape_radius
_is_confined = m.is_confined
_TERM_ESCAPED, _TERM_CONFINED, _TERM_ITER = m.TERM_ESCAPED, m.TERM_CONFINED, m.TERM_ITER


@m.njit
def _escape_time_orbit(c_re, c_im, path_points, stop):
    """A different kind of sequence engine, as small as can be - the orbit of 0 under z -> z^2 + c - that ends
    by the shared NaturalStopRule and the shared checks alone. Returns (why it ended, number of points)."""
    x = 0.0
    y = 0.0
    path_points[0, 0] = x
    path_points[0, 1] = y
    n = 1
    for _ in range(stop.max_iter):
        x, y = x * x - y * y + c_re, 2.0 * x * y + c_im
        path_points[n, 0] = x
        path_points[n, 1] = y
        n += 1
        if _beyond_escape_radius(x, y, stop.escape_radius):
            return _TERM_ESCAPED, n
        if _is_confined(path_points, n, stop.confine):
            return _TERM_CONFINED, n
    return _TERM_ITER, n


def test_another_engine_can_be_ended_by_the_same_natural_stop_rule_and_checks():
    path_points = np.empty((300, 2))
    stop = _stop(256)
    term, n = _escape_time_orbit(1.0, 0.0, path_points, stop)                    # 0, 1, 2: out at the third point
    assert (term, n) == (m.TERM_ESCAPED, 3) and np.allclose(path_points[:3, 0], [0.0, 1.0, 2.0])
    term, n = _escape_time_orbit(-0.1, 0.0, path_points, stop)                   # settles on a fixed point
    assert term == m.TERM_CONFINED and 4 < n < 100
    term, n = _escape_time_orbit(-1.0, 0.0, path_points, _stop(20))              # a cycle of 0 and -1: it never settles
    assert (term, n) == (m.TERM_ITER, 21)
    term, n = _escape_time_orbit(1.0, 0.0, path_points, _stop(256, escape_radius=100.0))   # a larger radius: later
    assert term == m.TERM_ESCAPED and n == 6 and path_points[5, 0] == 677.0      # 0, 1, 2, 5, 26, 677
    assert _escape_time_orbit(1.0, 0.0, path_points, _stop(256, escape_radius=1.5))[1] == 3


def test_the_ray_engine_still_ends_by_the_same_shared_checks_as_its_own_inline_ones_did():
    """trace_path now asks is_confined where it used to judge the lookback inline: a confined ray ends by
    TERM_CONFINED exactly when the shared check says its last lookback is confined."""
    rule = m.confine_rule(4, m.CONFINE_BOX, 3.0)                 # wider than the unit circle: 4 points confine
    path_points, path_point_tags, term, raw = _kernel_trace(0.1, 0.2, 30.0, [], circle=(R, R), max_iter=400, confine=rule)
    assert term == m.TERM_CONFINED and len(path_points) == 4
    assert m.is_confined(path_points, len(path_points), rule) and not m.is_confined(path_points, len(path_points) - 1, rule)
    path_rule = m.confine_rule(3, m.CONFINE_PATH, 2.5)           # two segments in the lookback, each under a diameter
    path_points, path_point_tags, term, raw = _kernel_trace(0.1, 0.2, 30.0, [], circle=(R, R), max_iter=400, confine=path_rule)
    assert m.is_confined(path_points, len(path_points), path_rule) == (term == m.TERM_CONFINED)


# --------------------------------------------------------------------------- #
# Named sequences of points: filtered views of one tagged path, each reduced to the same stored numbers
# --------------------------------------------------------------------------- #
def _reference_members(seq, path_point_tags, nb, npts):
    """Which points of a path are in the named sequence, written independently of the kernel."""
    if seq == m.SEQ_PATH:
        return list(range(npts))
    bounces = range(1, nb + 1)
    if seq == m.SEQ_BOUNCES:
        return list(bounces)
    wanted = m.TAG_BOUNDARY if seq == m.SEQ_BOUNDARY else m.TAG_SELF
    return [k for k in bounces if path_point_tags[k] == wanted]


def _reference_point(seq, reduction, path_points, path_point_tags, nb, npts):
    members = _reference_members(seq, path_point_tags, nb, npts)
    if not members:
        return path_points[npts - 1]                       # an empty sequence reports the path's end point
    xy = path_points[members]
    return {m.PRED_FIRST: lambda: xy[0], m.PRED_LAST: lambda: xy[-1], m.PRED_MEAN: lambda: xy.mean(axis=0),
            m.PRED_TOP: lambda: xy[np.argmax(xy[:, 1])], m.PRED_BOTTOM: lambda: xy[np.argmin(xy[:, 1])],
            m.PRED_RIGHT: lambda: xy[np.argmax(xy[:, 0])], m.PRED_LEFT: lambda: xy[np.argmin(xy[:, 0])]}[reduction]()


def _reference_scalar(seq, reduction, path_points, path_point_tags, nb, npts, max_iter):
    xy = path_points[_reference_members(seq, path_point_tags, nb, npts)]
    if reduction == m.SRED_COUNT:
        return min(len(xy) / max_iter, 1.0)
    if reduction == m.SRED_SHARE:
        return min(len(xy) / nb, 1.0) if nb > 0 else 0.0
    steps = np.diff(xy, axis=0) if len(xy) > 1 else np.zeros((0, 2))
    if reduction == m.SRED_LENGTH:
        return np.arctan(np.hypot(steps[:, 0], steps[:, 1]).sum() / 5.0) * 2.0 / np.pi
    angle = np.arctan2(steps[-1, 1], steps[-1, 0]) if len(steps) else 0.0
    return min(max((angle + np.pi) / (2.0 * np.pi), 0.0), 1.0)


def _stored_for_path(path_points, path_point_tags, nb, max_iter=20):
    """(characteristic points, stored metrics) the kernels make of a path - the whole stored layout."""
    npts = len(path_points)
    raw = np.zeros(m.N_RAW)
    raw[m.RAW_BOUNCES] = nb
    raw[m.RAW_SELF_K] = -1
    raw[m.RAW_TERM] = m.TERM_ITER
    out = np.full(m.N_STATS, np.nan)
    sp = np.full((m.N_POINTS, 2), np.nan)
    m.compute_run_measurements(path_points, npts, raw, max_iter, 1.0, out)
    m.compute_sequences(path_points, npts, path_point_tags, nb, max_iter, out, sp)
    return sp, out


def _check_stored_against_reference(path_points, path_point_tags, nb, max_iter=20, label=""):
    npts = len(path_points)
    sp, out = _stored_for_path(path_points, path_point_tags, nb, max_iter)
    assert np.isfinite(sp).all() and np.isfinite(out).all(), f"a slot was never written {label}"
    for seq in range(m.N_SEQS):
        for red in range(m.N_PRED):
            want = _reference_point(seq, red, path_points, path_point_tags, nb, npts)
            got = sp[m.point_slot(seq, red)]
            assert np.allclose(got, want, rtol=0, atol=1e-12), (label, "point", seq, red, got, want)
        for red in range(m.N_SRED):
            want = _reference_scalar(seq, red, path_points, path_point_tags, nb, npts, max_iter)
            assert np.isclose(out[m.stat_slot(seq, red)], want, rtol=0, atol=1e-12), (label, "scalar", seq, red)


#  a hand-made path: start, boundary, self, boundary, self bounces, then a distance-cut end point
_HAND_PTS = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0], [-1.0, 2.0], [0.0, -2.0], [3.0, 3.0]])
_HAND_KINDS = np.array([m.TAG_START, m.TAG_BOUNDARY, m.TAG_SELF, m.TAG_BOUNDARY, m.TAG_SELF, m.TAG_DISTCUT])


def test_each_named_sequence_is_a_filtered_view_of_the_one_tagged_path():
    members = {s: [k for k in range(6) if m.in_sequence(s, k, _HAND_KINDS)] for s in range(m.N_SEQS)}
    assert members == {m.SEQ_PATH: [0, 1, 2, 3, 4, 5],
                       m.SEQ_BOUNCES: [1, 2, 3, 4],        # not the start, nor the cut end point
                       m.SEQ_BOUNDARY: [1, 3], m.SEQ_SELF: [2, 4]}
    assert not any(m.in_sequence(s, 0, _HAND_KINDS) for s in (m.SEQ_BOUNCES, m.SEQ_BOUNDARY, m.SEQ_SELF))
    # an end point that is not a bounce, of whichever tag, is in no bounce sequence
    for end_tag in (m.TAG_DISTCUT, m.TAG_OBSTACLE_TERM, m.TAG_ESCAPED, m.TAG_PERPENDICULAR):
        path_point_tags = _HAND_KINDS.copy()
        path_point_tags[5] = end_tag
        assert [m.in_sequence(s, 5, path_point_tags) for s in range(m.N_SEQS)] == [True, False, False, False], end_tag


def test_a_ray_ended_by_perpendicular_incidence_ends_on_a_point_of_its_own_tag_that_is_no_bounce():
    """Regression test: the final point of such a ray used to be tagged TAG_BOUNDARY / TAG_SELF like a bounce and was
    kept out of the bounce sequences only by the bounce count (`k <= nb`), so a membership test by tag alone was wrong
    for it. It is now TAG_PERPENDICULAR, like the other end points that are not bounces."""
    path_points, path_point_tags, term, raw = _kernel_trace(0.3, 0.3, 45.0, [], circle=(R, R), max_iter=50)    # straight through the centre
    assert term == m.TERM_PERPENDICULAR and len(path_points) == 2 and raw[m.RAW_BOUNCES] == 0
    assert path_point_tags[-1] == m.TAG_PERPENDICULAR and np.allclose(path_points[-1], (0.5 ** 0.5, 0.5 ** 0.5))
    assert [m.in_sequence(s, 1, path_point_tags) for s in range(m.N_SEQS)] == [True, False, False, False]


def test_membership_by_tag_alone_equals_the_bounce_count_rule_on_every_ray_of_a_scene():
    """in_sequence decides by a point's tag alone; checked against the independent reference (the points
    path_points[1] .. path_points[nb] by their tags) on every ray of a scene with a circle and an ellipse, which includes rays ended
    by perpendicular incidence, with and without the segment multiplier."""
    ellipse = m.EllipseShape(1)
    ellipse.width, ellipse.height, ellipse.rotation, ellipse.cx, ellipse.cy = 0.7, 0.45, 25.0, 0.15, -0.1
    arrays = m.build_obstacle_arrays(_start_scene() + [ellipse])
    n, checked, perpendicular = 64, 0, 0
    for seg_mult in (1.0, 0.7):
        stop, ray = _stop(80), _ray(arrays, seg_mult=seg_mult)
        for j in range(n):
            for i in range(n):
                path_points, path_point_tags, raw = np.empty((82, 2)), np.empty(82), np.empty(m.N_RAW)
                term, npts = m.trace_path((i + 0.5) * 2.0 / n - 1.0, 1.0 - (j + 0.5) * 2.0 / n, path_points, path_point_tags, stop, ray, raw)
                nb = int(raw[m.RAW_BOUNCES])
                for seq in range(m.N_SEQS):
                    got = [k for k in range(npts) if m.in_sequence(seq, k, path_point_tags)]
                    assert got == _reference_members(seq, path_point_tags, nb, npts), (i, j, seq)
                checked += 1
                perpendicular += term == m.TERM_PERPENDICULAR
    assert perpendicular > 20 and checked == 2 * n * n


def test_every_path_point_tag_has_a_colour_for_the_segment_graphics():
    tags = {v for k, v in vars(m).items() if k.startswith("TAG_") and isinstance(v, float)}
    assert tags == set(m.TAG_COLOR), tags ^ set(m.TAG_COLOR)


def test_the_choices_a_coloration_could_always_make_are_aliases_that_still_pick_what_they_always_picked():
    """The 13 sample points and the metrics that are reductions of a sequence, on a path small enough to work out by
    hand - independent of the alias tables, which they must agree with."""
    sp, out = _stored_for_path(_HAND_PTS, _HAND_KINDS, 4, max_iter=20)
    expected = {m.POINT_START: (0, 0), m.POINT_END: (3, 3), m.POINT_ALL_MEAN: (0.5, 1.0),
                m.POINT_BOUNDARY_MEAN: (0.0, 1.5), m.POINT_SELF_MEAN: (1.0, 0.5),
                m.POINT_TOP: (2, 3), m.POINT_BOTTOM: (0, -2), m.POINT_RIGHT: (2, 3), m.POINT_LEFT: (-1, 2),
                m.POINT_SELF_TOP: (2, 3), m.POINT_SELF_BOTTOM: (0, -2), m.POINT_SELF_RIGHT: (2, 3),
                m.POINT_SELF_LEFT: (0, -2)}
    assert len(expected) == m.N_LEGACY_POINTS
    for point, xy in expected.items():
        assert np.allclose(sp[point], xy), (point, sp[point], xy)
    assert out[m.ST_BOUNCES] == 4 / 20
    path_length = np.hypot(*np.diff(_HAND_PTS, axis=0).T).sum()
    assert np.isclose(out[m.ST_LENGTH], np.arctan(path_length / 5.0) * 2.0 / np.pi)
    assert np.isclose(out[m.ST_LAST_DIR], (np.arctan2(5.0, 3.0) + np.pi) / (2.0 * np.pi))     # (0, -2) -> (3, 3)
    # no self bounce: every self choice is the end point, the boundary ones are still the boundary's
    path_point_tags = _HAND_KINDS.copy()
    path_point_tags[[2, 4]] = m.TAG_BOUNDARY
    sp, _out = _stored_for_path(_HAND_PTS, path_point_tags, 4)
    for point in (m.POINT_SELF_MEAN, m.POINT_SELF_TOP, m.POINT_SELF_BOTTOM, m.POINT_SELF_RIGHT, m.POINT_SELF_LEFT):
        assert np.allclose(sp[point], (3, 3)), point
    assert np.allclose(sp[m.POINT_BOUNDARY_MEAN], (0.5, 1.0))
    # no bounce at all: every bounce choice is the end point
    sp, out = _stored_for_path(_HAND_PTS[[0, 5]], _HAND_KINDS[[0, 5]], 0)
    assert all(np.allclose(sp[p], (3, 3)) for p in range(m.N_LEGACY_POINTS) if p != m.POINT_START)
    assert out[m.ST_BOUNCES] == 0.0


def test_every_stored_point_and_scalar_is_its_sequences_reduction_by_an_independent_reference():
    """Random paths with coarse coordinates (so there are ties for every extremum), every mix of tags,
    paths with no bounces, and one-point paths: all N_POINTS points and every scalar reduction."""
    rng = np.random.RandomState(7)
    for trial in range(400):
        npts = int(rng.randint(1, 20))
        nb = int(rng.randint(0, npts))
        path_points = np.round(rng.uniform(-1, 1, (npts, 2)), 1)
        path_point_tags = np.zeros(npts)
        for k in range(1, npts):
            path_point_tags[k] = (rng.choice([m.TAG_BOUNDARY, m.TAG_SELF]) if k <= nb else
                        rng.choice([m.TAG_DISTCUT, m.TAG_OBSTACLE_TERM, m.TAG_ESCAPED, m.TAG_PERPENDICULAR]))
        _check_stored_against_reference(path_points, path_point_tags, nb, max_iter=int(rng.randint(1, 30)), label=f"trial {trial}")


def test_a_sequence_with_fewer_than_two_points_has_no_direction_and_an_empty_one_reports_the_end_point():
    path_points = np.array([[0.0, 0.0], [0.25, -0.5]])
    sp, out = _stored_for_path(path_points, np.array([m.TAG_START, m.TAG_ESCAPED]), 0)
    assert out[m.stat_slot(m.SEQ_BOUNCES, m.SRED_LAST_DIR)] == 0.5 and out[m.stat_slot(m.SEQ_SELF, m.SRED_COUNT)] == 0.0
    assert out[m.stat_slot(m.SEQ_SELF, m.SRED_LENGTH)] == 0.0
    assert np.allclose(sp[m.point_slot(m.SEQ_SELF, m.PRED_FIRST)], (0.25, -0.5))
    assert out[m.stat_slot(m.SEQ_PATH, m.SRED_COUNT)] == 2 / 20


def test_a_pixel_is_pending_until_its_row_is_rendered_then_traced_or_has_no_ray():
    """pixel_state tells the three things apart on its own: rows not rendered yet stay PIXEL_PENDING, a rendered pixel
    outside every firing shape is PIXEL_NO_RAY (it used to be 0, the same as pending), and one that fires is
    PIXEL_TRACED, with its stats stored. A finished render leaves no pixel pending."""
    arrays = m.build_obstacle_arrays(_start_scene())           # the unit circle: the image's corners are outside it
    stop, ray, n = _stop(8), _ray(arrays), 16
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.full((n, n), m.PIXEL_PENDING, dtype=np.uint8)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, 6, n, n, m.WORLD_RECT_DEFAULT, stop, ray, 0)
    assert (pixel_state[6:] == m.PIXEL_PENDING).all()
    assert not (pixel_state[:6] == m.PIXEL_PENDING).any()
    assert pixel_state[0, 0] == m.PIXEL_NO_RAY and pixel_state[5, 8] == m.PIXEL_TRACED
    m.trace_rows(stats, characteristic_points, pixel_state, 6, n, n, n, m.WORLD_RECT_DEFAULT, stop, ray, 0)
    assert set(np.unique(pixel_state)) == {m.PIXEL_TRACED, m.PIXEL_NO_RAY}
    centre_x = (np.arange(n) + 0.5) * 2.0 / n - 1.0
    inside = centre_x[None, :] ** 2 + centre_x[::-1][:, None] ** 2 < 1.0
    assert np.array_equal(pixel_state == m.PIXEL_TRACED, inside)
    assert (stats[pixel_state == m.PIXEL_NO_RAY] == 0).all()


def test_a_rendered_pixels_stored_sequences_are_those_of_a_retrace_of_its_ray():
    """The whole stored layout, end to end: the arrays trace_rows fills hold, for each pixel, every reduction of
    every sequence of the path trace_path makes for that pixel's ray (checked with the independent reference)."""
    ellipse = m.EllipseShape(1)
    ellipse.width, ellipse.height, ellipse.rotation, ellipse.cx, ellipse.cy = 0.7, 0.45, 25.0, 0.15, -0.1
    arrays = m.build_obstacle_arrays(_start_scene() + [ellipse])
    stop, ray, n = _stop(60), _ray(arrays, seg_mult=0.9), 24
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((n, n), dtype=np.uint8)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, m.WORLD_RECT_DEFAULT, stop, ray, 0)
    checked = 0
    for j in range(0, n, 2):
        for i in range(0, n, 2):
            if pixel_state[j, i] != m.PIXEL_TRACED:
                continue
            path_points, path_point_tags, raw = np.empty((62, 2)), np.empty(62), np.empty(m.N_RAW)
            _term, npts = m.trace_path((i + 0.5) * 2.0 / n - 1.0, 1.0 - (j + 0.5) * 2.0 / n, path_points, path_point_tags, stop, ray, raw)
            nb = int(raw[m.RAW_BOUNCES])
            for seq in range(m.N_SEQS):
                for red in range(m.N_PRED):
                    want = _reference_point(seq, red, path_points, path_point_tags, nb, npts)
                    assert np.allclose(characteristic_points[j, i, m.point_slot(seq, red)], want, atol=1e-5), (i, j, seq, red)
                for red in range(m.N_SRED):
                    want = _reference_scalar(seq, red, path_points, path_point_tags, nb, npts, 60)
                    assert np.isclose(stats[j, i, m.stat_slot(seq, red)], want, atol=1e-5), (i, j, seq, red)
            checked += 1
    assert checked > 40


def test_the_ray_fraction_metrics_are_the_shares_of_the_self_and_boundary_sequences_for_every_ray():
    """ST_RAY_FRAC and ST_BOUNDARY_FRAC were once measured from a separate count of edge hits, which also counted a
    final hit at exactly perpendicular incidence that is no recorded bounce (a ray that bounced once and then ended
    that way got -1 and 2). They are now the Self and Boundary sequences' share of the bounces, like any sequence's."""
    lens = m.LensShape(1)
    lens.cx, lens.cy, lens.in_effect, lens.out_effect = -0.3, 0.3, m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_REFLECT
    poly = m.RegularPolygonShape(2)
    poly.radius, poly.cx, poly.cy = 0.4, 0.3, 0.4
    arrays = m.build_obstacle_arrays(_start_scene() + [lens, poly])
    n = 64
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((n, n), dtype=np.uint8)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, m.WORLD_RECT_DEFAULT, _stop(400), _ray(arrays), 0)
    fired = pixel_state == m.PIXEL_TRACED
    perpendicular = fired & (np.abs(stats[..., m.ST_TERMINATION] - m.TERM_STAT_VALUE[m.TERM_PERPENDICULAR]) < 0.01)
    assert perpendicular.sum() > 10, "the scene must include rays that end perpendicularly"
    count = lambda seq: np.round(stats[..., m.stat_slot(seq, m.SRED_COUNT)] * 400)
    bounces, own, edge = count(m.SEQ_BOUNCES), count(m.SEQ_SELF), count(m.SEQ_BOUNDARY)
    assert np.all((own + edge == bounces)[fired])
    safe = np.maximum(bounces, 1)
    assert np.allclose(stats[..., m.ST_RAY_FRAC][fired], (own / safe)[fired], atol=1e-6)
    assert np.allclose(stats[..., m.ST_BOUNDARY_FRAC][fired], (edge / safe)[fired], atol=1e-6)
    assert np.all(stats[..., m.ST_RAY_FRAC][fired] >= 0) and np.all(stats[..., m.ST_BOUNDARY_FRAC][fired] <= 1)
    assert np.allclose((stats[..., m.ST_RAY_FRAC] + stats[..., m.ST_BOUNDARY_FRAC])[fired & (bounces > 0)], 1.0, atol=1e-6)


def test_the_stored_layout_and_its_alias_tables_agree():
    assert m.N_POINTS == m.N_SEQS * m.N_PRED == 28 and m.N_STATS == 13 + m.N_SEQS * m.N_SRED - len(m.STAT_ALIASES) == 24
    assert m.point_slot(m.SEQ_SELF, m.PRED_TOP) == m.POINT_SELF_TOP and m.point_slot(m.SEQ_PATH, m.PRED_LAST) == m.POINT_END
    assert m.stat_slot(m.SEQ_PATH, m.SRED_LENGTH) == m.ST_LENGTH and m.stat_slot(m.SEQ_BOUNCES, m.SRED_COUNT) == m.ST_BOUNCES
    assert sorted(m.POINT_SLOT.ravel().tolist()) == list(range(m.N_POINTS))
    assert m.SEQ_PATH == 0 and m.SEQUENCE_LABELS[m.SEQ_PATH] == "Path"      # the sequence every engine offers


# --------------------------------------------------------------------------- #
# Buddha seeding: scattering from a grid of seed points laid over a zone
# --------------------------------------------------------------------------- #
def _seeding_scene():
    """A closed scene with self bounces (a circle and an ellipse), traced with a few iterations so the test is quick."""
    ellipse = m.EllipseShape(1)
    ellipse.width, ellipse.height, ellipse.rotation, ellipse.cx, ellipse.cy = 0.7, 0.45, 25.0, 0.15, -0.1
    return _stop(40), _ray(m.build_obstacle_arrays(_start_scene() + [ellipse]), seg_mult=0.9)


_NO_SLOTS = np.empty(0, dtype=np.int64)


def _scatter_seeds(n, zone, slots=(), end_terms=(), multi=(False, False, False, False), rows=None, fuzz_seed=0):
    stop, ray = _seeding_scene()
    counts = np.zeros((n, n))
    m.scatter_from_seeds(0 if rows is None else rows[0], n if rows is None else rows[1], m.WorldRect(*zone), m.WORLD_RECT_DEFAULT, stop, ray,
                         fuzz_seed,
                         np.array(slots, dtype=np.int64), np.array(end_terms, dtype=np.int64), *multi, counts)
    return counts


def test_the_zone_of_the_whole_world_seeds_exactly_the_pixels_of_an_image_covering_it():
    """The camera pixels' seeding is the zone (-1, 1, -1, 1): scatter_multi_buddha is that, and so is a seeding of the
    same options through the general kernel."""
    stop, ray = _seeding_scene()
    n = 24
    for multi in [(True, False, False, False), (False, True, False, False), (False, False, True, True)]:
        old = np.zeros((n, n))
        m.scatter_multi_buddha(0, n, m.WORLD_RECT_DEFAULT, stop, ray, 0, *multi, old)
        assert old.sum() > 0
        assert np.array_equal(old, _scatter_seeds(n, (-1.0, 1.0, -1.0, 1.0), multi=multi))


def test_seeding_the_stored_points_equals_scattering_those_stored_points_of_an_image_made_from_the_same_rays():
    """With the camera zone, a seeded scatter of the stored point of a slot is the counts of that slot's points as
    trace_rows stored them for the same pixels - so the tethered and the seeded way agree."""
    stop, ray = _seeding_scene()
    n = 24
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((n, n), dtype=np.uint8)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, m.WORLD_RECT_DEFAULT, stop, ray, 0)
    for slot in (m.POINT_ALL_MEAN, m.point_slot(m.SEQ_SELF, m.PRED_LAST), m.POINT_END):
        xs, ys = characteristic_points[..., slot, 0][pixel_state == m.PIXEL_TRACED], characteristic_points[..., slot, 1][pixel_state == m.PIXEL_TRACED]
        col = np.floor((xs + 1.0) * (n / 2.0)).astype(np.int64)
        row = np.floor((1.0 - ys) * (n / 2.0)).astype(np.int64)
        ok = (col >= 0) & (col < n) & (row >= 0) & (row < n)
        stored = np.zeros((n, n))
        np.add.at(stored, (row[ok], col[ok]), 1)
        seeded = _scatter_seeds(n, (-1.0, 1.0, -1.0, 1.0), slots=[slot])
        assert stored.sum() > 0 and np.allclose(stored, seeded), slot
    nonempty = 0
    for term in (m.TERM_CONFINED, m.TERM_PERPENDICULAR, m.TERM_ITER):
        mask = (pixel_state == m.PIXEL_TRACED) & (np.abs(stats[..., m.ST_TERMINATION] - m.TERM_STAT_VALUE[term]) < 0.01)
        xs, ys = characteristic_points[..., m.POINT_END, 0][mask], characteristic_points[..., m.POINT_END, 1][mask]
        col = np.floor((xs + 1.0) * (n / 2.0)).astype(np.int64)
        row = np.floor((1.0 - ys) * (n / 2.0)).astype(np.int64)
        ok = (col >= 0) & (col < n) & (row >= 0) & (row < n)
        stored = np.zeros((n, n))
        np.add.at(stored, (row[ok], col[ok]), 1)
        assert np.allclose(stored, _scatter_seeds(n, (-1.0, 1.0, -1.0, 1.0), end_terms=[term])), term
        nonempty += int(stored.sum() > 0)
    assert nonempty >= 2, "the scene must end rays in more than one way for this to mean anything"


def test_a_smaller_zone_has_the_same_number_of_seeds_closer_together_and_a_seed_outside_a_shape_fires_nothing():
    n = 16
    stop, ray = _seeding_scene()
    def seed_points(zone):
        h = w = n
        xs = zone[0] + (np.arange(w) + 0.5) * ((zone[1] - zone[0]) / w)
        ys = zone[3] - (np.arange(h) + 0.5) * ((zone[3] - zone[2]) / h)
        return np.meshgrid(xs, ys)
    big_x, big_y = seed_points((-1.0, 1.0, -1.0, 1.0))
    small_x, small_y = seed_points((-0.5, 0.5, -0.5, 0.5))
    assert big_x.size == small_x.size == n * n, "the zone does not change how many seeds there are"
    assert np.allclose(np.diff(small_x[0]), np.diff(big_x[0]) / 2), "they are twice as close"
    assert np.allclose(big_x[0], (np.arange(n) + 0.5) * (2.0 / n) - 1.0), "the whole world's seeds are the pixel centres"
    far = _scatter_seeds(n, (10.0, 12.0, 10.0, 12.0), multi=(True, True, True, True), slots=[m.POINT_END])
    assert far.sum() == 0, "a zone with no shape in it fires no ray"
    inner = _scatter_seeds(n, (-0.5, 0.5, -0.5, 0.5), slots=[m.POINT_START])
    assert inner.sum() == n * n, "a zone inside the shape: every seed fires, and each start point is the seed itself"
    # every start point is a seed, so each lands in its own cell of the image: with a zone twice as small
    # (quarter of the image) the starts all fall in the middle half of the counts
    assert inner[:n // 4].sum() == 0 and inner[-n // 4:].sum() == 0


def test_seeding_only_the_rows_asked_for_adds_only_those_seeds():
    n = 16
    zone = (-1.0, 1.0, -1.0, 1.0)
    whole = _scatter_seeds(n, zone, slots=[m.POINT_END], multi=(False, True, False, False))
    parts = _scatter_seeds(n, zone, slots=[m.POINT_END], multi=(False, True, False, False), rows=(0, 6)) \
        + _scatter_seeds(n, zone, slots=[m.POINT_END], multi=(False, True, False, False), rows=(6, n))
    assert np.array_equal(whole, parts)
