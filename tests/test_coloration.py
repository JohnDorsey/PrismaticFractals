"""Tests: coloration."""
import os
import tempfile
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


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
    img = app.colorize_view(_view_settings_from_config(v), stats, points, status, 4, 4,
                            m.DEFAULT_BUDDHA_BLUR, _stop(30), _ray(m.build_obstacle_arrays([])), 0)
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
    img = app.colorize_view(_view_settings_from_config(v), stats, points, status, 4, 4,
                            m.DEFAULT_BUDDHA_BLUR, _stop(30), _ray(m.build_obstacle_arrays([])), 0)
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


def test_the_buddha_blur_defaults_are_a_record_of_their_own():
    assert m.DEFAULT_BUDDHA_BLUR.radius == m.DEFAULT_BUDDHA_BLUR_RADIUS
    assert m.DEFAULT_BUDDHA_BLUR.type == m.DEFAULT_BUDDHA_BLUR_TYPE
    assert m.DEFAULT_SETTINGS[m.BuddhaBlurSettings] is m.DEFAULT_BUDDHA_BLUR
    assert m.DEFAULT_SETTINGS[m.RaySettings] is m.DEFAULT_RAY_SETTINGS


def test_loading_keyframe_settings_refreshes_the_buddha_blur():
    """apply_keyframe_settings and load_animated_setting write the blur controls like any other
    setting, so they must also trigger the (live) blur refresh."""
    import inspect
    for name in ("apply_keyframe_settings", "load_animated_setting"):
        assert "self.on_buddha_blur_changed()" in inspect.getsource(getattr(m.App, name)), name


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
    assert resolved.buddha_blur == m.BuddhaBlurSettings(radius=0.2, type=m.BUDDHA_BLUR_BOKEH)
    assert resolved.ray == app.current_ray_settings()


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
    # the entry of BUDDHA_POINT_CATALOG that is a stored point has that point's own number
    v.buddha_points[0] = _buddha_choice(m.POINT_ALL_MEAN)

    stop, ray = _stop(30), _ray(m.build_obstacle_arrays([]))

    no_blur = m.BuddhaBlurSettings(radius=0.0, type=m.BUDDHA_BLUR_BOKEH)
    img_no_blur = app.colorize_view(_view_settings_from_config(v), stats, points, status, size, size, no_blur, stop, ray, 0)
    lit_no_blur = (img_no_blur[8:13, 8:13, 0] > 0).sum()

    blur = m.BuddhaBlurSettings(radius=0.1, type=m.BUDDHA_BLUR_BOKEH)
    img_blur = app.colorize_view(_view_settings_from_config(v), stats, points, status, size, size, blur, stop, ray, 0)
    lit_blur = (img_blur[8:13, 8:13, 0] > 0).sum()

    assert lit_no_blur == 1, f"with no blur, exactly one pixel should be nonzero, got {lit_no_blur}"
    assert lit_blur > lit_no_blur, \
        f"blur should leave more pixels nonzero ({lit_blur}) than no blur ({lit_no_blur})"


def _fake_blur_refresh_app(buddha_channel=True):
    """An App stand-in with the real on_buddha_blur_changed / refresh_blur and recording stubs for
    everything they would redraw."""
    class FakeApp:
        pass
    app = FakeApp()
    app.calls, app.pending = [], []
    app.blur_job = None
    app.root = type("Root", (), {"after_idle": lambda self, fn: app.pending.append(fn) or "job"})()
    app.refresh_timeline_diffs = lambda: app.calls.append("timeline")
    app.recolor = lambda r0, r1: app.calls.append(("recolor", r0, r1))
    app.draw = lambda: app.calls.append("draw")
    app.update_preview = lambda i, j: app.calls.append(("preview", i, j))
    app.rebuild_view_list = lambda: app.calls.append("thumbnails")
    app.pinned_px = None
    app.rows_done = 37
    app.buddha_counts_blurred = ["a", "b", "c"]
    app.view_thumb_cache = {1: "stale"}
    view = m.ViewConfig(0)
    if buddha_channel:
        view.chan_src[1] = m.CHAN_SRC_BUDDHA
    app.views = [view]
    app.on_buddha_blur_changed = m.App.on_buddha_blur_changed.__get__(app)
    app.refresh_blur = m.App.refresh_blur.__get__(app)
    return app


def test_a_blur_change_recolors_the_image_once_the_slider_events_have_all_arrived():
    app = _fake_blur_refresh_app()
    for _ in range(3):
        app.on_buddha_blur_changed()                  # a slider drag: many events
    assert app.calls == ["timeline"] * 3 and len(app.pending) == 1      # one coalesced refresh
    app.pinned_px = (4, 5)
    app.pending.pop()()
    assert app.calls[3:] == [("recolor", 0, 37), "draw", ("preview", 4, 5), "thumbnails"]
    assert app.buddha_counts_blurred == [None] * 3 and app.view_thumb_cache == {}
    assert app.blur_job is None
    app.on_buddha_blur_changed()                      # and the next change schedules again
    assert len(app.pending) == 1


def test_a_blur_change_costs_nothing_visible_when_no_view_has_a_buddha_channel():
    app = _fake_blur_refresh_app(buddha_channel=False)
    app.on_buddha_blur_changed()
    app.pending.pop()()
    assert app.calls == ["timeline"]                  # no recolor, no redraw, no thumbnails...
    assert app.buddha_counts_blurred == [None] * 3 and app.view_thumb_cache == {}   # ...but no stale cache


def test_the_blurred_counts_follow_the_blur_controls_without_a_new_start():
    """The whole chain: counts blurred with one setting are recomputed with the next as soon as
    the blur is refreshed - from the same stored counts, no simulation involved."""
    app = _fake_current_image_app()
    app.live_view = lambda: app.views[0]
    app.views[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    counts = np.zeros((m.SIZE, m.SIZE), dtype=np.float64)
    counts[100, 100] = 50.0
    app.buddha_counts = [counts, None, None]
    app.buddha_counts_blurred = [None, None, None]
    current = [m.BuddhaBlurSettings(radius=0.0, type=m.BUDDHA_BLUR_BOKEH)]
    app.current_blur_settings = lambda: current[0]
    app.blurred_buddha_counts = m.App.blurred_buddha_counts.__get__(app)
    app.recolor = lambda r0, r1: None
    app.draw = lambda: None
    app.rebuild_view_list = lambda: None
    app.pinned_px = None
    app.rows_done = m.SIZE
    app.refresh_blur = m.App.refresh_blur.__get__(app)
    sharp = app.blurred_buddha_counts(0)
    assert np.array_equal(sharp, counts)
    current[0] = m.BuddhaBlurSettings(radius=0.05, type=m.BUDDHA_BLUR_BOKEH)
    assert app.blurred_buddha_counts(0) is sharp          # (stale until refresh_blur runs)
    app.refresh_blur()
    soft = app.blurred_buddha_counts(0)
    assert (soft > 0).sum() > 1 and abs(soft.sum() - counts.sum()) < 1e-6 * counts.sum() + 1e-9
    assert not np.array_equal(soft, sharp)


def test_blurred_buddha_counts_is_cached_until_invalidated():
    """The mechanism that keeps the live preview cheap: blurring is a whole-
    image pass, so it must not be recomputed on every recolor - only when
    the raw counts actually changed."""
    app = _fake_current_image_app()
    app.live_view = lambda: app.views[0]
    app.views[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.buddha_counts = [np.zeros((m.SIZE, m.SIZE), dtype=np.float64), None, None]
    app.buddha_counts_blurred = [None, None, None]
    app.current_blur_settings = lambda: m.BuddhaBlurSettings(radius=0.05, type=m.BUDDHA_BLUR_BOKEH)
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
    stop, ray = _stop(30), _ray(m.build_obstacle_arrays([]))
    blur = m.BuddhaBlurSettings(radius=0.15, type=m.BUDDHA_BLUR_GAUSSIAN)     # a large, expensive-looking radius
    t0 = time.time()
    app.colorize_view(_view_settings_from_config(v), stats, points, status, size, size, blur, stop, ray, 0)
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
    assert m.POINT_LABELS.index("Origin") == m.POINT_ORIGIN == m.N_POINTS
    assert m.POINT_LABELS.index("Top left corner") == m.POINT_TOP_LEFT_CORNER == m.N_POINTS + 1
    assert m.POINT_LABELS.index("Path: First point") == m.POINT_START == 0
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
    fresh.selfchecks.POINT_MENU_ORDER = fresh.POINT_MENU_ORDER[:-1]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "POINT_MENU_ORDER" in str(e)


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
    app.update_buttons = lambda: None
    app.pinned_px = None
    rebuild_calls = []
    app.rebuild_view_list = lambda: rebuild_calls.append(1)
    m.App.on_finished(app)
    assert len(rebuild_calls) == 1


# --------------------------------------------------------------------------- #
# Buddha coloration catalog
# --------------------------------------------------------------------------- #
def test_buddha_catalog_lengths_agree():
    assert len(m.DEFAULT_BUDDHA_POINTS) == m.N_BUDDHA_POINTS
    assert len(m.BUDDHA_POINT_LABELS) == m.N_BUDDHA_POINTS


def test_buddha_confined_channel_present():
    assert any("confined" in label.lower() for label in m.BUDDHA_POINT_LABELS)


def test_every_metric_and_point_a_dropdown_can_choose_is_a_plain_int():
    """Regression: OTHER_STATS was built from the numpy slot table, so choosing a sequence's metric from the
    dropdown stored a numpy integer, which the strict ChannelSettings record (and so saving a project) refuses."""
    for table in (m.OTHER_STATS, m.POINT_STATS, m.POINT_MENU_ORDER):
        assert all(type(value) is int for value in table), table
    for label in m.OTHER_LABELS:
        assert type(m.other_label_to_stat(label)) is int, label
    v = m.ViewConfig(0)
    v.kinds[1] = m.KIND_OTHER
    v.other_metric[1] = m.other_label_to_stat("Boundary bounces: Length (arctan)")
    v.point1_choice[0] = m.POINT_LABELS.index("Self bounces: Last point")
    channels = [m.view_channel_value(v, c) for c in range(3)]
    assert channels[1].stat == m.stat_slot(m.SEQ_BOUNDARY, m.SRED_LENGTH)
    assert m._channel_to_json(channels[1]) == {"source": "query", "kind": "other_metric", "metric": "boundary_bounces_length"}
    assert m._channel_to_json(channels[0])["point1"] == "self_bounces_last"


def test_every_choice_a_view_can_make_has_its_own_label_and_a_working_key():
    assert len(set(m.STAT_LABELS)) == m.N_STATS and len(set(m.POINT_LABELS)) == m.N_POINT_CHOICES
    assert len(set(m.BUDDHA_POINT_LABELS)) == m.N_BUDDHA_POINTS and len(set(m.STAT_SHORT)) == m.N_STATS
    for stat in range(m.N_STATS):
        assert m.CAT_METRIC.by_key[m.CAT_METRIC.key(stat)] == stat
    for point in range(m.N_POINT_CHOICES):
        assert m.CAT_POINT.by_key[m.CAT_POINT.key(point)] == point
    for index in range(m.N_BUDDHA_POINTS):
        assert m.CAT_BUDDHA_POINT.by_key[m.CAT_BUDDHA_POINT.key(index)] == index
    assert m.POINT_LABELS[m.point_slot(m.SEQ_SELF, m.PRED_TOP)] == "Self bounces: Uppermost point"
    assert m.STAT_LABELS[m.stat_slot(m.SEQ_BOUNDARY, m.SRED_SHARE)] == "Boundary bounces: Share of the bounces"
    assert m.STAT_LABELS[m.ST_TERMINATION] == "Termination reason"          # a measurement of the run, written out by hand
    slot = m.point_slot(m.SEQ_SELF, m.PRED_LAST)
    assert m.BUDDHA_POINT_CATALOG[slot][1:] == (m.BUDDHA_SIMPLE, slot, None)
