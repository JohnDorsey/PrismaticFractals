"""Tests: coloration."""
import os
import tempfile
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# Fixed (non-ray-dependent) point choices: Origin and (-1,1)
# --------------------------------------------------------------------------- #
def test_extract_point_xy_returns_constants_for_the_fixed_points():
    n = 5
    x, y = m.extract_point_xy(None, m.CHOICE_ORIGIN, None, n)
    assert len(x) == n and (x == 0.0).all() and (y == 0.0).all()
    x, y = m.extract_point_xy(None, m.CHOICE_POINT_NEG1_POS1, None, n)
    assert (x == -1.0).all() and (y == 1.0).all()


def test_extract_point_xy_still_reads_real_points_from_the_array():
    characteristic_points = np.zeros((3, 3, m.N_POINTS, 2))
    characteristic_points[..., m.POINT_START, :] = [0.5, 0.25]
    computed = np.ones((3, 3), dtype=bool)
    x, y = m.extract_point_xy(characteristic_points, m.POINT_START, computed, 9)
    assert x[0] == 0.5 and y[0] == 0.25


def test_resolve_point_xy_handles_both_constant_and_real_points():
    assert m.resolve_point_xy(np.zeros((m.N_POINTS, 2)), m.CHOICE_ORIGIN) == (0.0, 0.0)
    assert m.resolve_point_xy(np.zeros((m.N_POINTS, 2)), m.CHOICE_POINT_NEG1_POS1) == (-1.0, 1.0)
    ray_characteristic_points = np.zeros((m.N_POINTS, 2))
    ray_characteristic_points[m.POINT_END] = [0.3, -0.4]
    assert m.resolve_point_xy(ray_characteristic_points, m.POINT_END) == (0.3, -0.4)


def test_point_lists_include_the_three_fixed_points():
    assert m.POINT_LABELS[m.CHOICE_ORIGIN] == "Origin"
    assert m.POINT_LABELS[m.CHOICE_POINT_NEG1_POS1] == "(-1,1)"
    assert m.POINT_LABELS[m.CHOICE_TOP_LEFT_POINT] == "Top left point"
    assert m.CONSTANT_POINT_COORDS == {m.CHOICE_ORIGIN: (0.0, 0.0), m.CHOICE_POINT_NEG1_POS1: (-1.0, 1.0),
                                       m.CHOICE_TOP_LEFT_POINT: (-2.0, 2.0)}
    assert len(m.POINT_LABELS) == len(m.POINT_SHORT) == len(m.POINT_DESCRIPTIONS)
    assert len(m.POINT_LABELS) == m.N_POINT_CHOICES == m.N_POINTS + 3


def test_apply_coloration_produces_uniform_color_for_two_constant_points():
    """The real, end-to-end proof: a channel comparing two fixed points must
    give the SAME value everywhere, regardless of where any ray went -
    since the metric no longer depends on the ray at all."""
    app = _fake_render_app()
    stats = np.zeros((4, 4, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((4, 4, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.full((4, 4), m.PIXEL_TRACED, dtype=np.uint8)
    v = m.ColorationConfig(0)                  # mutable; converted for apply_coloration below
    v.point1_choice[0] = m.CHOICE_ORIGIN
    v.point2_choice[0] = m.CHOICE_POINT_NEG1_POS1
    v.point_metric[0] = m.ST_HDIST
    img = app.apply_coloration(_coloration_settings_from_config(v), stats, characteristic_points, pixel_state, 4, 4,
                            m.DEFAULT_BUDDHA_BLUR, _termination(30), _engine_args(m.build_obstacle_arrays([])), 0)
    assert (img[..., 0] == img[0, 0, 0]).all()
    assert 62 <= img[0, 0, 0] <= 65   # H dist(origin, (-1,1)) = 1, divided by DISTANCE_METRIC_DIVISOR 4 -> 0.25 -> ~63.75
    v.point2_choice[0] = m.CHOICE_TOP_LEFT_POINT
    img = app.apply_coloration(_coloration_settings_from_config(v), stats, characteristic_points, pixel_state, 4, 4,
                            m.DEFAULT_BUDDHA_BLUR, _termination(30), _engine_args(m.build_obstacle_arrays([])), 0)
    assert 126 <= img[0, 0, 0] <= 129   # H dist(origin, (-2,2)) = 2 -> 0.5 -> ~127.5


def test_coloration_info_lines_shows_the_new_point_names():
    v = m.ColorationConfig(0)
    v.point1_choice[0] = m.CHOICE_ORIGIN
    v.point2_choice[0] = m.CHOICE_POINT_NEG1_POS1
    lines = m.coloration_info_lines(v)
    assert any("Origin" in l and "(-1,1)" in l for l in lines)


# --------------------------------------------------------------------------- #
# "Constant" channel source - a dedicated ColorationConfig field (constant_choice),
# never packed into a slot whose meaning depends on context
# --------------------------------------------------------------------------- #
def test_other_labels_no_longer_offer_the_constant_options():
    assert "None (always 0)" not in m.OTHER_LABELS
    assert "None (always 1)" not in m.OTHER_LABELS
    assert len(m.OTHER_LABELS) == len(m.OTHER_METRICS)


def test_chan_src_labels_include_constant():
    assert m.CHAN_SRC_LABELS[m.CHAN_SRC_CONSTANT] == "Constant"
    assert len(m.CHAN_SRC_LABELS) == 3


def test_constant_label_metric_round_trip():
    assert m.constant_label_to_metric("Always 0") == m.NONE_ZERO
    assert m.constant_label_to_metric("Always 1") == m.NONE_ONE
    assert m.constant_metric_to_label(m.NONE_ZERO) == "Always 0"
    assert m.constant_metric_to_label(m.NONE_ONE) == "Always 1"


def test_coloration_config_has_a_dedicated_constant_choice_field():
    v = m.ColorationConfig(0)
    assert v.constant_choice == list(m.DEFAULT_CONSTANT_CHOICES)
    # setting it must never touch other_metric - they are fully independent
    v.constant_choice[0] = m.NONE_ONE
    assert v.other_metric[0] == m.DEFAULT_OTHER_METRICS[0]


def test_make_mapping_raises_if_constant_channel_has_no_constant_choices():
    """The specific safety property requested: a mistake here must crash,
    not silently misread some other slot."""
    try:
        m.make_mapping([m.METRIC_TYPE_OTHER] * 3, [0] * 3, [0] * 3,
                       chan_src=[m.CHAN_SRC_CONSTANT] * 3)
        assert False, "should have raised"
    except ValueError:
        pass


def test_make_mapping_resolves_constant_channels_from_constant_choices():
    metric_types = [m.METRIC_TYPE_OTHER] * 3
    point_metrics = [m.ST_HDIST] * 3
    other_metrics = [m.ST_BOUNCE_COUNT, m.ST_BOUNCE_COUNT, m.ST_BOUNCE_COUNT]   # never consulted
    chan_src = [m.CHAN_SRC_CONSTANT, m.CHAN_SRC_CONSTANT, m.CHAN_SRC_QUERY]
    constant_choices = [m.NONE_ZERO, m.NONE_ONE, m.NONE_ZERO]    # [2] unused (Query)
    mapping = m.make_mapping(metric_types, point_metrics, other_metrics, chan_src, constant_choices)
    assert mapping == (m.NONE_ZERO, m.NONE_ONE, m.ST_BOUNCE_COUNT)


def test_coloration_channel_value_and_decompose_chans_round_trip_constant_choice():
    v = m.ColorationConfig(0)
    v.chan_src[1] = m.CHAN_SRC_CONSTANT
    v.constant_choice[1] = m.NONE_ONE
    chans = [m.coloration_channel_value(v, c) for c in range(3)]
    assert isinstance(chans[1], m.ChannelSettings) and chans[1].constant == m.NONE_ONE
    vs = m.decompose_chans(v.color_model, chans)
    assert vs.constant_choice[1] == m.NONE_ONE
    assert vs.constant_choice[0] == m.DEFAULT_CONSTANT_CHOICES[0]   # untouched channel


def test_apply_coloration_renders_constant_channels_correctly():
    """End-to-end: Always-0 and Always-1 channels render as pure 0/255
    everywhere, regardless of the actual simulation data."""
    app = _fake_render_app()
    stats = np.random.RandomState(0).rand(4, 4, m.N_STATS).astype(np.float32)
    characteristic_points = np.zeros((4, 4, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.full((4, 4), m.PIXEL_TRACED, dtype=np.uint8)
    v = m.ColorationConfig(0)                  # mutable; converted for apply_coloration below
    v.chan_src[0] = m.CHAN_SRC_CONSTANT
    v.constant_choice[0] = m.NONE_ONE
    v.chan_src[1] = m.CHAN_SRC_CONSTANT
    v.constant_choice[1] = m.NONE_ZERO
    img = app.apply_coloration(_coloration_settings_from_config(v), stats, characteristic_points, pixel_state, 4, 4,
                            m.DEFAULT_BUDDHA_BLUR, _termination(30), _engine_args(m.build_obstacle_arrays([])), 0)
    assert (img[..., 0] == 255).all()
    assert (img[..., 1] == 0).all()


def test_coloration_info_lines_describes_a_constant_channel():
    v = m.ColorationConfig(0)
    v.chan_src[2] = m.CHAN_SRC_CONSTANT
    v.constant_choice[2] = m.NONE_ONE
    lines = m.coloration_info_lines(v)
    assert "B:Constant" in lines
    assert "  Always 1" in lines
    for line in lines:
        m.render_text_line(line, scale=1)   # every line must actually render


def test_coloration_info_lines_raises_if_the_metric_is_a_constant_sentinel_outside_constant_source():
    """The other half of the safety property: if other_metric somehow ends
    up holding a Constant-only sentinel while chan_src is Query, that must
    crash loudly rather than render as if it were a real metric."""
    v = m.ColorationConfig(0)
    v.metric_types[0] = m.METRIC_TYPE_OTHER
    v.other_metric[0] = m.NONE_ZERO      # chan_src[0] is still CHAN_SRC_QUERY
    try:
        m.coloration_info_lines(v)
        assert False, "should have raised"
    except ValueError:
        pass


def test_short_value_text_describes_a_constant_channel():
    v = m.ColorationConfig(0)
    v.chan_src[0] = m.CHAN_SRC_CONSTANT
    v.constant_choice[0] = m.NONE_ZERO
    val = m.coloration_channel_value(v, 0)
    text = m.short_value_text("coloration:0:chan0", val)
    assert text == "Const:0"


def test_short_value_text_raises_if_the_metric_is_a_constant_sentinel_outside_constant_source():
    v = m.ColorationConfig(0)
    v.metric_types[0] = m.METRIC_TYPE_OTHER
    v.other_metric[0] = m.NONE_ONE       # chan_src[0] is still CHAN_SRC_QUERY
    val = m.coloration_channel_value(v, 0)
    try:
        m.short_value_text("coloration:0:chan0", val)
        assert False, "should have raised"
    except ValueError:
        pass


def test_on_constant_changed_updates_the_dedicated_field_only():
    app = _fake_color_editor_app()
    coloration = app.selected_coloration_obj()
    coloration.chan_src[0] = m.CHAN_SRC_CONSTANT
    original_other_metric = coloration.other_metric[0]
    app.constant_vars[0].set("Always 1")
    app.on_constant_changed = m.App.on_constant_changed.__get__(app)
    app.on_constant_changed(0)
    assert coloration.constant_choice[0] == m.NONE_ONE
    assert coloration.other_metric[0] == original_other_metric, \
        "on_constant_changed must never touch other_metric"


def test_refresh_row_shows_constant_frame_only_for_constant_source():
    app = _fake_color_editor_app()
    coloration = app.selected_coloration_obj()
    coloration.chan_src[0] = m.CHAN_SRC_CONSTANT
    app.refresh_row(0)
    assert app.constant_frames[0].shown
    assert not app.query_frames[0].shown
    assert not app.buddha_frames[0].shown
    coloration.chan_src[0] = m.CHAN_SRC_QUERY
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
    """apply_setting_values (what selecting a keyframe loads through) and load_animated_setting write the blur controls like any other
    setting, so they must also trigger the (live) blur refresh."""
    import inspect
    for name in ("apply_setting_values", "load_animated_setting"):
        assert "self.on_buddha_blur_changed()" in inspect.getsource(getattr(m.App, name)), name


def test_capture_and_resolve_round_trip_buddha_blur_settings():
    app = _make_fake_app_for_snapshot()
    app.buddha_blur_radius_var.set("0.2")
    app.buddha_blur_type_var.set(m.BUDDHA_BLUR_LABELS[m.BUDDHA_BLUR_BOKEH])
    app.deco = {"outline": _StubVar(True)}
    app.obstacles = []
    app.colorations = []
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    data = app.capture_all_settings()
    assert data["buddha_blur_radius"] == 0.2
    assert data["buddha_blur_type"] == m.BUDDHA_BLUR_BOKEH

    kf = m.Keyframe(0, 0, data)
    resolved = m.resolve_frame_settings([kf], 0, m.DEFAULT_SETTING_VALUES)
    assert resolved.buddha_blur == m.BuddhaBlurSettings(radius=0.2, type=m.BUDDHA_BLUR_BOKEH)
    assert resolved.ray == app.current_ray_settings()


def test_apply_coloration_spreads_a_buddha_point_with_blur_enabled():
    """End-to-end: a single ray's Buddha point, with blur on, must leave a
    nonzero count on more of its (fired) neighboring pixels than with blur
    off. Two things this test design has to account for, both found by
    running it and checking the actual numbers rather than guessing:
    apply_coloration only colors pixels that are themselves "computed" (fired
    their own ray), so the neighborhood around the scattered point must
    also be marked as fired for the spread to be visible at all - and every
    computed pixel gets assigned a value (even exactly 0), overwriting the
    background, so "differs from background" is the wrong signal; "nonzero"
    is the right one."""
    app = _fake_render_app()
    size = 20
    stats = np.zeros((size, size, m.N_STATS), dtype=np.float32)
    characteristic_points = np.full((size, size, m.N_POINTS, 2), 100.0, dtype=np.float32)   # far outside
                                                                             # the [-1,1] grid
    pixel_state = np.zeros((size, size), dtype=np.uint8)
    pixel_state[8:13, 8:13] = m.PIXEL_TRACED                          # a small fired neighborhood
    characteristic_points[10, 10, m.POINT_ALL_MEAN] = [0.0, 0.0]   # only this one actually lands in-frame

    v = m.ColorationConfig(0)                  # mutable; converted for apply_coloration below
    v.chan_src[0] = m.CHAN_SRC_BUDDHA
    # the entry of BUDDHA_POINT_CATALOG that is a stored point has that point's own number
    v.buddha_points[0] = _buddha_choice(m.POINT_ALL_MEAN)

    termination, engine_args = _termination(30), _engine_args(m.build_obstacle_arrays([]))

    no_blur = m.BuddhaBlurSettings(radius=0.0, type=m.BUDDHA_BLUR_BOKEH)
    img_no_blur = app.apply_coloration(_coloration_settings_from_config(v), stats, characteristic_points, pixel_state, size, size, no_blur, termination, engine_args, 0)
    lit_no_blur = (img_no_blur[8:13, 8:13, 0] > 0).sum()

    blur = m.BuddhaBlurSettings(radius=0.1, type=m.BUDDHA_BLUR_BOKEH)
    img_blur = app.apply_coloration(_coloration_settings_from_config(v), stats, characteristic_points, pixel_state, size, size, blur, termination, engine_args, 0)
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
    _give_image_geometry(app)
    app.calls, app.pending = [], []
    app.blur_job = None
    app.root = type("Root", (), {"after_idle": lambda self, fn: app.pending.append(fn) or "job"})()
    app.refresh_timeline_diffs = lambda: app.calls.append("timeline")
    app.recolor = lambda r0, r1: app.calls.append(("recolor", r0, r1))
    app.draw = lambda: app.calls.append("draw")
    app.update_preview = lambda i, j: app.calls.append(("preview", i, j))
    app.rebuild_coloration_list = lambda: app.calls.append("thumbnails")
    app.pinned_px = None
    app.rows_done = 37
    app.buddha_counts_blurred = ["a", "b", "c"]
    app.coloration_thumb_cache = {1: "stale"}
    coloration = m.ColorationConfig(0)
    if buddha_channel:
        coloration.chan_src[1] = m.CHAN_SRC_BUDDHA
    app.colorations = [coloration]
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
    assert app.buddha_counts_blurred == [None] * 3 and app.coloration_thumb_cache == {}
    assert app.blur_job is None
    app.on_buddha_blur_changed()                      # and the next change schedules again
    assert len(app.pending) == 1


def test_a_blur_change_costs_nothing_visible_when_no_coloration_has_a_buddha_channel():
    app = _fake_blur_refresh_app(buddha_channel=False)
    app.on_buddha_blur_changed()
    app.pending.pop()()
    assert app.calls == ["timeline"]                  # no recolor, no redraw, no thumbnails...
    assert app.buddha_counts_blurred == [None] * 3 and app.coloration_thumb_cache == {}   # ...but no stale cache


def test_the_blurred_counts_follow_the_blur_controls_without_a_new_start():
    """The whole chain: counts blurred with one setting are recomputed with the next as soon as
    the blur is refreshed - from the same stored counts, no simulation involved."""
    app = _fake_current_image_app()
    app.live_coloration = lambda: app.colorations[0]
    app.colorations[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    counts = np.zeros((m.PREVIEW_BOX, m.PREVIEW_BOX), dtype=np.float64)
    counts[100, 100] = 50.0
    app.buddha_counts = [counts, None, None]
    app.buddha_counts_blurred = [None, None, None]
    current = [m.BuddhaBlurSettings(radius=0.0, type=m.BUDDHA_BLUR_BOKEH)]
    app.current_blur_settings = lambda: current[0]
    app.current_buddha_seed_settings = lambda: m.DEFAULT_BUDDHA_SEED
    app.current_engine_settings = lambda: m.DEFAULT_ENGINE_SETTINGS
    app.current_mandelbrot_settings = lambda: m.DEFAULT_MANDELBROT_SETTINGS
    app.mandelbrot_defines_figure = m.App.mandelbrot_defines_figure.__get__(app)
    app.blurred_buddha_counts = m.App.blurred_buddha_counts.__get__(app)
    app.recolor = lambda r0, r1: None
    app.draw = lambda: None
    app.rebuild_coloration_list = lambda: None
    app.pinned_px = None
    app.rows_done = m.PREVIEW_BOX
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
    app.live_coloration = lambda: app.colorations[0]
    app.colorations[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.buddha_counts = [np.zeros((m.PREVIEW_BOX, m.PREVIEW_BOX), dtype=np.float64), None, None]
    app.buddha_counts_blurred = [None, None, None]
    app.current_blur_settings = lambda: m.BuddhaBlurSettings(radius=0.05, type=m.BUDDHA_BLUR_BOKEH)
    app.current_buddha_seed_settings = lambda: m.DEFAULT_BUDDHA_SEED
    app.current_engine_settings = lambda: m.DEFAULT_ENGINE_SETTINGS
    app.current_mandelbrot_settings = lambda: m.DEFAULT_MANDELBROT_SETTINGS
    app.mandelbrot_defines_figure = m.App.mandelbrot_defines_figure.__get__(app)
    app.blurred_buddha_counts = m.App.blurred_buddha_counts.__get__(app)

    first = app.blurred_buddha_counts(0)
    second = app.blurred_buddha_counts(0)
    assert first is second, "must return the cached array, not recompute it"

    app.buddha_counts_blurred[0] = None      # simulate scatter_buddha_rows invalidating it
    third = app.blurred_buddha_counts(0)
    assert third is not first, "must recompute once actually invalidated"


def test_scatter_buddha_rows_invalidates_only_touched_channels():
    app = _fake_current_image_app()
    app.live_coloration = lambda: app.colorations[0]
    coloration = app.colorations[0]
    coloration.chan_src[0] = m.CHAN_SRC_BUDDHA
    coloration.buddha_points[0] = tuple(i == 0 for i in range(m.N_BUDDHA_POINTS))
    coloration.chan_src[1] = m.CHAN_SRC_BUDDHA
    coloration.buddha_points[1] = tuple(False for _ in range(m.N_BUDDHA_POINTS))   # nothing selected
    app.buddha_counts = [np.zeros((m.PREVIEW_BOX, m.PREVIEW_BOX), dtype=np.float64),
                         np.zeros((m.PREVIEW_BOX, m.PREVIEW_BOX), dtype=np.float64), None]
    app.buddha_counts_blurred = ["sentinel0", "sentinel1", None]
    app.pixel_state = np.full((m.PREVIEW_BOX, m.PREVIEW_BOX), m.PIXEL_TRACED, dtype=np.uint8)
    app.characteristic_points = np.zeros((m.PREVIEW_BOX, m.PREVIEW_BOX, m.N_POINTS, 2), dtype=np.float32)
    app.stats = np.zeros((m.PREVIEW_BOX, m.PREVIEW_BOX, m.N_STATS), dtype=np.float32)
    app.scatter_buddha_rows = m.App.scatter_buddha_rows.__get__(app)
    app.scatter_buddha_rows(0, m.PREVIEW_BOX)
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
    characteristic_points = np.random.RandomState(0).rand(size, size, m.N_POINTS, 2).astype(np.float32) * 2 - 1
    pixel_state = np.full((size, size), m.PIXEL_TRACED, dtype=np.uint8)   # every pixel fires
    v = m.ColorationConfig(0)                  # mutable; converted for apply_coloration below
    v.chan_src[0] = m.CHAN_SRC_BUDDHA
    v.buddha_points[0] = tuple(i == 0 for i in range(m.N_BUDDHA_POINTS))
    termination, engine_args = _termination(30), _engine_args(m.build_obstacle_arrays([]))
    blur = m.BuddhaBlurSettings(radius=0.15, type=m.BUDDHA_BLUR_GAUSSIAN)     # a large, expensive-looking radius
    t0 = time.time()
    app.apply_coloration(_coloration_settings_from_config(v), stats, characteristic_points, pixel_state, size, size, blur, termination, engine_args, 0)
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
# "Origin" and "(-1,1)" head the point dropdowns
# --------------------------------------------------------------------------- #
def test_the_point_dropdowns_list_the_three_fixed_points_first():
    assert m.POINT_MENU_LABELS[:3] == ["Origin", "Top left point", "(-1,1)"]
    assert m.POINT_MENU_ORDER[:3] == [m.CHOICE_ORIGIN, m.CHOICE_TOP_LEFT_POINT, m.CHOICE_POINT_NEG1_POS1]


def test_the_point_dropdowns_still_list_every_choice_exactly_once():
    assert sorted(m.POINT_MENU_ORDER) == list(range(m.N_POINT_CHOICES))
    assert sorted(m.POINT_MENU_LABELS) == sorted(m.POINT_LABELS)
    assert len(set(m.POINT_MENU_LABELS)) == m.N_POINT_CHOICES


def test_choosing_from_the_reordered_dropdown_still_gives_the_same_point_ids():
    """Only the display order changed: a label maps back to its unchanged id, so
    the ids stored in colorations, keyframes and the sample arrays are untouched."""
    assert m.POINT_LABELS.index("Origin") == m.CHOICE_ORIGIN == m.N_POINTS
    assert m.POINT_LABELS.index("(-1,1)") == m.CHOICE_POINT_NEG1_POS1 == m.N_POINTS + 1
    assert m.POINT_LABELS.index("Top left point") == m.CHOICE_TOP_LEFT_POINT == m.N_POINTS + 2
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


def test_selecting_a_coloration_makes_it_the_live_one():
    """The core simplification: there's no separate "focus" step anymore -
    selecting a coloration for editing is what makes it the one shown in the
    preview."""
    app = _fake_color_editor_app()
    assert app.live_coloration().id == 0
    app.select_coloration(1)
    assert app.live_coloration().id == 1
    assert app.live_coloration() is app.selected_coloration_obj()


def test_editing_the_selected_coloration_triggers_the_recolor_cascade():
    app = _fake_color_editor_app()
    app.color_model_var.set("HSV")
    app.on_color_model_changed()
    assert app.colorations[0].color_model == "HSV"
    assert len(app.recolor_calls) == 1
    assert len(app.stat_tag_calls) == 1


# --------------------------------------------------------------------------- #
# Performance fix: a Buddha rescatter (which can involve re-tracing every
# fired ray for a "multi" point option) must only happen when the specific
# thing that changed could actually affect Buddha data - never for an
# unrelated channel's edit, and never for a color model change
# --------------------------------------------------------------------------- #
def test_apply_mapping_change_skips_buddha_rebuild_for_unrelated_channel():
    """The core performance fix: editing a channel that isn't Buddha must
    not trigger a full Buddha rescatter, even if some OTHER channel on the
    same coloration IS Buddha (whose rescatter could be very expensive for a
    "multi" point option like All self bounces)."""
    app = _fake_color_editor_app()
    app.colorations[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.apply_mapping_change(1)   # channel 1 is plain Query, untouched
    assert app.rebuild_buddha_counts_calls == []


def test_apply_mapping_change_rebuilds_when_the_edited_channel_is_buddha():
    app = _fake_color_editor_app()
    app.colorations[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.apply_mapping_change(0)
    assert len(app.rebuild_buddha_counts_calls) == 1


def test_load_value_skips_buddha_rebuild_for_mode_field():
    """Mode (RGB<->HSV) never affects Buddha scattering at all - matches
    on_color_model_changed's own behavior, which never calls this either."""
    app = _fake_color_editor_app()
    app.colorations[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.load_value = m.App.load_value.__get__(app)
    app.load_value("coloration:0:color_model", "HSV")
    assert app.rebuild_buddha_counts_calls == []


def test_load_value_skips_buddha_rebuild_for_unrelated_channel_field():
    app = _fake_color_editor_app()
    app.colorations[0].chan_src[0] = m.CHAN_SRC_BUDDHA
    app.load_value = m.App.load_value.__get__(app)
    chan1_value = m.ChannelSettings(
        chan_src=m.CHAN_SRC_QUERY, metric_type=m.METRIC_TYPE_POINT_PAIR, metric_slot=m.ST_HDIST, point1=m.POINT_START,
        point2=m.POINT_END, buddha_points=m.DEFAULT_BUDDHA_POINTS, buddha_path_kinds=m.ALL_PATH_KINDS,
        buddha_compression=m.BUDDHA_COMPRESS_ARCTAN, buddha_brightness=0, constant=m.NONE_ZERO)
    app.load_value("coloration:0:chan1", chan1_value)
    assert app.rebuild_buddha_counts_calls == []


def test_load_value_rebuilds_when_the_loaded_channel_is_buddha():
    app = _fake_color_editor_app()
    app.load_value = m.App.load_value.__get__(app)
    chan0_value = m.ChannelSettings(
        chan_src=m.CHAN_SRC_BUDDHA, metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_BOUNCE_COUNT, point1=m.POINT_START,
        point2=m.POINT_END, buddha_points=m.DEFAULT_BUDDHA_POINTS, buddha_path_kinds=m.ALL_PATH_KINDS,
        buddha_compression=m.BUDDHA_COMPRESS_ARCTAN, buddha_brightness=0, constant=m.NONE_ZERO)
    app.load_value("coloration:0:chan0", chan0_value)
    assert len(app.rebuild_buddha_counts_calls) == 1


def test_delete_coloration_recolors_the_preview_to_match_the_new_selection():
    app = _fake_coloration_list_app()
    app.new_coloration()                     # ids: [0, 1]; 1 is now selected/live
    app.select_coloration(0)
    app.mapping = m.make_mapping(m.DEFAULT_METRIC_TYPES, m.DEFAULT_POINT_METRICS,
                                 m.DEFAULT_OTHER_METRICS, m.DEFAULT_CHAN_SRCS)
    recolor_calls = []
    app.recolor = lambda r0, r1: recolor_calls.append((r0, r1))
    app.delete_coloration()
    assert app.live_coloration().id == 1
    assert len(recolor_calls) == 1, \
        "deleting the selected (live) coloration must recolor the preview to match the new one"


def test_on_finished_refreshes_the_colorations_list():
    """The thumbnails must reflect a just-finished generation even if the
    Colorations tab isn't the active notebook tab - rebuild_coloration_list is a
    plain method call, not gated on tab visibility."""
    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    app.tick_id = "irrelevant"
    app.update_buttons = lambda: None
    app.pinned_px = None
    rebuild_calls = []
    app.rebuild_coloration_list = lambda: rebuild_calls.append(1)
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
    """Regression: OTHER_METRICS was built from the numpy slot table, so choosing a sequence's metric from the
    dropdown stored a numpy integer, which the strict ChannelSettings record (and so saving a project) refuses."""
    for table in (m.OTHER_METRICS, m.POINT_METRICS, m.POINT_MENU_ORDER):
        assert all(type(value) is int for value in table), table
    for label in m.OTHER_LABELS:
        assert type(m.other_label_to_metric(label)) is int, label
    v = m.ColorationConfig(0)
    v.metric_types[1] = m.METRIC_TYPE_OTHER
    v.other_metric[1] = m.other_label_to_metric("Boundary bounces: Length (arctan)")
    v.point1_choice[0] = m.POINT_LABELS.index("Self bounces: Last point")
    channels = [m.coloration_channel_value(v, c) for c in range(3)]
    assert channels[1].metric_slot == m.stat_slot(m.SEQ_BOUNDARY, m.SRED_LENGTH)
    assert m._channel_to_json(channels[1]) == {"source": "query", "kind": "other_metric", "metric": "boundary_bounces_length"}
    assert m._channel_to_json(channels[0])["point1"] == "self_bounces_last"


def test_every_choice_a_coloration_can_make_has_its_own_label_and_a_working_key():
    assert len(set(m.METRIC_LABELS)) == m.N_STATS and len(set(m.POINT_LABELS)) == m.N_POINT_CHOICES
    assert len(set(m.BUDDHA_POINT_LABELS)) == m.N_BUDDHA_POINTS and len(set(m.METRIC_SHORT)) == m.N_STATS
    for metric_slot in range(m.N_STATS):
        assert m.CAT_METRIC.by_key[m.CAT_METRIC.key(metric_slot)] == metric_slot
    for point in range(m.N_POINT_CHOICES):
        assert m.CAT_POINT.by_key[m.CAT_POINT.key(point)] == point
    for index in range(m.N_BUDDHA_POINTS):
        assert m.CAT_BUDDHA_POINT.by_key[m.CAT_BUDDHA_POINT.key(index)] == index
    assert m.POINT_LABELS[m.point_slot(m.SEQ_SELF, m.PRED_TOP)] == "Self bounces: Uppermost point"
    assert m.METRIC_LABELS[m.stat_slot(m.SEQ_BOUNDARY, m.SRED_SHARE)] == "Boundary bounces: Share of the bounces"
    assert m.METRIC_LABELS[m.ST_TERMINATION] == "Termination reason"          # a measurement of the run, written out by hand
    slot = m.point_slot(m.SEQ_SELF, m.PRED_LAST)
    assert m.BUDDHA_POINT_CATALOG[slot][1:] == (m.BUDDHA_SIMPLE, slot, None)


# --------------------------------------------------------------------------- #
# The Buddha seeding tab: greying, the zone's entries, the outline on the preview, and the image
# --------------------------------------------------------------------------- #
class _StateWidget:
    def __init__(self):
        self.states = []

    def state(self, spec):
        self.states.append(list(spec))


def _fake_seeding_app(camera=True, **zone):
    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    defaults = dict(x_start=-1.0, x_stop=1.0, y_start=-1.0, y_stop=1.0)
    defaults.update(zone)
    app.seed_zone = defaults
    app.buddha_seed_entries = [_StateWidget() for _ in range(4)]
    app.buddha_seed_labels = [_StateWidget() for _ in range(5)]
    app.seed_job = None
    app.calls, app.pending = [], []
    app.root = type("Root", (), {"after_idle": lambda self, fn: app.pending.append(fn) or "job"})()
    app.refresh_timeline_diffs = lambda: app.calls.append("timeline")
    app.update_seed_zone_outline = lambda: app.calls.append("outline")
    for name in ("current_buddha_seed_settings", "update_buddha_seed_state", "on_buddha_seed_camera_changed",
                 "on_buddha_seed_commit", "on_buddha_seeds_changed", "refresh_buddha_seeds"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    _give_settings_model(app, buddha_seed_camera=camera, **{f"buddha_seed_{field}": str(value) for field, value in defaults.items()})
    return app


def test_the_zone_controls_are_greyed_out_while_the_camera_pixels_are_the_seeds():
    app = _fake_seeding_app(camera=True)
    app.update_buddha_seed_state()
    assert all(w.states[-1] == ["disabled"] for w in app.buddha_seed_entries + app.buddha_seed_labels)
    app.buddha_seed_camera_var.set(False)
    app.on_buddha_seed_camera_changed()
    assert all(w.states[-1] == ["!disabled"] for w in app.buddha_seed_entries + app.buddha_seed_labels)
    assert app.calls == ["outline", "timeline"] and len(app.pending) == 1


def test_the_seeding_settings_read_the_controls_and_an_entry_holding_nonsense_counts_as_its_last_value():
    app = _fake_seeding_app(camera=False, x_start=-0.5)
    app.buddha_seed_x_stop_var.set("0.75")
    app.buddha_seed_y_start_var.set("-")                 # half typed
    got = app.current_buddha_seed_settings()
    assert got == m.BuddhaSeedSettings(camera_pixels=False, x_start=-0.5, x_stop=0.75, y_start=-1.0, y_stop=1.0)
    app.buddha_seed_camera_var.set(True)
    assert app.current_buddha_seed_settings().camera_pixels is True


def test_committing_a_zone_entry_tidies_it_puts_back_the_last_valid_value_and_recolours_once():
    app = _fake_seeding_app(camera=False)
    app.buddha_seed_x_start_var.set("-.5")
    app.buddha_seed_y_stop_var.set("")
    app.on_buddha_seed_commit()
    assert app.buddha_seed_x_start_var.get() == "-0.5" and app.model.value("buddha_seed_x_start") == -0.5
    assert app.buddha_seed_y_stop_var.get() == "1.0", "an empty entry goes back to its last valid value"
    assert len(app.pending) == 1
    app.on_buddha_seed_commit()                          # a second commit before the recolour: still one pending job
    assert len(app.pending) == 1


def test_the_zone_outline_is_drawn_only_while_the_seeding_tab_is_focused_over_the_zone_in_world_space():
    class Canvas:
        def __init__(self):
            self.rects, self.deleted = [], []

        def delete(self, tag):
            self.deleted.append(tag)
            self.rects = []

        def create_rectangle(self, *coords, **options):
            self.rects.append((coords, options))

    class Notebook:
        def __init__(self, selected):
            self.selected = selected

        def select(self):
            return self.selected

    app = _fake_seeding_app(camera=False, x_start=-0.5, x_stop=0.5, y_start=-1.0, y_stop=0.0)
    app.canvas = Canvas()
    app.buddha_seeding_tab, app.buddha_tab = "seeding", "buddha"
    app.controls_notebook, app.buddha_notebook = Notebook("buddha"), Notebook("seeding")
    app.seed_tab_focused = m.App.seed_tab_focused.__get__(app)
    del app.update_seed_zone_outline
    app.update_seed_zone_outline = m.App.update_seed_zone_outline.__get__(app)
    app.update_seed_zone_outline()
    (coords, options), = app.canvas.rects
    quarter = m.PREVIEW_BOX / 4.0                    # (the image covers the default world [-2, 2]: 4 units)
    assert coords == (1.5 * quarter, 2.0 * quarter, 2.5 * quarter, 3.0 * quarter), "x -0.5..0.5 and y -1..0 as pixels"
    assert options["outline"] == m.SEED_ZONE_OUTLINE and options["dash"] and options["tags"] == "seedzone"
    app.buddha_notebook.selected = "blur"                # the other sub-tab: invisible
    app.update_seed_zone_outline()
    assert app.canvas.rects == [] and app.canvas.deleted[-1] == "seedzone"
    app.buddha_notebook.selected = "seeding"
    app.controls_notebook.selected = "coloration"        # another tab of the controls: invisible
    app.update_seed_zone_outline()
    assert app.canvas.rects == []
    app.controls_notebook.selected = "buddha"            # visible even with the camera pixels as seeds
    app.buddha_seed_camera_var.set(True)
    app.update_seed_zone_outline()
    assert len(app.canvas.rects) == 1


def _buddha_image(app, points_choice, seeds, zone_scene=None):
    termination, engine_args = _termination(40), _engine_args(m.build_obstacle_arrays(_unit_scene()), seg_mult=0.9)
    n = 24
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((n, n), dtype=np.uint8)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, UNIT_WORLD, termination, engine_args, 0)
    v = m.ColorationConfig(0)
    v.chan_src[0] = m.CHAN_SRC_BUDDHA
    v.buddha_points[0] = _buddha_choice(*points_choice)
    v.buddha_brightness[0] = 3
    blur = m.BuddhaBlurSettings(radius=0.0, type=m.BUDDHA_BLUR_BOKEH)
    return app.apply_coloration(_coloration_settings_from_config(v), stats, characteristic_points, pixel_state, n, n, blur, termination, engine_args, 0, seeds,
                                UNIT_WORLD)


def test_a_buddha_image_seeded_from_the_whole_world_zone_is_the_camera_pixels_image_whatever_the_options():
    app = _fake_render_app()
    zone = m.BuddhaSeedSettings(camera_pixels=False, x_start=-1.0, x_stop=1.0, y_start=-1.0, y_stop=1.0)
    choices = [(m.POINT_ALL_MEAN,), (m.POINT_END, m.POINT_SELF_MEAN), (len(m.BUDDHA_POINT_CATALOG) - 3,),
               (len(m.BUDDHA_POINT_CATALOG) - 2, len(m.BUDDHA_POINT_CATALOG) - 1)]
    for choice in choices:
        tethered = _buddha_image(app, choice, None)
        assert tethered[..., 0].max() > 0
        assert np.array_equal(tethered, _buddha_image(app, choice, m.DEFAULT_BUDDHA_SEED)), "camera pixels"
        assert np.array_equal(tethered, _buddha_image(app, choice, zone)), choice


def test_shrinking_the_seeding_zone_changes_the_buddha_image_without_changing_how_many_seeds_there_are():
    app = _fake_render_app()
    whole = m.BuddhaSeedSettings(camera_pixels=False, x_start=-1.0, x_stop=1.0, y_start=-1.0, y_stop=1.0)
    small = whole.replace(x_start=-0.5, x_stop=0.5, y_start=-0.5, y_stop=0.5)
    start_points = (m.POINT_START,)
    a, b = _buddha_image(app, start_points, whole), _buddha_image(app, start_points, small)
    assert not np.array_equal(a, b)
    # every seed's start point is the seed itself: one count per seed that fires, in the zone - so the same total
    # in both (all of them fire inside the unit circle's disc here), packed into the middle for the small zone
    n = 24
    termination, engine_args = _termination(40), _engine_args(m.build_obstacle_arrays(_unit_scene()))
    for zone, expected_rows in ((whole, (0, n)), (small, (n // 4, 3 * n // 4))):
        counts = np.zeros((n, n))
        m.scatter_from_seeds(0, n, m.WorldRect(zone.x_start, zone.x_stop, zone.y_start, zone.y_stop), UNIT_WORLD,
                             termination, engine_args, 0,
                             np.array([m.POINT_START], dtype=np.int64), np.empty(0, dtype=np.int64),
                             False, False, False, False, False, ALL_TERMS_ALLOWED, counts)
        assert counts[:expected_rows[0]].sum() == 0 and counts[expected_rows[1]:].sum() == 0
    full = np.zeros((n, n))
    m.scatter_from_seeds(0, n, m.WorldRect(small.x_start, small.x_stop, small.y_start, small.y_stop), UNIT_WORLD,
                         termination, engine_args, 0,
                         np.array([m.POINT_START], dtype=np.int64), np.empty(0, dtype=np.int64),
                         False, False, False, False, False, ALL_TERMS_ALLOWED, full)
    assert full.sum() == n * n, "all n x n seeds of the shrunken zone fire, and all are scattered"


def test_refreshing_the_seeding_scatters_the_buddha_channels_again_and_recolours_but_only_if_there_are_any():
    app = _fake_seeding_app(camera=False)
    app.colorations = [m.ColorationConfig(0)]
    app.coloration_thumb_cache = {1: "stale"}
    app.rows_done = 7
    app.pinned_px = (3, 4)
    for name in ("rebuild_buddha_counts", "draw", "rebuild_coloration_list"):
        setattr(app, name, lambda name=name: app.calls.append(name))
    app.recolor = lambda r0, r1: app.calls.append(("recolor", r0, r1))
    app.update_preview = lambda i, j: app.calls.append(("preview", i, j))
    app.seed_job = "job"
    app.refresh_buddha_seeds()
    assert app.seed_job is None and app.coloration_thumb_cache == {}
    assert app.calls == [], "no coloration has a Buddha channel: nothing to scatter again"
    app.colorations[0].chan_src[1] = m.CHAN_SRC_BUDDHA
    app.refresh_buddha_seeds()
    assert app.calls == ["rebuild_buddha_counts", ("recolor", 0, 7), "draw", ("preview", 3, 4), "rebuild_coloration_list"]


# --------------------------------------------------------------------------- #
# The colour chain exists once (color_pipeline) and the Buddha counts are made by one function (buddha)
# --------------------------------------------------------------------------- #
def _chain_scene(n=24):
    """(termination, engine_args, stats, characteristic_points, pixel_state) of an n x n render of a circle and an ellipse."""
    ellipse = m.EllipseShape(1)
    ellipse.width, ellipse.height, ellipse.rotation, ellipse.cx, ellipse.cy = 0.7, 0.45, 25.0, 0.15, -0.1
    termination = _termination(50)
    engine_args = _engine_args(m.build_obstacle_arrays(_unit_scene() + [ellipse]), seg_mult=0.9)
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((n, n), dtype=np.uint8)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, UNIT_WORLD, termination, engine_args, 3)
    return termination, engine_args, stats, characteristic_points, pixel_state


def _chain_coloration():
    """Channel 0 compares two points, channel 1 is a metric, channel 2 a Buddha channel (simple options and a multi one)."""
    v = m.ColorationConfig(0)
    v.color_model = "HSV"
    v.chan_src[2] = m.CHAN_SRC_BUDDHA
    v.buddha_points[2] = _buddha_choice(m.POINT_ALL_MEAN, m.POINT_END, m.BUDDHA_POINT_LABELS.index("All segment midpoints"))
    v.buddha_brightness[2] = 1
    return v


def test_a_hovered_ray_is_coloured_by_the_chain_of_an_image_of_one_pixel():
    """Regression test: the number panel of a hovered ray had a chain of its own, scalar by scalar, next to the one that
    colours images. It is now the image chain on a 1 x 1 image, so the two cannot disagree."""
    termination, engine_args, stats, characteristic_points, pixel_state = _chain_scene()
    v = _coloration_settings_from_config(_chain_coloration())
    mapping = m.make_mapping(v.metric_types, v.point_metric, v.other_metric, v.chan_src, v.constant_choice)
    n = stats.shape[0]
    blur = m.BuddhaBlurSettings(radius=0.05, type=m.BUDDHA_BLUR_GAUSSIAN)
    counts = np.zeros((n, n))
    m.scatter_buddha_channel(counts, v.buddha_points[2], 0, n, UNIT_WORLD, None, pixel_state == m.PIXEL_TRACED,
                             stats, characteristic_points, termination, engine_args, 3)
    counts = m.blur_counts_image(counts, m.blur_kernel(blur, n, n))
    computed = pixel_state == m.PIXEL_TRACED
    blurred = [None, None, counts]
    whole = m.channel_values(v, mapping, stats, characteristic_points, computed, blurred)
    where = np.argwhere(computed)
    for j, i in where[::37]:
        one = m.ray_values(v, mapping, stats[j, i].astype(np.float64), characteristic_points[j, i].astype(np.float64),
                           [None, None, counts[j, i]])
        index = int(np.flatnonzero((where[:, 0] == j) & (where[:, 1] == i))[0])
        assert np.allclose(one, [whole[c][index] for c in range(3)], atol=1e-12), (i, j)
    assert m.ray_values(v, mapping, stats[where[0][0], where[0][1]], characteristic_points[where[0][0], where[0][1]],
                        [None, None, None])[2] == 0.0, "a Buddha channel with no counts is 0"


def test_paint_fills_the_background_where_no_ray_was_traced_and_colours_the_rest_as_color_pixels_does():
    termination, engine_args, stats, characteristic_points, pixel_state = _chain_scene()
    v = _coloration_settings_from_config(m.ColorationConfig(0))
    mapping = m.make_mapping(v.metric_types, v.point_metric, v.other_metric, v.chan_src, v.constant_choice)
    computed = pixel_state == m.PIXEL_TRACED
    assert 0 < computed.sum() < computed.size
    out = np.zeros(stats.shape[:2] + (3,), dtype=np.uint8)
    m.paint(out, v, mapping, stats, characteristic_points, computed, [None] * 3)
    assert (out[~computed] == m.BACKGROUND).all()
    assert np.array_equal(out[computed], m.color_pixels(v, mapping, stats, characteristic_points, computed, [None] * 3))
    nothing = np.full(out.shape, 7, dtype=np.uint8)
    m.paint(nothing, v, mapping, stats, characteristic_points, np.zeros_like(computed), [None] * 3)
    assert (nothing == m.BACKGROUND).all(), "no pixel computed: the whole image is the background"


def test_scattering_a_channels_rows_in_pieces_gives_the_counts_of_scattering_them_all_at_once():
    """The live image scatters the rows as they are traced; an export scatters them all: the same function, so the same
    counts - for the stored points (with a termination filter), the multi options, and a seed grid."""
    termination, engine_args, stats, characteristic_points, pixel_state = _chain_scene()
    n = stats.shape[0]
    fired = pixel_state == m.PIXEL_TRACED
    zone = m.BuddhaSeedSettings(camera_pixels=False, x_start=-0.5, x_stop=0.8, y_start=-0.6, y_stop=0.5)
    labels = m.BUDDHA_POINT_LABELS
    choices = {"stored": _buddha_choice(m.POINT_ALL_MEAN, labels.index("Path: Last point (confined)")),
               "multi": _buddha_choice(labels.index("All self bounces"), labels.index("Progressive means of bounce points")),
               "both": _buddha_choice(m.POINT_END, labels.index("All segment midpoints"))}
    for name, chosen in choices.items():
        for seeds in (None, zone):
            whole, pieces = np.zeros((n, n)), np.zeros((n, n))
            m.scatter_buddha_channel(whole, chosen, 0, n, UNIT_WORLD, seeds, fired, stats, characteristic_points,
                                     termination, engine_args, 3)
            for r0, r1 in ((0, 5), (5, 17), (17, n)):
                m.scatter_buddha_channel(pieces, chosen, r0, r1, UNIT_WORLD, seeds, fired[r0:r1], stats[r0:r1],
                                         characteristic_points[r0:r1], termination, engine_args, 3)
            assert whole.sum() > 0 and np.array_equal(whole, pieces), (name, seeds)
    nothing = np.zeros((n, n))
    assert not m.scatter_buddha_channel(nothing, choices["stored"], 0, n, UNIT_WORLD, None, np.zeros_like(fired),
                                        stats, characteristic_points, termination, engine_args, 3)
    assert nothing.sum() == 0, "no ray fired in the rows: nothing to scatter, and nothing reported"


def test_the_live_image_built_row_by_row_is_the_picture_an_export_of_the_same_data_makes():
    """Two paths to the same picture - recolor() of the live image after its rows were scattered as they were traced,
    and apply_coloration() of the finished data - now share the colour chain and the Buddha scatter, and must agree,
    for the camera's pixels and for a seed grid."""
    termination, engine_args, stats, characteristic_points, pixel_state = _chain_scene()
    n = stats.shape[0]
    blur = m.BuddhaBlurSettings(radius=0.04, type=m.BUDDHA_BLUR_TRIANGULAR)
    camera = m.BuddhaSeedSettings(camera_pixels=True, x_start=-1.0, x_stop=1.0, y_start=-1.0, y_stop=1.0)
    zone = camera.replace(camera_pixels=False, x_start=-0.5, x_stop=0.8, y_start=-0.6, y_stop=0.5)
    v = _chain_coloration()
    v.buddha_points[1] = _buddha_choice(m.POINT_TOP)
    v.chan_src[1] = m.CHAN_SRC_BUDDHA
    for seeds in (camera, zone):
        class LiveApp:
            pass
        app = LiveApp()
        app.img_h = app.img_w = n
        app.image_world_rect = UNIT_WORLD
        app.stats, app.characteristic_points, app.pixel_state = stats, characteristic_points, pixel_state
        app.img = np.zeros((n, n, 3), np.uint8)
        app.rows_done = 0
        app.live_coloration = lambda: v
        app.mapping = m.make_mapping(v.metric_types, v.point_metric, v.other_metric, v.chan_src, v.constant_choice)
        app.current_blur_settings = lambda: blur
        app.current_buddha_seed_settings = lambda seeds=seeds: seeds
        _give_generated(app, ray=m.DEFAULT_RAY_SETTINGS.replace(fuzz=m.DEFAULT_RAY_SETTINGS.fuzz.replace(rng_seed=3)),
                        termination=termination, engine_args=engine_args)
        for name in ("scatter_buddha_rows", "blurred_buddha_counts", "rebuild_buddha_counts", "recolor"):
            setattr(app, name, getattr(m.App, name).__get__(app))
        app.rebuild_buddha_counts()
        for r0, r1 in ((0, 6), (6, 15), (15, n)):
            app.rows_done = r1
            app.scatter_buddha_rows(r0, r1)
        app.recolor(0, n)
        export = m.App.apply_coloration(app, _coloration_settings_from_config(v), stats, characteristic_points, pixel_state,
                                        n, n, blur, termination, engine_args, 3, seeds, UNIT_WORLD)
        assert np.array_equal(app.img, export), seeds.camera_pixels
        assert (app.img != m.BACKGROUND).any()


# --------------------------------------------------------------------------- #
# Stored layouts: an image keeps only the slots its colorations read
# --------------------------------------------------------------------------- #
def test_a_stored_layout_names_the_slots_kept_and_refuses_the_others():
    layout = m.StoredLayout.of({m.ST_BOUNCE_COUNT, m.ST_TERMINATION}, [m.POINT_END, m.POINT_START, m.POINT_END])
    assert layout.stat_ids.tolist() == sorted([m.ST_BOUNCE_COUNT, m.ST_TERMINATION])
    assert layout.point_ids.tolist() == sorted([m.POINT_START, m.POINT_END])
    assert layout.stat_position(m.ST_TERMINATION) == layout.stat_ids.tolist().index(m.ST_TERMINATION)
    assert layout.point_position(m.POINT_END) == layout.point_ids.tolist().index(m.POINT_END)
    assert _raises(KeyError, lambda: layout.point_position(m.POINT_TOP)), "a slot that is not kept is never read as zeros"
    assert _raises(KeyError, lambda: layout.stat_position(m.ST_ANGLES))
    assert _raises(ValueError, lambda: m.StoredLayout.of([m.N_STATS], []))
    assert _raises(ValueError, lambda: m.StoredLayout.of([], [-1]))
    stats, points = layout.allocate(5, 7)
    assert stats.shape == (5, 7, 2) and points.shape == (5, 7, 2, 2) and stats.dtype == points.dtype == np.float32
    assert layout.bytes_per_pixel == 2 * 4 + 2 * 8 + 1
    other = m.StoredLayout.of([m.ST_ANGLES], [m.POINT_TOP])
    both = layout.union(other)
    assert both.stat_ids.tolist() == sorted({m.ST_BOUNCE_COUNT, m.ST_TERMINATION, m.ST_ANGLES})
    assert both.point_ids.tolist() == sorted({m.POINT_START, m.POINT_END, m.POINT_TOP})


def test_the_full_layout_keeps_every_slot_in_order_which_is_321_bytes_a_pixel():
    assert m.FULL_LAYOUT.stat_ids.tolist() == list(range(m.N_STATS))
    assert m.FULL_LAYOUT.point_ids.tolist() == list(range(m.N_POINTS))
    assert all(m.FULL_LAYOUT.stat_position(s) == s for s in range(m.N_STATS))
    assert all(m.FULL_LAYOUT.point_position(p) == p for p in range(m.N_POINTS))
    assert m.FULL_LAYOUT.bytes_per_pixel == 321


def test_trace_rows_stored_keeps_only_the_slots_asked_for_and_all_of_them_is_trace_rows():
    termination, engine_args, stats, characteristic_points, pixel_state = _chain_scene()
    n = stats.shape[0]

    def traced(layout):
        s, p = layout.allocate(n, n)
        state = np.zeros((n, n), dtype=np.uint8)
        m.trace_rows_stored(s, p, state, 0, n, n, n, UNIT_WORLD, termination, engine_args, 3, layout.stat_ids,
                            layout.point_ids)
        return s, p, state
    s, p, state = traced(m.FULL_LAYOUT)
    assert np.array_equal(s, stats) and np.array_equal(p, characteristic_points) and np.array_equal(state, pixel_state)
    some = m.StoredLayout.of([m.ST_TERMINATION, m.ST_BOUNCE_COUNT, m.ST_LOOP_AREA], [m.POINT_END, m.POINT_SELF_MEAN, m.POINT_START])
    s, p, state = traced(some)
    assert np.array_equal(s, stats[..., some.stat_ids]) and np.array_equal(p, characteristic_points[..., some.point_ids, :])
    assert np.array_equal(state, pixel_state)
    none = m.StoredLayout.of([], [])
    s, p, state = traced(none)
    assert s.shape == (n, n, 0) and p.shape == (n, n, 0, 2) and np.array_equal(state, pixel_state)


def test_a_demand_layout_keeps_what_the_colorations_read_and_nothing_else():
    default = _coloration_settings_from_config(m.ColorationConfig(0))      # two point pairs and a count of bounces
    layout = m.demand_layout([default])
    assert layout.stat_ids.tolist() == [m.ST_BOUNCE_COUNT] and layout.point_ids.tolist() == [m.POINT_START, m.POINT_END]
    assert layout.bytes_per_pixel == 21
    v = m.ColorationConfig(0)
    v.point1_choice[0], v.point2_choice[0] = m.CHOICE_ORIGIN, m.POINT_TOP           # a fixed point is not stored
    v.chan_src[1] = m.CHAN_SRC_CONSTANT
    v.chan_src[2] = m.CHAN_SRC_BUDDHA
    v.buddha_points[2] = _buddha_choice(m.BUDDHA_POINT_LABELS.index("All self bounces"))   # a multi option stores nothing
    only_top = m.demand_layout([_coloration_settings_from_config(v)])
    assert only_top.stat_ids.tolist() == [] and only_top.point_ids.tolist() == [m.POINT_TOP]
    v.buddha_points[2] = _buddha_choice(m.POINT_SELF_MEAN, m.BUDDHA_POINT_LABELS.index("Path: Last point (confined)"))
    settings = _coloration_settings_from_config(v)
    assert m.demand_layout([settings]).point_ids.tolist() == sorted([m.POINT_TOP, m.POINT_SELF_MEAN, m.POINT_END])
    assert m.demand_layout([settings]).stat_ids.tolist() == [m.ST_TERMINATION], "a termination filter needs that metric"
    zone = m.DEFAULT_BUDDHA_SEED.replace(camera_pixels=False)
    assert m.demand_layout([settings], zone).point_ids.tolist() == [m.POINT_TOP], "a seed grid reads no stored point"
    both = m.demand_layout([default, settings])
    assert both.stat_ids.tolist() == sorted([m.ST_BOUNCE_COUNT, m.ST_TERMINATION])
    assert both.point_ids.tolist() == sorted([m.POINT_START, m.POINT_END, m.POINT_TOP, m.POINT_SELF_MEAN])


def _random_coloration(rng):
    v = m.ColorationConfig(0)
    v.color_model = str(rng.choice(m.COLOR_MODEL_NAMES))
    other_metrics, point_metrics = sorted(m.OTHER_METRICS), sorted(m.POINT_METRICS)
    for c in range(3):
        v.chan_src[c] = int(rng.choice([m.CHAN_SRC_QUERY, m.CHAN_SRC_QUERY, m.CHAN_SRC_BUDDHA, m.CHAN_SRC_CONSTANT]))
        v.metric_types[c] = int(rng.choice([m.METRIC_TYPE_POINT_PAIR, m.METRIC_TYPE_OTHER]))
        v.point_metric[c] = int(rng.choice(point_metrics))
        v.other_metric[c] = int(rng.choice(other_metrics))
        v.point1_choice[c] = int(rng.randint(m.N_POINT_CHOICES))
        v.point2_choice[c] = int(rng.randint(m.N_POINT_CHOICES))
        v.buddha_points[c] = tuple(bool(rng.rand() < 0.12) for _ in range(m.N_BUDDHA_POINTS))
        v.buddha_compression[c] = int(rng.randint(2))
        v.buddha_path_kinds[c] = tuple(bool(rng.rand() < 0.7) for _ in range(3))
        v.buddha_brightness[c] = int(rng.randint(4))
        v.constant_choice[c] = int(rng.choice([m.NONE_ZERO, m.NONE_ONE]))
    return v


def test_a_demand_layout_keeps_everything_the_colour_chain_reads():
    """Random colorations (every source, metric and point choice, Buddha options of every sort) and both kinds of
    seeding: the image made from the slots a demand layout keeps is the image made from all of them. A channel source
    whose reads demand_layout forgot would fail here (or raise: an unkept slot is never read as zeros)."""
    termination, engine_args, stats, characteristic_points, pixel_state = _chain_scene()
    n = stats.shape[0]
    blur = m.BuddhaBlurSettings(radius=0.02, type=m.BUDDHA_BLUR_BOKEH)
    zone = m.DEFAULT_BUDDHA_SEED.replace(camera_pixels=False, x_start=-0.5, x_stop=0.8, y_start=-0.6, y_stop=0.5)
    rng = np.random.RandomState(11)
    apply = m.App.apply_coloration.__get__(type("A", (), {})())
    checked = 0
    for trial in range(60):
        settings = [_coloration_settings_from_config(_random_coloration(rng)) for _ in range(1 + trial % 2)]
        seeds = zone if trial % 3 == 2 else None
        layout = m.demand_layout(settings, seeds)
        kept_stats, kept_points = stats[..., layout.stat_ids], characteristic_points[..., layout.point_ids, :]
        for v in settings:
            want = apply(v, stats, characteristic_points, pixel_state, n, n, blur, termination, engine_args, 3, seeds,
                         UNIT_WORLD)
            got = apply(v, kept_stats, kept_points, pixel_state, n, n, blur, termination, engine_args, 3, seeds,
                        UNIT_WORLD, layout)
            assert np.array_equal(want, got), (trial, v)
            checked += 1
    assert checked > 60


def test_a_large_export_keeps_only_what_its_colorations_read():
    """render_full_image allocates the demand layout, not 321 bytes for every pixel: the same picture as the full
    layout, in a fraction of the memory (measured with tracemalloc, which sees numpy's arrays)."""
    import tracemalloc
    app = _fake_render_app()
    data = m.resolve_frame_settings([m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES))], 0, m.DEFAULT_SETTING_VALUES).replace(outline=False)
    obst_arrays = _starting_obst_arrays()
    n = 256
    buddha = m.ColorationConfig(1)
    buddha.chan_src[1] = m.CHAN_SRC_BUDDHA
    buddha.buddha_points[1] = _buddha_choice(m.POINT_ALL_MEAN, m.POINT_END)
    colorations = (_coloration_settings_from_config(m.ColorationConfig(0)), _coloration_settings_from_config(buddha))
    termination, engine_args = m.settings_natural_termination(data.natural_termination), m.path_args(data.engine.engine, m.settings_ray_args(data.ray), obst_arrays)
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((n, n), dtype=np.uint8)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, m.WORLD_RECT_DEFAULT, termination, engine_args, data.ray.fuzz.rng_seed)
    want = np.concatenate([app.apply_coloration(v, stats, characteristic_points, pixel_state, n, n, data.buddha_blur, termination,
                                                engine_args, data.ray.fuzz.rng_seed, data.buddha_seed) for v in colorations], axis=1)
    full_bytes = n * n * m.FULL_LAYOUT.bytes_per_pixel
    del stats, characteristic_points
    tracemalloc.start()
    try:
        got = app.render_full_image(data, n, n, obst_arrays, colorations, m.DEFAULT_FRAMING)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert np.array_equal(got, want)
    assert peak < 0.4 * full_bytes, (peak, full_bytes)


# --------------------------------------------------------------------------- #
# A Buddha channel scatters only the paths of the kinds it has ticked (escaping / non-escaping / terminated by obstacle)
# --------------------------------------------------------------------------- #
def _kinds_scene():
    """Three firing and non-firing obstacles that between them give rays every kind of end: a circle that reflects (its rays
    run out of iterations, or hit the perpendicular, or are terminated by the small circle in it), an ellipse beside it that
    lets rays out (they escape)."""
    big = _circle_obstacle(-0.5, 0.0, 0.5, R, R, fire=True, id_=0)
    small = _circle_obstacle(-0.5, 0.15, 0.1, T, T, fire=False, id_=1)
    out = m.EllipseShape(2)
    out.width, out.height, out.cx, out.cy = 0.6, 0.5, 0.55, 0.4
    out.in_effect, out.out_effect, out.fire_inside = R, P, True
    return [big, small, out]


def _kinds_trace(n=48):
    arrays = m.build_obstacle_arrays(_kinds_scene())
    termination, engine_args = _termination(12), _engine_args(arrays)
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    state = np.zeros((n, n), dtype=np.uint8)
    m.trace_rows(stats, points, state, 0, n, n, n, UNIT_WORLD, termination, engine_args, 0)
    return n, termination, engine_args, stats, points, state


def _pixel_xy(i, j, n, rect=UNIT_WORLD):
    """Where the ray of pixel (i, j) of an n x n image of `rect` starts: the kernels' own arithmetic, bit for bit (a chaotic
    path would notice a rounding difference)."""
    return (rect.x_start + (i + 0.5) * ((rect.x_stop - rect.x_start) / n),
            rect.y_stop - (j + 0.5) * ((rect.y_stop - rect.y_start) / n))


def _path_of(x, y, termination, engine_args):
    path_points, tags, raw = np.empty((termination.max_iter + 2, 2)), np.empty(termination.max_iter + 2), np.empty(m.N_RAW)
    term, npts = m.trace_path(x, y, path_points, tags, termination, engine_args.ray, engine_args.obstacles, raw)
    return term, path_points[:npts].copy()


def test_the_scene_for_the_path_kinds_has_every_kind_of_path_and_every_ending_that_is_non_escaping():
    n, termination, engine_args, stats, _points, state = _kinds_trace()
    ends = {}
    for j in range(n):
        for i in range(n):
            if state[j, i] == m.PIXEL_TRACED:
                term, _path = _path_of(*_pixel_xy(i, j, n), termination, engine_args)
                ends[term] = ends.get(term, 0) + 1
    assert {m.TERM_ESCAPED, m.TERM_OBSTACLE, m.TERM_ITER, m.TERM_PERPENDICULAR} <= set(ends), ends
    assert m.PATH_KIND_OF_TERM[m.TERM_PERPENDICULAR] == m.PATH_KIND_NON_ESCAPING, "a perpendicular collision ends a path, not escaping"
    assert {m.PATH_KIND_OF_TERM[t] for t in ends} == {0, 1, 2}


def test_a_buddha_channel_scatters_only_the_paths_of_the_kinds_ticked_whatever_it_scatters_and_from_where():
    """Property test over every set of ticked kinds, three options (a stored point; every point of the path; every segment
    midpoint) and both kinds of seeding, against a reference written from trace_path alone. All three ticked is no limit."""
    n, termination, engine_args, stats, points, state = _kinds_trace()
    fired = state == m.PIXEL_TRACED
    zone = m.BuddhaSeedSettings(camera_pixels=False, x_start=-0.8, x_stop=0.9, y_start=-0.7, y_stop=0.8)
    options = {"stored end point": _buddha_choice(m.POINT_END), "all path points": _buddha_choice(m.BUDDHA_POINT_LABELS.index("All path points")),
               "all midpoints": _buddha_choice(m.BUDDHA_POINT_LABELS.index("All segment midpoints"))}
    arrays = engine_args.obstacles
    checked = 0
    for ticked in range(8):
        kinds = tuple(bool(ticked >> k & 1) for k in range(3))
        for name, chosen in options.items():
            for seeds in (None, zone):
                got = np.zeros((n, n))
                m.scatter_buddha_channel(got, chosen, 0, n, UNIT_WORLD, seeds, fired, stats, points, termination, engine_args, 0, kinds=kinds)
                want = np.zeros((n, n))
                if seeds is None and name == "stored end point":                  # the stored points of the pixels whose path is of a kind ticked
                    allowed = np.zeros((n, n), dtype=bool)
                    for j in range(n):
                        for i in range(n):
                            if fired[j, i]:
                                term, _path = _path_of(*_pixel_xy(i, j, n), termination, engine_args)
                                allowed[j, i] = kinds[m.PATH_KIND_OF_TERM[term]]
                    m.scatter_buddha_channel(want, chosen, 0, n, UNIT_WORLD, None, allowed, stats, points, termination, engine_args, 0)
                else:
                    for j in range(n):
                        for i in range(n):
                            x, y = _pixel_xy(i, j, n, UNIT_WORLD if seeds is None else zone)
                            if not m.fires_ray(x, y, arrays):
                                continue
                            term, path = _path_of(x, y, termination, engine_args)
                            if not kinds[m.PATH_KIND_OF_TERM[term]]:
                                continue
                            if name == "all path points":
                                scattered = path
                            elif name == "all midpoints":
                                scattered = 0.5 * (path[:-1] + path[1:])
                            else:                                                # a stored point, from a seed grid: the path's last point
                                scattered = path[-1:]
                            for px, py in scattered:
                                m.scatter_one_buddha_point(want, px, py, UNIT_WORLD)
                assert np.array_equal(got, want), (kinds, name, seeds is None)
                checked += 1
                if kinds == m.ALL_PATH_KINDS:
                    assert got.sum() > 0
                if not any(kinds):
                    assert got.sum() == 0, "no kind ticked: nothing is scattered"
    assert checked == 8 * 3 * 2


def test_all_path_points_is_an_option_of_its_own_that_includes_the_starting_point():
    label = m.BUDDHA_POINT_LABELS[-1]
    assert label == "All path points" and m.BUDDHA_POINT_CATALOG[-1][1:] == (m.BUDDHA_MULTI, m.BUDDHA_MULTI_ALL_PATH, None)
    assert m.N_BUDDHA_POINTS == len(m.BUDDHA_POINT_CATALOG) == 36 and m.BUDDHA_POINT_SHORT[-1] == "AllPath"
    counts = np.zeros((8, 8))
    circle = _circle_obstacle(0.0, 0.0, 0.9, R, R, fire=True)
    arrays = m.build_obstacle_arrays([circle])
    m.scatter_multi_buddha(0, 8, UNIT_WORLD, _termination(5), _engine_args(arrays), 0, False, False, False, False, True,
                           ALL_TERMS_ALLOWED, counts)
    starts = np.zeros((8, 8))
    for j in range(8):
        for i in range(8):
            x, y = _pixel_xy(i, j, 8)
            if m.fires_ray(x, y, arrays):
                m.scatter_one_buddha_point(starts, x, y, UNIT_WORLD)
    assert (counts >= starts).all() and starts.sum() > 20 and counts.sum() > starts.sum(), "every start, and the points after it"


def test_a_demand_layout_keeps_the_termination_metric_for_a_channel_limited_to_some_kinds_of_path():
    v = m.ColorationConfig(0)
    v.chan_src[2] = m.CHAN_SRC_BUDDHA
    v.buddha_points[2] = _buddha_choice(m.POINT_END)
    assert m.ST_TERMINATION not in m.demand_layout([_coloration_settings_from_config(v)]).stat_ids.tolist()
    v.buddha_path_kinds[2] = (True, False, True)
    assert m.ST_TERMINATION in m.demand_layout([_coloration_settings_from_config(v)]).stat_ids.tolist()
    zone = m.DEFAULT_BUDDHA_SEED.replace(camera_pixels=False)
    assert m.ST_TERMINATION not in m.demand_layout([_coloration_settings_from_config(v)], zone).stat_ids.tolist(), \
        "a seed grid traces its rays again and reads nothing stored"
    v.buddha_points[2] = _buddha_choice(m.BUDDHA_POINT_LABELS.index("All path points"))
    assert m.ST_TERMINATION not in m.demand_layout([_coloration_settings_from_config(v)]).stat_ids.tolist(), "a multi option reads nothing stored"


def test_the_kind_of_path_of_every_ending_and_the_flags_the_kernels_take():
    assert m.PATH_KIND_OF_TERM == {m.TERM_ESCAPED: 0, m.TERM_OBSTACLE: 2, m.TERM_ITER: 1, m.TERM_DIST: 1, m.TERM_CONFINED: 1,
                                   m.TERM_PERPENDICULAR: 1}
    assert m.PATH_KIND_LABELS == ["Escaping", "Non-escaping", "Terminated by obstacle"] and m.N_TERM == 6
    flags = m.term_allowed_flags((True, False, False))
    assert flags.dtype == np.bool_ and [t for t in range(m.N_TERM) if flags[t]] == [m.TERM_ESCAPED]
    assert m.term_allowed_flags(m.ALL_PATH_KINDS).all() and not m.term_allowed_flags((False,) * 3).any()
    assert [t for t in range(m.N_TERM) if m.term_allowed_flags((False, True, False))[t]] == sorted(
        [m.TERM_ITER, m.TERM_DIST, m.TERM_CONFINED, m.TERM_PERPENDICULAR])


def test_the_path_kind_checkboxes_write_the_channels_three_flags_and_rebuild_the_counts():
    class FakeApp:
        pass

    class Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value
    app = FakeApp()
    app._color_editor_loading = False
    coloration = m.ColorationConfig(0)
    app.selected_coloration_obj = lambda: coloration
    app.buddha_kind_vars = [[Var(True), Var(False), Var(True)] for _ in range(3)]
    changed = []
    app.apply_mapping_change = lambda c: changed.append(c)
    m.App.on_buddha_path_kinds_changed(app, 1)
    assert coloration.buddha_path_kinds[1] == (True, False, True) and coloration.buddha_path_kinds[0] == m.ALL_PATH_KINDS
    assert changed == [1]
    app._color_editor_loading = True
    app.buddha_kind_vars[1][0].set(False)
    m.App.on_buddha_path_kinds_changed(app, 1)
    assert coloration.buddha_path_kinds[1] == (True, False, True) and changed == [1], "nothing while the editor is loading"


def test_a_limited_channel_says_which_paths_in_its_info_text_and_a_complete_one_says_nothing():
    v = m.ColorationConfig(0)
    v.chan_src[1] = m.CHAN_SRC_BUDDHA
    assert not any("paths" in line for line in m.coloration_info_lines(v))
    v.buddha_path_kinds[1] = (True, False, True)
    assert "  paths: Escaping, Terminated by obstacle" in m.coloration_info_lines(v)
    v.buddha_path_kinds[1] = (False, False, False)
    assert "  paths: none" in m.coloration_info_lines(v)
    short = m.short_value_text("coloration:0:chan1", m.coloration_channel_value(v, 1))
    assert "[]" in short or short.startswith("Buddha:") and "[" in short


def test_an_export_of_a_buddha_channel_with_only_multi_options_and_a_limited_kind_of_path_does_not_need_the_termination_metric():
    """Regression test: scatter_stored_points read the stored termination metric whenever a kind of path was unticked, even when
    every option the channel had traces its rays again (and so demand_layout had not kept the metric): an export raised KeyError."""
    app = _fake_render_app()
    config = m.ColorationConfig(0)
    config.chan_src[:] = [m.CHAN_SRC_BUDDHA, m.CHAN_SRC_CONSTANT, m.CHAN_SRC_CONSTANT]
    config.buddha_points[0] = tuple(label == "All self bounces" for label in m.BUDDHA_POINT_LABELS)
    config.buddha_path_kinds[0] = (True, False, False)
    data = m.resolve_frame_settings([], 0, {**m.DEFAULT_SETTING_VALUES, "outline": False, "iter": 30})
    image = app.render_full_image(data, 64, 64, _starting_obst_arrays(), (_coloration_settings_from_config(config),), m.DEFAULT_FRAMING)
    assert image.shape == (64, 64, 3)
