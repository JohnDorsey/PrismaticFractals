"""Tests: live_image (the image being generated, apart from any window). No Tk is needed for any of them."""
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


def _figure(iterations=8):
    """A figure of the starting obstacle (the circle that fills the default world), with a short iteration limit so that it renders at once."""
    return m.generated_figure(m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_TERMINATION_SETTINGS.replace(max_iter=iterations),
                              [m.default_starting_obstacle(0)])


def _live(framing=None, figure=None):
    figure = _figure() if figure is None else figure
    return m.LiveImage(m.DEFAULT_FRAMING if framing is None else framing, figure.layout), figure


def _buddha_coloration():
    coloration = m.ColorationConfig(0)
    coloration.chan_src[0] = m.CHAN_SRC_BUDDHA
    mapping = m.make_mapping(coloration.metric_types, coloration.point_metric, coloration.other_metric, coloration.chan_src,
                             coloration.constant_choice)
    return coloration, mapping


def test_a_new_live_image_is_a_blank_image_of_its_framing_with_nothing_computed():
    live, _figure_ = _live()
    assert (live.img_w, live.img_h) == m.framing_image_size(m.DEFAULT_FRAMING) and live.rows_done == 0
    assert live.world_rect == m.framing_world_rect(m.DEFAULT_FRAMING)
    assert (live.pixel_state == m.PIXEL_PENDING).all() and (live.img == m.BACKGROUND).all() and not live.stats.any()
    assert live.buddha_counts == [None] * 3 and live.gx.shape == live.gy.shape == (live.img_h, live.img_w)


def test_resetting_to_a_framing_of_the_same_shape_clears_the_buffers_it_has_and_another_shape_makes_new_ones():
    live, figure = _live()
    stats, img = live.stats, live.img
    live.advance(figure, 10.0)
    assert live.rows_done == live.img_h and live.pixel_state.any()
    assert live.reset(m.DEFAULT_FRAMING, figure.layout) is False
    assert live.stats is stats and live.img is img and live.rows_done == 0 and not live.stats.any() and (live.img == m.BACKGROUND).all()
    wide = m.FramingSnapshot(id=1, x=0.0, y=0.0, diagonal=m.DEFAULT_FRAMING_DIAGONAL, aspect=2.0)
    assert live.reset(wide, figure.layout) is True
    assert (live.img_w, live.img_h) == m.framing_image_size(wide) and live.img.shape == (live.img_h, live.img_w, 3)
    assert live.framing is wide and live.world_rect == m.framing_world_rect(wide)


def test_a_pixels_centre_and_the_canvas_position_of_that_centre_are_each_others_inverse():
    live, _figure_ = _live(m.FramingSnapshot(id=1, x=0.5, y=-0.25, diagonal=1.5, aspect=1.5))
    for i, j in ((0, 0), (3, 7), (live.img_w - 1, live.img_h - 1)):
        x, y = live.pixel_center(i, j)
        cx, cy = live.to_canvas(x, y)
        assert abs(cx - (i + 0.5)) < 1e-9 and abs(cy - (j + 0.5)) < 1e-9, (i, j)
        assert abs(live.gx[j, i] - x) < 1e-9 and abs(live.gy[j, i] - y) < 1e-9, "(the grid holds the same centres)"


def test_advancing_in_small_slices_fills_the_image_with_exactly_what_one_trace_of_all_its_rows_makes():
    live, figure = _live()
    firsts = []
    while live.rows_done < live.img_h:
        firsts.append(live.advance(figure, 0.0))                 # (a budget of nothing: one batch a call)
    assert firsts[0] == 0 and firsts == sorted(firsts) and len(firsts) > 1 and live.rows_done == live.img_h
    stats, points = figure.layout.allocate(live.img_h, live.img_w)
    state = np.full((live.img_h, live.img_w), m.PIXEL_PENDING, dtype=np.uint8)
    m.trace_rows(stats, points, state, 0, live.img_h, live.img_w, live.img_h, live.world_rect, figure.termination, figure.engine_args,
                 figure.rng_seed)
    assert np.array_equal(live.stats, stats) and np.array_equal(live.characteristic_points, points) and np.array_equal(live.pixel_state, state)
    assert live.advance(figure, 1.0) == live.img_h, "(nothing left to trace)"


def test_recolouring_rows_paints_only_those_rows_and_nothing_for_an_empty_range():
    live, figure = _live()
    live.advance(figure, 10.0)
    coloration, mapping = m.ColorationConfig(0), None
    mapping = m.make_mapping(coloration.metric_types, coloration.point_metric, coloration.other_metric, coloration.chan_src,
                             coloration.constant_choice)
    live.recolor(5, 5, coloration, mapping, figure, m.DEFAULT_BUDDHA_SEED, m.DEFAULT_BUDDHA_BLUR)
    assert (live.img == m.BACKGROUND).all()
    live.recolor(10, 20, coloration, mapping, figure, m.DEFAULT_BUDDHA_SEED, m.DEFAULT_BUDDHA_BLUR)
    assert (live.img[:10] == m.BACKGROUND).all() and (live.img[20:] == m.BACKGROUND).all()
    assert (live.img[10:20] != m.BACKGROUND).any(), "the fired pixels of those rows have their colours"


def test_a_buddha_channel_scatters_its_counts_blurs_them_once_and_loses_both_with_a_new_framing():
    live, figure = _live()
    coloration, mapping = _buddha_coloration()
    live.advance(figure, 10.0)
    live.rebuild_buddha_counts(coloration, figure, m.DEFAULT_BUDDHA_SEED)
    assert live.buddha_counts[0] is not None and live.buddha_counts[1] is None and live.buddha_counts[0].sum() > 0
    blur = m.BuddhaBlurSettings(radius=0.02, type=m.BUDDHA_BLUR_BOKEH)
    first = live.blurred_buddha_counts(0, blur)
    assert first is live.blurred_buddha_counts(0, blur), "(kept until the counts change)"
    assert live.blurred_buddha_counts(1, blur) is None, "(a channel not in Buddha mode has no counts)"
    live.invalidate_blur()
    assert live.blurred_buddha_counts(0, blur) is not first
    live.recolor(0, live.rows_done, coloration, mapping, figure, m.DEFAULT_BUDDHA_SEED, blur)
    assert (live.img != m.BACKGROUND).any()
    live.reset(m.DEFAULT_FRAMING, figure.layout)
    assert live.buddha_counts == [None] * 3 and live.buddha_counts_blurred == [None] * 3


def test_scattering_rows_adds_to_the_counts_of_the_buddha_channels_only_and_a_rebuild_starts_again():
    live, figure = _live()
    coloration, _mapping = _buddha_coloration()
    live.rebuild_buddha_counts(coloration, figure, m.DEFAULT_BUDDHA_SEED)
    assert live.buddha_counts[0].sum() == 0, "(nothing computed yet)"
    while live.rows_done < live.img_h:
        first = live.advance(figure, 0.0)
        live.scatter_buddha_rows(first, live.rows_done, coloration, figure, m.DEFAULT_BUDDHA_SEED)
    incremental = live.buddha_counts[0].copy()
    live.rebuild_buddha_counts(coloration, figure, m.DEFAULT_BUDDHA_SEED)
    assert np.array_equal(live.buddha_counts[0], incremental), "(row by row gives what one pass over all of them gives)"
    live.scatter_buddha_rows(3, 3, coloration, figure, m.DEFAULT_BUDDHA_SEED)
    assert np.array_equal(live.buddha_counts[0], incremental), "(an empty range adds nothing)"
