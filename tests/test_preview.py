"""Tests: preview."""
import os
import tempfile
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# A pinned ray preview stays visible even where no ray would fire
# --------------------------------------------------------------------------- #
def _fake_preview_app(pinned):
    import collections

    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    app.update_preview = m.App.update_preview.__get__(app)
    app.pinned_px = pinned
    app.hover_px = None
    app.obstacles = []                              # nothing at all fires a ray
    app.preview_settings = lambda: m.DEFAULT_RAY_SETTINGS
    app.preview_natural_termination_settings = lambda: m.DEFAULT_NATURAL_TERMINATION_SETTINGS
    app.hover_path_points = np.empty((m.ITER_LIMIT_MAX + 2, 2))
    app.hover_path_point_tags = np.empty(m.ITER_LIMIT_MAX + 2)
    app.hover_raw = np.empty(m.N_RAW)
    app.hover_vec = np.empty(m.N_STATS)
    app.hover_ray_characteristic_points = np.empty((m.N_POINTS, 2))
    app.component_values = lambda vec, ray_characteristic_points, x, y: (0.1, 0.2, 0.3)
    app.draw_segments = lambda path_points, path_point_tags, n: None
    app.draw_corkscrew = lambda path_points, path_point_tags, n: None
    app.deco = {k: _StubVar(k != "samples") for k in
                ("pending", "outline", "ray_path", "ray_start", "ray_end", "guides", "samples")}
    app.canvas = _StubWidget()
    app.hline_img = app.vline_img = None
    app.live_coloration = lambda: m.ColorationConfig(0)
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
    _give_image_geometry(app)
    app.on_click = m.App.on_click.__get__(app)
    app.pinned_px = pinned
    app.hover_px = "stale"
    app.obstacles = obstacles
    app.preview_settings = lambda: m.DEFAULT_RAY_SETTINGS
    app.preview_natural_termination_settings = lambda: m.DEFAULT_NATURAL_TERMINATION_SETTINGS
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
    strings += m.METRIC_LABELS
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
    v = m.ColorationConfig(0)
    lines = m.coloration_info_lines(v)
    assert lines[0] == f"Mode:{v.mode}"
    assert not any("Point comparison" in l or "Other" in l for l in lines), \
        "the metric's kind must never be shown - it's implied by the metric's name"
    letter = m.MODES[v.mode][0][0][0]
    expected_metric = m.METRIC_LABELS[v.point_metric[0]]
    assert f"{letter}:{expected_metric}" in lines


def test_coloration_info_lines_uses_long_names_not_short_ones():
    """The point of this turn's change: dropdown-length names, not the
    compact forms used elsewhere (e.g. the keyframe list)."""
    v = m.ColorationConfig(0)
    lines = m.coloration_info_lines(v)
    joined = " ".join(lines)
    assert "H dist" not in joined and "Horizontal distance" in joined


def test_point_comparison_metric_names_no_longer_say_between_points():
    for label in m.METRIC_LABELS[:4]:      # the four point-comparison metrics
        assert "between points" not in label
    assert all("between points" not in label for label in m.POINT_METRIC_LABELS)


def test_coloration_info_lines_shows_points_only_for_point_comparison_metrics():
    v = m.ColorationConfig(0)
    v.metric_types[2] = m.METRIC_TYPE_OTHER
    v.other_metric[2] = m.ST_BOUNCES
    lines = m.coloration_info_lines(v)
    letter2 = m.MODES[v.mode][2][0][0]
    idx = lines.index(f"{letter2}:{m.METRIC_LABELS[m.ST_BOUNCES]}")
    # the very next line (if any - this may be the last line) must NOT be a
    # Point 1/2 line for this channel
    assert idx + 1 == len(lines) or not lines[idx + 1].strip().startswith("P1:")


def test_coloration_info_lines_buddha_channel_shows_points_compression_brightness():
    v = m.ColorationConfig(0)
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


def test_coloration_info_lines_works_for_resolved_coloration_settings_too():
    """New image / video export pass resolved ColorationSettings (read-only
    tuples), not live ColorationConfig objects - the field names must line up."""
    vc = m.ColorationConfig(0)
    vc.chan_src[1] = m.CHAN_SRC_BUDDHA
    vs = _coloration_settings_from_config(vc)
    assert m.coloration_info_lines(vc) == m.coloration_info_lines(vs)


def test_draw_coloration_info_uses_pure_black_background_and_not_pure_white_text():
    assert m.COLORATION_INFO_BG == (0, 0, 0)
    assert m.COLORATION_INFO_FG != (255, 255, 255)
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img, [m.ColorationConfig(0), m.ColorationConfig(1)], 100)
    assert (img == m.COLORATION_INFO_BG).all(axis=-1).any()
    assert (img == m.COLORATION_INFO_FG).all(axis=-1).any()
    assert not (img == (255, 255, 255)).all(axis=-1).any()


def test_draw_coloration_info_stays_within_each_colorations_own_portion():
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img, [m.ColorationConfig(0), m.ColorationConfig(1)], 100)
    left_has_info = (img[:, :100] == m.COLORATION_INFO_BG).all(axis=-1).any()
    right_has_info = (img[:, 100:] == m.COLORATION_INFO_BG).all(axis=-1).any()
    assert left_has_info and right_has_info


def test_draw_coloration_info_does_not_crash_on_a_tiny_image():
    img = np.full((15, 20, 3), 50, dtype=np.uint8)
    v = m.ColorationConfig(0)
    v.chan_src[1] = m.CHAN_SRC_BUDDHA
    m.draw_coloration_info(img, [v], 20)   # must not raise


def _text_row_extent(img, background):
    """How many rows contain any non-background pixel - a rough proxy for
    the overlay's total rendered height."""
    return (img != background).any(axis=-1).any(axis=1).sum()


def test_coloration_info_text_scales_up_above_size_512():
    v = m.ColorationConfig(0)
    img_1x = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img_1x, [v], m.SIZE)
    img_2x = np.full((m.SIZE * 2, m.SIZE * 2, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img_2x, [v], m.SIZE * 2)
    h1 = _text_row_extent(img_1x, 128)
    h2 = _text_row_extent(img_2x, 128)
    assert h2 == h1 * 2, \
        f"text should be exactly 2x taller at 2x the coloration resolution, got {h1} vs {h2}"


def test_coloration_info_text_never_shrinks_below_the_size_512_baseline():
    v = m.ColorationConfig(0)
    img_baseline = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img_baseline, [v], m.SIZE)
    img_small = np.full((200, 200, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img_small, [v], 200)
    assert _text_row_extent(img_small, 128) == _text_row_extent(img_baseline, 128), \
        "text must stay at the baseline size for any resolution at or below SIZE"


def test_coloration_info_text_scale_uses_the_smaller_of_width_and_height():
    """An unusually elongated export (very wide, short coloration) shouldn't get
    oversized text just because one dimension is large."""
    v = m.ColorationConfig(0)
    img_square = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    m.draw_coloration_info(img_square, [v], m.SIZE)
    img_wide_short = np.full((m.SIZE, m.SIZE * 4, 3), 128, dtype=np.uint8)   # short height
    m.draw_coloration_info(img_wide_short, [v], m.SIZE * 4)
    assert _text_row_extent(img_wide_short, 128) == _text_row_extent(img_square, 128), \
        "text size should follow the smaller dimension (height here), not the wide one"


def test_bottom_margin_text_scales_with_the_colorations_own_size_not_the_margin():
    """Regression test: the margin's own height is only a quarter of
    coloration_width by design, but the text size must track coloration_width itself -
    it was previously computed from the (always smaller) margin height,
    which meant the text barely grew until an absurdly large export
    (coloration_width beyond ~3072) rather than scaling normally."""
    v = m.ColorationConfig(0)
    img_512 = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    out_512 = m.add_bottom_color_margin(img_512, [v], m.SIZE)
    h1 = _text_row_extent(out_512[m.SIZE:], m.COLORATION_INFO_BG)

    img_1080 = np.full((1080, 1080, 3), 128, dtype=np.uint8)
    out_1080 = m.add_bottom_color_margin(img_1080, [v], 1080)
    h2 = _text_row_extent(out_1080[1080:], m.COLORATION_INFO_BG)

    assert h2 == h1 * 2, f"expected exactly 2x text height at 1080 vs 512, got {h1} -> {h2}"


def test_left_margin_text_scales_with_the_colorations_own_size_not_the_margin():
    """Measures vertical extent (line height/spacing), not horizontal -
    horizontal extent can be legitimately clipped once text no longer fits
    the (narrow) margin width at a larger scale, which isn't itself a bug."""
    settings = _generation()
    img_512 = np.full((m.SIZE, m.SIZE, 3), 128, dtype=np.uint8)
    out_512 = m.add_left_info_margin(img_512, m.figure_info_blocks(*settings, _start_scene()))
    margin_w_512 = m.SIZE // 2
    h1 = (out_512[:, :margin_w_512] != m.COLORATION_INFO_BG).any(axis=-1).any(axis=1).sum()

    img_1080 = np.full((1080, 1080, 3), 128, dtype=np.uint8)
    out_1080 = m.add_left_info_margin(img_1080, m.figure_info_blocks(*settings, _start_scene()))
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
    coloration_snapshot = [m.snapshot_coloration(v) for v in app.colorations_to_export()]
    colorations = m.resolve_colorations(coloration_snapshot, keyframes, 0)

    rgb = app.render_full_image(data, 40, 40, obst_arrays, colorations, m.DEFAULT_FRAMING)
    rgb = m.apply_info_locations(rgb, colorations, 40, data.ray, data.natural_termination, [],
                                 "Overlay", "None", m.DEFAULT_FRAMING, "None", mutate_ok=True)
    assert (rgb == m.COLORATION_INFO_BG).all(axis=-1).any()


def test_video_render_worker_adds_figure_settings_left_margin_when_enabled():
    app = _fake_export_app()
    kf = m.Keyframe(0, 0, dict(m.DEFAULT_SETTING_VALUES))
    keyframes = [kf]
    obst_snapshot = [o.snapshot() for o in app.obstacles]
    data = m.resolve_frame_settings(keyframes, 0)
    obst_arrays = m.resolve_obstacle_arrays(obst_snapshot, keyframes, 0)
    coloration_snapshot = [m.snapshot_coloration(v) for v in app.colorations_to_export()]
    colorations = m.resolve_colorations(coloration_snapshot, keyframes, 0)

    rgb = app.render_full_image(data, 40, 40, obst_arrays, colorations, m.DEFAULT_FRAMING)
    original_width = rgb.shape[1]
    rgb = m.apply_info_locations(rgb, colorations, 40, data.ray, data.natural_termination,
                                 _start_scene(), "None", "Left margin", m.DEFAULT_FRAMING, "None", mutate_ok=True)
    assert rgb.shape[1] >= original_width + 40 // 2       # (more, at this tiny size)
    assert (rgb == m.COLORATION_INFO_BG).all(axis=-1).any()


def test_color_info_and_figure_settings_locations_default_to_none():
    app = _fake_export_app()
    assert app.color_info_location_var.get() == "None"
    assert app.figure_info_location_var.get() == "None"


def test_apply_info_locations_none_leaves_the_image_unchanged():
    settings = _generation()
    v = m.ColorationConfig(0)
    img = np.full((40, 40, 3), 77, dtype=np.uint8)
    out = m.apply_info_locations(img, [v], 40, *settings, [], "None", "None", m.DEFAULT_FRAMING, "None", mutate_ok=True)
    assert out.shape == img.shape
    assert (out == 77).all()


def test_apply_info_locations_margin_cases_never_mutate_when_not_ok():
    """Bottom/left margins always return a new array anyway (they can't
    mutate in place - the image gets bigger), but this locks in that
    mutate_ok=False is honored for "Overlay" specifically, which otherwise
    would draw straight into the caller's array."""
    settings = _generation()
    v = m.ColorationConfig(0)
    img = np.full((40, 40, 3), 77, dtype=np.uint8)
    original = img.copy()
    m.apply_info_locations(img, [v], 40, *settings, [], "Overlay", "None", m.DEFAULT_FRAMING, "None", mutate_ok=False)
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
    assert (img[:m.SIZE] == 42).all(), "the original coloration content must be untouched"


def test_write_with_left_margin_produces_a_wider_image():
    app = _fake_current_image_app()
    app.figure_info_location_var = _StubVar("Left margin")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    assert img.shape == (m.SIZE, m.SIZE + m.SIZE // 2, 3)
    assert (img[:, :m.SIZE // 2] == m.COLORATION_INFO_BG).all(axis=-1).any()
    assert (img[:, m.SIZE // 2:] == 42).all(), "the original coloration content must be untouched"


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
# The sequence preview's Corkscrew tab
# --------------------------------------------------------------------------- #
def _corkscrew_path():
    """A path of five points: right, up (a segment of no length between), left, down-left."""
    path_points = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 2.0], [-1.5, 2.0], [-2.5, 0.0]])
    path_point_tags = np.array([m.TAG_START, m.TAG_BOUNDARY, m.TAG_SELF, m.TAG_SELF, m.TAG_BOUNDARY, m.TAG_ESCAPED])
    return path_points, path_point_tags


def test_the_corkscrew_has_every_segment_one_unit_long_pointing_the_way_the_real_segment_points():
    path_points, path_point_tags = _corkscrew_path()
    vertices, start_tags, end_tags = m.corkscrew_steps(path_points, path_point_tags, len(path_points))
    steps = np.diff(vertices, axis=0)
    assert len(steps) == 4, "the segment of no length has no direction, so it is left out"
    assert np.allclose(np.hypot(steps[:, 0], steps[:, 1]), 1.0)
    real = np.diff(path_points, axis=0)
    real = real[np.hypot(real[:, 0], real[:, 1]) > 1e-12]
    assert np.allclose(steps, real / np.hypot(real[:, 0], real[:, 1])[:, None])
    assert np.allclose(vertices[0], 0.0)
    assert list(start_tags) == [m.TAG_START, m.TAG_SELF, m.TAG_SELF, m.TAG_BOUNDARY]
    assert list(end_tags) == [m.TAG_BOUNDARY, m.TAG_SELF, m.TAG_BOUNDARY, m.TAG_ESCAPED]


def test_the_corkscrew_of_a_sequence_with_no_segment_is_empty():
    for n in (0, 1):
        vertices, start_tags, end_tags = m.corkscrew_steps(np.zeros((3, 2)), np.zeros(3), n)
        assert vertices.shape == (1, 2) and len(start_tags) == len(end_tags) == 0
    vertices, start_tags, _end = m.corkscrew_steps(np.array([[1.0, 1.0], [1.0, 1.0]]), np.zeros(2), 2)
    assert len(start_tags) == 0, "a sequence that never moves has no direction to draw"


def test_the_corkscrew_starts_at_the_centre_and_stops_just_short_of_the_edge_at_any_canvas_size():
    rng = np.random.RandomState(4)
    for side in (80, 301, 640):
        for trial in range(20):
            n = int(rng.randint(2, 60))
            path_points = np.cumsum(rng.normal(size=(n, 2)) * rng.choice([0.01, 1.0, 300.0]), axis=0)
            vertices, _s, _e = m.corkscrew_steps(path_points, np.zeros(n), n)
            xy, segment = m.corkscrew_fit(vertices, side)
            margin = max(m.CORKSCREW_MARGIN_MIN, side * m.CORKSCREW_MARGIN)
            assert np.allclose(xy[0], side / 2.0), "starts at the centre"
            assert xy.min() >= margin - 1e-9 and xy.max() <= side - margin + 1e-9, "never touches the edge"
            assert abs(max(np.abs(xy - side / 2.0).max(), 0.0) - (side / 2.0 - margin)) < 1e-9, \
                "and goes almost to it: the farthest point is exactly a margin short"
            lengths = np.hypot(*np.diff(xy, axis=0).T)
            assert np.allclose(lengths, segment), "every segment the same length, in pixels"


def test_a_segment_pointing_up_in_the_sequence_points_up_on_the_canvas():
    xy, _segment = m.corkscrew_fit(np.array([[0.0, 0.0], [0.0, 1.0]]), 200)
    assert xy[1][1] < xy[0][1] and xy[1][0] == xy[0][0]


def test_the_corkscrew_canvas_is_square_as_tall_as_the_tab_allows_and_leaves_room_for_the_text():
    tall = m.corkscrew_side(512, 330)
    assert tall == 330 - 2 * m.PAD - 2, "as tall as the tab"
    assert m.corkscrew_side(512, 330) > m.corkscrew_side(512, 250)
    assert m.corkscrew_side(300, 600) == 300 - m.CORKSCREW_TEXT_W - 3 * m.PAD, "a narrow tab keeps the text's room"
    assert m.corkscrew_side(10, 10) == m.CORKSCREW_SIDE_MIN


def test_the_sequence_preview_has_a_bar_graph_and_a_corkscrew_tab_with_the_text_left_of_the_canvas():
    import inspect
    layout = inspect.getsource(m.App.build_info_quadrant)
    assert 'text="Sequence preview"' in layout and "Ray preview" not in layout
    assert 'text="Numbers"' in layout and 'text="Bar graph"' in layout and 'text="Corkscrew"' in layout
    assert '"Graphics"' not in layout
    tab = inspect.getsource(m.App.build_corkscrew_tab)
    assert tab.index("text.grid(row=0, column=0") < tab.index("self.corkscrew_canvas.grid(row=0, column=1")
    assert 'bg="black"' in tab and "tab.rowconfigure(0, weight=1)" in tab and '"<Configure>"' in tab


class _RecordingCanvas:
    def __init__(self):
        self.lines = []

    def delete(self, _tag):
        self.lines = []

    def create_line(self, x0, y0, x1, y1, **options):
        self.lines.append(((x0, y0, x1, y1), options))

    def configure(self, **options):
        self.options = options


def _fake_corkscrew_app(side=300):
    class FakeApp:
        pass
    app = FakeApp()
    app.corkscrew_side = side
    app.corkscrew_canvas = _RecordingCanvas()
    for name in ("draw_corkscrew", "redraw_corkscrew", "clear_corkscrew", "on_corkscrew_tab_resized"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    app.clear_corkscrew()
    return app


def test_drawing_a_corkscrew_lays_the_lines_end_to_end_from_the_centre_in_the_bar_graphs_colours():
    app = _fake_corkscrew_app(300)
    path_points, path_point_tags = _corkscrew_path()
    app.draw_corkscrew(path_points, path_point_tags, len(path_points))
    lines = app.corkscrew_canvas.lines
    ends = [(l[0][:2], l[0][2:]) for l in lines]
    assert np.allclose(ends[0][0], (150, 150)), "the first line starts at the centre"
    for (_a, end), (start, _b) in zip(ends, ends[1:]):
        assert np.allclose(end, start), "each line starts where the one before ended"
    assert all(0 < x < 300 and 0 < y < 300 for line in lines for x, y in (line[0][:2], line[0][2:]))
    colors = [l[1]["fill"] for l in lines]
    boundary, self_, escaped = (m.lerp_color(m.TAG_COLOR[k], m.TAG_COLOR[k], 0.0)
                                for k in (m.TAG_BOUNDARY, m.TAG_SELF, m.TAG_ESCAPED))
    assert self_ in colors, "a segment between two points of the same tag is one solid colour"
    assert any(c not in (boundary, self_, escaped) and c != m.lerp_color(m.TAG_COLOR[m.TAG_START], m.TAG_COLOR[m.TAG_START], 0.0)
               for c in colors), "a segment between two differently tagged points fades from one colour to the other"
    assert lines[0][1]["width"] == m.CORKSCREW_LINE_WIDTH


def test_a_segment_fades_from_its_start_tags_colour_to_its_end_tags_colour():
    app = _fake_corkscrew_app(300)
    path_points = np.array([[0.0, 0.0], [1.0, 0.0]])
    app.draw_corkscrew(path_points, np.array([m.TAG_START, m.TAG_ESCAPED]), 2)
    colors = [l[1]["fill"] for l in app.corkscrew_canvas.lines]
    assert len(colors) >= 2
    start, end = m.TAG_COLOR[m.TAG_START], m.TAG_COLOR[m.TAG_ESCAPED]
    channel = lambda c, i: int(c[1 + 2 * i:3 + 2 * i], 16)
    assert abs(channel(colors[0], 1) - start[1]) < abs(channel(colors[-1], 1) - start[1])
    assert abs(channel(colors[-1], 1) - end[1]) < abs(channel(colors[0], 1) - end[1])


def test_resizing_the_tab_keeps_the_canvas_square_and_redraws_the_corkscrew_to_fit():
    import types
    app = _fake_corkscrew_app(300)
    path_points, path_point_tags = _corkscrew_path()
    app.draw_corkscrew(path_points, path_point_tags, len(path_points))
    before = list(app.corkscrew_canvas.lines)
    app.on_corkscrew_tab_resized(types.SimpleNamespace(width=512, height=200))
    side = m.corkscrew_side(512, 200)
    assert app.corkscrew_side == side and app.corkscrew_canvas.options == {"width": side, "height": side}
    after = app.corkscrew_canvas.lines
    assert len(after) == len(before) and after != before
    assert np.allclose(after[0][0][:2], side / 2.0)
    app.on_corkscrew_tab_resized(types.SimpleNamespace(width=512, height=200))      # no change: nothing redrawn
    assert app.corkscrew_canvas.lines == after


def test_clearing_the_preview_clears_the_corkscrew_too():
    app = _fake_corkscrew_app(300)
    path_points, path_point_tags = _corkscrew_path()
    app.draw_corkscrew(path_points, path_point_tags, len(path_points))
    assert app.corkscrew_canvas.lines
    app.clear_corkscrew()
    assert app.corkscrew_canvas.lines == []
    app.on_corkscrew_tab_resized(__import__("types").SimpleNamespace(width=512, height=150))   # a resize still draws nothing
    assert app.corkscrew_canvas.lines == []


def test_previewing_a_ray_draws_its_corkscrew():
    app = _fake_preview_app(pinned=(100, 100))
    app.corkscrews = []
    app.draw_corkscrew = lambda path_points, path_point_tags, n: app.corkscrews.append((path_points[:n].copy(), path_point_tags[:n].copy(), n))
    app.update_preview(100, 100)
    assert len(app.corkscrews) == 1 and app.corkscrews[0][2] >= 1
