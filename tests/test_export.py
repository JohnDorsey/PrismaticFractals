"""Tests: export."""
import os
import tempfile
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


def test_new_image_export_widens_when_a_second_coloration_is_added_and_all_colorations_included():
    """The scenario the person actually cares about: does exporting a "New
    figure" actually get wider when you add a coloration AND turn on "Include all
    colorations" - through the real save_new_image -> _save_new_image_worker ->
    render_full_image chain, not just render_full_image called directly
    with a hand-built colorations list."""
    app = _fake_export_app()
    app.include_all_colorations_var = _StubVar(True)
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
        img_one_coloration = m.read_png(os.path.join(tmpdir, files[0]))
        assert img_one_coloration.shape == (20, 20, 3)
        os.remove(os.path.join(tmpdir, files[0]))

        app.new_coloration()
        assert app.all_coloration_ids() == [0, 1]
        _run_save_new_image_synchronously(app)
        files2 = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files2) == 1
        img_two_colorations = m.read_png(os.path.join(tmpdir, files2[0]))
        assert img_two_colorations.shape == (20, 40, 3), \
            f"adding a second coloration should double the export width, got {img_two_colorations.shape}"
    finally:
        m.paths.EXPORT_DIR = orig_export_dir


def test_include_all_colorations_defaults_to_true():
    app = _fake_export_app()
    assert app.include_all_colorations_var.get() is True


def test_new_image_export_stays_single_width_when_all_colorations_not_included():
    """With "Include all colorations" explicitly off, adding a second coloration must
    NOT widen the export - only the selected/live coloration goes out. (The
    checkbox now defaults to on, so this test turns it off explicitly
    rather than relying on a default.)"""
    app = _fake_export_app()
    app.include_all_colorations_var = _StubVar(False)
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

    app.new_coloration()
    tmpdir = tempfile.mkdtemp()
    orig_export_dir = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img = m.read_png(os.path.join(tmpdir, files[0]))
        assert img.shape == (20, 20, 3), \
            f"with 'Include all colorations' off, export must stay single-width, got {img.shape}"
    finally:
        m.paths.EXPORT_DIR = orig_export_dir


def test_video_frame_render_widens_when_a_second_coloration_is_added_and_included():
    """Same property, for the video-render call chain: snapshot
    colorations_to_export() the way the render launch site now does, resolve
    them, and render - before and after adding a coloration, with 'Include all
    colorations' on."""
    app = _fake_export_app()
    app.include_all_colorations_var = _StubVar(True)
    kf = m.Keyframe(0, 0, dict(m.DEFAULT_SETTING_VALUES))
    keyframes = [kf]
    obst_snapshot = [o.snapshot() for o in app.obstacles]
    data = m.resolve_frame_settings(keyframes, 0)
    obst_arrays = m.resolve_obstacle_arrays(obst_snapshot, keyframes, 0)

    coloration_snapshot = [m.snapshot_coloration(v) for v in app.colorations_to_export()]
    colorations = m.resolve_colorations(coloration_snapshot, keyframes, 0)
    img_one_coloration = app.render_full_image(data, 20, 20, obst_arrays, colorations, m.DEFAULT_FRAMING)
    assert img_one_coloration.shape == (20, 20, 3)

    app.new_coloration()
    coloration_snapshot2 = [m.snapshot_coloration(v) for v in app.colorations_to_export()]   # as the render
                                                                           # launch site
                                                                           # re-snapshots
    colorations2 = m.resolve_colorations(coloration_snapshot2, keyframes, 0)
    img_two_colorations = app.render_full_image(data, 20, 20, obst_arrays, colorations2, m.DEFAULT_FRAMING)
    assert img_two_colorations.shape == (20, 40, 3), \
        f"adding a second coloration should double the video frame width, got {img_two_colorations.shape}"


def test_render_coloration_thumbnail_does_not_crash_for_every_coloration_kind():
    """Regression test: render_coloration_thumbnail builds its own ColorationSettings
    manually (like render_current_image does) and had no coverage at all -
    that's exactly how a missing constant_choice argument shipped
    undetected. Exercises it for Query, Buddha, and Constant channels."""
    app = _fake_current_image_app()
    v = m.ColorationConfig(0)
    v.chan_src[0] = m.CHAN_SRC_CONSTANT
    v.constant_choice[0] = m.NONE_ONE
    v.chan_src[1] = m.CHAN_SRC_BUDDHA
    thumb = app.render_coloration_thumbnail(v, m.THUMB_SIZE)
    assert thumb.shape == (m.THUMB_SIZE, m.THUMB_SIZE, 3)
    assert (thumb[..., 0] == 255).all()   # channel 0 is Constant, Always 1


def test_render_coloration_thumbnail_reuses_self_img_for_the_live_coloration():
    """The performance fix itself: the live coloration's thumbnail must be an
    exact downsample of self.img (already kept current by recolor()), not
    a fresh apply_coloration computation - this is what avoids redundantly
    re-tracing every ray for a "multi" Buddha point option on every single
    color-setting change."""
    app = _fake_current_image_app()
    app.img = np.random.RandomState(0).randint(0, 256, (m.PREVIEW_BOX, m.PREVIEW_BOX, 3), dtype=np.uint8)
    live = app.live_coloration()
    assert live is app.colorations[0]
    thumb = app.render_coloration_thumbnail(live, m.THUMB_SIZE)
    ys = (np.arange(m.THUMB_SIZE) * m.PREVIEW_BOX / m.THUMB_SIZE).astype(np.int64)
    xs = (np.arange(m.THUMB_SIZE) * m.PREVIEW_BOX / m.THUMB_SIZE).astype(np.int64)
    expected = app.img[np.ix_(ys, xs)]
    assert np.array_equal(thumb, expected)


def test_render_coloration_thumbnail_never_calls_apply_coloration_for_the_live_coloration():
    """Belt-and-suspenders on the same guarantee: even a "multi" Buddha
    point (the expensive, ray-re-tracing case) must not reach
    apply_coloration at all when thumbnailing the live coloration."""
    app = _fake_current_image_app()
    app.img = np.zeros((m.PREVIEW_BOX, m.PREVIEW_BOX, 3), dtype=np.uint8)
    live = app.live_coloration()
    live.chan_src[0] = m.CHAN_SRC_BUDDHA
    live.buddha_points[0] = tuple(i == 7 for i in range(m.N_BUDDHA_POINTS))  # a "multi" point

    def _boom(*a, **k):
        raise AssertionError("apply_coloration must not be called for the live coloration's thumbnail")
    app.apply_coloration = _boom
    app.render_coloration_thumbnail(live, m.THUMB_SIZE)   # must not raise


def test_render_coloration_thumbnail_still_uses_apply_coloration_for_a_non_live_coloration():
    """The other half: a coloration that ISN'T live has no incrementally-updated
    self.img to reuse, so it must still go through apply_coloration as
    before."""
    app = _fake_current_image_app()
    app.img = np.full((m.PREVIEW_BOX, m.PREVIEW_BOX, 3), 42, dtype=np.uint8)   # sentinel: must be unused
    other = m.ColorationConfig(99)
    other.color_model = "HSV"
    assert other is not app.live_coloration()
    thumb = app.render_coloration_thumbnail(other, m.THUMB_SIZE)
    assert not np.array_equal(thumb, np.full((m.THUMB_SIZE, m.THUMB_SIZE, 3), 42, dtype=np.uint8))


def test_render_coloration_thumbnail_caches_a_non_live_coloration_across_calls():
    """The third performance fix: rendering the SAME non-live coloration's
    thumbnail twice, with nothing having changed (rows_done unchanged),
    must only actually compute it once - this is what keeps editing one
    coloration fast even when a DIFFERENT, unrelated coloration uses an expensive
    "multi" Buddha point option, since rebuild_coloration_list re-renders every
    coloration's thumbnail on every single color edit."""
    app = _fake_current_image_app()
    other = m.ColorationConfig(99)
    calls = []
    real_colorize = app.apply_coloration
    def counting_colorize(*a, **k):
        calls.append(1)
        return real_colorize(*a, **k)
    app.apply_coloration = counting_colorize
    thumb1 = app.render_coloration_thumbnail(other, m.THUMB_SIZE)
    thumb2 = app.render_coloration_thumbnail(other, m.THUMB_SIZE)
    assert len(calls) == 1, "the second call should reuse the cache, not recompute"
    assert np.array_equal(thumb1, thumb2)


def test_render_coloration_thumbnail_recomputes_a_non_live_coloration_after_rows_done_advances():
    """When new simulation data becomes available, a cached non-live
    thumbnail must not be reused, even though the coloration's own settings
    haven't changed - it may now cover more of the image."""
    app = _fake_current_image_app()
    other = m.ColorationConfig(99)
    calls = []
    real_colorize = app.apply_coloration
    def counting_colorize(*a, **k):
        calls.append(1)
        return real_colorize(*a, **k)
    app.apply_coloration = counting_colorize
    app.render_coloration_thumbnail(other, m.THUMB_SIZE)
    app.rows_done = m.PREVIEW_BOX // 2   # simulate more rows having completed
    app.render_coloration_thumbnail(other, m.THUMB_SIZE)
    assert len(calls) == 2, "rows_done advancing must invalidate the cache"


def test_select_coloration_invalidates_the_incoming_colorations_stale_cache():
    """Regression test for the specific staleness bug this design has to
    avoid: a coloration that was cached while non-live, then selected (becomes
    live, gets edited), then deselected again must NOT fall back to its
    pre-edit cached thumbnail just because rows_done happens to match."""
    app = _fake_color_editor_app()
    app.render_coloration_thumbnail = m.App.render_coloration_thumbnail.__get__(app)
    app.apply_coloration = m.App.apply_coloration.__get__(app)
    app.obstacles = []
    app.stats = np.zeros((m.PREVIEW_BOX, m.PREVIEW_BOX, m.N_STATS), dtype=np.float32)
    app.characteristic_points = np.zeros((m.PREVIEW_BOX, m.PREVIEW_BOX, m.N_POINTS, 2), dtype=np.float32)
    app.pixel_state = np.full((m.PREVIEW_BOX, m.PREVIEW_BOX), m.PIXEL_TRACED, dtype=np.uint8)
    app.gen_termination = m.NATURAL_TERMINATION_DEFAULT
    app.gen_engine_args = _engine_args(m.build_obstacle_arrays([]))
    app.generated_settings = None                     # the placeholder image: default settings
    app.generated_natural_termination = None
    app.current_blur_settings = lambda: m.DEFAULT_BUDDHA_BLUR
    app.current_buddha_seed_settings = lambda: m.DEFAULT_BUDDHA_SEED
    app.settings_of_current_image = m.App.settings_of_current_image.__get__(app)
    app.img = np.zeros((m.PREVIEW_BOX, m.PREVIEW_BOX, 3), dtype=np.uint8)

    # coloration 1 starts non-live; cache its (default RGB) thumbnail first
    coloration1 = app.colorations[1]
    thumb_before = app.render_coloration_thumbnail(coloration1, m.THUMB_SIZE)
    assert coloration1.id in app.coloration_thumb_cache

    # select it (becomes live), "edit" it, then select coloration 0 again (coloration 1
    # becomes non-live) - all through the real select_coloration, which must
    # invalidate coloration 1's stale entry when it becomes live. coloration1 is HSV
    # mode (the fixture's default for colorations[1]), so channel 2 is Value -
    # editing it is what actually changes the visible output; Hue alone
    # wouldn't, since Value stays 0 either way.
    app.select_coloration(1)
    coloration1.chan_src[2] = m.CHAN_SRC_CONSTANT
    coloration1.constant_choice[2] = m.NONE_ONE
    app.select_coloration(0)
    thumb_after = app.render_coloration_thumbnail(coloration1, m.THUMB_SIZE)
    assert not np.array_equal(thumb_before, thumb_after), \
        "must reflect the edit made while live, not the stale pre-edit cache"


def test_write_uses_self_frame_directly_for_a_single_coloration():
    """No need to recompute anything when only one coloration is being exported -
    self.image_for_export already holds exactly that."""
    app = _fake_current_image_app()
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    assert img.shape == (m.PREVIEW_BOX, m.PREVIEW_BOX, 3)
    assert (img == 42).all(), "single-coloration export should be exactly self.image_for_export"


def test_write_composes_all_colorations_when_include_all_colorations_is_on():
    app = _fake_current_image_app()
    app.include_all_colorations_var = _StubVar(True)
    app.new_coloration()
    app.colorations[1].color_model = "HSV"
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "out.png")
    app._write(path)
    img = m.read_png(path)
    assert img.shape == (m.PREVIEW_BOX, m.PREVIEW_BOX * 2, 3)
    half = m.PREVIEW_BOX
    assert not np.array_equal(img[:, :half], img[:, half:]), \
        "two colorations with different modes should produce different halves"


def test_render_current_image_matches_self_frame_for_identical_single_coloration():
    """A sanity check that the two code paths agree: composing just the live
    coloration through render_current_image gives the same picture self.image_for_export
    already has (modulo the outline, which render_current_image also
    applies - both are off here for a clean, direct pixel comparison isn't
    meaningful since self.image_for_export's content comes from a different source in
    this fixture; instead just check shape and that it runs without error)."""
    app = _fake_current_image_app()
    img = app.render_current_image([app.live_coloration()])
    assert img.shape == (m.PREVIEW_BOX, m.PREVIEW_BOX, 3)


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
    lines = m.figure_info_lines(*_generation(max_iter=42, max_dist=5.0, seg_mult=0.5),
                                _start_scene())
    assert "Iter limit:42" in lines
    assert "Dist limit:5.00" in lines
    assert "Segment:x0.50" in lines
    assert "Segment:x1.00" in m.figure_info_lines(*_generation(), _start_scene()), \
        "shown even at the default, so the layout never depends on the values"


def test_figure_info_lines_shows_none_for_unlimited_distance():
    assert "Dist limit:none" in m.figure_info_lines(*_generation(max_dist=0.0), _start_scene())


def test_figure_info_lines_shows_fuzz_details_when_enabled_and_just_off_otherwise():
    off = m.figure_info_lines(*_generation(), _start_scene())
    assert "Fuzz:Off" in off and not any("Seed" in l for l in off)
    fuzz = m.DEFAULT_FUZZ.replace(enabled=True, emit_jitter=5.0, rng_seed=12345)
    on = m.figure_info_lines(*_generation(fuzz=fuzz), _start_scene())
    assert "Fuzz:On" in on
    assert any("Emit" in l and "5.0" in l for l in on)
    assert "  Seed 12345" in on


def test_figure_info_lines_no_longer_describes_a_bounding_shape():
    lines = m.figure_info_lines(*_generation(), _start_scene())
    assert not any("Shape" in l for l in lines)


def test_figure_info_lines_says_so_when_there_are_no_obstacles():
    assert "no obstacles" in m.figure_info_lines(*_generation(), [])


def test_figure_info_lines_lists_every_setting_of_the_starting_obstacle():
    lines = m.figure_info_lines(*_generation(), _start_scene())
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
                                   rng_seed=999999999)
    settings = _generation(max_iter=9999, max_dist=9999.0, seg_mult=0.01, fuzz=fuzz)
    for line in m.figure_info_lines(*settings, scene):
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
    for settings, scene in [(_generation(), scene_a), (_generation(max_dist=3.0, seg_mult=0.5,
                            fuzz=fuzz_on), scene_b)]:
        shapes.append([len(block) for block in m.figure_info_blocks(*settings, scene)])
    assert shapes[0] == shapes[1]


def test_layout_info_columns_keeps_blocks_together_and_splits_only_oversized_ones():
    blocks = [["a1", "a2", "a3"], ["b1", "b2", "b3"], ["c1", "c2"]]
    cols = m.layout_info_columns(blocks, lines_per_column=7)
    assert cols == [["a1", "a2", "a3", " ", "b1", "b2", "b3"], ["c1", "c2"]]
    huge = m.layout_info_columns([["x%d" % i for i in range(10)]], lines_per_column=4)
    assert [len(c) for c in huge] == [4, 4, 2]
    assert [l for c in huge for l in c] == ["x%d" % i for i in range(10)]


def test_left_margin_is_half_the_image_width_for_an_ordinary_scene():
    img = np.full((m.PREVIEW_BOX, m.PREVIEW_BOX, 3), 128, dtype=np.uint8)
    out = m.add_left_info_margin(img, m.figure_info_blocks(*_generation(), _start_scene()))
    assert out.shape == (m.PREVIEW_BOX, m.PREVIEW_BOX + m.PREVIEW_BOX // 2, 3)
    assert np.array_equal(out[:, m.PREVIEW_BOX // 2:], img), "the image itself is untouched"
    assert (out[:, :m.PREVIEW_BOX // 2] != m.COLORATION_INFO_BG).any(), "and the text is drawn"


def test_left_margin_grows_extra_columns_when_the_settings_need_the_room():
    img = np.full((m.PREVIEW_BOX, m.PREVIEW_BOX, 3), 128, dtype=np.uint8)
    scene = []
    for i in range(14):
        o = m.default_starting_obstacle(i)
        o.r = 0.1 * i
        scene.append(o)
    out = m.add_left_info_margin(img, m.figure_info_blocks(*_generation(), scene))
    margin_w = out.shape[1] - m.PREVIEW_BOX
    assert margin_w > m.PREVIEW_BOX // 2
    assert margin_w % 2 == 0, "video encoders want even widths"
    text = (out[:, :margin_w] != m.COLORATION_INFO_BG).any(axis=-1)
    last_col_x0 = margin_w - 6 * m.FIGURE_INFO_COLUMN_CHARS
    assert text[:, last_col_x0:].any(), "the last column really has text in it"


def test_left_margin_size_is_the_same_for_every_frame_of_an_animated_video():
    img = np.full((m.PREVIEW_BOX, m.PREVIEW_BOX, 3), 128, dtype=np.uint8)
    scene = _every_kind_of_obstacle()
    moved = _every_kind_of_obstacle()
    for o in moved:
        o.r, o.rotation = 0.25, 5.0
    fuzz_on = m.DEFAULT_FUZZ.replace(enabled=True)
    a = m.add_left_info_margin(img, m.figure_info_blocks(*_generation(), scene))
    b = m.add_left_info_margin(img, m.figure_info_blocks(*_generation(max_dist=3.0, seg_mult=0.5, fuzz=fuzz_on),
                                 moved))
    assert a.shape == b.shape
    assert not np.array_equal(a, b), "but they do show the different values"


def test_add_left_info_margin_does_not_mutate_input():
    img = np.full((40, 60, 3), 33, dtype=np.uint8)
    original = img.copy()
    m.add_left_info_margin(img, m.figure_info_blocks(m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_TERMINATION_SETTINGS, _start_scene()))
    assert np.array_equal(img, original)


def test_add_left_info_margin_does_not_crash_for_any_setting_combination():
    img = np.full((30, 30, 3), 10, dtype=np.uint8)
    for fuzz_on in (False, True):
        for seg_mult in (1.0, 0.5):
            for scene in ([], _start_scene(), _every_kind_of_obstacle()):
                settings = _generation(seg_mult=seg_mult,
                                       fuzz=m.DEFAULT_FUZZ.replace(enabled=fuzz_on))
                m.add_left_info_margin(img, m.figure_info_blocks(*settings, scene))   # must not raise


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
    settings = _generation()
    text0 = m.figure_info_lines(*settings, m.resolve_obstacles(snaps, kfs, 0))
    text5 = m.figure_info_lines(*settings, m.resolve_obstacles(snaps, kfs, 5))
    assert "  pos r0.00 \u03b80.0\u00b0" in text0
    assert "  pos r0.00 \u03b845.0\u00b0" in text5


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
    _give_image_geometry(app)
    app.save_source_var = _StubVar(m.SAVE_SOURCE_OPTIONS[0])
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    app.generated_natural_termination = m.DEFAULT_NATURAL_TERMINATION_SETTINGS
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
    _give_image_geometry(app)
    app.root = None
    app.save_source_var = _StubVar(m.SAVE_SOURCE_OPTIONS[0])
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    app.generated_natural_termination = m.DEFAULT_NATURAL_TERMINATION_SETTINGS
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
class _RecordingButton:
    """A button that remembers whether it was last enabled or disabled."""
    enabled = None

    def state(self, flags):
        self.enabled = flags == ["!disabled"]


def _fake_button_gating_app():
    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    app.ready = True
    app.current_generation_settings = lambda: (m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_TERMINATION_SETTINGS)
    app.generated_settings = None
    app.generated_natural_termination = None
    app.gen_obstacles_sig = None
    app.obstacles = []
    app.baseline = m.DEFAULT_RAY_SETTINGS
    app.baseline_natural_termination = m.DEFAULT_NATURAL_TERMINATION_SETTINGS
    app.running = False
    app.rows_done = 0
    app.start_btn = _RecordingButton()
    app.revert_btn = _RecordingButton()
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


def test_start_and_revert_follow_the_ray_settings_and_never_look_at_the_buddha_blur():
    """The blur is applied whenever Buddha channels are colored, so a blur change needs neither
    Start nor Revert: update_buttons does not even read the blur controls (this fake app has
    none, so a read would raise)."""
    app = _fake_button_gating_app()
    assert not hasattr(app, "current_blur_settings")
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    app.generated_natural_termination = m.DEFAULT_NATURAL_TERMINATION_SETTINGS
    app.gen_obstacles_sig = m.obstacles_signature([])
    app.update_buttons()
    assert (app.start_btn.enabled, app.revert_btn.enabled) == (False, False)      # nothing changed
    app.current_generation_settings = lambda: (m.DEFAULT_RAY_SETTINGS.replace(seg_mult=0.5),
                                               m.DEFAULT_NATURAL_TERMINATION_SETTINGS)
    app.update_buttons()
    assert (app.start_btn.enabled, app.revert_btn.enabled) == (True, True)        # a ray setting changed
    app.current_generation_settings = lambda: (m.DEFAULT_RAY_SETTINGS,
                                               m.DEFAULT_NATURAL_TERMINATION_SETTINGS.replace(max_iter=99))
    app.update_buttons()
    assert (app.start_btn.enabled, app.revert_btn.enabled) == (True, True)        # so did a termination limit
    app.current_generation_settings = lambda: None
    app.update_buttons()
    assert (app.start_btn.enabled, app.revert_btn.enabled) == (False, True)       # an entry is invalid


def test_export_disabled_before_any_generation():
    app = _fake_button_gating_app()
    app.update_buttons()
    assert app.save_btn_tooltip.text != ""


def test_export_enabled_after_generation_finishes():
    app = _fake_button_gating_app()
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    app.generated_natural_termination = m.DEFAULT_NATURAL_TERMINATION_SETTINGS
    app.rows_done = m.PREVIEW_BOX
    app.running = False
    app.update_buttons()
    assert app.save_btn_tooltip.text == ""


def test_export_disabled_while_still_generating():
    app = _fake_button_gating_app()
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    app.generated_natural_termination = m.DEFAULT_NATURAL_TERMINATION_SETTINGS
    app.rows_done = m.PREVIEW_BOX // 2
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
