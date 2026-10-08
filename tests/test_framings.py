"""Tests: framings."""
import math

import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# A framing's arithmetic: its world rectangle, the size of its image, its text
# --------------------------------------------------------------------------- #
def test_the_default_framing_covers_exactly_the_world_every_image_covered_before_framings_existed():
    assert m.DEFAULT_FRAMING_DIAGONAL == 2.0 * math.sqrt(2.0)
    assert (m.DEFAULT_FRAMING.x, m.DEFAULT_FRAMING.y, m.DEFAULT_FRAMING.aspect) == (0.0, 0.0, 1.0)
    assert m.framing_world_rect(m.DEFAULT_FRAMING) == m.WORLD_RECT_DEFAULT == m.WorldRect(-1.0, 1.0, -1.0, 1.0)
    assert m.framing_image_size(m.DEFAULT_FRAMING) == (m.SIZE, m.SIZE)
    assert m.framing_outline_pixel(m.DEFAULT_FRAMING) == m.PIXEL


def test_a_framings_diagonal_and_aspect_ratio_make_its_window_around_its_centre():
    f = _framing(x=1.5, y=-2.0, diagonal=5.0, aspect=2.0)
    world_rect = m.framing_world_rect(f)
    width, height = world_rect.x_stop - world_rect.x_start, world_rect.y_stop - world_rect.y_start
    assert abs(width / height - 2.0) < 1e-12, "aspect ratio is width / height"
    assert abs(math.hypot(width, height) - 5.0) < 1e-12, "the diagonal is the diagonal"
    assert abs((world_rect.x_start + world_rect.x_stop) / 2 - 1.5) < 1e-12 and abs((world_rect.y_start + world_rect.y_stop) / 2 + 2.0) < 1e-12
    portrait = m.framing_world_rect(_framing(diagonal=5.0, aspect=0.5))
    assert (portrait.x_stop - portrait.x_start) < (portrait.y_stop - portrait.y_start)
    assert all(type(v) is float for v in world_rect), "numba is given floats"


def test_the_image_of_a_framing_has_its_aspect_ratio_in_the_largest_size_that_fits_the_512_box():
    sizes = {a: m.framing_image_size(_framing(aspect=a)) for a in (1.0, 2.0, 0.5, 16 / 9, 4.0 / 3.0)}
    assert sizes[1.0] == (512, 512) and sizes[2.0] == (512, 256) and sizes[0.5] == (256, 512)
    assert sizes[16 / 9] == (512, 288) and sizes[4.0 / 3.0] == (512, 384)
    assert m.framing_image_size(_framing(aspect=1000.0)) == (512, 1) and m.framing_image_size(_framing(aspect=0.001)) == (1, 512)
    assert m.framing_image_size(_framing(aspect=2.0), box=48) == (48, 24), "a thumbnail's box"


def test_a_framing_whose_numbers_leave_no_window_to_see_through_is_not_usable():
    assert m.framing_is_usable(m.DEFAULT_FRAMING)
    assert m.framing_is_usable(_framing(x=-3.0, y=7.5, diagonal=0.001, aspect=20.0))
    assert not m.framing_is_usable(_framing(x=1e300, diagonal=1e-300)), "no width left next to such a centre"
    assert not m.framing_is_usable(_framing(diagonal=float("inf")))
    assert not m.framing_is_usable(_framing(diagonal=0.0))


def test_an_outline_is_as_thick_in_a_zoomed_framing_as_in_the_whole_world():
    """The layers of an obstacle's outline are measured in the world, so a framing that zooms in would have drawn
    them as thick as the zoom: they are measured in the pixels of the preview now."""
    half = _framing(diagonal=m.DEFAULT_FRAMING_DIAGONAL / 2.0)
    assert abs(m.framing_outline_pixel(half) - m.PIXEL / 2.0) < 1e-15
    wide = _framing(aspect=4.0)
    width, height = m.framing_size(wide)
    assert m.framing_outline_pixel(wide) == width / m.SIZE, "the longer side is the 512 pixels"


def test_framing_snapshots_are_strict_frozen_copies_of_live_framings_and_same_framing_ignores_ids():
    live = m.FramingConfig(4)
    live.x, live.diagonal = 0.5, 3
    snap = m.snapshot_framing(live)
    assert (snap.id, snap.x, snap.y, snap.diagonal, snap.aspect) == (4, 0.5, 0.0, 3.0, 1.0)
    assert all(type(getattr(snap, f)) is float for f in m.FRAMING_FIELDS), "an int diagonal is stored as a float"
    live.x = 9.0
    assert snap.x == 0.5, "a snapshot does not follow the live framing"
    try:
        snap.x = 1.0
    except Exception:
        pass
    else:
        raise AssertionError("a snapshot must be frozen")
    try:
        m.FramingSnapshot(id=0, x="0", y=0.0, diagonal=1.0, aspect=1.0)
    except Exception:
        pass
    else:
        raise AssertionError("a snapshot checks the types of its fields")
    back = m.framing_config_from_snapshot(snap)
    assert (back.id, back.x, back.diagonal) == (4, 0.5, 3.0)
    assert m.same_framing(snap, _framing(id=99, x=0.5, diagonal=3.0)) and not m.same_framing(snap, m.DEFAULT_FRAMING)


def test_a_framing_describes_itself_in_the_lines_of_a_margin_block():
    lines = m.framing_info_block(_framing(x=0.25, y=-0.5, diagonal=4.0, aspect=1.5))
    assert lines == ["Framing", "  x 0.25", "  y -0.5", "  diagonal 4", "  aspect 1.5"]
    assert len(m.framing_info_block(m.DEFAULT_FRAMING)) == len(lines), "a block's height never depends on the numbers"


# --------------------------------------------------------------------------- #
# The kernels cover the framing's world rectangle
# --------------------------------------------------------------------------- #
def test_every_ray_of_a_panned_and_zoomed_window_starts_at_the_centre_of_its_pixel():
    world_rect = m.WorldRect(0.2, 0.6, -0.3, 0.1)
    w, h = 8, 6
    arrays = m.build_obstacle_arrays([_circle_obstacle(0.0, 0.0, 5.0, R, R, fire=True)])
    stats = np.zeros((h, w, m.N_STATS), dtype=np.float32)
    characteristic_points = np.zeros((h, w, m.N_POINTS, 2), dtype=np.float32)
    pixel_state = np.zeros((h, w), dtype=np.uint8)
    m.trace_rows(stats, characteristic_points, pixel_state, 0, h, w, h, world_rect, _stop(3), _ray(arrays), 0)
    assert (pixel_state == m.PIXEL_TRACED).all()
    gx, gy = m.make_grid(w, h, world_rect)
    assert np.allclose(characteristic_points[..., m.POINT_START, 0], gx, atol=1e-6) and np.allclose(characteristic_points[..., m.POINT_START, 1], gy, atol=1e-6)
    assert abs(gx[0, 0] - (0.2 + 0.5 * 0.4 / 8)) < 1e-12 and abs(gy[0, 0] - (0.1 - 0.5 * 0.4 / 6)) < 1e-12, \
        "the top left pixel's centre, in the world rectangle's corner"


def test_make_grid_of_the_whole_world_is_what_it_always_was():
    gx, gy = m.make_grid(5, 3)
    cx = (np.arange(5) + 0.5) * (2.0 / 5) - 1.0
    cy = (np.arange(3) + 0.5) * (2.0 / 3) - 1.0
    ex, ey = np.meshgrid(cx, -cy)
    assert np.array_equal(gx, ex) and np.array_equal(gy, ey)
    assert np.array_equal(gx, m.make_grid(5, 3, m.WORLD_RECT_DEFAULT)[0])


def test_a_buddha_point_is_counted_in_the_pixel_of_the_window_it_falls_in_and_not_at_all_outside_it():
    world_rect = m.WorldRect(0.0, 1.0, 0.0, 1.0)
    counts = np.zeros((4, 4))
    for x, y in ((0.1, 0.9), (0.9, 0.1), (0.5, 0.5)):
        m.scatter_one_buddha_point(counts, x, y, world_rect)
    assert counts[0, 0] == 1 and counts[3, 3] == 1 and counts[2, 2] == 1 and counts.sum() == 3
    for x, y in ((-0.1, 0.5), (1.1, 0.5), (0.5, 1.1), (0.5, -0.1)):
        m.scatter_one_buddha_point(counts, x, y, world_rect)
    assert counts.sum() == 3, "points outside the world rectangle are not in the image"
    whole = np.zeros((4, 4))
    m.scatter_one_buddha_point(whole, 0.1, 0.9, m.WORLD_RECT_DEFAULT)
    assert whole[0, 2] == 1, "the whole world's pixel of the same point is another"


def test_the_camera_pixels_seeds_of_a_window_are_the_pixels_of_that_window():
    """scatter_multi_buddha is the seeding of the camera's pixels: the zone is the world rectangle the counts cover."""
    stop = _stop(40)
    ray = _ray(m.build_obstacle_arrays([_circle_obstacle(0.5, 0.0, 0.6, R, R, fire=True)]))
    world_rect = m.WorldRect(0.2, 0.9, -0.3, 0.3)
    n = 20
    multi = (False, True, False, False)
    mine = np.zeros((n, n))
    m.scatter_multi_buddha(0, n, world_rect, stop, ray, 0, *multi, mine)
    general = np.zeros((n, n))
    none = np.empty(0, dtype=np.int64)
    m.scatter_from_seeds(0, n, world_rect, world_rect, stop, ray, 0, none, none, *multi, general)
    assert mine.sum() > 0 and np.array_equal(mine, general)
    whole = np.zeros((n, n))
    m.scatter_multi_buddha(0, n, m.WORLD_RECT_DEFAULT, stop, ray, 0, *multi, whole)
    assert not np.array_equal(mine, whole), "another world rectangle, other seeds and another image"
    elsewhere = np.zeros((n, n))
    m.scatter_from_seeds(0, n, m.WORLD_RECT_DEFAULT, world_rect, stop, ray, 0, none, none, *multi, elsewhere)
    assert not np.array_equal(elsewhere, general), "a zone of its own is not the world rectangle"


# --------------------------------------------------------------------------- #
# The image on screen: geometry, pixels and the world
# --------------------------------------------------------------------------- #
def _geometry_app(framing=None):
    class FakeApp:
        pass
    app = FakeApp()
    return _give_image_geometry(app, framing)


def test_set_image_framing_reshapes_every_buffer_and_forgets_what_was_computed():
    app = _geometry_app()
    assert app.stats.shape == (512, 512, m.N_STATS) and app.gx.shape == (512, 512)
    app.pixel_state[:] = m.PIXEL_TRACED
    app.rows_done = 300
    app.set_image_framing(_framing(aspect=2.0))
    assert (app.img_w, app.img_h) == (512, 256) and app.rows_done == 0
    assert app.stats.shape == (256, 512, m.N_STATS) and app.characteristic_points.shape == (256, 512, m.N_POINTS, 2)
    assert app.pixel_state.shape == (256, 512) and (app.pixel_state == m.PIXEL_PENDING).all()
    assert app.img.shape == app.frame.shape == app.display_frame.shape == (256, 512, 3)
    assert (app.img == m.BACKGROUND).all() and app.gx.shape == (256, 512)
    assert app.image_world_rect == m.framing_world_rect(_framing(aspect=2.0))
    assert app.outline_pixel == m.framing_outline_pixel(_framing(aspect=2.0)) and app.image_framing == _framing(aspect=2.0)


def test_set_image_framing_of_the_same_size_clears_the_buffers_it_has_instead_of_making_new_ones():
    app = _geometry_app()
    stats = app.stats
    app.stats[:] = 5
    app.set_image_framing(_framing(x=0.5))
    assert app.stats is stats and not stats.any(), "80 MB are not allocated again for a pan"
    assert app.image_world_rect.x_start == m.framing_world_rect(_framing(x=0.5)).x_start and app.gx[0, 0] > -1.0 + 0.4


def test_a_pixels_centre_is_its_grid_point_and_the_canvas_position_of_the_centre_is_the_pixel():
    for framing in (m.DEFAULT_FRAMING, _framing(x=0.3, y=-0.2, diagonal=1.7, aspect=2.5), _framing(diagonal=3.1, aspect=0.4)):
        app = _geometry_app(framing)
        for i, j in ((0, 0), (17, 3), (app.img_w - 1, app.img_h - 1), (app.img_w // 2, app.img_h // 3)):
            x, y = app.pixel_center(i, j)
            assert (x, y) == (float(app.gx[j, i]), float(app.gy[j, i])), "a hover ray starts where the image's did"
            cx, cy = app.world_to_canvas(x, y)
            assert abs(cx - (i + 0.5)) < 1e-9 and abs(cy - (j + 0.5)) < 1e-9
    xs, ys = app.world_to_canvas(np.array([app.image_world_rect.x_start, app.image_world_rect.x_stop]),
                                 np.array([app.image_world_rect.y_stop, app.image_world_rect.y_start]))
    assert np.allclose(xs, [0, app.img_w]) and np.allclose(ys, [0, app.img_h]), "the world rectangle's corners are the canvas'"


def test_an_obstacle_outline_is_placed_on_the_canvas_by_the_window_of_the_image():
    lune = m.LuneShape(0)
    default = np.array(lune.outline_coords())
    assert np.array_equal(default, np.array(lune.outline_coords(m.WORLD_RECT_DEFAULT, m.SIZE, m.SIZE)))
    world_rect = m.WorldRect(-2.0, 2.0, -1.0, 1.0)
    wide = np.array(lune.outline_coords(world_rect, 512, 256)).reshape(-1, 2)
    base = default.reshape(-1, 2)
    assert np.allclose(wide[:, 0], (base[:, 0] - 256) / 2.0 + 256, atol=1e-9), "x: half as far from the middle"
    assert np.allclose(wide[:, 1], base[:, 1] / 2.0, atol=1e-9), "y: the 2:1 canvas halves 512"


# --------------------------------------------------------------------------- #
# The list of framings and what a change of framing does to the image
# --------------------------------------------------------------------------- #
def _fake_framing_app(generated=False):
    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    _give_framings(app)
    for name in ("new_framing", "duplicate_framing", "delete_framing", "select_framing", "on_framing_changed",
                 "apply_selected_framing", "refresh_framing_controls_for_selected_framing", "on_framing_commit"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    app.canvas = app.photo = _StubWidget()
    app.generated_settings = m.DEFAULT_RAY_SETTINGS if generated else None
    app.pinned_px = (3, 4) if generated else None
    app.calls = []
    app.purged = []
    app.purge_keyframe_settings = app.purged.append
    def begin_generation():
        app.calls.append(("begin_generation", app.selected_framing_snapshot()))
        app.set_image_framing(app.selected_framing_snapshot())       # (as the real one starts by doing)
    app.begin_generation = begin_generation
    app.clear_overlay = lambda: app.calls.append("clear_overlay")
    app.refresh_live_preview_state = lambda: app.calls.append("refresh_live_preview_state")
    app.update_buttons = lambda: app.calls.append("update_buttons")
    app.rebuild_coloration_list = lambda: app.calls.append("rebuild_coloration_list")
    app.last_framing = {f: getattr(app.framings[0], f) for f in m.FRAMING_FIELDS}
    for field in m.FRAMING_FIELDS:
        setattr(app, f"framing_{field}_var", _StubVar(str(app.last_framing[field])))
    for name in ("validate_signed_float", "validate_positive_float"):
        setattr(app, name, getattr(m.App, name))
    return app


def test_new_and_duplicate_put_the_framing_after_the_selected_one_and_select_it_and_delete_goes_to_framing_0():
    app = _fake_framing_app()
    app.new_framing()
    assert app.all_framing_ids() == [0, 1] and app.selected_framing_id == 1 and app._framing_counter == 2
    assert m.same_framing(app.framings[1], m.DEFAULT_FRAMING), "a new framing sees the whole world"
    app.framings[1].x, app.framings[1].aspect = 0.5, 2.0
    app.select_framing(0)
    app.duplicate_framing()
    assert app.all_framing_ids() == [0, 2, 1] and app.selected_framing_id == 2
    app.select_framing(1)
    app.duplicate_framing()
    assert app.all_framing_ids() == [0, 2, 1, 3] and app.selected_framing_id == 3
    assert (app.framings[3].x, app.framings[3].aspect) == (0.5, 2.0), "a copy has the numbers of what it copies"
    app.delete_framing()
    assert app.all_framing_ids() == [0, 2, 1] and app.selected_framing_id == 0
    app.select_framing(2)
    app.delete_framing()
    app.delete_framing()                                             # framing 0 is selected and not alone
    assert app.all_framing_ids() == [1] and app.selected_framing_id == 1
    app.delete_framing()
    assert app.all_framing_ids() == [1], "the last framing cannot be deleted"
    assert app.selected_framing_obj() is app.framings[0] and app.selected_framing_snapshot().id == 1


def test_selecting_a_framing_before_anything_is_generated_only_gives_the_blank_image_its_shape():
    app = _fake_framing_app(generated=False)
    app.new_framing()
    assert "begin_generation" not in [c[0] if isinstance(c, tuple) else c for c in app.calls], "the same camera"
    app.calls.clear()
    app.framing_aspect_var.set("2")
    app.on_framing_commit()
    assert (app.img_w, app.img_h) == (512, 256) and app.calls.count("refresh_live_preview_state") == 1
    assert not any(isinstance(c, tuple) for c in app.calls), "nothing is generated: there is no figure yet"
    assert "rebuild_coloration_list" in app.calls


def test_selecting_or_editing_a_framing_after_a_generation_generates_the_same_figure_again_and_lets_go_of_the_pin():
    app = _fake_framing_app(generated=True)
    app.new_framing()
    assert not any(isinstance(c, tuple) for c in app.calls), "a copy of the same camera needs no new image"
    assert app.pinned_px == (3, 4), "and the pin is still the same place"
    app.framing_x_var.set("0.5")
    app.on_framing_commit()
    begins = [c for c in app.calls if isinstance(c, tuple)]
    assert len(begins) == 1 and begins[0][1].x == 0.5 and begins[0][1].id == 1
    assert app.pinned_px is None, "pixel (3, 4) is somewhere else in the new image"
    app.calls.clear()
    app.select_framing(0)
    assert [c[1].x for c in app.calls if isinstance(c, tuple)] == [0.0], "selecting another framing, another image"
    app.calls.clear()
    app.select_framing(0)
    assert app.calls == [], "selecting the selected one changes nothing"


def test_a_framing_entry_commits_only_valid_numbers_and_otherwise_goes_back_to_the_last_valid_ones():
    app = _fake_framing_app(generated=True)
    app.framing_x_var.set("abc")
    app.framing_diagonal_var.set("0")
    app.framing_aspect_var.set("-2")
    app.on_framing_commit()
    assert [app.framing_x_var.get(), app.framing_diagonal_var.get(), app.framing_aspect_var.get()] == \
        ["0.0", str(m.DEFAULT_FRAMING_DIAGONAL), "1.0"]
    assert not any(isinstance(c, tuple) for c in app.calls), "nothing changed, so nothing is generated"
    app.framing_x_var.set("1e300")
    app.framing_diagonal_var.set("1e-300")
    app.on_framing_commit()
    assert app.framings[0].x == 0.0 and app.framing_x_var.get() == "0.0", "a world rectangle of no width is refused, all of it"
    app.framing_x_var.set("-1.5")
    app.framing_y_var.set("0.25")
    app.framing_diagonal_var.set("4")
    app.framing_aspect_var.set("1.5")
    app.on_framing_commit()
    f = app.framings[0]
    assert (f.x, f.y, f.diagonal, f.aspect) == (-1.5, 0.25, 4.0, 1.5)
    assert sum(isinstance(c, tuple) for c in app.calls) == 1, "one generation for the four entries"
    app.on_framing_commit()
    assert sum(isinstance(c, tuple) for c in app.calls) == 1, "committing the same numbers again is nothing"


def test_begin_generation_makes_an_image_of_the_selected_framing_of_the_figure_already_held():
    """A new generation of the same figure: the settings it was started with stay (the controls may hold others
    meanwhile), the caches of the old image are dropped, the image takes the shape of the framing."""
    app = _fake_framing_app(generated=True)
    app.selected_framing_id = 0
    app.framings[0].aspect = 2.0
    app.calls.clear()
    app.tick_id = None
    app.root = _StubWidget()
    app.thumb_sims = {"old": 1}
    app.coloration_thumb_cache = {1: "stale"}
    app.rebuild_buddha_counts = lambda: app.calls.append("rebuild_buddha_counts")
    app.draw = lambda: None
    app.tick = lambda: app.calls.append("tick")
    app.pinned_px = None
    app.done = True
    app.running = False
    app.start_time = app.elapsed = app.next_tick = 0.0
    settings = m.DEFAULT_RAY_SETTINGS.replace(seg_mult=0.5)
    app.generated_settings = settings
    m.App.begin_generation(app)
    assert (app.img_w, app.img_h) == (512, 256) and app.rows_done == 0
    assert app.generated_settings is settings, "a new image, not a new figure"
    assert app.thumb_sims == {} and app.coloration_thumb_cache == {}
    assert app.running and not app.done and app.calls[-1] == "tick"


# --------------------------------------------------------------------------- #
# Thumbnails: framings are seen through coloration 0, colorations through framing 0
# --------------------------------------------------------------------------- #
def _thumbnail_app(generated=True):
    app = _fake_current_image_app()
    _give_framings(app)
    app.thumb_sims = {}
    app.simulated_thumbnail_data = m.App.simulated_thumbnail_data.__get__(app)
    app.render_thumbnail = m.App.render_thumbnail.__get__(app)
    app.render_framing_thumbnail = m.App.render_framing_thumbnail.__get__(app)
    app.gen_ray = _ray(m.build_obstacle_arrays(_start_scene()))
    app.gen_stop = _stop(20)
    app.generated_settings = m.DEFAULT_RAY_SETTINGS if generated else None
    app.generated_natural_stop = m.DEFAULT_NATURAL_STOP_SETTINGS if generated else None
    return app


def test_a_framings_thumbnail_before_any_generation_is_the_blank_image_in_its_shape_centred_on_grey():
    app = _thumbnail_app(generated=False)
    pic = app.render_framing_thumbnail(_framing(aspect=2.0), 48)
    assert pic.shape == (48, 48, 3)
    assert tuple(pic[0, 0]) == m.THUMB_BACKGROUND and tuple(pic[47, 47]) == m.THUMB_BACKGROUND
    assert tuple(pic[24, 24]) == tuple(m.BACKGROUND) and tuple(pic[12, 0]) == tuple(m.BACKGROUND)
    assert (pic[:12] == m.THUMB_BACKGROUND).all() and (pic[12:36] != m.THUMB_BACKGROUND).any()
    square = app.render_framing_thumbnail(m.DEFAULT_FRAMING, 48)
    assert (square == m.BACKGROUND).all(), "the whole world's is a square: no grey"


def test_a_framings_thumbnail_is_what_the_framing_sees_coloured_by_coloration_0_and_never_by_another():
    app = _thumbnail_app()
    app.colorations = [_white_coloration_config(), m.ColorationConfig(1)]
    background = np.array(m.BACKGROUND)
    whole = app.render_framing_thumbnail(m.DEFAULT_FRAMING, 48)
    zoomed_framing = _framing(diagonal=m.DEFAULT_FRAMING_DIAGONAL / 4.0)       # the middle quarter of the circle
    zoomed = app.render_framing_thumbnail(zoomed_framing, 48)
    assert not np.array_equal(whole, zoomed)
    assert (whole == background).all(axis=-1)[0, 0], "the whole world's corners are outside the circle"
    assert (whole == 255).all(axis=-1)[24, 24], "and coloration 0 (white) colours the rest"
    assert (zoomed == 255).all(), "the middle of the circle fills the zoomed framing's picture"
    app.colorations[1].mode = "HSV"
    assert np.array_equal(app.render_framing_thumbnail(m.DEFAULT_FRAMING, 48), whole), "coloration 1 plays no part"
    app.colorations[0] = m.ColorationConfig(0)
    assert not np.array_equal(app.render_framing_thumbnail(m.DEFAULT_FRAMING, 48), whole), "coloration 0 does"


def test_the_simulation_behind_thumbnails_is_made_once_per_framing_and_size():
    app = _thumbnail_app()
    first = app.simulated_thumbnail_data(m.DEFAULT_FRAMING, 48)
    assert app.simulated_thumbnail_data(_framing(id=7), 48) is first, "the same camera, whatever its id"
    assert app.simulated_thumbnail_data(m.DEFAULT_FRAMING, 24) is not first
    assert app.simulated_thumbnail_data(_framing(x=0.1), 48) is not first
    stats, characteristic_points, pixel_state, world_rect, w, h = first
    assert (w, h) == (48, 48) and world_rect == m.WORLD_RECT_DEFAULT and (pixel_state == m.PIXEL_TRACED).any()


def test_a_colorations_thumbnail_is_of_framing_0_even_when_the_image_on_screen_is_of_another_framing():
    app = _thumbnail_app()
    app.colorations = [m.ColorationConfig(0), m.ColorationConfig(1)]
    app.image_framing = _framing(x=0.4, id=5)
    app.img = None                                   # the image on screen is of no use to it
    want = app.render_thumbnail(app.framings[0], app.colorations[1].settings(), 48)
    assert np.array_equal(app.render_coloration_thumbnail(app.colorations[1], 48), want)
    live = app.render_coloration_thumbnail(app.colorations[0], 48)
    assert live.shape == (48, 48, 3), "even the live coloration's is a small simulation of framing 0"


# --------------------------------------------------------------------------- #
# Images of a framing
# --------------------------------------------------------------------------- #
def _white_coloration_config():
    """A coloration that paints every pixel whose ray fires white (all three channels constant 1)."""
    v = m.ColorationConfig(0)
    for c in range(3):
        v.chan_src[c] = m.CHAN_SRC_CONSTANT
        v.constant_choice[c] = m.NONE_ONE
    return v


def _render(framing, w, h, outline=False):
    app = _fake_render_app()
    kf = m.Keyframe(0, 0, {**m.DEFAULT_SETTING_VALUES, "outline": outline})
    data = m.resolve_frame_settings([kf], 0)
    colorations = (_coloration_settings_from_config(_white_coloration_config()),)
    return app.render_full_image(data, w, h, _starting_obst_arrays(), colorations, framing)


def test_an_image_of_a_framing_shows_only_the_part_of_the_world_it_covers():
    background = np.array(m.BACKGROUND)
    whole = _render(m.DEFAULT_FRAMING, 40, 40)
    assert (whole == background).all(axis=-1)[0, 0] and (whole == background).all(axis=-1)[-1, -1], "outside the circle"
    assert (whole == 255).all(axis=-1)[20, 20]
    zoomed = _render(_framing(diagonal=m.DEFAULT_FRAMING_DIAGONAL / 2.0), 40, 40)
    assert (zoomed == 255).all(), "the middle of the circle fills the picture"
    panned = _render(_framing(x=1.0), 40, 40)
    assert (panned[:, -10:] == background).all(), "to the right of the circle there is nothing"
    assert (panned[16:24, :4] == 255).all(), "its right edge is at the left of the picture"
    wide = _render(_framing(aspect=2.0), 80, 40)
    assert wide.shape == (40, 80, 3)


def test_the_two_layers_of_an_outline_are_as_thick_in_pixels_in_a_zoomed_image_as_in_the_whole_world_one():
    small = _circle_obstacle(0.0, 0.0, 0.4, R, R, fire=True)

    def thickness(framing):
        app = _fake_render_app()
        kf = m.Keyframe(0, 0, {**m.DEFAULT_SETTING_VALUES, "outline": True})
        data = m.resolve_frame_settings([kf], 0)
        colorations = (_coloration_settings_from_config(_white_coloration_config()),)
        img = app.render_full_image(data, 256, 256, m.build_obstacle_arrays([small]), colorations, framing)
        row = img[128]
        ring = m.OBST_EFFECT_RGB[R]
        return int((row == np.array(ring)).all(axis=-1).sum())
    whole = thickness(m.DEFAULT_FRAMING)
    zoomed = thickness(_framing(diagonal=m.DEFAULT_FRAMING_DIAGONAL / 2.0))
    assert 2 <= whole and abs(whole - zoomed) <= 2, (whole, zoomed)


# --------------------------------------------------------------------------- #
# Which coloration and which framing are used
# --------------------------------------------------------------------------- #
def test_the_selected_coloration_is_the_one_exported_unless_include_all_colorations_is_on():
    app = _fake_export_app()
    app.colorations = [m.ColorationConfig(0), m.ColorationConfig(1), m.ColorationConfig(2)]
    app.selected_coloration_id = 2
    app.include_all_colorations_var.set(False)
    assert [v.id for v in app.colorations_to_export()] == [2], "the selected one, not the first"
    app.include_all_colorations_var.set(True)
    assert [v.id for v in app.colorations_to_export()] == [0, 1, 2], "all of them, in list order"
    assert app.live_coloration().id == 2, "and the selected one is the preview's"


def test_exports_are_of_the_selected_framing_alone_there_being_no_include_all_framings_choice():
    app = _fake_export_app()
    app.framings.append(m.FramingConfig(1))
    app.framings[1].x = 0.5
    app.selected_framing_id = 1
    snap = app.selected_framing_snapshot()
    assert (snap.id, snap.x) == (1, 0.5)
    import inspect
    assert "include_all_framings" not in inspect.getsource(m.App.build_status_quadrant)


# --------------------------------------------------------------------------- #
# The left margin: framing settings below the figure's, a line of dashes between
# --------------------------------------------------------------------------- #
def test_the_left_margin_holds_the_figure_and_the_framing_settings_as_their_two_locations_say():
    args = (*_generation(), _start_scene())
    f = _framing(x=0.5, diagonal=3.0)
    assert m.left_margin_blocks(*args, f, "None", "None") == []
    figure = m.figure_info_blocks(*_generation(), _start_scene())
    assert m.left_margin_blocks(*args, f, "Left margin", "None") == figure
    only_framing = m.left_margin_blocks(*args, f, "None", "Left margin")
    assert only_framing == [m.framing_info_block(f)], "with no figure settings to be apart from, no line"
    both = m.left_margin_blocks(*args, f, "Left margin", "Left margin")
    assert both[:-1] == figure, "the figure's settings first ..."
    assert both[-1] == [m.FIGURE_INFO_SEPARATOR] + m.framing_info_block(f), "... the framing's below, after a line"
    assert set(m.FIGURE_INFO_SEPARATOR) == {"-"} and len(m.FIGURE_INFO_SEPARATOR) == m.FIGURE_INFO_COLUMN_CHARS
    assert "-" in m.FONT_5X7, "the pixel font can draw the line"


def test_the_framing_settings_block_comes_below_the_figures_in_the_margins_column_with_the_line_between():
    img = np.full((600, 600, 3), 90, dtype=np.uint8)
    blocks = m.left_margin_blocks(*_generation(), _start_scene(), _framing(x=0.5), "Left margin", "Left margin")
    columns = m.layout_info_columns(blocks, 60)
    flat = [line for col in columns for line in col]
    i = flat.index(m.FIGURE_INFO_SEPARATOR)
    assert flat[i - 1] == " " and flat[i + 1] == "Framing" and flat[i + 2] == "  x 0.5"
    assert flat[0].startswith("Iter limit"), "the figure's settings are above it"
    out = m.add_left_info_margin(img, blocks)
    assert out.shape[0] == 600 and out.shape[1] > 600 and np.array_equal(out[:, -600:], img)


def test_the_size_of_a_margin_with_both_blocks_never_depends_on_the_numbers_so_a_videos_frames_match():
    img = np.full((120, 160, 3), 90, dtype=np.uint8)
    shapes = set()
    for f in (m.DEFAULT_FRAMING, _framing(x=123456.789, y=-0.000123, diagonal=77.7, aspect=3.5)):
        blocks = m.left_margin_blocks(*_generation(), _start_scene(), f, "Left margin", "Left margin")
        shapes.add((m.add_left_info_margin(img, blocks).shape, tuple(len(b) for b in blocks)))
    assert len(shapes) == 1


def test_apply_info_locations_adds_the_framing_margin_when_its_location_says_so():
    img = np.full((120, 160, 3), 90, dtype=np.uint8)
    args = (img, [m.ColorationConfig(0)], 160, *_generation(), _start_scene(), "None", "None")
    assert m.apply_info_locations(*args, m.DEFAULT_FRAMING, "None", mutate_ok=True).shape == img.shape
    with_framing = m.apply_info_locations(*args, m.DEFAULT_FRAMING, "Left margin", mutate_ok=True)
    assert with_framing.shape[0] == 120 and with_framing.shape[1] > 160
    assert (with_framing[:, :-160] == m.COLORATION_INFO_FG).all(axis=-1).any(), "there is text in it"
    both = m.apply_info_locations(args[0], args[1], 160, *_generation(), _start_scene(), "None", "Left margin",
                                  m.DEFAULT_FRAMING, "Left margin", mutate_ok=True)
    assert both.shape[1] >= with_framing.shape[1]


def test_the_framing_settings_location_is_an_export_choice_of_its_own_that_a_project_keeps():
    assert m.FRAMING_INFO_LOCATIONS == ["None", "Left margin"]
    rows = {row[0]: row for row in m.EXPORT_OPTIONS}
    assert rows["framing_info_location"][1:3] == ("framing_info_location_var", "choice")
    assert rows["framing_info_location"][3].key("Left margin") == "left_margin"
    assert rows["framing_info_location"][3].key("None") == "none"
    import inspect
    source = inspect.getsource(m.App.build_status_quadrant)
    assert "Framing settings location:" in source and source.index("Figure settings location:") < source.index("Framing settings location:")


# --------------------------------------------------------------------------- #
# The Framing tab
# --------------------------------------------------------------------------- #
def test_the_two_lists_are_one_widget_so_they_look_and_behave_the_same():
    import inspect
    assert "ThumbnailStrip(" in inspect.getsource(m.App.build_colorations_list)
    assert "ThumbnailStrip(" in inspect.getsource(m.App.build_framing_controls)
    strip = inspect.getsource(m.ThumbnailStrip.__init__) + inspect.getsource(m.ThumbnailStrip.rebuild)
    for text in ('text="New"', 'text="Dupe"', 'text="Del"', "highlightthickness=self.outline_px"):
        assert text in strip


def test_the_framing_tabs_entries_are_x_y_diagonal_and_aspect_ratio_read_by_their_own_parsers():
    assert [row[0] for row in m.FRAMING_ENTRIES] == ["x", "y", "diagonal", "aspect"]
    assert [row[0] for row in m.FRAMING_ENTRIES] == list(m.FRAMING_FIELDS)
    parse = {row[0]: row[2] for row in m.FRAMING_ENTRIES}
    assert parse["x"]("-0.5") == -0.5 and parse["x"]("nan") is None and parse["x"]("") is None
    assert parse["diagonal"]("2.5") == 2.5 and parse["diagonal"]("0") is None and parse["diagonal"]("-1") is None
    assert parse["aspect"]("1e3") == 1000.0 and parse["aspect"]("inf") is None


# --------------------------------------------------------------------------- #
# Framings in keyframes: "framing:<id>:<field>", all four floats, so the camera glides
# --------------------------------------------------------------------------- #
def test_a_framings_four_numbers_are_keyframe_fields_that_glide_between_keyframes():
    assert [row[0] for row in m.FRAMING_FIELD_INFO] == list(m.FRAMING_FIELDS)
    assert all(row[2] for row in m.FRAMING_FIELD_INFO), "every one is a float: interpolated, never held"
    base = _framing(id=1, x=0.0, y=0.0, diagonal=2.0, aspect=1.0)
    keyframes = [m.Keyframe(0, 0, {"framing:1:x": 0.0, "framing:1:diagonal": 2.0, "framing:1:aspect": 1.0}),
                 m.Keyframe(1, 10, {"framing:1:x": 4.0, "framing:1:diagonal": 6.0, "framing:1:aspect": 3.0})]
    mid = m.resolve_framing(base, keyframes, 5)
    assert (mid.x, mid.diagonal, mid.aspect) == (2.0, 4.0, 2.0) and mid.id == 1
    assert mid.y == 0.0, "no keyframe defines y: the framing's own value"
    assert m.resolve_framing(base, keyframes, -3).x == 0.0 and m.resolve_framing(base, keyframes, 99).x == 4.0, "held outside"
    other = m.resolve_framing(_framing(id=2, x=7.0), keyframes, 5)
    assert other.x == 7.0, "another framing's keys are not this one's"
    assert all(type(getattr(mid, f)) is float for f in m.FRAMING_FIELDS)


def test_capture_all_settings_lists_every_framings_numbers_under_keys_of_its_own():
    app = _fake_render_app()
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    app.current_generation_settings = lambda: (m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_STOP_SETTINGS)
    app.current_blur_settings = lambda: m.DEFAULT_BUDDHA_BLUR
    app.current_buddha_seed_settings = lambda: m.DEFAULT_BUDDHA_SEED
    app.deco = {"outline": _StubVar(True)}
    app.obstacles, app.colorations = [], []
    second = m.FramingConfig(3)
    second.x, second.aspect = 0.5, 2.0
    app.framings.append(second)
    d = app.capture_all_settings()
    assert {k for k in d if k.startswith("framing:")} == {f"framing:{i}:{f}" for i in (0, 3) for f in m.FRAMING_FIELDS}
    assert d["framing:3:x"] == 0.5 and d["framing:3:aspect"] == 2.0 and d["framing:0:diagonal"] == m.DEFAULT_FRAMING_DIAGONAL


def _keyframe_framing_app(generated=True):
    app = _fake_framing_app(generated)
    for name in ("load_value", "on_framings_loaded", "setting_fallback", "setting_is_float", "all_setting_entries",
                 "all_setting_tabs", "widgets_for_setting"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    app.obstacles, app.colorations = [], []
    app.framing_x_entry = object()
    app.framing_aspect_entry = object()
    return app


def test_loading_framing_keys_only_sets_the_numbers_and_one_refresh_afterwards_generates_the_image_once():
    app = _keyframe_framing_app()
    app.new_framing()
    app.calls.clear()
    app.selected_framing_id = 1
    for field, value in (("x", 0.5), ("y", -0.25), ("diagonal", 2.0), ("aspect", 2.0)):
        app.load_value(f"framing:1:{field}", value)
    f = app.framings[1]
    assert (f.x, f.y, f.diagonal, f.aspect) == (0.5, -0.25, 2.0, 2.0)
    assert not any(isinstance(c, tuple) for c in app.calls), "four numbers are no reason for four generations"
    app.on_framings_loaded()
    assert sum(isinstance(c, tuple) for c in app.calls) == 1 and app.framing_x_var.get() == "0.5"
    assert "rebuild_coloration_list" in app.calls
    app.calls.clear()
    app.on_framings_loaded()
    assert app.calls == [], "nothing was loaded since"
    app.load_value("framing:77:x", 1.0)                              # a framing that is gone
    app.load_value("framing:1:colour", 1.0)                          # a field it does not have
    app.on_framings_loaded()
    assert app.calls == []


def test_a_mix_of_loaded_numbers_that_leaves_the_framing_no_window_does_not_generate_an_image():
    app = _keyframe_framing_app()
    app.load_value("framing:0:x", 1e300)
    app.load_value("framing:0:diagonal", 1e-300)
    app.on_framings_loaded()
    assert not any(isinstance(c, tuple) for c in app.calls), "the image stays as it was"
    assert "rebuild_coloration_list" in app.calls


def test_framing_keys_have_a_fallback_a_float_flag_a_section_of_the_settings_list_and_an_entry_widget():
    app = _keyframe_framing_app()
    app.new_framing()
    app.framings[1].x = 0.75
    assert app.setting_fallback("framing:1:x") == 0.75 and app.setting_fallback("framing:42:x") is None
    assert all(app.setting_is_float(f"framing:1:{f}") for f in m.FRAMING_FIELDS)
    entries = [e for e in app.all_setting_entries() if e.key.startswith("framing:")]
    assert [e.tab for e in entries] == ["Framing 0"] * 4 + ["Framing 1"] * 4
    assert [e.label for e in entries[:4]] == ["X", "Y", "Diagonal", "Aspect ratio"] and all(e.is_float for e in entries)
    assert entries[4].fallback == 0.75
    assert app.all_setting_tabs()[-2:] == ["Framing 0", "Framing 1"]
    app.selected_framing_id = 1
    assert app.widgets_for_setting("framing:1:x") == [app.framing_x_entry]
    assert app.widgets_for_setting("framing:0:x") == [], "only the selected framing's controls are on show"
    assert m.short_value_text("framing:1:x", 0.75) == "0.75"
    assert m.short_value_text("framing:1:diagonal", m.DEFAULT_FRAMING_DIAGONAL) == "2.82843"


def test_deleting_a_framing_removes_its_keyframe_settings_at_once():
    app = _keyframe_framing_app()
    app.new_framing()
    app.delete_framing()
    assert app.purged == ["framing:1:"], "the colon keeps framing 1 from taking framing 10's with it"


def test_a_video_frame_and_a_new_image_use_the_framing_resolved_for_that_frame():
    import inspect
    assert "resolve_framing(framing_snapshot, assigned_sorted, fn)" in inspect.getsource(m.App._render_worker)
    assert "resolve_framing(self.selected_framing_snapshot(), [fake_kf], 0)" in inspect.getsource(m.App.save_new_image)
