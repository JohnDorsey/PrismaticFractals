"""Tests: views."""
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


def test_render_full_image_with_a_single_view_matches_pre_views_behavior():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES))
    data = m.resolve_frame_settings([kf], 0)
    views = (_view_settings_from_config(m.ViewConfig(0)),)
    img = app.render_full_image(data, 40, 40, _starting_obst_arrays(), views)
    assert img.shape == (40, 40, 3)


def test_two_identical_views_render_as_two_identical_halves():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES))
    data = m.resolve_frame_settings([kf], 0)
    obst_arrays = _starting_obst_arrays()
    one_view = (_view_settings_from_config(m.ViewConfig(0)),)
    single = app.render_full_image(data, 50, 50, obst_arrays, one_view)
    doubled = app.render_full_image(data, 50, 50, obst_arrays, one_view * 2)
    assert doubled.shape == (50, 100, 3)
    assert np.array_equal(doubled[:, :50], doubled[:, 50:])
    assert np.array_equal(doubled[:, :50], single)


def test_two_different_views_render_as_two_different_halves():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES))
    data = m.resolve_frame_settings([kf], 0)
    v0 = m.ViewConfig(0)
    v1 = m.ViewConfig(1)
    v1.mode = "HSV"
    views = (_view_settings_from_config(v0), _view_settings_from_config(v1))
    combined = app.render_full_image(data, 50, 50, _starting_obst_arrays(), views)
    assert not np.array_equal(combined[:, :50], combined[:, 50:])


def test_view_config_defaults_match_the_module_defaults():
    v = m.ViewConfig(1)
    assert v.mode == m.DEFAULT_MODE
    assert list(v.chan_src) == list(m.DEFAULT_CHAN_SRCS)
    assert list(v.kinds) == list(m.DEFAULT_KINDS)


def test_view_config_per_channel_fields_are_independent_per_instance():
    """Regression coverage for the shallow-copy hazard: two ViewConfigs must
    never share the same underlying list."""
    a = m.ViewConfig(1)
    b = m.ViewConfig(2)
    a.chan_src[0] = 999
    assert b.chan_src[0] != 999


def test_resolve_views_falls_back_to_live_state():
    v = m.ViewConfig(7)
    snap = m.snapshot_view(v)
    resolved = m.resolve_views([snap], [], 0)
    assert len(resolved) == 1
    assert resolved[0].mode == v.mode
    assert list(resolved[0].chan_src) == list(v.chan_src)


def test_resolve_views_works_uniformly_for_id_zero():
    """id 0 must not be treated specially - it's an ordinary view like any
    other, resolved the same way."""
    v = m.ViewConfig(0)
    v.mode = "HSV"
    snap = m.snapshot_view(v)
    kf = m.Keyframe(1, 0, {"view:0:mode": "Grayscale"})
    resolved = m.resolve_views([snap], [kf], 0)[0]
    assert resolved.mode == "Grayscale"


def test_resolve_views_honors_keyframe_overrides_per_channel():
    """A keyframe can override one of a view's channels without disturbing
    the others."""
    v = m.ViewConfig(9)
    snap = m.snapshot_view(v)
    custom_chan = m.ChannelSettings(
        chan_src=m.CHAN_SRC_QUERY, kind=m.KIND_OTHER, stat=m.ST_BOUNCES, point1=m.POINT_START,
        point2=m.POINT_END, buddha_points=m.DEFAULT_BUDDHA_POINTS,
        buddha_compression=m.BUDDHA_COMPRESS_MODULUS, buddha_brightness=3, constant=m.NONE_ZERO)
    kf = m.Keyframe(1, 0, {"view:9:mode": "HSV", "view:9:chan0": custom_chan})
    resolved = m.resolve_views([snap], [kf], 0)[0]
    assert resolved.mode == "HSV"
    assert resolved.kinds[0] == m.KIND_OTHER and resolved.buddha_brightness[0] == 3
    assert resolved.kinds[1] == m.DEFAULT_KINDS[1]   # untouched channel keeps the live default


def test_capture_all_settings_has_no_first_view_special_case():
    app = _fake_render_app()
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    app.current_ray_settings = lambda: m.DEFAULT_RAY_SETTINGS
    app.current_generation_settings = lambda: (m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_STOP_SETTINGS)
    app.current_blur_settings = lambda: m.DEFAULT_BUDDHA_BLUR
    app.deco = {"outline": type("X", (), {"get": lambda self: True})()}
    v0 = m.ViewConfig(0)
    v1 = m.ViewConfig(5)
    app.obstacles = []
    app.views = [v0, v1]
    d = app.capture_all_settings()
    assert d is not None
    assert "mode" not in d and "chan0" not in d, \
        "the first view must not get fixed bare keys anymore"
    for v in (v0, v1):
        assert f"view:{v.id}:mode" in d and d[f"view:{v.id}:mode"] == v.mode
        assert all(f"view:{v.id}:chan{c}" in d for c in range(3))


def test_setting_fallback_handles_view_keys_without_crashing():
    """Regression test: setting_fallback used to fall through to
    DEFAULT_SETTING_VALUES[key] for anything that wasn't an obst: key,
    which raised KeyError for any view: key (view fields are dynamic, never
    in that dict) - this is exactly the path load_animated_setting takes."""
    app = _fake_render_app()
    app.views = [m.ViewConfig(0)]
    app.obstacles = []
    app.setting_fallback = m.App.setting_fallback.__get__(app)
    app.setting_is_float = m.App.setting_is_float.__get__(app)
    app.view_by_id = m.App.view_by_id.__get__(app)

    assert app.setting_fallback("view:0:mode") == m.DEFAULT_MODE
    assert app.setting_fallback("view:0:chan0") is not None
    assert app.setting_is_float("view:0:mode") is False
    assert app.setting_fallback("view:99:mode") is None   # a deleted view: no crash


def test_new_view_inserts_right_after_the_selected_one():
    app = _fake_view_list_app()
    app.new_view()
    assert app.all_view_ids() == [0, 1]
    assert app.selected_view_id == 1
    app.new_view()
    assert app.all_view_ids() == [0, 1, 2]
    assert app.selected_view_id == 2


def test_duplicate_view_copies_settings_and_inserts_after_source():
    app = _fake_view_list_app()
    app.new_view()
    app.new_view()
    app.select_view(0)
    app.duplicate_view()
    assert app.all_view_ids() == [0, 3, 1, 2]
    assert app.selected_view_id == 3
    dup = app.view_by_id(3)
    src = app.view_by_id(0)
    assert dup.mode == src.mode and list(dup.chan_src) == list(src.chan_src)


def test_duplicate_view_copy_does_not_share_list_state_with_source():
    """Regression coverage for the shallow-copy hazard in duplicate_view."""
    app = _fake_view_list_app()
    app.duplicate_view()
    dup = app.selected_view_obj()
    src = app.view_by_id(0)
    dup.chan_src[0] = 999
    assert src.chan_src[0] != 999


def test_delete_view_refuses_the_only_remaining_view():
    app = _fake_view_list_app()
    before = app.all_view_ids()
    app.delete_view()
    assert app.all_view_ids() == before, "the last remaining view must never be deletable"


def test_delete_view_allows_deleting_the_first_view_when_others_exist():
    """The core payoff of unifying the data model: deleting position 0 is
    now completely ordinary - whichever view is now first becomes the one
    live_view() (and so the top-left preview) shows, with no promotion step
    and no keyframe-key renaming needed."""
    app = _fake_view_list_app()
    app.new_view()                      # ids: [0, 1]
    app.select_view(0)
    app.delete_view()
    assert app.all_view_ids() == [1]
    assert app.selected_view_id == 1
    assert app.live_view().id == 1


def test_delete_view_removes_an_extra_view_and_keeps_the_first():
    app = _fake_view_list_app()
    app.new_view()
    app.delete_view()
    assert app.all_view_ids() == [0]
    assert app.selected_view_id == 0


# --------------------------------------------------------------------------- #
# ViewConfig channels and settings
# --------------------------------------------------------------------------- #
def test_a_view_config_takes_a_channel_record_and_makes_its_view_settings():
    v = m.ViewConfig(4)
    channel = m.ChannelSettings(chan_src=m.CHAN_SRC_BUDDHA, kind=m.KIND_OTHER, stat=m.ST_ANGLES,
                                point1=3, point2=5,
                                buddha_points=(False, True) + (False,) * (m.N_BUDDHA_POINTS - 2),
                                buddha_compression=m.BUDDHA_COMPRESS_MODULUS,
                                buddha_brightness=2, constant=m.NONE_ONE)
    v.set_channel(1, channel)
    assert m.view_channel_value(v, 1) == channel
    assert m.view_channel_value(v, 0) == m.view_channel_value(m.ViewConfig(0), 0)   # others untouched
    point = channel.replace(kind=m.KIND_POINT, stat=m.ST_EUCLIDEAN)
    v.set_channel(2, point)
    assert m.view_channel_value(v, 2) == point and v.point_metric[2] == m.ST_EUCLIDEAN
    settings = v.settings()
    assert isinstance(settings, m.ViewSettings) and settings.mode == v.mode
    assert settings.chan_src[1] == m.CHAN_SRC_BUDDHA


def test_a_view_config_made_from_a_snapshot_has_the_snapshots_contents():
    v = m.ViewConfig(7)
    v.mode = "HSV"
    v.set_channel(0, m.ChannelSettings(
        chan_src=m.CHAN_SRC_CONSTANT, kind=m.KIND_POINT, stat=m.ST_HDIST, point1=1, point2=2,
        buddha_points=(True,) + (False,) * (m.N_BUDDHA_POINTS - 1), buddha_compression=0,
        buddha_brightness=4, constant=m.NONE_ZERO))
    snap = m.snapshot_view(v)
    back = m.view_config_from_snapshot(snap)
    assert back is not v and back.id == 7 and m.snapshot_view(back) == snap
