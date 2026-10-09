"""Tests: colorations."""
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


def test_render_full_image_with_a_single_coloration_matches_pre_colorations_behavior():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES))
    data = m.resolve_frame_settings([kf], 0, m.DEFAULT_SETTING_VALUES)
    colorations = (_coloration_settings_from_config(m.ColorationConfig(0)),)
    img = app.render_full_image(data, 40, 40, _starting_obst_arrays(), colorations, m.DEFAULT_FRAMING)
    assert img.shape == (40, 40, 3)


def test_two_identical_colorations_render_as_two_identical_halves():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES))
    data = m.resolve_frame_settings([kf], 0, m.DEFAULT_SETTING_VALUES)
    obst_arrays = _starting_obst_arrays()
    one_coloration = (_coloration_settings_from_config(m.ColorationConfig(0)),)
    single = app.render_full_image(data, 50, 50, obst_arrays, one_coloration, m.DEFAULT_FRAMING)
    doubled = app.render_full_image(data, 50, 50, obst_arrays, one_coloration * 2, m.DEFAULT_FRAMING)
    assert doubled.shape == (50, 100, 3)
    assert np.array_equal(doubled[:, :50], doubled[:, 50:])
    assert np.array_equal(doubled[:, :50], single)


def test_two_different_colorations_render_as_two_different_halves():
    app = _fake_render_app()
    kf = m.Keyframe(1, 0, dict(m.DEFAULT_SETTING_VALUES))
    data = m.resolve_frame_settings([kf], 0, m.DEFAULT_SETTING_VALUES)
    v0 = m.ColorationConfig(0)
    v1 = m.ColorationConfig(1)
    v1.color_model = "HSV"
    colorations = (_coloration_settings_from_config(v0), _coloration_settings_from_config(v1))
    combined = app.render_full_image(data, 50, 50, _starting_obst_arrays(), colorations, m.DEFAULT_FRAMING)
    assert not np.array_equal(combined[:, :50], combined[:, 50:])


def test_coloration_config_defaults_match_the_module_defaults():
    v = m.ColorationConfig(1)
    assert v.color_model == m.DEFAULT_COLOR_MODEL
    assert list(v.chan_src) == list(m.DEFAULT_CHAN_SRCS)
    assert list(v.metric_types) == list(m.DEFAULT_METRIC_TYPES)


def test_coloration_config_per_channel_fields_are_independent_per_instance():
    """Regression coverage for the shallow-copy hazard: two ColorationConfigs must
    never share the same underlying list."""
    a = m.ColorationConfig(1)
    b = m.ColorationConfig(2)
    a.chan_src[0] = 999
    assert b.chan_src[0] != 999


def test_resolve_colorations_falls_back_to_live_state():
    v = m.ColorationConfig(7)
    snap = m.snapshot_coloration(v)
    resolved = m.resolve_colorations([snap], [], 0)
    assert len(resolved) == 1
    assert resolved[0].color_model == v.color_model
    assert list(resolved[0].chan_src) == list(v.chan_src)


def test_resolve_colorations_works_uniformly_for_id_zero():
    """id 0 must not be treated specially - it's an ordinary coloration like any
    other, resolved the same way."""
    v = m.ColorationConfig(0)
    v.color_model = "HSV"
    snap = m.snapshot_coloration(v)
    kf = m.Keyframe(1, 0, {"coloration:0:color_model": "Grayscale"})
    resolved = m.resolve_colorations([snap], [kf], 0)[0]
    assert resolved.color_model == "Grayscale"


def test_resolve_colorations_honors_keyframe_overrides_per_channel():
    """A keyframe can override one of a coloration's channels without disturbing
    the others."""
    v = m.ColorationConfig(9)
    snap = m.snapshot_coloration(v)
    custom_chan = m.ChannelSettings(
        chan_src=m.CHAN_SRC_QUERY, metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_BOUNCE_COUNT, point1=m.POINT_START,
        point2=m.POINT_END, buddha_points=m.DEFAULT_BUDDHA_POINTS,
        buddha_compression=m.BUDDHA_COMPRESS_MODULUS, buddha_brightness=3, constant=m.NONE_ZERO)
    kf = m.Keyframe(1, 0, {"coloration:9:color_model": "HSV", "coloration:9:chan0": custom_chan})
    resolved = m.resolve_colorations([snap], [kf], 0)[0]
    assert resolved.color_model == "HSV"
    assert resolved.metric_types[0] == m.METRIC_TYPE_OTHER and resolved.buddha_brightness[0] == 3
    assert resolved.metric_types[1] == m.DEFAULT_METRIC_TYPES[1]   # untouched channel keeps the live default


def test_capture_all_settings_has_no_first_coloration_special_case():
    app = _fake_render_app()
    app.capture_all_settings = m.App.capture_all_settings.__get__(app)
    app.current_ray_settings = lambda: m.DEFAULT_RAY_SETTINGS
    app.current_generation_settings = lambda: (m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_TERMINATION_SETTINGS)
    app.current_blur_settings = lambda: m.DEFAULT_BUDDHA_BLUR
    app.current_buddha_seed_settings = lambda: m.DEFAULT_BUDDHA_SEED
    app.deco = {"outline": type("X", (), {"get": lambda self: True})()}
    v0 = m.ColorationConfig(0)
    v1 = m.ColorationConfig(5)
    app.obstacles = []
    app.colorations = [v0, v1]
    d = app.capture_all_settings()
    assert d is not None
    assert "color_model" not in d and "chan0" not in d, \
        "the first coloration must not get fixed bare keys anymore"
    for v in (v0, v1):
        assert f"coloration:{v.id}:color_model" in d and d[f"coloration:{v.id}:color_model"] == v.color_model
        assert all(f"coloration:{v.id}:chan{c}" in d for c in range(3))


def test_setting_fallback_handles_coloration_keys_without_crashing():
    """Regression test: setting_fallback used to fall through to
    DEFAULT_SETTING_VALUES[key] for anything that wasn't an obst: key,
    which raised KeyError for any coloration: key (coloration fields are dynamic, never
    in that dict) - this is exactly the path load_animated_setting takes."""
    app = _fake_render_app()
    app.colorations = [m.ColorationConfig(0)]
    app.obstacles = []
    app.setting_fallback = m.App.setting_fallback.__get__(app)
    app.setting_is_float = m.App.setting_is_float.__get__(app)
    app.coloration_by_id = m.App.coloration_by_id.__get__(app)

    assert app.setting_fallback("coloration:0:color_model") == m.DEFAULT_COLOR_MODEL
    assert app.setting_fallback("coloration:0:chan0") is not None
    assert app.setting_is_float("coloration:0:color_model") is False
    assert app.setting_fallback("coloration:99:color_model") is None   # a deleted coloration: no crash


def test_new_coloration_inserts_right_after_the_selected_one():
    app = _fake_coloration_list_app()
    app.new_coloration()
    assert app.all_coloration_ids() == [0, 1]
    assert app.selected_coloration_id == 1
    app.new_coloration()
    assert app.all_coloration_ids() == [0, 1, 2]
    assert app.selected_coloration_id == 2


def test_duplicate_coloration_copies_settings_and_inserts_after_source():
    app = _fake_coloration_list_app()
    app.new_coloration()
    app.new_coloration()
    app.select_coloration(0)
    app.duplicate_coloration()
    assert app.all_coloration_ids() == [0, 3, 1, 2]
    assert app.selected_coloration_id == 3
    dup = app.coloration_by_id(3)
    src = app.coloration_by_id(0)
    assert dup.color_model == src.color_model and list(dup.chan_src) == list(src.chan_src)


def test_duplicate_coloration_copy_does_not_share_list_state_with_source():
    """Regression coverage for the shallow-copy hazard in duplicate_coloration."""
    app = _fake_coloration_list_app()
    app.duplicate_coloration()
    dup = app.selected_coloration_obj()
    src = app.coloration_by_id(0)
    dup.chan_src[0] = 999
    assert src.chan_src[0] != 999


def test_delete_coloration_refuses_the_only_remaining_coloration():
    app = _fake_coloration_list_app()
    before = app.all_coloration_ids()
    app.delete_coloration()
    assert app.all_coloration_ids() == before, "the last remaining coloration must never be deletable"


def test_delete_coloration_allows_deleting_the_first_coloration_when_others_exist():
    """The core payoff of unifying the data model: deleting position 0 is
    now completely ordinary - whichever coloration is now first becomes the one
    live_coloration() (and so the top-left preview) shows, with no promotion step
    and no keyframe-key renaming needed."""
    app = _fake_coloration_list_app()
    app.new_coloration()                      # ids: [0, 1]
    app.select_coloration(0)
    app.delete_coloration()
    assert app.all_coloration_ids() == [1]
    assert app.selected_coloration_id == 1
    assert app.live_coloration().id == 1


def test_delete_coloration_removes_an_extra_coloration_and_keeps_the_first():
    app = _fake_coloration_list_app()
    app.new_coloration()
    app.delete_coloration()
    assert app.all_coloration_ids() == [0]
    assert app.selected_coloration_id == 0


# --------------------------------------------------------------------------- #
# ColorationConfig channels and settings
# --------------------------------------------------------------------------- #
def test_a_coloration_config_takes_a_channel_record_and_makes_its_coloration_settings():
    v = m.ColorationConfig(4)
    channel = m.ChannelSettings(chan_src=m.CHAN_SRC_BUDDHA, metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_ANGLES,
                                point1=3, point2=5,
                                buddha_points=(False, True) + (False,) * (m.N_BUDDHA_POINTS - 2),
                                buddha_compression=m.BUDDHA_COMPRESS_MODULUS,
                                buddha_brightness=2, constant=m.NONE_ONE)
    v.set_channel(1, channel)
    assert m.coloration_channel_value(v, 1) == channel
    assert m.coloration_channel_value(v, 0) == m.coloration_channel_value(m.ColorationConfig(0), 0)   # others untouched
    point = channel.replace(metric_type=m.METRIC_TYPE_POINT_PAIR, metric_slot=m.ST_EUCLIDEAN)
    v.set_channel(2, point)
    assert m.coloration_channel_value(v, 2) == point and v.point_metric[2] == m.ST_EUCLIDEAN
    settings = v.settings()
    assert isinstance(settings, m.ColorationSettings) and settings.color_model == v.color_model
    assert settings.chan_src[1] == m.CHAN_SRC_BUDDHA


def test_a_coloration_config_made_from_a_snapshot_has_the_snapshots_contents():
    v = m.ColorationConfig(7)
    v.color_model = "HSV"
    v.set_channel(0, m.ChannelSettings(
        chan_src=m.CHAN_SRC_CONSTANT, metric_type=m.METRIC_TYPE_POINT_PAIR, metric_slot=m.ST_HDIST, point1=1, point2=2,
        buddha_points=(True,) + (False,) * (m.N_BUDDHA_POINTS - 1), buddha_compression=0,
        buddha_brightness=4, constant=m.NONE_ZERO))
    snap = m.snapshot_coloration(v)
    back = m.coloration_config_from_snapshot(snap)
    assert back is not v and back.id == 7 and m.snapshot_coloration(back) == snap
