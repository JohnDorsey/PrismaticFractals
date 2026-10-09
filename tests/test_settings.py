"""Tests: settings."""
import dataclasses
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# The default iteration limit
# --------------------------------------------------------------------------- #
def test_the_default_iteration_limit_is_256():
    assert m.DEFAULT_ITER == 256
    assert m.DEFAULT_NATURAL_TERMINATION_SETTINGS.max_iter == 256
    assert m.DEFAULT_SETTING_VALUES["iter"] == 256
    assert m.ITER_LIMIT_MIN <= m.DEFAULT_ITER <= m.ITER_LIMIT_MAX


def test_the_iteration_limit_entry_starts_at_the_default():
    import inspect
    assert "self.iter_var = tk.StringVar(value=str(DEFAULT_ITER))" in \
        inspect.getsource(m.App.build_limits_controls)


def test_the_export_image_choices_are_called_current_image_and_new_image():
    assert m.SAVE_SOURCE_OPTIONS == ["Current image", "New image"]


def test_strict_records_take_keyword_arguments_only_and_all_of_them():
    f = m.DEFAULT_FUZZ
    assert _raises(TypeError, lambda: m.FuzzParams(False, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0))
    assert _raises(TypeError, lambda: m.FuzzParams(enabled=False))          # the rest is missing
    assert _raises(TypeError, lambda: m.BuddhaBlurSettings(radius=0.1))
    assert _raises(TypeError, lambda: m.RaySettings(max_iter=1))
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f.as_dict(), "emit_jiter": 0.0}))
    assert _raises(TypeError, lambda: m.RaySettings(*dataclasses.astuple(m.DEFAULT_RAY_SETTINGS)))


def test_strict_records_cannot_be_indexed_iterated_unpacked_or_changed():
    s = m.DEFAULT_RAY_SETTINGS
    assert _raises(TypeError, lambda: s[0])
    assert _raises(TypeError, lambda: list(s))
    assert _raises(TypeError, lambda: tuple(m.DEFAULT_FUZZ))
    assert _raises(TypeError, lambda: len(s))
    assert _raises(dataclasses.FrozenInstanceError, lambda: setattr(s, "seg_mult", 5))
    assert _raises(dataclasses.FrozenInstanceError,
                   lambda: setattr(m.DEFAULT_NATURAL_TERMINATION_SETTINGS, "max_iter", 5))
    assert _raises(AttributeError, lambda: s.seg_mlt)
    # (Python 3.11 raises TypeError here, later versions AttributeError; both forbid it)
    assert _raises((AttributeError, TypeError, dataclasses.FrozenInstanceError),
                   lambda: setattr(s, "not_a_field", 1))


def test_strict_records_check_every_value_against_its_annotation():
    f = m.DEFAULT_FUZZ.as_dict()
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f, "rng_seed": True}))        # a bool is not an int
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f, "rng_seed": 1.0}))         # nor is a float
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f, "enabled": 1}))        # an int is not a bool
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f, "bnd_mult": "1.0"}))
    assert _raises(TypeError, lambda: m.FuzzParams(**{**f, "bnd_mult": True}))    # a bool is not a float
    m.FuzzParams(**{**f, "bnd_mult": 2})                                          # an int is a fine float
    s = m.DEFAULT_RAY_SETTINGS.as_dict()
    assert _raises(TypeError, lambda: m.RaySettings(**{**s, "fuzz": (False, 0.0)}))
    assert _raises(TypeError, lambda: m.KeyframeData(ray=m.DEFAULT_RAY_SETTINGS,
                                                     natural_termination=m.DEFAULT_NATURAL_TERMINATION_SETTINGS,
                                                     buddha_blur=0.0, buddha_seed=m.DEFAULT_BUDDHA_SEED,
                                                     outline=True))
    n = m.DEFAULT_NATURAL_TERMINATION_SETTINGS.as_dict()
    try:
        m.NaturalTerminationSettings(**{**n, "max_iter": "9"})
        raise AssertionError("expected TypeError")
    except TypeError as e:
        assert "NaturalTerminationSettings.max_iter must be int, got str" in str(e)
    assert _raises(TypeError, lambda: m.RaySettings(**{**s, "max_iter": 9}))      # (not a ray setting any more)


def test_a_strict_record_stores_a_list_given_for_a_tuple_field_as_a_tuple():
    vs = m.ColorationSettings(color_model="RGB", metric_types=[0, 1, 2], point_metric=(0, 0, 0), other_metric=(0, 0, 0),
                        point1_choice=(0, 0, 0), point2_choice=(0, 0, 0), chan_src=[0, 0, 0],
                        buddha_points=[(True,)] * 3, buddha_compression=(0, 0, 0),
                        buddha_brightness=(0, 0, 0), constant_choice=(0, 0, 0))
    assert vs.metric_types == (0, 1, 2) and isinstance(vs.chan_src, tuple)
    assert _raises(TypeError, lambda: vs.replace(color_model=5))


def test_replace_makes_a_modified_copy_and_equality_hash_and_pickle_work():
    import pickle
    a = m.DEFAULT_RAY_SETTINGS
    b = a.replace(seg_mult=0.5, fuzz=a.fuzz.replace(rng_seed=3))
    assert (b.seg_mult, b.fuzz.rng_seed, a.seg_mult, a.fuzz.rng_seed) == (0.5, 3, 1.0, 0)
    assert b != a and a == a.replace() and hash(a) == hash(a.replace())
    assert _raises(TypeError, lambda: a.replace(seg_mult="x"))
    with _ModuleRegistered():
        again = pickle.loads(pickle.dumps(b))
    assert again == b and isinstance(again.fuzz, m.FuzzParams)
    n = m.DEFAULT_NATURAL_TERMINATION_SETTINGS
    termination = n.replace(max_iter=7, escape_radius=1.5)
    assert (termination.max_iter, termination.escape_radius, n.max_iter, n.escape_radius) == (7, 1.5, 256, 2.0)
    assert termination != n and n == n.replace() and hash(n) == hash(n.replace())
    assert _raises(TypeError, lambda: n.replace(max_iter="x"))
    with _ModuleRegistered():
        assert pickle.loads(pickle.dumps(termination)) == termination


def test_records_pickle_by_field_name_not_position():
    """A dump or saved file must not depend on the order the fields are declared in
    (dataclass's own pickle state for a frozen, slotted class is a bare list)."""
    import pickle
    with _ModuleRegistered():
        state = m.DEFAULT_FUZZ.__reduce_ex__(2)[2]
        assert isinstance(state, dict) and state["rng_seed"] == 0 and state["bnd_mult"] == 1.0
        scrambled = dict(reversed(list(state.items())))
        clone = m.FuzzParams.__new__(m.FuzzParams)
        clone.__setstate__(scrambled)
        assert clone == m.DEFAULT_FUZZ
        assert pickle.loads(pickle.dumps(m.DEFAULT_FUZZ)) == m.DEFAULT_FUZZ


def test_a_keyframes_data_holds_the_ray_settings_the_natural_termination_and_the_buddha_blur_side_by_side():
    data = m.resolve_frame_settings([], 0, m.DEFAULT_SETTING_VALUES)
    assert [f.name for f in dataclasses.fields(m.KeyframeData)] == [
        "ray", "natural_termination", "buddha_blur", "buddha_seed", "outline"]
    assert isinstance(data.ray, m.RaySettings) and isinstance(data.buddha_blur, m.BuddhaBlurSettings)
    assert isinstance(data.natural_termination, m.NaturalTerminationSettings)
    assert data.natural_termination == m.DEFAULT_NATURAL_TERMINATION_SETTINGS and data.ray == m.DEFAULT_RAY_SETTINGS
    assert (data.buddha_blur.radius, data.buddha_blur.type) == (m.DEFAULT_BUDDHA_BLUR_RADIUS,
                                                                m.DEFAULT_BUDDHA_BLUR_TYPE)
    louder = data.replace(buddha_blur=data.buddha_blur.replace(radius=0.2))
    assert louder.buddha_blur.radius == 0.2 and louder.ray is data.ray


def test_the_coloration_snapshot_record_mirrors_the_slots_of_the_live_coloration():
    assert [f.name for f in dataclasses.fields(m.ColorationSnapshot)] == list(m.ColorationConfig.__slots__)


def test_snapshot_coloration_stores_the_channel_lists_as_tuples_unshared_with_the_live_coloration():
    v = m.ColorationConfig(2)
    snap = m.snapshot_coloration(v)
    assert isinstance(snap, m.ColorationSnapshot) and snap.id == 2 and snap.color_model == v.color_model
    assert all(isinstance(getattr(snap, f), tuple) for f in m.ColorationConfig.__slots__
               if f not in ("id", "color_model"))
    v.chan_src[0] = m.CHAN_SRC_BUDDHA
    assert snap.chan_src[0] != m.CHAN_SRC_BUDDHA                # a later live edit never reaches it
    assert m.coloration_channel_value(snap, 0) == m.coloration_channel_value(m.ColorationConfig(2), 0)


def test_resolved_obstacles_are_frozen_shapes_and_colorations_and_settings_are_strict_records():
    kfs = [m.Keyframe(0, 0, {"obst:1:r": 0.25})]
    o = m.EllipseShape(1)
    resolved = m.resolve_obstacles([o.snapshot()], kfs, 0)
    assert type(resolved[0]) is m.EllipseShape and resolved[0].r == 0.25
    assert _raises(AttributeError, lambda: setattr(resolved[0], "r", 1.0))     # frozen
    colorations = m.resolve_colorations([m.snapshot_coloration(m.ColorationConfig(0))], [], 0)
    assert isinstance(colorations[0], m.ColorationSettings)
    assert isinstance(m.resolve_frame_settings([], 0, m.DEFAULT_SETTING_VALUES), m.KeyframeData)
    assert isinstance(m.resolve_frame_settings([], 0, m.DEFAULT_SETTING_VALUES).ray, m.RaySettings)


def test_colorations_colored_from_the_stored_data_use_the_settings_the_image_was_generated_with():
    """Regression test: the two places that color a coloration from the stored data (the
    composed 'Current image' export and the non-live coloration thumbnails) built their
    settings from the limits alone, so the confinement rule and emission direction
    silently fell back to their defaults - and Buddha coloration re-traces rays
    with both. Every RaySettings argument is required now, and they pass the
    settings of the generated image."""
    generated, generated_termination = _generation(
        confine_lookback=9, confine_method=m.CONFINE_PATH, confine_threshold=0.123,
        emit_mode=m.EMIT_TOWARDS, emit_target_r=0.5, emit_target_theta=30.0, emit_offset=12.0)
    blur = m.BuddhaBlurSettings(radius=0.2, type=m.BUDDHA_BLUR_TRIANGULAR)
    app = _fake_current_image_app()
    _give_generated(app, ray=generated, natural_termination=generated_termination,
                    engine_args=m.settings_ray_args(generated, app.generated.engine_args.obstacles))
    app.current_blur_settings = lambda: blur
    app.current_buddha_seed_settings = lambda: m.DEFAULT_BUDDHA_SEED
    seen = []

    def fake_colorize(coloration, stats, characteristic_points, pixel_state, w, h, blur, termination, engine_args, rng_seed_base, seeds=None, world_rect=None):
        seen.append((blur, termination, engine_args, rng_seed_base))
        return np.zeros((h, w, 3), dtype=np.uint8)
    app.apply_coloration = fake_colorize
    app.render_current_image([m.ColorationConfig(0), m.ColorationConfig(1)])
    expected = (blur, app.generated.termination, app.generated.engine_args, generated.fuzz.rng_seed)
    assert seen == [expected] * 2
    assert seen[0][1].confine == m.ConfineRule(lookback=9, method=m.CONFINE_PATH, threshold=0.123)
    assert seen[0][2].emission.offset == 12.0
    seen.clear()
    app.coloration_thumb_cache = {}
    app.rows_done = 0
    app.live_coloration = lambda: app.colorations[0]
    app.render_coloration_thumbnail(m.ColorationConfig(1), m.THUMB_SIZE)
    assert seen == [expected]
    blur = m.BuddhaBlurSettings(radius=0.0, type=m.BUDDHA_BLUR_GAUSSIAN)   # the blur controls move on:
    expected = (blur,) + expected[1:]
    seen.clear()
    app.coloration_thumb_cache = {}
    app.render_current_image([m.ColorationConfig(0), m.ColorationConfig(1)])
    assert seen == [expected] * 2                                # the same generated image, the new blur
    _give_generated(app, generated=False)                        # before the first Start
    assert app.settings_of_current_image() is m.DEFAULT_RAY_SETTINGS
    assert app.natural_termination_of_current_image() is m.DEFAULT_NATURAL_TERMINATION_SETTINGS


# --------------------------------------------------------------------------- #
# The settings tables (RAY_SETTING_SPECS, BLUR_SETTING_SPECS, joined by
# SETTING_TABLES): one row per setting (path into its record, its control, how
# the control is written)
# --------------------------------------------------------------------------- #
def _varied_blur_settings():
    return m.BuddhaBlurSettings(radius=0.07, type=m.BUDDHA_BLUR_BOKEH)


def _varied_ray_settings():
    return m.RaySettings(
        seg_mult=0.8,
        fuzz=m.FuzzParams(enabled=True, emit_jitter=4.0, bnd_offset=1.5, bnd_jitter=2.5,
                          bnd_mult=1.25, self_offset=-3.0, self_jitter=6.0, self_mult=0.5,
                          rng_seed=424242),
        emit_mode=m.EMIT_TOWARDS, emit_target_r=0.4, emit_target_theta=33.0,
        emit_target_cx=-0.2, emit_target_cy=0.6, emit_offset=12.5)


def _varied_natural_termination_settings():
    return m.NaturalTerminationSettings(
        max_iter=77, max_dist=3.5, escape_radius=3.25, confine_lookback=9,
        confine_method=m.CONFINE_PATH, confine_threshold=0.03)


def _varied_seed_settings():
    return m.BuddhaSeedSettings(camera_pixels=False, x_start=-0.5, x_stop=0.25, y_start=-3.0, y_stop=2.0)


def test_each_settings_table_and_its_record_fields_are_the_same_set():
    assert set(m.SETTING_TABLES) == {m.NaturalTerminationSettings, m.RaySettings, m.BuddhaBlurSettings,
                                     m.BuddhaSeedSettings}
    assert m.SETTING_TABLES[m.BuddhaSeedSettings] is m.BUDDHA_SEED_SETTING_SPECS
    assert m.SETTING_TABLES[m.NaturalTerminationSettings] is m.NATURAL_TERMINATION_SETTING_SPECS
    assert m.SETTING_TABLES[m.RaySettings] is m.RAY_SETTING_SPECS
    assert m.SETTING_TABLES[m.BuddhaBlurSettings] is m.BLUR_SETTING_SPECS
    seen = set()
    for record_class, table in m.SETTING_TABLES.items():
        paths = sorted(spec.path for spec in table.values())
        assert paths == sorted(m.record_leaf_paths(record_class)), record_class
        assert len(set(paths)) == len(paths)
        assert not (seen & set(table)), "a setting key is in two tables"
        seen |= set(table)
    assert seen == {k for k, _t, _l in m.SETTING_INFO} - {"outline"}
    assert list(m.SETTING_SPECS) == (list(m.NATURAL_TERMINATION_SETTING_SPECS) + list(m.RAY_SETTING_SPECS)
                                     + list(m.BLUR_SETTING_SPECS) + list(m.BUDDHA_SEED_SETTING_SPECS))
    assert set(m.NATURAL_TERMINATION_SETTING_SPECS) == {"iter", "dist", "escape_radius", "confine_lookback",
                                                "confine_method", "confine_threshold"}
    assert set(m.BLUR_SETTING_SPECS) == {"buddha_blur_radius", "buddha_blur_type"}
    assert set(m.BUDDHA_SEED_SETTING_SPECS) == {"buddha_seed_camera", "buddha_seed_x_start", "buddha_seed_x_stop",
                                               "buddha_seed_y_start", "buddha_seed_y_stop"}
    assert all(spec.tab == "Buddha seeding" for spec in m.BUDDHA_SEED_SETTING_SPECS.values())
    assert m.DEFAULT_BUDDHA_SEED == m.BuddhaSeedSettings(camera_pixels=True, x_start=-1.0, x_stop=1.0,
                                                         y_start=-1.0, y_stop=1.0)


def test_self_checks_catch_a_settings_field_without_a_spec_row():
    fresh = load_module()
    del fresh.RAY_SETTING_SPECS["emit_offset"]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "SETTING_TABLES" in str(e) and "emit_offset" in str(e)
    fresh = load_module()
    del fresh.BLUR_SETTING_SPECS["buddha_blur_radius"]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "SETTING_TABLES" in str(e) and "buddha_blur_radius" in str(e)


def test_self_checks_catch_a_setting_key_that_two_tables_both_have():
    fresh = load_module()
    fresh.BLUR_SETTING_SPECS["seed"] = fresh.RAY_SETTING_SPECS["seed"]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "more than one table" in str(e)


def test_self_checks_catch_a_spec_row_with_a_bad_kind_or_path():
    fresh = load_module()
    fresh.RAY_SETTING_SPECS["seed"] = fresh.RAY_SETTING_SPECS["seed"].replace(kind="dial")
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "bad kind" in str(e)
    fresh = load_module()
    fresh.RAY_SETTING_SPECS["seed"] = fresh.RAY_SETTING_SPECS["seed"].replace(path=("fuzz", "sead"))
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "the RaySettings fields and the rows of its settings table disagree" in str(e)
    fresh = load_module()
    fresh.BLUR_SETTING_SPECS["buddha_blur_type"] = \
        fresh.BLUR_SETTING_SPECS["buddha_blur_type"].replace(path=("typ",))
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "the BuddhaBlurSettings fields and the rows of its settings table disagree" in str(e)


def test_settings_records_round_trip_through_the_key_dictionary():
    for settings in (m.DEFAULT_RAY_SETTINGS, _varied_ray_settings()):
        values = m.settings_to_dict(settings)
        assert set(values) == set(m.RAY_SETTING_SPECS)
        assert m.settings_from_dict(m.RaySettings, values) == settings
    for termination in (m.DEFAULT_NATURAL_TERMINATION_SETTINGS, _varied_natural_termination_settings()):
        values = m.settings_to_dict(termination)
        assert set(values) == set(m.NATURAL_TERMINATION_SETTING_SPECS)
        assert m.settings_from_dict(m.NaturalTerminationSettings, values) == termination
    for blur in (m.DEFAULT_BUDDHA_BLUR, _varied_blur_settings()):
        values = m.settings_to_dict(blur)
        assert set(values) == set(m.BLUR_SETTING_SPECS)
        assert m.settings_from_dict(m.BuddhaBlurSettings, values) == blur
    for seeds in (m.DEFAULT_BUDDHA_SEED, _varied_seed_settings()):
        values = m.settings_to_dict(seeds)
        assert set(values) == set(m.BUDDHA_SEED_SETTING_SPECS)
        assert m.settings_from_dict(m.BuddhaSeedSettings, values) == seeds
    ray, termination, blur = _varied_ray_settings(), _varied_natural_termination_settings(), _varied_blur_settings()
    seeds = _varied_seed_settings()
    values = m.settings_to_dict(ray, termination, blur, seeds)       # side by side in one dictionary...
    assert set(values) == set(m.SETTING_SPECS)
    assert m.settings_from_dict(m.RaySettings, values) == ray          # ...each record picks out its own
    assert m.settings_from_dict(m.NaturalTerminationSettings, values) == termination
    assert m.settings_from_dict(m.BuddhaBlurSettings, values) == blur
    assert m.settings_from_dict(m.BuddhaSeedSettings, values) == seeds
    assert values["seed"] == 424242 and values["emit_jitter"] == 4.0 and values["iter"] == 77
    assert values["buddha_blur_type"] == m.BUDDHA_BLUR_BOKEH and values["fuzz_on"] is True


def test_values_resolved_from_keyframes_are_converted_to_their_fields_types():
    """A keyframe can hand an int setting over as a float (or a bool as 1): int fields
    are rounded and bool fields made bool; float fields are kept as given."""
    values = dict(m.settings_to_dict(_varied_ray_settings(), _varied_natural_termination_settings(),
                                     _varied_blur_settings()))
    values.update({"iter": 77.4, "seed": 9.0, "confine_lookback": 8.6, "emit_mode": 1.0,
                   "fuzz_on": 1, "buddha_blur_type": 2.0, "dist": 2})
    got = m.settings_from_dict(m.RaySettings, values)
    termination = m.settings_from_dict(m.NaturalTerminationSettings, values)
    assert (termination.max_iter, got.fuzz.rng_seed, termination.confine_lookback, got.emit_mode) == (77, 9, 9, 1)
    assert got.fuzz.enabled is True and termination.max_dist == 2
    assert all(type(v) is int for v in (termination.max_iter, got.fuzz.rng_seed, termination.confine_lookback))
    blur = m.settings_from_dict(m.BuddhaBlurSettings, values)
    assert blur.type == 2 and type(blur.type) is int


def test_the_derived_setting_tables_follow_the_settings_field_types():
    floats = m.SETTING_FLOAT_KEYS
    assert {"dist", "escape_radius", "seg_mult", "confine_threshold", "emit_offset", "emit_target_r",
            "emit_jitter", "bnd_mult", "self_offset", "buddha_blur_radius"} <= floats
    assert not ({"iter", "seed", "confine_lookback", "confine_method", "emit_mode", "fuzz_on",
                 "buddha_blur_type", "outline"} & floats)
    assert m.DEFAULT_SETTING_VALUES == {
        **m.settings_to_dict(m.DEFAULT_NATURAL_TERMINATION_SETTINGS, m.DEFAULT_RAY_SETTINGS,
                             m.DEFAULT_BUDDHA_BLUR, m.DEFAULT_BUDDHA_SEED), "outline": True}
    assert m.SETTING_WIDGET_ATTRS["seed"] == "rng_seed_entry"
    assert m.SETTING_WIDGET_ATTRS["outline"] == "deco_outline_check"


def _fake_controls_app():
    """An App stand-in holding a control of the right kind for every spec row."""
    class FakeApp:
        pass
    app = FakeApp()
    for spec in m.SETTING_SPECS.values():
        setattr(app, spec.control, _StubVar(None))
    app.on_limit_var_changed = app.on_shape_param_changed = app.on_fuzz_changed = lambda: None
    app.set_setting_control = m.App.set_setting_control.__get__(app)
    app.revert_settings = m.App.revert_settings.__get__(app)
    return app


def _expected_control_value(spec, value):
    if spec.kind == "text":
        return str(value)
    if spec.kind == "int_text":
        return str(int(round(value)))
    if spec.kind == "float_text":
        return str(float(value))
    if spec.kind == "choice":
        return spec.names[int(round(value))]
    return value


def test_each_kind_of_control_is_written_the_way_its_kind_needs():
    app = _fake_controls_app()
    values = m.settings_to_dict(_varied_ray_settings(), _varied_natural_termination_settings(),
                                _varied_blur_settings())
    for key, value in values.items():
        app.set_setting_control(key, value)
        spec = m.SETTING_SPECS[key]
        assert getattr(app, spec.control).get() == _expected_control_value(spec, value), key
    assert app.iter_var.get() == "77" and app.confine_lookback_var.get() == "9"
    assert app.emit_target_r_var.get() == "0.4" and app.rng_seed_var.get() == "424242"
    assert app.emit_mode_var.get() == m.EMIT_MODE_NAMES[m.EMIT_TOWARDS]
    assert app.buddha_blur_type_var.get() == m.BUDDHA_BLUR_LABELS[m.BUDDHA_BLUR_BOKEH]
    assert app.fuzz_enabled_var.get() is True and app.bnd_mult_scale.get() == 1.25


def test_a_value_an_interpolating_keyframe_made_fractional_still_reaches_an_int_control():
    app = _fake_controls_app()
    app.set_setting_control("confine_lookback", 8.6)
    app.set_setting_control("emit_mode", 0.9)
    assert app.confine_lookback_var.get() == "9"
    assert app.emit_mode_var.get() == m.EMIT_MODE_NAMES[1]


def test_revert_writes_every_control_from_the_baseline_through_the_same_table():
    app = _fake_controls_app()
    app.baseline = _varied_ray_settings()
    app.baseline_natural_termination = _varied_natural_termination_settings()
    app.revert_settings()
    for key, value in m.settings_to_dict(app.baseline, app.baseline_natural_termination).items():
        spec = m.SETTING_SPECS[key]
        assert getattr(app, spec.control).get() == _expected_control_value(spec, value), key


# --------------------------------------------------------------------------- #
# ChannelSettings, SettingEntry, ObstacleRing; label and tab live in the spec rows
# --------------------------------------------------------------------------- #
def test_a_coloration_channel_is_stored_as_a_strict_record_and_decomposes_back():
    v = m.ColorationConfig(0)
    v.chan_src[2] = m.CHAN_SRC_BUDDHA
    v.buddha_points[2] = tuple(i == 1 for i in range(m.N_BUDDHA_POINTS))
    v.metric_types[1] = m.METRIC_TYPE_OTHER
    v.other_metric[1] = m.ST_BOUNCE_COUNT
    chans = [m.coloration_channel_value(v, c) for c in range(3)]
    assert all(isinstance(ch, m.ChannelSettings) for ch in chans)
    assert chans[1].metric_type == m.METRIC_TYPE_OTHER and chans[1].metric_slot == m.ST_BOUNCE_COUNT
    assert chans[2].chan_src == m.CHAN_SRC_BUDDHA and chans[2].buddha_points[1] is True
    vs = m.decompose_chans(v.color_model, chans)
    assert vs.other_metric[1] == m.ST_BOUNCE_COUNT and vs.point_metric[1] == m.DEFAULT_POINT_METRICS[1]
    assert vs.chan_src[2] == m.CHAN_SRC_BUDDHA
    assert vs.buddha_points[2] == v.buddha_points[2]
    assert chans[0] == m.coloration_channel_value(m.snapshot_coloration(v), 0)      # live coloration and snapshot agree
    assert _raises(TypeError, lambda: chans[0][0])                      # no positional access
    assert _raises(TypeError, lambda: m.ChannelSettings(*dataclasses.astuple(chans[0])))
    assert _raises(TypeError, lambda: chans[0].replace(metric_type="point"))


def test_all_setting_entries_are_strict_records_in_display_order():
    class FakeApp:
        pass
    app = FakeApp()
    _give_framings(app)
    o1, o2 = m.EllipseShape(4), m.EllipseShape(9)
    o2.width = 0.7
    app.obstacles = [o1, o2]
    app.colorations = [m.ColorationConfig(3)]
    app.live_setting_values = lambda: {**m.DEFAULT_SETTING_VALUES, "iter": 77}
    entries = m.App.all_setting_entries(app)
    assert all(isinstance(e, m.SettingEntry) for e in entries)
    static = entries[:len(m.SETTING_INFO)]
    assert [(e.key, e.tab, e.label) for e in static] == m.SETTING_INFO
    assert {e.key: e.fallback for e in static} == {**m.DEFAULT_SETTING_VALUES, "iter": 77}, \
        "a setting no keyframe defines is shown with the value the controls hold, not the default"
    assert [e.is_float for e in static] == [e.key in m.SETTING_FLOAT_KEYS for e in static]
    width = next(e for e in entries if e.key == "obst:9:width")
    assert (width.tab, width.fallback, width.is_float) == (m.obstacle_tab_title(1, o2), 0.7, True)
    chan = next(e for e in entries if e.key == "coloration:3:chan1")
    assert chan.tab == "Coloration 0" and chan.fallback == m.coloration_channel_value(app.colorations[0], 1)
    assert m.obstacle_tab_title(0, o1) == "Obstacle #1 (Ellipse)"
    assert [e.tab for e in entries].count(m.obstacle_tab_title(0, o1)) == len(o1.KEYFRAME_FIELDS)
    assert "obst:4:shape" not in [e.key for e in entries]       # a shape's kind is not a setting


def test_setting_info_is_derived_from_the_spec_rows_so_every_setting_has_a_label_and_tab():
    assert [row for row in m.SETTING_INFO if row[0] != "outline"] == [
        (key, spec.tab, spec.label) for key, spec in m.SETTING_SPECS.items()]
    assert m.SETTING_INFO[-1] == ("outline", "Decorations", "Outline")
    assert all(spec.label and spec.tab in m.SETTING_TABS_ORDER
               for spec in m.SETTING_SPECS.values())
    assert m.SETTING_LABEL["emit_offset"] == "Emission angle offset"
    assert m.SETTING_TAB["seed"] == "Mutations" and m.SETTING_TAB["seg_mult"] == "Mutations"
    assert m.SETTING_TAB["emit_mode"] == "Emissions" and m.SETTING_TAB["emit_jitter"] == "Mutations"
    assert m.SETTING_TAB["iter"] == "Termination"
    assert m.SETTING_TABS_ORDER[:3] == ["Emissions", "Termination", "Mutations"]


def _ring_app(generated):
    class FakeApp:
        pass
    app = FakeApp()
    o = m.default_starting_obstacle(0)
    o2 = m.EllipseShape(1)
    o2.r, o2.theta, o2.cx, o2.in_effect, o2.out_effect = 0.5, 90.0, 0.25, m.OBSTACLE_EFFECT_TERMINATE, m.OBSTACLE_EFFECT_PASS
    app.obstacles = [o, o2]
    arrays = m.build_obstacle_arrays(app.obstacles)
    _give_generated(app, generated=generated, engine_args=_engine_args(arrays))
    return app


def test_obstacle_rings_are_strict_records_for_both_the_live_and_the_generated_set():
    for generated in (False, True):
        rings = m.App.obstacle_ring_specs(_ring_app(generated))
        assert len(rings) == 2 and all(isinstance(r, m.ObstacleRing) for r in rings)
        first, second = rings
        assert type(first.shape) is int and type(first.in_effect) is int      # numpy scalars converted
        assert type(second.cx) is float and isinstance(second.bound, np.ndarray)
        assert (second.in_effect, second.out_effect) == (m.OBSTACLE_EFFECT_TERMINATE,
                                                         m.OBSTACLE_EFFECT_PASS)
        assert abs(second.cx - 0.25) < 1e-9 and abs(second.cy - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# NaturalTerminationSettings: the limits that end any sequence of points, a record beside RaySettings
# --------------------------------------------------------------------------- #
def test_self_checks_catch_a_natural_termination_field_without_a_row_and_a_row_with_a_bad_path():
    fresh = load_module()
    del fresh.NATURAL_TERMINATION_SETTING_SPECS["escape_radius"]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "SETTING_TABLES" in str(e) and "escape_radius" in str(e)
    fresh = load_module()
    fresh.NATURAL_TERMINATION_SETTING_SPECS["iter"] = fresh.NATURAL_TERMINATION_SETTING_SPECS["iter"].replace(path=("max_itr",))
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "the NaturalTerminationSettings fields and the rows of its settings table disagree" in str(e)


def test_the_natural_termination_and_ray_settings_hold_disjoint_settings_that_together_were_the_old_simulation_set():
    ray = {f.name for f in dataclasses.fields(m.RaySettings)}
    termination = {f.name for f in dataclasses.fields(m.NaturalTerminationSettings)}
    assert not (ray & termination)
    assert termination == {"max_iter", "max_dist", "escape_radius", "confine_lookback", "confine_method",
                    "confine_threshold"}
    assert set(m.NATURAL_TERMINATION_SETTING_SPECS) == {"iter", "dist", "escape_radius", "confine_lookback",
                                                "confine_method", "confine_threshold"}
    assert not (set(m.NATURAL_TERMINATION_SETTING_SPECS) & set(m.RAY_SETTING_SPECS))
    assert all(spec.tab == "Termination" for spec in m.NATURAL_TERMINATION_SETTING_SPECS.values())


def test_the_kernel_tuples_are_built_from_the_records():
    termination = m.settings_natural_termination(m.DEFAULT_NATURAL_TERMINATION_SETTINGS)
    assert isinstance(termination, m.NaturalTerminationRule) and termination == m.NATURAL_TERMINATION_DEFAULT
    assert termination._fields == ("max_iter", "max_dist", "escape_radius", "confine")
    assert termination.max_iter == 256 and termination.max_dist == float("inf") and termination.escape_radius == 2.0
    assert termination.confine == m.CONFINE_DEFAULT
    assert (type(termination.max_iter), type(termination.max_dist), type(termination.escape_radius)) == (int, float, float)
    limited = m.settings_natural_termination(m.DEFAULT_NATURAL_TERMINATION_SETTINGS.replace(
        max_iter=9, max_dist=2.5, escape_radius=7, confine_lookback=6, confine_method=m.CONFINE_PATH,
        confine_threshold=0.1))
    assert limited == m.NaturalTerminationRule(max_iter=9, max_dist=2.5, escape_radius=7.0,
                                        confine=m.ConfineRule(lookback=6, method=m.CONFINE_PATH, threshold=0.1))
    assert type(limited.escape_radius) is float                     # an int is made the float the kernels need
    arrays = _starting_obst_arrays()
    engine_args = m.settings_ray_args(_varied_ray_settings(), arrays)
    assert isinstance(engine_args, m.RayArgs) and engine_args._fields == ("seg_mult", "fuzz", "emission", "obstacles")
    assert engine_args.seg_mult == 0.8 and engine_args.fuzz == m.fuzz_rule(_varied_ray_settings().fuzz)
    assert engine_args.emission == m.settings_emission(_varied_ray_settings()) and engine_args.obstacles is arrays


# --------------------------------------------------------------------------- #
# What an image was generated from: one GeneratedFigure
# --------------------------------------------------------------------------- #
def test_a_generated_figure_holds_the_settings_the_kernel_arguments_and_the_obstacles_as_they_were():
    import dataclasses
    ray, termination = _varied_ray_settings(), _varied_natural_termination_settings()
    scene = [m.default_starting_obstacle(0), m.EllipseShape(3)]
    figure = m.generated_figure(ray, termination, scene)
    assert figure.ray is ray and figure.natural_termination is termination
    assert figure.termination == m.settings_natural_termination(termination)
    assert figure.engine_args.obstacles.shapes.tolist() == m.build_obstacle_arrays(scene).shapes.tolist()
    assert figure.engine_args.seg_mult == ray.seg_mult and figure.rng_seed == ray.fuzz.rng_seed
    assert figure.obstacles_signature == m.obstacles_signature(scene)
    assert figure.obstacle_states == {0: scene[0].state(), 3: scene[1].state()}
    assert [o.id for o in figure.obstacle_snapshots] == [0, 3]
    scene[0].r = 0.5                                              # the scene moves on: the figure does not
    assert figure.obstacles_signature != m.obstacles_signature(scene)
    assert _raises(dataclasses.FrozenInstanceError, lambda: setattr(figure, "ray", m.DEFAULT_RAY_SETTINGS))


def test_the_placeholder_figure_is_the_defaults_with_no_obstacle():
    figure = m.PLACEHOLDER_FIGURE
    assert figure.ray is m.DEFAULT_RAY_SETTINGS and figure.natural_termination is m.DEFAULT_NATURAL_TERMINATION_SETTINGS
    assert figure.termination == m.NATURAL_TERMINATION_DEFAULT
    assert figure.engine_args.seg_mult == m.DEFAULT_RAY_SETTINGS.seg_mult and figure.engine_args.emission == m.EMISSION_DEFAULT
    assert figure.engine_args.fuzz == m.fuzz_rule(m.DEFAULT_FUZZ) and len(figure.engine_args.obstacles.shapes) == 0
    assert figure.obstacles_signature == () and figure.obstacle_snapshots == () and figure.obstacle_states == {}
    assert m.settings_of(None) == (None, None) and m.settings_of(figure) == (figure.ray, figure.natural_termination)


def test_start_makes_the_whole_figure_at_once():
    class FakeApp:
        pass
    app = FakeApp()
    ray, termination = _varied_ray_settings(), _varied_natural_termination_settings()
    app.current_generation_settings = lambda: (ray, termination)
    app.tick_id = None
    app.obstacles = [m.default_starting_obstacle(0)]
    app.begin_generation = lambda: app.__dict__.setdefault("began", []).append(app.generated)
    assert getattr(app, "generated", None) is None
    m.App.start(app)
    assert app.began == [app.generated] and app.generated.ray is ray and app.generated.natural_termination is termination
    assert app.baseline is ray and app.baseline_natural_termination is termination        # Revert returns to them
    assert app.generated.obstacles_signature == m.obstacles_signature(app.obstacles)
