"""Tests: settings_model (the settings model, its parsers, and the binding of a control to it). No Tk is needed for any of them."""

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


def _model():
    return m.SettingsModel(m.MODEL_SETTING_SPECS)


def _listening(model):
    calls = []
    model.subscribe(lambda changes: calls.append(dict(changes)))
    return calls


def test_a_new_model_holds_the_defaults_and_makes_the_default_record():
    model = _model()
    assert model.record(m.NaturalTerminationSettings) == m.DEFAULT_NATURAL_TERMINATION_SETTINGS
    assert model.record(m.BuddhaBlurSettings) == m.DEFAULT_BUDDHA_BLUR and model.record(m.BuddhaSeedSettings) == m.DEFAULT_BUDDHA_SEED
    assert all(model.error(key) is None for key in model.keys())
    assert sorted(model.keys()) == sorted(m.MODEL_SETTING_SPECS)
    assert model.record(m.RaySettings) == m.DEFAULT_RAY_SETTINGS
    assert (set(m.NATURAL_TERMINATION_SETTING_SPECS) | set(m.BLUR_SETTING_SPECS) | set(m.BUDDHA_SEED_SETTING_SPECS) | set(m.RAY_SETTING_SPECS)
            == set(model.keys()))


def test_a_valid_text_changes_the_value_and_an_invalid_one_keeps_it_and_sets_the_error():
    model = _model()
    assert model.submit("confine_lookback", "9") is None and model.value("confine_lookback") == 9
    message = model.submit("confine_lookback", "1")
    assert "whole number from 2 to 999" in message and model.error("confine_lookback") == message
    assert model.value("confine_lookback") == 9, "the last valid value stays"
    assert model.record(m.NaturalTerminationSettings) is None, "no record while a text is unusable"
    assert model.submit("confine_lookback", "12") is None and model.error("confine_lookback") is None
    assert model.record(m.NaturalTerminationSettings).confine_lookback == 12


def test_an_error_in_one_setting_does_not_touch_the_others():
    model = _model()
    model.submit("confine_threshold", "0")
    model.submit("iter", "77")
    assert model.value("iter") == 77 and model.error("iter") is None
    assert model.error("confine_threshold") is not None


def test_set_goes_through_the_same_text_and_parser_as_typing():
    """A keyframe's interpolated 8.6 reaches an int setting as 9, a long distance is rounded to two places as an entry's text
    would be, and a value out of range is an error exactly as typed text would be."""
    model = _model()
    assert model.set("confine_lookback", 8.6) is None and model.value("confine_lookback") == 9
    assert model.set("dist", 12.3456) is None and model.value("dist") == 12.35
    assert model.set("confine_method", 1) is None and model.value("confine_method") == 1
    assert model.set("iter", 99999) is not None and model.value("iter") == m.DEFAULT_ITER and model.error("iter") is not None


def test_a_set_clears_the_error_of_an_earlier_bad_text():
    model = _model()
    model.submit("escape_radius", "abc")
    assert model.error("escape_radius") is not None
    model.set("escape_radius", 3.0)
    assert model.error("escape_radius") is None and model.value("escape_radius") == 3.0


def test_listeners_hear_each_change_once_with_its_origin_and_nothing_when_nothing_changed():
    model = _model()
    calls = _listening(model)
    model.submit("iter", "50")
    model.submit("iter", "50")                                   # the same value again: nothing changed
    model.submit("iter", "abc")                                  # the error appeared
    model.submit("iter", "abc")                                  # ... and is the same
    model.set("iter", 50)                                        # a program write is always reported, so the view can show it
    assert calls == [{"iter": "submit"}, {"iter": "submit"}, {"iter": "set"}]


def test_a_batch_tells_the_listeners_once_at_its_end_and_batches_nest():
    model = _model()
    calls = _listening(model)
    with model.batch():
        model.set("iter", 10)
        with model.batch():
            model.set("dist", 3)
        model.submit("escape_radius", "5")
        assert calls == [], "nothing is reported inside the block"
    assert calls == [{"iter": "set", "dist": "set", "escape_radius": "submit"}]


def test_the_model_makes_no_partial_records():
    model = _model()
    try:
        model.record(m.EngineSettings)
    except KeyError as exc:
        assert "EngineSettings" in str(exc)
    else:
        raise AssertionError("expected a KeyError for a record the model does not manage")


def test_every_text_a_setting_formats_to_is_read_back_as_the_same_value():
    varied = m.settings_to_dict(_varied_natural_termination_settings(), _varied_blur_settings(), _varied_seed_settings(), _varied_ray_settings())
    for key in m.MODEL_SETTING_SPECS:
        for value in (m.DEFAULT_SETTING_VALUES[key], varied[key]):
            model = _model()
            assert model.set(key, value) is None, key
            assert model.value(key) == value, key


def test_the_self_checks_catch_a_parser_that_cannot_read_its_own_default():
    fresh = load_module()
    fresh.selfchecks.MODEL_SETTING_SPECS = {**fresh.MODEL_SETTING_SPECS, "iter": fresh.MODEL_SETTING_SPECS["iter"].replace(parse=lambda text: None)}
    _breaks_a_self_check(fresh, "does not read back the text of its own default value")


# ---- settings whose unusable text is ignored, and checkboxes (the Buddha blur and seeding) --------------
def test_a_lenient_setting_ignores_text_that_is_no_number_clamps_one_that_is_too_big_and_never_holds_an_error():
    model = _model()
    assert model.submit("buddha_blur_radius", "0.2") is None and model.value("buddha_blur_radius") == 0.2
    assert model.submit("buddha_blur_radius", "-") is None and model.value("buddha_blur_radius") == 0.2 and model.error("buddha_blur_radius") is None
    assert model.submit("buddha_blur_radius", "0.9") is None and model.value("buddha_blur_radius") == m.BUDDHA_BLUR_RADIUS_MAX
    assert model.submit("buddha_blur_radius", "-3") is None and model.value("buddha_blur_radius") == m.BUDDHA_BLUR_RADIUS_MIN
    assert model.record(m.BuddhaBlurSettings) is not None, "a blur is never None"


def test_the_seeding_zone_keeps_the_last_valid_value_while_a_number_is_half_typed():
    model = _model()
    model.submit("buddha_seed_x_start", "-.5")
    assert model.value("buddha_seed_x_start") == -0.5
    for half in ("-", "", "1e", "abc"):
        model.submit("buddha_seed_x_start", half)
        assert model.value("buddha_seed_x_start") == -0.5 and model.error("buddha_seed_x_start") is None, half
    assert model.record(m.BuddhaSeedSettings).x_start == -0.5


def test_a_program_write_to_a_lenient_setting_is_clamped_and_still_reported_so_the_view_shows_what_was_kept():
    model = _model()
    calls = _listening(model)
    model.set("buddha_blur_radius", 5)
    assert model.value("buddha_blur_radius") == m.BUDDHA_BLUR_RADIUS_MAX and calls == [{"buddha_blur_radius": "set"}]


def test_a_checkbox_setting_holds_a_bool_and_shows_one():
    model = _model()
    assert model.value("buddha_seed_camera") is True
    model.submit("buddha_seed_camera", False)
    assert model.value("buddha_seed_camera") is False
    model.set("buddha_seed_camera", 1)
    assert model.value("buddha_seed_camera") is True and m.format_value("buddha_seed_camera", 0) is False


def test_the_blur_type_is_chosen_by_name_and_an_unknown_name_is_ignored():
    model = _model()
    model.submit("buddha_blur_type", m.BUDDHA_BLUR_LABELS[m.BUDDHA_BLUR_BOKEH])
    assert model.value("buddha_blur_type") == m.BUDDHA_BLUR_BOKEH
    model.submit("buddha_blur_type", "no such blur")
    assert model.value("buddha_blur_type") == m.BUDDHA_BLUR_BOKEH and model.error("buddha_blur_type") is None


# ---- the ray-bouncing engine's settings ---------------------------------------------------------
def test_the_emission_target_is_not_lenient_a_bad_number_is_an_error_and_there_is_no_ray_record_then():
    model = _model()
    message = model.submit("emit_target_theta", "abc")
    assert "Target position values must be numbers" in message and model.record(m.RaySettings) is None
    assert model.last_valid_record(m.RaySettings) == m.DEFAULT_RAY_SETTINGS, "a preview still has the last valid values"
    model.submit("emit_target_theta", "-720")
    assert model.record(m.RaySettings).emit_target_theta == -720.0


def test_a_bad_fuzzing_seed_is_flagged_but_never_stops_a_ray_record_and_blank_is_zero():
    model = _model()
    model.submit("seed", "123")
    assert model.submit("seed", "12x") is not None and model.error("seed") is not None
    assert model.record(m.RaySettings).fuzz.rng_seed == 123, "the last valid seed stands"
    assert model.submit("seed", "") is None and model.value("seed") == 0 and model.error("seed") is None


def test_the_ray_entries_clamp_round_and_format_as_their_old_number_entries_did():
    model = _model()
    model.submit("seg_mult", "5")
    assert model.value("seg_mult") == m.SEGMENT_MULT_MAX
    model.submit("seg_mult", "0.755")
    assert model.value("seg_mult") == 0.755, "the multiplier is not rounded (only its text is)"
    assert m.format_value("seg_mult", 0.755) == "0.76" and m.format_value("emit_jitter", 4.0) == "4.0" and m.format_value("bnd_mult", 1.25) == "1.25"
    model.submit("bnd_mult", "9.87654")
    assert model.value("bnd_mult") == m.FUZZ_MULT_MAX
    model.submit("bnd_mult", "1.23456")
    assert model.value("bnd_mult") == 1.235, "rounded to three places like the record always was"
    model.submit("emit_offset", "-45.678")
    assert model.value("emit_offset") == -45.68


def test_a_choice_of_emission_direction_is_by_name():
    model = _model()
    model.submit("emit_mode", m.EMIT_MODE_NAMES[m.EMIT_TOWARDS])
    assert model.record(m.RaySettings).emit_mode == m.EMIT_TOWARDS


# ---- the binding of a control to the model --------------------------------------------------
def _bound(key="iter"):
    model = _model()
    var, entry = _StubVar(m.format_value(key, model.value(key))), _StubEntry()
    binding = m.ModelBinding(model, key, var, entry, lambda value: m.format_value(key, value))
    heard = _listening(model)
    return model, var, entry, binding, heard


def test_typing_into_a_bound_variable_reaches_the_model_and_a_bad_text_makes_the_entry_red():
    model, var, entry, binding, _heard = _bound()
    var.set("40")
    assert model.value("iter") == 40
    var.set("abc")
    binding.show_error()
    assert model.error("iter") is not None and entry.style == "Invalid.TEntry"
    var.set("41")
    binding.show_error()
    assert entry.style == "TEntry"


def test_a_value_the_program_sets_is_written_into_the_variable_without_being_submitted_back():
    model, var, _entry, binding, heard = _bound()
    model.set("iter", 33)
    assert heard == [{"iter": "set"}]
    binding.write_value()
    assert var.get() == "33"
    assert heard == [{"iter": "set"}], "writing the variable must not report a typed text"


def test_committing_puts_the_last_valid_value_back_tidily_and_clears_the_error():
    model, var, _entry, binding, _heard = _bound("confine_threshold")
    var.set("4E-3")
    binding.commit()
    assert var.get() == "0.004" and model.error("confine_threshold") is None
    var.set("-1")
    assert model.error("confine_threshold") is not None
    binding.commit()
    assert var.get() == "0.004", "the bad text is replaced by the last valid value"
    assert model.error("confine_threshold") is None


def test_the_window_composes_its_messages_from_the_models_errors():
    app = type("FakeApp", (), {})()
    _give_settings_model(app)
    app.iter_var.set("0")
    app.confine_threshold_var.set("0")
    assert "Iteration limit" in app.limit_msg_var.get() and app.confine_msg_var.get().startswith("Threshold")
    app.iter_var.set("5")
    assert app.limit_msg_var.get() == "" and app.confine_msg_var.get() != ""
