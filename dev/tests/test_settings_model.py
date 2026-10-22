"""Tests: settings_model (the settings model, its parsers, and the binding of a control to it). No Tk is needed for any of them."""

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


def _model():
    return m.SettingsModel(m.NATURAL_TERMINATION_PARSERS)


def _listening(model):
    calls = []
    model.subscribe(lambda changes: calls.append(dict(changes)))
    return calls


def test_a_new_model_holds_the_defaults_and_makes_the_default_record():
    model = _model()
    assert model.record(m.NaturalTerminationSettings) == m.DEFAULT_NATURAL_TERMINATION_SETTINGS
    assert all(model.error(key) is None for key in model.keys())
    assert sorted(model.keys()) == sorted(m.NATURAL_TERMINATION_SETTING_SPECS)


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
        model.record(m.RaySettings)
    except KeyError as exc:
        assert "RaySettings" in str(exc)
    else:
        raise AssertionError("expected a KeyError for a record the model does not manage")


def test_every_text_a_setting_formats_to_is_read_back_as_the_same_value():
    for key in m.NATURAL_TERMINATION_PARSERS:
        for value in (m.DEFAULT_SETTING_VALUES[key], m.settings_to_dict(_varied_natural_termination_settings())[key]):
            model = _model()
            assert model.set(key, value) is None, key
            assert model.value(key) == value, key


def test_the_self_checks_catch_a_parser_that_cannot_read_its_own_default():
    fresh = load_module()
    fresh.selfchecks.NATURAL_TERMINATION_PARSERS = {
        **fresh.NATURAL_TERMINATION_PARSERS, "iter": fresh.SettingParser(parse=lambda text: None, message="never")}
    _breaks_a_self_check(fresh, "does not read back the text of its own default value")


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
