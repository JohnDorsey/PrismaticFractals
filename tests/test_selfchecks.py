"""Tests: selfchecks."""

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# Registry self-checks
# --------------------------------------------------------------------------- #
def test_self_checks_pass_on_the_real_file():
    """run_self_checks() (in light_bounces.py itself) already runs at import
    time, so if we got this far it already passed - this just re-runs it
    explicitly so a failure here points straight at the registry check
    rather than an opaque import error."""
    m.run_self_checks()


def test_self_checks_catch_a_missing_termination_color():
    """Regression test: TERM_PERPENDICULAR was once missing from the
    ray-end-color dict, causing a KeyError the first time a ray terminated
    that way. Confirms the check that would have caught it actually fires."""
    fresh = load_module()
    del fresh.TERM_END_COLOR[fresh.TERM_PERPENDICULAR]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "TERM_END_COLOR" in str(e)


def test_self_checks_catch_an_unknown_float_key():
    fresh = load_module()
    fresh.selfchecks.SETTING_FLOAT_KEYS = fresh.SETTING_FLOAT_KEYS | {"not_a_real_setting"}
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "SETTING_FLOAT_KEYS" in str(e)


def test_self_checks_catch_a_setting_info_mismatch():
    fresh = load_module()
    del fresh.DEFAULT_SETTING_VALUES["dist"]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "SETTING_INFO" in str(e)
