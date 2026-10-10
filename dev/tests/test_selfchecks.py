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


def test_self_checks_catch_a_sequence_catalog_of_the_wrong_length():
    fresh = load_module()
    fresh.selfchecks.SEQUENCE_LABELS = fresh.SEQUENCE_LABELS[:-1]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "SEQUENCE_LABELS" in str(e)


def test_self_checks_catch_a_point_alias_that_is_not_its_own_slot_and_a_slot_given_twice():
    fresh = load_module()
    rows = list(fresh.POINT_ALIASES)
    rows[2], rows[3] = rows[3], rows[2]
    fresh.selfchecks.POINT_ALIASES = tuple(rows)
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "point choice" in str(e)
    fresh = load_module()
    slots = fresh.POINT_SLOT.copy()
    slots[fresh.SEQ_SELF, fresh.PRED_LEFT] = slots[fresh.SEQ_PATH, fresh.PRED_FIRST]
    fresh.selfchecks.POINT_SLOT = slots
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "POINT_SLOT" in str(e)


def test_self_checks_catch_a_metric_alias_that_is_not_its_own_slot():
    fresh = load_module()
    fresh.selfchecks.STAT_ALIASES = {**fresh.STAT_ALIASES, fresh.ST_LENGTH: (fresh.SEQ_SELF, fresh.SRED_LENGTH)}
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "metric" in str(e)
