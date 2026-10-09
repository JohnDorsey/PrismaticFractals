"""Tests: engines (what each path engine offers and accepts, the one correction, switching engine)."""
import random

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


def _channel(**changes):
    """Channel 0 of a default coloration (a Query channel comparing two points), with `changes`."""
    return m.coloration_channel_value(m.ColorationConfig(1), 0).replace(**changes)


def _ticked(*labels):
    return tuple(label in labels for label in m.BUDDHA_POINT_LABELS)


def _corrected(settings, engine=None, **kwargs):
    return m.correct_settings(settings, engine or small_engine(), **kwargs)


def _in(corrections, *tabs):
    """The corrections made in these tabs (a scene of default settings has others, which these tests are not about)."""
    return [c for c in corrections if c.tab in tabs]


# --------------------------------------------------------------------------- #
# The descriptions
# --------------------------------------------------------------------------- #
def test_the_ray_engine_offers_every_choice_so_nothing_is_ever_corrected_under_it():
    odd = settings_dict_with_lists(
        obstacle_effects=[(m.OBSTACLE_EFFECT_REFLECT, m.OBSTACLE_EFFECT_TERMINATE), (m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_REFLECT)],
        channels=[_channel(metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_LOOP_AREA),
                  _channel(chan_src=m.CHAN_SRC_BUDDHA, buddha_points=_ticked("Self bounces: Mean point", "All self bounces",
                                                                              "Path: Last point (perpendicular collision)")),
                  _channel(point1=m.CHOICE_TOP_LEFT_POINT, point2=m.point_slot(m.SEQ_SELF, m.PRED_TOP))])
    odd["iter"], odd["seg_mult"] = 100000, 7.0
    corrected, corrections = m.correct_settings(odd, m.RAY_BOUNCING_ENGINE)
    assert corrected == odd and corrections == [] and corrected is not odd
    assert m.RAY_BOUNCING_ENGINE.metrics == frozenset(range(m.N_STATS)), "the ray engine's ids are the catalogs' own (D34)"
    assert m.RAY_BOUNCING_ENGINE.points == frozenset(range(m.N_POINT_CHOICES))
    assert m.RAY_BOUNCING_ENGINE.buddha_options == frozenset(range(m.N_BUDDHA_POINTS))
    assert m.ENGINE_DESCRIPTIONS == {m.ENGINE_RAY_BOUNCING: m.RAY_BOUNCING_ENGINE}
    assert m.engine_description(m.ENGINE_RAY_BOUNCING) is m.RAY_BOUNCING_ENGINE


def test_an_engine_without_a_description_says_so_by_name():
    try:
        m.engine_description(m.ENGINE_MANDELBROT)
    except KeyError as exc:
        assert "Mandelbrot" in str(exc) and "no description" in str(exc)
    else:
        raise AssertionError("expected a KeyError")


def test_an_engine_makes_no_use_of_the_settings_of_its_ignored_tabs_and_keys():
    engine = small_engine(ignored_keys=frozenset({"outline"}))
    assert engine.ignores("seg_mult") and engine.ignores("emit_mode") and engine.ignores("fuzz_on") and engine.ignores("outline")
    assert not engine.ignores("iter") and not engine.ignores("escape_radius") and not engine.ignores("engine")
    assert not any(m.RAY_BOUNCING_ENGINE.ignores(key) for key in m.DEFAULT_SETTING_VALUES)


def test_the_keys_of_obstacles_and_colorations_have_patterns_with_the_id_left_out():
    assert m.key_pattern("obst:12:in_effect") == "obst:*:in_effect"
    assert m.key_pattern("coloration:3:chan1") == "coloration:*:chan1"
    assert m.key_pattern("framing:2:x") == "framing:*:x" and m.key_pattern("iter") == "iter"


# --------------------------------------------------------------------------- #
# The correction: colour channels
# --------------------------------------------------------------------------- #
def test_a_metric_the_engine_does_not_offer_becomes_the_channels_usual_default_or_else_the_engines_own():
    path_count = m.stat_slot(m.SEQ_PATH, m.SRED_COUNT)
    values = settings_dict_with_lists(channels=[
        _channel(metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_LOOP_AREA),
        _channel(metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_BOUNCE_COUNT),
        _channel(metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_TERMINATION)])
    corrected, corrections = _corrected(values)
    assert [corrected[f"coloration:1:chan{c}"].metric_slot for c in range(3)] == [path_count, path_count, m.ST_TERMINATION]
    assert [(c.key, c.field, c.old, c.new) for c in _in(corrections, "Colorations")] == [
        ("coloration:1:chan0", "metric_slot", m.ST_LOOP_AREA, path_count),
        ("coloration:1:chan1", "metric_slot", m.ST_BOUNCE_COUNT, path_count)]
    first = _in(corrections, "Colorations")[0]
    assert (first.tab, first.label, first.old_text, first.new_text) == (
        "Colorations", "Coloration 1, channel 1: Metric", "Loop area / same-perimeter circle", "Path: Point count / limit")
    # the usual default of the channel is preferred where the engine offers it: channel 1's own is Vertical distance
    only_vertical = small_engine(metrics=small_engine().metrics - {m.ST_HDIST, m.ST_MANHATTAN}, default_point_metric=m.ST_VDIST)
    pair = lambda slot: settings_dict_with_lists(channels=[_channel(metric_slot=slot)] * 3)
    got, _ = _corrected(pair(m.ST_HDIST), only_vertical)
    assert [got[f"coloration:1:chan{c}"].metric_slot for c in range(3)] == [m.ST_VDIST] * 3, "(HDIST, MANHATTAN are not offered)"
    two = small_engine(metrics=small_engine().metrics - {m.ST_HDIST, m.ST_VDIST}, default_point_metric=m.ST_EUCLIDEAN)
    got, _ = _corrected(settings_dict_with_lists(channels=[_channel(metric_slot=m.ST_HDIST), _channel(metric_slot=m.ST_VDIST),
                                                             _channel(metric_slot=m.ST_HDIST)]), two)
    assert [got[f"coloration:1:chan{c}"].metric_slot for c in range(3)] == [m.ST_EUCLIDEAN, m.ST_EUCLIDEAN, m.ST_MANHATTAN]


def test_a_point_the_engine_does_not_offer_becomes_the_start_or_end_or_else_the_engines_own():
    bad = m.point_slot(m.SEQ_SELF, m.PRED_TOP)
    values = settings_dict_with_lists(channels=[_channel(point1=bad, point2=bad), _channel(point1=m.CHOICE_ORIGIN, point2=bad),
                                                _channel(point1=bad, point2=m.POINT_END)])
    corrected, corrections = _corrected(values)
    got = [(corrected[f"coloration:1:chan{c}"].point1, corrected[f"coloration:1:chan{c}"].point2) for c in range(3)]
    assert got == [(m.POINT_START, m.POINT_END), (m.CHOICE_ORIGIN, m.POINT_END), (m.POINT_START, m.POINT_END)]
    corrections = _in(corrections, "Colorations")
    assert [(c.key, c.field) for c in corrections] == [("coloration:1:chan0", "point1"), ("coloration:1:chan0", "point2"),
                                                       ("coloration:1:chan1", "point2"), ("coloration:1:chan2", "point1")]
    assert (corrections[0].label, corrections[0].old_text, corrections[0].new_text) == (
        "Coloration 1, channel 1: Point 1", "Self bounces: Uppermost point", "Path: First point")
    no_start = small_engine(points=small_engine().points - {m.POINT_START, m.POINT_END}, default_point=m.CHOICE_ORIGIN)
    got, _ = _corrected(settings_dict_with_lists(channels=[_channel(point1=bad, point2=bad)] * 3), no_start)
    assert (got["coloration:1:chan2"].point1, got["coloration:1:chan2"].point2) == (m.CHOICE_ORIGIN, m.CHOICE_ORIGIN)


def test_a_buddha_channel_loses_the_options_the_engine_does_not_offer_and_gets_the_default_only_if_none_is_left():
    mean_bounces, mean_path = "All bounces: Mean point", "Path: Mean point"
    values = settings_dict_with_lists(channels=[
        _channel(chan_src=m.CHAN_SRC_BUDDHA, buddha_points=_ticked(mean_bounces)),
        _channel(chan_src=m.CHAN_SRC_BUDDHA, buddha_points=_ticked(mean_bounces, "Path: Last point", "All segment midpoints")),
        _channel(chan_src=m.CHAN_SRC_BUDDHA, buddha_points=_ticked())])
    corrected, corrections = _corrected(values)
    corrections = _in(corrections, "Colorations")
    assert corrected["coloration:1:chan0"].buddha_points == _ticked(mean_path), "(none left: the engine's default option)"
    assert corrected["coloration:1:chan1"].buddha_points == _ticked("Path: Last point"), "(one is left: nothing is added)"
    assert corrected["coloration:1:chan2"] == values["coloration:1:chan2"], "(a channel that scatters nothing is as it was)"
    assert [(c.field, c.old_text, c.new_text) for c in corrections] == [
        ("buddha_points", mean_bounces, mean_path),
        ("buddha_points", f"Path: Last point, {mean_bounces}, All segment midpoints", "Path: Last point")]
    assert corrections[0].label == "Coloration 1, channel 1: Buddha points"


def test_a_channel_is_corrected_only_in_the_fields_it_uses():
    bad_metric, bad_point = m.ST_LOOP_AREA, m.point_slot(m.SEQ_SELF, m.PRED_TOP)
    values = settings_dict_with_lists(channels=[
        _channel(chan_src=m.CHAN_SRC_CONSTANT, metric_type=m.METRIC_TYPE_OTHER, metric_slot=bad_metric, point1=bad_point,
                 buddha_points=_ticked("Self bounces: Mean point")),
        _channel(chan_src=m.CHAN_SRC_BUDDHA, metric_type=m.METRIC_TYPE_OTHER, metric_slot=bad_metric, point1=bad_point),
        _channel(metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_TERMINATION, point1=bad_point, point2=bad_point,
                 buddha_points=_ticked("Self bounces: Mean point"))])
    corrected, corrections = _corrected(values)
    assert [(c.key, c.field) for c in _in(corrections, "Colorations")] == [("coloration:1:chan1", "buddha_points")]
    assert corrected["coloration:1:chan1"].metric_slot == bad_metric and corrected["coloration:1:chan1"].point1 == bad_point
    assert corrected["coloration:1:chan0"] == values["coloration:1:chan0"], "a constant channel uses no choice at all"
    assert corrected["coloration:1:chan2"] == values["coloration:1:chan2"], "an 'other' metric reads no points, a Query channel no Buddha option"


# --------------------------------------------------------------------------- #
# The correction: other settings
# --------------------------------------------------------------------------- #
def test_an_effect_the_engine_does_not_offer_becomes_its_fallback_and_the_others_are_left_alone():
    values = settings_dict_with_lists(obstacle_effects=[(m.OBSTACLE_EFFECT_REFLECT, m.OBSTACLE_EFFECT_TERMINATE),
                                                         (m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_REFLECT)])
    corrected, corrections = _corrected(values, titles={"obst:2": "Obstacle #2 (Lens)"})
    assert [corrected[f"obst:{i}:{f}"] for i in (1, 2) for f in ("in_effect", "out_effect")] == [
        m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_TERMINATE, m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_PASS]
    assert corrected["obst:1:fire_inside"] is True and corrected["obst:1:radius"] == 2.0
    assert [(c.tab, c.label, c.old_text, c.new_text) for c in _in(corrections, "Obstacles")] == [
        ("Obstacles", "Obstacle 1: Incoming ray effect", "Reflect", "Pass through"),
        ("Obstacles", "Obstacle #2 (Lens): Outgoing ray effect", "Reflect", "Pass through")]


def test_a_new_obstacle_starts_with_the_effects_the_engine_defaults_to():
    assert m.new_obstacle_effects(m.RAY_BOUNCING_ENGINE) == (m.OBSTACLE_EFFECT_REFLECT, m.OBSTACLE_EFFECT_REFLECT)
    assert m.new_obstacle_effects(small_engine()) == (m.OBSTACLE_EFFECT_PASS, m.OBSTACLE_EFFECT_PASS)
    try:
        m.new_obstacle_effects(small_engine(fallbacks={}))
    except ValueError as exc:
        assert "obst:*:in_effect" in str(exc)
    else:
        raise AssertionError("an engine that rejects the usual default and names no other must say so")


def test_a_number_outside_the_engines_range_is_clamped_and_a_whole_number_stays_whole():
    values = settings_dict_with_lists()
    values.update(iter=500, seg_mult=0.1, escape_radius=4.0, fuzz_on=True)
    original = dict(values)
    corrected, corrections = _corrected(values)
    assert values == original, "the input is not changed"
    assert corrected["iter"] == 100 and type(corrected["iter"]) is int
    assert corrected["seg_mult"] == 0.5 and corrected["escape_radius"] == 4.0 and corrected["fuzz_on"] is True
    assert [(c.tab, c.label, c.old_text, c.new_text) for c in _in(corrections, "Termination", "Mutations")] == [
        ("Termination", "Iteration limit", "500", "100"), ("Mutations", "Segment length multiplier", "0.1", "0.5")]
    half = small_engine(ranges={"iter": (10.5, 20.5), "escape_radius": (None, 3), "seg_mult": (2, None)})
    got, _ = _corrected({"iter": 1000, "escape_radius": 4.0, "seg_mult": 1.0}, half)
    assert got == {"iter": 20, "escape_radius": 3, "seg_mult": 2}
    got, _ = _corrected({"iter": 3}, half)
    assert got["iter"] == 11, "(a whole number is moved into the range, never out of it)"


def test_the_range_of_a_dynamic_setting_is_found_by_its_pattern_and_the_engine_key_is_never_touched():
    engine = small_engine(ranges={"obst:*:radius": (None, 1.5), "engine": (5, 6)})
    got, corrections = _corrected({"obst:4:radius": 2.0, "obst:5:radius": 1.0, "engine": 0}, engine)
    assert got == {"obst:4:radius": 1.5, "obst:5:radius": 1.0, "engine": 0}
    assert [(c.key, c.tab, c.label) for c in corrections] == [("obst:4:radius", "Obstacles", "Obstacle 4: radius")]


def test_a_keyframe_with_only_some_settings_is_corrected_for_what_it_holds():
    got, corrections = _corrected({"obst:1:in_effect": m.OBSTACLE_EFFECT_REFLECT, "iter": 50})
    assert got == {"obst:1:in_effect": m.OBSTACLE_EFFECT_PASS, "iter": 50} and len(corrections) == 1
    assert _corrected({}) == ({}, [])


def test_the_correction_is_idempotent_and_changes_nothing_else():
    rng = random.Random(7)
    engine = small_engine()
    for _ in range(200):
        values = _random_settings(rng)
        once, corrections = m.correct_settings(values, engine)
        again, more = m.correct_settings(once, engine)
        assert more == [] and again == once, (values, once, more)
        assert {k for k in values if values[k] != once[k]} == {c.key for c in corrections}
        assert m.correct_settings(values, m.RAY_BOUNCING_ENGINE) == (values, [])


# --------------------------------------------------------------------------- #
# The words
# --------------------------------------------------------------------------- #
def test_the_corrections_are_worded_as_the_popup_of_the_design_shows_them():
    values = settings_dict_with_lists(obstacle_effects=[(m.OBSTACLE_EFFECT_REFLECT, m.OBSTACLE_EFFECT_PASS)],
                                      channels=[_channel(metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_BOUNCE_COUNT)] * 3)
    values["iter"] = 500
    _corrected_values, corrections = _corrected(values)
    assert m.correction_message(corrections) == "\n".join([
        "Some settings that weren't supported by this engine have been reset:",
        "  in the Termination tab:",
        "    Iteration limit: 500 -> 100",
        "  in the Obstacles tab:",
        "    Obstacle 1: Incoming ray effect: Reflect -> Pass through",
        "  in the Colorations tab:",
        "    Coloration 1, channel 1: Metric: All bounces: Point count / limit -> Path: Point count / limit",
        "    Coloration 1, channel 2: Metric: All bounces: Point count / limit -> Path: Point count / limit",
        "    Coloration 1, channel 3: Metric: All bounces: Point count / limit -> Path: Point count / limit"])
    assert m.correction_message(corrections[:1], m.PROJECT_HEADER) == (
        "Some values from this project have been clamped or defaulted:\n  in the Termination tab:\n    Iteration limit: 500 -> 100")
    assert m.KEYFRAME_HEADER.startswith("Some values from this keyframe have been clamped or defaulted")
    assert [tab for tab, _items in m.group_corrections(list(reversed(corrections)))] == ["Termination", "Obstacles", "Colorations"]


# --------------------------------------------------------------------------- #
# Switching engine, memory, go back
# --------------------------------------------------------------------------- #
RAY = m.RAY_BOUNCING_ENGINE


def _double():
    return small_engine()


def _reflecting_scene():
    values = settings_dict_with_lists(
        obstacle_effects=[(m.OBSTACLE_EFFECT_REFLECT, m.OBSTACLE_EFFECT_TERMINATE)],
        channels=[_channel(metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_BOUNCE_COUNT),
                  _channel(chan_src=m.CHAN_SRC_BUDDHA, buddha_points=_ticked("All bounces: Mean point"), buddha_compression=0),
                  _channel(point1=m.point_slot(m.SEQ_SELF, m.PRED_TOP))])
    values["iter"] = 500
    return values


def test_switching_to_another_engine_corrects_and_switching_back_restores_what_it_reset():
    values = _reflecting_scene()
    engine = _double()
    out = m.switch_engine(values, RAY, engine)
    assert out.settings["engine"] == engine.engine and out.settings["iter"] == 100
    assert out.settings["obst:1:in_effect"] == m.OBSTACLE_EFFECT_PASS and out.settings["obst:1:out_effect"] == m.OBSTACLE_EFFECT_TERMINATE
    assert len(out.corrections) == 5 and out.restored == ()
    assert values["engine"] == m.ENGINE_RAY_BOUNCING and values["iter"] == 500, "the input is not changed"
    back = m.switch_engine(out.settings, engine, RAY, out.memory)
    assert back.settings == values, "nothing was changed meanwhile: everything is as it was"
    assert back.corrections == () and len(back.restored) == 5 and back.memory.kept == {}


def test_a_setting_changed_meanwhile_is_not_restored_but_the_others_are():
    values = _reflecting_scene()
    engine = _double()
    out = m.switch_engine(values, RAY, engine)
    edited = dict(out.settings)
    edited["iter"] = 50                                                              # the user, or a selected keyframe
    chan2 = edited["coloration:1:chan2"]
    edited["coloration:1:chan1"] = edited["coloration:1:chan1"].replace(buddha_compression=1)   # another field of a channel
    edited["coloration:1:chan0"] = edited["coloration:1:chan0"].replace(metric_slot=m.stat_slot(m.SEQ_PATH, m.SRED_LENGTH))
    back = m.switch_engine(edited, engine, RAY, out.memory)
    assert back.settings["iter"] == 50, "the value the user typed stays"
    assert back.settings["coloration:1:chan0"].metric_slot == m.stat_slot(m.SEQ_PATH, m.SRED_LENGTH)
    assert back.settings["coloration:1:chan1"].buddha_compression == 1
    assert back.settings["coloration:1:chan1"].buddha_points == _ticked("All bounces: Mean point"), "(field by field: the points come back)"
    assert back.settings["coloration:1:chan2"].point1 == m.point_slot(m.SEQ_SELF, m.PRED_TOP) != chan2.point1
    assert back.settings["obst:1:in_effect"] == m.OBSTACLE_EFFECT_REFLECT
    assert ("iter", "") not in back.restored and ("obst:1:in_effect", "") in back.restored


def test_a_setting_put_back_to_the_value_it_was_reset_to_counts_as_unchanged():
    values = _reflecting_scene()
    out = m.switch_engine(values, RAY, _double())
    edited = dict(out.settings, iter=60)
    edited["iter"] = 100
    assert m.switch_engine(edited, _double(), RAY, out.memory).settings["iter"] == 500


def test_go_back_undoes_a_switch_and_its_memory():
    values = _reflecting_scene()
    engine = _double()
    out = m.switch_engine(values, RAY, engine)
    settings, memory = m.go_back(out)
    assert settings == values and memory == m.EMPTY_ENGINE_MEMORY
    back = m.switch_engine(out.settings, engine, RAY, out.memory)
    settings, memory = m.go_back(back)
    assert settings == out.settings and memory == out.memory, "(going back from the way back leaves the first switch in place)"
    again = m.switch_engine(settings, engine, RAY, memory)
    assert again.settings == back.settings
    meanwhile = dict(out.settings, **{"obst:9:in_effect": 0})
    undone, _memory = m.go_back(back, meanwhile)
    assert undone["obst:9:in_effect"] == 0 and undone["iter"] == 100, "only what the switch touched is put back"


def test_switching_to_the_engine_already_in_use_changes_nothing():
    values = _reflecting_scene()
    out = m.switch_engine(values, RAY, RAY)
    assert out.settings == values and out.corrections == () and out.touched == ()
    assert m.go_back(out)[0] == values


def test_what_is_reset_is_remembered_afresh_every_time_the_engine_is_left_and_missing_settings_are_skipped():
    values = _reflecting_scene()
    engine = _double()
    first = m.switch_engine(values, RAY, engine)
    back = m.switch_engine(first.settings, engine, RAY, first.memory)
    second = m.switch_engine(back.settings, RAY, engine, back.memory)
    assert second.settings == first.settings and len(second.corrections) == 5
    gone = {k: v for k, v in first.settings.items() if not k.startswith("obst:1:")}
    back = m.switch_engine(gone, engine, RAY, first.memory)
    assert not any(k.startswith("obst:") for k in back.settings) and back.settings["iter"] == 500
    added = dict(first.settings, **{"obst:2:in_effect": m.OBSTACLE_EFFECT_PASS})
    assert m.switch_engine(added, engine, RAY, first.memory).settings["obst:2:in_effect"] == m.OBSTACLE_EFFECT_PASS


def test_ranges_that_differ_between_engines_are_applied_by_each_in_turn():
    a = small_engine(engine=1, name="A", ranges={"iter": (1, 100)})
    b = small_engine(engine=2, name="B", ranges={"iter": (50, 200)})
    values = _reflecting_scene()
    values["iter"] = 20
    to_a = m.switch_engine(values, RAY, a)
    to_b = m.switch_engine(to_a.settings, a, b, to_a.memory)
    assert (to_a.settings["iter"], to_b.settings["iter"]) == (20, 50) and to_b.settings["engine"] == 2
    assert [(c.old, c.new) for c in to_b.corrections if c.key == "iter"] == [(20, 50)]
    back_to_a = m.switch_engine(to_b.settings, b, a, to_b.memory)
    assert back_to_a.settings["iter"] == 20, "A remembered the 20 it had; nothing changed it since"


def test_switching_and_switching_back_with_nothing_changed_gives_back_the_same_settings():
    rng = random.Random(11)
    engine = _double()
    for _ in range(300):
        values = _random_settings(rng)
        out = m.switch_engine(values, RAY, engine)
        assert m.correct_settings(out.settings, engine)[1] == [], "everything is now accepted by the new engine"
        back = m.switch_engine(out.settings, engine, RAY, out.memory)
        assert back.settings == values and back.corrections == (), values
        assert m.go_back(out)[0] == values and m.go_back(back)[0] == out.settings


# --------------------------------------------------------------------------- #
# The self-checks
# --------------------------------------------------------------------------- #
def test_self_checks_catch_an_engine_description_that_is_inconsistent():
    fresh = load_module()
    fresh.selfchecks.ENGINE_DESCRIPTIONS = {0: fresh.RAY_BOUNCING_ENGINE.replace(metrics=frozenset(range(fresh.N_STATS + 1)))}
    _breaks_a_self_check(fresh, "not in the catalogs")
    fresh = load_module()
    fresh.selfchecks.ENGINE_DESCRIPTIONS = {0: fresh.RAY_BOUNCING_ENGINE.replace(default_point=fresh.CHOICE_ORIGIN, points=frozenset())}
    _breaks_a_self_check(fresh, "does not offer its own default")
    fresh = load_module()
    fresh.selfchecks.ENGINE_DESCRIPTIONS = {0: fresh.RAY_BOUNCING_ENGINE.replace(buddha_options=frozenset({fresh.POINT_ALL_MEAN}))}
    _breaks_a_self_check(fresh, "every choice there is")
    fresh = load_module()
    fresh.selfchecks.ENGINE_DESCRIPTIONS = {}
    _breaks_a_self_check(fresh, "needs a description")
    fresh = load_module()
    fresh.selfchecks.ENGINE_DESCRIPTIONS = {0: fresh.RAY_BOUNCING_ENGINE.replace(ranges={"obst:3:radius": (0, 1)})}
    _breaks_a_self_check(fresh, "key pattern")


# --------------------------------------------------------------------------- #
# Random settings
# --------------------------------------------------------------------------- #
def _random_channel(rng):
    src = rng.choice([m.CHAN_SRC_QUERY, m.CHAN_SRC_BUDDHA, m.CHAN_SRC_CONSTANT])
    metric_type = rng.choice([m.METRIC_TYPE_POINT_PAIR, m.METRIC_TYPE_OTHER])
    pool = m.POINT_METRICS if metric_type == m.METRIC_TYPE_POINT_PAIR else m.OTHER_METRICS
    flags = tuple(rng.random() < rng.choice([0.0, 0.1, 0.5]) for _ in range(m.N_BUDDHA_POINTS))
    return m.ChannelSettings(
        chan_src=src, metric_type=metric_type, metric_slot=rng.choice(pool), point1=rng.randrange(m.N_POINT_CHOICES),
        point2=rng.randrange(m.N_POINT_CHOICES), buddha_points=flags, buddha_path_kinds=(True, True, True),
        buddha_compression=rng.randrange(2), buddha_brightness=rng.randrange(5), constant=rng.choice([m.NONE_ZERO, m.NONE_ONE]))


def _random_settings(rng):
    effects = [(rng.randrange(3), rng.randrange(3)) for _ in range(rng.randrange(1, 4))]
    values = settings_dict_with_lists(effects, [_random_channel(rng) for _ in range(3)])
    values["iter"] = rng.choice([1, 20, 100, 101, 5000])
    values["seg_mult"] = rng.choice([0.0, 0.5, 0.75, 1.0, 3.0])
    return values
