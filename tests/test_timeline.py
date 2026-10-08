"""Tests: timeline."""
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# Deleting an obstacle or coloration removes its "obst:<id>:*" / "coloration:<id>:*"
# settings from every keyframe immediately (they used to be left behind as
# orphans: invisible in the Timeline's settings list, so impossible to remove)
# --------------------------------------------------------------------------- #
def _kf(id_, frame_index, data):
    return m.Keyframe(id_, frame_index, dict(data))


def test_remove_keyframe_keys_removes_only_that_ids_keys_from_every_keyframe():
    kfs = [_kf(0, 0, {"iter": 5, "obst:3:r": 0.5, "obst:3:theta": 10.0, "obst:30:r": 0.9}),
           _kf(1, 10, {"obst:3:r": 0.7, "coloration:3:mode": "RGB"}),
           _kf(2, 20, {"iter": 9})]
    removed = m.remove_keyframe_keys(kfs, "obst:3:")
    assert removed == 3
    assert kfs[0].data == {"iter": 5, "obst:30:r": 0.9}, \
        "obst:30:* must survive removal of obst:3:* (ids that merely start alike)"
    assert kfs[1].data == {"coloration:3:mode": "RGB"}, "coloration:3 is not obstacle 3"
    assert kfs[2].data == {"iter": 9}


def test_remove_keyframe_keys_with_no_matches_changes_nothing():
    kfs = [_kf(0, 0, {"iter": 5})]
    assert m.remove_keyframe_keys(kfs, "obst:1:") == 0
    assert kfs[0].data == {"iter": 5}


def _fake_delete_obstacle_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.obstacles = []
    app._obst_counter = 0
    app.selected_obstacle = None
    app.keyframes = []
    app.delete_obstacle = m.App.delete_obstacle.__get__(app)
    app.purge_keyframe_settings = m.App.purge_keyframe_settings.__get__(app)
    app.rebuild_obstacle_list = lambda: None
    app.on_shape_param_changed = lambda: None
    app.diff_refreshes = []
    app.refresh_timeline_diffs = lambda: app.diff_refreshes.append(1)
    return app


def test_delete_obstacle_purges_its_settings_from_every_keyframe():
    app = _fake_delete_obstacle_app()
    keep, doomed = m.EllipseShape(0), m.EllipseShape(1)
    app.obstacles = [keep, doomed]
    app.selected_obstacle = doomed
    app.keyframes = [_kf(0, 0, {"iter": 5, "obst:0:r": 0.1, "obst:1:r": 0.5,
                                "coloration:1:mode": "RGB"}),
                     _kf(1, 10, {"obst:1:theta": 45.0, "obst:1:effect": 1})]
    app.delete_obstacle()
    assert app.obstacles == [keep]
    assert app.keyframes[0].data == {"iter": 5, "obst:0:r": 0.1, "coloration:1:mode": "RGB"}
    assert app.keyframes[1].data == {}
    assert app.diff_refreshes, "the Timeline's settings list must be refreshed"


def test_delete_obstacle_with_nothing_selected_changes_no_keyframe():
    app = _fake_delete_obstacle_app()
    o = m.EllipseShape(0)
    app.obstacles = [o]
    app.keyframes = [_kf(0, 0, {"obst:0:r": 0.1})]
    app.delete_obstacle()
    assert app.obstacles == [o]
    assert app.keyframes[0].data == {"obst:0:r": 0.1}


def test_delete_coloration_purges_its_settings_from_every_keyframe():
    app = _fake_coloration_list_app()
    app.new_coloration()                       # coloration id 1, selected
    app.keyframes = [_kf(0, 0, {"iter": 5, "coloration:0:mode": "RGB", "coloration:1:mode": "HSV",
                                "coloration:1:chan0": ("x",), "obst:1:r": 0.2}),
                     _kf(1, 10, {"coloration:1:chan2": ("y",)})]
    app.delete_coloration()
    assert app.all_coloration_ids() == [0]
    assert app.keyframes[0].data == {"iter": 5, "coloration:0:mode": "RGB", "obst:1:r": 0.2}
    assert app.keyframes[1].data == {}


def test_refused_delete_of_the_only_coloration_purges_nothing():
    app = _fake_coloration_list_app()
    app.keyframes = [_kf(0, 0, {"coloration:0:mode": "RGB"})]
    app.delete_coloration()
    assert app.all_coloration_ids() == [0]
    assert app.keyframes[0].data == {"coloration:0:mode": "RGB"}


def test_render_keyframe_snapshot_is_unaffected_by_later_edits_to_the_keyframes():
    """Regression guard for the immediate cleanup above: editing (or purging)
    keyframe settings while a render is running must not change that
    render, so the render works from independent copies."""
    live = [_kf(0, 0, {"obst:1:r": 0.5}), _kf(1, 10, {"obst:1:r": 0.9})]
    snap = m.snapshot_keyframes(live)
    m.remove_keyframe_keys(live, "obst:1:")
    live[0].data["iter"] = 99
    assert [(k.frame_index, k.data) for k in snap] == [(0, {"obst:1:r": 0.5}),
                                                     (10, {"obst:1:r": 0.9})]
    assert snap[0].id == live[0].id


# --------------------------------------------------------------------------- #
# Collapsible sections of the keyframe settings list
# --------------------------------------------------------------------------- #
class _RecordingLabel:
    """Records the text of every ttk.Label built, and keeps its click handler."""
    made = []

    def __init__(self, _parent=None, text="", **_k):
        self.text = text
        self.handler = None
        _RecordingLabel.made.append(self)

    def grid(self, *a, **k):
        return self

    def bind(self, _event, fn):
        self.handler = fn


class _RecordingRow:
    """Stands in for the ttk.Frame/Button of one setting row; counts rows."""
    frames = 0

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: self


def _fake_diff_app():
    import types
    keyframe_cls = m.Keyframe
    tabs = ["Limits", "Fuzzing"]
    entries = [m.SettingEntry(key="max_iter", tab="Limits", label="Iteration limit",
                              is_float=False, fallback=256),
               m.SettingEntry(key="max_dist", tab="Limits", label="Distance limit",
                              is_float=True, fallback=1.0),
               m.SettingEntry(key="fuzz_seed", tab="Fuzzing", label="Seed",
                              is_float=False, fallback=0)]

    class FakeApp:
        pass
    app = FakeApp()
    _give_framings(app)
    app.diff_expanded = set()
    app.diff_presence_var = m.tk.StringVar(value="present or absent")
    app.diff_list_frame = types.SimpleNamespace(winfo_children=lambda: [])
    app.kf_modified = set()
    app.diff_save_all_btn = _StubWidget()
    app.diff_delete_all_btn = _StubWidget()
    app.small_font = app.small_bold_font = None
    app.kf = keyframe_cls(1, 0, {"max_iter": 10})
    app.keyframes = [app.kf]
    app.selected_keyframe = app.kf
    app.capture_all_settings = lambda show_error=True: {e.key: e.fallback for e in entries}
    app.all_setting_entries = lambda: entries
    app.all_setting_tabs = lambda: tabs
    app.setting_status = lambda kf, key, sk, i: "present" if key in kf.data else "none"
    app.rows = []
    app.add_setting_row = lambda row, kf, key, *a: app.rows.append(key)
    app.fit_diff_scroll = lambda: None
    for name in ("refresh_timeline_diffs", "_rebuild_timeline_diffs", "toggle_diff_section"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    return app


def _headers_of(app):
    _RecordingLabel.made = []
    saved = m.ttk.Label
    m.ttk.Label = _RecordingLabel
    try:
        app.refresh_timeline_diffs()
    finally:
        m.ttk.Label = saved
    return list(_RecordingLabel.made)


def test_section_header_text_shows_arrow_title_and_defined_count():
    assert m.diff_section_header_text("Limits", False, 1, 3) == "\u25b6 Limits  (1 of 3 set)"
    assert m.diff_section_header_text("Limits", True, 0, 3) == "\u25bc Limits  (0 of 3 set)"


def test_keyframe_settings_sections_start_collapsed():
    """Every section is collapsed by default: headers only, no setting rows."""
    app = _fake_diff_app()
    labels = _headers_of(app)
    assert [l.text for l in labels] == [
        "\u25b6 Limits  (1 of 2 set)", "\u25b6 Fuzzing  (0 of 1 set)"]
    assert app.rows == []


def test_clicking_a_section_header_expands_only_that_section_and_again_collapses():
    app = _fake_diff_app()
    _headers_of(app)[0].handler(None)          # expand Limits
    labels = _headers_of(app)
    assert labels[0].text.startswith("\u25bc Limits")
    assert labels[1].text.startswith("\u25b6 Fuzzing")
    app.rows = []
    _headers_of(app)
    assert app.rows == ["max_iter", "max_dist"]
    _headers_of(app)[0].handler(None)          # collapse again
    assert app.diff_expanded == set()
    app.rows = []
    _headers_of(app)
    assert app.rows == []


def test_expanded_sections_stay_expanded_across_rebuilds():
    app = _fake_diff_app()
    _headers_of(app)[1].handler(None)          # expand Fuzzing
    for _ in range(3):
        labels = _headers_of(app)
        assert labels[1].text.startswith("\u25bc Fuzzing")


def test_collapsing_does_not_change_what_load_all_save_all_delete_all_act_on():
    """Those buttons work from the keyframe's data and all_setting_entries, never
    from the visible rows, so collapsed sections are still included."""
    import inspect
    for name in ("save_modified_settings", "delete_all_diffs"):
        src = inspect.getsource(getattr(m.App, name))
        assert "diff_expanded" not in src, name


def test_presence_filter_truth_table():
    f = m.setting_passes_presence_filter
    assert m.DIFF_PRESENCE_CHOICES[-1] == "present or absent"   # the default shows everything
    for in_kf in (True, False):
        assert f(in_kf, "present or absent")
        assert f(in_kf, "present") == in_kf
        assert f(in_kf, "absent") == (not in_kf)


def test_keyframe_list_filters_by_presence_and_hides_empty_sections():
    """The keyframe stores only max_iter, so "present" shows just it (hiding the
    Fuzzing section), and "absent" shows the rest."""
    app = _fake_diff_app()
    app.diff_expanded = {"Limits", "Fuzzing"}

    def shown(presence):
        app.diff_presence_var.set(presence)
        app.rows = []
        labels = _headers_of(app)
        return app.rows, [l.text.split("  ")[0][2:] for l in labels]

    assert shown("present or absent")[0] == ["max_iter", "max_dist", "fuzz_seed"]
    assert shown("present") == (["max_iter"], ["Limits"])
    assert shown("absent") == (["max_dist", "fuzz_seed"], ["Limits", "Fuzzing"])
    for k in ("max_dist", "fuzz_seed"):
        app.kf.data[k] = 1
    # every section filtered away -> a placeholder, not a blank list
    labels = _headers_of(app)
    assert [l.text for l in labels] == ["(no settings match the filters)"]


def test_tooltip_popup_is_built_from_options_the_widget_class_accepts():
    """Regression test: Tooltip._show made a ttk.Label with padx/pady, which ttk
    labels reject (TclError: unknown option "-padx"), so no tooltip ever appeared.
    A stand-in ttk.Label that rejects those options like the real one does."""
    import types

    class StrictTtkLabel:
        def __init__(self, _parent=None, **kw):
            for bad in ("padx", "pady"):
                if bad in kw:
                    raise Exception(f'unknown option "-{bad}"')

        def pack(self, *a, **k):
            pass

    class Popup:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, _name):
            return lambda *a, **k: None

    saved = (m.ttk.Label, m.tk.Toplevel)
    m.ttk.Label, m.tk.Toplevel = StrictTtkLabel, Popup
    try:
        tip = m.Tooltip.__new__(m.Tooltip)
        tip.text, tip.tip = "hello", None
        tip.widget = types.SimpleNamespace(winfo_rootx=lambda: 0, winfo_rooty=lambda: 0,
                                           winfo_height=lambda: 10)
        tip._show()
        assert tip.tip is not None
    finally:
        m.ttk.Label, m.tk.Toplevel = saved


def test_floats_interpolate_linearly_and_everything_else_holds():
    kfs = [m.Keyframe(0, 0, {"max_dist": 2.0, "max_iter": 10}),
           m.Keyframe(1, 10, {"max_dist": 4.0, "max_iter": 50})]
    for frame, want in ((0, 2.0), (5, 3.0), (10, 4.0), (2, 2.4)):
        got = m.resolve_value("max_dist", kfs, frame, 0.0, True)
        assert abs(got - want) < 1e-12, (frame, got)
    for frame in (0, 5, 9):
        assert m.resolve_value("max_iter", kfs, frame, 0, False) == 10
    assert m.resolve_value("max_iter", kfs, 10, 0, False) == 50


# --------------------------------------------------------------------------- #
# Selecting a keyframe loads it; edits are marked modified (blue outlines)
# --------------------------------------------------------------------------- #
def test_detect_modified_keys_marks_changes_and_ignores_new_keys():
    d = m.detect_modified_keys
    assert d({"a": 1, "b": 2}, {"a": 1, "b": 2}, set()) == set()
    assert d({"a": 1, "b": 2}, {"a": 5, "b": 2}, set()) == {"a"}
    # sticky: the value going back to where it started does not clear the mark
    assert d({"a": 5}, {"a": 1}, {"a"}) == {"a"}
    assert d({"a": 1}, {"a": 1}, {"a"}) == {"a"}
    # a key that is new (a just-added obstacle) is not an edit ...
    assert d({"a": 1}, {"a": 1, "new": 3}, set()) == set()
    # ... and a key that vanished (a deleted obstacle) stops being modified
    assert d({"a": 1, "gone": 2}, {"a": 1}, {"gone"}) == set()


def test_self_checks_catch_a_missing_setting_widget_entry():
    fresh = load_module()
    del fresh.SETTING_WIDGET_ATTRS["seed"]
    try:
        fresh.run_self_checks()
        raise AssertionError("expected run_self_checks() to raise")
    except AssertionError as e:
        assert "SETTING_WIDGET_ATTRS" in str(e)


def _fake_kf_app():
    """K0 (frame 0) {a: 0.0, c: 0.0}, K1 (frame 5) {b: 5}, K2 (frame 10) {a: 10.0,
    c: 10.0}. a and c are animated through K1; d is defined by no keyframe."""
    class FakeApp:
        pass
    app = FakeApp()
    _give_framings(app)
    app.keyframes = [m.Keyframe(0, 0, {"a": 0.0, "c": 0.0}), m.Keyframe(1, 5, {"b": 5}),
                     m.Keyframe(2, 10, {"a": 10.0, "c": 10.0})]
    app._kf_counter = 10
    app.selected_keyframe = None
    app.kf_modified, app.kf_last_seen, app.kf_applying = set(), None, False
    app.current = {"a": 99.0, "b": 99, "c": 99.0, "d": 7}
    app.loaded = []
    app.capture_all_settings = lambda show_error=True: dict(app.current)
    app.all_setting_entries = lambda: [
        m.SettingEntry(key=k, tab="T", label=k, is_float=isinstance(v, float), fallback=v)
        for k, v in app.current.items()]
    app.setting_fallback = lambda key: app.current[key]
    app.setting_is_float = lambda key: isinstance(app.current[key], float)

    def load_value(key, val):
        app.loaded.append(key)
        app.current[key] = val
    app.load_value = load_value
    for name in ("on_limit_var_changed", "on_shape_param_changed", "on_fuzz_changed",
                 "on_buddha_blur_changed", "on_buddha_seeds_changed", "rebuild_keyframe_list", "refresh_timeline_diffs",
                 "refresh_modified_outlines"):
        setattr(app, name, lambda *a, **k: None)
    for name in ("select_keyframe", "deselect_keyframe", "apply_keyframe_settings",
                 "check_modified", "save_settings", "save_modified_settings",
                 "setting_status", "load_animated_setting", "new_keyframe",
                 "new_keyframe_empty", "delete_selected_keyframe"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    return app


def test_selecting_a_keyframe_loads_its_settings_and_the_ones_neighbors_animate():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[1])
    assert sorted(app.loaded) == ["a", "b", "c"]            # d: nobody defines it
    assert app.current == {"a": 5.0, "b": 5, "c": 5.0, "d": 7}   # a, c blended at frame 5
    assert app.selected_keyframe is app.keyframes[1]
    assert app.kf_modified == set() and app.kf_last_seen == app.current


def test_edits_are_marked_modified_and_stay_marked():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[1])
    assert app.check_modified() is False
    app.current["d"] = 8
    assert app.check_modified() is True and app.kf_modified == {"d"}
    app.current["d"] = 7                                    # typed back to the old value
    app.check_modified()
    assert app.kf_modified == {"d"}
    app.current["a"] = 1.5
    app.check_modified()
    assert app.kf_modified == {"a", "d"}


def test_loading_a_keyframe_never_counts_as_a_user_edit():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[0])
    app.select_keyframe(app.keyframes[2])                   # overwrites a and c
    assert app.kf_modified == set()
    app.kf_applying = True
    app.current["d"] = 0
    assert app.check_modified() is False and app.kf_modified == set()


def test_nothing_is_tracked_while_no_keyframe_is_selected():
    app = _fake_kf_app()
    app.current["d"] = 8
    assert app.check_modified() is False and app.kf_modified == set()


def test_deselecting_keeps_values_but_drops_modified_status_so_reselecting_overwrites():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[0])
    app.current["a"] = 42.0
    app.check_modified()
    assert app.kf_modified == {"a"}
    app.deselect_keyframe()
    assert app.selected_keyframe is None and app.kf_modified == set()
    assert app.current["a"] == 42.0                         # not lost immediately
    app.select_keyframe(app.keyframes[2])
    assert app.current["a"] == 10.0 and app.kf_modified == set()


def test_selecting_another_keyframe_equals_deselect_then_select():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[0])
    app.current["b"] = 1
    app.check_modified()
    app.select_keyframe(app.keyframes[1])
    assert app.kf_modified == set() and app.current["b"] == 5


def test_reselecting_the_selected_keyframe_changes_nothing():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[1])
    app.current["d"] = 8
    app.check_modified()
    app.loaded.clear()
    app.select_keyframe(app.keyframes[1])
    assert app.loaded == [] and app.kf_modified == {"d"} and app.current["d"] == 8


def test_save_all_modified_saves_only_modified_settings_and_clears_them():
    app = _fake_kf_app()
    kf = app.keyframes[1]
    app.select_keyframe(kf)
    app.current["d"] = 8                                    # absent from the keyframe
    app.current["b"] = 6                                    # present in the keyframe
    app.check_modified()
    assert app.kf_modified == {"b", "d"}
    app.save_modified_settings()
    assert kf.data == {"b": 6, "d": 8}                      # a and c (loaded, unedited) not saved
    assert app.kf_modified == set()


def test_saving_one_setting_clears_only_that_settings_mark():
    app = _fake_kf_app()
    kf = app.keyframes[1]
    app.select_keyframe(kf)
    app.current["d"] = 8
    app.current["b"] = 6
    app.check_modified()
    app.save_settings(kf, ["d"])
    assert kf.data == {"b": 5, "d": 8} and app.kf_modified == {"b"}


def test_new_empty_keyframe_is_selected_and_loads_only_what_neighbors_animate():
    app = _fake_kf_app()
    app.keyframes[2].frame_index = 20                       # next new one lands at 21
    app.new_keyframe_empty()
    kf = app.selected_keyframe
    assert kf.data == {} and kf.frame_index == 21 and kf is app.keyframes[-1]
    assert app.kf_modified == set()


def test_new_keyframe_from_current_is_selected_without_reloading_anything():
    app = _fake_kf_app()
    app.new_keyframe()
    kf = app.selected_keyframe
    assert kf.data == {"a": 99.0, "b": 99, "c": 99.0, "d": 7}
    assert app.loaded == [] and app.kf_modified == set()


def test_deleting_the_selected_keyframe_deselects_it_and_clears_modified():
    app = _fake_kf_app()
    app.select_keyframe(app.keyframes[1])
    app.current["d"] = 8
    app.check_modified()
    app.delete_selected_keyframe()
    assert app.selected_keyframe is None and app.kf_modified == set()
    assert len(app.keyframes) == 2


def test_settings_list_marks_modified_sections_and_save_button_state():
    app = _fake_diff_app()
    app.diff_expanded = {"Limits"}
    app.kf_modified = {"max_dist"}
    labels = _headers_of(app)
    assert labels[0].text == "\u25bc Limits  (1 of 2 set, 1 modified)"
    assert labels[1].text == "\u25b6 Fuzzing  (0 of 1 set)"


def test_a_setting_defined_on_one_side_only_is_animated_from_that_side():
    """The width example: #1 stores 1.4, #2 stores 1.5, #3 stores nothing. #3 gets
    1.5 (only earlier keyframes define it); #1 and #2 keep their own values; a
    keyframe before the first one to define a setting gets the first one's value;
    and once a later keyframe defines it too, #3 is blended like normal."""
    app = _fake_kf_app()
    app.current = {"w": 9.9}
    app.keyframes = [m.Keyframe(0, 0, {"w": 1.4}), m.Keyframe(1, 5, {"w": 1.5}),
                     m.Keyframe(2, 10, {})]
    early = m.Keyframe(3, -3, {})
    app.keyframes.append(early)
    app.select_keyframe(app.keyframes[0])
    assert app.current["w"] == 1.4
    app.select_keyframe(app.keyframes[2])
    assert app.current["w"] == 1.5                     # held from the last defining keyframe
    app.select_keyframe(early)
    assert app.current["w"] == 1.4                     # held from the first defining keyframe
    app.select_keyframe(app.keyframes[1])
    assert app.current["w"] == 1.5
    app.keyframes.append(m.Keyframe(4, 20, {"w": 2.5}))    # a keyframe after #3 defines it too
    app.select_keyframe(app.keyframes[2])
    assert abs(app.current["w"] - (1.5 + 1.0 / 3)) < 1e-9   # blended: frame 10 of 5..20


def test_frames_before_the_first_defining_keyframe_hold_its_value():
    """Regression test: resolve_value blended backwards past the first keyframe
    that defines a float setting (frame -3 of keyframes at 0 and 5 gave
    1.4 - 0.6 * (1.5 - 1.4)), instead of holding the first keyframe's value."""
    kfs = [m.Keyframe(0, 0, {"w": 1.4}), m.Keyframe(1, 5, {"w": 1.5})]
    assert m.resolve_value("w", kfs, -3, 0.0, True) == 1.4
    assert m.resolve_value("w", kfs, 0, 0.0, True) == 1.4
    assert abs(m.resolve_value("w", kfs, 2.5, 0.0, True) - 1.45) < 1e-12
    assert m.resolve_value("w", kfs, 50, 0.0, True) == 1.5


def test_settings_list_scroll_is_refitted_after_every_rebuild():
    """Regression test: with the list scrolled down, choosing the "present" filter
    (or collapsing a section) left it scrolled past its shorter content - a blank
    list - because the scroll area only followed the list frame's <Configure>
    event, which never comes when the canvas is showing none of the frame."""
    calls = []
    app = _fake_diff_app()
    app.fit_diff_scroll = lambda: calls.append("fit")
    app.refresh_timeline_diffs()
    app.toggle_diff_section("Limits")
    assert calls == ["fit", "fit"]

    class Canvas:
        def __init__(self):
            self.log = []
        def bbox(self, _tag):
            return (0, 0, 355, 116)
        def configure(self, **kw):
            self.log.append(("configure", kw))
        def yview(self):
            return (0.7, 0.78)
        def yview_moveto(self, f):
            self.log.append(("moveto", f))

    fit_app = _fake_diff_app()
    fit_app.diff_canvas = Canvas()
    fit_app.diff_list_frame = type("F", (), {"update_idletasks": lambda self: None})()
    m.App.fit_diff_scroll(fit_app)
    assert fit_app.diff_canvas.log == [("configure", {"scrollregion": (0, 0, 355, 116)}),
                                       ("moveto", 0.7)]


def _recorded_angle_entries(build_name):
    """Build a controls section on a fake app whose add_angle_entry only records
    (label, lo, hi) for each angle entry asked for."""
    calls = []

    class Handle:
        widget = None

    class FakeApp:
        root = _StubWidget()
        fuzz_ttk_widgets = []

        def add_angle_entry(self, parent, row, label, default, on_change=None,
                            lo=None, hi=None, fmt="{:.2f}"):
            calls.append((label, lo, hi))
            return Handle()

        def __getattr__(self, name):
            return lambda *a, **k: None
    app = FakeApp()
    getattr(m.App, build_name)(app, _StubWidget())
    return calls


def test_angle_offset_and_rotation_style_entries_have_no_limits():
    """The emission angle offset, and the fuzzing offsets, take any number of
    degrees (many revolutions, either direction)."""
    (emission,) = _recorded_angle_entries("build_emission_controls")
    assert emission[1:] == (None, None), emission
    fuzz = {label: (lo, hi) for label, lo, hi in _recorded_angle_entries("build_fuzzing_controls")}
    assert fuzz["Offset (\u00b0)"] == (None, None)


def test_fuzzing_jitter_ranges_stay_limited_to_0_through_180_degrees():
    """Jitter is a +/- range, so it keeps its 0..180 limits (only offsets and
    rotations are unlimited)."""
    jitters = [(lo, hi) for label, lo, hi in _recorded_angle_entries("build_fuzzing_controls")
               if label.startswith("Jitter")]
    assert len(jitters) == 3 and all(j == (0.0, 180.0) for j in jitters), jitters


def test_huge_emission_offsets_point_the_same_way_as_their_remainder():
    base = _angle_deg(_first_direction(0.2, 0.1, _emission(UP, offset=30.0)))
    for offset in (30.0 + 360.0, 30.0 - 720.0, 30.0 + 360.0 * 1000):
        got = _angle_deg(_first_direction(0.2, 0.1, _emission(UP, offset=offset)))
        assert abs(((got - base + 180.0) % 360.0) - 180.0) < 1e-6, offset


def test_default_moss_egg_fits_inside_the_view_at_every_apex_angle():
    """Regression test: at the old default size (1.0) the default egg (90 degree
    apex) reached y = 1.034, past the top edge of the view (which spans -1 to 1)."""
    g = np.linspace(-1.2, 1.2, 1201)
    gx, gy = np.meshgrid(g, g)
    for angle in range(60, 181, 10):
        bound = m.moss_egg_bound(float(angle), m.DEFAULT_MOSS_SIZE, 0.0)
        inside = m.inside_boundary_grid(gx, gy, m.SHAPE_MOSS_EGG, bound)
        assert np.abs(gy[inside]).max() < 1.0 and np.abs(gx[inside]).max() < 1.0, angle


class _WidgetsApp:
    """Just enough App to ask which controls show a setting."""


def test_every_setting_and_obstacle_field_resolves_to_a_control():
    import types
    app = _WidgetsApp()
    names = set(m.SETTING_WIDGET_ATTRS.values()) | {"mode_combo"}
    for name in names:
        setattr(app, name, types.SimpleNamespace(name=name))
    app.chan_blocks = [types.SimpleNamespace(name=f"block{c}") for c in range(3)]
    app.obs_widgets = {}
    for cls in m.SHAPE_CLASSES:
        for p in cls.PARAMS:
            app.obs_widgets[(cls, p.field)] = types.SimpleNamespace(name=f"{cls.__name__}.{p.field}")
    for field in m.EllipseShape.FIELD_NAMES:
        if field not in ("width", "height"):
            app.obs_widgets[(m.Shape, field)] = types.SimpleNamespace(name=f"Shape.{field}")
    v = m.ColorationConfig(2)
    app.selected_coloration_obj = lambda: v
    app.widgets_for_setting = m.App.widgets_for_setting.__get__(app)
    app.obstacle_widget = m.App.obstacle_widget.__get__(app)
    for key in m.SETTING_WIDGET_ATTRS:
        assert [w.name for w in app.widgets_for_setting(key)] == [m.SETTING_WIDGET_ATTRS[key]]
    for cls in m.SHAPE_CLASSES:
        o = cls(4)
        app.selected_obstacle = o
        for field in o.FIELD_NAMES:
            found = app.widgets_for_setting(f"obst:4:{field}")
            owner = cls.__name__ if field in {p.field for p in cls.PARAMS} else "Shape"
            assert [w.name for w in found] == [f"{owner}.{field}"], (cls.__name__, field)
            assert app.widgets_for_setting(f"obst:5:{field}") == []     # not the edited obstacle
    assert [w.name for w in app.widgets_for_setting("coloration:2:mode")] == ["mode_combo"]
    assert [w.name for w in app.widgets_for_setting("coloration:2:chan1")] == ["block1"]
    assert app.widgets_for_setting("coloration:3:chan1") == []
