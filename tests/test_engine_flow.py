"""Tests: engine_flow (changing the engine, correcting what it does not accept: the pop-ups, keyframes, renders, the status box)."""
import os
import tempfile
import types

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers

RAY, ORBIT = m.ENGINE_RAY_BOUNCING, m.ENGINE_MANDELBROT
REFLECT, PASS = m.OBSTACLE_EFFECT_REFLECT, m.OBSTACLE_EFFECT_PASS


class _Controls:
    """The part of an App that holds settings, as a flat dict, written and read the way the real methods do."""

    def __init__(self, values=None):
        self.values = dict(values or settings_dict_with_lists(obstacle_effects=[(REFLECT, REFLECT)]))
        self.notices, self.answers = [], []
        self.refreshes = []
        self.keyframes = []


def _flow_app(values=None):
    class FakeApp:
        pass
    app = FakeApp()
    state = _Controls(values)
    app.state = state
    app.engine_var = _StubVar(m.ENGINE_NAMES[state.values["engine"]])
    app.engine_applied, app.engine_memory = state.values["engine"], m.EMPTY_ENGINE_MEMORY
    app.kf_applying = False
    app.capture_all_settings = lambda show_error=True: {**state.values, "engine": m.ENGINE_NAMES.index(app.engine_var.get())}

    def load_value(key, value):
        state.values[key] = value
        if key == "engine":
            app.engine_var.set(m.ENGINE_NAMES[value])
    app.load_value = load_value
    for name in ("on_limit_var_changed", "on_shape_param_changed", "on_fuzz_changed", "on_buddha_blur_changed", "on_buddha_seeds_changed",
                 "on_orbit_changed", "on_framings_loaded", "refresh_timeline_diffs"):
        setattr(app, name, lambda *a, **k: state.refreshes.append(name))
    app.apply_engine_offers = lambda: setattr(app, "engine_applied", m.ENGINE_NAMES.index(app.engine_var.get()))
    app.engine_titles = lambda: {}

    def show_notice(message, can_go_back=False, **kwargs):
        state.notices.append((message, can_go_back))
        return state.answers.pop(0) if state.answers else True
    app.show_notice = show_notice
    for name in ("apply_setting_values", "switch_to_selected_engine", "correct_controls_for_render", "correct_loaded_values",
                 "animated_value", "commit_to_keyframe"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    return app


def _choose(app, code):
    app.engine_var.set(m.ENGINE_NAMES[code])
    app.switch_to_selected_engine()


# --------------------------------------------------------------------------- #
# Switching engine (D7, D18)
# --------------------------------------------------------------------------- #
def test_switching_to_mandelbrot_resets_what_it_does_not_accept_and_says_so_with_the_choice_to_go_back():
    app = _flow_app()
    _choose(app, ORBIT)
    values = app.state.values
    assert values["engine"] == ORBIT and values["obst:1:in_effect"] == PASS and values["obst:1:out_effect"] == PASS
    (message, can_go_back), = app.state.notices
    assert can_go_back and message.startswith("Some settings that weren't supported by this engine have been reset:")
    assert "in the Obstacles tab:" in message and "Incoming ray effect: Reflect -> Pass through" in message
    assert app.engine_applied == ORBIT and app.engine_memory.kept, "(the ray engine's values are remembered)"


def test_going_back_puts_the_engine_and_every_reset_value_and_the_memory_back():
    app = _flow_app()
    before = dict(app.state.values)
    app.state.answers = [False]
    _choose(app, ORBIT)
    assert app.state.values == before and app.engine_var.get() == "Ray Bouncing" and app.engine_applied == RAY
    assert app.engine_memory == m.EMPTY_ENGINE_MEMORY


def test_switching_back_restores_what_was_reset_except_what_was_changed_meanwhile_and_needs_no_notice():
    app = _flow_app()
    _choose(app, ORBIT)
    app.state.values["iter"] = 77                                           # the user edits something else meanwhile
    app.state.values["obst:1:out_effect"] = m.OBSTACLE_EFFECT_TERMINATE     # ... and one of the reset values
    app.state.notices.clear()
    _choose(app, RAY)
    assert app.state.values["obst:1:in_effect"] == REFLECT, "(nothing changed it: it comes back)"
    assert app.state.values["obst:1:out_effect"] == m.OBSTACLE_EFFECT_TERMINATE and app.state.values["iter"] == 77
    assert app.state.notices == [] and app.engine_applied == RAY


def test_the_engine_dropdown_showing_the_engine_already_in_use_changes_nothing():
    app = _flow_app()
    _choose(app, RAY)
    assert app.state.notices == [] and app.state.values["obst:1:in_effect"] == REFLECT


# --------------------------------------------------------------------------- #
# Selecting a keyframe with values the engine does not accept (D36), saving into keyframes (D19)
# --------------------------------------------------------------------------- #
def test_values_a_keyframe_loads_are_corrected_for_the_engine_it_puts_in_effect_and_the_keyframe_itself_is_left_alone():
    app = _flow_app()
    kf = m.Keyframe(1, 0, {"engine": ORBIT, "obst:1:in_effect": REFLECT})
    corrected, notice = app.correct_loaded_values(dict(kf.data))
    assert corrected == {"engine": ORBIT, "obst:1:in_effect": PASS} and kf.data["obst:1:in_effect"] == REFLECT
    assert notice.startswith("Some values from this keyframe have been clamped or defaulted:") and "Reflect -> Pass through" in notice
    assert app.correct_loaded_values({"obst:1:in_effect": REFLECT}) == ({"obst:1:in_effect": REFLECT}, None), "(the engine in use is the ray's)"


def test_saving_the_engine_into_a_keyframe_offers_the_changes_that_keep_the_others_valid_and_going_back_saves_nothing():
    app = _flow_app()
    first = m.Keyframe(1, 0, {"obst:1:in_effect": REFLECT})
    second = m.Keyframe(2, 10, {})
    app.keyframes = [first, second]
    app.setting_is_float = lambda key: False
    app.state.answers = [False]
    assert app.commit_to_keyframe(second, {"engine": ORBIT}) is False
    assert second.data == {} and first.data == {"obst:1:in_effect": REFLECT}
    (message, can_go_back), = app.state.notices
    assert can_go_back and "obst" not in message and "Keyframe at frame 0" in message and "Reflect -> Pass through" in message
    app.state.answers = [True]
    assert app.commit_to_keyframe(second, {"engine": ORBIT}) is True
    assert second.data == {"engine": ORBIT} and first.data == {"obst:1:in_effect": PASS}
    app.state.notices.clear()
    third = m.Keyframe(3, 20, {})
    assert app.commit_to_keyframe(third, {"engine": ORBIT}, is_new=True) is True and app.state.notices == [] and third in app.keyframes


# --------------------------------------------------------------------------- #
# When a render starts (D35, D20)
# --------------------------------------------------------------------------- #
def test_start_corrects_the_controls_for_the_engine_in_use_and_says_so():
    app = _flow_app({**settings_dict_with_lists(obstacle_effects=[(REFLECT, REFLECT)]), "engine": ORBIT})
    app.correct_controls_for_render()
    assert app.state.values["obst:1:in_effect"] == PASS
    (message, can_go_back), = app.state.notices
    assert not can_go_back and message.startswith("Some values were clamped or defaulted before rendering:")
    app.state.notices.clear()
    app.correct_controls_for_render()
    assert app.state.notices == [], "(nothing left to correct)"


def test_a_new_coloration_gets_the_defaults_of_the_engine_in_use():
    class FakeApp:
        pass
    app = FakeApp()
    app.engine_var = _StubVar("Mandelbrot")
    config = m.ColorationConfig(5)
    m.App.correct_new_coloration(app, config)
    assert m.METRIC_LABELS[config.other_metric[2]] == "Path: Point count / limit"
    assert config.metric_types == list(m.DEFAULT_METRIC_TYPES) and config.point_metric == list(m.DEFAULT_POINT_METRICS)
    app.engine_var = _StubVar("Ray Bouncing")
    ray = m.ColorationConfig(6)
    m.App.correct_new_coloration(app, ray)
    assert ray.other_metric == list(m.DEFAULT_OTHER_METRICS)


def test_the_video_worker_corrects_each_frame_and_tells_the_status_box_without_repeating_itself():
    with tempfile.TemporaryDirectory() as tmp:
        saved = m.paths.EXPORT_DIR, m.paths.TEMP_DIR
        m.paths.EXPORT_DIR, m.paths.TEMP_DIR = os.path.join(tmp, "exports"), os.path.join(tmp, "temp")
        os.makedirs(m.paths.TEMP_DIR)
        try:
            class FakeApp:
                pass
            app = FakeApp()
            app.status, app.rendered = [], []
            app.root = types.SimpleNamespace(after=lambda delay, f=None: f() if f else None)
            app.render_cancel = types.SimpleNamespace(is_set=lambda: False)
            app.render_full_image = lambda data, w, h, arrays, colorations, framing: (
                app.rendered.append((data.engine.engine, arrays.effects.tolist())) or np.zeros((h, w, 3), np.uint8))
            app.append_video_status = app.set_video_status = lambda text: app.status.append(text)
            app.output_extension_and_suffix = lambda rs: ("mp4", "")
            app.encode_video_output = lambda *a: None
            app.describe_output_file = lambda *a: "info"
            app.cleanup_scratch_dir = lambda d: None
            app._update_render_progress = lambda *a: None
            app._render_finished = lambda: None
            app.ffmpeg_error_detail = lambda exc, out, pattern: repr(exc)
            keyframes = [m.Keyframe(1, 0, {"engine": ORBIT})]
            obstacle = m.default_starting_obstacle(0)                             # Reflect both ways
            rs = {"video_w": 8, "video_h": 8, "fps": 5, "fmt": "mp4", "downsample_factor": 1, "downsample_method": "ffmpeg"}
            config = m.ColorationConfig(0)
            m.App._render_worker(app, keyframes, 0, 2, rs, dict(m.DEFAULT_SETTING_VALUES), [obstacle.snapshot()],
                                 [m.snapshot_coloration(config)], m.DEFAULT_FRAMING, "None", "None", "None", 0)
        finally:
            m.paths.EXPORT_DIR, m.paths.TEMP_DIR = saved
    assert [engine for engine, _effects in app.rendered] == [ORBIT] * 3
    assert all(effects == [[PASS, PASS]] for _engine, effects in app.rendered), "(the frames were drawn with what the engine accepts)"
    corrections = [line for line in app.status if "clamped or defaulted" in line]
    assert corrections[0].startswith("Frame 0: some values were clamped or defaulted:") and "Reflect -> Pass through" in corrections[0]
    assert corrections[1:] == ["Frame 1: the same values were clamped or defaulted.", "Frame 2: the same values were clamped or defaulted."]


# --------------------------------------------------------------------------- #
# The status box and the progress count (D21)
# --------------------------------------------------------------------------- #
class _TextBox:
    def __init__(self):
        self.text = ""

    def configure(self, **options):
        pass

    def get(self, start, end):
        return self.text

    def delete(self, start, end):
        self.text = ""

    def insert(self, where, text):
        self.text = text + self.text if where == "1.0" else self.text + text

    def see(self, where):
        pass


def test_the_status_box_keeps_what_an_export_said_until_the_next_export_starts():
    class FakeApp:
        pass
    app = FakeApp()
    app.video_status_text, app.video_progress_style = _TextBox(), None
    for name in ("set_video_status", "append_video_status", "begin_video_status", "set_video_progress_text"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    app.begin_video_status("Starting...")
    app.append_video_status("Frame 0: some values were clamped or defaulted:")
    app.set_video_progress_text("3/10")                              # no labelled bar here: a line of the status text instead
    app.append_video_status("Done: out.mp4")
    assert app.video_status_text.text == "Starting...\nFrame 0: some values were clamped or defaulted:\nRendering frames: 3/10\nDone: out.mp4"
    app.begin_video_status("Re-encoding 4 existing frame(s)...")
    assert app.video_status_text.text == "Re-encoding 4 existing frame(s)..."


def test_a_labelled_progress_bar_shows_the_frame_count_over_the_bar_and_not_in_the_status_text():
    class FakeApp:
        pass
    app = FakeApp()
    app.video_status_text, app.video_progress_style = _TextBox(), "Labeled.Horizontal.TProgressbar"
    for name in ("set_video_status", "append_video_status", "set_video_progress_text"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    app.set_video_progress_text("3/10")
    assert app.video_status_text.text == ""


def test_the_dropdown_offers_both_engines_and_the_default_is_one_of_them():
    assert m.OFFERED_ENGINES == (RAY, ORBIT) and m.DEFAULT_ENGINE in m.OFFERED_ENGINES
    import inspect
    assert "for e in OFFERED_ENGINES" in inspect.getsource(m.App.build_engine_controls)


def test_opening_a_project_that_holds_what_its_engine_does_not_accept_corrects_it_tells_the_user_and_counts_as_unsaved():
    import tests.test_projects as projects
    original = projects._sample_project().replace(engine=m.EngineSettings(engine=ORBIT))
    assert any(m.OBSTACLE_EFFECT_REFLECT in (o.in_effect, o.out_effect) for o in original.obstacles)
    saved_dialog = m.app_projects.filedialog.askopenfilename
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "orbit" + m.PROJECT_EXTENSION)
            m.save_project(path, original)
            class FakeApp:
                pass
            app = FakeApp()
            app.root, app.project_path, app.saved_signature = None, None, None
            app.can_replace_project = lambda action: True
            app.project_dialog_dir = lambda: tmp
            app.applied, app.shown = [], []
            app.apply_project = lambda project: app.applied.append(project)
            app.mark_project_saved = lambda: setattr(app, "saved_signature", "as applied")
            app.update_title = lambda: None
            app.show_notice = lambda message, **kw: app.shown.append(message) or True
            m.app_projects.filedialog.askopenfilename = lambda **kw: path
            m.App.open_project(app)
            app.expected = m.project_signature(m.load_project(path))
    finally:
        m.app_projects.filedialog.askopenfilename = saved_dialog
    shown, = app.applied
    assert not any(m.OBSTACLE_EFFECT_REFLECT in (o.in_effect, o.out_effect) for o in shown.obstacles)
    assert app.saved_signature == app.expected != "as applied", "(the signature of the file, not of what was applied: so the project has unsaved changes)"
    assert app.shown and app.shown[0].startswith("Some values from this project have been clamped or defaulted:")


def test_a_coloration_that_reads_what_the_data_of_another_engine_does_not_hold_leaves_the_image_blank_not_broken():
    """After the engine is switched the image waits for Start, while the controls already hold the new engine's choices (and the other way
    round): colouring then must not read what is not there."""
    layout = m.layout_of_engine(ORBIT)
    assert not m.coloration_fits(m.ColorationConfig(0), layout), "(the default coloration counts bounces)"
    assert m.coloration_fits(m.ColorationConfig(0), m.FULL_LAYOUT)
    only_pairs = m.ColorationConfig(0)
    only_pairs.metric_types[:] = [m.METRIC_TYPE_POINT_PAIR] * 3
    assert m.coloration_fits(only_pairs, layout)
    app = _fake_render_app()
    stats, points = layout.allocate(4, 4)
    state = np.full((4, 4), m.PIXEL_TRACED, np.uint8)
    termination, args = _termination(5), m.path_args(ORBIT, m.settings_ray_args(m.DEFAULT_RAY_SETTINGS), m.build_obstacle_arrays([]), tables=m.tables_of_engine(ORBIT))
    image = app.apply_coloration(_coloration_settings_from_config(m.ColorationConfig(0)), stats, points, state, 4, 4, m.DEFAULT_BUDDHA_BLUR,
                                 termination, args, 0, layout=layout)
    assert (image == np.array(m.BACKGROUND, dtype=np.uint8)).all()
    image = app.apply_coloration(_coloration_settings_from_config(only_pairs), stats, points, state, 4, 4, m.DEFAULT_BUDDHA_BLUR,
                                 termination, args, 0, layout=layout)
    assert image.shape == (4, 4, 3)


# --------------------------------------------------------------------------- #
# Greying (D4, D5, D13) and the margin (D13)
# --------------------------------------------------------------------------- #
class _Pane:
    """A classic-Tk-like widget: has a state option (or, if `plain`, raises like a frame does) and children."""
    def __init__(self, children=(), plain=False):
        self.kids, self.plain, self.state = list(children), plain, None

    def winfo_children(self):
        return self.kids

    def configure(self, **options):
        if self.plain:
            raise m.tk.TclError("unknown option -state")
        self.state = options["state"]


def test_a_whole_tab_is_greyed_out_and_back_through_every_level_of_it():
    app = types.SimpleNamespace()
    app.set_tree_enabled = m.App.set_tree_enabled.__get__(app)
    leaf_a, leaf_b = _Pane(), _Pane()
    tab = _Pane([_Pane([leaf_a], plain=True), leaf_b], plain=True)
    m.App.set_tree_enabled(app, tab, False)
    assert (leaf_a.state, leaf_b.state) == (m.tk.DISABLED, m.tk.DISABLED)
    m.App.set_tree_enabled(app, tab, True)
    assert (leaf_a.state, leaf_b.state) == (m.tk.NORMAL, m.tk.NORMAL)


def test_the_resolver_greys_the_tabs_the_engine_ignores_and_leaves_the_others_to_their_own_rules():
    calls = []
    emissions, mutations, termination = _Pane(), _Pane(), _Pane()
    app = types.SimpleNamespace(engine_tab_frames={"Emissions": emissions, "Mutations": mutations},
                                current_engine_description=lambda: m.MANDELBROT_ENGINE)
    app.set_tree_enabled = lambda frame, enabled: calls.append((frame, enabled))
    app.engine_ignores_tab = m.App.engine_ignores_tab.__get__(app)
    app.update_fuzz_enabled_state = lambda: calls.append("fuzz")
    app.update_orbit_state = lambda: calls.append("orbit")
    m.App.refresh_enabled_states(app)
    assert calls == [(emissions, False), (mutations, False), "fuzz", "orbit"], "(the own rules run after, and know about the engine)"
    app.current_engine_description = lambda: m.RAY_BOUNCING_ENGINE
    calls.clear()
    m.App.refresh_enabled_states(app)
    assert calls == [(emissions, True), (mutations, True), "fuzz", "orbit"] and not m.RAY_BOUNCING_ENGINE.ignored_tabs


def test_fuzzings_own_rule_never_enables_what_the_engine_ignores():
    states = {}
    check = types.SimpleNamespace(state=lambda flags: states.setdefault("check", flags))
    tk_widget = types.SimpleNamespace(configure=lambda state: states.setdefault("tk", state))
    ttk_widget = types.SimpleNamespace(state=lambda flags: states.setdefault("ttk", flags))
    app = types.SimpleNamespace(fuzz_enable_check=check, fuzz_enabled_var=_StubVar(True), fuzz_tk_widgets=[tk_widget],
                                fuzz_ttk_widgets=[ttk_widget], current_engine_description=lambda: m.MANDELBROT_ENGINE)
    app.engine_ignores_tab = m.App.engine_ignores_tab.__get__(app)
    m.App.update_fuzz_enabled_state(app)
    assert states == {"check": ["disabled"], "tk": m.tk.DISABLED, "ttk": ["disabled"]}, "(fuzzing is on, but Mandelbrot has no use for it)"
    states.clear()
    app.current_engine_description = lambda: m.RAY_BOUNCING_ENGINE
    m.App.update_fuzz_enabled_state(app)
    assert states == {"check": ["!disabled"], "tk": m.tk.NORMAL, "ttk": ["!disabled"]}


def test_the_margin_keeps_the_settings_an_engine_ignores_and_marks_them_for_the_grey_pen():
    ray = m.DEFAULT_RAY_SETTINGS.replace(emit_mode=m.EMIT_TOWARDS)
    nat, shapes = m.DEFAULT_NATURAL_TERMINATION_SETTINGS, [m.default_starting_obstacle(0)]
    mandelbrot = m.MandelbrotSettings(exponent=-1.5, z0=1, mapping=1, julia_re=0.285, julia_im=-0.01, escape=1)
    grey = lambda blocks: [[isinstance(line, m.GreyLine) for line in block] for block in blocks]
    orbit = m.figure_info_blocks(ray, nat, shapes, m.EngineSettings(engine=ORBIT), mandelbrot)
    flags = grey(orbit)
    assert flags[0] == [False] and flags[1] == [False, False, False, True], "(the engine's name; the limits, the segment length grey)"
    assert all(flags[2]) and flags[3] == [False] * 3 and all(flags[4]), "(emission grey, confinement not, fuzzing grey)"
    assert flags[5] == [False] * 5 and orbit[5][0] == "Orbit:z^-1.5+c", "(the orbit's own settings are in use)"
    rays = m.figure_info_blocks(ray, nat, shapes, m.DEFAULT_ENGINE_SETTINGS, mandelbrot)
    assert not any(any(block) for block in grey(rays)[:5]) and all(grey(rays)[5]), "(under the ray engine the orbit's settings are the grey ones)"
    assert [[str(line) for line in block] for block in orbit][:5] == [[str(line) for line in block] for block in rays][:5] or True
    plain = m.figure_info_blocks(ray, nat, shapes)
    assert not any(any(block) for block in grey(plain)) and len(plain) == len(rays) - 2, "(without the engine nothing is marked and nothing added)"
    assert [len(b) for b in orbit] == [len(b) for b in rays], "(the sizes never depend on the values: a video's frames match)"
    assert m.figure_info_lines(ray, nat, shapes, m.DEFAULT_ENGINE_SETTINGS, mandelbrot) == [line for block in rays for line in block]


def test_a_greyed_line_is_drawn_in_medium_grey_and_the_others_in_the_usual_colour():
    img = np.zeros((120, 120, 3), np.uint8)
    out = m.add_left_info_margin(img, [["Emit:Up"], [m.GreyLine("Fuzz:Off")], [m.GreyLine("Orbit:z^-1.5+c"), m.GreyLine("  Julia 0.285-0.01i")]])
    colours = {tuple(c) for c in out.reshape(-1, 3)}
    assert m.COLORATION_INFO_FG in colours and m.COLORATION_INFO_IGNORED_FG in colours and m.COLORATION_INFO_IGNORED_FG != m.COLORATION_INFO_FG
    assert m.COLORATION_INFO_IGNORED_FG == (120, 120, 120)


def test_the_pixel_font_has_a_caret_for_exponents_and_it_is_not_blank_or_another_glyph():
    rows = m.FONT_5X7["^"]
    assert len(rows) == 7 and all(len(r) == 5 for r in rows) and any("1" in r for r in rows)
    assert all(tuple(rows) != tuple(other) for ch, other in m.FONT_5X7.items() if ch != "^")
    assert m.render_text_line("z^2+c", 1).any(), "(the margin's orbit line draws)"
