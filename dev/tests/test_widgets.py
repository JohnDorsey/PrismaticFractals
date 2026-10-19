"""Tests: widgets."""
import os
import types

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# Tabs whose contents can outgrow them (depending on font size and display
# scaling) scroll instead of squeezing their contents
# --------------------------------------------------------------------------- #
def test_make_scrollable_tab_returns_a_canvas_and_a_separate_inner_frame():
    tab = object.__new__(m.tk.Frame)
    canvas, inner = m.make_scrollable_tab(tab)
    assert canvas is not None and inner is not None
    assert canvas is not inner, "widgets go in the inner frame, not the canvas itself"


def test_the_emissions_termination_mutations_and_obstacles_tabs_are_built_inside_scrollable_frames():
    import inspect
    source = inspect.getsource(m.App.build_controls_quadrant)
    assert "self.emissions_canvas, emissions_inner = make_scrollable_tab(emissions_tab)" in source
    assert "self.build_emission_controls(emissions_inner)" in source
    assert "self.termination_canvas, termination_inner = make_scrollable_tab(termination_tab)" in source
    assert "self.build_limits_controls(termination_inner)" in source
    assert "self.mutations_canvas, mutations_inner = make_scrollable_tab(mutations_tab)" in source
    assert "self.build_fuzzing_controls(mutations_inner)" in source
    assert "self.build_segment_mult_controls(mutations_inner)" in source
    assert "self.obst_canvas, obst_inner = make_scrollable_tab(obst_tab)" in source
    assert "self.build_obstacles_controls(obst_inner)" in source


class _TextVar:
    def __init__(self, text=""):
        self.text = text

    def get(self):
        return self.text

    def set(self, text):
        self.text = text


def test_an_exact_float_entry_keeps_the_typed_value_on_commit():
    """Regression: the blur radius entry rewrote what was typed to two decimals
    on Enter (0.123 -> 0.12). An `exact` handle shows the value as typed."""
    h = m.FloatEntryHandle(_TextVar("0.123"), "{:.2f}", 0.0, lo=0.0, hi=0.5, exact=True)
    h.commit()
    assert h.var.get() == "0.123" and h.get() == 0.123
    h.set(0.0)
    assert h.var.get() == "0.0"
    plain = m.FloatEntryHandle(_TextVar("0.123"), "{:.2f}", 0.0)
    plain.commit()
    assert plain.var.get() == "0.12" and plain.get() == 0.12


def test_the_buddha_tab_holds_a_seeding_tab_then_a_blur_tab_each_with_its_own_controls():
    import inspect
    source = inspect.getsource(m.App.build_controls_quadrant)
    assert 'notebook.add(buddha_tab, text="Buddha")' in source
    assert source.index('buddha_notebook.add(buddha_seeding_tab, text="Buddha seeding")') < \
        source.index('buddha_notebook.add(buddha_blur_tab, text="Buddha blur")'), "seeding is the leftmost sub-tab"
    assert "self.build_buddha_seeding_controls(buddha_seeding_tab)" in source
    assert "self.build_buddha_blur_controls(buddha_blur_tab)" in source
    assert '"<<NotebookTabChanged>>"' in source and "update_seed_zone_outline" in source


def test_the_perception_tab_holds_a_framing_tab_then_a_colorations_tab_each_with_its_own_controls():
    import inspect
    source = inspect.getsource(m.App.build_controls_quadrant)
    assert 'notebook.add(perception_tab, text="Perception")' in source
    assert source.index('perception_notebook.add(framing_tab, text="Framing")') < \
        source.index('perception_notebook.add(colorations_tab, text="Colorations")'), "Framing is the left tab"
    assert "self.build_framing_controls(framing_tab)" in source
    assert "self.build_color_controls(colorations_tab)" in source, "the controls the tab always held are in Colorations"


class _HelpButtonWithoutPlacing(m.HelpButton):
    def _place(self):                       # (the headless stubs have no geometry to place by)
        self.placed = True


def test_a_help_button_shows_its_text_on_a_click_and_hides_it_on_the_next_or_on_a_click_elsewhere_or_on_the_popup():
    button = _HelpButtonWithoutPlacing(None, "what this panel does")
    assert button.text == "what this panel does" and not button.shown
    button.toggle()
    assert button.shown and button.placed
    button.toggle()
    assert not button.shown
    button.show()
    button._clicked_elsewhere(types.SimpleNamespace(widget=button))
    assert button.shown, "(a click on the button itself is its own command's business)"
    button._clicked_elsewhere(types.SimpleNamespace(widget=object()))
    assert not button.shown
    button.show()
    button.hide()
    button.hide()
    assert not button.shown
    assert not _HelpButtonWithoutPlacing(None, "").show() and not _HelpButtonWithoutPlacing(None, "").shown, "(nothing to show, no popup)"


def test_the_bottom_middle_holds_the_export_designer_and_a_notebook_of_the_image_and_video_tabs_and_the_bottom_right_is_empty():
    import inspect
    layout = inspect.getsource(m.App.build_status_quadrant)
    assert "self.bottom_notebook = ttk.Notebook(container)" in layout
    assert 'add(exp, text="Image")' in layout and 'add(self.video_tab, text="Video")' in layout
    assert "self.bottom_right_panel = ttk.LabelFrame(self.root" in layout
    video = inspect.getsource(m.App.build_video_panel)
    assert "vp = self.video_tab" in video and "self.root, text" not in video, "(the video controls are in the tab, not a panel of their own)"
    assert "wraplength" not in video, "(no help text takes room in the tab: it is behind the ? button)"
    assert "HelpButton(" in video and "self.video_help_text()" in video
    assert inspect.getsource(m.App.build_video_panel).count("bottom_right_panel") == 0, "(nothing is put into the empty panel)"


def test_the_video_help_text_names_the_folders_relative_to_the_program_and_survives_another_drive():
    app = types.SimpleNamespace()
    saved = m.paths.SCRIPT_DIR, m.paths.TEMP_DIR, m.paths.EXPORT_DIR
    try:
        root = os.path.abspath(os.sep + "program")
        m.paths.SCRIPT_DIR, m.paths.EXPORT_DIR = root, os.path.join(root, "data", "exports")
        m.paths.TEMP_DIR = os.path.join(m.paths.EXPORT_DIR, "TEMP")
        text = m.App.video_help_text(app)
        assert "data/exports/TEMP/" in text and "saved in data/exports/." in text and "Re-encode" in text
        if os.name == "nt":                                 # (relpath raises across drives)
            m.paths.EXPORT_DIR = "D:\\elsewhere\\exports"
            m.paths.TEMP_DIR = "D:\\elsewhere\\exports\\TEMP"
            assert "D:/elsewhere/exports/TEMP/" in m.App.video_help_text(app)
    finally:
        m.paths.SCRIPT_DIR, m.paths.TEMP_DIR, m.paths.EXPORT_DIR = saved

