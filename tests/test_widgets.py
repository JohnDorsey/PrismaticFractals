"""Tests: widgets."""

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


def test_the_simulation_and_obstacles_tabs_are_built_inside_scrollable_frames():
    import inspect
    source = inspect.getsource(m.App.build_controls_quadrant)
    assert "self.sim_canvas, sim_inner = make_scrollable_tab(sim_tab)" in source
    assert "self.build_limits_controls(sim_inner)" in source
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
