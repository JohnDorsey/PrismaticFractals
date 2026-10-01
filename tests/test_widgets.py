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
